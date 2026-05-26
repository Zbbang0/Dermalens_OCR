import cv2
import os
import numpy as np
from datetime import datetime


class ImagePreprocessor:
    """
    Dermalens OCR 이미지 전처리 클래스

    목적:
    1. 화장품 성분표 이미지에서 글자가 최대한 잘 추출되도록 전처리한다.
    2. 특정 이미지, 특정 라벨, 특정 위치에 맞춘 전처리가 아니라 일반화된 전처리를 수행한다.
    3. PaddleOCR이 여러 조건에서 인식할 수 있도록 전처리 이미지를 여러 버전으로 생성한다.
    4. 원본 이미지는 반드시 보존한다.

    [이전 수정 사항]

    (G) _load_image_safe() 신설 — 한글 경로 대응
    (H) _inverted_preprocess() 신설 — 어두운 배경 + 밝은 글씨 대응
    (I) _detect_dark_background() 신설 — 어두운 배경 자동 감지

    [이번 수정 사항 — 1단계 OCR/전처리 강화]

    (J) _get_auto_scale() 배율 상향
        기존: 1400px 이하 → 1.6, 2200px 이하 → 1.3, 그 이상 → 1.1
        수정: 1400px 이하 → 2.0, 2200px 이하 → 1.6, 3500px 이하 → 1.4
              → 작은 성분 글씨도 PaddleOCR이 인식 가능한 크기(20px+)로 확보

    (K) _detect_dark_background() 정확도 향상
        기존: 평균 밝기 < 100 단독 판단
              → 일부분만 어두운 라벨에서 오판
        수정: 평균 밝기 + 어두운 픽셀 비율 복합 판단
              평균 < 110 이면서 어두운 픽셀(<80) 비율 35% 이상일 때만 dark
              매우 어두운 경우(평균 < 75)는 비율 무관

    (L) _measure_brightness_stats() 신설
        밝기 통계를 dict로 반환 → variant info에 담아 run_ocr 점수 계산에 활용

    (M) _inverted_preprocess() 강화
        기존: invert → CLAHE(2.5, 8x8) → denoise → adaptive_sharpen
        수정: invert → CLAHE(4.0, 4x4) → denoise → adaptive_sharpen → 언샤프 마스킹
              → 어두운 배경 라벨에서 글자 경계가 훨씬 또렷해짐

    (N) _high_contrast_binary_preprocess() 신규 variant
        Otsu 전역 이진화 + 모폴로지 OPEN
        → 저화질, 노이즈 많은 스마트폰 라벨 사진에서 효과 큼
        → soft_binary(적응형 이진화)와 상보적

    (O) _color_clahe_preprocess() 신규 variant
        LAB 색공간의 L 채널에만 CLAHE 적용 후 BGR 복원
        → 컬러 정보 보존하면서 대비 강화
        → 그레이 변환 시 정보 손실이 큰 컬러 라벨에 강함

    (P) preprocess_variants_and_save()에 신규 variant 2개 등록
        총 8개 variant: original, inverted, layout_safe, readable,
                       ocr_enhanced, soft_binary, high_contrast_binary, color_clahe
        → 라벨 종류가 다양해도 best variant가 항상 1개 이상 보장
    """

    # =========================================================
    # 1. 이미지 품질 측정
    # =========================================================

    @staticmethod
    def _get_auto_scale(width, height):
        """
        원본 이미지 크기에 따라 OCR용 확대 배율 자동 결정
        min_side 기준 (짧은 쪽이 충분히 커야 텍스트 인식이 잘 됨)

        [수정] 배율 상향
        기존: 1400px → 1.6, 2200px → 1.3, 그 이상 → 1.1
              → 라벨 사진(1000~1500px)에서 성분 글씨 폭 8~12px로 작게 남음
              → PaddleOCR이 작은 글씨 누락하거나 오인식
        수정: 1400px → 2.0, 2200px → 1.6, 그 이상 → 1.4
              → 글자 높이 20px 이상 확보, PaddleOCR 정확도 향상

        주의: 너무 키우면 메모리 폭증 + 노이즈도 같이 확대됨
        """

        min_side = min(width, height)

        if min_side <= 400:
            return 3.0

        elif min_side <= 600:
            return 2.5

        elif min_side <= 900:
            return 2.2

        elif min_side <= 1400:
            return 2.0   # 기존 1.6 → 2.0

        elif min_side <= 2200:
            return 1.6   # 기존 1.3 → 1.6

        elif min_side <= 3500:
            return 1.4   # 기존 1.1 → 1.4

        else:
            return 1.2   # 매우 큰 이미지도 약간은 확대

    @staticmethod
    def _calculate_blur_score(gray_image):
        if gray_image is None:
            return 0.0

        try:
            return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())

        except Exception:
            return 0.0

    @staticmethod
    def _calculate_contrast_score(gray_image):
        if gray_image is None:
            return 0.0

        try:
            return float(np.std(gray_image))

        except Exception:
            return 0.0

    @staticmethod
    def _build_image_quality_info(image):
        if image is None:
            return {
                "width": 0,
                "height": 0,
                "blur_score": 0.0,
                "contrast_score": 0.0
            }

        height, width = image.shape[:2]

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        except Exception:
            gray = image

        return {
            "width": int(width),
            "height": int(height),
            "blur_score": ImagePreprocessor._calculate_blur_score(gray),
            "contrast_score": ImagePreprocessor._calculate_contrast_score(gray)
        }

    @staticmethod
    def _detect_dark_background(image):
        """
        이미지가 어두운 배경인지 자동 감지.

        [수정] 평균 밝기 단독 → 평균 + 어두운 픽셀 비율 + 표준편차 복합 판단
        기존: mean < 100 단독 판단
              → 일부분만 어둡고 나머지는 밝은 라벨에서 오판
              → 그레이 배경 라벨에서 dark 오판
        수정: mean < 110 이면서 어두운 픽셀(< 80) 비율 35% 이상일 때만 dark
              → 진짜 어두운 배경 라벨만 정확히 식별

        팬틴 프레스티지처럼 금색 배경 + 흰 글씨 라벨이 대표적인 케이스.
        """

        if image is None:
            return False

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            mean_brightness = float(np.mean(gray))

            # 어두운 픽셀(< 80) 비율
            dark_pixel_ratio = float(np.sum(gray < 80)) / gray.size

            # 평균 밝기와 어두운 픽셀 비율을 함께 판단
            if mean_brightness < 110 and dark_pixel_ratio >= 0.35:
                return True

            # 매우 어두운 경우 (평균 < 75)는 ratio 조건 없이도 dark
            if mean_brightness < 75:
                return True

            return False

        except Exception:
            return False

    @staticmethod
    def _measure_brightness_stats(image):
        """
        이미지의 밝기 통계 측정.
        info dict에 담아 run_ocr 점수 계산에 활용.
        """

        if image is None:
            return {
                "mean_brightness": 0.0,
                "dark_pixel_ratio": 0.0,
                "is_dark_background": False
            }

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            mean_brightness = float(np.mean(gray))
            dark_pixel_ratio = float(np.sum(gray < 80)) / gray.size

            return {
                "mean_brightness": mean_brightness,
                "dark_pixel_ratio": dark_pixel_ratio,
                "is_dark_background": ImagePreprocessor._detect_dark_background(image)
            }

        except Exception:
            return {
                "mean_brightness": 0.0,
                "dark_pixel_ratio": 0.0,
                "is_dark_background": False
            }

    # =========================================================
    # 2. 이미지 로드 (한글 경로 대응)
    # =========================================================

    @staticmethod
    def _load_image_safe(input_path):
        """
        한글/특수문자 경로 대응 이미지 로드.

        기존: cv2.imread() → Windows에서 한글 경로 None 반환
        수정: np.fromfile() + cv2.imdecode() 조합

        모든 이미지 로드는 이 함수를 통해 수행한다.
        """

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"이미지 파일 없음: {input_path}")

        try:
            # 한글 경로 대응: numpy로 바이트 읽기 후 decode
            img_array = np.fromfile(input_path, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if image is not None:
                return image

        except Exception:
            pass

        # fallback: 일반 imread
        image = cv2.imread(input_path)

        if image is None:
            raise ValueError(f"이미지 로드 실패: {input_path}")

        return image

    # =========================================================
    # 3. 기본 이미지 처리 함수
    # =========================================================

    @staticmethod
    def _resize_image(image, scale):
        if image is None:
            return image

        if scale <= 0:
            scale = 1.0

        if abs(scale - 1.0) < 0.01:
            return image.copy()

        height, width = image.shape[:2]

        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )

        return resized

    @staticmethod
    def _resize_for_ocr(image):
        height, width = image.shape[:2]

        scale = ImagePreprocessor._get_auto_scale(width, height)

        resized = ImagePreprocessor._resize_image(image, scale)

        new_height, new_width = resized.shape[:2]

        return resized, scale, width, height, new_width, new_height

    @staticmethod
    def _apply_clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

        return clahe.apply(gray_image)

    @staticmethod
    def _denoise(gray_image, strength=3):
        return cv2.fastNlMeansDenoising(
            gray_image,
            None,
            h=strength,
            templateWindowSize=7,
            searchWindowSize=21
        )

    @staticmethod
    def _adaptive_sharpen(gray_image, blur_score=None, scale=1.0):
        if blur_score is None:
            blur_score = ImagePreprocessor._calculate_blur_score(gray_image)

        scale_factor = max(scale * scale, 1.0)
        normalized_blur = blur_score / scale_factor

        blur = cv2.GaussianBlur(gray_image, (0, 0), 1.0)

        if normalized_blur < 50:
            alpha = 1.55
            beta = -0.55

        elif normalized_blur < 120:
            alpha = 1.40
            beta = -0.40

        else:
            alpha = 1.25
            beta = -0.25

        sharpened = cv2.addWeighted(gray_image, alpha, blur, beta, 0)

        return sharpened

    @staticmethod
    def _light_sharpen(gray_image):
        blur = cv2.GaussianBlur(gray_image, (0, 0), 1.0)
        sharpened = cv2.addWeighted(gray_image, 1.3, blur, -0.3, 0)

        return sharpened

    @staticmethod
    def _gamma_correction(gray_image, gamma=1.0):
        if gamma <= 0:
            gamma = 1.0

        inv_gamma = 1.0 / gamma

        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")

        return cv2.LUT(gray_image, table)

    @staticmethod
    def _to_bgr(image):
        if image is None:
            return image

        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image

    @staticmethod
    def _deskew_light(image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            height, width = image.shape[:2]

            diag = float(np.sqrt(width ** 2 + height ** 2))
            hough_threshold = max(40, int(diag * 0.04))
            min_line_length = max(40, int(diag * 0.12))
            max_line_gap = max(10, int(diag * 0.01))

            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=hough_threshold,
                minLineLength=min_line_length,
                maxLineGap=max_line_gap
            )

            if lines is None:
                return image

            angles = []

            for line in lines:
                x1, y1, x2, y2 = line[0]

                dx = x2 - x1
                dy = y2 - y1

                if dx == 0:
                    continue

                angle = np.degrees(np.arctan2(dy, dx))

                if -8 <= angle <= 8:
                    angles.append(angle)

            if not angles:
                return image

            median_angle = float(np.median(angles))

            if abs(median_angle) < 0.5:
                return image

            if abs(median_angle) > 5:
                return image

            center = (width // 2, height // 2)

            matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

            rotated = cv2.warpAffine(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

            return rotated

        except Exception:
            return image

    # =========================================================
    # 4. 전처리 variant 생성
    # =========================================================

    @staticmethod
    def _original_variant(image):
        quality = ImagePreprocessor._build_image_quality_info(image)

        info = {
            "variant": "original",
            "purpose": "원본 이미지",
            "scale": 1.0,
            "original_size": {
                "width": quality["width"],
                "height": quality["height"]
            },
            "processed_size": {
                "width": quality["width"],
                "height": quality["height"]
            },
            "blur_score_before": quality["blur_score"],
            "contrast_score_before": quality["contrast_score"],
            "blur_score_after": quality["blur_score"],
            "contrast_score_after": quality["contrast_score"]
        }

        return image.copy(), info

    @staticmethod
    def _inverted_preprocess(image):
        """
        어두운 배경 + 밝은 글씨 대응 전처리.

        대상:
        - 금색/검정/진한 배경에 흰 글씨로 인쇄된 화장품 라벨
        - 팬틴 프레스티지, 럭셔리 라인 등

        [수정] CLAHE clip_limit 상향 + 언샤프 마스킹 추가
        기존: invert → CLAHE(2.5) → denoise → adaptive_sharpen
              → 반전 후에도 글자 경계가 흐릿하면 OCR이 약함
        수정: invert → CLAHE(4.0, tile 4x4) → denoise → adaptive_sharpen
              → 언샤프 마스킹 추가 → 글자 경계 또렷
              → tile_grid_size 8x8 → 4x4 (지역 대비 더 강하게)

        처리 순서:
        1. 자동 확대 (min_side 기준)
        2. 그레이스케일 변환
        3. 이미지 반전 (bitwise_not) ← 핵심
           → 흰 글씨가 검정 글씨로, 어두운 배경이 밝은 배경으로
        4. CLAHE 강화 대비 향상 (clip 4.0, tile 4x4)
        5. 노이즈 제거
        6. 적응형 샤프닝
        7. 언샤프 마스킹 추가 (글자 엣지 강화)

        효과:
        - 반전 전: PaddleOCR이 배경과 글씨를 구분 못해 글자 누락/오인식
        - 반전 후: 일반 흰 배경 + 검정 글씨로 처리되어 OCR 정확도 향상
        - clip_limit 4.0 + 4x4 tile로 작은 글씨 영역도 충분한 대비 확보
        """

        brightness_stats = ImagePreprocessor._measure_brightness_stats(image)

        resized, scale, old_w, old_h, new_w, new_h = ImagePreprocessor._resize_for_ocr(image)

        resized = ImagePreprocessor._deskew_light(resized)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        blur_score_before = ImagePreprocessor._calculate_blur_score(gray)
        contrast_score_before = ImagePreprocessor._calculate_contrast_score(gray)

        # 핵심: 이미지 반전
        inverted = cv2.bitwise_not(gray)

        # [수정] 반전 후 CLAHE 강화: clip 2.5→4.0, tile 8x8→4x4
        contrast = ImagePreprocessor._apply_clahe(
            inverted,
            clip_limit=4.0,
            tile_grid_size=(4, 4)
        )

        denoised = ImagePreprocessor._denoise(contrast, strength=3)

        processed = ImagePreprocessor._adaptive_sharpen(
            denoised,
            blur_score=blur_score_before,
            scale=scale
        )

        # [추가] 언샤프 마스킹 — 글자 경계 추가 강화
        unsharp_blur = cv2.GaussianBlur(processed, (0, 0), 1.5)
        processed = cv2.addWeighted(processed, 1.5, unsharp_blur, -0.5, 0)

        processed_bgr = ImagePreprocessor._to_bgr(processed)

        final_h, final_w = processed_bgr.shape[:2]

        info = {
            "variant": "inverted",
            "purpose": "어두운 배경 + 밝은 글씨 대응 강화 반전 전처리",
            "scale": scale,
            "original_size": {
                "width": old_w,
                "height": old_h
            },
            "processed_size": {
                "width": final_w,
                "height": final_h
            },
            "blur_score_before": blur_score_before,
            "contrast_score_before": contrast_score_before,
            "blur_score_after": ImagePreprocessor._calculate_blur_score(processed),
            "contrast_score_after": ImagePreprocessor._calculate_contrast_score(processed),
            "is_inverted": True,
            "source_is_dark_background": brightness_stats["is_dark_background"],
            "source_mean_brightness": brightness_stats["mean_brightness"],
            "source_dark_pixel_ratio": brightness_stats["dark_pixel_ratio"]
        }

        return processed_bgr, info

    @staticmethod
    def _layout_safe_preprocess(image):
        original_quality = ImagePreprocessor._build_image_quality_info(image)

        old_w = original_quality["width"]
        old_h = original_quality["height"]

        scale = ImagePreprocessor._get_auto_scale(old_w, old_h)

        resized = ImagePreprocessor._resize_image(image, scale)

        resized = ImagePreprocessor._deskew_light(resized)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        blur_score_before = ImagePreprocessor._calculate_blur_score(gray)
        contrast_score_before = ImagePreprocessor._calculate_contrast_score(gray)

        contrast = ImagePreprocessor._apply_clahe(
            gray,
            clip_limit=1.6,
            tile_grid_size=(8, 8)
        )

        processed_bgr = ImagePreprocessor._to_bgr(contrast)

        new_h, new_w = processed_bgr.shape[:2]

        info = {
            "variant": "layout_safe",
            "purpose": "레이아웃 보존형 OCR 전처리",
            "scale": scale,
            "original_size": {
                "width": old_w,
                "height": old_h
            },
            "processed_size": {
                "width": new_w,
                "height": new_h
            },
            "blur_score_before": blur_score_before,
            "contrast_score_before": contrast_score_before,
            "blur_score_after": ImagePreprocessor._calculate_blur_score(contrast),
            "contrast_score_after": ImagePreprocessor._calculate_contrast_score(contrast)
        }

        return processed_bgr, info

    @staticmethod
    def _readable_preprocess(image):
        resized, scale, old_w, old_h, new_w, new_h = ImagePreprocessor._resize_for_ocr(image)

        resized = ImagePreprocessor._deskew_light(resized)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        blur_score_before = ImagePreprocessor._calculate_blur_score(gray)
        contrast_score_before = ImagePreprocessor._calculate_contrast_score(gray)

        contrast = ImagePreprocessor._apply_clahe(
            gray,
            clip_limit=2.0,
            tile_grid_size=(8, 8)
        )

        denoised = ImagePreprocessor._denoise(contrast, strength=3)

        processed = ImagePreprocessor._adaptive_sharpen(
            denoised,
            blur_score=blur_score_before,
            scale=scale
        )

        processed_bgr = ImagePreprocessor._to_bgr(processed)

        final_h, final_w = processed_bgr.shape[:2]

        info = {
            "variant": "readable",
            "purpose": "일반 OCR 가독성 전처리",
            "scale": scale,
            "original_size": {
                "width": old_w,
                "height": old_h
            },
            "processed_size": {
                "width": final_w,
                "height": final_h
            },
            "blur_score_before": blur_score_before,
            "contrast_score_before": contrast_score_before,
            "blur_score_after": ImagePreprocessor._calculate_blur_score(processed),
            "contrast_score_after": ImagePreprocessor._calculate_contrast_score(processed)
        }

        return processed_bgr, info

    @staticmethod
    def _ocr_enhanced_preprocess(image):
        resized, scale, old_w, old_h, new_w, new_h = ImagePreprocessor._resize_for_ocr(image)

        resized = ImagePreprocessor._deskew_light(resized)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        blur_score_before = ImagePreprocessor._calculate_blur_score(gray)
        contrast_score_before = ImagePreprocessor._calculate_contrast_score(gray)

        if contrast_score_before < 45:
            gray = ImagePreprocessor._gamma_correction(gray, gamma=1.15)

        contrast = ImagePreprocessor._apply_clahe(
            gray,
            clip_limit=2.5,
            tile_grid_size=(8, 8)
        )

        denoise_strength = 3

        if blur_score_before > 180:
            denoise_strength = 2

        elif blur_score_before < 50:
            denoise_strength = 3

        denoised = ImagePreprocessor._denoise(contrast, strength=denoise_strength)

        sharpened = ImagePreprocessor._adaptive_sharpen(
            denoised,
            blur_score=blur_score_before,
            scale=scale
        )

        blur = cv2.GaussianBlur(sharpened, (0, 0), 0.8)

        processed = cv2.addWeighted(sharpened, 1.10, blur, -0.10, 0)

        processed_bgr = ImagePreprocessor._to_bgr(processed)

        final_h, final_w = processed_bgr.shape[:2]

        info = {
            "variant": "ocr_enhanced",
            "purpose": "작은 글자 OCR 강화 전처리",
            "scale": scale,
            "original_size": {
                "width": old_w,
                "height": old_h
            },
            "processed_size": {
                "width": final_w,
                "height": final_h
            },
            "blur_score_before": blur_score_before,
            "contrast_score_before": contrast_score_before,
            "blur_score_after": ImagePreprocessor._calculate_blur_score(processed),
            "contrast_score_after": ImagePreprocessor._calculate_contrast_score(processed)
        }

        return processed_bgr, info

    @staticmethod
    def _soft_binary_preprocess(image):
        resized, scale, old_w, old_h, new_w, new_h = ImagePreprocessor._resize_for_ocr(image)

        resized = ImagePreprocessor._deskew_light(resized)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        blur_score_before = ImagePreprocessor._calculate_blur_score(gray)
        contrast_score_before = ImagePreprocessor._calculate_contrast_score(gray)

        contrast = ImagePreprocessor._apply_clahe(
            gray,
            clip_limit=2.0,
            tile_grid_size=(8, 8)
        )

        blur = cv2.GaussianBlur(contrast, (3, 3), 0)

        block_size = max(11, (new_h // 60) | 1)

        if block_size % 2 == 0:
            block_size += 1

        if contrast_score_before < 30:
            adaptive_c = 4
        elif contrast_score_before < 50:
            adaptive_c = 6
        else:
            adaptive_c = 8

        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            adaptive_c
        )

        processed_bgr = ImagePreprocessor._to_bgr(binary)

        final_h, final_w = processed_bgr.shape[:2]

        info = {
            "variant": "soft_binary",
            "purpose": "보조용 약한 이진화 전처리",
            "scale": scale,
            "block_size": block_size,
            "adaptive_c": adaptive_c,
            "original_size": {
                "width": old_w,
                "height": old_h
            },
            "processed_size": {
                "width": final_w,
                "height": final_h
            },
            "blur_score_before": blur_score_before,
            "contrast_score_before": contrast_score_before,
            "blur_score_after": ImagePreprocessor._calculate_blur_score(binary),
            "contrast_score_after": ImagePreprocessor._calculate_contrast_score(binary)
        }

        return processed_bgr, info

    @staticmethod
    def _high_contrast_binary_preprocess(image):
        """
        [신규 variant] 고대비 전역 이진화 전처리.

        대상:
        - 저화질, 노이즈 많은 스마트폰 사진
        - 라벨이 작게 찍힌 사진
        - 흰 배경 + 검정 글씨인데 OCR이 약한 케이스

        soft_binary는 adaptive(지역 이진화)라 그라데이션/그림자에 강하지만,
        깔끔한 라벨에서는 오히려 노이즈가 추가됨.
        high_contrast_binary는 Otsu(전역 이진화) + 모폴로지로 깔끔한 라벨에서 강함.

        처리 순서:
        1. 자동 확대
        2. 그레이스케일
        3. 강한 CLAHE (clip 3.5)
        4. 가벼운 가우시안 블러 (3x3)
        5. Otsu 이진화
        6. 모폴로지 OPEN (작은 노이즈 제거)
        """

        resized, scale, old_w, old_h, new_w, new_h = ImagePreprocessor._resize_for_ocr(image)

        resized = ImagePreprocessor._deskew_light(resized)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        blur_score_before = ImagePreprocessor._calculate_blur_score(gray)
        contrast_score_before = ImagePreprocessor._calculate_contrast_score(gray)

        # 강한 CLAHE
        contrast = ImagePreprocessor._apply_clahe(
            gray,
            clip_limit=3.5,
            tile_grid_size=(8, 8)
        )

        # 가벼운 블러로 잡티 제거
        smoothed = cv2.GaussianBlur(contrast, (3, 3), 0)

        # Otsu 이진화 - 전역 임계값 자동 결정
        _, binary = cv2.threshold(
            smoothed,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 모폴로지 OPEN으로 작은 점 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        processed_bgr = ImagePreprocessor._to_bgr(cleaned)

        final_h, final_w = processed_bgr.shape[:2]

        info = {
            "variant": "high_contrast_binary",
            "purpose": "고대비 전역 이진화 (저화질/노이즈 많은 사진 대응)",
            "scale": scale,
            "original_size": {
                "width": old_w,
                "height": old_h
            },
            "processed_size": {
                "width": final_w,
                "height": final_h
            },
            "blur_score_before": blur_score_before,
            "contrast_score_before": contrast_score_before,
            "blur_score_after": ImagePreprocessor._calculate_blur_score(cleaned),
            "contrast_score_after": ImagePreprocessor._calculate_contrast_score(cleaned)
        }

        return processed_bgr, info

    @staticmethod
    def _color_clahe_preprocess(image):
        """
        [신규 variant] 컬러 보존 CLAHE 전처리.

        대상:
        - PaddleOCR이 컬러 정보를 활용해 잘 잡는 라벨
        - 색상 라벨이나 컬러 텍스트가 있는 라벨
        - 그레이 변환 시 손실되는 정보가 큰 케이스

        그레이로 변환하지 않고 LAB 색공간의 L 채널에만 CLAHE 적용 후
        다시 BGR로 복원. 색상 정보 보존하면서 대비만 강화.

        처리 순서:
        1. 자동 확대
        2. BGR → LAB 변환
        3. L 채널에만 CLAHE (clip 2.5, tile 8x8)
        4. LAB → BGR 복원
        5. 가벼운 언샤프 마스킹
        """

        resized, scale, old_w, old_h, new_w, new_h = ImagePreprocessor._resize_for_ocr(image)

        resized = ImagePreprocessor._deskew_light(resized)

        gray_for_metrics = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        blur_score_before = ImagePreprocessor._calculate_blur_score(gray_for_metrics)
        contrast_score_before = ImagePreprocessor._calculate_contrast_score(gray_for_metrics)

        # LAB 색공간에서 L 채널에만 CLAHE
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        color_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # 가벼운 언샤프 마스킹
        unsharp_blur = cv2.GaussianBlur(color_enhanced, (0, 0), 1.0)
        processed_bgr = cv2.addWeighted(color_enhanced, 1.3, unsharp_blur, -0.3, 0)

        final_h, final_w = processed_bgr.shape[:2]

        # 결과 측정용
        try:
            gray_after = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY)
            blur_score_after = ImagePreprocessor._calculate_blur_score(gray_after)
            contrast_score_after = ImagePreprocessor._calculate_contrast_score(gray_after)
        except Exception:
            blur_score_after = 0.0
            contrast_score_after = 0.0

        info = {
            "variant": "color_clahe",
            "purpose": "컬러 정보 보존 LAB-CLAHE 전처리",
            "scale": scale,
            "original_size": {
                "width": old_w,
                "height": old_h
            },
            "processed_size": {
                "width": final_w,
                "height": final_h
            },
            "blur_score_before": blur_score_before,
            "contrast_score_before": contrast_score_before,
            "blur_score_after": blur_score_after,
            "contrast_score_after": contrast_score_after
        }

        return processed_bgr, info

    # 기존 이름 호환용
    @staticmethod
    def _basic_preprocess(image):
        return ImagePreprocessor._readable_preprocess(image)

    @staticmethod
    def _enhanced_preprocess(image):
        return ImagePreprocessor._ocr_enhanced_preprocess(image)

    # =========================================================
    # 5. 저장 관련 함수
    # =========================================================

    @staticmethod
    def _safe_filename(value):
        text = str(value).strip()

        if not text:
            return "image"

        unsafe_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', " "]

        for char in unsafe_chars:
            text = text.replace(char, "_")

        return text

    @staticmethod
    def _save_image(image, output_path):
        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        saved = cv2.imwrite(output_path, image)

        if not saved:
            raise ValueError(f"이미지 저장 실패: {output_path}")

        return output_path

    @staticmethod
    def _build_output_path(output_dir, prefix, name, ext="png"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = ImagePreprocessor._safe_filename(name)

        return os.path.join(
            output_dir,
            f"{prefix}_{safe_name}_{timestamp}.{ext}"
        )

    # =========================================================
    # 6. 외부 호출 메서드
    # =========================================================

    @staticmethod
    def preprocess_image(input_path, output_path=None):
        """
        OCR용 대표 이미지 전처리 (기존 코드 호환용)
        한글 경로 대응 _load_image_safe() 사용
        """

        print("[전처리] 이미지 로드 중...")

        image = ImagePreprocessor._load_image_safe(input_path)

        print("[전처리] 원본 이미지 로드 완료")

        height, width = image.shape[:2]

        print(f"[전처리] 원본 크기: {width}x{height}")

        processed, info = ImagePreprocessor._readable_preprocess(image)

        print(
            f"[전처리] readable 전처리 완료 "
            f"({info['original_size']['width']}x{info['original_size']['height']} "
            f"→ {info['processed_size']['width']}x{info['processed_size']['height']})"
        )

        if output_path:
            ImagePreprocessor._save_image(processed, output_path)
            print(f"[전처리] 저장 완료: {output_path}")

        print("[전처리] 기본 전처리 완료")

        return processed

    @staticmethod
    def preprocess_variants_and_save(input_path, output_dir=None):
        """
        전처리 이미지 여러 버전 생성 및 저장.

        [수정]
        - _load_image_safe() 사용 → 한글 경로 대응
        - _detect_dark_background() → 어두운 배경이면 inverted variant 자동 추가
        - inverted variant가 preferred_variants에 포함되어 run_ocr.py에서 점수 비교
        - high_contrast_binary, color_clahe variant 신규 추가
          → 라벨 종류가 다양한 케이스(다양한 배경/색상/화질)에서
            best variant가 항상 1개 이상은 보장되도록 다양성 확보

        반환:
        [
            { "variant": "original",              ... },
            { "variant": "inverted",              ... },  ← 어두운 배경 대응
            { "variant": "layout_safe",           ... },
            { "variant": "readable",              ... },
            { "variant": "ocr_enhanced",          ... },
            { "variant": "soft_binary",           ... },  ← 적응형 이진화
            { "variant": "high_contrast_binary",  ... },  ← 전역 이진화 (신규)
            { "variant": "color_clahe",           ... }   ← 컬러 보존 (신규)
        ]
        """

        print("[전처리] 이미지 로드 중...")

        image = ImagePreprocessor._load_image_safe(input_path)

        filename = os.path.basename(input_path)
        name, _ = os.path.splitext(filename)

        if output_dir is None:
            output_dir = os.path.join("outputs", "preprocess")

        os.makedirs(output_dir, exist_ok=True)

        original_height, original_width = image.shape[:2]

        print(f"[전처리] 원본 크기: {original_width}x{original_height}")

        # 어두운 배경 자동 감지
        is_dark = ImagePreprocessor._detect_dark_background(image)
        if is_dark:
            print("[전처리] 어두운 배경 감지 → inverted variant 우선 후보")

        variants = []

        # ---------------------------------------------
        # original
        # ---------------------------------------------
        try:
            print("[전처리] original variant 등록 중...")

            original_image, original_info = ImagePreprocessor._original_variant(image)

            variants.append(
                {
                    "variant": "original",
                    "image": original_image,
                    "path": input_path,
                    "info": original_info
                }
            )

            print("[전처리] original 등록 완료")

        except Exception as error:
            print(f"[전처리 경고] original 등록 실패: {error}")

        # ---------------------------------------------
        # inverted (어두운 배경 감지 시 항상 생성)
        # 어두운 배경이 아니어도 보조 variant로 생성
        # → run_ocr.py에서 점수 비교 후 자동 선택
        # ---------------------------------------------
        try:
            print("[전처리] inverted 전처리 생성 중...")

            inverted_image, inverted_info = ImagePreprocessor._inverted_preprocess(image)

            inverted_path = ImagePreprocessor._build_output_path(
                output_dir=output_dir,
                prefix="processed_inverted",
                name=name,
                ext="png"
            )

            ImagePreprocessor._save_image(inverted_image, inverted_path)

            variants.append(
                {
                    "variant": "inverted",
                    "image": inverted_image,
                    "path": inverted_path,
                    "info": inverted_info
                }
            )

            print(f"[전처리] inverted 저장 완료: {inverted_path}")

        except Exception as error:
            print(f"[전처리 경고] inverted 생성 실패: {error}")

        # ---------------------------------------------
        # layout_safe
        # ---------------------------------------------
        try:
            print("[전처리] layout_safe 전처리 생성 중...")

            layout_safe_image, layout_safe_info = ImagePreprocessor._layout_safe_preprocess(image)

            layout_safe_path = ImagePreprocessor._build_output_path(
                output_dir=output_dir,
                prefix="processed_layout_safe",
                name=name,
                ext="png"
            )

            ImagePreprocessor._save_image(layout_safe_image, layout_safe_path)

            variants.append(
                {
                    "variant": "layout_safe",
                    "image": layout_safe_image,
                    "path": layout_safe_path,
                    "info": layout_safe_info
                }
            )

            print(f"[전처리] layout_safe 저장 완료: {layout_safe_path}")

        except Exception as error:
            print(f"[전처리 경고] layout_safe 생성 실패: {error}")

        # ---------------------------------------------
        # readable
        # ---------------------------------------------
        try:
            print("[전처리] readable 전처리 생성 중...")

            readable_image, readable_info = ImagePreprocessor._readable_preprocess(image)

            readable_path = ImagePreprocessor._build_output_path(
                output_dir=output_dir,
                prefix="processed_readable",
                name=name,
                ext="png"
            )

            ImagePreprocessor._save_image(readable_image, readable_path)

            variants.append(
                {
                    "variant": "readable",
                    "image": readable_image,
                    "path": readable_path,
                    "info": readable_info
                }
            )

            print(f"[전처리] readable 저장 완료: {readable_path}")

        except Exception as error:
            print(f"[전처리 경고] readable 생성 실패: {error}")

        # ---------------------------------------------
        # ocr_enhanced
        # ---------------------------------------------
        try:
            print("[전처리] ocr_enhanced 전처리 생성 중...")

            ocr_enhanced_image, ocr_enhanced_info = ImagePreprocessor._ocr_enhanced_preprocess(image)

            ocr_enhanced_path = ImagePreprocessor._build_output_path(
                output_dir=output_dir,
                prefix="processed_ocr_enhanced",
                name=name,
                ext="png"
            )

            ImagePreprocessor._save_image(ocr_enhanced_image, ocr_enhanced_path)

            variants.append(
                {
                    "variant": "ocr_enhanced",
                    "image": ocr_enhanced_image,
                    "path": ocr_enhanced_path,
                    "info": ocr_enhanced_info
                }
            )

            print(f"[전처리] ocr_enhanced 저장 완료: {ocr_enhanced_path}")

        except Exception as error:
            print(f"[전처리 경고] ocr_enhanced 생성 실패: {error}")

        # ---------------------------------------------
        # soft_binary (기존 - 적응형 이진화)
        # ---------------------------------------------
        try:
            print("[전처리] soft_binary 전처리 생성 중...")

            soft_binary_image, soft_binary_info = ImagePreprocessor._soft_binary_preprocess(image)

            soft_binary_path = ImagePreprocessor._build_output_path(
                output_dir=output_dir,
                prefix="processed_soft_binary",
                name=name,
                ext="png"
            )

            ImagePreprocessor._save_image(soft_binary_image, soft_binary_path)

            variants.append(
                {
                    "variant": "soft_binary",
                    "image": soft_binary_image,
                    "path": soft_binary_path,
                    "info": soft_binary_info
                }
            )

            print(f"[전처리] soft_binary 저장 완료: {soft_binary_path}")

        except Exception as error:
            print(f"[전처리 경고] soft_binary 생성 실패: {error}")

        # ---------------------------------------------
        # [신규] high_contrast_binary (전역 이진화)
        # 저화질/노이즈 많은 라벨 사진에 강함
        # ---------------------------------------------
        try:
            print("[전처리] high_contrast_binary 전처리 생성 중...")

            hcb_image, hcb_info = ImagePreprocessor._high_contrast_binary_preprocess(image)

            hcb_path = ImagePreprocessor._build_output_path(
                output_dir=output_dir,
                prefix="processed_high_contrast_binary",
                name=name,
                ext="png"
            )

            ImagePreprocessor._save_image(hcb_image, hcb_path)

            variants.append(
                {
                    "variant": "high_contrast_binary",
                    "image": hcb_image,
                    "path": hcb_path,
                    "info": hcb_info
                }
            )

            print(f"[전처리] high_contrast_binary 저장 완료: {hcb_path}")

        except Exception as error:
            print(f"[전처리 경고] high_contrast_binary 생성 실패: {error}")

        # ---------------------------------------------
        # [신규] color_clahe (LAB 컬러 보존 CLAHE)
        # PaddleOCR이 컬러 정보로 잘 잡는 라벨에 강함
        # ---------------------------------------------
        try:
            print("[전처리] color_clahe 전처리 생성 중...")

            cc_image, cc_info = ImagePreprocessor._color_clahe_preprocess(image)

            cc_path = ImagePreprocessor._build_output_path(
                output_dir=output_dir,
                prefix="processed_color_clahe",
                name=name,
                ext="png"
            )

            ImagePreprocessor._save_image(cc_image, cc_path)

            variants.append(
                {
                    "variant": "color_clahe",
                    "image": cc_image,
                    "path": cc_path,
                    "info": cc_info
                }
            )

            print(f"[전처리] color_clahe 저장 완료: {cc_path}")

        except Exception as error:
            print(f"[전처리 경고] color_clahe 생성 실패: {error}")

        print(f"[전처리] 전처리 이미지 여러 버전 생성 완료: {len(variants)}개")

        return variants

    @staticmethod
    def preprocess_and_save(input_path, output_dir=None):
        """기존 코드 호환용."""

        filename = os.path.basename(input_path)
        name, _ = os.path.splitext(filename)

        if output_dir is None:
            output_dir = os.path.join("outputs", "preprocess")

        os.makedirs(output_dir, exist_ok=True)

        output_path = ImagePreprocessor._build_output_path(
            output_dir=output_dir,
            prefix="processed",
            name=name,
            ext="png"
        )

        processed_image = ImagePreprocessor.preprocess_image(input_path, output_path)

        return processed_image, output_path