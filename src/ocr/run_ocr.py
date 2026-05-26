import os
import re
import cv2
import statistics
from datetime import datetime
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from google.cloud import vision
from google.oauth2 import service_account

from src.ocr.preprocess import ImagePreprocessor
from src.utils.file_io import FileIO

# Google Vision 인증 키 경로(GOOGLE_APPLICATION_CREDENTIALS)는 .env 에 정의돼 있다.
# OCRRunner 가 단독 import 되어도 동작하도록 모듈 로딩 시점에 한 번 로딩.
load_dotenv()


class OCRRunner:
    """
    Dermalens OCR Runner

    Claude API를 사용하지 않는 OCR 중심 구조.

    역할:
    1. 화장품 라벨 이미지를 OCR 한다.
    2. 원본 이미지와 여러 전처리 이미지 variant를 모두 OCR 한다.
    3. OCR 결과 중 가장 안정적인 결과를 선택한다.
    4. 텍스트뿐 아니라 bbox 좌표, 줄 정보, confidence를 보존한다.
    5. 이후 section_detector.py에서 제품명/용량/성분/사용방법/주의사항/기타를
       일반화해서 분류할 수 있도록 데이터를 제공한다.

    중요:
    - 이 파일에서는 제품명/성분/사용방법/주의사항을 확정 분류하지 않는다.
    - 성분표마다 순서와 배치가 다르므로 OCR 단계에서는 전체 텍스트와 좌표를 최대한 보존한다.
    - 성분 API 검증은 이 파일에서 하지 않는다.
    - 성분 API 전/후 결과는 postprocess.py + ingredient_api.py + main.py에서 최종 JSON에 담는다.

    [수정 사항]

    (A) _group_blocks_into_lines() — y축 임계값 동적 계산
        기존: line별 avg_height * 0.75 고정값
              → 작은 이미지는 다른 줄이 합쳐지고, 큰 이미지는 한 줄이 쪼개짐
        수정: 전체 블록의 global_median_height를 먼저 계산 후
              y_threshold = max(global_median_height * 0.55, 6.0) 적용
              → 이미지 해상도/폰트 크기에 무관하게 줄 그룹화 안정화

    (B) _merge_line_text() — global_median_height 인자 추가
        _group_blocks_into_lines()에서 계산한 global_median_height를
        _merge_line_text()로 전달해 공백 판단 기준도 동일 스케일로 맞춤

    (C) _need_space_between_blocks() — 공백 판단 임계값 상향 조정
        기존: normal_gap_threshold = median_height * 0.25
              → 고해상도 이미지에서 불필요한 공백이 너무 많이 삽입됨
        수정: normal_gap_threshold = median_height * 0.35 (0.25→0.35)
              small_gap_threshold = median_height * 0.15 (0.12→0.15)
              최소값 5px (4→5)

    (D) _build_layout_text_from_lines() — 구역 경계 임계값 동적화
        기존: median_height * 1.8 고정
              → 이미지마다 줄 간격이 달라서 구역 분리가 부정확
        수정: _calc_section_break_threshold() 신설
              전체 라인 y gap의 75th percentile * 1.5를 임계값으로 사용
              → 대부분의 줄 간격보다 현저히 큰 gap만 구역 경계로 판단

    (E) _score_full_ocr_result() — variant별 가중치 균등화
        기존: readable +0.08, original +0.06 등 특정 variant 편향
              → 어두운/고해상도 이미지에서 오히려 다른 variant가 더 나을 수 있음
        수정: 모든 variant에 동일한 +0.03 적용
              → 실제 텍스트 품질(confidence, 라인 수, 키워드 포함)로만 판단

    [이번 수정 사항 — 2단계 OCR 강화]

    (F) __init__() — PaddleOCR 옵션 강화 + 호환성 fallback
        기존: lang, use_angle_cls, show_log만 사용
              → 작은 글씨 박스 누락, 박스 끝 잘림
        수정: det_db_box_thresh=0.3 (작은 글씨도 박스 잡힘)
              det_db_unclip_ratio=2.0 (박스 확장으로 글자 끝까지 포함)
              drop_score=0.15 (일단 다 뽑고 min_confidence로 필터)
              rec_batch_num=6 (많은 박스 처리시 안정성)
              + try-except로 PaddleOCR 버전 호환성 보장

    (G) preferred_variants에 신규 variant 2개 추가
        high_contrast_binary (Otsu 전역 이진화 - 저화질 라벨)
        color_clahe (LAB 컬러 보존 CLAHE - 컬러 라벨)

    (H) _score_full_ocr_result() — variant_info 기반 컨텍스트 보너스
        기존: variant 이름과 무관하게 동일 보정
              → 어두운 배경 라벨에서 inverted가 명확히 더 나아도 차이 적음
        수정: variant_info["source_is_dark_background"]가 True일 때
              - inverted variant +0.10 보너스
              - 비-inverted variant -0.05 감점
              이진화 계열에서 confidence ≥ 0.85면 +0.05 보너스
              bad_chars 감점 강화 0.04 → 0.06

    (I) _clean_common_text() — replace_map 확장
        25개 → 약 60개로 확장
        - 단위 추가 (㎕, ㎍, ㎤)
        - 전각 숫자/% → 반각 정규화
        - 라벨 장식 기호(▶, ★, ✓ 등) 제거
        - 가운뎃점, 괄호 변형 정규화
    """

    def __init__(self):
        # ========================================================
        # Google Vision API 초기화
        #
        # PaddleOCR → Google Cloud Vision (DOCUMENT_TEXT_DETECTION) 으로 교체.
        # DOCUMENT_TEXT_DETECTION 은 한글 라벨처럼 작은 글씨/다양한 폰트가
        # 섞인 문서형 이미지에 가장 적합한 모드.
        #
        # 인증:
        # - .env 의 GOOGLE_APPLICATION_CREDENTIALS 가 가리키는 JSON 키 사용
        # - 없으면 ADC(Application Default Credentials) 로 fallback
        # ========================================================

        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if cred_path and os.path.exists(cred_path):
            credentials = service_account.Credentials.from_service_account_file(cred_path)
            self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            print(f"[OCR 초기화] Google Vision 클라이언트 생성 완료 (key: {cred_path})")
        else:
            self.vision_client = vision.ImageAnnotatorClient()
            print("[OCR 초기화] Google Vision 클라이언트 생성 완료 (ADC 사용)")

        # 한국어 + 영어 hint - 화장품 라벨 대응
        self.vision_image_context = vision.ImageContext(language_hints=["ko", "en"])

        # confidence 임계값 (PaddleOCR 때와 동일 의미)
        # Google Vision 의 word.confidence 가 0.0~1.0 범위라 그대로 사용
        self.min_confidence = 0.20
        self.min_text_length = 1

        # Google Vision 은 1장당 1회 호출만 수행.
        # PaddleOCR 시절엔 전처리 variant 여러 개를 OCR 한 뒤 점수로 골랐지만
        # Vision DOCUMENT_TEXT_DETECTION 은 작은 글씨/색배경/뒤집힘에 자체적으로 강건해서
        # "original" 1개만으로 충분하며, 호출 수 = 비용 이므로 9배 절감.
        # 추후 특정 케이스에서 품질이 부족하면 inverted 등을 추가하면 됨.
        self.preferred_variants = ["original"]

        self.ocr_init_options = {
            "engine": "google_vision_document_text_detection",
            "language_hints": ["ko", "en"]
        }

    # =========================================================
    # 1. 전체 이미지 OCR 실행
    # =========================================================

    def run(
        self,
        image_path: str,
        save_text: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        전체 이미지 OCR 실행 함수

        Parameters
        ----------
        image_path : str
            분석할 이미지 경로

        save_text : bool
            OCR raw text 저장 여부

        Returns
        -------
        dict | None
            {
                "success": True,
                "mode": "full_image_ocr_with_bbox",
                "image_path": "...",
                "selected_variant": "readable",
                "selected_variant_path": "...",
                "raw_text": "...",
                "layout_text": "...",
                "ocr_blocks": [...],
                "ocr_lines": [...],
                "image_size": {
                    "width": 원본 이미지 width,
                    "height": 원본 이미지 height
                },
                "selected_variant_size": {
                    "width": 선택된 전처리 이미지 width,
                    "height": 선택된 전처리 이미지 height
                },
                "ocr_summary": {...},
                "variant_results": [...],
                "saved_text_path": "...",
                "created_at": "..."
            }
        """

        print("[진행중] 전체 이미지 OCR 시작")

        if not image_path:
            print("[오류] image_path가 비어 있습니다.")
            return None

        if not os.path.exists(image_path):
            print(f"[오류] 이미지 파일을 찾을 수 없습니다: {image_path}")
            return None

        original_image = cv2.imread(image_path)

        if original_image is None:
            print(f"[오류] 이미지를 읽을 수 없습니다: {image_path}")
            return None

        image_height, image_width = original_image.shape[:2]

        print(f"[정보] 원본 이미지 크기: {image_width}x{image_height}")

        variants = self._build_full_image_variants(
            image_path=image_path,
            original_image=original_image
        )

        if not variants:
            print("[오류] OCR 대상 이미지 variant를 생성하지 못했습니다.")
            return None

        all_variant_results = []

        for variant in variants:
            variant_name = variant.get("variant", "unknown")
            variant_image = variant.get("image")
            variant_path = variant.get("path")
            variant_info = variant.get("info", {})

            if variant_image is None:
                continue

            print(f"[진행중] {variant_name} OCR 수행 중...")

            try:
                one_result = self._run_ocr_on_variant(
                    variant_name=variant_name,
                    variant_image=variant_image,
                    variant_path=variant_path,
                    variant_info=variant_info,
                    original_image_size={
                        "width": image_width,
                        "height": image_height
                    }
                )

                if one_result is None:
                    print(f"[경고] {variant_name} OCR 결과 없음")
                    continue

                all_variant_results.append(one_result)

                print(
                    f"[완료] {variant_name} OCR 완료 "
                    f"(block: {len(one_result.get('ocr_blocks', []))}개, "
                    f"line: {len(one_result.get('ocr_lines', []))}개, "
                    f"confidence: {one_result.get('confidence_avg', 0):.3f}, "
                    f"score: {one_result.get('score', 0):.3f})"
                )

            except Exception as error:
                print(f"[경고] {variant_name} OCR 실패: {error}")

        if not all_variant_results:
            print("[경고] 전체 OCR 결과가 비어 있습니다.")
            return None

        best_result = self._select_best_variant_result(all_variant_results)

        raw_text = best_result.get("raw_text", "")
        layout_text = best_result.get("layout_text", raw_text)

        saved_text_path = None

        if save_text:
            try:
                saved_text_path = FileIO.save_raw_ocr_text(raw_text)
                print(f"[완료] OCR 텍스트 저장 완료: {saved_text_path}")

            except Exception as error:
                print(f"[경고] OCR 텍스트 저장 실패: {error}")

        final_result = {
            "success": True,
            "mode": "full_image_ocr_with_bbox",
            "image_path": image_path,
            "selected_variant": best_result.get("variant"),
            "selected_variant_path": best_result.get("variant_path"),
            "raw_text": raw_text,
            "layout_text": layout_text,
            "ocr_blocks": best_result.get("ocr_blocks", []),
            "ocr_lines": best_result.get("ocr_lines", []),
            "image_size": {
                "width": image_width,
                "height": image_height
            },
            "selected_variant_size": best_result.get(
                "selected_variant_size",
                {
                    "width": image_width,
                    "height": image_height
                }
            ),
            "ocr_summary": {
                "block_count": len(best_result.get("ocr_blocks", [])),
                "line_count": len(best_result.get("ocr_lines", [])),
                "confidence_avg": best_result.get("confidence_avg", 0.0),
                "score": best_result.get("score", 0.0),
                "text_length": len(raw_text),
                "selected_variant": best_result.get("variant")
            },
            "variant_results": self._summarize_variant_results(all_variant_results),
            "saved_text_path": saved_text_path,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print(
            f"[완료] 전체 이미지 OCR 완료 "
            f"- 선택 variant: {final_result['selected_variant']}"
        )

        return final_result

    # =========================================================
    # 2. 단일 variant OCR
    # =========================================================

    def _run_ocr_on_variant(
        self,
        variant_name: str,
        variant_image,
        variant_path: Optional[str],
        variant_info: Dict[str, Any],
        original_image_size: Dict[str, int]
    ) -> Optional[Dict[str, Any]]:

        if variant_image is None:
            return None

        variant_height, variant_width = variant_image.shape[:2]

        raw_ocr_result = self._call_google_vision(variant_image)

        ocr_blocks = self._extract_blocks_from_ocr_result(raw_ocr_result)

        if not ocr_blocks:
            return None

        ocr_lines = self._group_blocks_into_lines(ocr_blocks)

        if not ocr_lines:
            return None

        raw_text = self._build_raw_text_from_lines(ocr_lines)
        layout_text = self._build_layout_text_from_lines(ocr_lines)

        if not raw_text:
            return None

        confidence_avg = self._average_confidence_from_blocks(ocr_blocks)

        score = self._score_full_ocr_result(
            raw_text=raw_text,
            ocr_blocks=ocr_blocks,
            ocr_lines=ocr_lines,
            confidence_avg=confidence_avg,
            variant_name=variant_name,
            variant_info=variant_info
        )

        return {
            "variant": variant_name,
            "variant_path": variant_path,
            "variant_info": variant_info,
            "raw_text": raw_text,
            "layout_text": layout_text,
            "ocr_blocks": ocr_blocks,
            "ocr_lines": ocr_lines,
            "confidence_avg": confidence_avg,
            "score": score,
            "image_size": original_image_size,
            "selected_variant_size": {
                "width": variant_width,
                "height": variant_height
            }
        }

    # =========================================================
    # 3. 이미지 variant 생성
    # =========================================================

    def _build_full_image_variants(
        self,
        image_path: str,
        original_image
    ) -> List[Dict[str, Any]]:
        """
        전체 이미지 OCR용 variant 생성

        원칙:
        - original은 무조건 포함한다.
        - preprocess.py에서 만든 여러 전처리 variant를 가져온다.
        - 각 variant를 모두 OCR에 넣고, 결과 점수로 최종 선택한다.
        """

        variants = []

        variants.append(
            {
                "variant": "original",
                "image": original_image,
                "path": image_path,
                "info": {
                    "variant": "original",
                    "description": "원본 이미지"
                }
            }
        )

        try:
            preprocessed_variants = ImagePreprocessor.preprocess_variants_and_save(
                image_path
            )

            if isinstance(preprocessed_variants, list):
                for item in preprocessed_variants:
                    if not isinstance(item, dict):
                        continue

                    variant_name = item.get("variant", "unknown")

                    if variant_name not in self.preferred_variants:
                        continue

                    # original은 위에서 이미 넣었으므로 중복 제거
                    if variant_name == "original":
                        continue

                    variant_image = item.get("image")

                    if variant_image is None:
                        continue

                    variants.append(
                        {
                            "variant": variant_name,
                            "image": variant_image,
                            "path": item.get("path"),
                            "info": item.get("info", {})
                        }
                    )

        except AttributeError:
            print("[경고] preprocess_variants_and_save()가 없어 preprocess_and_save()를 사용합니다.")

            try:
                processed_image, processed_path = ImagePreprocessor.preprocess_and_save(
                    image_path
                )

                if processed_image is not None:
                    variants.append(
                        {
                            "variant": "processed",
                            "image": processed_image,
                            "path": processed_path,
                            "info": {
                                "variant": "processed",
                                "description": "기본 전처리 이미지"
                            }
                        }
                    )

            except Exception as error:
                print(f"[경고] 기본 전처리 실패: {error}")

        except Exception as error:
            print(f"[경고] 전처리 variant 생성 실패: {error}")

        variants = self._deduplicate_variants(variants)

        print(f"[정보] OCR 대상 variant 수: {len(variants)}개")

        for variant in variants:
            print(f" - {variant.get('variant')}")

        return variants

    def _deduplicate_variants(
        self,
        variants: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        result = []
        seen = set()

        for item in variants:
            variant_name = item.get("variant", "unknown")
            path = item.get("path") or ""

            key = f"{variant_name}|{path}"

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    # =========================================================
    # 4-1. Google Vision 호출 + PaddleOCR 호환 포맷 변환
    # =========================================================

    def _call_google_vision(self, variant_image) -> List[List[Any]]:
        """
        Google Vision DOCUMENT_TEXT_DETECTION 호출 후
        PaddleOCR 의 결과 포맷 [[ [box, (text, conf)], ... ]] 로 변환한다.

        word 단위로 풀어줘야 기존 _group_blocks_into_lines() 가
        PaddleOCR 때와 동일하게 동작한다 (성분명 한 단어 = 1 block).

        Parameters
        ----------
        variant_image : numpy.ndarray
            cv2 BGR 이미지

        Returns
        -------
        list
            PaddleOCR 호환 결과: [[ [box4점, (text, conf)], ... ]]
            (page 1개로 감싼 구조)
        """

        if variant_image is None:
            return [[]]

        # cv2 (BGR ndarray) → PNG 바이트로 인코딩
        success, encoded = cv2.imencode(".png", variant_image)
        if not success:
            return [[]]

        content = encoded.tobytes()
        image = vision.Image(content=content)

        response = self.vision_client.document_text_detection(
            image=image,
            image_context=self.vision_image_context,
        )

        if response.error.message:
            raise RuntimeError(f"[Google Vision 오류] {response.error.message}")

        annotation = response.full_text_annotation
        if not annotation or not annotation.pages:
            return [[]]

        word_items: List[List[Any]] = []

        # 페이지 → 블록 → 문단 → 단어 순서로 풀기
        for page in annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        text = "".join(s.text for s in word.symbols).strip()
                        if not text:
                            continue

                        confidence = float(word.confidence) if word.confidence else 0.0

                        vertices = word.bounding_box.vertices
                        if not vertices or len(vertices) < 4:
                            continue

                        # PaddleOCR 호환 box: 4점 (x, y) 좌표
                        box = [[float(v.x or 0), float(v.y or 0)] for v in vertices]

                        # PaddleOCR 포맷: [box, (text, confidence)]
                        word_items.append([box, (text, confidence)])

        # PaddleOCR 은 [page_result] 로 감싸서 반환하므로 동일하게 맞춤
        return [word_items]

    # =========================================================
    # 4. PaddleOCR 호환 결과에서 block 추출
    # =========================================================

    def _extract_blocks_from_ocr_result(
        self,
        raw_ocr_result
    ) -> List[Dict[str, Any]]:

        blocks = []

        if not raw_ocr_result:
            return blocks

        block_index = 0

        for page in raw_ocr_result:
            if page is None:
                continue

            for word_info in page:
                try:
                    box = word_info[0]
                    text = str(word_info[1][0]).strip()
                    confidence = float(word_info[1][1])

                    if confidence < self.min_confidence:
                        continue

                    text = self._clean_common_text(text)

                    if not text:
                        continue

                    if len(text) < self.min_text_length:
                        continue

                    if self._is_noise_text(text):
                        continue

                    x_values = [float(point[0]) for point in box]
                    y_values = [float(point[1]) for point in box]

                    x1 = min(x_values)
                    x2 = max(x_values)
                    y1 = min(y_values)
                    y2 = max(y_values)

                    width = x2 - x1
                    height = y2 - y1

                    if width <= 0 or height <= 0:
                        continue

                    block = {
                        "block_index": block_index,
                        "text": text,
                        "confidence": confidence,
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        },
                        "points": [
                            {
                                "x": float(point[0]),
                                "y": float(point[1])
                            }
                            for point in box
                        ],
                        "x_min": x1,
                        "y_min": y1,
                        "x_max": x2,
                        "y_max": y2,
                        "center_x": (x1 + x2) / 2,
                        "center_y": (y1 + y2) / 2,
                        "width": width,
                        "height": height,
                        "line_index": None
                    }

                    blocks.append(block)
                    block_index += 1

                except Exception:
                    continue

        return sorted(
            blocks,
            key=lambda item: (
                item.get("center_y", 0),
                item.get("x_min", 0)
            )
        )

    # =========================================================
    # 5. block을 line으로 그룹화
    # =========================================================

    def _group_blocks_into_lines(
        self,
        blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        block들을 같은 줄로 그룹화한다.

        [수정] y축 임계값 동적 계산
        기존: line별 avg_height * 0.75 고정
              → 블록을 하나씩 추가하면서 해당 라인의 avg_height로 판단하기 때문에
                이미지 크기/폰트마다 같은 줄이 합쳐지거나 쪼개지는 문제 발생

        수정: 루프 시작 전 전체 블록의 global_median_height를 먼저 계산
              y_threshold = max(global_median_height * 0.55, 6.0) 로 고정
              → 이미지 스케일에 맞는 일관된 기준 적용
              → 0.55: 줄 높이의 절반 이하 차이면 같은 줄
              → 최소 6px: 매우 작은 이미지에서도 기준 보장
        """

        if not blocks:
            return []

        # [수정] 전체 블록 height 중앙값으로 y_threshold 사전 계산
        all_heights = [
            block.get("height", 0)
            for block in blocks
            if block.get("height", 0) > 0
        ]

        global_median_height = float(statistics.median(all_heights)) if all_heights else 14.0

        # 줄 높이의 55% 이내 차이면 같은 줄, 최소 6px 보장
        y_threshold = max(global_median_height * 0.55, 6.0)

        sorted_blocks = sorted(
            blocks,
            key=lambda item: (
                item.get("center_y", 0),
                item.get("x_min", 0)
            )
        )

        line_groups = []

        for block in sorted_blocks:
            matched_line = None

            for line in line_groups:
                # [수정] 고정된 y_threshold 사용 (기존: line별 avg_height * 0.75)
                if abs(block["center_y"] - line["center_y"]) <= y_threshold:
                    matched_line = line
                    break

            if matched_line is None:
                matched_line = {
                    "center_y": block["center_y"],
                    "avg_height": block["height"],
                    "blocks": []
                }
                line_groups.append(matched_line)

            matched_line["blocks"].append(block)

            centers = [
                item["center_y"]
                for item in matched_line["blocks"]
            ]

            heights = [
                item["height"]
                for item in matched_line["blocks"]
            ]

            matched_line["center_y"] = sum(centers) / len(centers)
            matched_line["avg_height"] = sum(heights) / len(heights)

        ocr_lines = []

        line_groups = sorted(
            line_groups,
            key=lambda item: item.get("center_y", 0)
        )

        for line_index, line in enumerate(line_groups):
            line_blocks = sorted(
                line.get("blocks", []),
                key=lambda item: item.get("x_min", 0)
            )

            # [수정] global_median_height 전달
            line_text = self._merge_line_text(
                blocks=line_blocks,
                global_median_height=global_median_height
            )

            if not line_text:
                continue

            if self._is_noise_text(line_text):
                continue

            confidences = [
                block.get("confidence", 0.0)
                for block in line_blocks
            ]

            confidence_avg = (
                sum(confidences) / len(confidences)
                if confidences
                else 0.0
            )

            x1 = min(block.get("x_min", 0) for block in line_blocks)
            y1 = min(block.get("y_min", 0) for block in line_blocks)
            x2 = max(block.get("x_max", 0) for block in line_blocks)
            y2 = max(block.get("y_max", 0) for block in line_blocks)

            for block in line_blocks:
                block["line_index"] = line_index

            ocr_lines.append(
                {
                    "line_index": line_index,
                    "text": line_text,
                    "confidence": confidence_avg,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                    "center_x": (x1 + x2) / 2,
                    "center_y": (y1 + y2) / 2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "blocks": line_blocks
                }
            )

        return sorted(
            ocr_lines,
            key=lambda item: (
                item.get("center_y", 0),
                item.get("x_min", 0)
            )
        )

    def _merge_line_text(
        self,
        blocks: List[Dict[str, Any]],
        global_median_height: float = 14.0
    ) -> str:
        """
        같은 줄의 block들을 텍스트로 합친다.

        [수정] global_median_height 인자 추가
        _group_blocks_into_lines()에서 계산한 이미지 스케일 기준값을
        공백 판단에도 동일하게 적용한다.
        라인 내 블록이 1개뿐인 경우 fallback으로 사용한다.
        """

        if not blocks:
            return ""

        blocks = sorted(
            blocks,
            key=lambda item: item.get("x_min", 0)
        )

        heights = [
            block.get("height", 0)
            for block in blocks
            if block.get("height", 0) > 0
        ]

        widths = [
            block.get("width", 0)
            for block in blocks
            if block.get("width", 0) > 0
        ]

        # 라인 내 median이 없으면 global 값으로 fallback
        median_height = statistics.median(heights) if heights else global_median_height
        median_width = statistics.median(widths) if widths else 20

        merged_parts = []
        previous_block = None

        for block in blocks:
            text = self._clean_common_text(block.get("text", ""))

            if not text:
                continue

            if previous_block is None:
                merged_parts.append(text)
                previous_block = block
                continue

            gap = block.get("x_min", 0) - previous_block.get("x_max", 0)

            if self._need_space_between_blocks(
                previous_text=merged_parts[-1],
                current_text=text,
                gap=gap,
                median_height=median_height,
                median_width=median_width
            ):
                merged_parts.append(" ")

            merged_parts.append(text)
            previous_block = block

        merged = "".join(merged_parts)

        return self._clean_common_text(merged)

    # =========================================================
    # 6. OCR 텍스트 생성
    # =========================================================

    def _build_raw_text_from_lines(
        self,
        ocr_lines: List[Dict[str, Any]]
    ) -> str:

        texts = []

        for line in ocr_lines:
            text = self._clean_common_text(line.get("text", ""))

            if not text:
                continue

            texts.append(text)

        return "\n".join(texts).strip()

    def _build_layout_text_from_lines(
        self,
        ocr_lines: List[Dict[str, Any]]
    ) -> str:
        """
        좌표 기반 layout text 생성

        목적:
        - section_detector.py가 위아래 흐름을 파악할 수 있게 함
        - 줄 사이 간격이 큰 경우 빈 줄을 넣어 구역 분리 힌트를 제공

        [수정] 구역 경계 임계값 동적화
        기존: median_height * 1.8 고정
              → 이미지마다 줄 간격이 달라 구역 분리가 부정확
              → 작은 이미지에선 모든 줄에 빈 줄이 삽입되거나
                큰 이미지에선 구역 경계를 못 잡음

        수정: _calc_section_break_threshold() 신설
              전체 라인 y gap의 75th percentile * 1.5를 임계값으로 사용
              → 대부분의 일반 줄 간격보다 현저히 큰 gap만 구역 경계로 판단
              → 이미지 스케일에 무관하게 동작
        """

        if not ocr_lines:
            return ""

        sorted_lines = sorted(
            ocr_lines,
            key=lambda item: (
                item.get("center_y", 0),
                item.get("x_min", 0)
            )
        )

        # [수정] 동적 임계값 계산
        section_break_threshold = self._calc_section_break_threshold(sorted_lines)

        result_lines = []
        previous_y = None

        for line in sorted_lines:
            text = self._clean_common_text(line.get("text", ""))

            if not text:
                continue

            current_y = line.get("center_y", 0)

            if previous_y is not None:
                y_gap = current_y - previous_y

                # [수정] 고정 median_height * 1.8 → 동적 threshold
                if y_gap > section_break_threshold:
                    result_lines.append("")

            result_lines.append(text)
            previous_y = current_y

        return "\n".join(result_lines).strip()

    def _calc_section_break_threshold(
        self,
        ocr_lines: List[Dict[str, Any]]
    ) -> float:
        """
        줄 간 y gap 분포에서 구역 경계 판단 임계값을 동적으로 계산한다.

        방식:
        - 인접 라인 간 y gap을 모두 수집
        - 75th percentile * 1.5를 임계값으로 사용
        - 이 값보다 큰 gap은 구역 경계(빈 줄 삽입)로 판단

        fallback:
        - 라인이 부족하거나 gap이 없으면 median_height * 2.0 사용
        """

        if len(ocr_lines) < 2:
            return self._median_line_height(ocr_lines) * 2.0

        sorted_lines = sorted(
            ocr_lines,
            key=lambda item: item.get("center_y", 0)
        )

        y_gaps = []

        for i in range(1, len(sorted_lines)):
            prev_y = sorted_lines[i - 1].get("center_y", 0)
            curr_y = sorted_lines[i].get("center_y", 0)
            gap = curr_y - prev_y

            if gap > 0:
                y_gaps.append(gap)

        if not y_gaps:
            return self._median_line_height(ocr_lines) * 2.0

        y_gaps.sort()

        # 75th percentile
        p75_idx = int(len(y_gaps) * 0.75)
        p75 = y_gaps[min(p75_idx, len(y_gaps) - 1)]

        threshold = p75 * 1.5

        # 최소값 보장 (median_height * 1.5 이상)
        median_h = self._median_line_height(ocr_lines)
        threshold = max(threshold, median_h * 1.5)

        return threshold

    def _median_line_height(
        self,
        ocr_lines: List[Dict[str, Any]]
    ) -> float:

        heights = [
            line.get("height", 0)
            for line in ocr_lines
            if line.get("height", 0) > 0
        ]

        if not heights:
            return 14.0

        return float(statistics.median(heights))

    # =========================================================
    # 7. OCR variant 선택 점수
    # =========================================================

    def _select_best_variant_result(
        self,
        variant_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        OCR variant 중 가장 안정적인 결과 선택

        기준:
        - 평균 confidence
        - 읽을 수 있는 문자 비율
        - block 수
        - line 수
        - 화장품 라벨에서 자주 보이는 구조 단서
        - 용량 패턴
        - URL/QR 관련 패턴
        - OCR 노이즈 감점
        """

        if not variant_results:
            raise ValueError("variant_results가 비어 있습니다.")

        sorted_results = sorted(
            variant_results,
            key=lambda item: item.get("score", 0),
            reverse=True
        )

        print("[정보] OCR variant 점수 순위")

        for item in sorted_results:
            print(
                f" - {item.get('variant')}: "
                f"score={item.get('score', 0):.3f}, "
                f"conf={item.get('confidence_avg', 0):.3f}, "
                f"blocks={len(item.get('ocr_blocks', []))}, "
                f"lines={len(item.get('ocr_lines', []))}, "
                f"text_len={len(item.get('raw_text', ''))}"
            )

        return sorted_results[0]

    def _score_full_ocr_result(
        self,
        raw_text: str,
        ocr_blocks: List[Dict[str, Any]],
        ocr_lines: List[Dict[str, Any]],
        confidence_avg: float,
        variant_name: str,
        variant_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        OCR 결과 품질 점수 계산

        [이전 수정] variant별 가중치 균등화 (+0.03 동일 적용)

        [이번 수정] is_dark_background 정보 반영 + bad_chars 감점 강화
        기존: variant 이름과 관계없이 모든 variant +0.03
              → 어두운 배경 이미지에서 inverted가 명백히 더 나은데도
                다른 variant가 비슷한 점수로 선택될 수 있음
        수정:
          - variant_info에 source_is_dark_background=True가 있으면
            inverted variant에 +0.10 보너스 (어두운 배경 라벨에서 inverted 우대)
          - 이진화 계열(soft_binary, high_contrast_binary)이 텍스트는 짧지만
            confidence가 매우 높으면 +0.05 보너스
          - bad_chars 감점 0.04 → 0.06 (글자 깨짐이 심하면 더 강하게 감점)

        Parameters
        ----------
        variant_info : dict | None
            preprocess.py에서 만든 variant info (source_is_dark_background 포함)
        """

        if not raw_text:
            return 0.0

        score = float(confidence_avg)

        readable_count = len(
            re.findall(
                r"[가-힣A-Za-z0-9,.;:()\[\]/\-·%]",
                raw_text
            )
        )

        readable_ratio = readable_count / max(len(raw_text), 1)

        # 읽을 수 있는 문자가 많을수록 가산
        score += readable_ratio * 0.35

        # block/line이 적당히 많아야 작은 글자 추출이 된 것으로 판단
        score += min(len(ocr_blocks) * 0.003, 0.30)
        score += min(len(ocr_lines) * 0.010, 0.30)

        # 모든 variant 동일 기본 보정값 (특정 variant 편향 제거)
        score += 0.03

        # ────────────────────────────────────────────────────────
        # [신규] 컨텍스트 의존 보너스
        # ────────────────────────────────────────────────────────

        # 1) 어두운 배경 이미지에서 inverted variant에 보너스
        if variant_info is not None:
            is_dark_source = bool(
                variant_info.get("source_is_dark_background", False)
                or variant_info.get("is_dark_background", False)
            )

            if is_dark_source and variant_name == "inverted":
                score += 0.10
                # 어두운 배경에서 inverted가 아닌 variant는 약간 감점
                # (이미 inverted +0.10 했으므로 상대적 우열은 0.10 + 0.05 = 0.15)

            if is_dark_source and variant_name in ("original", "readable", "ocr_enhanced", "color_clahe"):
                score -= 0.05

        # 2) 이진화 계열에서 confidence가 매우 높은 경우 보너스
        #    (이진화는 텍스트 길이가 짧을 수 있어 다른 보너스가 적게 붙음)
        if variant_name in ("soft_binary", "high_contrast_binary") and confidence_avg >= 0.85:
            score += 0.05

        # ────────────────────────────────────────────────────────
        # 라벨 컨텐츠 기반 보너스 (기존)
        # ────────────────────────────────────────────────────────

        # 라벨 구간 단서가 있으면 OCR이 중요한 정보를 잡은 것으로 판단
        if self._contains_section_keywords(raw_text):
            score += 0.20

        # 용량 패턴
        if self._contains_capacity_pattern(raw_text):
            score += 0.08

        # URL/QR 관련 패턴
        if self._contains_url_pattern(raw_text):
            score += 0.05

        # 성분형 문장 구조: 콤마가 많고 한글/영문 명사가 연속되는 경우
        if self._looks_like_ingredient_text(raw_text):
            score += 0.18

        # ────────────────────────────────────────────────────────
        # 감점 (강화)
        # ────────────────────────────────────────────────────────

        # [수정] 지나치게 이상한 문자 감점 강화 (0.04 → 0.06)
        bad_count = self._count_bad_ocr_chars(raw_text)
        score -= min(bad_count * 0.06, 0.60)

        # 너무 짧으면 감점
        if len(raw_text.strip()) < 10:
            score -= 0.30

        # 줄 수가 너무 적으면 라벨 전체 추출 실패 가능성
        if len(ocr_lines) < 3:
            score -= 0.20

        return max(score, 0.0)

    def _summarize_variant_results(
        self,
        variant_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        summaries = []

        for item in variant_results:
            summaries.append(
                {
                    "variant": item.get("variant"),
                    "variant_path": item.get("variant_path"),
                    "score": item.get("score", 0.0),
                    "confidence_avg": item.get("confidence_avg", 0.0),
                    "block_count": len(item.get("ocr_blocks", [])),
                    "line_count": len(item.get("ocr_lines", [])),
                    "text_length": len(item.get("raw_text", "")),
                    "selected_variant_size": item.get("selected_variant_size", {})
                }
            )

        return sorted(
            summaries,
            key=lambda item: item.get("score", 0),
            reverse=True
        )

    # =========================================================
    # 8. 텍스트 정리
    # =========================================================

    def _clean_common_text(
        self,
        text: Any
    ) -> str:
        """
        [수정] replace_map 확장 - 한글 OCR 흔한 오인식 패턴 보강
        - 단위 표기 변형 추가 (㎕, ㎍ 등)
        - 한글 OCR이 자주 잘못 잡는 특수 기호/유사문자 매핑 추가
        - 화살표/체크 마크 등 라벨에 자주 쓰이는 기호 정리
        """

        if text is None:
            return ""

        text = str(text)

        text = text.replace("\r", " ")
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")

        replace_map = {
            # 콤마/세미콜론/콜론
            "，": ",",
            "、": ",",
            "；": ";",
            "：": ":",

            # 가운뎃점
            "ㆍ": "·",
            "・": "·",

            # 괄호
            "（": "(",
            "）": ")",
            "［": "[",
            "］": "]",
            "【": "[",
            "】": "]",
            "「": "[",
            "」": "]",
            "『": "[",
            "』": "]",

            # 단위 (확장)
            "㎖": "ml",
            "ｍｌ": "ml",
            "ＭＬ": "ml",
            "㎘": "kl",
            "ｇ": "g",
            "Ｇ": "g",
            "㎎": "mg",
            "㎏": "kg",
            "㎕": "ul",
            "㎍": "ug",
            "㎤": "cm3",

            # 대시
            "–": "-",
            "—": "-",
            "−": "-",
            "‐": "-",
            "‑": "-",

            # 인용부호
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",

            # 로마자
            "Ⅰ": "I",
            "Ⅱ": "II",
            "Ⅲ": "III",
            "Ⅳ": "IV",
            "Ⅴ": "V",

            # 라벨에 자주 등장하는 화살표/장식 (제거)
            "▶": "",
            "▷": "",
            "◀": "",
            "◁": "",
            "→": " ",
            "←": " ",
            "↑": " ",
            "↓": " ",
            "•": "·",
            "●": "·",
            "○": " ",
            "■": " ",
            "□": " ",
            "◆": " ",
            "◇": " ",
            "★": " ",
            "☆": " ",
            "✓": " ",
            "✔": " ",

            # 전각 숫자 → 반각 숫자
            "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
            "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",

            # 전각 % → 반각
            "％": "%",
            "＆": "&",
            "＋": "+",
            "／": "/",
            "＿": "_",
        }

        for old, new in replace_map.items():
            text = text.replace(old, new)

        # 공백 정리
        text = re.sub(r"\s+", " ", text)

        # 쉼표/세미콜론/콜론 주변 정리
        text = re.sub(r"\s*,\s*", ", ", text)
        text = re.sub(r"\s*;\s*", "; ", text)
        text = re.sub(r"\s*:\s*", ": ", text)

        # 닫는 기호 앞 공백 제거
        text = re.sub(r"\s+([,.;:)\]\}])", r"\1", text)

        # 여는 기호 뒤 공백 제거
        text = re.sub(r"([(\[\{])\s+", r"\1", text)

        # 쉼표 뒤 공백 보강
        text = re.sub(r"([,;])(?=[가-힣A-Za-z0-9])", r"\1 ", text)

        return text.strip()

    def _need_space_between_blocks(
        self,
        previous_text: str,
        current_text: str,
        gap: float,
        median_height: float,
        median_width: float
    ) -> bool:
        """
        block 사이에 공백이 필요한지 판단한다.

        [수정] 임계값 상향 조정
        기존: normal_gap_threshold = max(median_height * 0.25, median_width * 0.08, 4)
              small_gap_threshold  = max(median_height * 0.12, 2)
              → 고해상도 이미지에서 gap이 크게 나와 불필요한 공백 과다 삽입

        수정: normal_gap_threshold = max(median_height * 0.35, median_width * 0.10, 5)
              small_gap_threshold  = max(median_height * 0.15, 3)
              → 불필요한 공백 삽입 감소
              → 한글 성분명이 자연스럽게 붙어야 하는 경우 보존
        """

        if gap <= 0:
            return False

        previous_text = previous_text or ""
        current_text = current_text or ""

        if not previous_text or not current_text:
            return False

        # 붙어 있어야 자연스러운 문자
        if previous_text[-1:] in ["(", "[", "{", "/", "-", "·", ",", ".", ":", "&"]:
            return False

        if current_text[:1] in [")", "]", "}", "/", "-", "·", ",", ".", ":", "%"]:
            return False

        # 숫자 + 단위는 붙어도 됨: 750 ml은 나중에 정규화 가능
        if re.search(r"\d$", previous_text) and re.search(
            r"^(ml|mL|ML|g|G|mg|kg|oz|%)",
            current_text
        ):
            return True

        # [수정] 임계값 상향: 0.25 → 0.35, 0.08 → 0.10, 4 → 5
        normal_gap_threshold = max(
            median_height * 0.35,
            median_width * 0.10,
            5
        )

        # [수정] 임계값 상향: 0.12 → 0.15, 2 → 3
        small_gap_threshold = max(
            median_height * 0.15,
            3
        )

        if gap >= normal_gap_threshold:
            return True

        if gap >= small_gap_threshold:
            if re.search(r"[가-힣A-Za-z0-9)]$", previous_text) and re.search(
                r"^[가-힣A-Za-z0-9(]",
                current_text
            ):
                return True

        return False

    # =========================================================
    # 9. OCR 품질 판단 유틸
    # =========================================================

    def _contains_section_keywords(
        self,
        text: str
    ) -> bool:

        if not text:
            return False

        patterns = [
            r"제품\s*명",
            r"상품\s*명",
            r"품\s*명",
            r"화장품\s*명",
            r"전\s*성\s*분",
            r"성\s*분",
            r"주\s*성\s*분",
            r"사용\s*방법",
            r"사용\s*법",
            r"용\s*법",
            r"사용\s*시",
            r"주의\s*사항",
            r"사용\s*시\s*의\s*주의\s*사항",
            r"보관\s*방법",
            r"내용\s*량",
            r"용\s*량",
            r"중\s*량",
            r"제조\s*번호",
            r"제조\s*일자",
            r"사용\s*기한",
            r"사용\s*기간",
            r"Ingredients?",
            r"How\s*to\s*use",
            r"Directions?",
            r"Cautions?",
            r"Warnings?",
            r"Precautions?",
            r"Net\s*Wt",
            r"Volume",
            r"Capacity"
        ]

        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True

        return False

    def _contains_capacity_pattern(
        self,
        text: str
    ) -> bool:

        if not text:
            return False

        return bool(
            re.search(
                r"\d+(?:\.\d+)?\s?(ml|mL|ML|g|G|kg|KG|mg|MG|oz|OZ|fl\.?\s?oz|매|pcs|ea|개)",
                text,
                flags=re.IGNORECASE
            )
        )

    def _contains_url_pattern(
        self,
        text: str
    ) -> bool:

        if not text:
            return False

        return bool(
            re.search(
                r"(https?://|www\.|\.com|\.co\.kr|\.kr|qr|QR)",
                text,
                flags=re.IGNORECASE
            )
        )

    def _looks_like_ingredient_text(
        self,
        text: str
    ) -> bool:
        """
        성분표일 가능성이 있는 텍스트 구조 판단

        주의:
        - 여기서 성분을 확정하지 않는다.
        - OCR variant 선택 점수에만 사용한다.
        """

        if not text:
            return False

        comma_count = text.count(",")
        korean_or_english_words = re.findall(
            r"[가-힣A-Za-z]{2,}",
            text
        )

        ingredient_like_suffixes = [
            "추출물",
            "오일",
            "수",
            "애씨드",
            "글리세린",
            "부틸렌글라이콜",
            "하이드록사이드",
            "클로라이드",
            "레이트",
            "올",
            "테아릴",
            "스테아레이트",
            "폴리머",
            "Acid",
            "Water",
            "Extract",
            "Oil",
            "Glycerin",
            "Glycol",
            "Chloride"
        ]

        suffix_hit = 0

        for suffix in ingredient_like_suffixes:
            if suffix.lower() in text.lower():
                suffix_hit += 1

        if comma_count >= 3 and len(korean_or_english_words) >= 6:
            return True

        if suffix_hit >= 2:
            return True

        return False

    def _average_confidence_from_blocks(
        self,
        blocks: List[Dict[str, Any]]
    ) -> float:

        confidences = [
            float(block.get("confidence", 0.0))
            for block in blocks
        ]

        if not confidences:
            return 0.0

        return sum(confidences) / len(confidences)

    def _is_noise_text(
        self,
        text: str
    ) -> bool:

        text = str(text).strip()

        if not text:
            return True

        # 같은 문자 반복
        if len(set(text)) == 1 and len(text) >= 5:
            return True

        # 기호만 있는 경우
        if re.fullmatch(r"[-_=~.ㆍ·,，;；:：|/\\]+", text):
            return True

        readable_count = len(
            re.findall(
                r"[가-힣A-Za-z0-9]",
                text
            )
        )

        if len(text) >= 4 and readable_count == 0:
            return True

        # 너무 긴 무의미 영문 반복
        if re.fullmatch(r"[A-Za-z]{1,2}", text) and len(text) <= 2:
            return False

        return False

    def _count_bad_ocr_chars(
        self,
        text: str
    ) -> int:

        if not text:
            return 0

        bad_chars = re.findall(r"[<>□■◆◇★☆♣♠♥♡※\ufffd]", text)
        bad_chars += re.findall(r"[Yy]{2,}", text)
        bad_chars += re.findall(r"[UuDdOoLlIi]{4,}", text)

        return len(bad_chars)
