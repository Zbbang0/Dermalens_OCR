"""
OCR 파이프라인 호출용 래퍼.

목적
----
기존 main.py 의 CLI 실행 흐름(main())은 그대로 두고,
서버(server.py)에서 이미지 경로 리스트를 받아 동일한 분석 단계를 수행한 뒤
최종 결과(final_result)를 반환하기 위한 진입점만 새로 제공한다.

main.py 의 단계별 헬퍼 함수를 그대로 재사용하므로
분석 로직은 main.py 한 곳에서만 관리된다 (중복/변경 없음).

분석 단계 (main() 과 동일)
--------------------------
1. 이미지별 OCR / 구간 탐지 / 후처리
2. 여러 이미지 결과 병합
3. GPT 정밀 분류/추출 (가능할 때)
4. QR / URL 분석 보강
5. 성분 API 검증
6. 최종 JSON 생성
"""

from src.main import (
    analyze_single_image,
    merge_image_analysis_results,
    apply_gpt_extraction,
    run_qr_analysis,
    run_ingredient_api_validation,
    build_final_result,
    normalize_list,
)

from src.ocr.run_ocr import OCRRunner
from src.ocr.section_detector import OCRSectionDetector
from src.ocr.postprocess import IngredientPostprocessor
from src.ocr.qr_reader import QRReader

from src.api.qr_analyzer import QRAnalyzer

from src.ai.gpt_extractor import GPTExtractor


class OCRPipeline:
    """
    분석 객체를 1회만 초기화해 재사용하는 파이프라인.

    서버에서 요청마다 OCRRunner(Google Vision 클라이언트) 등을 새로 만들면
    초기화 비용이 크므로, 객체를 한 번만 생성해 두고 analyze()만 반복 호출한다.

    QR / GPT 등 보조 객체는 초기화에 실패해도 파이프라인이 동작하도록
    main() 과 동일하게 None 으로 처리한다.
    """

    def __init__(self):
        # 필수 객체 — 실패 시 예외를 그대로 올린다.
        self.ocr_runner = OCRRunner()
        self.section_detector = OCRSectionDetector()
        self.postprocessor = IngredientPostprocessor()

        # 보조 객체 — 실패해도 분석은 계속 (main() 과 동일 정책).
        try:
            self.qr_reader = QRReader()
        except Exception as error:
            print(f"[경고] QRReader 초기화 실패 (QR 인식 생략): {error}")
            self.qr_reader = None

        try:
            self.qr_analyzer = QRAnalyzer()
        except Exception as error:
            print(f"[경고] QRAnalyzer 초기화 실패 (QR 분석 생략): {error}")
            self.qr_analyzer = None

        try:
            self.gpt_extractor = GPTExtractor()
        except Exception as error:
            print(f"[경고] GPTExtractor 초기화 실패 (규칙 기반 분류 사용): {error}")
            self.gpt_extractor = None

    def analyze(self, image_paths):
        """
        이미지 경로 리스트를 받아 전체 분석을 수행하고 final_result 를 반환한다.

        Parameters
        ----------
        image_paths : list[str]
            분석할 이미지 경로 목록

        Returns
        -------
        dict
            build_final_result() 가 만든 최종 JSON
            (result.ingredients_verified = 성분 API 검증 후 확정 성분)

        Raises
        ------
        ValueError
            유효한 이미지가 없을 때
        RuntimeError
            모든 이미지 분석에 실패했을 때
        """

        valid_paths = [p for p in (image_paths or []) if p]
        if not valid_paths:
            raise ValueError("분석할 이미지 경로가 없습니다.")

        # 1. 이미지별 OCR / 구간 탐지 / 후처리
        image_analysis_results = []
        for image_index, image_path in enumerate(valid_paths, start=1):
            image_analysis_results.append(
                analyze_single_image(
                    image_path=image_path,
                    image_index=image_index,
                    total_count=len(valid_paths),
                    ocr_runner=self.ocr_runner,
                    section_detector=self.section_detector,
                    postprocessor=self.postprocessor,
                )
            )

        successful = [
            item for item in image_analysis_results
            if isinstance(item, dict) and item.get("success")
        ]
        if not successful:
            raise RuntimeError("분석에 성공한 이미지가 없습니다.")

        # 2. 여러 이미지 결과 병합
        merged = merge_image_analysis_results(image_analysis_results)

        # 3. GPT 정밀 분류/추출 (gpt_extractor 가 None 이면 규칙 기반 결과 유지)
        merged = apply_gpt_extraction(
            merged=merged,
            image_paths=valid_paths,
            gpt_extractor=self.gpt_extractor,
        )

        # 4. QR / URL 분석 보강
        ocr_meta = merged.get("ocr_meta", {}) or {}
        raw_section = ocr_meta.get("raw_section_text", {}) or {}
        ocr_text_for_qr = "\n".join([
            str(raw_section.get("qr_url", "")),
            str(ocr_meta.get("raw_text", "")),
            str(raw_section.get("product_name", "")),
        ])

        qr_analysis = run_qr_analysis(
            image_paths=valid_paths,
            ocr_text=ocr_text_for_qr,
            qr_reader=self.qr_reader,
            qr_analyzer=self.qr_analyzer,
        )

        # 5. 성분 API 검증
        ingredients_raw = normalize_list(merged.get("ingredients_raw", []))
        ingredient_api_validation = run_ingredient_api_validation(ingredients_raw)

        merged["ingredients_verified"] = normalize_list(
            ingredient_api_validation.get("ingredients_verified", [])
            or ingredient_api_validation.get("ingredients_after_api", [])
            or ingredient_api_validation.get("verified_ingredient_names", [])
        )

        # 6. 최종 JSON 생성
        final_result = build_final_result(
            image_paths=valid_paths,
            image_analysis_results=image_analysis_results,
            merged_postprocessed_result=merged,
            ingredient_api_validation=ingredient_api_validation,
            qr_analysis=qr_analysis,
        )

        return final_result
