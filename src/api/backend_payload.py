"""
백엔드(DB) 전송용 payload 어댑터.

백엔드 요청 스펙 (POST /api/analysis/ocr-result/)
--------------------------------------------------
필수 필드:
  - user_id     : 앱(프론트)이 OCR 호출 시 전달 → OCR 은 그대로 백엔드로 전달
  - ingredients : 검증된 성분(after_api) 배열  ← result.ingredients_verified
  - raw_text    : OCR 원문 (기본 포함)         ← result.ocr_meta.raw_text

주의
----
기존 file_io.py 의 build_server_payload() 는 변경하지 않는다.
그쪽은 파일 저장/내부용 풍부한 구조를 유지하고,
이 함수는 백엔드 스펙에 정확히 맞춘 전용 어댑터다.
"""

from src.main import normalize_list


def build_backend_payload(final_result, user_id):
    """
    final_result(파이프라인 결과)를 백엔드 스펙 payload 로 변환한다.

    Parameters
    ----------
    final_result : dict
        OCRPipeline.analyze() / build_final_result() 결과
    user_id : str | int
        앱에서 전달받은 사용자 식별자 (그대로 전달)

    Returns
    -------
    dict
        {
          "user_id": ...,
          "ingredients": [...],   # after_api(검증된 성분) 배열
          "raw_text": "..."
        }
    """

    if not isinstance(final_result, dict):
        final_result = {}

    result = final_result.get("result", {}) or {}
    ocr_meta = result.get("ocr_meta", {}) or {}

    # ingredients = 검증된 성분(after_api) 배열
    ingredients_after_api = normalize_list(
        result.get("ingredients_verified", [])
    )

    # raw_text = OCR 원문 (기본 포함)
    raw_text = ocr_meta.get("raw_text", "") or ""

    return {
        "user_id": user_id,
        "ingredients": ingredients_after_api,
        "raw_text": raw_text,
    }
