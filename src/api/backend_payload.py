"""
백엔드(DB) 전송용 payload 어댑터.

백엔드 요청 스펙 (POST /api/analysis/ocr-result/)
--------------------------------------------------
  - user_id        : 앱/백엔드가 전달한 사용자 식별자 → 그대로 전달
  - ingredients    : 검증된 성분(after_api) 배열       ← result.ingredients_verified
  - raw_text       : OCR 원문                          ← result.ocr_meta.raw_text
  - image_url      : 분석한 이미지 URL (선택)          ← 요청으로 받은 값 그대로
  - ocr_confidence : OCR 평균 신뢰도 (0.0~1.0)         ← final_result.accuracy.ocr_confidence
  - product_name   : 제품명                            ← result.product_name
  - capacity       : 용량                              ← result.capacity
  - usage          : 사용방법 (문자열 리스트)          ← result.usage
  - cautions       : 주의사항 (문자열 리스트)          ← result.cautions
  - effects        : 효능/장점 (문자열 리스트)         ← result.effects

주의
----
기존 file_io.py 의 build_server_payload() 는 변경하지 않는다.
그쪽은 파일 저장/내부용 풍부한 구조를 유지하고,
이 함수는 백엔드 스펙에 정확히 맞춘 전용 어댑터다.
"""

from src.main import normalize_list


def build_backend_payload(final_result, user_id, image_url=None):
    """
    final_result(파이프라인 결과)를 백엔드 스펙 payload 로 변환한다.

    Parameters
    ----------
    final_result : dict
        OCRPipeline.analyze() / build_final_result() 결과
    user_id : str | int
        앱에서 전달받은 사용자 식별자 (그대로 전달)
    image_url : str | None
        분석한 이미지 URL. 있으면 payload 에 포함 (백엔드 스펙상 선택값).

    Returns
    -------
    dict
        {
          "user_id": ...,
          "ingredients": [...],       # after_api(검증된 성분) 배열
          "raw_text": "...",
          "ocr_confidence": 0.0~1.0,
          "product_name": "...",
          "capacity": "...",
          "usage": [...],
          "cautions": [...],
          "effects": [...],
          "image_url": "..."          # image_url 인자가 있을 때만
        }
    """

    if not isinstance(final_result, dict):
        final_result = {}

    result = final_result.get("result", {}) or {}
    ocr_meta = result.get("ocr_meta", {}) or {}
    accuracy = final_result.get("accuracy", {}) or {}

    # ingredients = 검증된 성분(after_api) 배열
    ingredients_after_api = normalize_list(
        result.get("ingredients_verified", [])
    )

    # raw_text = OCR 원문
    raw_text = ocr_meta.get("raw_text", "") or ""

    # ocr_confidence = OCR 평균 신뢰도 (accuracy 블록에서 추출)
    try:
        ocr_confidence = float(accuracy.get("ocr_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        ocr_confidence = 0.0

    # 제품 메타: 누락 시 upstream 이 "확인 불가" 로 채워두므로 값 그대로 전달
    product_name = result.get("product_name") or ""
    capacity = result.get("capacity") or ""

    # 텍스트 리스트(사용방법/주의사항/효능): 항상 list 로 정규화
    usage = normalize_list(result.get("usage", []))
    cautions = normalize_list(result.get("cautions", []))
    effects = normalize_list(result.get("effects", []))

    payload = {
        "user_id": user_id,
        "ingredients": ingredients_after_api,
        "raw_text": raw_text,
        "ocr_confidence": round(ocr_confidence, 4),
        "product_name": product_name,
        "capacity": capacity,
        "usage": usage,
        "cautions": cautions,
        "effects": effects,
    }

    # image_url 은 선택값 — 전달받은 경우에만 포함
    if image_url:
        payload["image_url"] = image_url

    return payload
