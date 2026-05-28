"""
백엔드(DB) 전송용 payload 어댑터.

백엔드 요청 스펙 (POST /api/analysis/ocr-result/)
--------------------------------------------------
호출자 정보
  - user_id                   : 앱/백엔드가 전달한 사용자 식별자 → 그대로 전달
  - image_url                 : 분석한 이미지 URL (선택, 받은 값 그대로)

핵심 OCR 결과
  - product_name              : 제품명                                    ← result.product_name
  - capacity                  : 용량                                      ← result.capacity
  - ingredients               : 검증 후 확정 성분 배열                    ← result.ingredients_verified
  - raw_text                  : OCR 원문                                  ← result.ocr_meta.raw_text
  - usage                     : 사용방법 (문자열 리스트)                  ← result.usage
  - cautions                  : 주의사항 (문자열 리스트)                  ← result.cautions
  - effects                   : 효능/장점 (문자열 리스트)                 ← result.effects

분석 메타
  - ocr_confidence            : OCR 평균 신뢰도 (0.0~1.0)                 ← accuracy.ocr_confidence
  - analyzed_at               : 분석 시각 "YYYY-MM-DD HH:MM:SS"           ← final_result.analyzed_at

성분 검증 정확도/상세
  - ingredient_match_rate     : 성분 검증 성공률 (%, 0.0~100.0)           ← accuracy.ingredient_match_rate
  - ingredient_verified_count : 검증 성공 성분 수                         ← accuracy.ingredient_verified_count
  - ingredient_total_count    : 검증 대상 총 성분 수                      ← accuracy.ingredient_total_count
  - ingredients_raw           : API 검증 전 OCR 추출 성분 배열            ← result.ingredients_raw
  - unverified_ingredients    : 검증 실패 성분 객체 리스트                ← result.ingredient_api_validation.unverified_ingredients
  - ingredient_api_status     : "success" / "partial_success" / "unknown" ← result.ingredient_api_validation.status

QR
  - qr_codes                  : QR 원본 값 리스트                         ← result.qr_info.qr_codes
  - qr_urls                   : QR 안 URL 리스트                          ← result.qr_info.urls

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
        백엔드 스펙에 맞춘 평평한 payload (모듈 상단 docstring 참고).
    """

    if not isinstance(final_result, dict):
        final_result = {}

    result = final_result.get("result", {}) or {}
    ocr_meta = result.get("ocr_meta", {}) or {}
    accuracy = final_result.get("accuracy", {}) or {}
    qr_info = result.get("qr_info", {}) or {}
    # ingredient_api_validation 은 빌더 버전에 따라 result 안 또는 top-level 에 있음.
    # 어느 위치든 읽을 수 있게 양쪽 다 시도한다.
    api_validation = (
        result.get("ingredient_api_validation")
        or final_result.get("ingredient_api_validation")
        or {}
    )

    # ── 핵심 OCR 결과 ─────────────────────────────────────────────────
    ingredients_after_api = normalize_list(result.get("ingredients_verified", []))
    ingredients_raw = normalize_list(result.get("ingredients_raw", []))
    raw_text = ocr_meta.get("raw_text", "") or ""

    # 제품 메타: 누락 시 upstream 이 "확인 불가" 로 채워두므로 값 그대로 전달
    product_name = result.get("product_name") or ""
    capacity = result.get("capacity") or ""

    # 텍스트 리스트(사용방법/주의사항/효능): 항상 list 로 정규화
    usage = normalize_list(result.get("usage", []))
    cautions = normalize_list(result.get("cautions", []))
    effects = normalize_list(result.get("effects", []))

    # ── 분석 메타 ────────────────────────────────────────────────────
    try:
        ocr_confidence = float(accuracy.get("ocr_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        ocr_confidence = 0.0

    analyzed_at = final_result.get("analyzed_at", "") or ""

    # ── 성분 검증 정확도/상세 ────────────────────────────────────────
    try:
        ingredient_match_rate = float(
            accuracy.get("ingredient_match_rate", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        ingredient_match_rate = 0.0

    try:
        ingredient_verified_count = int(
            accuracy.get("ingredient_verified_count", 0) or 0
        )
    except (TypeError, ValueError):
        ingredient_verified_count = 0

    try:
        ingredient_total_count = int(
            accuracy.get("ingredient_total_count", 0) or 0
        )
    except (TypeError, ValueError):
        ingredient_total_count = 0

    # unverified_ingredients 는 객체 리스트 ({ocr_name, query_name, reason, similarity})
    unverified_ingredients = api_validation.get("unverified_ingredients", []) or []
    if not isinstance(unverified_ingredients, list):
        unverified_ingredients = []

    ingredient_api_status = api_validation.get("status", "unknown") or "unknown"

    # ── QR ───────────────────────────────────────────────────────────
    qr_codes = normalize_list(qr_info.get("qr_codes", []))
    qr_urls = normalize_list(qr_info.get("urls", []))

    # ── 최종 payload ─────────────────────────────────────────────────
    payload = {
        # 호출자 정보
        "user_id": user_id,

        # 핵심 OCR 결과
        "product_name": product_name,
        "capacity": capacity,
        "ingredients": ingredients_after_api,
        "raw_text": raw_text,
        "usage": usage,
        "cautions": cautions,
        "effects": effects,

        # 분석 메타
        "ocr_confidence": round(ocr_confidence, 4),
        "analyzed_at": analyzed_at,

        # 성분 검증 정확도/상세
        "ingredient_match_rate": round(ingredient_match_rate, 2),
        "ingredient_verified_count": ingredient_verified_count,
        "ingredient_total_count": ingredient_total_count,
        "ingredients_raw": ingredients_raw,
        "unverified_ingredients": unverified_ingredients,
        "ingredient_api_status": ingredient_api_status,

        # QR
        "qr_codes": qr_codes,
        "qr_urls": qr_urls,
    }

    # image_url 은 선택값 — 전달받은 경우에만 포함
    if image_url:
        payload["image_url"] = image_url

    return payload
