"""
build_backend_payload() 검증 스크립트.

실제 outputs/server_payload/*.json 파일을 final_result 로 넣어
백엔드로 전송될 payload 가 의도한 모든 필드(호출자 정보 + 핵심 OCR 결과
+ 분석 메타 + 성분 검증 상세 + QR)를 포함하는지 확인한다.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.backend_payload import build_backend_payload


def main():
    samples = sorted(glob.glob("outputs/server_payload/server_payload_*.json"))
    if not samples:
        print("샘플 파일이 없습니다.")
        return

    sample_path = samples[-1]
    print(f"입력 샘플: {sample_path}\n")

    with open(sample_path, encoding="utf-8") as f:
        final_result = json.load(f)

    # 저장된 server_payload 는 ingredients.{after_api,api_failed_candidates} 같은
    # 중첩 구조라 build_final_result 의 평평한 키로 맞춰준다(원본 파일 변경 X).
    result = final_result.get("result", {}) or {}
    ingredients_block = result.get("ingredients", {}) or {}
    if isinstance(ingredients_block, dict):
        result.setdefault(
            "ingredients_verified", ingredients_block.get("after_api", [])
        )
        result.setdefault(
            "ingredients_raw", ingredients_block.get("before_api", [])
        )
    # qr_info / ingredient_api_validation 은 saved 파일에서도 같은 키.
    final_result["result"] = result

    payload = build_backend_payload(
        final_result,
        user_id="verify-user",
        image_url="https://example.com/img.jpg",
    )

    print("=== 백엔드 payload 키 목록 (순서대로) ===")
    for key in payload.keys():
        print(f"  - {key}")

    print(f"\n=== 총 키 수: {len(payload)} ===")

    print("\n=== 호출자 정보 ===")
    print(f"  user_id   = {payload.get('user_id')!r}")
    print(f"  image_url = {payload.get('image_url')!r}")

    print("\n=== 핵심 OCR 결과 ===")
    print(f"  product_name  = {payload.get('product_name')!r}")
    print(f"  capacity      = {payload.get('capacity')!r}")
    print(f"  ingredients   = ({len(payload.get('ingredients', []))}개)  앞3: {payload.get('ingredients', [])[:3]}")
    print(f"  raw_text      = (길이 {len(payload.get('raw_text', ''))}자)")
    print(f"  usage         = ({len(payload.get('usage', []))}개)")
    for i, item in enumerate(payload.get("usage", []), 1):
        prev = (item[:70] + "...") if len(item) > 70 else item
        print(f"      {i}. {prev}")
    print(f"  cautions      = ({len(payload.get('cautions', []))}개)")
    for i, item in enumerate(payload.get("cautions", []), 1):
        prev = (item[:70] + "...") if len(item) > 70 else item
        print(f"      {i}. {prev}")
    print(f"  effects       = ({len(payload.get('effects', []))}개)")
    for i, item in enumerate(payload.get("effects", []), 1):
        prev = (item[:70] + "...") if len(item) > 70 else item
        print(f"      {i}. {prev}")

    print("\n=== 분석 메타 ===")
    print(f"  ocr_confidence = {payload.get('ocr_confidence')}")
    print(f"  analyzed_at    = {payload.get('analyzed_at')!r}")

    print("\n=== 성분 검증 정확도/상세 ===")
    print(f"  ingredient_match_rate     = {payload.get('ingredient_match_rate')}%")
    print(f"  ingredient_verified_count = {payload.get('ingredient_verified_count')}")
    print(f"  ingredient_total_count    = {payload.get('ingredient_total_count')}")
    print(f"  ingredient_api_status     = {payload.get('ingredient_api_status')!r}")
    print(f"  ingredients_raw           = ({len(payload.get('ingredients_raw', []))}개)")
    print(f"  unverified_ingredients    = ({len(payload.get('unverified_ingredients', []))}개)")
    for i, item in enumerate(payload.get("unverified_ingredients", []), 1):
        if isinstance(item, dict):
            print(f"      {i}. ocr_name={item.get('ocr_name')!r}  reason={item.get('reason')!r}")
        else:
            print(f"      {i}. {item}")

    print("\n=== QR ===")
    print(f"  qr_codes = ({len(payload.get('qr_codes', []))}개)")
    for i, item in enumerate(payload.get("qr_codes", []), 1):
        prev = (item[:80] + "...") if len(item) > 80 else item
        print(f"      {i}. {prev}")
    print(f"  qr_urls  = ({len(payload.get('qr_urls', []))}개) {payload.get('qr_urls', [])}")

    print("\n=== 검증 결론 ===")
    required = [
        "user_id", "image_url",
        "product_name", "capacity", "ingredients", "raw_text",
        "usage", "cautions", "effects",
        "ocr_confidence", "analyzed_at",
        "ingredient_match_rate", "ingredient_verified_count",
        "ingredient_total_count", "ingredients_raw",
        "unverified_ingredients", "ingredient_api_status",
        "qr_codes", "qr_urls",
    ]
    missing = [k for k in required if k not in payload]
    if missing:
        print(f"  X 누락된 키: {missing}")
    else:
        print(f"  OK 의도한 {len(required)}개 키가 모두 payload 에 들어있음.")


if __name__ == "__main__":
    main()
