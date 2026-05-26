from typing import Dict, Any, List


class ResultMerger:
    """
    로컬 OCR 분석 결과와 Claude Vision 이미지 분석 결과를 병합하는 클래스

    병합 원칙:
    1. 이 클래스에서는 성분 API 검증을 수행하지 않는다.
    2. 이 클래스에서는 성분 API 검증 결과를 최종 성분으로 덮어쓰지 않는다.
    3. OCR 결과와 Claude Vision 결과를 먼저 병합한다.
    4. 병합된 ingredients는 '성분 후보' 성격으로 유지한다.
    5. 성분 API 검증 결과는 main.py에서 ingredient_api_validation 필드로 별도 추가한다.

    최종 흐름:
    OCR 1차 분류 결과
    + Claude Vision 독립 분석 결과
    → ResultMerger에서 병합
    → main.py에서 병합 ingredients / ingredient_candidates를 IngredientAPI로 검증
    → 최종 JSON 아래에 ingredient_api_validation 별도 첨부
    """

    def merge(
        self,
        local_result: Dict[str, Any],
        claude_result: Dict[str, Any],
        api_verified_ingredients: List[str] = None
    ) -> Dict[str, Any]:
        """
        OCR 결과와 Claude 결과 병합

        Parameters
        ----------
        local_result : dict
            postprocess.py에서 나온 OCR 1차 분류 결과

        claude_result : dict
            claude_analyzer.py에서 나온 Claude Vision 분석 결과

        api_verified_ingredients : list[str] | None
            기존 코드 호환용 인자.
            여기서는 최종 ingredients를 결정하는 데 사용하지 않는다.
            실제 API 검증 결과는 main.py에서 ingredient_api_validation으로 따로 붙인다.

        Returns
        -------
        dict
            병합 결과
        """

        if api_verified_ingredients is None:
            api_verified_ingredients = []

        local_inner = self._unwrap_local_result(local_result)
        claude_inner = claude_result if isinstance(claude_result, dict) else {}

        merge_log = []

        final_product_name = self._merge_text_field(
            field_name="product_name",
            local_value=local_inner.get("product_name"),
            claude_value=claude_inner.get("product_name"),
            claude_confidence=self._get_confidence(claude_inner, "product_name"),
            merge_log=merge_log,
            claude_priority_threshold=0.60
        )

        final_capacity = self._merge_text_field(
            field_name="capacity",
            local_value=local_inner.get("capacity"),
            claude_value=claude_inner.get("capacity"),
            claude_confidence=self._get_confidence(claude_inner, "capacity"),
            merge_log=merge_log,
            claude_priority_threshold=0.55
        )

        final_usage = self._merge_list_field(
            field_name="usage",
            local_value=local_inner.get("usage"),
            claude_value=claude_inner.get("usage"),
            claude_confidence=self._get_confidence(claude_inner, "usage"),
            merge_log=merge_log,
            claude_priority_threshold=0.55
        )

        final_cautions = self._merge_list_field(
            field_name="cautions",
            local_value=local_inner.get("cautions"),
            claude_value=claude_inner.get("cautions"),
            claude_confidence=self._get_confidence(claude_inner, "cautions"),
            merge_log=merge_log,
            claude_priority_threshold=0.55
        )

        final_qr_url = self._merge_qr_field(
            local_result=local_inner,
            claude_result=claude_inner,
            merge_log=merge_log
        )

        final_ingredient_candidates = self._merge_ingredient_candidates(
            local_ingredients=local_inner.get("ingredients", []),
            local_ingredient_candidates=local_inner.get("ingredient_candidates", []),
            claude_ingredients=claude_inner.get("ingredients", []),
            merge_log=merge_log
        )

        final_ingredients = (
            final_ingredient_candidates
            if final_ingredient_candidates
            else ["확인 불가"]
        )

        final_result = {
            "product_name": final_product_name,
            "capacity": final_capacity,
            "ingredients": final_ingredients,
            "ingredient_candidates": final_ingredient_candidates,
            "usage": final_usage,
            "cautions": final_cautions,
            "qr_url": final_qr_url,

            # 원본 OCR 텍스트 보존
            "raw_text": local_inner.get("raw_text", ""),
            "layout_text": local_inner.get("layout_text", ""),

            # API 검증은 main.py에서 나중에 여기에 붙임
            "ingredient_api_validation": {
                "status": "not_run",
                "verified_ingredients": [],
                "unverified_ingredients": [],
                "api_success_results": [],
                "api_failed_results": [],
                "api_all_results": []
            }
        }

        return {
            "success": True,
            "result": final_result,
            "analysis_sources": {
                "local_ocr_result": local_inner,
                "claude_vision_result": claude_inner,

                # 기존 코드 호환용으로 보존만 함.
                # 여기서는 최종 ingredients에 반영하지 않음.
                "api_verified_ingredients_input_ignored": api_verified_ingredients
            },
            "merge_log": merge_log
        }

    # =========================================================
    # 1. 필드 병합
    # =========================================================

    def _merge_text_field(
        self,
        field_name: str,
        local_value: Any,
        claude_value: Any,
        claude_confidence: float,
        merge_log: List[Dict[str, Any]],
        claude_priority_threshold: float = 0.6
    ) -> str:
        local_value = self._normalize_text(local_value)
        claude_value = self._normalize_text(claude_value)

        local_valid = self._is_valid_text(local_value)
        claude_valid = self._is_valid_text(claude_value)

        if claude_valid and claude_confidence >= claude_priority_threshold:
            final = claude_value
            reason = (
                f"Claude Vision confidence {claude_confidence}가 "
                f"기준 {claude_priority_threshold} 이상이므로 Claude 결과 우선"
            )

        elif claude_valid and not local_valid:
            final = claude_value
            reason = "로컬 OCR 결과가 없고 Claude Vision 결과가 유효하므로 Claude 결과 사용"

        elif local_valid and not claude_valid:
            final = local_value
            reason = "Claude Vision 결과가 확인 불가이므로 로컬 OCR 결과 사용"

        elif claude_valid and local_valid:
            # 둘 다 유효하지만 Claude가 이미지 전체 맥락을 직접 보므로 우선
            final = claude_value
            reason = "양쪽 모두 유효하지만 Claude Vision 이미지 직접 분석 결과를 우선 반영"

        else:
            final = "확인 불가"
            reason = "양쪽 모두 유효한 값을 제공하지 못함"

        merge_log.append(
            {
                "field": field_name,
                "local": local_value,
                "claude": claude_value,
                "claude_confidence": claude_confidence,
                "final": final,
                "reason": reason
            }
        )

        return final

    def _merge_list_field(
        self,
        field_name: str,
        local_value: Any,
        claude_value: Any,
        claude_confidence: float,
        merge_log: List[Dict[str, Any]],
        claude_priority_threshold: float = 0.55
    ) -> List[str]:
        local_list = self._to_clean_list(local_value)
        claude_list = self._to_clean_list(claude_value)

        local_valid = len(local_list) > 0
        claude_valid = len(claude_list) > 0

        if claude_valid and claude_confidence >= claude_priority_threshold:
            final = claude_list
            reason = (
                f"Claude Vision confidence {claude_confidence}가 "
                f"기준 {claude_priority_threshold} 이상이므로 Claude 결과 우선"
            )

        elif claude_valid and not local_valid:
            final = claude_list
            reason = "로컬 OCR 결과가 없고 Claude Vision 결과가 유효하므로 Claude 결과 사용"

        elif local_valid and not claude_valid:
            final = local_list
            reason = "Claude Vision 결과가 확인 불가이므로 로컬 OCR 결과 사용"

        elif claude_valid and local_valid:
            final = self._dedupe_list(claude_list + local_list)
            reason = "Claude Vision 결과를 앞에 두고 로컬 OCR 결과를 보조로 병합"

        else:
            final = ["확인 불가"]
            reason = "양쪽 모두 유효한 값을 제공하지 못함"

        merge_log.append(
            {
                "field": field_name,
                "local": local_list,
                "claude": claude_list,
                "claude_confidence": claude_confidence,
                "final": final,
                "reason": reason
            }
        )

        return final

    def _merge_qr_field(
        self,
        local_result: Dict[str, Any],
        claude_result: Dict[str, Any],
        merge_log: List[Dict[str, Any]]
    ) -> str:
        local_qr_list = self._to_clean_list(
            local_result.get("qr_codes", [])
        )

        local_qr_url = self._normalize_text(
            local_result.get("qr_url", "")
        )

        claude_qr_url = self._normalize_text(
            claude_result.get("qr_url", "")
        )

        claude_confidence = self._get_confidence(
            claude_result,
            "qr_url"
        )

        local_candidates = []

        if self._is_valid_text(local_qr_url):
            local_candidates.append(local_qr_url)

        for item in local_qr_list:
            if self._is_valid_text(item):
                local_candidates.append(item)

        local_candidates = self._dedupe_list(local_candidates)

        claude_valid = self._is_valid_text(claude_qr_url)
        local_valid = len(local_candidates) > 0

        if claude_valid and claude_confidence >= 0.50:
            final = claude_qr_url
            reason = "Claude Vision QR/URL confidence가 기준 이상이므로 Claude 결과 사용"

        elif claude_valid and not local_valid:
            final = claude_qr_url
            reason = "로컬 QR/URL 결과가 없고 Claude 결과가 유효함"

        elif local_valid and not claude_valid:
            final = local_candidates[0]
            reason = "Claude QR/URL 결과가 확인 불가이므로 로컬 OCR/QR 결과 사용"

        elif claude_valid and local_valid:
            final = claude_qr_url
            reason = "양쪽 모두 유효하지만 Claude Vision 결과를 우선 사용"

        else:
            final = "확인 불가"
            reason = "양쪽 모두 QR/URL 값을 제공하지 못함"

        merge_log.append(
            {
                "field": "qr_url",
                "local": local_candidates,
                "claude": claude_qr_url,
                "claude_confidence": claude_confidence,
                "final": final,
                "reason": reason
            }
        )

        return final

    # =========================================================
    # 2. 성분 후보 병합
    # =========================================================

    def _merge_ingredient_candidates(
        self,
        local_ingredients: Any,
        local_ingredient_candidates: Any,
        claude_ingredients: Any,
        merge_log: List[Dict[str, Any]]
    ) -> List[str]:
        """
        성분 후보 병합

        중요:
        - 여기서 API 검증 성분을 최우선으로 두지 않는다.
        - API 검증 전 단계이므로 OCR + Claude 후보를 최대한 보존한다.
        - 최종 검증은 main.py에서 수행한다.
        """

        local_list = self._to_clean_ingredient_list(local_ingredients)
        local_candidate_list = self._to_clean_ingredient_list(local_ingredient_candidates)
        claude_list = self._to_clean_ingredient_list(claude_ingredients)

        # Claude는 이미지 전체 맥락을 직접 보므로 앞에 둔다.
        # 단, OCR 후보도 API 검증 대상으로 보내야 하므로 버리지 않는다.
        merged = self._dedupe_list(
            claude_list
            + local_candidate_list
            + local_list
        )

        merge_log.append(
            {
                "field": "ingredients",
                "local_ingredients": local_list,
                "local_ingredient_candidates": local_candidate_list,
                "claude_vision_ingredients": claude_list,
                "final_candidates": merged,
                "reason": (
                    "성분 API 검증 전 단계이므로 Claude 성분 후보와 로컬 OCR 성분 후보를 병합함. "
                    "API 검증 결과는 main.py에서 ingredient_api_validation 필드로 별도 추가 예정"
                )
            }
        )

        return merged

    # 기존 코드 호환용
    def _merge_ingredients(
        self,
        local_ingredients: Any,
        local_ingredient_candidates: Any,
        claude_ingredients: Any,
        api_verified_ingredients: List[str],
        merge_log: List[Dict[str, Any]]
    ) -> List[str]:
        """
        기존 코드에서 _merge_ingredients를 호출할 가능성을 고려한 호환 메서드.

        주의:
        - api_verified_ingredients는 사용하지 않는다.
        - API 검증 결과는 main.py에서 별도 필드로 붙여야 한다.
        """

        return self._merge_ingredient_candidates(
            local_ingredients=local_ingredients,
            local_ingredient_candidates=local_ingredient_candidates,
            claude_ingredients=claude_ingredients,
            merge_log=merge_log
        )

    def _merge_candidates(
        self,
        local_candidates: Any,
        claude_candidates: Any
    ) -> List[str]:
        local_list = self._to_clean_ingredient_list(local_candidates)
        claude_list = self._to_clean_ingredient_list(claude_candidates)

        return self._dedupe_list(claude_list + local_list)

    # =========================================================
    # 3. 값 정규화
    # =========================================================

    def _unwrap_local_result(self, local_result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(local_result, dict):
            return {}

        if "result" in local_result and isinstance(local_result.get("result"), dict):
            return local_result.get("result", {})

        return local_result

    def _get_confidence(self, claude_result: Dict[str, Any], field_name: str) -> float:
        if not isinstance(claude_result, dict):
            return 0.0

        confidence = claude_result.get("confidence", {})

        if not isinstance(confidence, dict):
            return 0.0

        value = confidence.get(field_name, 0.0)

        try:
            number = float(value)

            if number < 0:
                return 0.0

            if number > 1:
                return 1.0

            return number

        except Exception:
            return 0.0

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return "확인 불가"

        if isinstance(value, list):
            cleaned = self._to_clean_list(value)

            if cleaned:
                return " ".join(cleaned)

            return "확인 불가"

        if isinstance(value, dict):
            return "확인 불가"

        if not isinstance(value, str):
            value = str(value)

        value = value.strip()

        if not value:
            return "확인 불가"

        if value in [
            "없음",
            "null",
            "None",
            "none",
            "N/A",
            "n/a",
            "unknown",
            "Unknown"
        ]:
            return "확인 불가"

        return value

    def _to_clean_list(self, value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, str):
            value = [value]

        if isinstance(value, dict):
            value = [str(value)]

        if not isinstance(value, list):
            value = [str(value)]

        cleaned = []

        for item in value:
            if item is None:
                continue

            if isinstance(item, dict):
                continue

            text = str(item).strip()

            if not text:
                continue

            if text in [
                "확인 불가",
                "없음",
                "null",
                "None",
                "none",
                "N/A",
                "n/a",
                "unknown",
                "Unknown"
            ]:
                continue

            cleaned.append(text)

        return self._dedupe_list(cleaned)

    def _to_clean_ingredient_list(self, value: Any) -> List[str]:
        raw_list = self._to_clean_list(value)
        cleaned = []

        for item in raw_list:
            text = self._clean_ingredient_text(item)

            if not text:
                continue

            if not self._is_valid_ingredient_candidate(text):
                continue

            cleaned.append(text)

        return self._dedupe_list(cleaned)

    def _clean_ingredient_text(self, text: Any) -> str:
        if text is None:
            return ""

        text = str(text).strip()

        if not text:
            return ""

        replace_map = {
            "，": ",",
            "、": ",",
            "；": ";",
            "ㆍ": "·",
            "：": ":"
        }

        for old, new in replace_map.items():
            text = text.replace(old, new)

        text = text.strip(" ,.;:：/·ㆍ-[]{}()")

        # 성분명 앞에 붙은 라벨 제거
        labels = [
            "전성분",
            "전 성분",
            "성분",
            "주성분",
            "주 성분",
            "ingredients",
            "ingredient"
        ]

        for label in labels:
            if text.lower().startswith(label.lower()):
                text = text[len(label):].strip(" :：-")

        text = " ".join(text.split())

        return text.strip()

    def _is_valid_ingredient_candidate(self, text: str) -> bool:
        if not text:
            return False

        invalid_values = {
            "",
            "확인 불가",
            "없음",
            "null",
            "None",
            "none",
            "N/A",
            "n/a",
            "unknown",
            "Unknown"
        }

        if text in invalid_values:
            return False

        if len(text) < 2:
            return False

        if len(text) > 80:
            return False

        lower_text = text.lower()

        forbidden_keywords = [
            "제품명",
            "제품 명",
            "품명",
            "용량",
            "내용량",
            "중량",
            "사용방법",
            "사용 방법",
            "사용법",
            "주의사항",
            "주의",
            "제조원",
            "제조업자",
            "책임판매업자",
            "판매업자",
            "제조번호",
            "제조일자",
            "사용기한",
            "유통기한",
            "고객센터",
            "고객상담",
            "소비자상담",
            "주소",
            "www",
            "http",
            "https",
            ".com",
            ".co.kr",
            "manufacturer",
            "distributor",
            "warning",
            "caution",
            "direction",
            "how to use"
        ]

        compact = self._compact(text)

        for keyword in forbidden_keywords:
            if self._compact(keyword) in compact:
                return False

        if self._looks_like_url(text):
            return False

        if self._looks_like_phone_number(text):
            return False

        if self._looks_like_capacity(text):
            return False

        if text.replace(".", "").isdigit():
            return False

        if not any(
            [
                self._has_korean(text),
                self._has_english(text)
            ]
        ):
            return False

        if lower_text in [
            "for",
            "the",
            "and",
            "with",
            "from",
            "use",
            "made",
            "korea",
            "usa"
        ]:
            return False

        return True

    # =========================================================
    # 4. 유효성 판단
    # =========================================================

    def _is_valid_text(self, value: str) -> bool:
        invalid_values = {
            "",
            "확인 불가",
            "없음",
            "null",
            "None",
            "none",
            "N/A",
            "n/a",
            "unknown",
            "Unknown"
        }

        return value not in invalid_values

    def _looks_like_url(self, text: str) -> bool:
        lower = str(text).lower()

        return (
            "http://" in lower
            or "https://" in lower
            or "www." in lower
            or ".com" in lower
            or ".co.kr" in lower
            or ".kr" in lower
            or ".net" in lower
            or ".org" in lower
        )

    def _looks_like_phone_number(self, text: str) -> bool:
        import re

        return bool(
            re.search(
                r"\d{2,4}-\d{3,4}-\d{4}",
                str(text)
            )
        )

    def _looks_like_capacity(self, text: str) -> bool:
        import re

        return bool(
            re.search(
                r"\b\d+(?:\.\d+)?\s?(ml|mL|ML|g|G|kg|KG|mg|MG|oz|OZ|매|개|pcs|ea)\b",
                str(text)
            )
        )

    def _has_korean(self, text: str) -> bool:
        import re

        return bool(re.search(r"[가-힣]", str(text)))

    def _has_english(self, text: str) -> bool:
        import re

        return bool(re.search(r"[A-Za-z]", str(text)))

    # =========================================================
    # 5. 중복 제거
    # =========================================================

    def _dedupe_list(self, values: List[str]) -> List[str]:
        result = []
        seen = set()

        for value in values or []:
            if value is None:
                continue

            value = str(value).strip()

            if not value:
                continue

            key = self._compact(value)

            if not key:
                continue

            if key not in seen:
                seen.add(key)
                result.append(value)

        return result

    def _compact(self, value: Any) -> str:
        return (
            str(value)
            .lower()
            .replace(" ", "")
            .replace("\n", "")
            .replace("\t", "")
            .replace("\r", "")
            .replace("-", "")
            .replace("_", "")
            .replace(":", "")
            .replace("：", "")
            .replace(".", "")
            .replace(",", "")
            .replace("，", "")
            .replace(";", "")
            .replace("；", "")
            .replace("/", "")
            .replace("·", "")
            .replace("ㆍ", "")
            .replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .replace("{", "")
            .replace("}", "")
        )