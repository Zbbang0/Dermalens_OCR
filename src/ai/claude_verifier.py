import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from anthropic import Anthropic


class ClaudeVerifier:
    """
    Dermalens OCR 결과를 Claude로 2차 검증하는 클래스

    역할:
    1. 기존 OCR/후처리 결과를 Claude에 전달
    2. Claude가 제품명, 용량, 성분, 사용방법, 주의사항, URL/QR 등을 재분류
    3. 기존 결과와 Claude 결과를 비교
    4. 최종 JSON 구조로 병합
    """

    def __init__(self):
        load_dotenv()

        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 ANTHROPIC_API_KEY를 추가하세요."
            )

        self.client = Anthropic(api_key=self.api_key)

    def verify(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        OCR 결과를 Claude에 보내 검증/보정한다.

        Parameters
        ----------
        ocr_result : dict
            기존 OCR 파이프라인에서 생성한 결과 JSON

        Returns
        -------
        dict
            Claude 검증 결과 + 최종 병합 결과
        """

        prompt = self._build_prompt(ocr_result)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                temperature=0,
                system=self._system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            text = response.content[0].text.strip()
            claude_result = self._safe_json_loads(text)

            merged_result = self._merge_results(
                original_result=ocr_result,
                claude_result=claude_result
            )

            return {
                "success": True,
                "original_result": ocr_result,
                "claude_result": claude_result,
                "final_result": merged_result
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "original_result": ocr_result,
                "claude_result": None,
                "final_result": ocr_result
            }

    def _system_prompt(self) -> str:
        return """
너는 화장품 패키지 OCR 결과를 검증하는 분석 시스템이다.

목표:
- OCR로 추출된 전체 텍스트를 보고 제품명, 용량, 성분, 사용방법, 주의사항, QR/URL을 분류한다.
- 성분은 화장품 전성분 문맥에 맞게 정리한다.
- OCR 오타가 의심되면 자연스럽게 보정하되, 근거 없이 새로운 성분을 추가하지 않는다.
- 성분 API 검증 결과가 있으면 API 검증 성분을 우선 신뢰한다.
- OCR 텍스트에 없는 정보는 "확인 불가"로 둔다.
- 결과는 반드시 JSON만 출력한다.
- 설명 문장, 마크다운, 코드블록은 출력하지 않는다.
""".strip()

    def _build_prompt(self, ocr_result: Dict[str, Any]) -> str:
        return f"""
아래는 Dermalens OCR 시스템의 1차 분석 결과다.

너의 작업:
1. raw_text를 기준으로 전체 문맥을 다시 확인한다.
2. 제품명, 용량, 성분, 사용방법, 주의사항, QR/URL을 재분류한다.
3. ingredients는 성분 API 검증 결과가 있다면 그 결과를 우선 반영한다.
4. OCR 오타로 보이는 성분명은 화장품 성분명 형태로 보정한다.
5. 단, raw_text와 성분 후보에 전혀 없는 성분은 새로 만들지 않는다.
6. 최종 결과는 아래 JSON 스키마를 정확히 따른다.

출력 JSON 스키마:
{{
  "product_name": "문자열 또는 확인 불가",
  "capacity": "문자열 또는 확인 불가",
  "ingredients": ["성분1", "성분2"],
  "usage": "문자열 또는 확인 불가",
  "cautions": "문자열 또는 확인 불가",
  "qr_url": "문자열 또는 확인 불가",
  "corrections": [
    {{
      "before": "OCR 원문 또는 기존 값",
      "after": "보정 값",
      "reason": "보정 이유"
    }}
  ],
  "confidence": {{
    "product_name": 0.0,
    "capacity": 0.0,
    "ingredients": 0.0,
    "usage": 0.0,
    "cautions": 0.0,
    "qr_url": 0.0
  }}
}}

1차 OCR 분석 결과:
{json.dumps(ocr_result, ensure_ascii=False, indent=2)}
""".strip()

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        """
        Claude 응답에서 JSON만 안전하게 파싱한다.
        혹시 코드블록이 섞여도 최대한 복구한다.
        """

        text = text.strip()

        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("Claude 응답에서 JSON 객체를 찾을 수 없습니다.")

        json_text = text[start:end + 1]
        return json.loads(json_text)

    def _merge_results(
        self,
        original_result: Dict[str, Any],
        claude_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        기존 OCR 결과와 Claude 결과를 병합한다.

        기본 원칙:
        - 성분은 Claude 결과를 우선 사용하되, 빈 배열이면 기존 결과 사용
        - 제품명/용량/사용법/주의사항/QR은 Claude가 "확인 불가"가 아니면 Claude 값 사용
        - Claude도 모르면 기존 값 유지
        """

        original_inner = original_result.get("result", original_result)

        final = {
            "product_name": self._choose_text(
                claude_result.get("product_name"),
                original_inner.get("product_name")
            ),
            "capacity": self._choose_text(
                claude_result.get("capacity"),
                original_inner.get("capacity")
            ),
            "ingredients": self._choose_list(
                claude_result.get("ingredients"),
                original_inner.get("ingredients")
            ),
            "usage": self._choose_text(
                claude_result.get("usage"),
                original_inner.get("usage")
            ),
            "cautions": self._choose_text(
                claude_result.get("cautions"),
                original_inner.get("cautions")
            ),
            "qr_url": self._choose_text(
                claude_result.get("qr_url"),
                original_inner.get("qr_url")
            ),
            "corrections": claude_result.get("corrections", []),
            "confidence": claude_result.get("confidence", {})
        }

        return final

    def _choose_text(self, claude_value: Any, original_value: Any) -> str:
        invalid_values = [None, "", "확인 불가", "없음", "null", "None"]

        if isinstance(claude_value, str) and claude_value.strip() not in invalid_values:
            return claude_value.strip()

        if isinstance(original_value, str) and original_value.strip() not in invalid_values:
            return original_value.strip()

        return "확인 불가"

    def _choose_list(self, claude_value: Any, original_value: Any) -> List[str]:
        if isinstance(claude_value, list) and len(claude_value) > 0:
            return self._clean_list(claude_value)

        if isinstance(original_value, list) and len(original_value) > 0:
            return self._clean_list(original_value)

        return []

    def _clean_list(self, values: List[Any]) -> List[str]:
        cleaned = []

        for value in values:
            if not isinstance(value, str):
                continue

            value = value.strip()

            if not value:
                continue

            if value in cleaned:
                continue

            cleaned.append(value)

        return cleaned