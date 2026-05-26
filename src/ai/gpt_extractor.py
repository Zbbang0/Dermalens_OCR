import os
import json
import base64
import mimetypes
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


class GPTExtractor:
    """
    OpenAI GPT(Vision) 기반 화장품 라벨 정밀 분류/추출 클래스.

    입력:
    - 사용자가 등록한 원본 이미지 (사진 자체)
    - Google Vision DOCUMENT_TEXT_DETECTION 으로 뽑은 OCR raw 텍스트

    출력:
    - 제품명 / 용량 / 전성분 / 사용방법 / 주의사항 / 효능효과 / QR·URL 을
      라벨에 적힌 그대로 분류한 구조화 JSON

    설계 원칙:
    - 사진을 1차 근거(ground truth)로 본다. Vision raw 텍스트는 작은 글씨/저화질
      구간에서 글자를 보강하는 보조 입력이다.
    - 라벨에 실제로 적혀 있지 않은 값은 만들어내지 않는다 (없으면 "확인 불가").
    - 특정 브랜드/제품/성분에 맞춘 규칙을 쓰지 않는다. 한국어 화장품 라벨의
      일반적인 구조만 사용한다.
    - 출력은 JSON 객체만 한다.
    """

    SUPPORTED_MEDIA_TYPES = {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    }

    def __init__(self):
        load_dotenv()

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY가 없습니다. .env 파일에 OpenAI API 키를 설정하세요."
            )

        self.client = OpenAI(api_key=self.api_key)

    # =========================================================
    # 메인: 이미지 + Vision raw -> 구조화 추출
    # =========================================================

    def extract(
        self,
        image_paths: List[str],
        vision_raw_text: str = "",
        vision_layout_text: str = "",
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        image_paths : list[str]
            사용자가 등록한 원본 이미지 경로 목록
        vision_raw_text : str
            Google Vision OCR raw 텍스트 (줄바꿈 보존)
        vision_layout_text : str
            Vision layout 텍스트 (구역 경계 빈 줄 포함) — 선택

        Returns
        -------
        dict
        """

        valid_image_paths = self._filter_valid_images(image_paths or [])

        if not valid_image_paths:
            return self._empty_result("GPT에 전달할 유효한 이미지가 없습니다.")

        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": self._build_user_text(
                    vision_raw_text=vision_raw_text,
                    vision_layout_text=vision_layout_text,
                ),
            }
        ]

        for image_path in valid_image_paths:
            content.append(self._build_image_block(image_path))

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": content},
                ],
            )

            response_text = response.choices[0].message.content or ""

            print("\n" + "=" * 70)
            print("GPT 정밀 분류/추출 원본 응답")
            print("=" * 70)
            print(response_text)

            parsed = self._safe_json_loads(response_text)
            return self._normalize_result(parsed, response_text)

        except Exception as error:
            return self._empty_result(str(error))

    # =========================================================
    # 이미지 처리
    # =========================================================

    def _filter_valid_images(self, image_paths: List[str]) -> List[str]:
        valid = []

        for image_path in image_paths:
            if not image_path or not isinstance(image_path, str):
                continue

            if not os.path.exists(image_path):
                print(f"[GPT 경고] 이미지 파일 없음: {image_path}")
                continue

            if self._guess_media_type(image_path) not in self.SUPPORTED_MEDIA_TYPES:
                print(f"[GPT 경고] 지원하지 않는 이미지 형식: {image_path}")
                continue

            valid.append(image_path)

        return valid

    def _build_image_block(self, image_path: str) -> Dict[str, Any]:
        media_type = self._guess_media_type(image_path)

        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{encoded}",
                "detail": "high",
            },
        }

    def _guess_media_type(self, image_path: str) -> str:
        media_type, _ = mimetypes.guess_type(image_path)

        if media_type:
            return media_type

        lower = image_path.lower()

        if lower.endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        if lower.endswith(".png"):
            return "image/png"
        if lower.endswith(".webp"):
            return "image/webp"
        if lower.endswith(".gif"):
            return "image/gif"

        return "image/jpeg"

    # =========================================================
    # 프롬프트
    # =========================================================

    def _system_prompt(self) -> str:
        return """
너는 한국 화장품 패키지/라벨 이미지를 읽고 정보를 정확히 분류·추출하는 분석 시스템이다.

입력:
- 화장품 이미지(사진) 1장 이상
- 같은 이미지를 OCR(Google Vision)로 뽑은 raw 텍스트

판단 우선순위:
- 사진을 1차 근거로 본다. 글자가 작거나 흐릿해 사진만으로 애매한 부분은 OCR raw 텍스트로 보강한다.
- 사진과 OCR이 다르면, 사진에서 사람이 실제로 읽을 수 있는 표기를 우선한다.

원칙:
- 라벨에 실제로 적혀 있는 내용만 추출한다. 추측으로 새 값을 만들지 않는다.
- 항목이 라벨에 없으면 빈 값으로 둔다(문자열은 "확인 불가", 리스트는 []).
- 특정 브랜드/제품/성분에 맞춘 가정 없이, 한국어 화장품 라벨의 일반 구조만 사용한다.
- 전성분은 라벨에 적힌 순서를 유지하고, 성분 하나를 한 항목으로 분리한다.
- OCR이 한 성분을 여러 조각으로 쪼갠 경우 하나의 성분명으로 합친다.
- 명백한 OCR 오타(깨진 글자)는 자연스럽게 보정하되, 근거 없는 성분을 추가하지 않는다.
- 제조원/판매원/주소/고객센터/제조번호/사용기한 등은 성분이나 제품명에 넣지 않는다.
- 출력은 JSON 객체만 한다. 설명, 마크다운, 코드블록을 출력하지 않는다.
""".strip()

    def _build_user_text(
        self,
        vision_raw_text: str,
        vision_layout_text: str,
    ) -> str:
        raw_text = (vision_raw_text or "").strip() or "(OCR 텍스트 없음)"
        layout_text = (vision_layout_text or "").strip()

        layout_section = ""
        if layout_text:
            layout_section = (
                "\n\n[참고: 레이아웃 보존 OCR 텍스트 — 빈 줄은 구역 경계 힌트]\n"
                + layout_text
            )

        return f"""
아래 화장품 이미지와 OCR raw 텍스트를 함께 보고, 라벨 정보를 정확히 분류·추출해라.

[OCR raw 텍스트]
{raw_text}{layout_section}

반드시 아래 JSON 스키마 그대로 출력해라. 키 이름과 타입을 정확히 지켜라.

{{
  "product_name": "제품명 문자열, 없으면 \\"확인 불가\\"",
  "capacity": "용량/내용량 문자열(예: 50ml, 175g), 없으면 \\"확인 불가\\"",
  "ingredients": ["전성분을 라벨 순서대로 한 성분씩", "..."],
  "usage": ["사용방법 문장/항목을 분리해서", "..."],
  "cautions": ["주의사항 항목을 분리해서", "..."],
  "effects": ["효능/효과/주요특징 항목", "..."],
  "qr_codes": ["QR/바코드로 읽히는 값이 보이면", "..."],
  "urls": ["라벨에 적힌 URL/도메인", "..."]
}}

규칙:
- ingredients는 전성분 표기 구간에서만 추출한다. 제품명/용량/사용방법/주의사항 텍스트를 성분으로 넣지 마라.
- 각 리스트는 중복 없이, 라벨에 적힌 순서를 유지해라.
- 값이 전혀 없으면 문자열은 "확인 불가", 리스트는 [] 로 둬라.
- JSON 외 다른 문장을 출력하지 마라.
""".strip()

    # =========================================================
    # 응답 파싱 / 정규화
    # =========================================================

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("GPT 응답이 비어 있습니다.")

        text = text.strip()

        if text.startswith("```"):
            text = (
                text.replace("```json", "")
                .replace("```JSON", "")
                .replace("```", "")
                .strip()
            )

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or start >= end:
            raise ValueError("GPT 응답에서 JSON 객체를 찾을 수 없습니다.")

        return json.loads(text[start : end + 1])

    def _normalize_result(
        self,
        parsed: Dict[str, Any],
        raw_response: str,
    ) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            return self._empty_result(
                "GPT 응답이 dict 형식이 아닙니다.", raw_response=raw_response
            )

        return {
            "success": True,
            "product_name": self._clean_text(parsed.get("product_name")),
            "capacity": self._clean_text(parsed.get("capacity")),
            "ingredients": self._clean_list(parsed.get("ingredients")),
            "usage": self._clean_list(parsed.get("usage")),
            "cautions": self._clean_list(parsed.get("cautions")),
            "effects": self._clean_list(parsed.get("effects")),
            "qr_codes": self._clean_list(parsed.get("qr_codes")),
            "urls": self._clean_list(parsed.get("urls")),
            "raw_response": raw_response,
            "error": None,
        }

    _INVALID_VALUES = {
        "", "확인 불가", "없음", "null", "none", "n/a", "na", "unknown",
    }

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return "확인 불가"

        if isinstance(value, list):
            cleaned = self._clean_list(value)
            return cleaned[0] if cleaned else "확인 불가"

        text = str(value).strip()

        if not text or text.lower() in self._INVALID_VALUES:
            return "확인 불가"

        return text

    def _clean_list(self, value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, str):
            value = [value]

        if not isinstance(value, list):
            value = [value]

        results: List[str] = []
        seen = set()

        for item in value:
            if item is None:
                continue

            text = str(item).strip()

            if not text or text.lower() in self._INVALID_VALUES:
                continue

            key = text.lower().replace(" ", "")

            if key in seen:
                continue

            seen.add(key)
            results.append(text)

        return results

    # =========================================================
    # 실패 결과
    # =========================================================

    def _empty_result(
        self,
        error_message: str,
        raw_response: str = "",
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "product_name": "확인 불가",
            "capacity": "확인 불가",
            "ingredients": [],
            "usage": [],
            "cautions": [],
            "effects": [],
            "qr_codes": [],
            "urls": [],
            "raw_response": raw_response,
            "error": error_message,
        }
