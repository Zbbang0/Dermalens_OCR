import os
import json
import base64
import mimetypes
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
from anthropic import Anthropic

try:
    from PIL import Image
except Exception:
    Image = None


class ClaudeAnalyzer:
    """
    Claude Vision 기반 화장품 이미지 구간 탐지 클래스

    변경된 핵심 역할:
    - Claude가 이미지에서 제품명, 용량, 전성분, 사용방법, 주의사항, QR/URL 영역을 찾는다.
    - Claude는 최종 값을 완성하지 않는다.
    - Claude는 각 정보가 위치한 이미지 구간 bbox를 반환한다.
    - 반환된 bbox는 이후 section_cropper.py에서 crop하는 데 사용한다.
    - crop된 구간 이미지를 OCRRunner가 다시 OCR한다.

    전체 목표 흐름:
    이미지 입력
    → Claude Vision 구간 탐지
    → bbox 기준 crop
    → 구간별 OCR
    → OCR 텍스트 후처리
    → 성분만 공공데이터 API 검증
    → 최종 JSON 생성

    주의:
    - 성분 API 검증은 여기서 하지 않는다.
    - OCR도 여기서 하지 않는다.
    - 이 클래스는 "구간 탐지"만 담당한다.
    """

    # 프로젝트에서 사용할 표준 section_type
    VALID_SECTION_TYPES = {
        "product_name",
        "capacity",
        "ingredients",
        "usage",
        "cautions",
        "qr_url",
        "unknown"
    }

    SECTION_ALIASES = {
        "product": "product_name",
        "productname": "product_name",
        "product_name": "product_name",
        "제품명": "product_name",
        "상품명": "product_name",
        "브랜드": "product_name",

        "capacity": "capacity",
        "volume": "capacity",
        "weight": "capacity",
        "size": "capacity",
        "용량": "capacity",
        "중량": "capacity",
        "내용량": "capacity",

        "ingredient": "ingredients",
        "ingredients": "ingredients",
        "all_ingredients": "ingredients",
        "full_ingredients": "ingredients",
        "전성분": "ingredients",
        "성분": "ingredients",
        "주성분": "ingredients",
        "원료": "ingredients",

        "usage": "usage",
        "how_to_use": "usage",
        "directions": "usage",
        "사용법": "usage",
        "사용방법": "usage",
        "사용 방법": "usage",

        "caution": "cautions",
        "cautions": "cautions",
        "warning": "cautions",
        "warnings": "cautions",
        "precautions": "cautions",
        "주의": "cautions",
        "주의사항": "cautions",
        "사용시주의사항": "cautions",
        "사용 시 주의사항": "cautions",

        "qr": "qr_url",
        "qr_url": "qr_url",
        "url": "qr_url",
        "link": "qr_url",
        "barcode": "qr_url",
        "qrcode": "qr_url",
        "qr코드": "qr_url",
        "URL": "qr_url",
    }

    def __init__(self):
        load_dotenv()

        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY가 없습니다. .env 파일에 Claude API 키를 설정하세요."
            )

        self.client = Anthropic(api_key=self.api_key)

    # =========================================================
    # 1. Claude Vision 구간 탐지 메인 함수
    # =========================================================

    def analyze_images(
        self,
        image_paths: List[str],
        extra_image_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        여러 장의 화장품 이미지를 Claude Vision으로 분석하여
        제품명, 용량, 전성분, 사용방법, 주의사항, QR/URL 구간을 탐지한다.

        Parameters
        ----------
        image_paths : list[str]
            사용자가 입력한 원본 이미지 경로 목록

        extra_image_paths : list[str] | None
            전처리된 이미지 경로 목록
            예:
            - outputs/preprocess/processed_basic_sample1.jpg
            - outputs/preprocess/processed_enhanced_sample1.jpg

        Returns
        -------
        dict
            {
              "success": bool,
              "mode": "section_detection",
              "images": [...],
              "detected_sections": [
                {
                  "image_index": 0,
                  "image_path": "...",
                  "source_type": "original" 또는 "preprocessed",
                  "section_type": "ingredients",
                  "label": "전성분",
                  "bbox": {
                    "x1": 10,
                    "y1": 100,
                    "x2": 900,
                    "y2": 700
                  },
                  "confidence": 0.91,
                  "reason": "전성분 제목 아래 긴 텍스트 영역"
                }
              ],
              "section_summary": {
                "product_name": true,
                "capacity": false,
                "ingredients": true,
                "usage": true,
                "cautions": true,
                "qr_url": false
              },
              "image_analysis_note": "...",
              "error": None
            }
        """

        original_image_paths = self._filter_valid_images(image_paths or [])
        preprocessed_image_paths = self._filter_valid_images(extra_image_paths or [])

        image_items = self._build_image_items(
            original_image_paths=original_image_paths,
            preprocessed_image_paths=preprocessed_image_paths
        )

        if not image_items:
            return self._empty_result(
                "Claude에 전달할 유효한 이미지가 없습니다."
            )

        print("[Claude] Vision 구간 탐지 이미지 준비 완료")
        print(f"[Claude] 원본 이미지 수: {len(original_image_paths)}")
        print(f"[Claude] 전처리 이미지 수: {len(preprocessed_image_paths)}")
        print(f"[Claude] 전체 전달 이미지 수: {len(image_items)}")

        content = []

        for image_item in image_items:
            image_block = self._build_image_block(image_item["image_path"])
            content.append(image_block)

        content.append(
            {
                "type": "text",
                "text": self._build_prompt(image_items=image_items)
            }
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=6000,
                temperature=0,
                system=self._system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )

            response_text = self._extract_response_text(response)

            print("\n" + "=" * 70)
            print("Claude Vision 구간 탐지 원본 응답")
            print("=" * 70)
            print(response_text)

            parsed = self._safe_json_loads(response_text)
            normalized = self._normalize_result(
                result=parsed,
                image_items=image_items
            )

            return normalized

        except Exception as error:
            return self._empty_result(str(error), image_items=image_items)

    # =========================================================
    # 2. 이미지 처리
    # =========================================================

    def _filter_valid_images(self, image_paths: List[str]) -> List[str]:
        valid = []

        for image_path in image_paths:
            if not image_path:
                continue

            if not isinstance(image_path, str):
                continue

            if not os.path.exists(image_path):
                print(f"[Claude 경고] 이미지 파일 없음: {image_path}")
                continue

            media_type = self._guess_media_type(image_path)

            if media_type not in [
                "image/jpeg",
                "image/png",
                "image/gif",
                "image/webp"
            ]:
                print(f"[Claude 경고] 지원하지 않는 이미지 형식: {image_path}")
                continue

            valid.append(image_path)

        return valid

    def _build_image_items(
        self,
        original_image_paths: List[str],
        preprocessed_image_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Claude에게 전달할 이미지 목록을 메타데이터와 함께 구성한다.

        image_index는 Claude에게 전달되는 순서와 동일해야 한다.
        Claude가 반환하는 image_index는 이 순서를 기준으로 한다.
        """

        image_items = []
        seen = set()

        def add_item(path: str, source_type: str):
            abs_path = os.path.abspath(path)

            if abs_path in seen:
                return

            seen.add(abs_path)

            width, height = self._get_image_size(path)

            image_items.append(
                {
                    "image_index": len(image_items),
                    "image_path": path,
                    "abs_path": abs_path,
                    "source_type": source_type,
                    "width": width,
                    "height": height
                }
            )

        for path in original_image_paths:
            add_item(path, "original")

        for path in preprocessed_image_paths:
            add_item(path, "preprocessed")

        return image_items

    def _get_image_size(self, image_path: str) -> Tuple[int, int]:
        """
        이미지 크기 반환.
        PIL이 없거나 실패하면 0, 0 반환.
        """

        if Image is None:
            return 0, 0

        try:
            with Image.open(image_path) as image:
                width, height = image.size
                return int(width), int(height)

        except Exception:
            return 0, 0

    def _build_image_block(self, image_path: str) -> Dict[str, Any]:
        media_type = self._guess_media_type(image_path)

        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data
            }
        }

    def _guess_media_type(self, image_path: str) -> str:
        media_type, _ = mimetypes.guess_type(image_path)

        if media_type:
            return media_type

        lower_path = image_path.lower()

        if lower_path.endswith(".jpg") or lower_path.endswith(".jpeg"):
            return "image/jpeg"

        if lower_path.endswith(".png"):
            return "image/png"

        if lower_path.endswith(".webp"):
            return "image/webp"

        if lower_path.endswith(".gif"):
            return "image/gif"

        return "image/jpeg"

    # =========================================================
    # 3. 프롬프트
    # =========================================================

    def _system_prompt(self) -> str:
        return """
너는 화장품 패키지 이미지의 정보 구간을 탐지하는 Vision Layout Analyzer다.

너의 역할:
- 이미지에서 제품명, 용량, 전성분, 사용방법, 주의사항, QR/URL이 있는 구간을 찾는다.
- 값을 직접 완성해서 추출하지 않는다.
- OCR 결과 텍스트를 받지 않는다.
- 이미지 자체의 시각적 구조, 제목, 문단 배치, 텍스트 밀도, 라벨명을 보고 구간을 판단한다.
- 출력은 반드시 JSON만 한다.
- 설명, 마크다운, 코드블록은 출력하지 않는다.

중요 원칙:
- 너는 최종 성분명을 확정하지 않는다.
- 너는 성분 API 검증을 하지 않는다.
- 너는 OCR을 대신하지 않는다.
- 너는 crop을 위한 bbox 구간만 제공한다.
- bbox는 반드시 이미지 픽셀 좌표 기준으로 제공한다.
- bbox는 해당 정보 구간 전체가 포함되도록 약간 여유 있게 잡는다.
- bbox가 너무 작아서 글자가 잘리면 안 된다.
- 불확실한 구간은 confidence를 낮게 설정한다.
- 보이지 않는 항목은 detected_sections에 넣지 않는다.
- 여러 이미지가 같은 제품의 다른 면이면 각각 필요한 구간을 모두 찾는다.
- 원본 이미지와 전처리 이미지가 함께 있으면 원본의 실제 레이아웃을 우선하고, 전처리 이미지는 글자 가독성 보조로만 사용한다.
""".strip()

    def _build_prompt(self, image_items: List[Dict[str, Any]]) -> str:
        image_list_text = "\n".join(
            [
                (
                    f"- image_index={item['image_index']}, "
                    f"source_type={item['source_type']}, "
                    f"width={item['width']}, height={item['height']}, "
                    f"path={item['image_path']}"
                )
                for item in image_items
            ]
        )

        return f"""
아래 화장품 이미지들을 보고 정보 구간을 탐지해라.

전달 이미지 목록:
{image_list_text}

탐지해야 할 section_type:
1. product_name
   - 제품명 또는 상품명이 있는 구간
   - 브랜드명, 제조원, 판매원, 회사명과 혼동하지 말 것
   - 보통 패키지 앞면 상단/중앙의 큰 텍스트일 수 있음

2. capacity
   - 용량, 중량, 내용량 구간
   - 예: 50ml, 100 mL, 175g, 1.69 fl. oz., 10매, 30ea 등
   - 단위가 있는 짧은 텍스트 구간

3. ingredients
   - 전성분, 성분, 주성분, Ingredients 제목과 그 아래 또는 옆에 있는 성분 나열 구간
   - 쉼표, 줄바꿈, 중점, 슬래시 등으로 성분이 길게 나열된 문단
   - 제조원, 주소, 고객센터, 사용기한, 제조번호는 제외할 것
   - 단, crop할 때 성분 문단 전체가 들어가도록 여유 있게 잡을 것

4. usage
   - 사용법, 사용방법, How to use, Directions 구간
   - 제품 사용 절차나 사용량 설명 문장

5. cautions
   - 주의사항, 사용 시 주의사항, Caution, Warning, Precautions 구간
   - 피부 이상, 보관, 눈에 들어갔을 때, 어린이 손이 닿지 않는 곳 등 경고 문장

6. qr_url
   - QR 코드, 바코드, URL, 도메인, 웹사이트 주소, QR 관련 문구가 있는 구간

bbox 작성 규칙:
- bbox는 반드시 픽셀 좌표로 작성한다.
- 좌표 형식은 x1, y1, x2, y2를 사용한다.
- x1, y1은 좌상단 좌표다.
- x2, y2는 우하단 좌표다.
- x1 < x2, y1 < y2 조건을 지켜라.
- 이미지 밖 좌표를 쓰지 마라.
- 구간 전체 텍스트가 잘리지 않게 약간 넓게 잡아라.
- 여러 줄 문단은 전체 문단을 하나의 bbox로 잡아라.
- 같은 section_type이 여러 곳에 있으면 각각 별도 section으로 반환해도 된다.
- 전처리 이미지에만 더 잘 보이는 경우에도 section을 반환할 수 있다.
- 단, image_index는 반드시 위 전달 이미지 목록의 image_index를 사용해라.

반드시 아래 JSON 형식만 출력해라.

{{
  "detected_sections": [
    {{
      "image_index": 0,
      "source_type": "original",
      "section_type": "ingredients",
      "label": "전성분",
      "bbox": {{
        "x1": 10,
        "y1": 200,
        "x2": 900,
        "y2": 700
      }},
      "confidence": 0.95,
      "reason": "전성분 제목 아래에 쉼표로 나열된 긴 성분 문단이 있음"
    }}
  ],
  "image_analysis_note": "이미지에서 어떤 정보 구간을 찾았는지 간단히 요약"
}}

주의:
- JSON 외 다른 문장을 출력하지 마라.
- 마크다운 코드블록을 출력하지 마라.
- 최종 제품명, 최종 성분명, 최종 사용방법 문장을 완성해서 쓰지 마라.
- detected_sections에는 구간 정보만 넣어라.
- 보이지 않는 항목은 억지로 만들지 마라.
- bbox 좌표는 숫자만 사용해라.
- confidence는 0.0부터 1.0 사이 숫자로 둬라.
""".strip()

    # =========================================================
    # 4. 응답 파싱
    # =========================================================

    def _extract_response_text(self, response) -> str:
        if not response:
            return ""

        if not getattr(response, "content", None):
            return ""

        texts = []

        for block in response.content:
            if getattr(block, "type", None) == "text":
                texts.append(block.text)

            elif hasattr(block, "text"):
                texts.append(block.text)

        return "\n".join(texts).strip()

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        if not text:
            raise ValueError(
                "Claude Vision 응답이 비어 있습니다."
            )

        text = text.strip()

        if text.startswith("```"):
            text = (
                text
                .replace("```json", "")
                .replace("```JSON", "")
                .replace("```", "")
                .strip()
            )

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or start >= end:
            raise ValueError(
                "Claude Vision 응답에서 JSON 객체를 찾을 수 없습니다."
            )

        json_text = text[start:end + 1]

        try:
            return json.loads(json_text)

        except json.JSONDecodeError as error:
            repaired = self._repair_json_text(json_text)

            try:
                return json.loads(repaired)

            except Exception:
                raise ValueError(
                    f"Claude Vision JSON 파싱 실패: {error}"
                )

    def _repair_json_text(self, text: str) -> str:
        """
        Claude 응답이 거의 JSON이지만 일부 따옴표/쉼표 문제로 깨진 경우 최소 보정.
        """

        repaired = text.strip()

        repaired = repaired.replace("“", "\"")
        repaired = repaired.replace("”", "\"")
        repaired = repaired.replace("‘", "'")
        repaired = repaired.replace("’", "'")

        # trailing comma 제거
        repaired = repaired.replace(", }", " }")
        repaired = repaired.replace(",}", "}")
        repaired = repaired.replace(", ]", " ]")
        repaired = repaired.replace(",]", "]")

        return repaired

    # =========================================================
    # 5. 결과 정규화
    # =========================================================

    def _normalize_result(
        self,
        result: Dict[str, Any],
        image_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return self._empty_result(
                "Claude Vision 응답이 dict 형식이 아닙니다.",
                image_items=image_items
            )

        raw_sections = result.get("detected_sections", [])

        if not isinstance(raw_sections, list):
            raw_sections = []

        normalized_sections = []

        for raw_section in raw_sections:
            section = self._normalize_section(
                raw_section=raw_section,
                image_items=image_items
            )

            if section is None:
                continue

            normalized_sections.append(section)

        normalized_sections = self._remove_duplicate_sections(
            normalized_sections
        )

        section_summary = self._build_section_summary(normalized_sections)

        image_analysis_note = self._normalize_string_value(
            result.get("image_analysis_note"),
            default="Claude Vision 구간 탐지 완료"
        )

        return {
            "success": True,
            "mode": "section_detection",
            "images": self._public_image_items(image_items),
            "detected_sections": normalized_sections,
            "section_summary": section_summary,
            "image_analysis_note": image_analysis_note,
            "error": None
        }

    def _normalize_section(
        self,
        raw_section: Any,
        image_items: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw_section, dict):
            return None

        image_index = self._to_int(
            raw_section.get("image_index"),
            default=0
        )

        if image_index < 0 or image_index >= len(image_items):
            return None

        image_item = image_items[image_index]

        section_type = self._normalize_section_type(
            raw_section.get("section_type")
        )

        if section_type == "unknown":
            label_for_guess = self._normalize_string_value(
                raw_section.get("label"),
                default=""
            )
            section_type = self._normalize_section_type(label_for_guess)

        if section_type == "unknown":
            return None

        bbox = self._normalize_bbox(
            raw_bbox=raw_section.get("bbox"),
            image_width=image_item.get("width", 0),
            image_height=image_item.get("height", 0)
        )

        if bbox is None:
            return None

        confidence = self._normalize_confidence(
            raw_section.get("confidence")
        )

        label = self._normalize_string_value(
            raw_section.get("label"),
            default=self._default_label(section_type)
        )

        reason = self._normalize_string_value(
            raw_section.get("reason"),
            default=""
        )

        source_type = self._normalize_string_value(
            raw_section.get("source_type"),
            default=image_item.get("source_type", "unknown")
        )

        return {
            "image_index": image_index,
            "image_path": image_item.get("image_path"),
            "source_type": source_type,
            "section_type": section_type,
            "label": label,
            "bbox": bbox,
            "confidence": confidence,
            "reason": reason
        }

    def _normalize_section_type(self, value: Any) -> str:
        if value is None:
            return "unknown"

        text = str(value).strip()

        if not text:
            return "unknown"

        key = (
            text
            .lower()
            .strip()
            .replace(" ", "")
            .replace("-", "_")
        )

        if key in self.VALID_SECTION_TYPES:
            return key

        if key in self.SECTION_ALIASES:
            return self.SECTION_ALIASES[key]

        # 한글 원문이 공백 포함 상태로 들어오는 경우 보정
        compact_original = (
            str(value)
            .strip()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
        )

        if compact_original in self.SECTION_ALIASES:
            return self.SECTION_ALIASES[compact_original]

        return "unknown"

    def _normalize_bbox(
        self,
        raw_bbox: Any,
        image_width: int,
        image_height: int
    ) -> Optional[Dict[str, int]]:
        if not isinstance(raw_bbox, dict):
            return None

        x1 = self._to_int(raw_bbox.get("x1"), default=None)
        y1 = self._to_int(raw_bbox.get("y1"), default=None)
        x2 = self._to_int(raw_bbox.get("x2"), default=None)
        y2 = self._to_int(raw_bbox.get("y2"), default=None)

        if None in [x1, y1, x2, y2]:
            return None

        # 좌표 순서가 반대로 온 경우 보정
        if x1 > x2:
            x1, x2 = x2, x1

        if y1 > y2:
            y1, y2 = y2, y1

        # 이미지 크기를 알고 있으면 범위 보정
        if image_width and image_width > 0:
            x1 = max(0, min(x1, image_width - 1))
            x2 = max(0, min(x2, image_width))

        else:
            x1 = max(0, x1)
            x2 = max(0, x2)

        if image_height and image_height > 0:
            y1 = max(0, min(y1, image_height - 1))
            y2 = max(0, min(y2, image_height))

        else:
            y1 = max(0, y1)
            y2 = max(0, y2)

        # 너무 작은 박스 제거
        if x2 - x1 < 5:
            return None

        if y2 - y1 < 5:
            return None

        return {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        }

    def _remove_duplicate_sections(
        self,
        sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        같은 이미지, 같은 section_type, 거의 같은 bbox가 중복으로 들어온 경우 제거.
        confidence가 높은 항목을 우선한다.
        """

        if not sections:
            return []

        sorted_sections = sorted(
            sections,
            key=lambda item: item.get("confidence", 0.0),
            reverse=True
        )

        result = []

        for section in sorted_sections:
            duplicated = False

            for saved in result:
                if self._is_similar_section(section, saved):
                    duplicated = True
                    break

            if not duplicated:
                result.append(section)

        # 다시 image_index, y1 기준으로 정렬
        result = sorted(
            result,
            key=lambda item: (
                item.get("image_index", 0),
                item.get("bbox", {}).get("y1", 0),
                item.get("bbox", {}).get("x1", 0)
            )
        )

        return result

    def _is_similar_section(
        self,
        section_a: Dict[str, Any],
        section_b: Dict[str, Any]
    ) -> bool:
        if section_a.get("image_index") != section_b.get("image_index"):
            return False

        if section_a.get("section_type") != section_b.get("section_type"):
            return False

        bbox_a = section_a.get("bbox", {})
        bbox_b = section_b.get("bbox", {})

        iou = self._calculate_iou(bbox_a, bbox_b)

        return iou >= 0.75

    def _calculate_iou(
        self,
        bbox_a: Dict[str, int],
        bbox_b: Dict[str, int]
    ) -> float:
        try:
            ax1, ay1, ax2, ay2 = (
                bbox_a["x1"],
                bbox_a["y1"],
                bbox_a["x2"],
                bbox_a["y2"]
            )
            bx1, by1, bx2, by2 = (
                bbox_b["x1"],
                bbox_b["y1"],
                bbox_b["x2"],
                bbox_b["y2"]
            )

            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

            union_area = area_a + area_b - inter_area

            if union_area <= 0:
                return 0.0

            return inter_area / union_area

        except Exception:
            return 0.0

    def _build_section_summary(
        self,
        sections: List[Dict[str, Any]]
    ) -> Dict[str, bool]:
        summary = {
            "product_name": False,
            "capacity": False,
            "ingredients": False,
            "usage": False,
            "cautions": False,
            "qr_url": False
        }

        for section in sections:
            section_type = section.get("section_type")

            if section_type in summary:
                summary[section_type] = True

        return summary

    def _public_image_items(
        self,
        image_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        result = []

        for item in image_items:
            result.append(
                {
                    "image_index": item.get("image_index"),
                    "image_path": item.get("image_path"),
                    "source_type": item.get("source_type"),
                    "width": item.get("width"),
                    "height": item.get("height")
                }
            )

        return result

    # =========================================================
    # 6. 공통 정규화 유틸
    # =========================================================

    def _normalize_string_value(
        self,
        value: Any,
        default: str = "확인 불가"
    ) -> str:
        if value is None:
            return default

        if isinstance(value, list):
            cleaned = [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

            if not cleaned:
                return default

            return " ".join(cleaned).strip()

        text = str(value).strip()

        if not text:
            return default

        if text.lower() in ["none", "null", "unknown", "n/a", "na"]:
            return default

        return text

    def _normalize_confidence(self, value: Any) -> float:
        try:
            number = float(value)

            if number < 0:
                return 0.0

            if number > 1:
                return 1.0

            return number

        except Exception:
            return 0.0

    def _to_int(self, value: Any, default: Optional[int] = 0) -> Optional[int]:
        try:
            if value is None:
                return default

            if isinstance(value, bool):
                return int(value)

            if isinstance(value, int):
                return value

            if isinstance(value, float):
                return int(round(value))

            text = str(value).strip()

            if not text:
                return default

            return int(round(float(text)))

        except Exception:
            return default

    def _default_label(self, section_type: str) -> str:
        labels = {
            "product_name": "제품명",
            "capacity": "용량",
            "ingredients": "전성분",
            "usage": "사용방법",
            "cautions": "주의사항",
            "qr_url": "QR/URL",
            "unknown": "알 수 없음"
        }

        return labels.get(section_type, "알 수 없음")

    # =========================================================
    # 7. 실패 결과
    # =========================================================

    def _empty_result(
        self,
        error_message: str,
        image_items: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        image_items = image_items or []

        return {
            "success": False,
            "mode": "section_detection",
            "images": self._public_image_items(image_items),
            "detected_sections": [],
            "section_summary": {
                "product_name": False,
                "capacity": False,
                "ingredients": False,
                "usage": False,
                "cautions": False,
                "qr_url": False
            },
            "image_analysis_note": "Claude Vision 구간 탐지 실패 또는 유효 이미지 없음",
            "error": error_message
        }