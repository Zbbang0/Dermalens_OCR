import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

import cv2


class SectionCropper:
    """
    Claude가 탐지한 bbox 구간을 실제 이미지 crop 파일로 저장하는 클래스

    역할:
    - ClaudeAnalyzer가 반환한 detected_sections를 입력받는다.
    - 각 section의 image_path와 bbox를 기준으로 이미지를 자른다.
    - 제품명, 용량, 전성분, 사용방법, 주의사항, QR/URL 구간별 crop 이미지를 저장한다.
    - 이후 OCRRunner가 crop 이미지를 구간별로 OCR할 수 있도록 결과 목록을 반환한다.

    입력 예시:
    {
        "detected_sections": [
            {
                "image_index": 0,
                "image_path": "images/sample1.jpg",
                "section_type": "ingredients",
                "label": "전성분",
                "bbox": {
                    "x1": 10,
                    "y1": 200,
                    "x2": 900,
                    "y2": 700
                },
                "confidence": 0.95
            }
        ]
    }

    반환 예시:
    {
        "success": True,
        "cropped_sections": [
            {
                "section_type": "ingredients",
                "label": "전성분",
                "source_image_path": "images/sample1.jpg",
                "crop_image_path": "outputs/sections/ingredients/ingredients_20260512_120000_001.jpg",
                "bbox": {...},
                "confidence": 0.95
            }
        ],
        "failed_sections": []
    }
    """

    DEFAULT_OUTPUT_DIR = os.path.join("outputs", "sections")

    SECTION_DIR_NAMES = {
        "product_name": "product_name",
        "capacity": "capacity",
        "ingredients": "ingredients",
        "usage": "usage",
        "cautions": "cautions",
        "qr_url": "qr_url",
        "unknown": "unknown"
    }

    def __init__(
        self,
        output_dir: Optional[str] = None,
        padding_ratio: float = 0.03,
        min_padding_px: int = 8
    ):
        """
        Parameters
        ----------
        output_dir : str | None
            crop 이미지 저장 기본 폴더.
            None이면 outputs/sections 사용.

        padding_ratio : float
            bbox 주변 여백 비율.
            OCR에서 글자가 잘리지 않도록 약간 넓게 자르기 위함.

        min_padding_px : int
            최소 padding 픽셀.
        """

        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        self.padding_ratio = padding_ratio
        self.min_padding_px = min_padding_px

        self._ensure_dir(self.output_dir)

    # =========================================================
    # 1. 메인 함수
    # =========================================================

    def crop_sections(
        self,
        claude_section_result: Dict[str, Any],
        save_debug_json: bool = True
    ) -> Dict[str, Any]:
        """
        ClaudeAnalyzer 결과에서 detected_sections를 읽어
        각 영역을 crop 이미지로 저장한다.

        Parameters
        ----------
        claude_section_result : dict
            ClaudeAnalyzer.analyze_images() 반환값

        save_debug_json : bool
            crop 결과 디버그 JSON 저장 여부

        Returns
        -------
        dict
            crop 성공/실패 결과
        """

        detected_sections = []

        if isinstance(claude_section_result, dict):
            detected_sections = claude_section_result.get("detected_sections", [])

        if not isinstance(detected_sections, list):
            detected_sections = []

        if not detected_sections:
            return {
                "success": False,
                "cropped_sections": [],
                "failed_sections": [
                    {
                        "reason": "Claude detected_sections가 비어 있습니다."
                    }
                ],
                "saved_at": self._now_string()
            }

        cropped_sections = []
        failed_sections = []

        for index, section in enumerate(detected_sections, start=1):
            try:
                cropped = self._crop_one_section(
                    section=section,
                    section_order=index
                )

                if cropped:
                    cropped_sections.append(cropped)

                else:
                    failed_sections.append(
                        {
                            "section": section,
                            "reason": "crop 결과가 비어 있습니다."
                        }
                    )

            except Exception as error:
                failed_sections.append(
                    {
                        "section": section,
                        "reason": str(error)
                    }
                )

        result = {
            "success": len(cropped_sections) > 0,
            "cropped_sections": cropped_sections,
            "failed_sections": failed_sections,
            "saved_at": self._now_string()
        }

        if save_debug_json:
            self._save_debug_json(result)

        return result

    # =========================================================
    # 2. 단일 section crop
    # =========================================================

    def _crop_one_section(
        self,
        section: Dict[str, Any],
        section_order: int
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(section, dict):
            raise ValueError("section 형식이 dict가 아닙니다.")

        source_image_path = section.get("image_path")

        if not source_image_path:
            raise ValueError("section에 image_path가 없습니다.")

        if not os.path.exists(source_image_path):
            raise FileNotFoundError(f"원본 이미지 파일을 찾을 수 없습니다: {source_image_path}")

        bbox = section.get("bbox")

        if not isinstance(bbox, dict):
            raise ValueError("section에 bbox가 없습니다.")

        section_type = self._normalize_section_type(
            section.get("section_type")
        )

        label = self._normalize_string(
            section.get("label"),
            default=self._default_label(section_type)
        )

        image = cv2.imread(source_image_path)

        if image is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {source_image_path}")

        image_height, image_width = image.shape[:2]

        normalized_bbox = self._normalize_bbox(
            bbox=bbox,
            image_width=image_width,
            image_height=image_height
        )

        if normalized_bbox is None:
            raise ValueError("bbox 좌표가 유효하지 않습니다.")

        padded_bbox = self._apply_padding(
            bbox=normalized_bbox,
            image_width=image_width,
            image_height=image_height
        )

        x1 = padded_bbox["x1"]
        y1 = padded_bbox["y1"]
        x2 = padded_bbox["x2"]
        y2 = padded_bbox["y2"]

        cropped_image = image[y1:y2, x1:x2]

        if cropped_image is None or cropped_image.size == 0:
            raise ValueError("crop된 이미지가 비어 있습니다.")

        crop_dir = self._get_section_output_dir(section_type)
        self._ensure_dir(crop_dir)

        crop_filename = self._build_crop_filename(
            section_type=section_type,
            section_order=section_order
        )

        crop_image_path = os.path.join(crop_dir, crop_filename)

        saved = cv2.imwrite(crop_image_path, cropped_image)

        if not saved:
            raise ValueError(f"crop 이미지 저장 실패: {crop_image_path}")

        return {
            "section_type": section_type,
            "label": label,
            "source_image_path": source_image_path,
            "crop_image_path": crop_image_path,
            "source_type": section.get("source_type", "unknown"),
            "image_index": section.get("image_index", None),
            "bbox_original": normalized_bbox,
            "bbox_padded": padded_bbox,
            "confidence": self._normalize_confidence(
                section.get("confidence")
            ),
            "reason": self._normalize_string(
                section.get("reason"),
                default=""
            ),
            "width": int(cropped_image.shape[1]),
            "height": int(cropped_image.shape[0])
        }

    # =========================================================
    # 3. bbox 처리
    # =========================================================

    def _normalize_bbox(
        self,
        bbox: Dict[str, Any],
        image_width: int,
        image_height: int
    ) -> Optional[Dict[str, int]]:
        try:
            x1 = self._to_int(bbox.get("x1"))
            y1 = self._to_int(bbox.get("y1"))
            x2 = self._to_int(bbox.get("x2"))
            y2 = self._to_int(bbox.get("y2"))

            if None in [x1, y1, x2, y2]:
                return None

            if x1 > x2:
                x1, x2 = x2, x1

            if y1 > y2:
                y1, y2 = y2, y1

            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))

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

        except Exception:
            return None

    def _apply_padding(
        self,
        bbox: Dict[str, int],
        image_width: int,
        image_height: int
    ) -> Dict[str, int]:
        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        box_width = x2 - x1
        box_height = y2 - y1

        pad_x = max(
            self.min_padding_px,
            int(round(box_width * self.padding_ratio))
        )

        pad_y = max(
            self.min_padding_px,
            int(round(box_height * self.padding_ratio))
        )

        padded_x1 = max(0, x1 - pad_x)
        padded_y1 = max(0, y1 - pad_y)
        padded_x2 = min(image_width, x2 + pad_x)
        padded_y2 = min(image_height, y2 + pad_y)

        return {
            "x1": int(padded_x1),
            "y1": int(padded_y1),
            "x2": int(padded_x2),
            "y2": int(padded_y2)
        }

    # =========================================================
    # 4. 저장 경로 처리
    # =========================================================

    def _get_section_output_dir(self, section_type: str) -> str:
        dir_name = self.SECTION_DIR_NAMES.get(section_type, "unknown")
        return os.path.join(self.output_dir, dir_name)

    def _build_crop_filename(
        self,
        section_type: str,
        section_order: int
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_section_type = self._safe_filename(section_type)

        return f"{safe_section_type}_{timestamp}_{section_order:03d}.jpg"

    def _save_debug_json(self, result: Dict[str, Any]) -> None:
        try:
            debug_dir = os.path.join(self.output_dir, "_debug")
            self._ensure_dir(debug_dir)

            filename = f"section_crop_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            path = os.path.join(debug_dir, filename)

            with open(path, "w", encoding="utf-8") as file:
                json.dump(
                    result,
                    file,
                    ensure_ascii=False,
                    indent=2
                )

            print(f"[SectionCropper] crop 결과 JSON 저장 완료: {path}")

        except Exception as error:
            print(f"[SectionCropper 경고] crop 결과 JSON 저장 실패: {error}")

    def _ensure_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    # =========================================================
    # 5. 정규화 유틸
    # =========================================================

    def _normalize_section_type(self, value: Any) -> str:
        if value is None:
            return "unknown"

        text = str(value).strip()

        if not text:
            return "unknown"

        key = (
            text
            .lower()
            .replace(" ", "")
            .replace("-", "_")
        )

        aliases = {
            "product": "product_name",
            "productname": "product_name",
            "product_name": "product_name",
            "제품명": "product_name",
            "상품명": "product_name",

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

            "usage": "usage",
            "how_to_use": "usage",
            "directions": "usage",
            "사용법": "usage",
            "사용방법": "usage",

            "caution": "cautions",
            "cautions": "cautions",
            "warning": "cautions",
            "warnings": "cautions",
            "precautions": "cautions",
            "주의": "cautions",
            "주의사항": "cautions",
            "사용시주의사항": "cautions",

            "qr": "qr_url",
            "qrcode": "qr_url",
            "qr_url": "qr_url",
            "url": "qr_url",
            "link": "qr_url",
            "barcode": "qr_url",
            "qr코드": "qr_url",
        }

        if key in aliases:
            return aliases[key]

        valid = {
            "product_name",
            "capacity",
            "ingredients",
            "usage",
            "cautions",
            "qr_url",
            "unknown"
        }

        if key in valid:
            return key

        return "unknown"

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

    def _normalize_string(
        self,
        value: Any,
        default: str = ""
    ) -> str:
        if value is None:
            return default

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

    def _to_int(self, value: Any) -> Optional[int]:
        try:
            if value is None:
                return None

            if isinstance(value, bool):
                return int(value)

            if isinstance(value, int):
                return value

            if isinstance(value, float):
                return int(round(value))

            text = str(value).strip()

            if not text:
                return None

            return int(round(float(text)))

        except Exception:
            return None

    def _safe_filename(self, value: Any) -> str:
        text = str(value).strip()

        if not text:
            return "unknown"

        unsafe_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', " "]

        for char in unsafe_chars:
            text = text.replace(char, "_")

        return text

    def _now_string(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")