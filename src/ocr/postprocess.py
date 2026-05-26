import re
from typing import Dict, Any, List, Optional


class IngredientPostprocessor:
    """
    Dermalens OCR 후처리 클래스

    Claude API를 사용하지 않는 구조.

    역할:
    1. section_detector.py 결과 또는 OCRRunner.run() 직접 결과를 받는다.
    2. 제품명 / 용량 / 성분 / 사용방법 / 주의사항 / QR URL을 정리한다.
    3. 성분은 API 검증 전 후보 목록으로 만든다.
    4. 성분 API 검증은 여기서 하지 않는다.
       main.py에서 IngredientAPI 호출 후 ingredients_verified를 붙인다.

    [수정 사항]

    (A) DB 전송용 최종 JSON 구조 재설계
        - 요청 필드: product_name / capacity / ingredients_raw /
          ingredients_verified / usage / cautions / qr_info
        - 중복 필드(local_ocr_result, ingredient_candidates) 제거
        - ocr_meta로 OCR 내부 데이터 분리

    (B) _classify_line_section() 개선
        - section_detector와 동일한 헤더 길이 필터 적용
        - 긴 내용 줄에 키워드가 있어도 섹션 전환 안 함

    (C) _looks_like_usage_sentence() 개선
        - "사용" 단독 키워드 제거 → 오분류 방지
        - 명확한 사용방법 표현만 사용

    (D) _looks_like_caution_sentence() 개선
        - "주의" 단독 키워드 제거 → 오분류 방지
        - 명확한 주의사항 표현만 사용

    (E) _should_continue_section() 개선
        - ingredients: hint_words 목록 의존 제거
        - 쉼표 1개 이상 OR 구분자 포함이면 계속
        - gap_tolerance 도입: 비성분 라인 연속 N개까지 허용

    지원 입력 구조 1: section_detector.py 결과

    {
        "success": True,
        "section_text_map": { "product_name": [...], ... },
        "merged_text_by_section": { "product_name": "...", ... },
        "raw_text": "...",
        "ocr_lines": [...],
        "ocr_blocks": [...]
    }

    지원 입력 구조 2: OCRRunner.run() 직접 결과

    {
        "success": True,
        "raw_text": "...",
        "ocr_lines": [...],
        "ocr_blocks": [...]
    }

    최종 DB 전송 JSON 구조:
    {
        "product_name": "...",
        "capacity": "...",
        "ingredients_raw": [...],       ← API 검증 전 후보
        "ingredients_verified": [...],  ← API 검증 후 (main.py에서 채움)
        "usage": [...],
        "cautions": [...],
        "qr_info": {
            "qr_codes": [...],
            "urls": [...]
        },
        "ocr_meta": {
            "raw_text": "...",
            "raw_section_text": {...},
            "selected_variant": "...",
            "section_detection_summary": {...}
        }
    }
    """

    def __init__(self):
        self.section_types = [
            "product_name",
            "capacity",
            "ingredients",
            "usage",
            "cautions",
            "effects",
            "qr_url"
        ]

        # ── 헤더 줄로 인정할 최대 길이 ─────────────────────────────────
        # section_detector와 동일 기준 적용
        self.header_max_length = 25

        # ── ingredients 보조 구간 추정 시 gap 허용 개수 ─────────────────
        self.ingredient_gap_tolerance = 3

        # ── 키워드 목록 ─────────────────────────────────────────────────
        # 복합 표현 → 단일 표현 순으로 정렬 (오분류 방지)

        self.product_keywords = [
            "제품명", "제품 명", "품명", "상품명",
            "제품이름", "제품 이름",
            "product name", "item name"
        ]

        self.capacity_keywords = [
            "내용량", "순중량", "충전량",
            "용량", "중량",
            "net wt", "net weight", "volume", "capacity", "contents"
        ]

        self.ingredient_keywords = [
            "전성분", "전 성분", "전 성 분",
            "주요성분", "주요 성분",
            "주성분", "주 성분", "주 성 분",
            "성분명", "원료명",
            "성분", "원료",
            "ingredient list", "main ingredients", "key ingredients",
            "ingredients", "ingredient"
        ]

        self.usage_keywords = [
            "사용방법", "사용 방법",
            "사용순서", "사용 순서",
            "사용법", "용법",
            "how to use", "directions", "direction", "usage", "use"
        ]

        # 복합 표현 우선
        self.caution_keywords = [
            "사용상주의사항", "사용 시의 주의사항",
            "사용시주의사항", "사용할 때의 주의사항",
            "사용 상 주의사항",
            "주의사항",
            "경고", "주의",
            "warning", "caution", "precautions", "precaution",
            "화기주의", "가연성"
        ]

        # 효능/장점 — 명시적 헤더 라벨 (헤더 줄에서만 매칭)
        self.effects_header_keywords = [
            "효능효과", "효능 효과", "효능", "효과",
            "주요기능", "주요 기능", "기능성", "기능",
            "주요특징", "주요 특징", "제품특징", "제품 특징", "특징",
            "장점",
            "benefits", "benefit", "effects", "effect",
            "features", "feature", "claims"
        ]

        # 효능/장점 — 본문 단서 단어 (광고 카피 안에서 effects 분류 단서)
        self.effects_content_keywords = [
            "진정", "보습", "수분", "촉촉", "탄력", "광채",
            "케어", "보호", "완화", "개선", "도움",
            "영양", "윤기", "생기", "활력",
            "주름", "미백", "재생", "회복", "쿨링",
            "선사", "전달"
        ]

        self.qr_keywords = [
            "웹사이트", "사이트", "홈페이지", "homepage",
            "www", "http", "https", "url", "qr",
            ".com", ".co.kr", ".kr", ".net", ".org", ".io", ".ai"
        ]

        self.manufacturer_keywords = [
            "제조원", "제조업자", "책임판매업자", "판매업자",
            "화장품책임판매업자", "제조판매업자",
            "고객센터", "고객상담실", "소비자상담", "소비자상담실",
            "주소", "제조번호", "제조일자", "사용기한", "유통기한",
            "품질보증기준", "제품개발", "기술지원",
            "manufacturer", "distributor", "customer center",
            "exp", "mfg", "lot", "batch",
            "분쟁해결", "공정거래", "교환", "반품",
            "분리수거", "분리배출", "빈용기", "재활용",
            "제조국", "원산지", "made in", "판매원", "수입원",
            "barcode", "바코드", "전화", "문의"
        ]

        self.storage_keywords = [
            "보관방법", "보관 방법", "보관상 주의사항",
            "직사광선", "고온", "저온", "보관", "storage"
        ]

        self.stop_section_keywords = (
            self.product_keywords
            + self.capacity_keywords
            + self.usage_keywords
            + self.caution_keywords
            + self.effects_header_keywords
            + self.qr_keywords
            + self.manufacturer_keywords
            + self.storage_keywords
        )

        self.noise_words = [
            "for", "cfor", "the", "and", "from", "with", "use",
            "made", "in", "korea", "usa",
            "소비자가", "만드는신문", "신문",
            "brand", "nobrand", "no brand"
        ]

        self.meta_noise_keywords = [
            "brand", "상표", "이마트", "nobrand", "no brand",
            "소비자", "신문", "제조", "주소", "상담",
            "공급업자", "원산지", "품질보증",
            "corporation", "manufacturer", "distributor",
            "고객센터", "판매업자", "책임판매업자"
        ]

        # 성분 문장 오분류 패널티 키워드
        self.sentence_negative_keywords = [
            "사용후", "사용 후", "사용하지", "사용하고", "사용하기",
            "바릅니다", "발라", "흡수", "마사지", "세안", "도포",
            "화기", "화기주의", "가연성", "가스", "환기", "불",
            "버리지", "장소", "보관", "경우", "반드시",
            "하십시오", "하세요", "합니다", "됩니다", "있습니다",
            "상담", "고객", "소비자", "제조", "판매", "주소",
            "교환", "반품", "분쟁", "공정거래",
            "분리수거", "분리배출", "용기", "상품", "상표",
            "원산지", "품질보증", "욕실", "녹슬지"
        ]

    # =========================================================
    # 1. 메인 처리
    # =========================================================

    def process(self, ocr_or_section_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        OCR 결과 또는 section_detector 결과를 받아 DB 전송용 JSON을 만든다.

        주의:
        - 여기서 성분 API 검증은 하지 않는다.
        - ingredients_raw는 OCR 기반 후보 목록이다.
        - ingredients_verified는 main.py에서 IngredientAPI 검증 후 채운다.
        """

        if not ocr_or_section_result:
            return self._empty_result()

        print("[후처리] OCR 결과 정리 시작")

        raw_text = ocr_or_section_result.get("raw_text", "") or ""
        layout_text = ocr_or_section_result.get("layout_text", "") or raw_text
        selected_variant = ocr_or_section_result.get("selected_variant", "")
        section_detection_summary = ocr_or_section_result.get("section_detection_summary", {})

        merged_text_by_section = ocr_or_section_result.get("merged_text_by_section", {}) or {}
        section_text_map = ocr_or_section_result.get("section_text_map", {}) or {}

        # section_detector 미연결 시 raw OCR 기반 보조 구간 추정
        if not merged_text_by_section and not section_text_map:
            detected = self._build_section_text_from_raw_ocr(ocr_or_section_result)
            merged_text_by_section = detected.get("merged_text_by_section", {})
            section_text_map = detected.get("section_text_map", {})

        # ── 각 섹션 텍스트 추출 ────────────────────────────────────────
        product_text = self._get_section_text(merged_text_by_section, section_text_map, "product_name")
        capacity_text = self._get_section_text(merged_text_by_section, section_text_map, "capacity")
        ingredients_text = self._get_section_text(merged_text_by_section, section_text_map, "ingredients")
        usage_text = self._get_section_text(merged_text_by_section, section_text_map, "usage")
        cautions_text = self._get_section_text(merged_text_by_section, section_text_map, "cautions")
        effects_text = self._get_section_text(merged_text_by_section, section_text_map, "effects")
        qr_text = self._get_section_text(merged_text_by_section, section_text_map, "qr_url")

        # ── raw_text 기반 fallback ─────────────────────────────────────
        fallback = self._build_section_text_from_raw_text(raw_text)
        fallback_sections = fallback.get("raw_section_text", {})

        if not product_text:
            product_text = fallback_sections.get("product_name", "")
        if not capacity_text:
            capacity_text = fallback_sections.get("capacity", "")
        if not ingredients_text:
            ingredients_text = fallback_sections.get("ingredients", "")
        if not usage_text:
            usage_text = fallback_sections.get("usage", "")
        if not cautions_text:
            cautions_text = fallback_sections.get("cautions", "")
        if not effects_text:
            effects_text = fallback_sections.get("effects", "")
        if not qr_text:
            qr_text = fallback_sections.get("qr_url", "")

        # ── 필드별 정제 ────────────────────────────────────────────────
        product_name = self.extract_product_name(product_text)
        capacity = self.extract_capacity(capacity_text)

        ingredient_section_text = self.extract_ingredient_section_text(ingredients_text)
        ingredients_raw = self.extract_ingredient_candidates([ingredient_section_text])

        # 성분 후보가 없으면 raw_text에서 보완
        if not ingredients_raw and raw_text:
            guessed = self._guess_ingredient_text_from_raw(raw_text)
            ingredients_raw = self.extract_ingredient_candidates([guessed])
            if guessed and not ingredient_section_text:
                ingredient_section_text = guessed
                ingredients_text = guessed

        usage = self.extract_usage(usage_text)
        cautions = self.extract_cautions(cautions_text)
        effects = self.extract_effects(effects_text)
        qr_codes = self.extract_qr_candidates(qr_text)

        # QR raw_text 보완
        if raw_text:
            qr_codes = self.remove_duplicates(
                qr_codes + self.extract_qr_candidates(raw_text)
            )

        # ── 확인 불가 처리 ─────────────────────────────────────────────
        product_name = product_name or "확인 불가"
        capacity = capacity or "확인 불가"
        usage = usage or ["확인 불가"]
        cautions = cautions or ["확인 불가"]
        effects = effects or ["확인 불가"]

        # ── raw_section_text (OCR 메타용) ──────────────────────────────
        raw_section_text = {
            "product_name": product_text,
            "capacity": capacity_text,
            "ingredients": ingredients_text,
            "usage": usage_text,
            "cautions": cautions_text,
            "effects": effects_text,
            "qr_url": qr_text
        }

        # ── QR 정보 구조화 ─────────────────────────────────────────────
        urls = [q for q in qr_codes if self._looks_like_url(q)]
        qr_info = {
            "qr_codes": qr_codes if qr_codes else ["확인 불가"],
            "urls": urls if urls else ["확인 불가"]
        }

        # ── 최종 DB 전송 JSON ──────────────────────────────────────────
        result = {
            # ── 핵심 필드 ──────────────────────────────────────────────
            "product_name": product_name,
            "capacity": capacity,

            # API 검증 전 OCR 기반 성분 후보
            "ingredients_raw": ingredients_raw,

            # API 검증 후 확정 성분 (main.py에서 채움)
            "ingredients_verified": [],

            "usage": usage,
            "cautions": cautions,
            "effects": effects,
            "qr_info": qr_info,

            # ── OCR 내부 메타 (DB 저장 또는 디버깅용) ──────────────────
            "ocr_meta": {
                "raw_text": raw_text,
                "layout_text": layout_text,
                "raw_section_text": raw_section_text,
                "ingredient_section_text": ingredient_section_text,
                "selected_variant": selected_variant,
                "section_detection_summary": section_detection_summary,
                "ocr_lines": ocr_or_section_result.get("ocr_lines", []),
                "ocr_blocks": ocr_or_section_result.get("ocr_blocks", []),
                "detected_sections": ocr_or_section_result.get("detected_sections", [])
            }
        }

        print("[후처리] OCR 결과 정리 완료")

        return result

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "product_name": "확인 불가",
            "capacity": "확인 불가",
            "ingredients_raw": [],
            "ingredients_verified": [],
            "usage": ["확인 불가"],
            "cautions": ["확인 불가"],
            "effects": ["확인 불가"],
            "qr_info": {
                "qr_codes": ["확인 불가"],
                "urls": ["확인 불가"]
            },
            "ocr_meta": {
                "raw_text": "",
                "layout_text": "",
                "raw_section_text": {
                    "product_name": "",
                    "capacity": "",
                    "ingredients": "",
                    "usage": "",
                    "cautions": "",
                    "effects": "",
                    "qr_url": ""
                },
                "ingredient_section_text": "",
                "selected_variant": "",
                "section_detection_summary": {},
                "ocr_lines": [],
                "ocr_blocks": [],
                "detected_sections": []
            }
        }

    # =========================================================
    # 2. section text 추출
    # =========================================================

    def _get_section_text(
        self,
        merged_text_by_section: Dict[str, Any],
        section_text_map: Dict[str, Any],
        section_type: str
    ) -> str:
        text = merged_text_by_section.get(section_type, "")

        if text:
            return self._clean_general_text_preserve_newline(text)

        values = section_text_map.get(section_type, [])

        if isinstance(values, list):
            joined = "\n".join(
                str(item).strip()
                for item in values
                if str(item).strip()
            )
            return self._clean_general_text_preserve_newline(joined)

        if isinstance(values, str):
            return self._clean_general_text_preserve_newline(values)

        return ""

    # =========================================================
    # 3. section_detector 미연결 시 raw OCR 기반 보조 구간 추정
    # =========================================================

    def _build_section_text_from_raw_ocr(
        self,
        ocr_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        raw_text = ocr_result.get("raw_text", "") or ""
        ocr_lines = ocr_result.get("ocr_lines", []) or []

        if ocr_lines:
            lines = [
                self._clean_general_text_preserve_newline(line.get("text", ""))
                for line in ocr_lines
                if self._clean_general_text_preserve_newline(line.get("text", ""))
            ]
        else:
            lines = [
                self._clean_general_text_preserve_newline(line)
                for line in str(raw_text).splitlines()
                if self._clean_general_text_preserve_newline(line)
            ]

        sections = self._detect_sections_from_lines(lines)

        return {
            "section_text_map": {
                section_type: [sections.get(section_type, "")]
                if sections.get(section_type, "")
                else []
                for section_type in self.section_types
            },
            "merged_text_by_section": sections
        }

    def _build_section_text_from_raw_text(
        self,
        raw_text: str
    ) -> Dict[str, Any]:
        lines = [
            self._clean_general_text_preserve_newline(line)
            for line in str(raw_text or "").splitlines()
            if self._clean_general_text_preserve_newline(line)
        ]

        sections = self._detect_sections_from_lines(lines)

        return {"raw_section_text": sections}

    def _detect_sections_from_lines(
        self,
        lines: List[str]
    ) -> Dict[str, str]:
        """
        section_detector.py가 없거나 실패했을 때 쓰는 보조 구간 추정.

        [수정]
        - _classify_line_section()에 헤더 길이 필터 적용
        - _should_continue_section()의 ingredients 판단 완화
        - gap_tolerance 도입: 비성분 라인 연속 N개까지 허용
        """

        sections = {
            "product_name": "",
            "capacity": "",
            "ingredients": "",
            "usage": "",
            "cautions": "",
            "effects": "",
            "qr_url": ""
        }

        if not lines:
            return sections

        active_section = None
        ingredient_gap_count = 0

        collected = {
            "product_name": [],
            "capacity": [],
            "ingredients": [],
            "usage": [],
            "cautions": [],
            "effects": [],
            "qr_url": []
        }

        for line in lines:
            line = self._clean_general_text_preserve_newline(line)

            if not line:
                continue

            detected_section = self._classify_line_section(line)

            if detected_section:
                active_section = detected_section
                ingredient_gap_count = 0

                value = self._remove_label_by_section(line, detected_section)

                if value:
                    collected[detected_section].append(value)

                continue

            if active_section:
                if self._line_starts_any_other_section(line, active_section):
                    active_section = None
                    ingredient_gap_count = 0

                elif active_section == "ingredients":
                    if self._should_continue_ingredients_section(line, ingredient_gap_count):
                        if self._has_ingredient_structure(line):
                            ingredient_gap_count = 0
                        else:
                            ingredient_gap_count += 1
                        collected["ingredients"].append(line)
                        continue
                    else:
                        active_section = None
                        ingredient_gap_count = 0

                elif self._should_continue_section(line, active_section):
                    collected[active_section].append(line)
                    continue

                else:
                    active_section = None
                    ingredient_gap_count = 0

            # 명시적 label 없이 추정 가능한 줄
            if self._looks_like_capacity_text(line):
                collected["capacity"].append(line)
                continue

            if self._looks_like_url(line):
                collected["qr_url"].append(line)
                continue

            if self._looks_like_ingredient_list_text(line):
                collected["ingredients"].append(line)
                continue

            if self._looks_like_usage_sentence(line):
                collected["usage"].append(line)
                continue

            if self._looks_like_caution_sentence(line):
                collected["cautions"].append(line)
                continue

            if self._looks_like_effects_sentence(line):
                collected["effects"].append(line)
                continue

        # 제품명이 없으면 상단 후보에서 보조 추정
        if not collected["product_name"]:
            product_candidate = self._guess_product_name_from_lines(lines)
            if product_candidate:
                collected["product_name"].append(product_candidate)

        for section_type, values in collected.items():
            cleaned_values = self.remove_duplicates([
                self._clean_general_text_preserve_newline(v)
                for v in values
                if self._clean_general_text_preserve_newline(v)
            ])

            if section_type == "ingredients":
                sections[section_type] = self._prepare_ingredient_section_text(
                    " ".join(cleaned_values)
                )
            elif section_type in ["usage", "cautions", "effects"]:
                sections[section_type] = "\n".join(cleaned_values)
            else:
                sections[section_type] = " ".join(cleaned_values)

        return sections

    def _is_header_line(self, text: str) -> bool:
        """
        섹션 헤더 줄인지 판단한다.
        section_detector._is_section_header_line()과 동일 기준.

        헤더 조건:
        1. 길이가 header_max_length(25자) 이하
        2. "라벨: 값" 형태 — 콜론 앞이 15자 이하
        3. URL/QR 텍스트
        """

        if not text:
            return False

        if self._looks_like_url(text):
            return True

        if len(text) <= self.header_max_length:
            return True

        colon_match = re.match(r"^(.{1,20})[：:]\s*(.+)", text)
        if colon_match:
            label_part = colon_match.group(1).strip()
            if len(label_part) <= 15:
                return True

        return False

    def _classify_line_section(self, line: str) -> Optional[str]:
        """
        [수정] 헤더 줄인지 먼저 확인한다.
        헤더가 아니면 키워드가 있어도 섹션 전환하지 않는다.
        탐지 순서: cautions → ingredients → usage → capacity → qr → product_name
        """

        if not line:
            return None

        # 헤더 줄이 아니면 무시
        if not self._is_header_line(line):
            return None

        compact = self._compact(line)

        if self._contains_keyword(compact, self.caution_keywords):
            return "cautions"

        if self._contains_keyword(compact, self.ingredient_keywords):
            return "ingredients"

        if self._contains_keyword(compact, self.usage_keywords):
            return "usage"

        if self._contains_keyword(compact, self.capacity_keywords):
            return "capacity"

        if self._contains_keyword(compact, self.effects_header_keywords):
            return "effects"

        if self._contains_keyword(compact, self.qr_keywords):
            return "qr_url"

        if self._contains_keyword(compact, self.product_keywords):
            return "product_name"

        return None

    def _remove_label_by_section(self, text: str, section_type: str) -> str:
        if section_type == "product_name":
            return self._remove_section_label(text, self.product_keywords)

        if section_type == "capacity":
            return self._remove_section_label(text, self.capacity_keywords)

        if section_type == "ingredients":
            return self._remove_section_label(text, self.ingredient_keywords)

        if section_type == "usage":
            return self._remove_section_label(text, self.usage_keywords)

        if section_type == "cautions":
            return self._remove_section_label(text, self.caution_keywords)

        if section_type == "effects":
            return self._remove_section_label(text, self.effects_header_keywords)

        if section_type == "qr_url":
            return text

        return text

    def _line_starts_any_other_section(self, line: str, current_section: str) -> bool:
        detected = self._classify_line_section(line)
        return detected is not None and detected != current_section

    def _should_continue_ingredients_section(
        self,
        line: str,
        gap_count: int
    ) -> bool:
        """
        ingredients 섹션 연속 판단 (완화된 조건).

        [수정]
        - 기존: hint_words 목록에 의존하거나 쉼표 2개 이상 필요
        - 수정: 쉼표 1개 이상 OR 구분자 포함이면 계속
        - gap_tolerance 이내 비성분 라인은 허용
        """

        if not line:
            return gap_count < self.ingredient_gap_tolerance

        if self._line_starts_any_other_section(line, "ingredients"):
            return False

        if self._looks_like_usage_sentence(line):
            return False

        if self._looks_like_caution_sentence(line):
            return False

        if self._is_manufacturer_or_meta_text(line):
            return False

        if self._looks_like_url(line):
            return False

        if self._has_ingredient_structure(line):
            return True

        if self._count_separators(line) >= 1:
            return True

        tokens = self._split_by_separators(line)
        if len(tokens) >= 2:
            return True

        if gap_count < self.ingredient_gap_tolerance:
            return True

        return False

    def _has_ingredient_structure(self, text: str) -> bool:
        """구분자 + 토큰 기반 성분 구조 판단 (특정 성분명 목록 미사용)."""

        if not text:
            return False

        sep = self._count_separators(text)
        tokens = self._split_by_separators(text)

        if sep >= 2 and len(tokens) >= 3:
            return True

        if sep >= 1 and len(tokens) >= 2:
            return True

        return False

    def _count_separators(self, text: str) -> int:
        if not text:
            return 0

        return (
            text.count(",")
            + text.count("，")
            + text.count("、")
            + text.count(";")
            + text.count("；")
            + text.count("·")
            + text.count("ㆍ")
        )

    def _split_by_separators(self, text: str) -> List[str]:
        if not text:
            return []

        tokens = re.split(r"[,，、;；·ㆍ/\n]+", text)
        result = []

        for token in tokens:
            token = token.strip(" ,.;:：/·ㆍ-()[]{}")

            if not token or len(token) < 2 or len(token) > 50:
                continue

            if re.fullmatch(r"\d+", token):
                continue

            if re.fullmatch(r"[a-zA-Z]{1,3}", token):
                continue

            result.append(token)

        return result

    def _should_continue_section(self, line: str, section_type: str) -> bool:
        if not line:
            return False

        if section_type == "usage":
            if self._looks_like_ingredient_list_text(line):
                return False

            if self._is_manufacturer_or_meta_text(line):
                return False

            return self._is_valid_sentence_section_text(line)

        if section_type == "cautions":
            if self._looks_like_ingredient_list_text(line):
                return False

            return self._is_valid_sentence_section_text(line)

        if section_type == "effects":
            if self._looks_like_ingredient_list_text(line):
                return False

            if self._looks_like_url(line):
                return False

            if self._looks_like_capacity_text(line):
                return False

            return self._is_valid_sentence_section_text(line)

        if section_type == "capacity":
            return self._looks_like_capacity_text(line)

        if section_type == "qr_url":
            return self._looks_like_url(line) or self._contains_keyword(
                self._compact(line), self.qr_keywords
            )

        if section_type == "product_name":
            return False

        return False

    def _guess_product_name_from_lines(self, lines: List[str]) -> str:
        candidates = []
        top_lines = lines[: min(len(lines), 10)]

        for line in top_lines:
            line = self._clean_general_text(line)

            if not line:
                continue

            if self._is_possible_product_name(line):
                candidates.append(line)

        if not candidates:
            return ""

        candidates.sort(
            key=lambda item: (
                len(item),
                bool(re.search(r"[A-Za-z]", item)),
                bool(re.search(r"[가-힣]", item))
            ),
            reverse=True
        )

        return candidates[0]

    # =========================================================
    # 4. 제품명 추출
    # =========================================================

    def extract_product_name(self, text: str) -> str:
        if not text:
            return ""

        text = self._clean_general_text(text)
        text = self._remove_section_label(text, self.product_keywords)
        text = self._remove_product_noise(text)

        # [강화] 광고 카피 어구가 제품명에 붙어 있으면 잘라낸다
        text = self._strip_advertising_copy_from_product_name(text)

        if not self._is_possible_product_name(text):
            candidates = self._split_lines_or_sentences(text)

            valid_candidates = []
            for c in candidates:
                c = self._strip_advertising_copy_from_product_name(c)
                if self._is_possible_product_name(c):
                    valid_candidates.append(c)

            if valid_candidates:
                # 짧고 광고 어구 없는 후보를 선호 (한국어 화장품 제품명은
                # 대개 3~25자 수준의 브랜드+카테고리 조합)
                valid_candidates.sort(
                    key=lambda item: (
                        # 광고 어미가 없을수록 좋음
                        0 if not self._has_advertising_suffix(item) else 1,
                        # 길이가 6~40 사이에 있을수록 좋음
                        abs(len(item) - 18),
                        # 한글 포함 선호
                        0 if re.search(r"[가-힣]", item) else 1,
                    )
                )
                return valid_candidates[0]

            return ""

        return text

    def _has_advertising_suffix(self, text: str) -> bool:
        if not text:
            return False
        return bool(re.search(
            r"(?:는|한|된|해|하여|되어|위한|로서|에서|함유|주는|전달|선사)$",
            text.strip()
        ))

    def _is_possible_product_name(self, text: str) -> bool:
        if not text:
            return False

        text = self._clean_general_text(text)
        compact = self._compact(text)

        if not compact:
            return False

        if len(text) < 2 or len(text) > 80:
            return False

        if self._is_noise_line(text):
            return False

        forbidden_keywords = (
            self.ingredient_keywords
            + self.usage_keywords
            + self.caution_keywords
            + self.capacity_keywords
            + self.manufacturer_keywords
            + self.qr_keywords
            + self.storage_keywords
        )

        if self._contains_keyword(compact, forbidden_keywords):
            return False

        if self._looks_like_capacity_text(text):
            return False

        if self._looks_like_ingredient_list_text(text):
            return False

        if self._looks_like_url(text):
            return False

        if self._looks_like_phone_number(text):
            return False

        if self._sentence_penalty_score(text) >= 2:
            return False

        if not re.search(r"[가-힣A-Za-z]", text):
            return False

        return True

    def _remove_product_noise(self, text: str) -> str:
        if not text:
            return ""

        text = self._clean_general_text(text)

        remove_patterns = [
            r"^\s*brand\s*[:：]?\s*",
            r"^\s*브랜드\s*[:：]?\s*",
            r"^\s*상품\s*[:：]?\s*"
        ]

        for pattern in remove_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        return text.strip()

    # =========================================================
    # 5. 용량 추출
    # =========================================================

    def extract_capacity(self, text: str) -> str:
        if not text:
            return ""

        text = self._clean_general_text(text)
        text = self._normalize_capacity_text(text)
        text = self._remove_section_label(text, self.capacity_keywords)

        if self._is_forbidden_capacity_context(text):
            return ""

        patterns = [
            r"\d+(?:\.\d+)?\s?(?:ml|mL|ML|g|G|kg|KG|mg|MG|oz|OZ|fl\.?\s?oz|매|pcs|ea|개)",
            r"\d+\s?[xX×]\s?\d+(?:\.\d+)?\s?(?:ml|mL|ML|g|G|kg|KG|mg|MG|oz|OZ|매|pcs|ea|개)?",
            r"\d+(?:\.\d+)?\s?(?:fl\.?\s?oz)"
        ]

        matches = []

        for pattern in patterns:
            found = re.findall(pattern, text, flags=re.IGNORECASE)
            matches.extend(found)

        cleaned = []

        for item in matches:
            item = self._clean_general_text(item)

            if not item:
                continue

            if self._is_likely_manufacture_number_context(text, item):
                continue

            cleaned.append(item)

        cleaned = self.remove_duplicates(cleaned)

        if cleaned:
            return " / ".join(cleaned)

        guessed = self._guess_capacity_from_ocr_digit_error(text)

        if guessed:
            return guessed

        return text if self._looks_like_capacity_text(text) else ""

    def _normalize_capacity_text(self, text: str) -> str:
        text = str(text)

        replace_map = {
            "㎖": "ml", "ｍｌ": "ml", "ＭＬ": "ml",
            "ｍL": "ml", "Ｍl": "ml",
            "ｍｇ": "mg", "ＭＧ": "mg",
            "ｇ": "g", "Ｇ": "g"
        }

        for old, new in replace_map.items():
            text = text.replace(old, new)

        text = re.sub(r"(\d)\s*(mL|ML|ml)", r"\1ml", text, flags=re.IGNORECASE)
        text = re.sub(r"(\d)\s*(mg|MG)", r"\1mg", text, flags=re.IGNORECASE)
        text = re.sub(r"(\d)\s*(g|G)", r"\1g", text)
        text = re.sub(r"(\d)\s*(매|pcs|ea|개)", r"\1\2", text, flags=re.IGNORECASE)
        text = re.sub(r"(\d)\s*[xX×]\s*(\d)", r"\1 x \2", text)

        return text.strip()

    def _looks_like_capacity_text(self, text: str) -> bool:
        if not text:
            return False

        if self._is_forbidden_capacity_context(text):
            return False

        return bool(
            re.search(
                r"\d+(?:\.\d+)?\s?(?:ml|mL|ML|g|G|kg|KG|mg|MG|oz|OZ|fl\.?\s?oz|매|pcs|ea|개)",
                text,
                flags=re.IGNORECASE
            )
        )

    def _guess_capacity_from_ocr_digit_error(self, text: str) -> str:
        if not text:
            return ""

        compact = self._compact(text)

        if not self._contains_keyword(compact, self.capacity_keywords):
            return ""

        if self._is_forbidden_capacity_context(text):
            return ""

        match = re.search(r"(?<!\d)(\d{2,4})9(?!\d)", text)

        if match:
            number = match.group(1)

            try:
                number_int = int(number)
            except ValueError:
                return ""

            if 5 <= number_int <= 1000:
                return f"{number}g"

        return ""

    def _is_forbidden_capacity_context(self, text: str) -> bool:
        compact = self._compact(text)
        forbidden = [
            "제조번호", "제조일자", "사용기한", "유통기한",
            "exp", "mfg", "별도표", "lot", "batch"
        ]
        return self._contains_keyword(compact, forbidden)

    def _is_likely_manufacture_number_context(self, text: str, candidate: str) -> bool:
        index = text.find(candidate)

        if index == -1:
            return False

        start = max(0, index - 20)
        end = min(len(text), index + len(candidate) + 20)
        context = text[start:end]
        compact = self._compact(context)

        manufacture_words = [
            "제조번호", "제조일자", "사용기한", "유통기한",
            "exp", "mfg", "lot", "batch", "별도표"
        ]

        return self._contains_keyword(compact, manufacture_words)

    # =========================================================
    # 6. 성분 영역 / 후보 추출
    # =========================================================

    def extract_ingredient_section_text(self, text: str) -> str:
        if not text:
            return ""

        return self._prepare_ingredient_section_text(text)

    def extract_ingredient_candidates(self, ingredient_texts: List[str]) -> List[str]:
        candidates = []

        for text in ingredient_texts or []:
            if not text:
                continue

            cleaned = self._prepare_ingredient_section_text(text)

            if not cleaned:
                continue

            candidates.extend(self._split_ingredient_text(cleaned))

        candidates = self.normalize_candidates(candidates)
        candidates = self.remove_duplicates(candidates)

        return candidates

    def _split_ingredient_text(self, text: str) -> List[str]:
        if not text:
            return []

        text = self._prepare_ingredient_section_text(text)

        if not text:
            return []

        # [완화] 숫자 사이 쉼표(예: "(4, 800 ppm)", "1, 2-헥산다이올")는 구분자에서 제외
        # OCR로 들어온 성분 텍스트가 공백 구분인데 숫자 쉼표만 있는 경우
        # comma-split이 잘못 발동되어 모든 후보가 길이 80자 초과로 탈락하는 문제 방지
        non_numeric_comma_count = self._count_non_numeric_commas(text)

        if non_numeric_comma_count >= 1:
            rough_items = re.split(r"[,，、]+", text)

        elif re.search(r"[;；|]+", text):
            rough_items = re.split(r"[;；|]+", text)

        elif re.search(r"[·ㆍ]", text):
            rough_items = re.split(r"[·ㆍ]+", text)

        else:
            rough_items = self._split_without_explicit_separator(text)

        results = []

        for item in rough_items:
            item = self._clean_ingredient_candidate(item)

            if not item:
                continue

            # [완화] 쉼표로 잘렸지만 여전히 매우 긴 토큰 (공백 구분 성분 묶음)은
            # 추가로 공백 분할을 시도해서 개별 성분명을 뽑아낸다
            if len(item) > 40 and " " in item and self._count_separators(item) == 0:
                for sub in self._split_without_explicit_separator(item):
                    sub = self._clean_ingredient_candidate(sub)
                    if sub and self._is_valid_ingredient_candidate(sub):
                        results.append(sub)
                continue

            if self._is_valid_ingredient_candidate(item):
                results.append(item)

        return results

    def _count_non_numeric_commas(self, text: str) -> int:
        """숫자-쉼표-숫자 패턴(예: '4, 800', '1, 2-')을 제외한 쉼표 개수."""
        if not text:
            return 0

        total = (
            text.count(",")
            + text.count("，")
            + text.count("、")
        )

        numeric_commas = len(re.findall(r"\d\s*[,，、]\s*\d", text))

        return max(0, total - numeric_commas)

    def _split_without_explicit_separator(self, text: str) -> List[str]:
        if not text:
            return []

        text = self._clean_general_text(text)

        if not text:
            return []

        text = self._insert_soft_delimiter_after_ingredient_suffix(text)

        if self._has_comma_separator(text):
            return re.split(r"[,，、]+", text)

        tokens = [token.strip() for token in text.split() if token.strip()]

        if len(tokens) <= 1:
            return [text]

        results = []

        for token in tokens:
            token = self._clean_ingredient_candidate(token)

            if self._is_valid_ingredient_candidate(token):
                results.append(token)

        return results if results else [text]

    def _prepare_ingredient_section_text(self, text: str) -> str:
        if not text:
            return ""

        text = self.clean_text(text)
        text = self._normalize_ingredient_text(text)
        text = self._remove_section_label(text, self.ingredient_keywords)
        text = self._cut_text_before_stop_section(text)
        text = self._remove_short_english_noise_inside_line(text)
        text = self._restore_comma_separated_ingredient_text(text)
        text = self._insert_soft_delimiter_after_ingredient_suffix(text)
        # [추가] 성분 쉼표 누락 복원 (OCR이 쉼표 없이 붙여 읽은 경우 대응)
        text = self._restore_missing_commas_between_ingredients(text)
        text = self._restore_comma_separated_ingredient_text(text)

        return text.strip()

    def _normalize_ingredient_text(self, text: str) -> str:
        if not text:
            return ""

        text = str(text)

        replace_map = {
            "，": ",", "、": ",", "；": ";", "ㆍ": "·", "：": ":"
        }

        for old, new in replace_map.items():
            text = text.replace(old, new)

        text = re.sub(r"1\s*,?\s*2\s*-\s*", "1,2-", text)
        text = re.sub(r"(\d)\s*ppm", r"\1ppm", text, flags=re.IGNORECASE)
        text = re.sub(r"(피이지|PEG)\s*[-]?\s*(\d+)", r"\1-\2", text, flags=re.IGNORECASE)
        text = re.sub(r"(피피지|PPG)\s*[-]?\s*(\d+)", r"\1-\2", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _insert_soft_delimiter_after_ingredient_suffix(self, text: str) -> str:
        if not text:
            return ""

        if "," in text:
            return text

        suffix_patterns = [
            "정제수", "글리세린", "부틸렌글라이콜", "프로필렌글라이콜",
            "다이프로필렌글라이콜", "스테아릭애씨드", "미리스틱애씨드",
            "라우릭애씨드", "팔미틱애씨드",
            "하이드록사이드", "스테아레이트", "팔미테이트",
            "미리스테이트", "라우레이트",
            "추출물", "오일", "버터", "왁스", "알코올",
            "토코페롤", "판테놀", "카보머", "트로메타민",
            "이디티에이", "하이알루로네이트", "하이알루로닉",
            "페녹시에탄올", "리모넨", "리날룰", "아데노신",
            "다이올", "나이아신아마이드", "콜라겐", "유비퀴논",
            "알란토인", "베타인", "잔탄검"
        ]

        for pattern in suffix_patterns:
            text = re.sub(
                rf"({re.escape(pattern)})(?=[가-힣A-Za-z0-9])",
                r"\1, ",
                text
            )

        text = re.sub(
            r"(피이지-?\d+[A-Za-z]*)(?=[가-힣])",
            r"\1, ",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"(PEG-?\d+[A-Za-z]*)(?=[A-Za-z가-힣])",
            r"\1, ",
            text,
            flags=re.IGNORECASE
        )

        return text

    def _restore_missing_commas_between_ingredients(self, text: str) -> str:
        """
        OCR이 쉼표 없이 성분명을 붙여 읽은 경우 쉼표를 복원한다.

        예:
        "정제수소라우레스설페이트소듐클로라이드"
        → "정제수, 소라우레스설페이트, 소듐클로라이드"

        방식:
        - 알려진 성분 시작 단어(접두사) 앞에 쉼표 삽입
        - 이미 쉼표가 충분히 있는 텍스트는 건드리지 않음 (쉼표 3개 이상)
        - 한글 성분명 기준으로만 동작 (영문 성분은 공백으로 이미 구분됨)
        """

        if not text:
            return ""

        # 쉼표가 이미 충분하면 건드리지 않음
        if text.count(",") >= 3:
            return text

        # 성분 시작 접두사 목록 (이 단어 앞에 쉼표 삽입)
        # 가장 긴 것 먼저 (짧은 것이 먼저 매칭되면 오류)
        ingredient_starters = [
            # 소듐 계열
            "소듐라우레스", "소듐라우릴", "소듐클로라이드", "소듐시트레이트",
            "소듐하이드록사이드", "소듐벤조에이트", "소듐자일렌",
            "소듐이디티에이", "소듐살리실레이트",
            "다이소듐이디티에이", "트라이소듐이디티에이",
            "소듐", "소둠", "소돔",
            # 포타슘 계열
            "포타슘하이드록사이드", "포타슘",
            # 코카미도 계열
            "코카미도프로필베타인", "코카미도프로필", "코카미드",
            # 글리세린/글라이콜 계열
            "글리세린", "부틸렌글라이콜", "프로필렌글라이콜",
            "다이프로필렌글라이콜", "헥실렌글라이콜",
            # 정제수
            "정제수",
            # 알코올 계열
            "스테아릴알코올", "세틸알코올", "세테아릴알코올",
            "베헤닐알코올", "라우릴알코올",
            # 추출물
            "알로에베라추출물", "녹차추출물", "카밀레추출물",
            # 향료/색소
            "향료", "색소",
            # 기타 공통 시작어
            "구아하이드록시", "잔탄검", "카보머",
            "나이아신아마이드", "판테놀", "아데노신",
            "토코페롤", "알란토인", "베타인",
            "하이알루로닉애씨드", "하이알루로네이트",
            "페녹시에탄올", "메칠파라벤", "에칠파라벤",
            "시트릭애씨드", "락틱애씨드", "말릭애씨드",
            "피이지", "폴리쿼터늄",
            "트라이에탄올아민", "트로메타민",
            "테트라소듐", "테트라",
            "에탄올", "이소프로판올",
            "향료", "물"
        ]

        result = text

        for starter in ingredient_starters:
            # 문자열 시작 이후에 나오는 starter 앞에 쉼표 삽입
            # (이미 쉼표나 공백이 앞에 있으면 삽입 안 함)
            pattern = f"(?<=[가-힣a-zA-Z0-9])({re.escape(starter)})"
            replacement = f", \\1"
            result = re.sub(pattern, replacement, result)

        return result

    def _restore_comma_separated_ingredient_text(self, text: str) -> str:
        if not text:
            return ""

        text = str(text)
        text = text.replace("，", ",").replace("、", ",")
        text = text.replace("；", ";").replace("ㆍ", "·").replace("：", ":")
        text = re.sub(r"\s*,\s*", ", ", text)
        text = re.sub(r",\s*,+", ", ", text)
        text = re.sub(r"\s*;\s*", "; ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip(" ,;:")

    def _clean_ingredient_candidate(self, text: str) -> str:
        if not text:
            return ""

        text = str(text)
        text = text.replace("\n", " ").replace("\t", " ")
        text = self._normalize_ingredient_text(text)
        text = self._remove_section_label(text, self.ingredient_keywords)
        text = self._cut_text_before_stop_section(text)
        text = self._remove_short_english_noise_inside_line(text)

        text = re.sub(r"^[\s:：,.;/·ㆍ\-\[\](){}]+", "", text)
        text = re.sub(r"[\s:：,.;/·ㆍ\-\[\](){}]+$", "", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _is_valid_ingredient_candidate(self, text: str) -> bool:
        if not text:
            return False

        text = self._clean_general_text(text)
        compact = self._compact(text)

        if not compact:
            return False

        if len(text) < 2 or len(text) > 80:
            return False

        if self._is_noise_word(text):
            return False

        forbidden_keywords = (
            self.product_keywords
            + self.usage_keywords
            + self.caution_keywords
            + self.capacity_keywords
            + self.manufacturer_keywords
            + self.qr_keywords
            + self.storage_keywords
        )

        if self._contains_keyword(compact, forbidden_keywords):
            return False

        if self._contains_keyword(compact, self.meta_noise_keywords):
            return False

        if self._looks_like_url(text):
            return False

        if self._looks_like_phone_number(text):
            return False

        if self._looks_like_capacity_text(text):
            return False

        if re.fullmatch(r"[\d\s.]+", text):
            return False

        if self._sentence_penalty_score(text) >= 2:
            return False

        if not re.search(r"[가-힣a-zA-Z]", text):
            return False

        if re.fullmatch(r"[a-zA-Z]{1,4}", text):
            return False

        return True

    def normalize_candidates(self, candidate_list: List[str]) -> List[str]:
        results = []

        for candidate in candidate_list or []:
            candidate = self._clean_ingredient_candidate(candidate)

            if not self._is_valid_ingredient_candidate(candidate):
                continue

            results.append(candidate)

        return results

    def _has_comma_separator(self, text: str) -> bool:
        if not text:
            return False

        return bool(re.search(r"[,，、]", text))

    # =========================================================
    # 7. 사용방법 / 주의사항 추출
    # =========================================================

    def extract_usage(self, text: str) -> List[str]:
        """
        사용방법 추출.

        강화 사항:
        - 텍스트 내부에 "효능효과", "용법용량", "주의사항" 같은 인라인 라벨이 끼어
          있으면 자기 섹션(usage) 텍스트만 골라낸다.
        - 분리된 개별 문장/항목을 list로 유지한다 (한 줄로 합치지 않는다).
        """
        usage_text = self._extract_inline_section_text(text, "usage")
        if not usage_text:
            return []

        return self._extract_sentence_section(
            text=usage_text,
            target_keywords=self.usage_keywords
        )

    def extract_cautions(self, text: str) -> List[str]:
        """
        주의사항 추출.

        강화 사항:
        - 인라인 라벨 분리로 "주의사항" 자체 영역만 골라낸다.
        - 번호 매김(1./2./3./4. 또는 가)/나)/다)) 기준으로 개별 항목 분리.
        - 메타 키워드 부분 일치를 허용한다 (전문의/상담 등).
        """
        cautions_text = self._extract_inline_section_text(text, "cautions")
        if not cautions_text:
            return []

        cleaned_text = self._remove_section_label(cautions_text, self.caution_keywords)
        if not cleaned_text:
            return []

        numbered_items = self._split_cautions_by_numbering(cleaned_text)
        if numbered_items:
            results = []
            for item in numbered_items:
                if not item:
                    continue
                if self._is_valid_sentence_section_text(item, allow_meta=True):
                    results.append(item)
            results = self.remove_duplicates(results)
            if results:
                return results

        return self._extract_sentence_section(
            text=cleaned_text,
            target_keywords=self.caution_keywords,
            allow_meta=True
        )

    def extract_effects(self, text: str) -> List[str]:
        """
        효능/효과 추출.

        강화 사항:
        - 텍스트 내부에 "효능효과" 인라인 라벨이 있으면 해당 청크만 추출.
        - 광고 카피로 시작하는 텍스트도 effects_content_keywords가 충분하면 허용.
        """
        effects_text = self._extract_inline_section_text(text, "effects")
        if not effects_text:
            return []

        return self._extract_sentence_section(
            text=effects_text,
            target_keywords=self.effects_header_keywords
        )

    # =========================================================
    # 7-A. 인라인 섹션 라벨 분리 / 주의사항 번호 매김 분리
    # =========================================================

    # 다른 섹션의 인라인 라벨 — 이 단어가 텍스트 중간에 나오면 거기서 끊는다
    _INLINE_USAGE_LABELS = [
        "사용방법", "사용법", "용법용량", "용법", "사용순서", "사용 방법", "사용 순서"
    ]
    _INLINE_CAUTION_LABELS = [
        "사용상의 주의사항", "사용할 때의 주의사항", "사용시의 주의사항",
        "사용시 주의사항", "사용상주의사항", "주의사항", "경고"
    ]
    _INLINE_EFFECTS_LABELS = [
        "효능효과", "효능 효과", "효능", "효과",
        "주요기능", "기능성", "주요특징", "제품특징"
    ]
    _INLINE_INGREDIENT_LABELS = [
        "전성분", "주요성분", "주성분", "성분명", "원료명"
    ]

    def _extract_inline_section_text(self, text: str, target_section: str) -> str:
        """
        하나의 섹션 텍스트 안에 다른 섹션 라벨이 인라인으로 끼어 있을 때
        target_section에 해당하는 청크만 골라 반환한다.

        예) usage_text가
            "사용방법 ... 줍니다 [미백 주름개선 효능효과 ... 도움을 준다 용법용량 ... 바른다"
            target_section="usage" → "사용방법 ... 줍니다", "용법용량 ... 바른다" 만 모음
            target_section="effects" → "효능효과 ... 도움을 준다" 만 모음

        반환은 \n 결합 텍스트. 매칭이 없으면 입력 텍스트 전체 반환 (호환성).
        """
        if not text:
            return ""

        text = self._clean_general_text_preserve_newline(text)

        label_to_section = []
        for lbl in self._INLINE_USAGE_LABELS:
            label_to_section.append((lbl, "usage"))
        for lbl in self._INLINE_CAUTION_LABELS:
            label_to_section.append((lbl, "cautions"))
        for lbl in self._INLINE_EFFECTS_LABELS:
            label_to_section.append((lbl, "effects"))
        for lbl in self._INLINE_INGREDIENT_LABELS:
            label_to_section.append((lbl, "ingredients"))

        # 텍스트에서 라벨이 등장하는 위치(가장 긴 매칭 우선)를 찾는다
        matches = []
        for label, section in label_to_section:
            for m in re.finditer(re.escape(label), text, flags=re.IGNORECASE):
                matches.append((m.start(), m.end(), section, label))

        if not matches:
            return text

        # 같은 위치 시작이면 더 긴 라벨이 우선
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        # 겹치는 매칭 제거 (앞 매칭 끝 이전에 시작하면 스킵)
        filtered = []
        last_end = -1
        for start, end, section, label in matches:
            if start < last_end:
                continue
            filtered.append((start, end, section, label))
            last_end = end

        # 각 매칭을 경계로 청크 생성: 청크 i 는 매칭 i 라벨 이후 ~ 다음 매칭 시작 까지
        # 매칭 이전의 prefix 청크는 정보 부족 — 라우팅에 사용하지 않는다
        chunks_by_section: Dict[str, List[str]] = {}
        for i, (start, end, section, label) in enumerate(filtered):
            chunk_start = end
            chunk_end = filtered[i + 1][0] if (i + 1) < len(filtered) else len(text)
            body = text[chunk_start:chunk_end].strip(" :,.;·ㆍ[]()\n\t")
            if not body:
                continue
            chunks_by_section.setdefault(section, []).append(body)

        target_chunks = chunks_by_section.get(target_section, [])

        # target_section 라벨이 한 번도 안 나왔으면, 라벨 이전 prefix가
        # 해당 섹션의 본문일 가능성 (예: section_detector가 이미 분류한 영역)
        # → 첫 매칭 라벨 이전 prefix를 target_section의 후보로 추가
        if filtered:
            prefix = text[: filtered[0][0]].strip(" :,.;·ㆍ[]()\n\t")
            if prefix:
                # prefix는 target_section 라벨이 없었으므로
                # 외부에서 이미 분류한 target_section 영역으로 간주
                target_chunks.insert(0, prefix)

        if not target_chunks:
            return ""

        return "\n".join(target_chunks).strip()

    def _split_cautions_by_numbering(self, text: str) -> List[str]:
        """
        주의사항 텍스트를 번호 매김 기준으로 항목 분리한다.

        지원하는 번호 패턴:
        - "1.", "2.", "3.", "4."  (아라비아 숫자 + 마침표)
        - "1)", "2)", "3)"          (아라비아 숫자 + 닫는 괄호)
        - "가)", "나)", "다)"        (한글 + 닫는 괄호)
        - "①", "②", "③"            (원숫자)

        번호 표지를 만나면 그 앞 텍스트를 항목으로 끊는다.
        """
        if not text:
            return []

        # 텍스트 한 줄로 평탄화 (번호 매김은 줄 안에 섞여 들어오는 경우가 많음)
        flat = re.sub(r"\s+", " ", text).strip()
        if not flat:
            return []

        # 번호 표지 패턴 (앞에 공백 또는 시작)
        # \b 가 한글에서 잘 안 먹어서 명시적 lookbehind 사용
        numbering_pattern = re.compile(
            r"(?:(?<=\s)|(?<=^))"
            r"(?:"
            r"\d{1,2}\s*[.\)]\s*"          # 1. 2) 등
            r"|"
            r"[가-힣]\s*\)\s*"              # 가) 나) 등
            r"|"
            r"[①-⑳㈀-㈎]\s*"               # 원숫자/원문자
            r")"
        )

        # split with position tracking
        splits = []
        last_idx = 0
        for m in numbering_pattern.finditer(flat):
            if m.start() > last_idx:
                splits.append(flat[last_idx:m.start()].strip())
            last_idx = m.start()
        splits.append(flat[last_idx:].strip())

        # 결과 정리: 너무 짧은 조각/번호만 있는 조각 제거
        results = []
        for piece in splits:
            piece = piece.strip(" :,.;·ㆍ[]()\n\t")
            if not piece:
                continue
            if len(piece) < 4:
                continue
            results.append(piece)

        # 번호 매김이 실제로 의미 있게 분리되었을 때만 반환
        # (한 항목만 나왔다면 분리 효과가 없음 → 빈 list로)
        if len(results) <= 1:
            return []

        return results

    def _strip_advertising_copy_from_product_name(self, text: str) -> str:
        """
        제품명 텍스트에서 광고 카피 어구를 제거한다.

        - "보습감을 전달하여", "선사해 주는", "도움을 주는" 등의 동사구 이후 제거
        - "-한", "-된", "-는", "-해", "-하여", "-여서" 등 연결어미로 끝나는 어구 제거
        - 제거 후 단어 수가 극단적으로 줄어들면 (원본 어휘의 30% 미만) 원본 반환
        """
        if not text:
            return ""

        original = text.strip()

        # 한국어 화장품 라벨의 보편적 광고 표현 패턴 (특정 제품 어휘 X)
        cut_patterns = [
            # ① "[명사]감을 전달/선사/제공/부여하여" 같은 광고 동사구
            #    — 보습감/수분감/탄력감 등 다양한 명사+감 일반화
            r"\s*[가-힣]{1,4}감을?\s*(?:전달|선사|제공|부여|선물)하[여는을며고지]\s*",
            # ② 광고 동사 단독: "전달하여", "선사하는" 등 (앞 명사 누락된 경우)
            r"\s*(?:전달|선사|부여|선물)하[여는을며고지]\s*",
            # ③ "X에 도움을 주는" — 화장품법 표준 효능 표현
            r"\s*(?:에\s*)?도움을\s*주[는며고지]\s*",
            # ④ 광고 표현: "함유되어"
            r"\s*함유되어\s*",
            # ⑤ "위해 만든", "위한"
            r"\s*위해\s*만든\s*",
            # ⑥ "마치 ~ 같은/처럼/듯한" 비유 표현
            r"\s*마치\s+[가-힣A-Za-z]+\s+(?:같은|처럼|듯한)\s*",
        ]

        cut_index = None
        for pattern in cut_patterns:
            match = re.search(pattern, original)
            if match and match.start() > 0:
                if cut_index is None or match.start() < cut_index:
                    cut_index = match.start()

        if cut_index is not None and cut_index >= 3:
            head = original[:cut_index].strip(" ,.-")
            if head and len(head) >= 3:
                # 잘라낸 결과가 너무 짧으면 (한 단어) 원본 반환
                if len(head.split()) >= 1 and len(head) >= 4:
                    return head

        return original

    def _extract_sentence_section(
        self,
        text: str,
        target_keywords: List[str],
        allow_meta: bool = False
    ) -> List[str]:
        if not text:
            return []

        text = self._clean_general_text_preserve_newline(text)
        text = self._remove_section_label(text, target_keywords)

        if not text:
            return []

        candidates = self._split_lines_or_sentences(text)
        results = []

        for candidate in candidates:
            candidate = self._clean_sentence_text(candidate)

            if not candidate:
                continue

            if not self._is_valid_sentence_section_text(candidate, allow_meta=allow_meta):
                continue

            results.append(candidate)

        results = self.remove_duplicates(results)

        if not results and self._is_valid_sentence_section_text(text, allow_meta=allow_meta):
            results = [text]

        return results

    def _clean_sentence_text(self, text: str) -> str:
        if not text:
            return ""

        text = self._clean_general_text(text)
        text = re.sub(r"^[\s:：,.;/·ㆍ\-]+", "", text)
        text = re.sub(r"[\s:：,.;/·ㆍ\-]+$", "", text)

        return text.strip()

    def _is_valid_sentence_section_text(
        self,
        text: str,
        allow_meta: bool = False
    ) -> bool:
        if not text:
            return False

        text = self._clean_general_text(text)

        if len(text) < 2:
            return False

        if self._looks_like_capacity_text(text):
            return False

        if self._looks_like_url(text):
            return False

        if self._looks_like_phone_number(text):
            return False

        if self._looks_like_ingredient_list_text(text):
            return False

        # [완화] cautions처럼 "전문의 ... 상담할것" 등 메타 키워드 부분일치가
        # 흔한 섹션은 allow_meta=True로 호출되어 필터를 우회한다.
        if not allow_meta and self._is_manufacturer_or_meta_text(text):
            return False

        return True

    def _looks_like_usage_sentence(self, text: str) -> bool:
        """
        [수정] "사용", "use" 단독 키워드 제거.
        기존에 이 단어들이 있어서 성분 내용 줄이 usage로 오분류됨.
        명확한 사용방법 표현만 남긴다.
        """

        if not text:
            return False

        compact = self._compact(text)

        usage_signals = [
            "사용방법", "사용법", "사용순서",
            "바르", "발라", "도포", "흡수", "마사지",
            "세안", "눈가", "얼굴", "피부", "적당량",
            "apply", "massage", "rinse"
        ]

        if self._contains_keyword(compact, usage_signals):
            if not self._has_ingredient_structure(text):
                return True

        return False

    def _looks_like_caution_sentence(self, text: str) -> bool:
        """
        [수정] "주의" 단독 키워드 제거.
        기존에 이 단어가 있어서 성분 내용 줄이 cautions로 오분류됨.
        명확한 주의사항 표현만 남긴다.
        """

        if not text:
            return False

        compact = self._compact(text)

        caution_signals = [
            "주의사항", "사용상주의",
            "피부이상", "붉은반점",
            "부어오름", "가려움", "자극",
            "상처", "습진", "피부염",
            "직사광선", "어린이", "화기", "경고",
            "warning", "caution", "precaution"
        ]

        return self._contains_keyword(compact, caution_signals)

    def _looks_like_effects_sentence(self, text: str) -> bool:
        """
        효능/장점 광고 카피 라인 판별.

        조건:
        - effects_content_keywords 단서 단어 포함
        - 사용방법/주의사항 문장이 아님 (명령형/금지형 어미 없음)
        - 성분 나열·URL·용량·메타 텍스트가 아님
        """

        if not text:
            return False

        if self._looks_like_url(text):
            return False

        if self._looks_like_capacity_text(text):
            return False

        if self._looks_like_ingredient_list_text(text):
            return False

        if self._is_manufacturer_or_meta_text(text):
            return False

        if self._looks_like_usage_sentence(text):
            return False

        if self._looks_like_caution_sentence(text):
            return False

        compact = self._compact(text)

        return self._contains_keyword(compact, self.effects_content_keywords)

    def _split_lines_or_sentences(self, text: str) -> List[str]:
        if not text:
            return []

        raw_parts = re.split(r"[\n\r]+", str(text))

        if len(raw_parts) <= 1:
            raw_parts = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s+", str(text))

        results = []

        for part in raw_parts:
            part = self._clean_general_text(part)

            if part:
                results.append(part)

        return results

    # =========================================================
    # 8. QR / URL 추출
    # =========================================================

    def extract_qr_candidates(self, text: str) -> List[str]:
        if not text:
            return []

        text = self._restore_url_like_text(text)
        results = []

        url_patterns = [
            r"https?://[^\s]+",
            r"www\.[^\s]+",
            r"[a-zA-Z0-9.-]+\.(?:com|co\.kr|kr|net|org|io|ai)[^\s]*"
        ]

        for pattern in url_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            results.extend(matches)

        cleaned_results = []

        for item in results:
            item = item.strip(" ,.;:：()[]{}<>")

            if item:
                cleaned_results.append(item)

        if not cleaned_results and self._contains_keyword(
            self._compact(text), self.qr_keywords
        ):
            cleaned_results.append(text)

        return self.remove_duplicates(cleaned_results)

    def _restore_url_like_text(self, text: str) -> str:
        if not text:
            return ""

        text = self._clean_general_text(text)
        text = re.sub(r"\bwww([a-zA-Z0-9]+)co\.?kr\b", r"www.\1.co.kr", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwww([a-zA-Z0-9]+)\.com\b", r"www.\1.com", text, flags=re.IGNORECASE)
        text = re.sub(r"(https?)\s*:\s*/\s*/", r"\1://", text, flags=re.IGNORECASE)

        return text.strip()

    # =========================================================
    # 9. raw_text 기반 성분 추정
    # =========================================================

    def _guess_ingredient_text_from_raw(self, raw_text: str) -> str:
        if not raw_text:
            return ""

        lines = [
            self._clean_general_text(line)
            for line in str(raw_text).splitlines()
            if self._clean_general_text(line)
        ]

        collected = []
        mode = False
        gap_count = 0

        # 헤더로만 인정할 엄격한 키워드 (loose "성분"/"원료" 제외)
        # 이유: 광고 카피 "성분이 함유되어" 가 "성분" 부분 매칭으로
        #       ingredients-mode 를 켜는 오류 방지
        strict_ingredient_headers = [
            "전성분", "전 성분", "전 성 분",
            "주요성분", "주요 성분",
            "주성분", "주 성분", "주 성 분",
            "성분명", "원료명",
            "ingredient list", "main ingredients", "key ingredients",
            "ingredients", "ingredient"
        ]

        for line in lines:
            compact = self._compact(line)

            if not mode:
                # [완화] 엄격한 헤더 키워드만 사용 (성분/원료 단독은 제외)
                if self._contains_keyword(compact, strict_ingredient_headers):
                    mode = True
                    value = self._remove_section_label(line, self.ingredient_keywords)

                    if value:
                        collected.append(value)

                    continue

                if self._looks_like_ingredient_list_text(line):
                    mode = True
                    collected.append(line)
                    continue

                continue

            if self._contains_keyword(compact, self.stop_section_keywords):
                break

            if self._is_manufacturer_or_meta_text(line):
                break

            if self._looks_like_usage_sentence(line):
                break

            if self._looks_like_caution_sentence(line):
                break

            if self._has_ingredient_structure(line):
                gap_count = 0
                collected.append(line)
            elif gap_count < self.ingredient_gap_tolerance:
                gap_count += 1
                collected.append(line)
            else:
                break

        joined = " ".join(collected)
        return self._prepare_ingredient_section_text(joined)

    # =========================================================
    # 10. 공통 텍스트 정리
    # =========================================================

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = str(text)

        text = re.sub(
            r"[^가-힣a-zA-Z0-9\s,./·ㆍ;:+\-()_%\[\]：:]",
            " ",
            text
        )

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _clean_general_text(self, text: Any) -> str:
        if not text:
            return ""

        text = str(text)
        text = text.replace("\n", " ").replace("\t", " ")

        replace_map = {
            "，": ",", "、": ",", "；": ";", "ㆍ": "·", "：": ":",
            "（": "(", "）": ")",
            "㎖": "ml", "ｍｌ": "ml", "ＭＬ": "ml",
            "ｇ": "g", "Ｇ": "g", "㎎": "mg", "㎏": "kg",
            "–": "-", "—": "-", "−": "-"
        }

        for old, new in replace_map.items():
            text = text.replace(old, new)

        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*,\s*", ", ", text)
        text = re.sub(r"\s*;\s*", "; ", text)
        text = re.sub(r"\s*:\s*", ": ", text)
        text = re.sub(r"\s+([,.;:)\]\}])", r"\1", text)
        text = re.sub(r"([(\[\{])\s+", r"\1", text)

        return text.strip()

    def _clean_general_text_preserve_newline(self, text: Any) -> str:
        if not text:
            return ""

        text = str(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

        lines = []

        for line in text.split("\n"):
            line = self._clean_general_text(line)

            if line:
                lines.append(line)

        return "\n".join(lines).strip()

    def _remove_short_english_noise_inside_line(self, text: str) -> str:
        if not text:
            return ""

        tokens = str(text).split()
        cleaned_tokens = []
        compact_noise_words = [self._compact(word) for word in self.noise_words]

        for token in tokens:
            stripped = token.strip(" ,.;:：/·ㆍ-()[]{}")

            if not stripped:
                continue

            if self._compact(stripped) in compact_noise_words:
                continue

            cleaned_tokens.append(token)

        return " ".join(cleaned_tokens).strip()

    def _remove_section_label(self, text: str, keywords: List[str]) -> str:
        if not text:
            return ""

        result = str(text).strip()

        for keyword in keywords:
            pattern = re.escape(keyword)

            result = re.sub(
                rf"^\s*{pattern}\s*[:：]?\s*",
                "",
                result,
                flags=re.IGNORECASE
            )

        return result.strip()

    def _cut_text_before_stop_section(self, text: str) -> str:
        if not text:
            return ""

        cut_index = None

        for keyword in self.stop_section_keywords:
            match = re.search(re.escape(keyword), text, flags=re.IGNORECASE)

            if match:
                index = match.start()

                if index > 0:
                    if cut_index is None or index < cut_index:
                        cut_index = index

        if cut_index is not None:
            return text[:cut_index].strip()

        return text.strip()

    # =========================================================
    # 11. 판별 유틸
    # =========================================================

    def _looks_like_ingredient_list_text(self, text: str) -> bool:
        if not text:
            return False

        text = self._clean_general_text(text)

        if self._is_noise_line(text):
            return False

        if self._is_manufacturer_or_meta_text(text):
            return False

        if self._sentence_penalty_score(text) >= 2:
            return False

        # [완화] 광고/효능 문장 오분류 방지
        # 효능 카피 단어가 2개 이상이고 구분자가 거의 없으면 ingredient_list 아님
        # 예: "...진정과 보호에 도움을 주고 건조한 피부에 깊고 진한"
        compact_for_effects = self._compact(text)
        effects_signal_count = sum(
            1 for kw in self.effects_content_keywords
            if self._compact(kw) and self._compact(kw) in compact_for_effects
        )
        if effects_signal_count >= 2 and self._count_separators(text) == 0:
            return False

        # 구분자 기반 판단 (특정 성분명 목록 미사용)
        # 단, 숫자 쉼표(예: "1, 2-", "(4, 800 ppm)")는 제외
        sep = self._count_separators(text)
        numeric_commas = len(re.findall(r"\d\s*[,，、]\s*\d", text))
        effective_sep = max(0, sep - numeric_commas)

        if effective_sep >= 2:
            return True

        if effective_sep >= 1:
            tokens = self._split_by_separators(text)
            if len(tokens) >= 2:
                return True

        # 토큰 수 기반 판단
        tokens = re.split(r"[,，、;；·ㆍ/\s]+", text)
        tokens = [t.strip() for t in tokens if t.strip()]

        if len(tokens) >= 5:
            valid_count = sum(
                1 for t in tokens
                if 2 <= len(t) <= 40 and not self._is_sentence_like(t)
            )

            # [완화] 토큰 수 휴리스틱 강화:
            # 효능/주의/사용 단어가 섞여 있으면 성분 목록이 아님
            usage_or_caution_signals = (
                "도움" in text or "준다" in text or "주고" in text
                or "함유되어" in text or "성분이" in text
                or "줍니다" in text or "바릅니다" in text
                or "하십시오" in text or "마세요" in text
            )

            if valid_count >= 5 and not usage_or_caution_signals:
                # 한글 성분 어미 패턴이 2개 이상이어야 진짜 성분 목록
                ingredient_suffix_hits = sum(
                    1 for t in tokens
                    if re.search(
                        r"(?:추출물|오일|버터|왁스|애[씨시]드|에이트|레이트|이드|"
                        r"클로라이드|글라이콜|폴리머|에테르|하이드록사이드|"
                        r"설페이트|페이트|향료|색소|비타민)",
                        t
                    )
                )
                if ingredient_suffix_hits >= 2:
                    return True

        return False

    def _is_sentence_like(self, text: str) -> bool:
        return self._sentence_penalty_score(text) >= 1

    def _sentence_penalty_score(self, text: str) -> int:
        if not text:
            return 0

        compact = self._compact(text)
        score = 0

        for keyword in self.sentence_negative_keywords:
            if self._compact(keyword) in compact:
                score += 1

        if re.search(r"(하십시오|하세요|합니다|됩니다|있습니다|하지말것|버리지말것|사용하지)", text):
            score += 2

        if len(text) >= 45 and text.count(",") == 0:
            score += 1

        return score

    def _is_manufacturer_or_meta_text(self, text: str) -> bool:
        if not text:
            return False

        compact = self._compact(text)

        if self._contains_keyword(compact, self.manufacturer_keywords):
            return True

        if self._contains_keyword(compact, self.meta_noise_keywords):
            return True

        if self._looks_like_phone_number(text):
            return True

        if re.search(r"서울|경기|인천|부산|대구|광주|대전|울산|충북|충남|청주|주소", text):
            return True

        return False

    def _is_noise_line(self, text: str) -> bool:
        if not text:
            return True

        compact = self._compact(text)

        if not compact:
            return True

        for noise in self.noise_words:
            if self._compact(noise) == compact:
                return True

        if "소비자가" in text or "신문" in text:
            return True

        if re.fullmatch(r"[a-zA-Z]{1,4}", text):
            return True

        if re.fullmatch(r"[가-힣]{1,2}", text):
            return True

        return False

    def _is_noise_word(self, text: str) -> bool:
        if not text:
            return True

        compact = self._compact(text)

        if not compact:
            return True

        for noise in self.noise_words:
            if self._compact(noise) == compact:
                return True

        if re.fullmatch(r"[a-zA-Z]{1,4}", text):
            return True

        return False

    def _looks_like_url(self, text: str) -> bool:
        if not text:
            return False

        return bool(
            re.search(
                r"https?://|www\.|\.com|\.co\.kr|\.kr|\.net|\.org|\.io|\.ai",
                text,
                flags=re.IGNORECASE
            )
        )

    def _looks_like_phone_number(self, text: str) -> bool:
        if not text:
            return False

        return bool(re.search(r"\d{2,4}-\d{3,4}-\d{4}", text))

    # =========================================================
    # 12. 리스트 / 키워드 유틸
    # =========================================================

    def remove_duplicates(self, items: List[str]) -> List[str]:
        results = []
        seen = set()

        for item in items or []:
            if item is None:
                continue

            text = str(item).strip()

            if not text:
                continue

            key = self._compact_for_dedup(text)

            if key not in seen:
                seen.add(key)
                results.append(text)

        return results

    def _contains_keyword(self, compact_text: str, keywords: List[str]) -> bool:
        for keyword in keywords:
            key = self._compact(keyword)

            if key and key in compact_text:
                return True

        return False

    def _compact(self, text: Any) -> str:
        return (
            str(text)
            .lower()
            .replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
            .replace("-", "").replace("_", "").replace(":", "").replace("：", "")
            .replace(".", "").replace(",", "").replace("，", "")
            .replace(";", "").replace("；", "").replace("/", "")
            .replace("·", "").replace("ㆍ", "")
            .replace("[", "").replace("]", "")
            .replace("(", "").replace(")", "")
            .replace("{", "").replace("}", "")
        )

    def _compact_for_dedup(self, text: Any) -> str:
        return (
            str(text)
            .lower()
            .replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
        )
