import re
from typing import Dict, Any, List, Optional


class OCRSectionDetector:
    """
    Dermalens OCR 구간 탐지 클래스

    Claude API 없이 PaddleOCR 결과만 이용해서 구간을 탐지한다.

    핵심 원칙:
    1. 특정 화장품 이미지 구조에 맞추지 않는다.
    2. 특정 성분명 목록에 의존하지 않는다.
    3. 이 파일은 성분을 확정하지 않는다.
    4. 성분 API 검증은 postprocess.py 또는 ingredient_api.py 단계에서 수행한다.
    5. OCR 결과를 제품명/용량/성분/사용방법/주의사항/QR/기타로 분류한다.

    [수정 사항 요약]

    (A) _is_section_header_line() 신설
        - 짧고 단독으로 의미가 완결되는 라벨 줄만 섹션 헤더로 인정한다.
        - 긴 내용 줄에 키워드가 포함되어도 헤더로 취급하지 않는다.
        - "전성분: 정제수, 글리세린, ..." 같은 콜론형 헤더도 인식한다.

    (B) _detect_explicit_section_label() 개선
        - 헤더 줄 여부를 먼저 확인한다.
        - 키워드 탐지 순서 재정렬: 복합 표현 → 단일 표현 순 (오분류 방지).
        - cautions를 ingredients/usage보다 먼저 탐지한다.

    (C) _detect_sections_from_lines() 개선
        - ingredients 구간에 별도 gap 카운터를 도입한다.
        - _should_continue_ingredients() 분리: 완화된 조건으로 판단.

    (D) _should_continue_ingredients() 신설
        - 기존: 쉼표 2개 이상 + 토큰 3개 이상이어야 계속
        - 수정: 쉼표 1개 이상 OR 토큰 2개 이상이면 계속
        - ingredient_gap_tolerance 이내 비성분 라인은 허용.

    (E) _looks_like_ingredient_structure() 임계값 완화
        - 기존: 구분자 2개 이상 + 토큰 3개 이상
        - 수정: 구분자 1개 이상 + 토큰 2개 이상 (완화)

    (F) _guess_ingredient_lines() 전면 개선
        - 기존: 가장 점수 높은 1개 라인 기준 위아래 확장
        - 수정: 전체 라인 후보 수집 → 클러스터링 → 최고 클러스터 선택
        - 2단 컬럼 성분표, 줄바꿈이 많은 성분표 모두 처리 가능.

    (G) _looks_like_usage_sentence() 개선
        - "사용" 단독 키워드 제거 → 오분류 방지.
        - 명확한 사용방법 표현만 사용.

    (H) _looks_like_caution_sentence() 개선
        - "주의" 단독 키워드 제거 → 오분류 방지.
        - 명확한 주의사항 표현만 사용.

    [이번 수정 사항 — 3단계 분류 로직 일반화]

    핵심 원칙:
    - 특정 단어("케라틴", "마사지", "팬틴" 등) 의존 금지
    - 한국어 화장품 라벨의 구조적 특성 (어미, 조사, 접미사) 만 사용
    - 한국어 화장품 라벨의 일반적 형식 (단위, 회사 접미사, 주소 표지) 만 사용

    (I) __init__에 일반화 정규식 패턴 추가
        - USAGE_IMPERATIVE_PATTERN: "~세요/십시오/합니다/냅니다" 등 명령형 어미
        - CAUTION_IMPERATIVE_PATTERN: "~할 것/말 것/마세요/금지/자제/피하" 등 금지/명령
        - COMPANY_SUFFIX_PATTERN: 주식회사/유한회사/Co.,Ltd/Inc/Corp/LLC/GmbH
        - ADDRESS_PATTERN: 특별시/광역시/-시/-구/-동/-로 ##/-길 ##/-번지/-층
        - PHONE_PATTERN: 한국/국제 전화번호 포맷
        - MANUFACTURE_META_PATTERN: EXP/MFG/LOT/사용기한/제조번호 등
        - DATE_PATTERN: 다양한 날짜 형식
        - KOREAN_INGREDIENT_SUFFIX_PATTERN: 한글 성분 보편 어미
          (-추출물/-오일/-산/-올/-이드/-페이트/-클로라이드/-에이트 등)
        - ADVERTISING_ENDING_PATTERN: 광고 카피 종결 어미
          (-한/-된/-의/-로/-함유/-증가된 등)

    (J) _looks_like_caution_sentence() 패턴 기반 강화
        기존: 단어 매칭만 ("피부이상", "직사광선" 등)
              → "보관할 것", "찢지 말 것" 등 일반 라벨 문구 누락
        수정: CAUTION_IMPERATIVE_PATTERN 추가
              한국어 화장품 라벨의 금지/명령 어미 보편 패턴

    (K) _looks_like_advertising_copy() 신설
        광고 카피의 구조적 특성으로 판별:
        - 형용사형 종결 어미 (광고 특유)
        - 명령형/평서형 종결 어미 없음
        - 쉼표 거의 없음 (성분 나열 아님)
        - 한글 성분 어미가 여러 개면 광고 아님

    (L) _is_manufacturer_or_meta_text() 패턴 기반 강화
        기존: 하드코딩 키워드 + "서울/경기..." 지역명
              → 외국 라벨, 새 회사명, OCR 깨짐에 약함
        수정: COMPANY/ADDRESS/PHONE/MANUFACTURE_META/DATE 패턴
              한국 행정구역과 화장품 라벨 메타 정보의 보편 형식

    (M) _detect_implicit_section() 분류 우선순위 재정렬
        기존: URL → capacity → ingredients → cautions → usage
              → 광고/회사정보/제조정보가 ingredients/usage로 흡수됨
        수정: URL → 메타 → 광고 → capacity → ingredients → cautions → usage
              메타와 광고를 우선 분리해서 others로 보냄

    (N) _looks_like_usage_sentence() 패턴 기반 일반화
        기존: "바르", "마사지" 등 특정 동사 의존
              → 샴푸/바디/클렌저 등 다른 동사 라벨 누락
        수정: USAGE_IMPERATIVE_PATTERN으로 명령/평서 어미 일반 매칭
              + 광고/메타/주의사항 배제 조건 강화
    """

    def __init__(self):
        self.section_types = [
            "product_name",
            "capacity",
            "ingredients",
            "usage",
            "cautions",
            "effects",
            "qr_url",
            "others"
        ]

        # ── 섹션 헤더 라벨 탐지용 키워드 ──────────────────────────────
        # 성분명 힌트가 아니라, 라벨명 탐지용이다.
        # 순서 중요: 복합·명확한 표현 → 짧은 단어 순.

        self.product_keywords = [
            "제품명", "제품 명", "품명", "상품명",
            "제품이름", "제품 이름",
            "product name", "item name", "name"
        ]

        self.capacity_keywords = [
            "내용량", "순중량", "충전량",
            "용량", "중량",
            "net wt", "net weight", "volume", "capacity", "contents"
        ]

        # 복합 표현 → 단일 표현 순 (오분류 방지에 중요)
        self.ingredient_keywords = [
            "전성분", "전 성분", "전 성 분",
            "주요성분", "주요 성분",
            "주성분", "주 성분", "주 성 분",
            "성분명", "원료명",
            "성분", "원료",
            "ingredient list", "main ingredients", "key ingredients",
            "ingredients", "ingredient"
        ]

        # 복합 표현 → 단일 표현 순
        self.usage_keywords = [
            "사용방법", "사용 방법",
            "사용순서", "사용 순서",
            "사용법", "용법",
            "how to use", "directions", "direction", "usage", "use"
        ]

        # 복합 표현 우선 (사용상주의사항이 주의사항보다 먼저)
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

        # 효능/장점 — 본문 단서 단어 (광고 카피 줄 안에서 매칭 시 effects로 분류)
        # 길이 짧은 어근 위주. 사용방법·주의사항과 겹치지 않는 단어만.
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

        # 기타/메타 정보 판별용 (보조용 — 핵심은 self.META_* 정규식들)
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

        # 성분으로 착각하기 쉬운 문장/메타 표현
        self.ingredient_negative_keywords = [
            "사용후", "사용 후", "사용하지", "사용하고", "사용하기",
            "바릅니다", "발라", "흡수", "마사지", "세안", "도포",
            "주의", "화기", "화기주의", "가연성", "가스", "환기", "불",
            "버리지", "장소", "보관", "경우", "반드시",
            "하십시오", "하세요", "합니다", "됩니다", "있습니다",
            "상담", "고객", "소비자", "제조", "판매", "주소",
            "교환", "반품", "분쟁", "공정거래",
            "분리수거", "분리배출", "용기", "상품", "상표",
            "원산지", "품질보증", "제조번호", "사용기한", "유통기한"
        ]

        # 헤더 줄로 인정할 최대 텍스트 길이
        # 이보다 긴 줄은 키워드가 있어도 헤더로 취급하지 않음
        self.header_max_length = 25

        # ingredients 구간 내 비성분 라인 연속 허용 개수
        # 이만큼 연속으로 비성분 라인이 나오면 ingredients 구간 종료
        self.ingredient_gap_tolerance = 3

        # =====================================================================
        # [신규] 일반화 패턴 (정규식 기반)
        #
        # 원칙:
        # - 특정 단어("케라틴", "글리세린", "마사지", "팬틴") 의존 금지
        # - 한국어의 구조적 특성 (어미, 조사, 접미사) 만 사용
        # - 한국어 화장품 라벨의 일반적 형식만 사용 (단위, 회사 접미사 등)
        # =====================================================================

        # ── 명령형 어미 패턴 (한국어 일반화) ───────────────────────────
        # 사용방법 문장에 자주 나오는 어미: "~하세요", "~십시오", "~합니다", "~냅니다"
        self.USAGE_IMPERATIVE_PATTERN = re.compile(
            r"(?:하|발라|적[셔신]|뿌리|짜|덜|문지[르러]|닦|씻|적[셔시]|두[드들]|"
            r"바|펴|얇게|넓게|적당량|소량|"
            r"세요|십시오|십시요|시오|합니다|냅니다|줍니다|줍시다|드립니다)"
            r"(?:\s*[.!]?\s*)?$",
            flags=re.IGNORECASE
        )

        # ── ㄹ 받침 음절 동적 생성 (한국어 일반화) ────────────────────
        # 한국어에서 "~할 것/말 것/낼 것/먹을 것/쓸 것" 등 ㄹ 받침 + "것" 종결은
        # 미래형/명령형 어미로 화장품 라벨 주의사항에 보편적으로 등장
        # 음절 코드: (코드 - 0xAC00) % 28 == 8 인 한글 음절들이 ㄹ 받침
        #            (총 399개 음절: 갈, 결, 골, 굴, 글, 끌, 날, 늘, 달, 들, ... 할)
        _rieul_syllables = "".join(
            chr(code)
            for code in range(0xAC00, 0xD7A4)
            if (code - 0xAC00) % 28 == 8
        )

        # 주의사항 문장에 자주 나오는 금지/명령형 어미
        # [수정] ㄹ받침 음절 동적 생성으로 진정한 일반화
        #   - 모든 ㄹ받침 음절 + "것" → 미래/명령형 어미
        #   - 명령형 평서: ~마세요/마십시오/마라/말아라
        #   - 금지 명사: 금지/자제
        #   - 부정 명령: 하지말/하지마/않도록/안돼
        self.CAUTION_IMPERATIVE_PATTERN = re.compile(
            r"(?:"
            # ① ㄹ 받침 음절 + (공백 0~1) + "것" — 진짜 일반화
            r"[" + _rieul_syllables + r"]\s*것"
            r"|"
            # ② 명령형 평서: ~마세요/마십시오/마라/말아라
            r"마세요|마십시오|마라|말아라"
            r"|"
            # ③ 금지/제한 명사형
            r"금지|자제"
            r"|"
            # ④ 피하다
            r"피하[세십]|피할\s*것"
            r"|"
            # ⑤ 부정 명령
            r"하지\s*(?:말|마|않)|않도록|안\s*돼|안돼"
            r")",
            flags=re.IGNORECASE
        )

        # ── 회사/법인 접미사 (전 세계 일반화) ──────────────────────────
        # 한국: 주식회사, 유한회사, (주)
        # 영문: Co.,Ltd / Inc / Corp / GmbH / LLC
        self.COMPANY_SUFFIX_PATTERN = re.compile(
            r"(?:주식\s*회사|유한\s*회사|유한책임\s*회사|\(\s*주\s*\)|"
            r"co\.?\s*,?\s*ltd\.?|inc\.?|corp\.?|llc\.?|gmbh|s\.?a\.?|"
            r"판매\s*유한\s*회사|책임판매업자|제조판매업자|화장품책임판매업자)",
            flags=re.IGNORECASE
        )

        # ── 한국 주소 표지 (행정구역/도로명) ──────────────────────────
        # [수정] "샤워 시", "사용 시" 같은 일반 부사 표현과 충돌 방지
        #   화장품 라벨에서 "~ 시" 부사 사용이 매우 흔함 → "시" 단독 매칭은 위험
        #   명확한 패턴(특별시/광역시/주소 끝 표지)만 유지
        self.ADDRESS_PATTERN = re.compile(
            r"(?:"
            # ① 명확한 행정구역 단어 (조합어로만)
            r"특별시|광역시|특별자치시|특별자치도"
            r"|"
            # ② 도시명 + 구/군 + 공백 + 한글 — "강남구 테헤란", "해운대구 우동"
            #    "시"는 부사 충돌 위험이 높아 제외
            r"[가-힣]{2,}\s*(?:구|군)\s+[가-힣]"
            r"|"
            # ③ 도로명: ~로 + 숫자, ~길 + 숫자 (이미 숫자 필수라 안전)
            r"[가-힣]{2,}\s*(?:로|길)\s*\d+"
            r"|"
            # ④ 번지/호/층 (숫자 + 행정 표지)
            r"\d+\s*(?:번지|층|호)"
            r")"
        )

        # ── 전화번호 (한국/국제 일반화) ──────────────────────────────
        # 080-xxxx-xxxx, 02-xxxx-xxxx, 010-xxxx-xxxx, 1577-xxxx 등
        self.PHONE_PATTERN = re.compile(
            r"(?<!\d)"
            r"(?:\d{2,4}[-\s.]\d{3,4}[-\s.]\d{4}|"
            r"1\d{3}[-\s.]?\d{4}|"
            r"\+?\d{1,3}[-\s.]\d{1,4}[-\s.]\d{3,4}[-\s.]\d{4})"
            r"(?!\d)"
        )

        # ── 제조 메타 정보 패턴 (EXP / MFG / LOT / 사용기한 / 제조번호) ──
        # 라벨에 흔히 인쇄되는 식별자
        self.MANUFACTURE_META_PATTERN = re.compile(
            r"(?:exp|mfg|lot|batch|prod|use\s*by|best\s*by|"
            r"제조\s*번호|제조\s*일자|제조\s*년월일|"
            r"사용\s*기한|유통\s*기한|소비\s*기한|"
            r"표시\s*숫자|별도\s*표[시기])"
            r"(?:\s*[:.]?\s*\S+)?",
            flags=re.IGNORECASE
        )

        # ── 날짜 패턴 (제조일/사용기한 인식) ─────────────────────────
        self.DATE_PATTERN = re.compile(
            r"(?:20\d{2}|19\d{2})[.\-/년]\s*\d{1,2}[.\-/월]\s*\d{0,2}|"
            r"\d{4}\.\d{2}\.\d{2}|"
            r"\d{2}/\d{2}/\d{2,4}|"
            r"mm\s*/\s*y{1,4}"
            ,
            flags=re.IGNORECASE
        )

        # ── 한글 성분 어미 패턴 (일반화된 명사 접미사) ─────────────────
        # 화장품 성분명에 보편적으로 나타나는 한글 접미사 (특정 성분명 X)
        self.KOREAN_INGREDIENT_SUFFIX_PATTERN = re.compile(
            r"(?:추출물|오일|버터|왁스|"
            r"애[씨시]드|산\b|"
            r"올\b|에놀|"
            r"에이트|레이트|"
            r"이드|아이드|마이드|"
            r"클로라이드|"
            r"글라이콜|글리세린|"
            r"폴리머|폴리올|"
            r"에테르|에스테르|"
            r"아민|아민산|"
            r"하이드록사이드|"
            r"설페이트|페이트|"
            r"실리콘|실란|"
            r"세틸|세테아릴|스테아릴|라우릴|"
            r"향료|색소|"
            r"비타민|에센스|"
            r"단백질)"
        )

        # ── 영문 성분명 패턴 (영어식 성분 표기) ───────────────────────
        # CamelCase 단어 또는 PascalCase 단어 연속 (예: Sodium Chloride)
        self.ENGLISH_INGREDIENT_PATTERN = re.compile(
            r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{1,}){0,3}\b"
        )

        # ── 광고 카피 패턴 (수식 어구 / 강조 표현) ──────────────────────
        # 광고 카피의 특성:
        # 1) 형용사형/수식형 (한 X, 된 X, 의 X)
        # 2) 강조 수치 (XX% 증가/함유)
        # 3) 명령형/평서형 어미 없음 (-세요/-십시오/-니다 등)
        # 4) 쉼표 거의 없음 (성분처럼 나열 안 함)
        #
        # [수정] 끝에서만 매칭 → 텍스트 어디든 포함되면 매칭
        #        + %증가/%함유 같은 강조 표현 추가
        self.ADVERTISING_ENDING_PATTERN = re.compile(
            r"(?:"
            # ① 강조 수치 패턴 — 광고에서 매우 흔함 ("200% 증가된", "30% 함유")
            r"\d+\s*%\s*(?:증가|함유|보강|강화|개선|상승|향상)"
            r"|"
            # ② 광고용 형용사 수식 어구 (어디에 있어도 OK)
            r"(?:증가된|보강된|개선된|강화된|향상된|풍부한|진한|새로운|"
            r"가득한|담은|만든|위한|에서|을\s*위한|로서|함유)"
            r"|"
            # ③ "마치 ~ 같은/듯한" 비유 표현 (광고에 자주)
            r"마치\s*[가-힣]+|"
            r"[가-힣]+\s*(?:같은|듯한|처럼)"
            r")"
        )

        # ── 한국어 한 글자 어미 보조 (퍼지 매칭용) ────────────────────
        # 짧은 헤더 단어가 OCR로 1글자 깨졌을 때 인식
        self.MAX_FUZZY_DISTANCE = 1
        self.FUZZY_MIN_LENGTH = 3   # 3글자 이상인 헤더만 퍼지 매칭 적용
        self.FUZZY_MAX_LENGTH = 10  # 10글자 이상은 정확 매칭만

    # =========================================================
    # 1. 메인 구간 탐지
    # =========================================================

    def detect(self, ocr_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not ocr_result:
            return self._empty_result(reason="ocr_result가 비어 있습니다.")

        print("[구간탐지] OCR 기반 구간 탐지 시작")

        raw_text = ocr_result.get("raw_text", "") or ""
        layout_text = ocr_result.get("layout_text", "") or raw_text
        ocr_lines = ocr_result.get("ocr_lines", []) or []
        ocr_blocks = ocr_result.get("ocr_blocks", []) or []

        normalized_lines = self._normalize_ocr_lines(
            ocr_lines=ocr_lines,
            raw_text=raw_text
        )

        if not normalized_lines:
            return self._empty_result(
                reason="구간 탐지에 사용할 OCR line이 없습니다.",
                raw_text=raw_text,
                layout_text=layout_text,
                ocr_lines=ocr_lines,
                ocr_blocks=ocr_blocks
            )

        sections = self._detect_sections_from_lines(normalized_lines)

        sections = self._fill_missing_sections_by_global_guess(
            sections=sections,
            lines=normalized_lines
        )

        sections = self._fill_others_section(
            sections=sections,
            lines=normalized_lines
        )

        section_text_map = self._build_section_text_map(sections)
        merged_text_by_section = self._build_merged_text_by_section(section_text_map)
        detected_sections = self._build_detected_section_metadata(sections)

        result = {
            "success": True,
            "mode": "ocr_bbox_section_detection_generalized",
            "section_text_map": section_text_map,
            "merged_text_by_section": merged_text_by_section,
            "detected_sections": detected_sections,
            "section_detection_summary": {
                "line_count": len(normalized_lines),
                "block_count": len(ocr_blocks),
                "detected_section_count": len(
                    [
                        section_type
                        for section_type in self.section_types
                        if merged_text_by_section.get(section_type)
                    ]
                ),
                "detected_section_types": [
                    section_type
                    for section_type in self.section_types
                    if merged_text_by_section.get(section_type)
                ]
            },
            "raw_text": raw_text,
            "layout_text": layout_text,
            "ocr_lines": normalized_lines,
            "ocr_blocks": ocr_blocks,
            "source_ocr_summary": ocr_result.get("ocr_summary", {}),
            "selected_variant": ocr_result.get("selected_variant", "")
        }

        print("[구간탐지] OCR 기반 구간 탐지 완료")

        return result

    # =========================================================
    # 2. OCR line 정규화
    # =========================================================

    def _normalize_ocr_lines(
        self,
        ocr_lines: List[Dict[str, Any]],
        raw_text: str = ""
    ) -> List[Dict[str, Any]]:
        normalized = []

        if ocr_lines:
            for index, line in enumerate(ocr_lines):
                if not isinstance(line, dict):
                    continue

                text = self._clean_text(line.get("text", ""))

                if not text:
                    continue

                if self._is_noise_line(text):
                    continue

                bbox = line.get("bbox", {}) or {}

                x1 = self._to_float(bbox.get("x1", line.get("x_min", 0)))
                y1 = self._to_float(bbox.get("y1", line.get("y_min", 0)))
                x2 = self._to_float(bbox.get("x2", line.get("x_max", 0)))
                y2 = self._to_float(bbox.get("y2", line.get("y_max", 0)))

                normalized.append(
                    {
                        "line_index": line.get("line_index", index),
                        "text": text,
                        "confidence": self._to_float(line.get("confidence", 0.0)),
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        },
                        "x_min": x1,
                        "y_min": y1,
                        "x_max": x2,
                        "y_max": y2,
                        "center_x": self._to_float(line.get("center_x", (x1 + x2) / 2)),
                        "center_y": self._to_float(line.get("center_y", (y1 + y2) / 2)),
                        "width": self._to_float(line.get("width", x2 - x1)),
                        "height": self._to_float(line.get("height", y2 - y1)),
                        "blocks": line.get("blocks", [])
                    }
                )

        else:
            for index, raw_line in enumerate(str(raw_text or "").splitlines()):
                text = self._clean_text(raw_line)

                if not text:
                    continue

                if self._is_noise_line(text):
                    continue

                normalized.append(
                    {
                        "line_index": index,
                        "text": text,
                        "confidence": 0.0,
                        "bbox": {
                            "x1": 0.0,
                            "y1": float(index),
                            "x2": 0.0,
                            "y2": float(index)
                        },
                        "x_min": 0.0,
                        "y_min": float(index),
                        "x_max": 0.0,
                        "y_max": float(index),
                        "center_x": 0.0,
                        "center_y": float(index),
                        "width": 0.0,
                        "height": 1.0,
                        "blocks": []
                    }
                )

        return sorted(
            normalized,
            key=lambda item: (
                item.get("center_y", 0),
                item.get("x_min", 0)
            )
        )

    # =========================================================
    # 3. line 기반 구간 탐지
    # =========================================================

    def _detect_sections_from_lines(
        self,
        lines: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        sections = self._empty_section_line_map()

        active_section = None
        active_start_y = None
        ingredient_gap_count = 0

        for line in lines:
            text = line.get("text", "")

            # ── 명시적 섹션 헤더 감지 ──────────────────────────────────
            # _is_section_header_line()이 True인 짧은 라벨 줄만 헤더로 처리.
            # 긴 내용 줄에 키워드가 있어도 섹션이 바뀌지 않는다.
            detected_section = self._detect_explicit_section_label(text)

            if detected_section:
                active_section = detected_section
                active_start_y = line.get("center_y", 0)
                ingredient_gap_count = 0

                value = self._remove_section_label_by_type(
                    text=text,
                    section_type=detected_section
                )

                if value:
                    copied = dict(line)
                    copied["text"] = value
                    copied["section_reason"] = f"{detected_section} label value"
                    sections[detected_section].append(copied)

                continue

            # ── 묵시적 섹션 감지 ───────────────────────────────────────
            implicit_section = self._detect_implicit_section(line)

            if implicit_section:
                copied = dict(line)
                copied["section_reason"] = f"implicit {implicit_section}"
                sections[implicit_section].append(copied)

                if implicit_section in ["ingredients", "usage", "cautions"]:
                    active_section = implicit_section
                    active_start_y = line.get("center_y", 0)
                    ingredient_gap_count = 0

                continue

            # ── 활성 섹션 이어가기 ─────────────────────────────────────
            if active_section:
                if active_section == "ingredients":
                    # ingredients는 완화된 조건 + gap 허용
                    if self._should_continue_ingredients(line, ingredient_gap_count):
                        if self._looks_like_ingredient_structure(text):
                            ingredient_gap_count = 0
                        else:
                            ingredient_gap_count += 1

                        copied = dict(line)
                        copied["section_reason"] = "continued from ingredients"
                        sections["ingredients"].append(copied)
                        continue
                    else:
                        active_section = None
                        active_start_y = None
                        ingredient_gap_count = 0

                elif self._should_continue_active_section(
                    line=line,
                    active_section=active_section,
                    active_start_y=active_start_y
                ):
                    copied = dict(line)
                    copied["section_reason"] = f"continued from {active_section}"
                    sections[active_section].append(copied)
                    continue

                else:
                    active_section = None
                    active_start_y = None
                    ingredient_gap_count = 0

        return sections

    def _is_section_header_line(self, text: str) -> bool:
        """
        주어진 줄이 섹션 헤더 줄인지 판단한다.

        헤더로 인정하는 조건 (하나라도 충족하면 헤더):
        1. 텍스트 길이가 header_max_length(25자) 이하
        2. "라벨: 값" 형태 — 콜론 앞 라벨이 15자 이하
           예: "전성분: 정제수, 글리세린, ..." → 콜론 앞 "전성분"은 5자
        3. URL/QR 관련 텍스트 — 길이 제한 없이 허용

        False이면 _detect_explicit_section_label()에서
        키워드가 있어도 섹션 전환을 하지 않는다.
        """

        if not text:
            return False

        text = self._clean_text(text)

        # URL은 길이 관계없이 허용
        if self._looks_like_url(text):
            return True

        # 짧은 줄은 헤더 가능성 높음
        if len(text) <= self.header_max_length:
            return True

        # "라벨: 값" 형태 — 콜론 앞이 짧으면 헤더로 인정
        colon_match = re.match(r"^(.{1,20})[：:]\s*(.+)", text)
        if colon_match:
            label_part = colon_match.group(1).strip()
            if len(label_part) <= 15:
                return True

        return False

    def _detect_explicit_section_label(self, text: str) -> Optional[str]:
        """
        섹션 헤더 라벨을 탐지한다.

        핵심 변경:
        - _is_section_header_line()으로 헤더 줄 여부를 먼저 확인한다.
        - 헤더 줄이 아니면 키워드가 있어도 섹션 전환하지 않는다.
        - 탐지 순서: cautions → ingredients → usage → capacity → qr → product_name
          (복합 표현이 단일 표현보다 먼저 매칭되도록 키워드 리스트도 정렬됨)
        """

        if not text:
            return None

        # 핵심: 헤더 줄이 아니면 무시
        if not self._is_section_header_line(text):
            return None

        compact = self._compact(text)

        # cautions 우선 (사용상주의사항이 usage보다 먼저)
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

    def _detect_implicit_section(
        self,
        line: Dict[str, Any]
    ) -> Optional[str]:
        """
        [수정] 분류 우선순위 재정렬

        기존: URL → capacity → ingredients → cautions → usage 순서
              → 회사정보/제조정보/광고 카피가 ingredients나 usage로 들어감

        수정: 새 우선순위
              1) URL
              2) 메타 정보 (회사/주소/전화/제조정보/날짜) → others
              3) 광고 카피 → others
              4) capacity
              5) ingredients (성분 구조)
              6) cautions (명령형 금지 어미)
              7) usage (명령형 지시 어미)
        """

        text = line.get("text", "")

        if not text:
            return None

        # 1) URL
        if self._looks_like_url(text):
            return "qr_url"

        # [완화] 강한 주의사항 신호는 광고/메타 검사보다 우선
        # 한 줄에 "함유" (광고 단서) 와 "보관할 것/상담할것/자제할것" (주의 단서) 가
        # 함께 있는 경우, 광고로 흡수되지 않도록 cautions를 먼저 시도한다.
        if self._has_strong_caution_signal(text):
            return "cautions"

        # 2) [신규] 메타 정보를 일찍 분리해서 others로
        if self._is_manufacturer_or_meta_text(text):
            return "others"

        # 3) [신규] 광고 카피 → effects(효능 단서 있을 때) 또는 others
        if self._looks_like_advertising_copy(text):
            if self._contains_keyword(
                self._compact(text), self.effects_content_keywords
            ):
                return "effects"
            return "others"

        # 4) 용량
        if self._looks_like_capacity_text(text):
            return "capacity"

        # 5) 성분 구조 (쉼표 나열 + 한글 성분 어미)
        if self._looks_like_ingredient_list_text(text):
            return "ingredients"

        # 6) 주의사항 (명령형 금지 어미)
        if self._looks_like_caution_sentence(text):
            return "cautions"

        # 7) 사용방법 (명령형 지시 어미)
        if self._looks_like_usage_sentence(text):
            return "usage"

        return None

    def _has_strong_caution_signal(self, text: str) -> bool:
        """
        강한 주의사항 신호 판정.

        목적: "+활성-TECA'30,000 ppm 함유 ... 상담할것 자제할것 ..." 처럼
        한 줄에 광고성 단어(함유)와 주의사항 명령형이 섞여 있는 경우,
        광고/메타 분류로 흡수되지 않도록 cautions로 직접 분류한다.

        조건 (하나라도 충족):
        - CAUTION_IMPERATIVE_PATTERN 매칭 2회 이상
        - CAUTION_IMPERATIVE_PATTERN 1회 + 번호 매김 패턴(1./2./3./4. 또는 가)/나)/다))
        - CAUTION_IMPERATIVE_PATTERN 1회 + 명확한 주의 단어
          (주의사항/자제/보관할/피해서/어린이/상처)

        주의: 여기서 의도적으로 _is_manufacturer_or_meta_text() 사전 차단을 하지 않는다.
              "전문의" 가 "문의" 키워드 부분일치로 메타로 잘못 분류되어
              caution이 잘리는 회귀가 있었기 때문.
        """

        if not text:
            return False

        # 광고로만 보이는 매우 짧은 텍스트는 제외
        if len(text.strip()) < 10:
            return False

        # 전화번호/회사 접미사가 명확하면 정말 메타 (caution 신호보다 우선)
        if self.PHONE_PATTERN.search(text):
            return False
        if self.COMPANY_SUFFIX_PATTERN.search(text):
            return False
        if self.MANUFACTURE_META_PATTERN.search(text):
            return False

        imperative_matches = self.CAUTION_IMPERATIVE_PATTERN.findall(text)
        imperative_count = len(imperative_matches)

        if imperative_count >= 2:
            return True

        if imperative_count >= 1:
            # 번호 매김 ("1. 2. 3. 4." 또는 "가) 나) 다)")
            if re.search(r"\b\d+\s*[.\)]\s*[가-힣]", text):
                return True
            if re.search(r"[가-힣]\s*\)\s*[가-힣]", text):
                return True

            # 명확한 주의 단어
            compact = self._compact(text)
            caution_words = ["주의사항", "자제", "보관할", "피해서", "어린이", "상처"]
            if self._contains_keyword(compact, caution_words):
                return True

        return False

    def _should_continue_ingredients(
        self,
        line: Dict[str, Any],
        gap_count: int
    ) -> bool:
        """
        ingredients 섹션을 계속 이어갈지 판단한다.

        완화된 조건:
        - 기존: 쉼표 2개 이상 + 토큰 3개 이상이어야 계속
        - 수정: 쉼표 1개 이상 OR 토큰 2개 이상이면 계속
        - ingredient_gap_tolerance 이내의 비성분 라인은 통과 허용

        덕분에 성분표가 여러 줄로 나뉘어도 끊기지 않는다:
            정제수, 글리세린,           ← ingredients 시작
            부틸렌글라이콜, 판테놀,     ← 계속 (쉼표 있음)
            나이아신아마이드            ← 계속 (gap 허용)

        종료 조건:
        - 다른 섹션의 명시적 헤더가 나옴
        - 사용방법/주의사항 문장이 나옴
        - 제조/메타 텍스트가 나옴
        - URL이 나옴
        - 비성분 라인이 ingredient_gap_tolerance번 연속
        """

        text = line.get("text", "")

        if not text:
            return gap_count < self.ingredient_gap_tolerance

        detected = self._detect_explicit_section_label(text)
        if detected and detected != "ingredients":
            return False

        if self._looks_like_usage_sentence(text):
            return False

        if self._looks_like_caution_sentence(text):
            return False

        if self._is_manufacturer_or_meta_text(text):
            return False

        if self._looks_like_url(text):
            return False

        if self._looks_like_ingredient_structure(text):
            return True

        if self._count_ingredient_separators(text) >= 1:
            return True

        tokens = self._split_potential_ingredient_tokens(text)
        if len(tokens) >= 2:
            return True

        if gap_count < self.ingredient_gap_tolerance:
            return True

        return False

    def _should_continue_active_section(
        self,
        line: Dict[str, Any],
        active_section: str,
        active_start_y: Optional[float]
    ) -> bool:
        text = line.get("text", "")

        if not text:
            return False

        detected = self._detect_explicit_section_label(text)

        if detected and detected != active_section:
            return False

        if active_section != "cautions" and self._is_manufacturer_or_meta_text(text):
            return False

        if active_section == "usage":
            if self._looks_like_ingredient_list_text(text):
                return False

            if self._looks_like_caution_sentence(text):
                return False

            return self._is_valid_sentence_text(text)

        if active_section == "cautions":
            if self._looks_like_ingredient_list_text(text):
                return False

            if self._looks_like_url(text):
                return False

            return self._is_valid_sentence_text(text, allow_meta=True)

        if active_section == "capacity":
            return self._looks_like_capacity_text(text)

        if active_section == "effects":
            if self._looks_like_ingredient_list_text(text):
                return False

            if self._looks_like_url(text):
                return False

            if self._looks_like_capacity_text(text):
                return False

            return self._is_valid_sentence_text(text)

        if active_section == "qr_url":
            return self._looks_like_url(text)

        if active_section == "product_name":
            return False

        return False

    # =========================================================
    # 4. 누락 구간 보완
    # =========================================================

    def _fill_missing_sections_by_global_guess(
        self,
        sections: Dict[str, List[Dict[str, Any]]],
        lines: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not sections.get("product_name"):
            product_line = self._guess_product_name_line(lines)

            if product_line:
                copied = dict(product_line)
                copied["section_reason"] = "guessed product name from top lines"
                sections["product_name"].append(copied)

        if not sections.get("capacity"):
            capacity_line = self._guess_capacity_line(lines)

            if capacity_line:
                copied = dict(capacity_line)
                copied["section_reason"] = "guessed capacity pattern"
                sections["capacity"].append(copied)

        if not sections.get("ingredients"):
            ingredient_lines = self._guess_ingredient_lines(lines)

            for line in ingredient_lines:
                copied = dict(line)
                copied["section_reason"] = "guessed ingredient block by text structure"
                sections["ingredients"].append(copied)

        if not sections.get("qr_url"):
            for line in lines:
                if self._looks_like_url(line.get("text", "")):
                    copied = dict(line)
                    copied["section_reason"] = "guessed qr/url"
                    sections["qr_url"].append(copied)

        return sections

    def _fill_others_section(
        self,
        sections: Dict[str, List[Dict[str, Any]]],
        lines: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        used_keys = set()

        for section_type, section_lines in sections.items():
            if section_type == "others":
                continue

            for line in section_lines:
                used_keys.add(self._line_key(line))

        for line in lines:
            key = self._line_key(line)

            if key in used_keys:
                continue

            copied = dict(line)
            copied["section_reason"] = "unclassified other text"
            sections["others"].append(copied)

        return sections

    def _guess_product_name_line(
        self,
        lines: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not lines:
            return None

        top_lines = lines[: min(len(lines), 10)]
        candidates = []

        for line in top_lines:
            text = line.get("text", "")

            if not self._is_possible_product_name(text):
                continue

            score = self._score_product_name_line(
                line=line,
                all_lines=lines
            )

            candidates.append(
                {
                    "line": line,
                    "score": score
                }
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return candidates[0]["line"]

    def _score_product_name_line(
        self,
        line: Dict[str, Any],
        all_lines: List[Dict[str, Any]]
    ) -> float:
        text = line.get("text", "")
        y = line.get("center_y", 0)

        max_y = max(
            [item.get("center_y", 0) for item in all_lines],
            default=1
        )

        y_ratio = y / max(max_y, 1)

        score = 1.0

        score += max(0.0, 1.0 - y_ratio)

        if 3 <= len(text) <= 45:
            score += 0.4

        if re.search(r"[A-Za-z]", text):
            score += 0.2

        if re.search(r"[가-힣]", text):
            score += 0.2

        if self._is_manufacturer_or_meta_text(text):
            score -= 1.3

        if self._looks_like_capacity_text(text):
            score -= 1.2

        if self._looks_like_ingredient_list_text(text):
            score -= 1.8

        if self._looks_like_usage_sentence(text):
            score -= 1.4

        if self._looks_like_caution_sentence(text):
            score -= 1.4

        if self._looks_like_url(text):
            score -= 1.2

        return score

    def _guess_capacity_line(
        self,
        lines: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        candidates = []

        for line in lines:
            text = line.get("text", "")

            if not self._looks_like_capacity_text(text):
                continue

            score = 1.0

            if self._contains_keyword(self._compact(text), self.capacity_keywords):
                score += 0.4

            if self._is_manufacturer_or_meta_text(text):
                score -= 1.0

            candidates.append(
                {
                    "line": line,
                    "score": score
                }
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return candidates[0]["line"]

    def _guess_ingredient_lines(
        self,
        lines: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        성분표 구간을 전역적으로 추측한다.

        기존 방식의 문제:
        - 가장 점수 높은 1개 라인을 기준으로 위아래 확장
        - 성분표 레이아웃이 다르거나 줄바꿈이 많으면 한쪽만 잡힘

        개선된 방식:
        1. 전체 라인에서 성분 후보 라인 모두 수집 (양수 점수인 것)
        2. 인접한 후보들을 gap_tolerance 기준으로 클러스터로 묶음
        3. 가장 점수 높은 클러스터를 반환
        → 레이아웃, 줄바꿈 수, 2단 컬럼에 무관하게 동작
        """

        if not lines:
            return []

        # 1단계: 전체에서 성분 후보 수집
        candidate_indices = []

        for index, line in enumerate(lines):
            text = line.get("text", "")

            if not text:
                continue

            score = self._score_ingredient_line(text)

            if score > 0:
                candidate_indices.append(
                    {
                        "index": index,
                        "score": score
                    }
                )

        if not candidate_indices:
            return []

        # 2단계: 인접 후보를 클러스터로 묶기
        clusters = self._cluster_ingredient_candidates(
            candidate_indices=candidate_indices,
            all_lines=lines
        )

        if not clusters:
            return []

        # 3단계: 최고 클러스터 선택
        best_cluster = self._select_best_ingredient_cluster(clusters, lines)

        return best_cluster if best_cluster else []

    def _cluster_ingredient_candidates(
        self,
        candidate_indices: List[Dict[str, Any]],
        all_lines: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        후보 라인 인덱스들을 연속 구간(cluster)으로 묶는다.

        gap_tolerance 이내의 비후보 라인은 클러스터에 포함한다.
        """

        if not candidate_indices:
            return []

        sorted_candidates = sorted(candidate_indices, key=lambda c: c["index"])

        clusters = []
        current_indices = [sorted_candidates[0]["index"]]

        for i in range(1, len(sorted_candidates)):
            prev_idx = sorted_candidates[i - 1]["index"]
            curr_idx = sorted_candidates[i]["index"]
            gap = curr_idx - prev_idx - 1

            if gap <= self.ingredient_gap_tolerance:
                # 사이 라인(비후보)도 포함
                for j in range(prev_idx + 1, curr_idx):
                    current_indices.append(j)
                current_indices.append(curr_idx)
            else:
                clusters.append(current_indices)
                current_indices = [curr_idx]

        clusters.append(current_indices)

        # 인덱스 → 실제 라인 변환
        result = []
        for cluster_indices in clusters:
            cluster_lines = []

            for idx in sorted(set(cluster_indices)):
                if 0 <= idx < len(all_lines):
                    cluster_lines.append(all_lines[idx])

            if cluster_lines:
                result.append(cluster_lines)

        return result

    def _select_best_ingredient_cluster(
        self,
        clusters: List[List[Dict[str, Any]]],
        all_lines: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        클러스터 중 성분표일 가능성이 가장 높은 것을 선택한다.

        점수 기준:
        - 성분 구조 라인 수 (많을수록 좋음)
        - 구분자 총합 (많을수록 좋음)
        - 총 텍스트 길이 (길수록 좋음)
        - 제조/사용법/주의사항 비율 (낮을수록 좋음)
        """

        scored = []

        for cluster in clusters:
            if not cluster:
                continue

            score = 0.0
            ingredient_like_count = 0
            separator_total = 0
            total_text_len = 0

            for line in cluster:
                text = line.get("text", "")

                if not text:
                    continue

                total_text_len += len(text)
                separator_total += self._count_ingredient_separators(text)

                if self._looks_like_ingredient_structure(text):
                    ingredient_like_count += 1
                    score += 2.0

                if self._is_manufacturer_or_meta_text(text):
                    score -= 1.5

                if self._looks_like_usage_sentence(text):
                    score -= 1.0

                if self._looks_like_caution_sentence(text):
                    score -= 1.0

            score += min(separator_total * 0.3, 3.0)
            score += min(total_text_len * 0.005, 2.0)

            # 성분 구조 라인 1개 이상이어야 후보
            if ingredient_like_count == 0:
                continue

            scored.append(
                {
                    "cluster": cluster,
                    "score": score
                }
            )

        if not scored:
            return []

        scored.sort(key=lambda item: item["score"], reverse=True)

        return scored[0]["cluster"]

    def _score_ingredient_line(self, text: str) -> float:
        """단일 라인의 성분 후보 점수. 양수이면 후보."""

        text = self._clean_text(text)

        if self._is_manufacturer_or_meta_text(text):
            return -1.0

        if self._looks_like_usage_sentence(text):
            return -1.0

        if self._looks_like_caution_sentence(text):
            return -1.0

        if self._looks_like_capacity_text(text):
            return -1.0

        if self._looks_like_url(text):
            return -1.0

        score = 0.0

        separator_count = self._count_ingredient_separators(text)
        score += min(separator_count * 0.5, 3.0)

        tokens = self._split_potential_ingredient_tokens(text)

        if len(tokens) >= 4:
            score += 1.0

        if self._token_sequence_is_ingredient_like(tokens):
            score += 1.5

        if len(text) >= 30:
            score += 0.5

        return score

    def _should_include_neighbor_as_ingredient(
        self,
        text: str
    ) -> bool:
        if not text:
            return False

        explicit = self._detect_explicit_section_label(text)

        if explicit in ["usage", "cautions", "capacity", "qr_url", "product_name"]:
            return False

        if self._is_manufacturer_or_meta_text(text):
            return False

        if self._looks_like_usage_sentence(text):
            return False

        if self._looks_like_caution_sentence(text):
            return False

        if self._looks_like_capacity_text(text):
            return False

        if self._looks_like_url(text):
            return False

        return self._looks_like_ingredient_structure(text)

    # =========================================================
    # 5. 결과 생성
    # =========================================================

    def _build_section_text_map(
        self,
        sections: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[str]]:
        section_text_map = {
            section_type: []
            for section_type in self.section_types
        }

        for section_type in self.section_types:
            lines = sections.get(section_type, []) or []

            sorted_lines = sorted(
                lines,
                key=lambda item: (
                    item.get("center_y", 0),
                    item.get("x_min", 0)
                )
            )

            for line in sorted_lines:
                text = self._clean_text(line.get("text", ""))

                if not text:
                    continue

                if text not in section_text_map[section_type]:
                    section_text_map[section_type].append(text)

        return section_text_map

    def _build_merged_text_by_section(
        self,
        section_text_map: Dict[str, List[str]]
    ) -> Dict[str, str]:
        merged = {
            section_type: ""
            for section_type in self.section_types
        }

        for section_type, texts in section_text_map.items():
            if not texts:
                merged[section_type] = ""
                continue

            if section_type == "ingredients":
                merged[section_type] = " ".join(texts)

            elif section_type in ["usage", "cautions", "effects", "others"]:
                merged[section_type] = "\n".join(texts)

            else:
                merged[section_type] = " ".join(texts)

        return merged

    def _build_detected_section_metadata(
        self,
        sections: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        results = []

        for section_type in self.section_types:
            lines = sections.get(section_type, []) or []

            if not lines:
                continue

            x1 = min(line.get("x_min", 0) for line in lines)
            y1 = min(line.get("y_min", 0) for line in lines)
            x2 = max(line.get("x_max", 0) for line in lines)
            y2 = max(line.get("y_max", 0) for line in lines)

            reasons = []

            for line in lines:
                reason = line.get("section_reason", "")

                if reason and reason not in reasons:
                    reasons.append(reason)

            results.append(
                {
                    "section_type": section_type,
                    "text_count": len(lines),
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "reasons": reasons,
                    "preview": " / ".join(
                        [
                            line.get("text", "")
                            for line in lines[:3]
                        ]
                    )
                }
            )

        return results

    # =========================================================
    # 6. 성분 구조 판별: 특정 성분명 힌트 미사용
    # =========================================================

    def _looks_like_ingredient_list_text(
        self,
        text: str
    ) -> bool:
        if not text:
            return False

        text = self._clean_text(text)

        if self._is_noise_line(text):
            return False

        if self._is_manufacturer_or_meta_text(text):
            return False

        if self._ingredient_sentence_penalty_score(text) >= 2:
            return False

        if self._looks_like_url(text):
            return False

        if self._looks_like_capacity_text(text):
            return False

        return self._looks_like_ingredient_structure(text)

    def _looks_like_ingredient_structure(
        self,
        text: str
    ) -> bool:
        """
        특정 성분명 없이 성분 구간인지 판단한다.

        [수정] 임계값 완화 + 한글 성분 어미 단독 신호 추가:
        - 기존: 구분자 2개 이상 + 토큰 3개 이상
        - 1차 수정: 구분자 1개 이상 + 토큰 2개 이상
        - [신규] 구분자가 0개여도 KOREAN_INGREDIENT_SUFFIX_PATTERN이
          2개 이상 매칭되면 성분으로 인정
          → OCR이 쉼표를 다 놓쳐서 한 줄로 붙어버린 경우 대응
          예: "소등라우레스션페이트소동클로라이드"
              → "페이트", "클로라이드" 2개 매칭 → ingredients
        """

        if not text:
            return False

        text = self._clean_text(text)

        separator_count = self._count_ingredient_separators(text)
        tokens = self._split_potential_ingredient_tokens(text)

        # 강한 신호 (기존과 동일)
        if separator_count >= 2 and len(tokens) >= 3:
            return True

        # 완화된 신호 (1차 수정)
        if separator_count >= 1 and len(tokens) >= 2:
            return True

        if self._token_sequence_is_ingredient_like(tokens):
            return True

        # [신규] 구분자 없어도 한글 성분 어미가 2개 이상이면 성분
        #   OCR이 쉼표 다 놓친 경우 (어두운 라벨, 작은 글씨에서 흔함)
        suffix_matches = self.KOREAN_INGREDIENT_SUFFIX_PATTERN.findall(text)
        if len(suffix_matches) >= 2:
            # 단, 광고 카피나 메타가 아닐 때만
            if not self._looks_like_advertising_copy(text):
                if not self._is_manufacturer_or_meta_text(text):
                    return True

        return False

    def _count_ingredient_separators(self, text: str) -> int:
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

    def _split_potential_ingredient_tokens(
        self,
        text: str
    ) -> List[str]:
        if not text:
            return []

        tokens = re.split(r"[,，、;；·ㆍ/\n]+", text)

        cleaned_tokens = []

        for token in tokens:
            token = self._clean_text(token)
            token = token.strip(" ,.;:：/·ㆍ-()[]{}")

            if not token:
                continue

            if len(token) < 2:
                continue

            if len(token) > 50:
                continue

            if re.fullmatch(r"\d+", token):
                continue

            if re.fullmatch(r"[a-zA-Z]{1,3}", token):
                continue

            if self._looks_like_capacity_text(token):
                continue

            if self._looks_like_url(token):
                continue

            cleaned_tokens.append(token)

        return cleaned_tokens

    def _token_sequence_is_ingredient_like(
        self,
        tokens: List[str]
    ) -> bool:
        if not tokens:
            return False

        if len(tokens) < 4:
            return False

        valid_count = 0

        for token in tokens:
            if self._token_looks_like_ingredient_candidate(token):
                valid_count += 1

        ratio = valid_count / max(len(tokens), 1)

        return valid_count >= 4 and ratio >= 0.65

    def _token_looks_like_ingredient_candidate(
        self,
        token: str
    ) -> bool:
        if not token:
            return False

        token = self._clean_text(token)

        if len(token) < 2:
            return False

        if len(token) > 50:
            return False

        if re.fullmatch(r"\d+", token):
            return False

        if re.fullmatch(r"[a-zA-Z]{1,3}", token):
            return False

        if self._looks_like_url(token):
            return False

        if self._looks_like_capacity_text(token):
            return False

        if self._looks_like_usage_sentence(token):
            return False

        if self._looks_like_caution_sentence(token):
            return False

        if self._is_manufacturer_or_meta_text(token):
            return False

        if not re.search(r"[가-힣A-Za-z]", token):
            return False

        return True

    # =========================================================
    # 7. label 제거
    # =========================================================

    def _remove_section_label_by_type(
        self,
        text: str,
        section_type: str
    ) -> str:
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
            return self._remove_section_label(text, self.qr_keywords)

        return text

    def _remove_section_label(
        self,
        text: str,
        keywords: List[str]
    ) -> str:
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

    # =========================================================
    # 8. 기타 판별 함수
    # =========================================================

    def _is_possible_product_name(
        self,
        text: str
    ) -> bool:
        if not text:
            return False

        text = self._clean_text(text)
        compact = self._compact(text)

        if not compact:
            return False

        if len(text) < 2:
            return False

        if len(text) > 80:
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

        if self._ingredient_sentence_penalty_score(text) >= 2:
            return False

        if not re.search(r"[가-힣A-Za-z]", text):
            return False

        return True

    def _looks_like_capacity_text(
        self,
        text: str
    ) -> bool:
        if not text:
            return False

        compact = self._compact(text)

        forbidden = [
            "제조번호", "제조일자", "사용기한", "유통기한",
            "exp", "mfg", "별도표", "lot", "batch",
            "전화", "고객센터", "소비자상담"
        ]

        if self._contains_keyword(compact, forbidden):
            return False

        return bool(
            re.search(
                r"\d+(?:\.\d+)?\s?(?:ml|mL|ML|g|G|kg|KG|mg|MG|oz|OZ|fl\.?\s?oz|매|pcs|ea|개|ea|EA)",
                text,
                flags=re.IGNORECASE
            )
        )

    def _looks_like_usage_sentence(
        self,
        text: str
    ) -> bool:
        """
        [수정] 명령형 지시 어미 패턴 기반 (특정 동사 의존 X)

        기존: "바르", "발라", "도포", "마사지", "세안" 등 특정 동사 의존
              → 다양한 화장품(샴푸, 바디, 헤어, 클렌저)의 다른 동사 누락

        수정: USAGE_IMPERATIVE_PATTERN 정규식 추가
              - "~하세요", "~십시오", "~합니다", "~냅니다" 등 명령/평서 종결
              - 광고 카피/주의사항/성분 구조 배제 조건 추가
              - 짧은 깨진 텍스트 배제 (최소 5자 + 한글 3자 이상)
        """

        if not text:
            return False

        text_strip = text.strip()

        # [신규] 너무 짧거나 한글이 너무 적으면 OCR 깨짐일 가능성 → 분류 안 함
        if len(text_strip) < 5:
            return False

        korean_chars = re.findall(r"[가-힣]", text_strip)
        if len(korean_chars) < 3:
            return False

        # 광고/메타/성분과 구분되도록 먼저 배제
        if self._looks_like_advertising_copy(text):
            return False

        if self._looks_like_ingredient_structure(text):
            return False

        # 주의사항(금지 어미)이 더 강한 우선순위
        if self.CAUTION_IMPERATIVE_PATTERN.search(text):
            return False

        # 1) 일반화된 명령형/평서형 종결 어미 패턴
        if self.USAGE_IMPERATIVE_PATTERN.search(text):
            return True

        # 2) 기존 시그널 키워드 (보조 — 화장품 라벨 일반 표현)
        #    특정 동사가 아니라 "사용방법"이라는 라벨 자체 등을 인식
        compact = self._compact(text)

        usage_signals = [
            "사용방법", "사용법", "사용순서",
            "apply", "massage", "rinse", "wash",
            "directions", "directions for use"
        ]

        if self._contains_keyword(compact, usage_signals):
            if not self._looks_like_ingredient_structure(text):
                return True

        return False

    def _looks_like_caution_sentence(
        self,
        text: str
    ) -> bool:
        """
        [수정] 명령형/금지 어미 패턴 추가 (특정 단어 의존 X)

        기존: 명확한 단어(피부이상, 자극, 직사광선 등)만 매칭
              → "보관할 것", "찢지 말 것", "자제할 것" 등 일반 라벨 문구 누락

        수정: CAUTION_IMPERATIVE_PATTERN 정규식 추가
              - "~할 것", "~말 것", "~마세요", "~금지", "~피하세요" 같은
                금지/명령형 어미를 일반화된 패턴으로 인식
              - 키워드 매칭과 패턴 매칭 OR 결합
        """

        if not text:
            return False

        compact = self._compact(text)

        # 1) 일반화된 명령형/금지 어미 패턴 (한국어 화장품 라벨 보편적)
        if self.CAUTION_IMPERATIVE_PATTERN.search(text):
            # 단, 광고 카피나 메타가 아닐 때만
            if not self._looks_like_advertising_copy(text):
                if not self._is_manufacturer_or_meta_text(text, suppress_self=True):
                    return True

        # 2) 기존 시그널 키워드 (피부이상, 직사광선 등 화장품 일반 표현)
        caution_signals = [
            "주의사항", "사용상주의",
            "피부이상", "붉은반점",
            "부어오름", "가려움", "자극",
            "상처", "습진", "피부염",
            "직사광선", "어린이", "화기", "경고",
            "warning", "caution", "precaution"
        ]

        return self._contains_keyword(compact, caution_signals)

    def _looks_like_advertising_copy(
        self,
        text: str
    ) -> bool:
        """
        [신규] 광고 카피 라인 판별 (특정 단어 의존 X)

        광고 카피의 구조적 특징:
        1) 형용사형/수식형 종결 어미 ("~한", "~된", "~의", "~로", "~함유")
        2) 명령형/평서형 종결 어미 없음 (-세요, -십시오, -니다)
        3) 쉼표 거의 없음 (성분처럼 나열 안 함)
        4) 길이가 어느 정도 있음 (한 단어짜리는 광고 아님)

        예시:
        - "케라틴은 ~ 성분으로"      → "성분으로" (-로 종결) → 광고
        - "마치 살통에 간 듯한 향"   → "~한 향" → 광고
        - "프로비타민 풍부한"        → "풍부한" → 광고
        - "200% 증가된 케라틴"       → "증가된" → 광고
        """

        if not text:
            return False

        text_strip = text.strip()

        # 너무 짧으면 광고가 아닐 가능성 (제품명이거나 단순 라벨)
        if len(text_strip) < 8:
            return False

        # 1) 형용사형/수식형 종결 어미가 있어야 함
        if not self.ADVERTISING_ENDING_PATTERN.search(text_strip):
            return False

        # 2) 명령형/평서형 종결 어미가 있으면 광고 아님 (사용방법/주의사항)
        if self.USAGE_IMPERATIVE_PATTERN.search(text_strip):
            return False

        if self.CAUTION_IMPERATIVE_PATTERN.search(text_strip):
            return False

        # 3) 쉼표가 많으면 성분 나열일 가능성 (광고 아님)
        if text_strip.count(",") >= 2:
            return False

        # 4) 한글 성분 어미 패턴이 여러 개 있으면 성분 (광고 아님)
        ingredient_suffix_count = len(
            self.KOREAN_INGREDIENT_SUFFIX_PATTERN.findall(text_strip)
        )
        if ingredient_suffix_count >= 2:
            return False

        return True

    def _is_valid_sentence_text(
        self,
        text: str,
        allow_meta: bool = False
    ) -> bool:
        if not text:
            return False

        text = self._clean_text(text)

        if len(text) < 2:
            return False

        if self._looks_like_capacity_text(text):
            return False

        if self._looks_like_url(text):
            return False

        if self._looks_like_ingredient_list_text(text):
            return False

        if not allow_meta and self._is_manufacturer_or_meta_text(text):
            return False

        return True

    def _is_manufacturer_or_meta_text(
        self,
        text: str,
        suppress_self: bool = False
    ) -> bool:
        """
        [수정] 회사/주소/제조정보를 패턴 기반으로 강화 (특정 회사명 의존 X)

        기존: 키워드 리스트(manufacturer_keywords) + "서울/경기..." 하드코딩
              → 외국 라벨, 새 회사명, OCR 깨짐에 약함

        수정: 다음 패턴들을 추가
              - COMPANY_SUFFIX_PATTERN: 주식회사, 유한회사, Co.,Ltd, Inc 등
              - ADDRESS_PATTERN: ~로/길/시/구/번지/층/호
              - PHONE_PATTERN: 다양한 전화번호 형식
              - MANUFACTURE_META_PATTERN: EXP/MFG/LOT/사용기한/제조번호
              - DATE_PATTERN: 날짜 형식

        suppress_self: 다른 분류 함수에서 호출 시 재귀 방지
        """

        if not text:
            return False

        # ── 1. 일반화된 패턴 매칭 ──────────────────────────────────────

        # 회사/법인 접미사
        if self.COMPANY_SUFFIX_PATTERN.search(text):
            return True

        # 한국 주소 표지
        if self.ADDRESS_PATTERN.search(text):
            return True

        # 전화번호
        if self.PHONE_PATTERN.search(text):
            return True

        # 제조 메타 (EXP/MFG/LOT/사용기한 등)
        if self.MANUFACTURE_META_PATTERN.search(text):
            return True

        # 날짜
        if self.DATE_PATTERN.search(text):
            return True

        # ── 2. 기존 키워드 리스트 (보조) ──────────────────────────────

        compact = self._compact(text)

        if self._contains_keyword(compact, self.manufacturer_keywords):
            return True

        return False

    def _looks_like_phone_number(
        self,
        text: str
    ) -> bool:
        """[수정] PHONE_PATTERN으로 일원화"""
        if not text:
            return False

        return bool(self.PHONE_PATTERN.search(text))

    def _looks_like_url(
        self,
        text: str
    ) -> bool:
        if not text:
            return False

        return bool(
            re.search(
                r"https?://|www\.|\.com|\.co\.kr|\.kr|\.net|\.org|\.io|\.ai",
                text,
                flags=re.IGNORECASE
            )
        )

    def _ingredient_sentence_penalty_score(
        self,
        text: str
    ) -> int:
        if not text:
            return 0

        compact = self._compact(text)
        score = 0

        for keyword in self.ingredient_negative_keywords:
            if self._compact(keyword) in compact:
                score += 1

        if re.search(r"(하십시오|하세요|합니다|됩니다|있습니다|하지말것|버리지말것|사용하지)", text):
            score += 2

        if len(text) >= 45 and self._count_ingredient_separators(text) == 0:
            score += 1

        return score

    def _is_noise_line(
        self,
        text: str
    ) -> bool:
        if not text:
            return True

        compact = self._compact(text)

        if not compact:
            return True

        noise_words = [
            "for", "cfor", "the", "and", "from", "with",
            "made", "in",
            "소비자가", "만드는신문", "신문",
            "brand", "nobrand", "no brand"
        ]

        for noise in noise_words:
            if self._compact(noise) == compact:
                return True

        if "소비자가" in text or "신문" in text:
            return True

        if re.fullmatch(r"[a-zA-Z]{1,3}", text):
            return True

        if re.fullmatch(r"[가-힣]{1}", text):
            return True

        if re.fullmatch(r"[-_=~.ㆍ·,，;；:：|/\\]+", text):
            return True

        return False

    # =========================================================
    # 9. 공통 유틸
    # =========================================================

    def _empty_section_line_map(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "product_name": [],
            "capacity": [],
            "ingredients": [],
            "usage": [],
            "cautions": [],
            "effects": [],
            "qr_url": [],
            "others": []
        }

    def _empty_result(
        self,
        reason: str = "",
        raw_text: str = "",
        layout_text: str = "",
        ocr_lines: Optional[List[Dict[str, Any]]] = None,
        ocr_blocks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "mode": "ocr_bbox_section_detection_generalized",
            "reason": reason,
            "section_text_map": {
                "product_name": [],
                "capacity": [],
                "ingredients": [],
                "usage": [],
                "cautions": [],
                "effects": [],
                "qr_url": [],
                "others": []
            },
            "merged_text_by_section": {
                "product_name": "",
                "capacity": "",
                "ingredients": "",
                "usage": "",
                "cautions": "",
                "effects": "",
                "qr_url": "",
                "others": ""
            },
            "detected_sections": [],
            "section_detection_summary": {
                "line_count": len(ocr_lines or []),
                "block_count": len(ocr_blocks or []),
                "detected_section_count": 0,
                "detected_section_types": []
            },
            "raw_text": raw_text,
            "layout_text": layout_text,
            "ocr_lines": ocr_lines or [],
            "ocr_blocks": ocr_blocks or []
        }

    def _clean_text(
        self,
        text: Any
    ) -> str:
        if text is None:
            return ""

        text = str(text)

        replace_map = {
            "\r": " ",
            "\n": " ",
            "\t": " ",
            "，": ",",
            "、": ",",
            "；": ";",
            "：": ":",
            "ㆍ": "·",
            "（": "(",
            "）": ")",
            "［": "[",
            "］": "]",
            "【": "[",
            "】": "]",
            "㎖": "ml",
            "ｍｌ": "ml",
            "ＭＬ": "ml",
            "ｇ": "g",
            "Ｇ": "g",
            "㎎": "mg",
            "㎏": "kg",
            "–": "-",
            "—": "-",
            "−": "-"
        }

        for old, new in replace_map.items():
            text = text.replace(old, new)

        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*,\s*", ", ", text)
        text = re.sub(r"\s*;\s*", "; ", text)
        text = re.sub(r"\s*:\s*", ": ", text)
        text = re.sub(r"\s+([,.;:)\]\}])", r"\1", text)
        text = re.sub(r"([(\[\{])\s+", r"\1", text)
        text = re.sub(r"([,;])(?=[가-힣A-Za-z0-9])", r"\1 ", text)

        return text.strip()

    def _contains_keyword(
        self,
        compact_text: str,
        keywords: List[str]
    ) -> bool:
        for keyword in keywords:
            key = self._compact(keyword)

            if key and key in compact_text:
                return True

        return False

    def _compact(
        self,
        text: Any
    ) -> str:
        return (
            str(text)
            .lower()
            .replace(" ", "")
            .replace("\n", "")
            .replace("\r", "")
            .replace("\t", "")
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

    def _to_float(
        self,
        value: Any,
        default: float = 0.0
    ) -> float:
        try:
            return float(value)

        except Exception:
            return default

    def _line_key(self, line: Dict[str, Any]) -> str:
        text = self._clean_text(line.get("text", ""))
        x = round(self._to_float(line.get("x_min", 0)), 1)
        y = round(self._to_float(line.get("y_min", 0)), 1)

        return f"{text}|{x}|{y}"