import re
import requests

from difflib import SequenceMatcher
from src.utils.config import config


class IngredientAPI:
    """
    Dermalens 성분 API 검증 클래스

    역할:
    1. postprocess.py에서 추출한 ingredients_raw를 입력받는다.
    2. 후보 문자열이 한 줄로 붙어 있어도 줄바꿈/콤마/세미콜론/가운뎃점 기준으로 개별 성분 후보로 분리한다.
    3. 제품명, 용량, 사용방법, 주의사항, QR, 기타 텍스트에는 사용하지 않는다.
    4. 성분 후보에 대해서만 공공데이터포털 화장품 성분 API 검증을 수행한다.
    5. API에서 매칭된 표준 성분명을 ingredients_verified로 반환한다.

    [수정 사항]

    (A) 입력/반환 필드명 postprocess.py 새 구조에 맞게 통일
        - 입력: ingredients_raw (기존 ingredient_candidates, ingredients_before_api 혼재 → 통일)
        - 반환: ingredients_verified 추가 (postprocess.py 최종 JSON 구조와 일치)
        - 반환: ingredients_after_api, ingredients 하위 호환 유지

    (B) _clean_query() — 공백 제거 방지
        기존: [^가-힣a-zA-Z0-9,\\-/] 정규식으로 공백까지 제거됨
              → "Sodium Hyaluronate" → "SodiumHyaluronate" → API 검색 실패
        수정: \\s 허용으로 공백 보존 후 연속 공백만 정리

    (C) _is_invalid_candidate() — 최대 길이 완화
        기존: len > 70 → 복합 성분명도 탈락
        수정: max_candidate_length = 90 으로 완화

    (D) _split_by_space_when_safe() — 영어 토큰 수에 따라 분리 기준 차별화
        기존: 영어 토큰 2개 이상이면 무조건 통째로 유지
              → "Water Glycerin Niacinamide" (3개 성분)도 하나로 보내 API 검색 실패
        수정: 영어 토큰 정확히 2개 → 유지 ("Sodium Hyaluronate" 보존)
              영어 토큰 3개 이상 → 개별 분리

    (E) _generate_query_candidates() — 의미 있는 변형만 생성
        기존: 대시/콤마/슬래시 무조건 제거 변형 생성 → 중복 후보 발생
        수정: 해당 문자가 실제로 있을 때만 변형 생성

    반환 구조:
    {
        "status": "success | partial_success | all_failed | no_ingredient_candidates | no_valid_ingredient_candidates",
        "ingredients_verified": [...],      ← postprocess.py 최종 JSON 필드
        "ingredients_after_api": [...],     ← 하위 호환
        "ingredients": [...],               ← 하위 호환
        "verified_ingredient_names": [...],
        "verified_ingredients": [...],
        "unverified_ingredients": [...],
        "api_success_results": [...],
        "api_failed_results": [...],
        "api_all_results": [...],
        "verified_count": 0,
        "unverified_count": 0,
        "total_checked_count": 0
    }
    """

    def __init__(self):
        self.api_key = config.PUBLIC_DATA_API_KEY
        self.api_url = config.COSMETIC_INGREDIENT_API_URL

        if not self.api_key:
            raise ValueError(".env 파일에 PUBLIC_DATA_API_KEY가 없습니다.")

        if not self.api_url:
            raise ValueError(".env 파일에 COSMETIC_INGREDIENT_API_URL이 없습니다.")

        self.cache = {}

        # OCR 오타가 있을 수 있으므로 너무 높이면 실제 성분도 탈락함
        self.match_threshold = 0.72

        # 이 기준 이상이면 바로 성공 처리
        self.strong_match_threshold = 0.90

        # [수정] 성분명 최대 길이 완화 (70 → 90)
        # 복합 성분명이 70자를 넘는 경우가 있음
        self.max_candidate_length = 90

        # 성분 후보에서 제외할 문장/메타 정보 키워드
        self.invalid_keywords = [
            "사용", "사용법", "사용방법", "사용후", "사용 후",
            "주의", "주의사항", "보관", "보관방법",
            "화장품", "고객", "상담", "제조", "판매", "교환", "반품",
            "제품명", "용량", "중량", "내용량", "원산지",
            "마사지", "도포", "씻어", "세안", "바르", "흡수",
            "www", "http", "https", "사이트", "주소", "홈페이지",
            "소비자", "품질보증", "공정거래", "분쟁해결",
            "EXP", "MFG", "LOT", "Batch",
            "가연성", "화기주의", "전화", "문의",
            "제조번호", "제조일자", "사용기한", "유통기한",
            "별도표", "별도", "표기", "분리수거", "분리배출",
            "확인불가", "확인 불가",
            "qr", "url", "barcode", "바코드"
        ]

        # OCR 오타 보정용
        # 구간 탐지용이 아니라 API 검색어 보정용이다.
        # 최종 성분명은 반드시 API 결과의 표준 성분명을 사용한다.
        self.ocr_hint_map = {
            "소듬": "소듐",
            "이소듬": "다이소듐",
            "다이소듬": "다이소듐",
            "주출물": "추출물",
            "수출물": "추출물",
            "글라이볼": "글라이콜",
            "글라이클": "글라이콜",
            "핵산다이올": "헥산다이올",
            "텍산다이올": "헥산다이올",
            "12헥산다이올": "1,2-헥산다이올",
            "12핵산다이올": "1,2-헥산다이올",
            "12텍산다이올": "1,2-헥산다이올",
            "12-헥산다이올": "1,2-헥산다이올",
            "12-핵산다이올": "1,2-헥산다이올",
            "12-텍산다이올": "1,2-헥산다이올",
            "비즈악스": "비즈왁스",
            "비즈왁쓰": "비즈왁스",
            "포타슷하이드록사이드": "포타슘하이드록사이드",
            "포타숩하이드록사이드": "포타슘하이드록사이드",
            "포타슘 하이드록사이드": "포타슘하이드록사이드",
            "소듐 하이드록사이드": "소듐하이드록사이드",
            "하이알루로내이트": "하이알루로네이트",
            "하이알루로네이트": "하이알루로네이트",
            "나이아신 아마이드": "나이아신아마이드",
            "펜타이드": "펩타이드",
            "트라이펜타이드": "트라이펩타이드",
            "테트라펜타이드": "테트라펩타이드"
        }

    # =========================================================
    # 1. API 검색
    # =========================================================

    def search_ingredient(self, ingredient_name, num_of_rows=20):
        """
        단일 성분명을 공공데이터 API에서 검색한다.
        """

        query = self._clean_query(ingredient_name)

        if not query:
            return {
                "success": False,
                "query": ingredient_name,
                "error": "빈 검색어",
                "items": []
            }

        cache_key = self._compact(query)

        if cache_key in self.cache:
            return self.cache[cache_key]

        params = {
            "serviceKey": self.api_key,
            "pageNo": 1,
            "numOfRows": num_of_rows,
            "type": "json",
            "INGR_KOR_NAME": query
        }

        try:
            response = requests.get(
                self.api_url,
                params=params,
                timeout=10
            )

            if response.status_code != 200:
                result = {
                    "success": False,
                    "query": query,
                    "error": f"HTTP 오류: {response.status_code}",
                    "items": []
                }
                self.cache[cache_key] = result
                return result

            try:
                data = response.json()

            except ValueError:
                result = {
                    "success": False,
                    "query": query,
                    "error": "JSON 응답 변환 실패",
                    "items": []
                }
                self.cache[cache_key] = result
                return result

            items = self._extract_items(data)

            result = {
                "success": True,
                "query": query,
                "items": items
            }

            self.cache[cache_key] = result
            return result

        except requests.exceptions.RequestException as error:
            result = {
                "success": False,
                "query": query,
                "error": str(error),
                "items": []
            }
            self.cache[cache_key] = result
            return result

    # =========================================================
    # 2. 여러 성분 후보 검증
    # =========================================================

    def match_ingredients(self, candidate_list):
        """
        ingredients_raw(postprocess.py 출력)를 API로 검증한다.

        입력 예:
        [
            "정제수, 글리세린, 스테아릭애씨드",
            "피이지-8\\n프로필렌글라이콜"
        ]

        처리:
        - 줄바꿈 분리
        - 콤마 분리
        - 세미콜론/가운뎃점/파이프 분리
        - 공백 연결 후보 보조 분리
        - API 검증

        반환:
        - ingredients_verified: 검증 성공한 표준 성분명 (postprocess.py 최종 JSON 필드)
        - ingredients_after_api: 동일 (하위 호환)
        - unverified_ingredients: 검증 실패 후보
        - api_all_results: 전체 검증 상세
        """

        success_results = []
        failed_results = []
        all_results = []
        verified_ingredient_names = []
        verified_ingredients = []
        unverified_ingredients = []
        seen_query_keys = set()
        seen_verified_names = set()

        if not candidate_list:
            return self._build_match_result(
                status="no_ingredient_candidates",
                verified_ingredient_names=[],
                verified_ingredients=[],
                unverified_ingredients=[],
                success_results=[],
                failed_results=[],
                all_results=[]
            )

        expanded_candidates = self._expand_candidate_list(candidate_list)

        if not expanded_candidates:
            return self._build_match_result(
                status="no_valid_ingredient_candidates",
                verified_ingredient_names=[],
                verified_ingredients=[],
                unverified_ingredients=[],
                success_results=[],
                failed_results=[],
                all_results=[]
            )

        for candidate in expanded_candidates:
            original_candidate = str(candidate).strip()

            if not original_candidate or original_candidate == "확인 불가":
                continue

            cleaned = self._clean_query(original_candidate)

            if not cleaned:
                failed = self._build_failed_result(
                    ocr_name=original_candidate,
                    query_name="",
                    reason="정리 후 빈 성분 후보"
                )

                failed_results.append(failed)
                all_results.append(failed)
                unverified_ingredients.append(
                    self._build_unverified_item(failed)
                )
                continue

            query_key = self._compact(cleaned)

            if query_key in seen_query_keys:
                continue

            seen_query_keys.add(query_key)

            matched = self.match_ingredient(
                candidate_name=cleaned,
                original_name=original_candidate
            )

            all_results.append(matched)

            if matched.get("matched"):
                success_results.append(matched)

                matched_name = (
                    matched.get("matched_name_kr")
                    or matched.get("query_name")
                    or cleaned
                )

                matched_name_key = self._compact(matched_name)

                if matched_name and matched_name_key not in seen_verified_names:
                    seen_verified_names.add(matched_name_key)
                    verified_ingredient_names.append(matched_name)

                    verified_ingredients.append(
                        {
                            "ocr_name": matched.get("ocr_name"),
                            "query_name": matched.get("query_name"),
                            "matched_name_kr": matched.get("matched_name_kr"),
                            "matched_name_en": matched.get("matched_name_en"),
                            "cas_no": matched.get("cas_no"),
                            "definition": matched.get("definition"),
                            "similarity": matched.get("similarity"),
                            "source": matched.get("source")
                        }
                    )

            else:
                failed_results.append(matched)
                unverified_ingredients.append(
                    self._build_unverified_item(matched)
                )

        if not success_results and failed_results:
            status = "all_failed"

        elif success_results and failed_results:
            status = "partial_success"

        elif success_results and not failed_results:
            status = "success"

        else:
            status = "no_result"

        return self._build_match_result(
            status=status,
            verified_ingredient_names=verified_ingredient_names,
            verified_ingredients=verified_ingredients,
            unverified_ingredients=unverified_ingredients,
            success_results=success_results,
            failed_results=failed_results,
            all_results=all_results
        )

    def _build_match_result(
        self,
        status,
        verified_ingredient_names,
        verified_ingredients,
        unverified_ingredients,
        success_results,
        failed_results,
        all_results
    ):
        """
        [수정] ingredients_verified 필드 추가
        postprocess.py 최종 JSON의 "ingredients_verified" 필드와 일치.
        main.py에서 이 필드를 postprocess 결과에 붙인다.
        """

        return {
            "status": status,

            # [수정] postprocess.py 최종 JSON 필드명과 일치
            "ingredients_verified": verified_ingredient_names,

            # 하위 호환 필드
            "ingredients_after_api": verified_ingredient_names,
            "ingredients": verified_ingredient_names,
            "verified_ingredient_names": verified_ingredient_names,

            # 상세 결과
            "verified_ingredients": verified_ingredients,
            "unverified_ingredients": unverified_ingredients,

            "api_success_results": success_results,
            "api_failed_results": failed_results,
            "api_all_results": all_results,

            "verified_count": len(verified_ingredient_names),
            "unverified_count": len(unverified_ingredients),
            "total_checked_count": len(all_results)
        }

    def _build_unverified_item(self, matched_result):
        if not isinstance(matched_result, dict):
            return {
                "ocr_name": str(matched_result),
                "query_name": "",
                "reason": "검증 실패",
                "similarity": 0
            }

        return {
            "ocr_name": matched_result.get("ocr_name"),
            "query_name": matched_result.get("query_name"),
            "reason": matched_result.get("reason"),
            "similarity": matched_result.get("similarity")
        }

    # =========================================================
    # 3. 단일 성분 검증
    # =========================================================

    def match_ingredient(self, candidate_name, original_name=None):
        original_name = str(original_name or candidate_name).strip()

        cleaned_name = self._clean_query(candidate_name)
        cleaned_name = self._normalize_for_search_hint(cleaned_name)
        cleaned_name = self._clean_query(cleaned_name)

        if not cleaned_name:
            return self._build_failed_result(
                ocr_name=original_name,
                query_name="",
                reason="빈 성분 후보"
            )

        if self._is_invalid_candidate(cleaned_name):
            return self._build_failed_result(
                ocr_name=original_name,
                query_name=cleaned_name,
                reason="성분 후보가 아닌 문장/노이즈로 판단됨"
            )

        return self._match_with_query_candidates(
            ingredient_name=cleaned_name,
            original_name=original_name
        )

    def _match_with_query_candidates(self, ingredient_name, original_name=None):
        original_name = original_name or ingredient_name
        query_candidates = self._generate_query_candidates(ingredient_name)

        best_match = None

        for query in query_candidates:
            result = self._match_single_api(
                ingredient_name=query,
                original_name=original_name,
                print_log=True
            )

            if best_match is None:
                best_match = result

            elif result.get("similarity", 0) > best_match.get("similarity", 0):
                best_match = result

            if result.get("matched") and result.get("similarity", 0) >= self.strong_match_threshold:
                return result

        if best_match is None:
            return self._build_failed_result(
                ocr_name=original_name,
                query_name=ingredient_name,
                reason="검색 후보 생성 실패"
            )

        return best_match

    def _match_single_api(self, ingredient_name, original_name=None, print_log=False):
        original_name = original_name or ingredient_name

        query_name = self._normalize_for_search_hint(ingredient_name)
        query_name = self._clean_query(query_name)

        if not query_name:
            return self._build_failed_result(
                ocr_name=original_name,
                query_name="",
                reason="검색어 정리 후 빈 값"
            )

        if print_log:
            print(f"[API 검증] OCR='{original_name}' → 검색어='{query_name}'")

        result = self.search_ingredient(query_name)
        used_query = query_name

        if not result.get("success") or not result.get("items"):
            # [수정] 실제로 다른 변형일 때만 재시도
            alt_queries = []

            if "-" in query_name:
                alt_queries.append(query_name.replace("-", ""))

            if "," in query_name:
                alt_queries.append(query_name.replace(",", ""))

            if "/" in query_name:
                alt_queries.append(query_name.replace("/", ""))

            for alt_query in alt_queries:
                alt_query = self._clean_query(alt_query)

                if not alt_query or alt_query == query_name:
                    continue

                alt_result = self.search_ingredient(alt_query)

                if alt_result.get("success") and alt_result.get("items"):
                    result = alt_result
                    used_query = alt_query
                    break

        if not result.get("success") or not result.get("items"):
            return self._build_failed_result(
                ocr_name=original_name,
                query_name=used_query,
                reason="API 검색 결과 없음"
            )

        best_item = None
        best_score = 0.0

        for item in result.get("items", []):
            kor_name = self._get_value(
                item,
                ["INGR_KOR_NAME", "ingrKorName", "INGR_NM", "name"]
            )

            eng_name = self._get_value(
                item,
                ["INGR_ENG_NAME", "ingrEngName", "ENG_NM", "engName"]
            )

            alias_name = self._get_value(
                item,
                ["INGR_ALIAS", "ingrAlias", "ALIAS", "alias"]
            )

            candidate_scores = []

            if kor_name:
                candidate_scores.append(self._calculate_score(used_query, kor_name))
                candidate_scores.append(self._calculate_score(original_name, kor_name))

            if eng_name:
                candidate_scores.append(self._calculate_score(used_query, eng_name))
                candidate_scores.append(self._calculate_score(original_name, eng_name))

            if alias_name:
                candidate_scores.append(self._calculate_score(used_query, alias_name))
                candidate_scores.append(self._calculate_score(original_name, alias_name))

            if not candidate_scores:
                continue

            score = max(candidate_scores)

            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None:
            return self._build_failed_result(
                ocr_name=original_name,
                query_name=used_query,
                reason="API item에서 성분명 필드 없음"
            )

        matched_name_kr = self._get_value(
            best_item,
            ["INGR_KOR_NAME", "ingrKorName", "INGR_NM", "name"]
        )

        matched_name_en = self._get_value(
            best_item,
            ["INGR_ENG_NAME", "ingrEngName", "ENG_NM", "engName"]
        )

        cas_no = self._get_value(
            best_item,
            ["CAS_NO", "casNo", "CASNo", "CAS"]
        )

        definition = self._get_value(
            best_item,
            ["INGR_DEF", "ingrDef", "DEFINITION", "definition"]
        )

        matched = best_score >= self.match_threshold

        return {
            "ocr_name": original_name,
            "query_name": used_query,
            "matched": matched,
            "matched_name_kr": matched_name_kr if matched else None,
            "matched_name_en": matched_name_en if matched else None,
            "cas_no": cas_no if matched else None,
            "definition": definition if matched else None,
            "similarity": round(best_score, 3),
            "raw_item": best_item if matched else None,
            "source": "public_data_api" if matched else None,
            "reason": "API 존재 검증 통과" if matched else "API 존재 검증 유사도 기준 미달"
        }

    # =========================================================
    # 4. 후보 확장 / 분리
    # =========================================================

    def _expand_candidate_list(self, candidate_list):
        """
        ingredients_raw에서 넘어온 후보를 API 검색 전 개별 성분으로 분리한다.

        분리 기준:
        - 줄바꿈 / 콤마 / 세미콜론 / 가운뎃점 / 파이프 / 슬래시
        - 명시적 성분 라벨 제거

        특정 성분명 목록으로 분리하지 않는다.
        """

        results = []
        seen = set()

        for candidate in candidate_list or []:
            if candidate is None:
                continue

            if isinstance(candidate, dict):
                candidate = (
                    candidate.get("name")
                    or candidate.get("ingredient")
                    or candidate.get("ocr_name")
                    or candidate.get("query_name")
                    or ""
                )

            text = str(candidate).strip()

            if not text or text == "확인 불가":
                continue

            pieces = self._split_possible_joined_candidate(text)

            for piece in pieces:
                piece = self._clean_query(piece)
                piece = self._normalize_for_search_hint(piece)
                piece = self._clean_query(piece)

                if not piece:
                    continue

                if self._is_invalid_candidate(piece):
                    continue

                key = self._compact(piece)

                if key in seen:
                    continue

                seen.add(key)
                results.append(piece)

        return results

    def _split_possible_joined_candidate(self, text):
        """
        하나의 긴 OCR 문자열에서 개별 성분 후보를 분리한다.

        예:
        "전성분 정제수, 글리세린, 스테아릭애씨드"
        → ["정제수", "글리세린", "스테아릭애씨드"]

        "정제수\\n글리세린\\n스테아릭애씨드"
        → ["정제수", "글리세린", "스테아릭애씨드"]
        """

        if not text:
            return []

        text = self._normalize_candidate_text_for_split(text)
        text = self._remove_ingredient_section_label(text)

        if not text:
            return []

        line_parts = [
            part.strip()
            for part in re.split(r"[\r\n]+", text)
            if part.strip()
        ]

        if not line_parts:
            line_parts = [text]

        final_parts = []

        for line in line_parts:
            line = self._normalize_candidate_text_for_split(line)
            line = self._remove_ingredient_section_label(line)

            if not line:
                continue

            parts = re.split(r"[,，、;；|·ㆍ]+", line)

            for part in parts:
                part = self._normalize_candidate_text_for_split(part)
                part = part.strip(" ,;:：/·ㆍ-|()[]{}")

                if not part:
                    continue

                sub_parts = self._split_by_space_when_safe(part)

                for sub in sub_parts:
                    sub = self._normalize_candidate_text_for_split(sub)
                    sub = sub.strip(" ,;:：/·ㆍ-|()[]{}")

                    if sub:
                        final_parts.append(sub)

        return self._deduplicate_text_list(final_parts)

    def _split_by_space_when_safe(self, text):
        """
        공백 기준 분리를 보수적으로 수행한다.

        [수정] 영어 토큰 수에 따라 분리 기준 차별화
        기존: 영어 토큰 2개 이상이면 무조건 통째로 유지
              → "Water Glycerin Niacinamide" (3개 성분)도 하나로 보내 API 검색 실패

        수정:
        - 영어 토큰 정확히 2개 → 통째로 유지 ("Sodium Hyaluronate" 보존)
        - 영어 토큰 3개 이상 → 개별 토큰으로 분리
        - 한글 혼합: 기존과 동일 (토큰 3개 이상이고 유효 토큰 3개 이상이면 분리)
        """

        if not text:
            return []

        text = str(text).strip()

        if " " not in text:
            return [text]

        tokens = [
            token.strip()
            for token in text.split()
            if token.strip()
        ]

        if len(tokens) <= 1:
            return [text]

        # 영어 토큰 수 계산
        english_tokens = [
            t for t in tokens
            if re.fullmatch(r"[A-Za-z\-]+", t)
        ]

        # [수정] 영어 토큰 정확히 2개 → 유지 (예: "Sodium Hyaluronate")
        if len(english_tokens) == 2 and len(tokens) == 2:
            return [text]

        # [수정] 영어 토큰 3개 이상 → 개별 분리
        if len(english_tokens) >= 3:
            return [t for t in tokens if t]

        # 한글 혼합: 유효 토큰 3개 이상이면 분리
        valid_token_count = 0

        for token in tokens:
            token = token.strip(" ,;:：/·ㆍ-|()[]{}")

            if 2 <= len(token) <= 45 and re.search(r"[가-힣A-Za-z]", token):
                valid_token_count += 1

        if len(tokens) >= 3 and valid_token_count >= 3:
            return tokens

        return [text]

    def _generate_query_candidates(self, text):
        """
        API 검색어 후보 생성.

        [수정] 의미 있는 변형만 생성
        기존: 대시/콤마/슬래시 무조건 제거 변형 추가 → 원본과 같으면 중복 후보
        수정: 해당 문자가 실제로 있을 때만 제거 변형을 생성

        특정 성분 목록 기반 후보 생성은 하지 않는다.
        """

        if not text:
            return []

        text = self._normalize_for_search_hint(text)
        text = self._clean_query(text)

        if not text:
            return []

        candidates = []

        def add(value):
            value = self._normalize_for_search_hint(value)
            value = self._clean_query(value)

            if not value:
                return

            if len(value) <= 1:
                return

            if value not in candidates:
                candidates.append(value)

        # 원본
        add(text)

        # [수정] 해당 문자가 있을 때만 변형 추가
        if "-" in text:
            add(text.replace("-", ""))

        if "," in text:
            add(text.replace(",", ""))

        # ppm 숫자 제거 (있을 때만)
        if re.search(r"\d+ppm", text, flags=re.IGNORECASE):
            no_ppm = re.sub(r"\d+ppm", "", text, flags=re.IGNORECASE)
            add(no_ppm)

        # 슬래시 분리 (있을 때만)
        if "/" in text:
            add(text.replace("/", ""))
            for part in text.split("/"):
                add(part.strip())

        return candidates[:8]

    # =========================================================
    # 5. 실패 결과 생성 / 후보 검증
    # =========================================================

    def _build_failed_result(self, ocr_name, query_name=None, reason=""):
        return {
            "ocr_name": ocr_name,
            "query_name": query_name or ocr_name,
            "matched": False,
            "matched_name_kr": None,
            "matched_name_en": None,
            "cas_no": None,
            "definition": None,
            "similarity": 0,
            "raw_item": None,
            "source": None,
            "reason": reason
        }

    def _is_invalid_candidate(self, text):
        """
        [수정] 최대 길이를 self.max_candidate_length(90)로 완화
        기존 70자 고정에서 복합 성분명이 탈락하는 문제 방지.
        """

        if not text:
            return True

        text = str(text).strip()

        if len(text) <= 1:
            return True

        if text.isdigit():
            return True

        if not re.search(r"[가-힣a-zA-Z]", text):
            return True

        # [수정] 70 → self.max_candidate_length (90)
        if len(text) > self.max_candidate_length:
            return True

        lower_text = text.lower()

        for keyword in self.invalid_keywords:
            if keyword.lower() in lower_text:
                return True

        digit_count = sum(char.isdigit() for char in text)

        if digit_count >= len(text) * 0.5:
            return True

        if re.fullmatch(r"[a-zA-Z]{1,4}", text):
            if text.lower() not in ["peg", "ppg"]:
                return True

        if re.search(
            r"https?://|www\.|\.com|\.co\.kr|\.kr",
            text,
            flags=re.IGNORECASE
        ):
            return True

        return False

    # =========================================================
    # 6. 유사도 계산
    # =========================================================

    def _calculate_score(self, text1, text2):
        text1 = self._normalize_for_search_hint(self._clean_query(text1))
        text2 = self._normalize_for_search_hint(self._clean_query(text2))

        if not text1 or not text2:
            return 0.0

        if text1 == text2:
            return 1.0

        compact1 = self._compact(text1)
        compact2 = self._compact(text2)

        if compact1 == compact2:
            return 1.0

        similarity = SequenceMatcher(
            None,
            compact1,
            compact2
        ).ratio()

        if compact1 in compact2 or compact2 in compact1:
            similarity += 0.10

        length_diff = abs(len(compact1) - len(compact2))

        if length_diff >= 10:
            similarity *= 0.78

        elif length_diff >= 8:
            similarity *= 0.84

        elif length_diff >= 5:
            similarity *= 0.90

        return round(min(similarity, 1.0), 3)

    # =========================================================
    # 7. 텍스트 정규화
    # =========================================================

    def _normalize_candidate_text_for_split(self, text):
        if not text:
            return ""

        text = str(text)

        replace_map = {
            "，": ",",
            "、": ",",
            "；": ";",
            "ㆍ": "·",
            "：": ":",
            "\r\n": "\n",
            "\r": "\n",
            "\t": " ",
            "㎖": "ml",
            "ｍｌ": "ml",
            "ＭＬ": "ml",
            "ｇ": "g",
            "Ｇ": "g",
            "–": "-",
            "—": "-",
            "−": "-"
        }

        for old, new in replace_map.items():
            text = text.replace(old, new)

        text = re.sub(r"\s*,\s*", ", ", text)
        text = re.sub(r"\s*;\s*", "; ", text)
        text = re.sub(r"\s*:\s*", ": ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)

        return text.strip()

    def _remove_ingredient_section_label(self, text):
        if not text:
            return ""

        remove_patterns = [
            r"^\s*전\s*성\s*분\s*[:：]?\s*",
            r"^\s*성\s*분\s*[:：]?\s*",
            r"^\s*주\s*성\s*분\s*[:：]?\s*",
            r"^\s*주요\s*성\s*분\s*[:：]?\s*",
            r"^\s*원료\s*명?\s*[:：]?\s*",
            r"^\s*ingredients?\s*[:：]?\s*",
            r"^\s*ingredient\s*list\s*[:：]?\s*"
        ]

        result = str(text).strip()

        changed = True

        while changed:
            before = result

            for pattern in remove_patterns:
                result = re.sub(
                    pattern,
                    "",
                    result,
                    flags=re.IGNORECASE
                ).strip()

            changed = before != result

        return result.strip()

    def _normalize_for_search_hint(self, text):
        if not text:
            return ""

        text = str(text)

        text = text.replace("，", ",")
        text = text.replace("、", ",")
        text = text.replace("；", ";")
        text = text.replace("ㆍ", "·")
        text = text.replace("：", ":")

        text = re.sub(r"나이아신\s*아마이드", "나이아신아마이드", text)
        text = re.sub(r"1\s*,?\s*2\s*-\s*", "1,2-", text)
        text = re.sub(r"(\d)\s*ppm", r"\1ppm", text, flags=re.IGNORECASE)

        for wrong, correct in self.ocr_hint_map.items():
            text = text.replace(wrong, correct)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _clean_query(self, text):
        """
        API 검색어를 정리한다.

        [수정] 공백 보존
        기존: [^가-힣a-zA-Z0-9,\\-/] 정규식 → 공백까지 제거
              → "Sodium Hyaluronate" → "SodiumHyaluronate" → API 검색 실패

        수정: \\s 포함으로 공백 허용
              탭/줄바꿈은 위에서 공백으로 치환했으므로 스페이스만 남음
              → 영문 성분명 공백 보존
        """

        if text is None:
            return ""

        text = str(text).strip()

        text = self._remove_ingredient_section_label(text)

        text = text.replace("，", ",")
        text = text.replace("、", ",")
        text = text.replace("；", ";")
        text = text.replace("ㆍ", "·")
        text = text.replace("：", ":")

        text = text.replace("\r", " ")
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")

        # [수정] \s → 공백(스페이스) 허용
        # 기존: [^가-힣a-zA-Z0-9,\-/] → 공백 제거
        # 수정: [^가-힣a-zA-Z0-9,\-/\s] → 공백 보존
        text = re.sub(
            r"[^가-힣a-zA-Z0-9,\-/\s]",
            "",
            text
        )

        # 연속 공백 정리
        text = re.sub(r" +", " ", text)

        text = re.sub(r",+", ",", text)
        text = text.strip(" ,;:/-")

        return text.strip()

    def _looks_like_english_ingredient_phrase(self, text):
        if not text:
            return False

        if not re.search(r"[A-Za-z]", text):
            return False

        tokens = [
            token.strip()
            for token in str(text).split()
            if token.strip()
        ]

        if len(tokens) <= 1:
            return False

        english_token_count = sum(
            1
            for token in tokens
            if re.fullmatch(r"[A-Za-z\-]+", token)
        )

        return english_token_count >= 2

    def _deduplicate_text_list(self, items):
        results = []
        seen = set()

        for item in items or []:
            if item is None:
                continue

            text = str(item).strip()

            if not text:
                continue

            key = self._compact(text)

            if key in seen:
                continue

            seen.add(key)
            results.append(text)

        return results

    # =========================================================
    # 8. API 응답 파싱
    # =========================================================

    def _extract_items(self, data):
        """
        공공데이터포털 JSON 응답 구조가 서비스별로 약간 다를 수 있어
        body/items/item 구조를 최대한 유연하게 처리한다.
        """

        if not isinstance(data, dict):
            return []

        try:
            body = data.get("body", {})

            if isinstance(body, dict):
                items = body.get("items", [])
                parsed = self._parse_items_container(items)

                if parsed:
                    return parsed

            response = data.get("response", {})

            if isinstance(response, dict):
                body = response.get("body", {})

                if isinstance(body, dict):
                    items = body.get("items", [])
                    parsed = self._parse_items_container(items)

                    if parsed:
                        return parsed

            items = data.get("items", [])
            parsed = self._parse_items_container(items)

            if parsed:
                return parsed

            return []

        except AttributeError:
            return []

    def _parse_items_container(self, items):
        if not items:
            return []

        if isinstance(items, dict):
            if "item" in items:
                item = items.get("item", [])

                if isinstance(item, list):
                    return item

                if isinstance(item, dict):
                    return [item]

                return []

            return [items]

        if isinstance(items, list):
            return items

        return []

    def _get_value(self, item, possible_keys):
        if not isinstance(item, dict):
            return None

        for key in possible_keys:
            if key in item and item[key]:
                return str(item[key]).strip()

        return None

    def _compact(self, text):
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
            .replace(",", "")
            .replace("/", "")
            .replace("·", "")
            .replace("ㆍ", "")
            .replace(".", "")
            .replace(";", "")
            .replace("；", "")
        )