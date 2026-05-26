import cv2
import re
from urllib.parse import urlparse


class QRReader:
    """
    Dermalens QR / URL 추출 클래스

    역할:
    1. 이미지에서 QR 코드를 직접 인식한다.
    2. OCR 텍스트에서 URL 후보를 추출한다.
    3. QR/URL 값을 정리하고 중복 제거한다.
    4. 최종 JSON에 넣기 쉬운 qr_result / qr_analysis_base 구조를 만든다.

    주의:
    - 이 파일은 QR/URL을 '찾는 역할'이다.
    - QR URL에 접속해서 페이지 내용을 분석하는 기능은 src/api/qr_analyzer.py에서 처리한다.
    - QR이 없을 수도 있으므로 없으면 확인 불가 구조로 반환한다.
    """

    def __init__(self):
        self.detector = cv2.QRCodeDetector()

        self.allowed_url_domains = [
            "com", "co.kr", "kr", "net", "org", "io", "ai",
            "shop", "mall", "beauty", "me", "app", "store",
            "brand", "global", "co", "jp", "cn"
        ]

        self.invalid_qr_keywords = [
            "제조번호", "제조일자", "사용기한", "유통기한",
            "소비자상담", "고객센터", "주소", "전화",
            "용량", "중량", "내용량", "전성분", "성분",
            "사용방법", "주의사항", "품질보증", "분리수거",
            "제조원", "판매원", "책임판매업자", "제조업자"
        ]

    # =========================================================
    # 1. QR 코드 직접 인식
    # =========================================================

    def read_qr(self, image_path):
        """
        이미지에서 QR 코드를 직접 인식한다.

        여러 전처리 버전을 시도한다.
        - 원본
        - 그레이스케일
        - 대비 보정
        - 이진화
        - 확대 이미지
        - 샤프닝 이미지
        - CLAHE 이미지
        """

        image = cv2.imread(image_path)

        if image is None:
            print("[경고] QR 인식용 이미지 로드 실패")
            return []

        results = []

        for variant in self._build_qr_image_variants(image):
            try:
                results.extend(
                    self._decode_qr_from_image(variant)
                )
            except Exception:
                continue

        valid_results = []

        for item in results:
            cleaned = self._clean_qr_value(item)

            if self._is_valid_qr_or_url(cleaned):
                valid_results.append(cleaned)

        return self._remove_duplicates(valid_results)

    def _build_qr_image_variants(self, image):
        """
        QR 인식용 이미지 variant 생성
        """

        variants = []

        if image is None:
            return variants

        variants.append(image)

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variants.append(gray)
        except Exception:
            return variants

        try:
            enhanced = cv2.equalizeHist(gray)
            variants.append(enhanced)
        except Exception:
            pass

        try:
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )
            clahe_img = clahe.apply(gray)
            variants.append(clahe_img)
        except Exception:
            pass

        try:
            _, binary = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            variants.append(binary)
        except Exception:
            pass

        try:
            scaled = cv2.resize(
                gray,
                None,
                fx=1.5,
                fy=1.5,
                interpolation=cv2.INTER_CUBIC
            )
            variants.append(scaled)
        except Exception:
            pass

        try:
            scaled_big = cv2.resize(
                gray,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_CUBIC
            )
            variants.append(scaled_big)
        except Exception:
            pass

        try:
            sharpened = self._sharpen_image(gray)
            variants.append(sharpened)
        except Exception:
            pass

        return variants

    # =========================================================
    # 2. 이미지에서 QR 디코딩
    # =========================================================

    def _decode_qr_from_image(self, image):
        results = []

        # 단일 QR 인식
        try:
            data, points, _ = self.detector.detectAndDecode(image)

            if data:
                results.append(data)

        except Exception:
            pass

        # 여러 QR 인식
        try:
            retval, decoded_info, points, _ = self.detector.detectAndDecodeMulti(image)

            if retval and decoded_info:
                for item in decoded_info:
                    if item:
                        results.append(item)

        except Exception:
            pass

        return results

    def _sharpen_image(self, gray_image):
        """
        QR 경계가 흐릴 때 보조적으로 사용하는 샤프닝 처리
        """

        try:
            blurred = cv2.GaussianBlur(gray_image, (0, 0), 3)
            sharpened = cv2.addWeighted(gray_image, 1.5, blurred, -0.5, 0)
            return sharpened

        except Exception:
            return gray_image

    # =========================================================
    # 3. OCR 텍스트에서 URL 후보 추출
    # =========================================================

    def extract_urls_from_text(self, text):
        """
        OCR 텍스트에서 URL 후보를 추출한다.

        OCR에서 URL은 다음처럼 깨질 수 있다.
        - www . brand . com
        - http : // example . com
        - example . co . kr
        이를 최대한 복원한 뒤 URL 후보로 추출한다.
        """

        if not text:
            return []

        normalized_text = self._normalize_ocr_url_text(text)

        patterns = [
            r"https?://[^\s<>()\[\]{}\"']+",
            r"www\.[^\s<>()\[\]{}\"']+",
            r"\b[a-zA-Z0-9][a-zA-Z0-9.-]*\.(?:com|co\.kr|kr|net|org|io|ai|shop|mall|beauty|me|app|store|global|jp|cn)[^\s<>()\[\]{}\"']*"
        ]

        results = []

        for pattern in patterns:
            matches = re.findall(
                pattern,
                normalized_text,
                flags=re.IGNORECASE
            )

            for match in matches:
                cleaned = self._clean_qr_value(match)

                if self._is_valid_qr_or_url(cleaned):
                    results.append(cleaned)

        return self._remove_duplicates(results)

    # =========================================================
    # 4. QR + OCR URL 통합
    # =========================================================

    def read_all(self, image_path, ocr_text=""):
        """
        QR 직접 인식 결과와 OCR URL 후보를 통합한다.

        반환 구조:
        {
            "qr_detected": [...],
            "url_candidates": [...],
            "all_qr_values": [...],
            "qr_result": {
                "detected": true,
                "url": "...",
                "urls": [...],
                "raw_values": [...]
            },
            "qr_analysis_base": {...}
        }
        """

        print("[진행중] QR 코드 이미지 직접 인식 중...")

        qr_results = self.read_qr(image_path)

        print(f"[완료] QR 직접 인식 결과: {len(qr_results)}개")

        print("[진행중] OCR 텍스트 기반 URL 추출 중...")

        url_results = self.extract_urls_from_text(ocr_text)

        print(f"[완료] OCR URL 후보 추출 결과: {len(url_results)}개")

        all_values = self._remove_duplicates(qr_results + url_results)

        qr_analysis_base = self.build_qr_analysis_base(
            qr_detected=qr_results,
            url_candidates=url_results,
            all_qr_values=all_values
        )

        qr_result = self.build_qr_result(
            qr_detected=qr_results,
            url_candidates=url_results,
            all_qr_values=all_values
        )

        return {
            "qr_detected": qr_results if qr_results else ["확인 불가"],
            "url_candidates": url_results if url_results else ["확인 불가"],
            "all_qr_values": all_values if all_values else ["확인 불가"],
            "qr_result": qr_result,
            "qr_analysis_base": qr_analysis_base
        }

    def build_qr_result(self, qr_detected, url_candidates, all_qr_values):
        """
        최종 JSON result.qr에 넣기 좋은 구조 생성
        """

        clean_qr_detected = self._remove_duplicates(qr_detected)
        clean_url_candidates = self._remove_duplicates(url_candidates)
        clean_all_values = self._remove_duplicates(all_qr_values)

        urls = []

        for value in clean_all_values:
            if self._looks_like_url(value):
                urls.append(
                    self._normalize_url_for_request(value)
                )

        urls = self._remove_duplicates(urls)

        detected = bool(clean_qr_detected or clean_url_candidates or urls)

        primary_url = urls[0] if urls else "확인 불가"

        return {
            "detected": detected,
            "url": primary_url,
            "urls": urls if urls else [],
            "direct_qr_values": clean_qr_detected,
            "ocr_url_values": clean_url_candidates,
            "raw_values": clean_all_values,
            "status": "detected" if detected else "not_detected"
        }

    def build_qr_analysis_base(self, qr_detected, url_candidates, all_qr_values):
        """
        최종 JSON의 qr_analysis에 들어갈 기본 구조를 만든다.

        여기서는 URL에 접속하지 않는다.
        URL 접속 및 페이지 분석은 QRAnalyzer에서 수행한다.
        """

        clean_qr_detected = self._remove_duplicates(qr_detected)
        clean_url_candidates = self._remove_duplicates(url_candidates)
        clean_all_values = self._remove_duplicates(all_qr_values)

        url_values = []
        non_url_values = []

        for value in clean_all_values:
            if self._looks_like_url(value):
                url_values.append(
                    self._normalize_url_for_request(value)
                )
            else:
                non_url_values.append(value)

        url_values = self._remove_duplicates(url_values)
        non_url_values = self._remove_duplicates(non_url_values)

        return {
            "qr_codes": clean_all_values if clean_all_values else ["확인 불가"],
            "ocr_qr_codes": clean_url_candidates if clean_url_candidates else [],
            "direct_qr_results": clean_qr_detected if clean_qr_detected else [],
            "url_candidates": url_values,
            "non_url_qr_values": non_url_values,
            "analysis_ready_urls": url_values,
            "analysis_status": "ready" if url_values else "no_url_to_analyze",
            "analysis_note": "QR/URL 값 추출 완료. URL 내용 분석은 QRAnalyzer 단계에서 수행"
        }

    # =========================================================
    # 5. OCR URL 텍스트 보정
    # =========================================================

    def _normalize_ocr_url_text(self, text):
        if not text:
            return ""

        text = str(text)

        # OCR 줄바꿈 때문에 URL이 끊기는 경우 보완
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")

        replacements = {
            "h t t p": "http",
            "h t t p s": "https",
            "http : //": "http://",
            "https : //": "https://",
            "http: //": "http://",
            "https: //": "https://",
            "http ://": "http://",
            "https ://": "https://",
            "www .": "www.",
            ". com": ".com",
            ". co . kr": ".co.kr",
            ". co.kr": ".co.kr",
            ". kr": ".kr",
            ". net": ".net",
            ". org": ".org",
            ". io": ".io",
            ". ai": ".ai",
            ". shop": ".shop",
            ". mall": ".mall",
            ". beauty": ".beauty",
            ". me": ".me",
            ". app": ".app",
            ". store": ".store",
            ". global": ".global",
            ". jp": ".jp",
            ". cn": ".cn"
        }

        normalized = text

        for wrong, correct in replacements.items():
            normalized = re.sub(
                re.escape(wrong),
                correct,
                normalized,
                flags=re.IGNORECASE
            )

        normalized = re.sub(
            r"(https?)\s*:\s*/\s*/",
            r"\1://",
            normalized,
            flags=re.IGNORECASE
        )

        normalized = re.sub(
            r"www\s*\.\s*",
            "www.",
            normalized,
            flags=re.IGNORECASE
        )

        normalized = re.sub(
            r"([a-zA-Z0-9])\s*\.\s*(com|kr|net|org|io|ai|shop|mall|beauty|me|app|store|global|jp|cn)",
            r"\1.\2",
            normalized,
            flags=re.IGNORECASE
        )

        normalized = re.sub(
            r"\.co\s*\.\s*kr",
            ".co.kr",
            normalized,
            flags=re.IGNORECASE
        )

        normalized = re.sub(r"\s+", " ", normalized)

        return normalized.strip()

    # =========================================================
    # 6. QR / URL 값 정리
    # =========================================================

    def _clean_qr_value(self, value):
        if value is None:
            return ""

        value = str(value).strip()

        if not value:
            return ""

        value = value.replace("\n", "")
        value = value.replace("\t", "")
        value = value.strip()

        value = re.sub(r"^[\s\"'`<>\[\](){}]+", "", value)
        value = re.sub(r"[\s\"'`<>\[\](){},.;]+$", "", value)

        if self._looks_like_url(value):
            value = re.sub(r"\s+", "", value)
            value = self._normalize_url_for_request(value)

        return value.strip()

    def _normalize_url_for_request(self, value):
        """
        URL 요청이 가능하도록 형태를 정리한다.

        예:
        www.example.com → https://www.example.com
        example.com → https://example.com
        """

        if not value:
            return ""

        value = str(value).strip()
        value = value.replace(" ", "")

        if value.startswith("http://") or value.startswith("https://"):
            return value

        if value.startswith("www."):
            return "https://" + value

        if self._looks_like_domain(value):
            return "https://" + value

        return value

    # =========================================================
    # 7. QR / URL 유효성 판단
    # =========================================================

    def _is_valid_qr_or_url(self, value):
        if not value:
            return False

        if value == "확인 불가":
            return False

        value = str(value).strip()

        if len(value) < 4:
            return False

        compact = self._compact(value)

        for keyword in self.invalid_qr_keywords:
            if self._compact(keyword) in compact:
                return False

        # 숫자만 있으면 제조번호/바코드일 가능성이 높음
        if re.fullmatch(r"\d+", value):
            return False

        # 전화번호 제거
        if re.fullmatch(r"\d{2,4}-\d{3,4}-\d{4}", value):
            return False

        # 일반 URL이면 통과
        if self._looks_like_url(value):
            return True

        # URL이 아닌 QR 텍스트도 가능하므로 영문/한글이 포함된 긴 문자열은 유지
        if len(value) >= 8 and re.search(r"[a-zA-Z가-힣]", value):
            return True

        return False

    def _looks_like_url(self, value):
        if not value:
            return False

        value = str(value).strip()

        patterns = [
            r"^https?://",
            r"^www\.",
            r"^[a-zA-Z0-9][a-zA-Z0-9.-]*\.(com|co\.kr|kr|net|org|io|ai|shop|mall|beauty|me|app|store|global|jp|cn)"
        ]

        for pattern in patterns:
            if re.search(
                pattern,
                value,
                flags=re.IGNORECASE
            ):
                return True

        return False

    def _looks_like_domain(self, value):
        if not value:
            return False

        value = str(value).strip()

        return bool(
            re.search(
                r"^[a-zA-Z0-9][a-zA-Z0-9.-]*\.(com|co\.kr|kr|net|org|io|ai|shop|mall|beauty|me|app|store|global|jp|cn)",
                value,
                flags=re.IGNORECASE
            )
        )

    # =========================================================
    # 8. 중복 제거 / 유틸
    # =========================================================

    def _remove_duplicates(self, items):
        results = []
        seen = set()

        for item in items or []:
            if item is None:
                continue

            item = str(item).strip()

            if not item:
                continue

            if item == "확인 불가":
                continue

            key = item.lower().replace(" ", "")

            if key not in seen:
                results.append(item)
                seen.add(key)

        return results

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
            .replace(".", "")
            .replace(",", "")
            .replace("/", "")
            .replace("·", "")
            .replace("ㆍ", "")
            .replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
        )