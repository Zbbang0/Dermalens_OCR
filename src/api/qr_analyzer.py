import re
import requests
from html import unescape


class QRAnalyzer:
    def __init__(self):
        self.timeout = 7

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
        }

    def analyze(self, qr_analysis_base=None, qr_values=None):
        base = qr_analysis_base or {}

        qr_codes = base.get("qr_codes", [])
        ocr_qr_codes = base.get("ocr_qr_codes", [])
        direct_qr_results = base.get("direct_qr_results", [])
        url_candidates = base.get("url_candidates", [])
        analysis_ready_urls = base.get("analysis_ready_urls", [])

        all_values = []

        for group in [
            qr_codes,
            ocr_qr_codes,
            direct_qr_results,
            url_candidates,
            analysis_ready_urls,
            qr_values or []
        ]:
            if isinstance(group, str):
                group = [group]

            for item in group:
                if item and item != "확인 불가" and item not in all_values:
                    all_values.append(item)

        urls = []

        for value in all_values:
            if self._looks_like_url(value):
                urls.append(self._normalize_url(value))

        urls = self._remove_duplicates(urls)

        page_analysis_results = []
        resolved_urls = []
        page_titles = []
        meta_descriptions = []
        product_name_hints = []

        for url in urls:
            result = self._analyze_url(url)
            page_analysis_results.append(result)

            if result.get("success"):
                if result.get("resolved_url"):
                    resolved_urls.append(result.get("resolved_url"))

                if result.get("page_title"):
                    page_titles.append(result.get("page_title"))
                    product_name_hints.append(result.get("page_title"))

                if result.get("meta_description"):
                    meta_descriptions.append(result.get("meta_description"))

        return {
            "qr_codes": all_values if all_values else ["확인 불가"],
            "ocr_qr_codes": ocr_qr_codes if isinstance(ocr_qr_codes, list) else [ocr_qr_codes],
            "direct_qr_results": direct_qr_results if isinstance(direct_qr_results, list) else [direct_qr_results],
            "url_candidates": urls,
            "resolved_urls": self._remove_duplicates(resolved_urls),
            "page_titles": self._remove_duplicates(page_titles),
            "meta_descriptions": self._remove_duplicates(meta_descriptions),
            "product_name_hints": self._remove_duplicates(product_name_hints),
            "page_analysis_results": page_analysis_results,
            "analysis_status": "analyzed" if page_analysis_results else "no_url_to_analyze",
            "analysis_summary": "QR/URL 분석 보강 완료" if page_analysis_results else "분석 가능한 QR/URL 없음",
            "analysis_note": "QR/URL 정보는 제품 정보 확인 및 분석 보강용으로 사용"
        }

    def _analyze_url(self, url):
        result = {
            "input_url": url,
            "success": False,
            "status_code": None,
            "resolved_url": None,
            "page_title": None,
            "meta_description": None,
            "error": None
        }

        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=True
            )

            result["status_code"] = response.status_code
            result["resolved_url"] = response.url

            if response.status_code < 200 or response.status_code >= 400:
                result["error"] = f"HTTP 오류: {response.status_code}"
                return result

            content_type = response.headers.get("Content-Type", "")

            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                result["error"] = f"HTML 문서가 아님: {content_type}"
                return result

            html = self._get_response_text_with_correct_encoding(response)

            if not html.strip():
                result["error"] = "HTML 내용 비어 있음"
                return result

            result["page_title"] = self._extract_title(html)
            result["meta_description"] = self._extract_meta_description(html)
            result["success"] = True

            return result

        except Exception as e:
            result["error"] = str(e)
            return result

    def _get_response_text_with_correct_encoding(self, response):
        """
        한글 페이지 제목이 ë\x89´... 형태로 깨지는 문제 방지용.

        requests가 ISO-8859-1 등으로 잘못 추정하면 response.text가 깨질 수 있다.
        apparent_encoding을 우선 적용하고, 실패하면 utf-8로 재시도한다.
        """

        if response is None:
            return ""

        try:
            if response.apparent_encoding:
                response.encoding = response.apparent_encoding

            text = response.text

            if self._looks_like_broken_korean(text):
                try:
                    text = response.content.decode("utf-8", errors="replace")
                except Exception:
                    pass

            if self._looks_like_broken_korean(text):
                try:
                    text = response.content.decode("euc-kr", errors="replace")
                except Exception:
                    pass

            return text or ""

        except Exception:
            try:
                return response.content.decode("utf-8", errors="replace")
            except Exception:
                return response.text or ""

    def _looks_like_broken_korean(self, text):
        if not text:
            return False

        broken_patterns = [
            "ë",
            "ì",
            "í",
            "ê",
            "Â",
            "Ã",
            "\x89",
            "\x8a",
            "\x9c"
        ]

        hit_count = 0

        for pattern in broken_patterns:
            if pattern in text:
                hit_count += 1

        return hit_count >= 2

    def _extract_title(self, html):
        if not html:
            return ""

        match = re.search(
            r"<title[^>]*>(.*?)</title>",
            html,
            flags=re.IGNORECASE | re.DOTALL
        )

        if not match:
            return ""

        return self._clean_html_text(match.group(1))

    def _extract_meta_description(self, html):
        if not html:
            return ""

        patterns = [
            r'<meta\s+[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\'][^>]*>',
            r'<meta\s+[^>]*content=["\'](.*?)["\'][^>]*name=["\']description["\'][^>]*>',
            r'<meta\s+[^>]*property=["\']og:description["\'][^>]*content=["\'](.*?)["\'][^>]*>',
            r'<meta\s+[^>]*content=["\'](.*?)["\'][^>]*property=["\']og:description["\'][^>]*>'
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                html,
                flags=re.IGNORECASE | re.DOTALL
            )

            if match:
                return self._clean_html_text(match.group(1))

        return ""

    def _clean_html_text(self, text):
        if not text:
            return ""

        text = unescape(str(text))
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = text.replace("\r", " ")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _looks_like_url(self, value):
        if not value:
            return False

        value = str(value).strip()

        return bool(
            re.search(
                r"^(https?://|www\.|[a-zA-Z0-9][a-zA-Z0-9.-]*\.(com|co\.kr|kr|net|org|io|ai|shop|mall|beauty|me|app|store))",
                value,
                flags=re.IGNORECASE
            )
        )

    def _normalize_url(self, value):
        value = str(value).strip().replace(" ", "")

        if value.startswith("http://") or value.startswith("https://"):
            return value

        if value.startswith("www."):
            return "https://" + value

        return "https://" + value

    def _remove_duplicates(self, items):
        results = []
        seen = set()

        for item in items:
            if not item:
                continue

            text = str(item).strip()

            if not text or text == "확인 불가":
                continue

            key = text.lower().replace(" ", "")

            if key not in seen:
                seen.add(key)
                results.append(text)

        return results