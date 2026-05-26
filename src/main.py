from datetime import datetime
import os
import re
import time
import traceback

from src.ocr.run_ocr import OCRRunner
from src.ocr.section_detector import OCRSectionDetector
from src.ocr.postprocess import IngredientPostprocessor
from src.ocr.qr_reader import QRReader


from src.api.ingredient_api import IngredientAPI
from src.api.qr_analyzer import QRAnalyzer

from src.ai.gpt_extractor import GPTExtractor

from src.utils.file_io import FileIO


# =========================================================
# 출력 로그 함수 — 진행상황 가독성 개선
# =========================================================

# ANSI 색상 코드 (터미널 지원 여부 체크)
_USE_COLOR = os.getenv("NO_COLOR") is None and os.isatty(1) if hasattr(os, "isatty") else False

def _c(text, code):
    """ANSI 색상 적용. 터미널이 아니면 그대로 반환."""
    if _USE_COLOR:
        return f"\033[{code}m{text}\033[0m"
    return text

def _green(t):   return _c(t, "32")
def _yellow(t):  return _c(t, "33")
def _red(t):     return _c(t, "31")
def _cyan(t):    return _c(t, "36")
def _bold(t):    return _c(t, "1")
def _dim(t):     return _c(t, "2")


# 전체 단계 수 (진행률 표시용)
_TOTAL_STEPS = 10
_step_start_times = {}


def print_section(title, step=None):
    """
    단계 구분선 출력.
    step 인자를 주면 진행률 표시.
    """
    print()
    print(_bold("=" * 70))

    if step is not None:
        bar_filled = int((step / _TOTAL_STEPS) * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        pct = int(step / _TOTAL_STEPS * 100)
        header = _bold(_cyan(f"[{step:2d}/{_TOTAL_STEPS}] [{bar}] {pct:3d}%  {title}"))
        _step_start_times[step] = time.time()
    else:
        header = _bold(_cyan(f"  {title}"))

    print(header)
    print(_bold("=" * 70))


def print_step_done(step, title=None):
    """단계 완료 + 소요 시간."""
    elapsed = ""
    if step in _step_start_times:
        secs = time.time() - _step_start_times[step]
        elapsed = _dim(f"  ({secs:.1f}s)")
    label = title or ""
    print(_green(f"  ✔ {label} 완료{elapsed}"))


def print_sub_step(message):
    print(_dim(f"  ▸ {message}..."))


def print_done(message):
    print(_green(f"  ✔ {message}"))


def print_warning(message):
    print(_yellow(f"  ⚠ {message}"))


def print_error(message):
    print(_red(f"  ✖ {message}"))


def print_info(message):
    print(_dim(f"    {message}"))


def print_table_row(label, value, indent=4):
    label_str = f"{label}:"
    print(f"{' ' * indent}{_bold(label_str):30s} {value}")


def print_summary_box(title, rows):
    """
    작은 요약 박스 출력.
    rows: [(label, value), ...]
    """
    width = 60
    print()
    print(_dim("  ┌" + "─" * (width - 2) + "┐"))
    print(_dim("  │") + _bold(f" {title}".ljust(width - 2)) + _dim("│"))
    print(_dim("  ├" + "─" * (width - 2) + "┤"))
    for label, value in rows:
        line = f" {label}: {value}"
        print(_dim("  │") + line.ljust(width - 2) + _dim("│"))
    print(_dim("  └" + "─" * (width - 2) + "┘"))


# =========================================================
# 공통 보조 함수
# =========================================================

def normalize_list(items):
    """
    리스트 정리 함수

    역할:
    - None 제거
    - 빈 문자열 제거
    - '확인 불가' 제거
    - 중복 제거
    - 기존 순서 유지
    """

    if not items:
        return []

    if isinstance(items, str):
        items = [items]

    if not isinstance(items, list):
        items = [items]

    results = []
    seen = set()

    for item in items:
        if item is None:
            continue

        if isinstance(item, dict):
            item = (
                item.get("matched_name_kr")
                or item.get("name")
                or item.get("ocr_name")
                or item.get("query_name")
                or item.get("url")
                or str(item)
            )

        text = str(item).strip()

        if not text:
            continue

        if text in [
            "확인 불가", "없음", "null", "None",
            "none", "N/A", "n/a", "unknown", "Unknown"
        ]:
            continue

        key = (
            text.lower()
            .replace(" ", "")
            .replace("\n", "")
            .replace("\t", "")
            .replace("\r", "")
        )

        if key not in seen:
            seen.add(key)
            results.append(text)

    return results


def safe_list(items):
    cleaned = normalize_list(items)
    return cleaned if cleaned else ["확인 불가"]


def value_or_unknown(value):
    if value is None:
        return "확인 불가"

    if isinstance(value, list):
        cleaned = normalize_list(value)
        return cleaned[0] if cleaned else "확인 불가"

    if isinstance(value, dict):
        if "url" in value:
            return value_or_unknown(value.get("url"))
        if "value" in value:
            return value_or_unknown(value.get("value"))
        return "확인 불가"

    text = str(value).strip()

    if not text:
        return "확인 불가"

    if text in [
        "확인 불가", "없음", "null", "None",
        "none", "N/A", "n/a", "unknown", "Unknown"
    ]:
        return "확인 불가"

    return text


def safe_save(method_name, data):
    try:
        save_method = getattr(FileIO, method_name, None)

        if save_method is None:
            print_warning(f"FileIO.{method_name} 함수 없음 — 저장 생략")
            return None

        return save_method(data)

    except Exception as error:
        print_error(f"FileIO.{method_name} 저장 실패: {error}")
        return None


def choose_best_text(values):
    cleaned = normalize_list(values)
    return cleaned[0] if cleaned else "확인 불가"


def merge_text_values(values):
    cleaned = normalize_list(values)
    return "\n\n".join(cleaned) if cleaned else ""


def merge_fragmented_text(items):
    """
    OCR이 한 문장을 여러 줄로 쪼개놓은 경우, 한 줄로 합쳐 가독성을 높인다.

    - 모든 fragment를 공백으로 연결
    - 다중 공백/줄바꿈 정규화
    - 빈 값이면 ["확인 불가"] 반환
    """
    cleaned = normalize_list(items)
    if not cleaned:
        return ["확인 불가"]

    merged = " ".join(cleaned)
    merged = re.sub(r"\s+", " ", merged).strip()

    return [merged] if merged else ["확인 불가"]


def compute_accuracy_block(ocr_confidence_avg, ingredient_api_validation):
    """
    server payload 최상단에 들어갈 accuracy 블록 생성.

    - ocr_confidence: PaddleOCR/Vision 평균 신뢰도 (0.0 ~ 1.0)
    - ingredient_match_rate: 성분 API 검증 성공률 (0.0 ~ 100.0, %)
    """
    try:
        ocr_conf = float(ocr_confidence_avg or 0.0)
    except (TypeError, ValueError):
        ocr_conf = 0.0

    if not isinstance(ingredient_api_validation, dict):
        ingredient_api_validation = {}

    verified = int(ingredient_api_validation.get("verified_count", 0) or 0)
    total = int(ingredient_api_validation.get("total_checked_count", 0) or 0)

    match_rate = (verified / total * 100.0) if total > 0 else 0.0

    return {
        "ocr_confidence": round(ocr_conf, 4),
        "ingredient_match_rate": round(match_rate, 2),
        "ingredient_verified_count": verified,
        "ingredient_total_count": total
    }


# =========================================================
# QR 관련 보조 함수
# =========================================================

def extract_qr_result_fields(qr_result):
    if not isinstance(qr_result, dict):
        return [], [], [], {}

    qr_inner = qr_result.get("qr_result", {}) or {}

    direct_qr_results = normalize_list(
        qr_result.get("qr_detected", [])
        or qr_result.get("direct_qr_results", [])
        or qr_inner.get("direct_qr_values", [])
        or qr_result.get("qr_codes", [])
        or qr_result.get("decoded_qr", [])
    )

    url_candidates = normalize_list(
        qr_result.get("url_candidates", [])
        or qr_result.get("urls", [])
        or qr_result.get("ocr_urls", [])
        or qr_inner.get("urls", [])
        or qr_inner.get("ocr_url_values", [])
    )

    all_qr_values = normalize_list(
        qr_result.get("all_qr_values", [])
        or qr_inner.get("raw_values", [])
        or direct_qr_results
        or url_candidates
    )

    return direct_qr_results, url_candidates, all_qr_values, qr_inner


def extract_url_candidates_from_text(text):
    if not text:
        return []

    normalized_text = str(text)
    normalized_text = normalized_text.replace("\n", " ").replace("\t", " ")
    normalized_text = re.sub(r"www\s*\.\s*", "www.", normalized_text, flags=re.IGNORECASE)
    normalized_text = re.sub(r"(https?)\s*:\s*/\s*/", r"\1://", normalized_text, flags=re.IGNORECASE)
    normalized_text = re.sub(
        r"([a-zA-Z0-9])\s*\.\s*(com|co\.kr|kr|net|org|io|ai|shop|mall|beauty|me|app|store|global)",
        r"\1.\2", normalized_text, flags=re.IGNORECASE
    )

    url_patterns = [
        r"https?://[^\s]+",
        r"www\.[^\s]+",
        r"[a-zA-Z0-9.-]+\.(?:com|co\.kr|kr|net|org|io|ai|shop|mall|beauty|me|app|store|global)[^\s]*"
    ]

    results = []
    for pattern in url_patterns:
        matches = re.findall(pattern, normalized_text, flags=re.IGNORECASE)
        results.extend(matches)

    cleaned = [str(item).strip(" ,.;:：()[]{}<>") for item in results if item]
    return normalize_list(cleaned)


def normalize_url_for_result(value):
    if not value:
        return ""

    value = str(value).strip().replace(" ", "")

    if value.startswith("http://") or value.startswith("https://"):
        return value

    if value.startswith("www."):
        return "https://" + value

    if re.search(
        r"^[a-zA-Z0-9][a-zA-Z0-9.-]*\.(com|co\.kr|kr|net|org|io|ai|shop|mall|beauty|me|app|store|global)",
        value, flags=re.IGNORECASE
    ):
        return "https://" + value

    return value


def build_basic_qr_result(qr_analysis):
    if not isinstance(qr_analysis, dict):
        qr_analysis = {}

    url_candidates = normalize_list(
        qr_analysis.get("url_candidates", [])
        or qr_analysis.get("analysis_ready_urls", [])
        or qr_analysis.get("urls", [])
    )

    qr_codes = normalize_list(
        qr_analysis.get("qr_codes", [])
        or qr_analysis.get("direct_qr_results", [])
        or qr_analysis.get("ocr_qr_codes", [])
    )

    urls = []
    for value in url_candidates + qr_codes:
        if re.search(r"https?://|www\.|\.com|\.co\.kr|\.kr|\.net|\.org|\.io|\.ai", value, flags=re.IGNORECASE):
            urls.append(normalize_url_for_result(value))

    urls = normalize_list(urls)
    detected = bool(qr_codes or urls)

    return {
        "detected": detected,
        "url": urls[0] if urls else "확인 불가",
        "urls": urls,
        "raw_values": normalize_list(qr_codes + urls),
        "status": "detected" if detected else "not_detected"
    }


def run_qr_analysis(image_paths, ocr_text, qr_reader, qr_analyzer):
    all_direct_qr_results = []
    all_url_candidates = []
    all_qr_values = []
    qr_reader_inner_results = []

    for image_path in image_paths:
        if qr_reader is None:
            continue

        try:
            print_sub_step(f"QR 직접 인식: {os.path.basename(image_path)}")

            qr_result = qr_reader.read_all(
                image_path=image_path,
                ocr_text=ocr_text
            )

            direct_qr_results, url_candidates, all_values, qr_inner = extract_qr_result_fields(qr_result)

            all_direct_qr_results.extend(direct_qr_results)
            all_url_candidates.extend(url_candidates)
            all_qr_values.extend(all_values)

            if qr_inner:
                qr_reader_inner_results.append(qr_inner)

            print_done(f"QR 직접 인식 완료: {os.path.basename(image_path)}")

        except Exception as error:
            print_warning(f"QR 인식 실패: {os.path.basename(image_path)} / {error}")

    ocr_url_candidates = extract_url_candidates_from_text(ocr_text)

    qr_codes = normalize_list(
        all_direct_qr_results + all_url_candidates + all_qr_values + ocr_url_candidates
    )

    url_candidates = normalize_list(all_url_candidates + ocr_url_candidates)

    qr_analysis_base = {
        "qr_codes": qr_codes if qr_codes else ["확인 불가"],
        "ocr_qr_codes": ocr_url_candidates,
        "direct_qr_results": normalize_list(all_direct_qr_results),
        "url_candidates": url_candidates,
        "analysis_ready_urls": url_candidates,
        "qr_reader_inner_results": qr_reader_inner_results,
        "analysis_status": "ready" if url_candidates else "no_url_to_analyze",
        "analysis_note": "QR/URL 정보는 제품 정보 확인 및 분석 보강용으로 사용"
    }

    if qr_codes and qr_analyzer is not None:
        try:
            print_sub_step("QR/URL 내용 분석 보강")

            qr_analysis = qr_analyzer.analyze(
                qr_analysis_base=qr_analysis_base,
                qr_values=qr_codes
            )

            if isinstance(qr_analysis, dict):
                merged = dict(qr_analysis_base)
                merged.update(qr_analysis)
                print_done("QR/URL 분석 보강 완료")
                return merged

        except Exception as error:
            print_warning(f"QR/URL 분석 보강 실패: {error}")

    return qr_analysis_base


# =========================================================
# 성분 API 검증 관련 함수
# =========================================================

def build_empty_api_validation(status="not_run", error_message=None, ingredient_candidates=None):
    ingredient_candidates = normalize_list(ingredient_candidates or [])

    unverified_ingredients = [
        {
            "ocr_name": item,
            "query_name": item,
            "reason": status,
            "similarity": 0
        }
        for item in ingredient_candidates
    ]

    result = {
        "status": status,

        # [수정] postprocess.py 새 구조 필드명 추가
        "ingredients_verified": [],

        # 하위 호환
        "ingredients": [],
        "ingredients_after_api": [],
        "verified_ingredient_names": [],
        "verified_ingredients": [],
        "unverified_ingredients": unverified_ingredients,
        "api_success_results": [],
        "api_failed_results": [],
        "api_all_results": [],
        "verified_count": 0,
        "unverified_count": len(unverified_ingredients),
        "total_checked_count": len(ingredient_candidates)
    }

    if error_message:
        result["error"] = error_message

    return result


def run_ingredient_api_validation(ingredient_candidates):
    """
    공공데이터포털 성분 API 검증.

    중요:
    - 성분 API는 제품명/용량/사용방법/주의사항에는 쓰지 않는다.
    - ingredients_raw에 대해서만 검증한다.
    """

    clean_candidates = normalize_list(ingredient_candidates)

    if not clean_candidates:
        print_warning("API 검증 가능한 성분 후보 없음")
        return build_empty_api_validation(
            status="no_ingredient_candidates",
            ingredient_candidates=[]
        )

    try:
        print_sub_step("IngredientAPI 객체 생성")
        ingredient_api = IngredientAPI()
        print_done("IngredientAPI 객체 생성 완료")

        print_sub_step(f"성분 후보 {len(clean_candidates)}개 API 검증 수행")
        api_result = ingredient_api.match_ingredients(clean_candidates)
        print_done("성분 API 검증 완료")

        if not isinstance(api_result, dict):
            print_warning("IngredientAPI 결과가 dict 형식이 아님")
            api_result = {}

        # [수정] ingredients_verified 우선 참조
        verified_ingredient_names = normalize_list(
            api_result.get("ingredients_verified", [])
            or api_result.get("ingredients_after_api", [])
            or api_result.get("verified_ingredient_names", [])
            or api_result.get("ingredients", [])
        )

        verified_ingredients = (
            api_result.get("verified_ingredients", [])
            if isinstance(api_result.get("verified_ingredients", []), list) else []
        )

        unverified_ingredients = (
            api_result.get("unverified_ingredients", [])
            if isinstance(api_result.get("unverified_ingredients", []), list) else []
        )

        api_success_results = (
            api_result.get("api_success_results", [])
            if isinstance(api_result.get("api_success_results", []), list) else []
        )

        api_failed_results = (
            api_result.get("api_failed_results", [])
            if isinstance(api_result.get("api_failed_results", []), list) else []
        )

        api_all_results = (
            api_result.get("api_all_results", [])
            if isinstance(api_result.get("api_all_results", []), list) else []
        )

        if not api_all_results:
            api_all_results = api_success_results + api_failed_results

        status = api_result.get("status", "success")

        return {
            "status": status,

            # [수정] postprocess.py 새 구조 필드명 추가
            "ingredients_verified": verified_ingredient_names,

            # 하위 호환
            "ingredients": verified_ingredient_names,
            "ingredients_after_api": verified_ingredient_names,
            "verified_ingredient_names": verified_ingredient_names,

            "verified_ingredients": verified_ingredients,
            "unverified_ingredients": unverified_ingredients,
            "api_success_results": api_success_results,
            "api_failed_results": api_failed_results,
            "api_all_results": api_all_results,
            "verified_count": len(verified_ingredient_names),
            "unverified_count": len(unverified_ingredients),
            "total_checked_count": len(api_all_results) if api_all_results else len(clean_candidates)
        }

    except Exception as error:
        print_error(f"성분 API 검증 실패: {error}")
        traceback.print_exc()

        return build_empty_api_validation(
            status="api_error",
            error_message=str(error),
            ingredient_candidates=clean_candidates
        )


# =========================================================
# 이미지별 OCR → 구간탐지 → 후처리
# =========================================================

def analyze_single_image(
    image_path,
    image_index,
    total_count,
    ocr_runner,
    section_detector,
    postprocessor
):
    """
    단일 이미지 분석

    흐름:
    1. 전체 이미지 OCR
    2. OCR bbox / line 기반 구간 탐지
    3. 후처리
    """

    single_result = {
        "image_index": image_index,
        "image_path": image_path,
        "ocr_result": {},
        "section_result": {},
        "postprocessed_result": {},
        "success": False,
        "error": None
    }

    basename = os.path.basename(image_path)

    try:
        # OCR
        print_sub_step(f"[{image_index}/{total_count}] OCR 수행: {basename}")
        t0 = time.time()

        ocr_result = ocr_runner.run(image_path=image_path, save_text=True)

        if not isinstance(ocr_result, dict) or not ocr_result.get("success"):
            raise RuntimeError("OCR 결과가 비어 있거나 실패했습니다.")

        safe_save("save_ocr_result", ocr_result)

        ocr_secs = time.time() - t0
        line_count = len(ocr_result.get("ocr_lines", []))
        block_count = len(ocr_result.get("ocr_blocks", []))
        variant = ocr_result.get("selected_variant", "-")
        print_done(
            f"OCR 완료 — {line_count}줄 / {block_count}블록 / variant={variant} "
            + _dim(f"({ocr_secs:.1f}s)")
        )

        # 구간 탐지
        print_sub_step(f"[{image_index}/{total_count}] 구간 탐지: {basename}")
        t1 = time.time()

        section_result = section_detector.detect(ocr_result)

        if not isinstance(section_result, dict):
            raise RuntimeError("section_detector 결과가 dict 형식이 아닙니다.")

        safe_save("save_section_detection_result", section_result)

        det_types = section_result.get("section_detection_summary", {}).get("detected_section_types", [])
        det_secs = time.time() - t1
        print_done(
            f"구간 탐지 완료 — 탐지된 구간: {det_types} "
            + _dim(f"({det_secs:.1f}s)")
        )

        # 후처리
        print_sub_step(f"[{image_index}/{total_count}] 후처리: {basename}")
        t2 = time.time()

        postprocessed_result = postprocessor.process(section_result)

        if not isinstance(postprocessed_result, dict):
            raise RuntimeError("postprocessor 결과가 dict 형식이 아닙니다.")

        safe_save("save_postprocessed_result", postprocessed_result)

        raw_count = len(normalize_list(postprocessed_result.get("ingredients_raw", [])))
        pp_secs = time.time() - t2
        print_done(
            f"후처리 완료 — 성분 후보 {raw_count}개 "
            + _dim(f"({pp_secs:.1f}s)")
        )

        single_result.update(
            {
                "ocr_result": ocr_result,
                "section_result": section_result,
                "postprocessed_result": postprocessed_result,
                "success": True
            }
        )

    except Exception as error:
        print_error(f"이미지 분석 실패: {basename} / {error}")
        traceback.print_exc()
        single_result["error"] = str(error)

    return single_result


# =========================================================
# 카테고리 cross-routing — 인라인 라벨 재분류
# =========================================================

# usage 항목 안의 effects/cautions 인라인 라벨 식별
_INLINE_LABELS_TO_SECTION = [
    ("효능효과", "effects"),
    ("효능 효과", "effects"),
    ("주요기능", "effects"),
    ("기능성 화장품", "effects"),
    ("주요특징", "effects"),
    ("제품특징", "effects"),

    ("사용방법", "usage"),
    ("사용 방법", "usage"),
    ("사용순서", "usage"),
    ("사용 순서", "usage"),
    ("용법용량", "usage"),
    ("용법", "usage"),
    ("사용법", "usage"),

    ("사용상의 주의사항", "cautions"),
    ("사용할 때의 주의사항", "cautions"),
    ("사용시의 주의사항", "cautions"),
    ("사용시 주의사항", "cautions"),
    ("사용상주의사항", "cautions"),
    ("주의사항", "cautions"),
    ("경고", "cautions"),
]


def _split_by_inline_labels(text):
    """
    한 텍스트 안의 인라인 라벨 위치를 찾아 각 청크를 (section, body) 쌍의 list로 반환.
    라벨 앞 prefix는 입력 섹션(default_section)으로 간주된다 (외부 처리).
    """
    if not text:
        return []

    flat = re.sub(r"\s+", " ", str(text)).strip()
    if not flat:
        return []

    matches = []
    for label, section in _INLINE_LABELS_TO_SECTION:
        for m in re.finditer(re.escape(label), flat, flags=re.IGNORECASE):
            matches.append((m.start(), m.end(), section, label))

    if not matches:
        return [(None, flat)]

    # 같은 위치는 더 긴 라벨이 우선
    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # 겹치는 매칭 제거
    filtered = []
    last_end = -1
    for start, end, section, label in matches:
        if start < last_end:
            continue
        filtered.append((start, end, section, label))
        last_end = end

    chunks = []
    prefix = flat[: filtered[0][0]].strip(" :,.;·ㆍ[]()\n\t")
    if prefix:
        chunks.append((None, prefix))  # 호출자가 default section으로 분류

    for i, (start, end, section, label) in enumerate(filtered):
        body_start = end
        body_end = filtered[i + 1][0] if (i + 1) < len(filtered) else len(flat)
        body = flat[body_start:body_end].strip(" :,.;·ㆍ[]()\n\t")
        if body:
            chunks.append((section, body))

    return chunks


def _split_cautions_into_numbered_items(text):
    """
    주의사항 텍스트를 번호 매김 기준으로 항목 분리.

    지원 패턴:
    - 1. 2) 3.    (아라비아 숫자 + . 또는 ))
    - 가) 나) 다)  (한글 + ))
    - ① ② ③      (원숫자)
    """
    if not text:
        return []

    flat = re.sub(r"\s+", " ", str(text)).strip()
    if not flat:
        return []

    numbering_pattern = re.compile(
        r"(?:(?<=\s)|(?<=^))"
        r"(?:"
        r"\d{1,2}\s*[.\)]\s*"
        r"|"
        r"[가-힣]\s*\)\s*"
        r"|"
        r"[①-⑳㈀-㈎]\s*"
        r")"
    )

    splits = []
    last_idx = 0
    for m in numbering_pattern.finditer(flat):
        if m.start() > last_idx:
            splits.append(flat[last_idx:m.start()].strip())
        last_idx = m.start()
    splits.append(flat[last_idx:].strip())

    results = []
    for piece in splits:
        piece = piece.strip(" :,.;·ㆍ[]()\n\t")
        if not piece:
            continue
        if len(piece) < 4:
            continue
        results.append(piece)

    if len(results) <= 1:
        return []

    return results


# =========================================================
# 카테고리 list 잡음 정제 (잡음 A/B/C)
# =========================================================
#
# 일반 패턴만 사용 (특정 화장품·성분·브랜드·제품 카피 어휘 의존 금지).
# 한국어 라벨의 구조적 패턴(기호 prefix, 어미, 어절 수,
# '함유'/'ppm' 같은 일반 라벨 키워드)만 사용한다.
#
# A. usage list에 효능/광고 prefix 잔존
#    - 특수 기호로 시작 + 길이 ≤ 15 + 동사 종결 어미 없음
#    - → effects로 라우팅
#
# B. cautions 항목 끝에 성분 광고 어구 매달림
#    - [+*] + 짧은 어구 + (ppm 표기) + '함유'
#    - → 해당 sub-string 분리하여 effects로 이동, 원 항목은 잘린 형태 유지
#
# C. 번호 매김 분리 후 짧은 헤더 잔존
#    - 어절 < 3 + 길이 < 10 + 동사 종결 어미 없음
#    - → 폐기

# 항목 시작부의 광고/효능 prefix 기호.
# (번호 매김 '1.', '가)', '①' 등은 정상 항목이므로 제외)
_PREFIX_NOISE_SYMBOLS = "[*+★※◆◇■□●○▶▷►"

# 한국어 종결 어미 — 명령/평서/문어체 종결 기준.
# 의미: 이 패턴이 텍스트에 나타나면 '완결된 문장'으로 본다.
_VERB_ENDING_PATTERN = re.compile(
    r"("
    r"세요|십시오|시오|하세요|마세요"
    r"|합니다|입니다|됩니다|있습니다|없습니다"
    r"|한다|된다|있다|없다|이다"
    r"|하십시오|주십시오"
    r"|바른다|바릅니다|바르세요"
    r"|사용한다|사용하십시오|사용하세요|사용합니다"
    r"|주의|금지|중단|보관|섭취|문의|상담"
    r")(?:[.!?]|\s*$)"
)

# 매달린 성분 광고 어구 패턴.
# 예) +병풀단백질추출물 함유, *활성-XXX (30,000 ppm) 함유
# - [+*] 또는 공백 후 [+*]
# - 짧은 어구 (한글/영문/숫자/하이픈/어포스트로피)
# - 선택적 (xx ppm) 표기
# - 끝에 '함유'
_TRAILING_AD_PATTERN = re.compile(
    r"\s*[+*]\s*"
    r"[가-힣A-Za-z0-9\-\'\u2018\u2019]"
    r"[가-힣A-Za-z0-9\-\'\u2018\u2019\s,]*?"
    r"(?:\(\s*\d[\d,\s]*\s*ppm\s*\))?"
    r"\s*함유"
    r"\s*\.?\s*$",
    re.IGNORECASE,
)


def _count_eojeol(text):
    """공백 기준 어절 수."""
    if not text:
        return 0
    return len([w for w in str(text).split() if w.strip()])


def _has_verb_ending(text):
    """한국어 종결 어미 존재 여부."""
    if not text:
        return False
    return bool(_VERB_ENDING_PATTERN.search(str(text)))


def _starts_with_noise_symbol(text):
    """특수 기호 prefix로 시작하는지."""
    if not text:
        return False
    stripped = str(text).lstrip()
    if not stripped:
        return False
    return stripped[0] in _PREFIX_NOISE_SYMBOLS


def _is_usage_noise_fragment(text):
    """
    잡음 A 판정.
    - 특수 기호로 시작
    - 길이 ≤ 15
    - 동사 종결 어미 없음
    """
    if not text:
        return False
    flat = str(text).strip()
    if not _starts_with_noise_symbol(flat):
        return False
    if len(flat) > 15:
        return False
    if _has_verb_ending(flat):
        return False
    return True


def _is_short_header_fragment(text):
    """
    잡음 C 판정.
    - 어절 수 < 3
    - 길이 < 10
    - 동사 종결 어미 없음
    """
    if not text:
        return True
    flat = str(text).strip()
    if not flat:
        return True
    if _has_verb_ending(flat):
        return False
    if _count_eojeol(flat) >= 3:
        return False
    if len(flat) >= 10:
        return False
    return True


def _strip_trailing_ad_phrase(text):
    """
    잡음 B 처리.
    항목 끝에 매달린 [+*]<어구>(<ppm>) 함유 패턴을 잘라낸다.

    반환: (정제된 본문, 잘라낸 광고 어구 또는 None)
    """
    if not text:
        return text, None

    flat = str(text)
    match = _TRAILING_AD_PATTERN.search(flat)
    if not match:
        return flat, None

    stripped_body = flat[: match.start()].rstrip(" ,;·ㆍ")
    ad_phrase = flat[match.start():].strip(" .")
    # 본문이 너무 짧아지면 통째로 effects로 옮기는 게 안전
    if len(stripped_body) < 4:
        return "", ad_phrase
    return stripped_body, ad_phrase


def _clean_section_items(usage_items, caution_items, effects_items):
    """
    cross_route_sections 출력 단계의 마지막 정제.

    수행 작업:
      A. usage 항목 중 특수 기호 prefix + 단축 명사구 토막 → effects로 이동
      B. cautions 항목 끝의 [+*]<어구> 함유 → 분리하여 effects로 이동
      C. 어절 < 3 + 길이 < 10 + 동사 종결 없음 항목 → 폐기

    제품명·성분·QR·effects 본문에는 손대지 않는다.
    """
    cleaned_usage = []
    cleaned_cautions = []
    cleaned_effects = list(effects_items or [])

    # ─── A. usage 정제 ──────────────────────────────────
    for item in usage_items or []:
        if not item:
            continue
        text = str(item).strip()
        if not text:
            continue

        if _is_usage_noise_fragment(text):
            # 광고/효능 prefix → effects로 이동
            cleaned_effects.append(text)
            continue

        if _is_short_header_fragment(text):
            # C. 무의미한 짧은 헤더 → 폐기
            continue

        cleaned_usage.append(text)

    # ─── B. cautions 정제 ───────────────────────────────
    for item in caution_items or []:
        if not item:
            continue
        text = str(item).strip()
        if not text:
            continue

        body, ad_phrase = _strip_trailing_ad_phrase(text)

        if ad_phrase:
            cleaned_effects.append(ad_phrase)
            if not body:
                # 항목 자체가 광고 어구뿐이면 cautions에서 제거
                continue
            text = body

        if _is_short_header_fragment(text):
            # C. 무의미한 짧은 헤더 → 폐기
            continue

        cleaned_cautions.append(text)

    # ─── C. effects 정제 (짧은 헤더 토막만 제거) ───────
    deduped_effects = []
    for item in cleaned_effects:
        if not item:
            continue
        text = str(item).strip()
        if not text:
            continue
        if _is_short_header_fragment(text):
            continue
        deduped_effects.append(text)

    return (
        normalize_list(cleaned_usage),
        normalize_list(cleaned_cautions),
        normalize_list(deduped_effects),
    )


def cross_route_sections(usage_items, caution_items, effects_items):
    """
    카테고리 정확도 강화:

    1. usage 항목 안에 effects/cautions 인라인 라벨이 있으면 적절한 섹션으로 이동
    2. effects 항목 안에 usage/cautions 인라인 라벨이 있으면 적절한 섹션으로 이동
    3. cautions 항목 안에 usage/effects 인라인 라벨이 있으면 적절한 섹션으로 이동
    4. cautions는 번호 매김 기준으로 추가 분리
    5. 라우팅 후 잔존 잡음(A/B/C) 정제 — _clean_section_items

    각 라벨 청크는 해당 섹션 list에 개별 항목으로 추가된다 (한 줄 합치기 X).
    """
    usage_items = normalize_list(usage_items)
    caution_items = normalize_list(caution_items)
    effects_items = normalize_list(effects_items)

    out_usage = []
    out_cautions = []
    out_effects = []

    def _route(items, default_section):
        for item in items:
            chunks = _split_by_inline_labels(item)
            if not chunks:
                continue

            for section, body in chunks:
                target = section or default_section
                if not body:
                    continue
                if target == "usage":
                    out_usage.append(body)
                elif target == "cautions":
                    out_cautions.append(body)
                elif target == "effects":
                    out_effects.append(body)

    _route(usage_items, "usage")
    _route(caution_items, "cautions")
    _route(effects_items, "effects")

    # cautions는 번호 매김으로 한번 더 분리
    expanded_cautions = []
    for item in out_cautions:
        numbered = _split_cautions_into_numbered_items(item)
        if numbered:
            expanded_cautions.extend(numbered)
        else:
            expanded_cautions.append(item)

    # [강화] 라우팅 후 잔존 잡음 정제 (A: usage prefix, B: cautions trailing,
    # C: short header fragments). 일반 패턴만 사용.
    return _clean_section_items(
        normalize_list(out_usage),
        normalize_list(expanded_cautions),
        normalize_list(out_effects),
    )


def merge_image_analysis_results(image_analysis_results):
    """
    여러 장 이미지 분석 결과 병합.

    [수정] postprocess.py 새 구조(ingredients_raw) 반영
    """

    successful_results = [
        item
        for item in image_analysis_results
        if isinstance(item, dict) and item.get("success")
    ]

    if not successful_results:
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
                "raw_section_text": {},
                "ingredient_section_text": "",
                "selected_variant": "",
                "section_detection_summary": {},
                "ocr_lines": [],
                "ocr_blocks": [],
                "detected_sections": []
            },
            "image_analysis_results": image_analysis_results
        }

    product_names = []
    capacities = []
    ingredients_raw = []
    usage_items = []
    caution_items = []
    effects_items = []
    qr_codes = []
    qr_urls = []

    ocr_confidences = []

    raw_texts = []
    layout_texts = []

    raw_section_texts = {
        "product_name": [],
        "capacity": [],
        "ingredients": [],
        "usage": [],
        "cautions": [],
        "effects": [],
        "qr_url": []
    }

    all_detected_sections = []
    all_ocr_lines = []
    all_ocr_blocks = []

    for item in successful_results:
        postprocessed = item.get("postprocessed_result", {}) or {}
        section_result = item.get("section_result", {}) or {}
        ocr_result = item.get("ocr_result", {}) or {}
        ocr_meta = postprocessed.get("ocr_meta", {}) or {}

        product_names.append(postprocessed.get("product_name"))
        capacities.append(postprocessed.get("capacity"))

        # [수정] ingredients_raw 참조
        ingredients_raw.extend(
            normalize_list(
                postprocessed.get("ingredients_raw", [])
            )
        )

        usage_items.extend(normalize_list(postprocessed.get("usage", [])))
        caution_items.extend(normalize_list(postprocessed.get("cautions", [])))
        effects_items.extend(normalize_list(postprocessed.get("effects", [])))

        qr_info = postprocessed.get("qr_info", {}) or {}
        qr_codes.extend(normalize_list(qr_info.get("qr_codes", [])))
        qr_urls.extend(normalize_list(qr_info.get("urls", [])))

        raw_texts.append(
            ocr_meta.get("raw_text") or ocr_result.get("raw_text", "")
        )
        layout_texts.append(
            ocr_meta.get("layout_text") or ocr_result.get("layout_text", "")
        )

        one_raw_section = ocr_meta.get("raw_section_text", {}) or {}
        for key in raw_section_texts:
            value = one_raw_section.get(key, "")
            if value:
                raw_section_texts[key].append(value)

        detected_sections = ocr_meta.get("detected_sections", [])
        if isinstance(detected_sections, list):
            all_detected_sections.extend(detected_sections)

        ocr_lines = ocr_meta.get("ocr_lines", []) or ocr_result.get("ocr_lines", [])
        ocr_blocks = ocr_meta.get("ocr_blocks", []) or ocr_result.get("ocr_blocks", [])

        if isinstance(ocr_lines, list):
            all_ocr_lines.extend(ocr_lines)

        if isinstance(ocr_blocks, list):
            all_ocr_blocks.extend(ocr_blocks)

        ocr_summary = ocr_result.get("ocr_summary", {}) or {}
        conf_value = ocr_summary.get("confidence_avg")
        if isinstance(conf_value, (int, float)) and conf_value > 0:
            ocr_confidences.append(float(conf_value))

    final_raw_section = {
        key: merge_text_values(values)
        for key, values in raw_section_texts.items()
    }

    ingredients_raw = normalize_list(ingredients_raw)
    qr_codes_clean = normalize_list(qr_codes)
    qr_urls_clean = normalize_list(qr_urls)

    ocr_confidence_avg = (
        sum(ocr_confidences) / len(ocr_confidences)
        if ocr_confidences
        else 0.0
    )

    # [강화] 카테고리 cross-routing
    # 한 항목 안에 다른 카테고리의 인라인 라벨이 끼어 있으면 재분류한다.
    # 예: usage 항목 안의 "효능효과 ..." 부분은 effects로 이동.
    usage_items, caution_items, effects_items = cross_route_sections(
        usage_items=usage_items,
        caution_items=caution_items,
        effects_items=effects_items
    )

    return {
        "product_name": choose_best_text(product_names),
        "capacity": choose_best_text(capacities),

        # [수정] 새 구조 필드명
        "ingredients_raw": ingredients_raw,
        "ingredients_verified": [],  # API 검증 후 채움

        "usage": safe_list(usage_items),
        "cautions": safe_list(caution_items),
        "effects": safe_list(effects_items),

        "ocr_confidence_avg": round(ocr_confidence_avg, 4),

        "qr_info": {
            "qr_codes": qr_codes_clean if qr_codes_clean else ["확인 불가"],
            "urls": qr_urls_clean if qr_urls_clean else ["확인 불가"]
        },

        "ocr_meta": {
            "raw_text": merge_text_values(raw_texts),
            "layout_text": merge_text_values(layout_texts),
            "raw_section_text": final_raw_section,
            "ingredient_section_text": final_raw_section.get("ingredients", ""),
            "selected_variant": "",
            "section_detection_summary": {
                "image_count": len(successful_results),
                "detected_section_count": len(all_detected_sections),
                "ocr_line_count": len(all_ocr_lines),
                "ocr_block_count": len(all_ocr_blocks)
            },
            "ocr_lines": all_ocr_lines,
            "ocr_blocks": all_ocr_blocks,
            "detected_sections": all_detected_sections
        },

        "image_analysis_results": image_analysis_results
    }


# =========================================================
# GPT 정밀 분류/추출 — 규칙 기반 분류 결과 대체
# =========================================================

def apply_gpt_extraction(merged, image_paths, gpt_extractor):
    """
    사진 + Google Vision raw 텍스트를 GPT(Vision)에 함께 보내
    제품명/용량/전성분/사용방법/주의사항/효능/QR·URL을 정밀 분류·추출한 뒤
    규칙 기반(section_detector + postprocess) 결과를 대체한다.

    실패 시 merged를 그대로 둔다(규칙 기반 결과 fallback).
    """

    if gpt_extractor is None:
        print_warning("GPTExtractor 미초기화 — 규칙 기반 결과 유지")
        return merged

    ocr_meta = merged.get("ocr_meta", {}) or {}
    raw_text = ocr_meta.get("raw_text", "") or ""
    layout_text = ocr_meta.get("layout_text", "") or ""

    print_sub_step(f"GPT 정밀 분류/추출 요청 (이미지 {len(image_paths)}장)")

    gpt_result = gpt_extractor.extract(
        image_paths=image_paths,
        vision_raw_text=raw_text,
        vision_layout_text=layout_text,
    )

    if not isinstance(gpt_result, dict) or not gpt_result.get("success"):
        error = gpt_result.get("error") if isinstance(gpt_result, dict) else "알 수 없음"
        print_warning(f"GPT 추출 실패 — 규칙 기반 결과 유지 ({error})")
        return merged

    # ── 규칙 기반 결과를 GPT 결과로 대체 ──────────────────
    merged["product_name"] = value_or_unknown(gpt_result.get("product_name"))
    merged["capacity"] = value_or_unknown(gpt_result.get("capacity"))

    merged["ingredients_raw"] = normalize_list(gpt_result.get("ingredients", []))

    merged["usage"] = safe_list(gpt_result.get("usage", []))
    merged["cautions"] = safe_list(gpt_result.get("cautions", []))
    merged["effects"] = safe_list(gpt_result.get("effects", []))

    # GPT가 읽은 QR/URL을 기존 qr_info에 합쳐 둔다 (5단계 QR 분석에서 추가 보강).
    existing_qr = merged.get("qr_info", {}) or {}
    merged["qr_info"] = {
        "qr_codes": normalize_list(
            list(existing_qr.get("qr_codes", []) or [])
            + gpt_result.get("qr_codes", [])
        ) or ["확인 불가"],
        "urls": normalize_list(
            list(existing_qr.get("urls", []) or [])
            + gpt_result.get("urls", [])
        ) or ["확인 불가"],
    }

    merged["gpt_extraction"] = {
        "model": getattr(gpt_extractor, "model", ""),
        "raw_response": gpt_result.get("raw_response", ""),
    }

    print_done(
        "GPT 정밀 분류/추출 완료 — "
        f"성분 {len(merged['ingredients_raw'])}개 / "
        f"사용방법 {len(safe_list(merged['usage']))}개 / "
        f"주의사항 {len(safe_list(merged['cautions']))}개"
    )

    return merged


# =========================================================
# 최종 JSON 생성
# =========================================================

def build_final_result(
    image_paths,
    image_analysis_results,
    merged_postprocessed_result,
    ingredient_api_validation,
    qr_analysis
):
    """
    DB 전송용 최종 JSON 생성.

    [수정] postprocess.py 새 구조에 맞게 재정렬:
    - ingredients_raw: API 검증 전 OCR 추출 성분
    - ingredients_verified: API 검증 후 확정 성분
    - qr_info: { qr_codes, urls }
    - ocr_meta: OCR 내부 데이터 (디버깅/저장용)
    """

    if not isinstance(merged_postprocessed_result, dict):
        merged_postprocessed_result = {}

    if not isinstance(ingredient_api_validation, dict):
        ingredient_api_validation = build_empty_api_validation("not_run")

    if not isinstance(qr_analysis, dict):
        qr_analysis = {}

    ingredients_raw = normalize_list(
        merged_postprocessed_result.get("ingredients_raw", [])
    )

    # [수정] ingredients_verified 우선 참조
    ingredients_verified = normalize_list(
        ingredient_api_validation.get("ingredients_verified", [])
        or ingredient_api_validation.get("ingredients_after_api", [])
        or ingredient_api_validation.get("verified_ingredient_names", [])
        or ingredient_api_validation.get("ingredients", [])
    )

    qr_result = build_basic_qr_result(qr_analysis)

    merged_qr_info = merged_postprocessed_result.get("qr_info", {}) or {}
    all_qr_codes = normalize_list(
        merged_qr_info.get("qr_codes", [])
        + qr_analysis.get("qr_codes", [])
        + qr_analysis.get("url_candidates", [])
    )
    all_qr_urls = normalize_list(
        merged_qr_info.get("urls", [])
        + qr_result.get("urls", [])
    )

    successful_image_count = len(
        [item for item in image_analysis_results
         if isinstance(item, dict) and item.get("success")]
    )

    ocr_meta = merged_postprocessed_result.get("ocr_meta", {}) or {}

    accuracy = compute_accuracy_block(
        ocr_confidence_avg=merged_postprocessed_result.get("ocr_confidence_avg", 0.0),
        ingredient_api_validation=ingredient_api_validation
    )

    result_inner = {
        # ── 핵심 필드 (postprocess.py 새 구조와 동일) ─────────────────
        "product_name": value_or_unknown(
            merged_postprocessed_result.get("product_name")
        ),
        "capacity": value_or_unknown(
            merged_postprocessed_result.get("capacity")
        ),

        # API 검증 전 OCR 추출 성분
        "ingredients_raw": ingredients_raw,

        # API 검증 후 확정 성분
        "ingredients_verified": ingredients_verified,

        "usage": safe_list(merged_postprocessed_result.get("usage", [])),
        "cautions": safe_list(merged_postprocessed_result.get("cautions", [])),
        "effects": safe_list(merged_postprocessed_result.get("effects", [])),

        "qr_info": {
            "qr_codes": all_qr_codes if all_qr_codes else ["확인 불가"],
            "urls": all_qr_urls if all_qr_urls else ["확인 불가"]
        },

        # ── OCR 내부 메타 ─────────────────────────────────────────────
        "ocr_meta": {
            "raw_text": ocr_meta.get("raw_text", ""),
            "layout_text": ocr_meta.get("layout_text", ""),
            "raw_section_text": ocr_meta.get("raw_section_text", {}),
            "ingredient_section_text": ocr_meta.get("ingredient_section_text", ""),
            "section_detection_summary": ocr_meta.get("section_detection_summary", {}),
            "input_image_paths": image_paths
        },

        # ── API 검증 상세 ─────────────────────────────────────────────
        "ingredient_api_validation": {
            "status": ingredient_api_validation.get("status", "unknown"),
            "verified_ingredient_names": ingredients_verified,
            "verified_ingredients": ingredient_api_validation.get("verified_ingredients", []),
            "unverified_ingredients": ingredient_api_validation.get("unverified_ingredients", []),
            "api_success_results": ingredient_api_validation.get("api_success_results", []),
            "api_failed_results": ingredient_api_validation.get("api_failed_results", []),
            "api_all_results": ingredient_api_validation.get("api_all_results", []),
            "verified_count": ingredient_api_validation.get("verified_count", 0),
            "unverified_count": ingredient_api_validation.get("unverified_count", 0),
            "total_checked_count": ingredient_api_validation.get("total_checked_count", 0)
        },

        # ── 처리 요약 ─────────────────────────────────────────────────
        "processing_summary": {
            "input_image_count": len(image_paths),
            "successful_image_analysis_count": successful_image_count,
            "ocr_line_count": len(ocr_meta.get("ocr_lines", [])),
            "ocr_block_count": len(ocr_meta.get("ocr_blocks", [])),
            "detected_section_count": len(ocr_meta.get("detected_sections", [])),
            "ingredient_raw_count": len(ingredients_raw),
            "ingredient_verified_count": ingredient_api_validation.get("verified_count", 0),
            "ingredient_unverified_count": ingredient_api_validation.get("unverified_count", 0)
        },

        # ── 디버그 ────────────────────────────────────────────────────
        "debug_info": {
            "pipeline": "full_image_ocr → section_detection → postprocess → qr_analysis → ingredient_api",
            "image_analysis_success": [
                {
                    "image_path": item.get("image_path"),
                    "success": item.get("success"),
                    "error": item.get("error")
                }
                for item in image_analysis_results
                if isinstance(item, dict)
            ]
        }
    }

    final_result = {
        "accuracy": accuracy,
        "success": True,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": "full_image_ocr → section_detection → postprocess → qr_analysis → ingredient_api → final_json",
        "result": result_inner
    }

    return final_result


def build_server_payload_from_final(final_result):
    """
    DB 전송용 payload 생성.

    [수정] 새 구조(ingredients_raw / ingredients_verified / qr_info) 반영
    """

    try:
        build_method = getattr(FileIO, "build_server_payload", None)

        if build_method:
            return build_method(final_result, include_debug=False)

    except Exception as error:
        print_warning(f"FileIO.build_server_payload 실패, 내부 생성 사용: {error}")

    result = final_result.get("result", {}) if isinstance(final_result, dict) else {}
    api_validation = result.get("ingredient_api_validation", {})
    qr_info = result.get("qr_info", {})

    accuracy = final_result.get("accuracy") if isinstance(final_result, dict) else None
    if not isinstance(accuracy, dict):
        accuracy = compute_accuracy_block(
            ocr_confidence_avg=result.get("ocr_meta", {}).get("confidence_avg", 0.0),
            ingredient_api_validation=api_validation
        )

    return {
        "accuracy": accuracy,
        "success": bool(final_result.get("success", True)),
        "analyzed_at": final_result.get(
            "analyzed_at",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ),
        "result": {
            "product_name": value_or_unknown(result.get("product_name")),
            "capacity": value_or_unknown(result.get("capacity")),

            "ingredients_raw": normalize_list(result.get("ingredients_raw", [])),
            "ingredients_verified": normalize_list(result.get("ingredients_verified", [])),

            # [강화] merge_fragmented_text 제거 — 항목 단위 정확도 우선
            # 사용방법/주의사항/효능은 분리된 list로 그대로 보낸다.
            "usage": safe_list(result.get("usage", [])),
            "cautions": safe_list(result.get("cautions", [])),
            "effects": safe_list(result.get("effects", [])),

            "qr_info": {
                "qr_codes": normalize_list(qr_info.get("qr_codes", [])),
                "urls": normalize_list(qr_info.get("urls", []))
            }
        },
        "ingredient_api_validation": {
            "status": api_validation.get("status", "unknown"),
            "verified_count": api_validation.get("verified_count", 0),
            "unverified_count": api_validation.get("unverified_count", 0),
            "total_checked_count": api_validation.get("total_checked_count", 0),
            "verified_ingredient_names": normalize_list(
                api_validation.get("verified_ingredient_names", [])
            ),
            "unverified_ingredients": api_validation.get("unverified_ingredients", [])
        }
    }


# =========================================================
# 최종 출력
# =========================================================

def print_final_summary(final_result):
    result = final_result.get("result", {})
    api_val = result.get("ingredient_api_validation", {})
    qr_info = result.get("qr_info", {})
    summary = result.get("processing_summary", {})
    accuracy = final_result.get("accuracy", {}) or {}

    print_section("최종 분석 결과 요약")

    # 정확도 박스
    print_summary_box("정확도", [
        ("OCR 신뢰도",     f"{accuracy.get('ocr_confidence', 0):.4f}"),
        ("성분 매칭률",     f"{accuracy.get('ingredient_match_rate', 0):.2f}%"),
        ("성분 검증 성공",  f"{accuracy.get('ingredient_verified_count', 0)} / {accuracy.get('ingredient_total_count', 0)}"),
    ])

    # 기본 정보 박스
    print_summary_box("제품 기본 정보", [
        ("제품명",   result.get("product_name", "확인 불가")),
        ("용량",     result.get("capacity", "확인 불가")),
        ("QR/URL",   qr_info.get("urls", ["확인 불가"])[0] if qr_info.get("urls") else "확인 불가"),
    ])

    # 성분 정보 박스
    raw_count = len(normalize_list(result.get("ingredients_raw", [])))
    ver_count = api_val.get("verified_count", 0)
    unver_count = api_val.get("unverified_count", 0)

    print_summary_box("성분 분석 결과", [
        ("OCR 추출 후보",   f"{raw_count}개"),
        ("API 검증 성공",   f"{ver_count}개"),
        ("API 검증 실패",   f"{unver_count}개"),
        ("API 상태",        api_val.get("status", "-")),
    ])

    # 처리 요약 박스
    print_summary_box("처리 요약", [
        ("입력 이미지",     f"{summary.get('input_image_count', 0)}개"),
        ("분석 성공",       f"{summary.get('successful_image_analysis_count', 0)}개"),
        ("OCR 줄 수",       f"{summary.get('ocr_line_count', 0)}개"),
        ("OCR 블록 수",     f"{summary.get('ocr_block_count', 0)}개"),
        ("탐지 구간 수",    f"{summary.get('detected_section_count', 0)}개"),
    ])

    # 사용방법 / 주의사항
    print()
    print(_bold("  사용방법:"))
    for item in safe_list(result.get("usage", [])):
        print_info(f"• {item}")

    print()
    print(_bold("  주의사항:"))
    for item in safe_list(result.get("cautions", [])):
        print_info(f"• {item}")

    # 성분 목록
    print()
    print(_bold(f"  OCR 추출 성분 ({raw_count}개):"))
    for i, ing in enumerate(normalize_list(result.get("ingredients_raw", [])), 1):
        print_info(f"{i:3d}. {ing}")

    print()
    print(_bold(f"  API 검증 성공 성분 ({ver_count}개):"))
    verified = normalize_list(result.get("ingredients_verified", []))
    if verified:
        for i, ing in enumerate(verified, 1):
            print(_green(f"       {i:3d}. {ing}"))
    else:
        print_info("없음")

    # DB 전송 JSON 구조 미리보기
    print_section("DB 전송 JSON 구조 미리보기")
    print_table_row("success",                     final_result.get("success"))
    print_table_row("analyzed_at",                 final_result.get("analyzed_at"))
    print_table_row("result.product_name",         result.get("product_name"))
    print_table_row("result.capacity",             result.get("capacity"))
    print_table_row("result.ingredients_raw",      f"{raw_count}개")
    print_table_row("result.ingredients_verified", f"{ver_count}개")
    print_table_row("result.usage",                f"{len(safe_list(result.get('usage', [])))}개")
    print_table_row("result.cautions",             f"{len(safe_list(result.get('cautions', [])))}개")
    print_table_row("result.qr_info.urls",         qr_info.get("urls", ["확인 불가"])[0] if qr_info.get("urls") else "확인 불가")
    print_table_row("ingredient_api_validation.status",   api_val.get("status"))
    print_table_row("ingredient_api_validation.verified", f"{ver_count}개 / {api_val.get('total_checked_count', 0)}개 검증")


# =========================================================
# main
# =========================================================

def main():
    """
    Dermalens OCR 전체 실행 흐름

    실행 순서:
    1단계  이미지 목록 설정
    2단계  공통 객체 초기화
    3단계  이미지별 OCR / 구간 탐지 / 후처리
    4단계  여러 이미지 결과 병합
    5단계  QR / URL 분석 보강
    6단계  성분 API 검증
    7단계  중간 분석 결과 저장
    8단계  최종 JSON 생성
    9단계  DB 전송용 payload 생성
    10단계 최종 결과 저장 + 요약 출력
    """

    _start = time.time()

    print()
    print(_bold("=" * 70))
    print(_bold(_cyan("  Dermalens OCR 화장품 정보 분석 시작")))
    print(_bold("=" * 70))

    # =====================================================
    # 1단계 - 분석할 이미지 목록
    # =====================================================

    print_section("이미지 목록 설정", step=1)

    image_paths = [
        "images/rnt1.jpg",
        #"images/sample1.2.jpg",
        # "images/sample2.jpg",
    ]

    valid_image_paths = []

    for index, path in enumerate(image_paths, start=1):
        if os.path.exists(path):
            print_done(f"{index}. {path}")
            valid_image_paths.append(path)
        else:
            print_warning(f"{index}. 파일 없음: {path}")

    image_paths = valid_image_paths

    if not image_paths:
        print_error("분석할 유효 이미지가 없습니다.")
        return

    print_info(f"총 {len(image_paths)}개 이미지 분석 예정")
    print_step_done(1, "이미지 목록 설정")

    # =====================================================
    # 2단계 - 공통 객체 초기화
    # =====================================================

    print_section("공통 분석 객체 초기화", step=2)

    try:
        print_sub_step("OCR 엔진 (PaddleOCR) 초기화")
        ocr_runner = OCRRunner()
        print_done("OCR 엔진 초기화 완료")
    except Exception as error:
        print_error(f"OCRRunner 초기화 실패: {error}")
        traceback.print_exc()
        return

    try:
        print_sub_step("구간 탐지 객체 생성")
        section_detector = OCRSectionDetector()
        print_done("OCRSectionDetector 생성 완료")
    except Exception as error:
        print_error(f"OCRSectionDetector 초기화 실패: {error}")
        traceback.print_exc()
        return

    try:
        print_sub_step("후처리 객체 생성")
        postprocessor = IngredientPostprocessor()
        print_done("IngredientPostprocessor 생성 완료")
    except Exception as error:
        print_error(f"IngredientPostprocessor 초기화 실패: {error}")
        traceback.print_exc()
        return

    try:
        print_sub_step("QR 리더 객체 생성")
        qr_reader = QRReader()
        print_done("QRReader 생성 완료")
    except Exception as error:
        print_warning(f"QRReader 초기화 실패 (QR 인식 생략): {error}")
        qr_reader = None

    try:
        print_sub_step("QR 분석기 객체 생성")
        qr_analyzer = QRAnalyzer()
        print_done("QRAnalyzer 생성 완료")
    except Exception as error:
        print_warning(f"QRAnalyzer 초기화 실패 (QR 분석 생략): {error}")
        qr_analyzer = None

    try:
        print_sub_step("GPT 정밀 추출기 객체 생성")
        gpt_extractor = GPTExtractor()
        print_done(f"GPTExtractor 생성 완료 (model={gpt_extractor.model})")
    except Exception as error:
        print_warning(f"GPTExtractor 초기화 실패 (규칙 기반 분류 사용): {error}")
        gpt_extractor = None

    print_step_done(2, "공통 객체 초기화")

    # =====================================================
    # 3단계 - 이미지별 OCR / 구간 탐지 / 후처리
    # =====================================================

    print_section("이미지별 OCR / 구간 탐지 / 후처리", step=3)

    image_analysis_results = []

    for image_index, image_path in enumerate(image_paths, start=1):
        result = analyze_single_image(
            image_path=image_path,
            image_index=image_index,
            total_count=len(image_paths),
            ocr_runner=ocr_runner,
            section_detector=section_detector,
            postprocessor=postprocessor
        )
        image_analysis_results.append(result)

    successful_results = [
        item for item in image_analysis_results
        if isinstance(item, dict) and item.get("success")
    ]

    print_info(f"성공: {len(successful_results)}개 / 실패: {len(image_paths) - len(successful_results)}개")

    if not successful_results:
        print_error("분석에 성공한 이미지가 없습니다.")
        return

    print_step_done(3, "이미지별 분석")

    # =====================================================
    # 4단계 - 여러 이미지 결과 병합
    # =====================================================

    print_section("이미지 결과 병합", step=4)
    print_sub_step(f"{len(successful_results)}개 이미지 결과 병합")

    merged = merge_image_analysis_results(image_analysis_results)

    raw_count = len(normalize_list(merged.get("ingredients_raw", [])))
    print_info(f"제품명: {merged.get('product_name')}")
    print_info(f"용량:   {merged.get('capacity')}")
    print_info(f"성분 후보: {raw_count}개")
    print_info(f"사용방법: {len(safe_list(merged.get('usage', [])))}개")
    print_info(f"주의사항: {len(safe_list(merged.get('cautions', [])))}개")

    print_step_done(4, "이미지 결과 병합")

    # =====================================================
    # 4.5단계 - GPT 정밀 분류/추출 (규칙 기반 분류 대체)
    # =====================================================

    print_section("GPT 정밀 분류/추출 (사진 + Vision raw)")

    merged = apply_gpt_extraction(
        merged=merged,
        image_paths=image_paths,
        gpt_extractor=gpt_extractor
    )

    raw_count = len(normalize_list(merged.get("ingredients_raw", [])))
    print_info(f"제품명: {merged.get('product_name')}")
    print_info(f"용량:   {merged.get('capacity')}")
    print_info(f"성분 후보: {raw_count}개")

    # =====================================================
    # 5단계 - QR / URL 분석 보강
    # =====================================================

    print_section("QR / URL 분석 보강", step=5)

    ocr_meta = merged.get("ocr_meta", {})
    raw_section = ocr_meta.get("raw_section_text", {})

    ocr_text_for_qr = "\n".join([
        str(raw_section.get("qr_url", "")),
        str(ocr_meta.get("raw_text", "")),
        str(raw_section.get("product_name", "")),
    ])

    qr_analysis = run_qr_analysis(
        image_paths=image_paths,
        ocr_text=ocr_text_for_qr,
        qr_reader=qr_reader,
        qr_analyzer=qr_analyzer
    )

    safe_save("save_qr_result", qr_analysis)
    qr_result = build_basic_qr_result(qr_analysis)
    print_info(f"QR 감지: {qr_result.get('detected')} / URL: {qr_result.get('url')}")

    print_step_done(5, "QR / URL 분석 보강")

    # =====================================================
    # 6단계 - 성분 API 검증
    # =====================================================

    print_section("성분 후보 API 검증", step=6)

    ingredients_raw = normalize_list(merged.get("ingredients_raw", []))

    print_info(f"API 검증 대상: {len(ingredients_raw)}개")

    ingredient_api_validation = run_ingredient_api_validation(ingredients_raw)

    # [수정] 새 필드명으로 병합 결과 업데이트
    ingredients_verified = normalize_list(
        ingredient_api_validation.get("ingredients_verified", [])
        or ingredient_api_validation.get("ingredients_after_api", [])
        or ingredient_api_validation.get("verified_ingredient_names", [])
    )

    merged["ingredients_verified"] = ingredients_verified

    # API 검증 결과 출력
    print()
    api_all = ingredient_api_validation.get("api_all_results", [])
    if api_all:
        success_cnt = ingredient_api_validation.get("verified_count", 0)
        fail_cnt = ingredient_api_validation.get("unverified_count", 0)
        print(_bold(f"    검증 결과: {success_cnt}개 성공 / {fail_cnt}개 실패"))
        print()

        for r in api_all:
            if not isinstance(r, dict):
                continue
            if r.get("matched"):
                print(_green(
                    f"    ✔ {r.get('ocr_name')!r:25s} → {r.get('matched_name_kr')}"
                    f"  ({r.get('similarity')})"
                ))
            else:
                print(_dim(
                    f"    ✘ {r.get('ocr_name')!r:25s} → 미매칭 ({r.get('reason')})"
                ))
    else:
        print_info("검증 결과 없음")

    safe_save("save_matched_ingredients", ingredient_api_validation)

    print_step_done(6, f"성분 API 검증 (성공 {ingredient_api_validation.get('verified_count',0)}개)")

    # =====================================================
    # 7단계 - 중간 분석 결과 저장
    # =====================================================

    print_section("중간 분석 결과 저장", step=7)

    full_analysis_result = {
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": "full_image_ocr → section_detection → postprocess → qr_analysis → ingredient_api",
        "input_image_paths": image_paths,
        "image_analysis_results": image_analysis_results,
        "merged_postprocessed_result": merged,
        "qr_analysis": qr_analysis,
        "ingredient_api_validation": ingredient_api_validation
    }

    for method, label in [
        ("save_full_analysis",       "전체 분석 JSON"),
        ("save_analysis_report",     "분석 리포트"),
        ("save_candidate_ingredients", "성분 후보"),
    ]:
        print_sub_step(f"{label} 저장")
        path = safe_save(method, full_analysis_result if method != "save_candidate_ingredients" else ingredients_raw)
        if path:
            print_info(f"→ {path}")

    print_step_done(7, "중간 결과 저장")

    # =====================================================
    # 8단계 - 최종 JSON 생성
    # =====================================================

    print_section("최종 DB 전송 JSON 생성", step=8)
    print_sub_step("최종 JSON 조립")

    final_result = build_final_result(
        image_paths=image_paths,
        image_analysis_results=image_analysis_results,
        merged_postprocessed_result=merged,
        ingredient_api_validation=ingredient_api_validation,
        qr_analysis=qr_analysis
    )

    print_done("최종 JSON 생성 완료")
    print_step_done(8, "최종 JSON 생성")

    # =====================================================
    # 9단계 - DB 전송용 payload 생성
    # =====================================================

    print_section("DB 전송용 payload 생성", step=9)
    print_sub_step("payload 조립")

    server_payload = build_server_payload_from_final(final_result)

    print_done("payload 생성 완료")
    print_step_done(9, "payload 생성")

    # =====================================================
    # 10단계 - 최종 결과 저장 + 요약 출력
    # =====================================================

    print_section("최종 결과 저장", step=10)

    for method, data, label in [
        ("save_final_result",  final_result,   "최종 결과 JSON"),
        ("save_server_payload", server_payload, "서버 payload"),
    ]:
        print_sub_step(f"{label} 저장")
        path = safe_save(method, data)
        if path:
            print_info(f"→ {path}")

    print_step_done(10, "최종 결과 저장")

    # ── 최종 요약 출력 ──────────────────────────────────
    print_final_summary(final_result)

    total_secs = time.time() - _start

    print()
    print(_bold("=" * 70))
    print(_bold(_green(f"  Dermalens OCR 분석 완료  (총 소요 시간: {total_secs:.1f}초)")))
    print(_bold("=" * 70))
    print()


if __name__ == "__main__":
    main()