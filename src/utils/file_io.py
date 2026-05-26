import os
import re
import json
from datetime import datetime


class FileIO:
    """
    Dermalens 파일 저장 유틸 클래스

    현재 파이프라인:
    이미지 입력
    → 전처리
    → 전체 이미지 OCR
    → OCR bbox / line 기반 구간 탐지
    → 후처리
    → QR/URL 분석
    → 성분 API 검증
    → 최종 JSON 저장
    → DB 전송용 payload 저장

    저장 대상:
    - raw OCR text
    - layout text
    - OCR result
    - section detection result
    - postprocess result
    - ingredient candidates
    - ingredient API result
    - QR result
    - final result
    - server payload
    - 사람이 읽기 쉬운 report
    """

    # =========================================================
    # 1. 기본 파일/폴더 처리
    # =========================================================

    @staticmethod
    def ensure_directory(path):
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def timestamp_filename(prefix, ext):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{now}.{ext}"

    @staticmethod
    def save_text(filepath, text):
        FileIO.ensure_directory(os.path.dirname(filepath))

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(str(text))

        return filepath

    @staticmethod
    def save_json(filepath, data):
        FileIO.ensure_directory(os.path.dirname(filepath))

        cleaned = FileIO.clean_for_json(data)

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(cleaned, file, ensure_ascii=False, indent=4)

        return filepath

    @staticmethod
    def load_json(filepath):
        if not os.path.exists(filepath):
            return None

        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)

    # =========================================================
    # 2. OCR / 구간 탐지 / 중간 텍스트 저장
    # =========================================================

    @staticmethod
    def save_raw_ocr_text(text):
        filename = FileIO.timestamp_filename("raw_text", "txt")
        filepath = os.path.join("outputs", "raw_text", filename)

        FileIO.save_text(filepath, text)
        return filepath

    @staticmethod
    def save_layout_text(text):
        filename = FileIO.timestamp_filename("layout_text", "txt")
        filepath = os.path.join("outputs", "layout_text", filename)

        FileIO.save_text(filepath, text)
        return filepath

    @staticmethod
    def save_ocr_result(ocr_result):
        filename = FileIO.timestamp_filename("ocr_result", "json")
        filepath = os.path.join("outputs", "ocr_result", filename)

        FileIO.save_json(filepath, ocr_result)
        return filepath

    @staticmethod
    def save_section_detection_result(section_result):
        filename = FileIO.timestamp_filename("section_detection_result", "json")
        filepath = os.path.join("outputs", "section_detection", filename)

        FileIO.save_json(filepath, section_result)
        return filepath

    @staticmethod
    def save_postprocessed_result(postprocessed_result):
        filename = FileIO.timestamp_filename("postprocessed_result", "json")
        filepath = os.path.join("outputs", "postprocess", filename)

        FileIO.save_json(filepath, postprocessed_result)
        return filepath

    @staticmethod
    def save_qr_result(qr_result):
        filename = FileIO.timestamp_filename("qr_result", "json")
        filepath = os.path.join("outputs", "qr", filename)

        FileIO.save_json(filepath, qr_result)
        return filepath

    @staticmethod
    def save_image_analysis_results(image_analysis_results):
        filename = FileIO.timestamp_filename("image_analysis_results", "json")
        filepath = os.path.join("outputs", "image_analysis", filename)

        FileIO.save_json(filepath, image_analysis_results)
        return filepath

    # =========================================================
    # 3. 이전 구조 호환용 저장 함수
    # =========================================================

    @staticmethod
    def save_section_ocr_result(section_ocr_result):
        """
        이전 crop 구간별 OCR 구조 호환용.
        현재 구조에서는 section_detection_result 사용 권장.
        """

        filename = FileIO.timestamp_filename("section_ocr_result", "json")
        filepath = os.path.join("outputs", "section_ocr", filename)

        FileIO.save_json(filepath, section_ocr_result)
        return filepath

    @staticmethod
    def save_claude_section_result(claude_section_result):
        """
        이전 Claude 구간 탐지 구조 호환용.
        현재 구조에서는 사용하지 않는다.
        """

        filename = FileIO.timestamp_filename("claude_section_result", "json")
        filepath = os.path.join("outputs", "legacy_claude_sections", filename)

        FileIO.save_json(filepath, claude_section_result)
        return filepath

    @staticmethod
    def save_crop_result(crop_result):
        """
        이전 SectionCropper 구조 호환용.
        현재 구조에서는 사용하지 않는다.
        """

        filename = FileIO.timestamp_filename("crop_result", "json")
        filepath = os.path.join("outputs", "legacy_sections", "_debug", filename)

        FileIO.save_json(filepath, crop_result)
        return filepath

    # =========================================================
    # 4. 성분 후보 / API 검증 결과 저장
    # =========================================================

    @staticmethod
    def save_candidate_ingredients(candidate_list):
        filename = FileIO.timestamp_filename("ingredient_candidates", "json")
        filepath = os.path.join("outputs", "candidates", filename)

        clean_candidates = FileIO.clean_list(candidate_list)

        data = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "candidate_count": len(clean_candidates),
            "ingredient_candidates": clean_candidates,
            "ingredients_before_api": clean_candidates
        }

        FileIO.save_json(filepath, data)
        return filepath

    @staticmethod
    def save_matched_ingredients(matched_results):
        """
        성분 API 검증 결과 저장.

        입력:
        - IngredientAPI.match_ingredients() 반환 dict
        또는
        - API raw result list
        """

        filename = FileIO.timestamp_filename("ingredient_matched", "json")
        filepath = os.path.join("outputs", "matched", filename)

        cleaned_input = FileIO.clean_for_json(matched_results)

        if isinstance(cleaned_input, dict):
            status = cleaned_input.get("status", "unknown")
            all_results = cleaned_input.get("api_all_results", [])
            success_results = cleaned_input.get("api_success_results", [])
            failed_results = cleaned_input.get("api_failed_results", [])

            normalized_ingredients = (
                cleaned_input.get("verified_ingredient_names", [])
                or cleaned_input.get("ingredients_after_api", [])
                or cleaned_input.get("ingredients", [])
                or FileIO.extract_normalized_ingredients(all_results)
            )

            verified_ingredients = cleaned_input.get("verified_ingredients", [])
            unverified_ingredients = cleaned_input.get("unverified_ingredients", [])

        else:
            status = "success"
            all_results = cleaned_input if isinstance(cleaned_input, list) else []

            success_results = [
                item
                for item in all_results
                if isinstance(item, dict) and item.get("matched")
            ]

            failed_results = [
                item
                for item in all_results
                if isinstance(item, dict) and not item.get("matched")
            ]

            normalized_ingredients = FileIO.extract_normalized_ingredients(all_results)

            verified_ingredients = [
                {
                    "ocr_name": item.get("ocr_name"),
                    "query_name": item.get("query_name"),
                    "matched_name_kr": item.get("matched_name_kr"),
                    "matched_name_en": item.get("matched_name_en"),
                    "cas_no": item.get("cas_no"),
                    "definition": item.get("definition"),
                    "similarity": item.get("similarity"),
                    "source": item.get("source")
                }
                for item in success_results
                if isinstance(item, dict)
            ]

            unverified_ingredients = [
                {
                    "ocr_name": item.get("ocr_name"),
                    "query_name": item.get("query_name"),
                    "reason": item.get("reason"),
                    "similarity": item.get("similarity")
                }
                for item in failed_results
                if isinstance(item, dict)
            ]

        data = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "total_count": len(all_results),
            "success_count": len(success_results),
            "failed_count": len(failed_results),
            "normalized_count": len(normalized_ingredients),

            "ingredients_after_api": FileIO.clean_list(normalized_ingredients),
            "verified_ingredient_names": FileIO.clean_list(normalized_ingredients),
            "normalized_ingredients": FileIO.clean_list(normalized_ingredients),

            "verified_ingredients": verified_ingredients,
            "unverified_ingredients": unverified_ingredients,

            "api_success_results": success_results,
            "api_failed_results": failed_results,
            "api_all_results": all_results,

            # 기존 코드 호환
            "success_results": success_results,
            "failed_results": failed_results,
            "all_results": all_results
        }

        FileIO.save_json(filepath, data)
        return filepath

    # =========================================================
    # 5. 전체 분석 / 최종 결과 저장
    # =========================================================

    @staticmethod
    def save_full_analysis(result_dict):
        filename = FileIO.timestamp_filename("full_analysis", "json")
        filepath = os.path.join("outputs", "analysis", filename)

        FileIO.save_json(filepath, result_dict)
        return filepath

    @staticmethod
    def save_final_result(final_result):
        filename = FileIO.timestamp_filename("final_result", "json")
        filepath = os.path.join("outputs", "final", filename)

        FileIO.save_json(filepath, final_result)
        return filepath

    @staticmethod
    def save_server_payload(server_payload):
        filename = FileIO.timestamp_filename("server_payload", "json")
        filepath = os.path.join("outputs", "server_payload", filename)

        FileIO.save_json(filepath, server_payload)
        return filepath

    # =========================================================
    # 6. DB 전송용 payload 생성 보조
    # =========================================================

    @staticmethod
    def build_server_payload(
        final_result,
        include_debug=False
    ):
        """
        DB 또는 백엔드로 넘기기 좋은 payload 구조 생성.

        핵심:
        - 성분 API 검증 전/후를 둘 다 보낸다.
        - OCR raw/debug 정보는 include_debug=True일 때만 포함한다.
        """

        if not isinstance(final_result, dict):
            final_result = {}

        result_data = final_result.get("result", final_result)

        ingredient_api_validation = (
            result_data.get("ingredient_api_validation", {})
            or final_result.get("ingredient_api_validation", {})
        )

        qr_data = (
            result_data.get("qr", {})
            or result_data.get("qr_result", {})
            or final_result.get("qr_result", {})
            or {}
        )

        # [강화] postprocess의 qr_info 키도 fallback으로 인식한다.
        # main.py의 build_final_result는 result.qr_info 구조로 만들어
        # 기존 코드의 result.qr 매칭에서 빠지던 문제가 있었음.
        if not qr_data:
            qr_info = result_data.get("qr_info", {}) or {}
            urls_list = FileIO.clean_list(qr_info.get("urls", []))
            qr_codes_list = FileIO.clean_list(qr_info.get("qr_codes", []))
            if urls_list or qr_codes_list:
                qr_data = {
                    "detected": bool(urls_list or qr_codes_list),
                    "url": urls_list[0] if urls_list else "확인 불가",
                    "urls": urls_list,
                    "raw_values": FileIO.clean_list(qr_codes_list + urls_list)
                }

        ingredients_before_api = (
            result_data.get("ingredients_before_api", [])
            or result_data.get("ingredient_candidates", [])
            or result_data.get("ingredients", [])
        )

        ingredients_after_api = (
            result_data.get("ingredients_after_api", [])
            or ingredient_api_validation.get("ingredients_after_api", [])
            or ingredient_api_validation.get("verified_ingredient_names", [])
            or ingredient_api_validation.get("ingredients", [])
        )

        accuracy = final_result.get("accuracy") if isinstance(final_result, dict) else None
        if not isinstance(accuracy, dict):
            ocr_meta = result_data.get("ocr_meta", {}) or {}
            accuracy = FileIO.compute_accuracy_block(
                ocr_confidence_avg=(
                    result_data.get("ocr_confidence_avg")
                    or ocr_meta.get("confidence_avg", 0.0)
                ),
                ingredient_api_validation=ingredient_api_validation
            )

        payload = {
            "accuracy": accuracy,
            "success": bool(final_result.get("success", True)),
            "analyzed_at": final_result.get(
                "analyzed_at",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            "result": {
                "product_name": FileIO.value_or_unknown(result_data.get("product_name")),
                "capacity": FileIO.value_or_unknown(result_data.get("capacity")),
                # [강화] merge_fragmented_text 사용 중단.
                # 사용방법/주의사항/효능은 항목 단위 list로 그대로 전달한다.
                # (한 줄로 합치면 카테고리별 정확도가 떨어진다.)
                "usage": FileIO.list_or_unknown(result_data.get("usage", [])),
                "cautions": FileIO.list_or_unknown(result_data.get("cautions", [])),
                "effects": FileIO.list_or_unknown(result_data.get("effects", [])),
                "ingredients": {
                    "before_api": FileIO.clean_list(ingredients_before_api),
                    "after_api": FileIO.clean_list(ingredients_after_api),
                    "api_failed_candidates": ingredient_api_validation.get("unverified_ingredients", [])
                },
                "qr": {
                    "detected": bool(qr_data.get("detected", False)),
                    "url": FileIO.value_or_unknown(qr_data.get("url")),
                    "urls": FileIO.clean_list(qr_data.get("urls", [])),
                    "raw_values": FileIO.clean_list(qr_data.get("raw_values", []))
                },
                # [강화] qr_info 구조도 함께 노출 (클라이언트 호환 + 신구조)
                "qr_info": {
                    "qr_codes": FileIO.clean_list(
                        (result_data.get("qr_info", {}) or {}).get("qr_codes", [])
                        or qr_data.get("raw_values", [])
                    ),
                    "urls": FileIO.clean_list(
                        (result_data.get("qr_info", {}) or {}).get("urls", [])
                        or qr_data.get("urls", [])
                    )
                },
                "others": FileIO.clean_list(result_data.get("others", []))
            },
            "ingredient_api_validation": {
                "status": ingredient_api_validation.get("status", "unknown"),
                "verified_ingredient_names": FileIO.clean_list(
                    ingredient_api_validation.get("verified_ingredient_names", [])
                    or ingredient_api_validation.get("ingredients_after_api", [])
                    or ingredient_api_validation.get("ingredients", [])
                ),
                "unverified_ingredients": ingredient_api_validation.get("unverified_ingredients", []),
                "verified_count": ingredient_api_validation.get("verified_count", 0),
                "unverified_count": ingredient_api_validation.get("unverified_count", 0),
                "total_checked_count": ingredient_api_validation.get("total_checked_count", 0)
            }
        }

        if include_debug:
            payload["debug"] = {
                "raw_text": result_data.get("raw_text", ""),
                "layout_text": result_data.get("layout_text", ""),
                "raw_section_text": result_data.get("raw_section_text", {}),
                "section_detection_summary": result_data.get("section_detection_summary", {}),
                "ocr_summary": result_data.get("ocr_summary", {}),
                "detected_sections": result_data.get("detected_sections", [])
            }

        return FileIO.clean_for_json(payload)

    # =========================================================
    # 7. 사람이 읽기 쉬운 리포트 저장
    # =========================================================

    @staticmethod
    def save_analysis_report(result_dict):
        filename = FileIO.timestamp_filename("analysis_report", "txt")
        filepath = os.path.join("outputs", "reports", filename)

        result_data = FileIO.unwrap_result(result_dict)
        lines = []

        lines.append("Dermalens OCR 분석 결과")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"저장 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        if "merged_postprocessed_result" in result_data:
            merged_postprocessed_result = result_data.get("merged_postprocessed_result", {})
            ingredient_api_validation = result_data.get("ingredient_api_validation", {})
            qr_analysis = result_data.get("qr_analysis", {})
            image_analysis_results = result_data.get("image_analysis_results", [])

            lines.append("[파이프라인 요약]")
            lines.append("전체 이미지 OCR → OCR bbox/line 기반 구간 탐지 → 후처리 → QR/URL 분석 → 성분 API 검증")
            lines.append("")

            lines.append("[입력 이미지]")
            FileIO.append_list_or_unknown(lines, result_data.get("input_image_paths", []))
            lines.append("")

            FileIO._append_image_analysis_summary(
                lines=lines,
                image_analysis_results=image_analysis_results
            )

            FileIO._append_result_sections(
                lines=lines,
                result_data=merged_postprocessed_result,
                ingredient_api_validation=ingredient_api_validation,
                qr_analysis=qr_analysis
            )

        elif "postprocessed_result" in result_data:
            postprocessed = result_data.get("postprocessed_result", {})
            ingredient_api_validation = result_data.get("ingredient_api_validation", {})
            qr_analysis = result_data.get("qr_analysis", {})
            section_ocr_result = result_data.get("section_ocr_result", {})

            lines.append("[파이프라인 요약]")
            lines.append("이전 구조: 구간 OCR → 후처리 → 성분 API 검증")
            lines.append("")

            lines.append("[입력 이미지]")
            FileIO.append_list_or_unknown(lines, result_data.get("input_image_paths", []))
            lines.append("")

            merged_text_by_section = section_ocr_result.get("merged_text_by_section", {})

            if merged_text_by_section:
                lines.append("[구간별 OCR 텍스트]")
                for section_type in [
                    "product_name",
                    "capacity",
                    "ingredients",
                    "usage",
                    "cautions",
                    "effects",
                    "qr_url",
                    "others"
                ]:
                    lines.append(f"\n<{section_type}>")
                    lines.append(str(merged_text_by_section.get(section_type, "") or "확인 불가"))
                lines.append("")

            FileIO._append_result_sections(
                lines=lines,
                result_data=postprocessed,
                ingredient_api_validation=ingredient_api_validation,
                qr_analysis=qr_analysis
            )

        else:
            ingredient_api_validation = result_data.get("ingredient_api_validation", {})
            qr_analysis = (
                result_data.get("qr_analysis", {})
                or result_data.get("qr", {})
                or result_data.get("qr_result", {})
            )

            FileIO._append_result_sections(
                lines=lines,
                result_data=result_data,
                ingredient_api_validation=ingredient_api_validation,
                qr_analysis=qr_analysis
            )

        report_text = "\n".join(lines)

        FileIO.save_text(filepath, report_text)
        return filepath

    @staticmethod
    def _append_image_analysis_summary(lines, image_analysis_results):
        lines.append("[이미지별 분석 요약]")

        if not image_analysis_results:
            lines.append("확인 불가")
            lines.append("")
            return

        success_count = 0
        failed_count = 0

        for index, item in enumerate(image_analysis_results, start=1):
            if not isinstance(item, dict):
                continue

            image_path = item.get("image_path", "")
            success = item.get("success", False)
            error = item.get("error")

            if success:
                success_count += 1
            else:
                failed_count += 1

            section_result = item.get("section_result", {}) or {}
            ocr_result = item.get("ocr_result", {}) or {}
            postprocessed_result = item.get("postprocessed_result", {}) or {}

            section_summary = section_result.get("section_detection_summary", {})
            ocr_summary = ocr_result.get("ocr_summary", {})

            lines.append(f"{index}. 이미지: {image_path}")
            lines.append(f"   - 성공 여부: {success}")
            lines.append(f"   - 오류: {error if error else '없음'}")
            lines.append(f"   - OCR line 수: {ocr_summary.get('line_count', 0)}")
            lines.append(f"   - OCR block 수: {ocr_summary.get('block_count', 0)}")
            lines.append(f"   - 탐지 구간 수: {section_summary.get('detected_section_count', 0)}")
            lines.append(f"   - 제품명: {FileIO.value_or_unknown(postprocessed_result.get('product_name'))}")
            lines.append(f"   - 용량: {FileIO.value_or_unknown(postprocessed_result.get('capacity'))}")
            lines.append(f"   - 성분 후보 수: {len(FileIO.clean_list(postprocessed_result.get('ingredient_candidates', [])))}")
            lines.append("")

        lines.append(f"분석 성공 이미지 수: {success_count}개")
        lines.append(f"분석 실패 이미지 수: {failed_count}개")
        lines.append("")

    @staticmethod
    def _append_result_sections(lines, result_data, ingredient_api_validation=None, qr_analysis=None):
        ingredient_api_validation = ingredient_api_validation or {}
        qr_analysis = qr_analysis or {}

        lines.append("[제품 정보]")
        lines.append(f"제품명: {FileIO.value_or_unknown(result_data.get('product_name'))}")
        lines.append(f"용량: {FileIO.value_or_unknown(result_data.get('capacity'))}")
        lines.append("")

        ingredients_before_api = (
            result_data.get("ingredients_before_api", [])
            or result_data.get("ingredient_candidates", [])
            or result_data.get("ingredients", [])
        )

        ingredients_after_api = (
            result_data.get("ingredients_after_api", [])
            or result_data.get("api_verified_ingredients", [])
            or ingredient_api_validation.get("verified_ingredient_names", [])
            or ingredient_api_validation.get("ingredients_after_api", [])
            or ingredient_api_validation.get("ingredients", [])
        )

        lines.append("[성분 API 검증 전 후보]")
        FileIO.append_list_or_unknown(lines, ingredients_before_api)
        lines.append("")

        lines.append("[성분 API 검증 후 성분]")
        FileIO.append_list_or_unknown(lines, ingredients_after_api)
        lines.append("")

        lines.append("[API 검증 요약]")
        lines.append(f"상태: {ingredient_api_validation.get('status', '확인 불가')}")
        lines.append(f"검증 성공: {ingredient_api_validation.get('verified_count', 0)}개")
        lines.append(f"검증 실패: {ingredient_api_validation.get('unverified_count', 0)}개")
        lines.append(f"총 검증 대상: {ingredient_api_validation.get('total_checked_count', 0)}개")
        lines.append("")

        api_success_results = (
            ingredient_api_validation.get("api_success_results", [])
            or result_data.get("api_success_results", [])
        )

        api_failed_results = (
            ingredient_api_validation.get("api_failed_results", [])
            or result_data.get("api_failed_results", [])
        )

        if api_success_results:
            lines.append("[API 성공 상세]")
            for idx, item in enumerate(api_success_results, start=1):
                if not isinstance(item, dict):
                    lines.append(f"{idx}. {item}")
                    continue

                ocr_name = item.get("ocr_name", "")
                matched_name = item.get("matched_name_kr", "")
                similarity = item.get("similarity", "")
                lines.append(f"{idx}. OCR: {ocr_name} → 표준명: {matched_name} / 유사도: {similarity}")
            lines.append("")

        if api_failed_results:
            lines.append("[API 실패 상세]")
            for idx, item in enumerate(api_failed_results, start=1):
                if not isinstance(item, dict):
                    lines.append(f"{idx}. {item}")
                    continue

                ocr_name = item.get("ocr_name", "")
                query_name = item.get("query_name", "")
                reason = item.get("reason", "")
                lines.append(f"{idx}. OCR: {ocr_name} / 검색어: {query_name} / 사유: {reason}")
            lines.append("")

        lines.append("[사용방법]")
        usage = result_data.get("usage", [])
        FileIO.append_list_or_unknown(lines, usage)
        lines.append("")

        lines.append("[주의사항]")
        cautions = result_data.get("cautions", [])
        FileIO.append_list_or_unknown(lines, cautions)
        lines.append("")

        lines.append("[효능/장점]")
        effects = result_data.get("effects", [])
        FileIO.append_list_or_unknown(lines, effects)
        lines.append("")

        lines.append("[QR / URL]")
        qr_codes = (
            result_data.get("qr_codes", [])
            or result_data.get("qr_urls", [])
            or qr_analysis.get("qr_codes", [])
            or qr_analysis.get("urls", [])
            or qr_analysis.get("url_candidates", [])
        )
        FileIO.append_list_or_unknown(lines, qr_codes)
        lines.append("")

        others = result_data.get("others", [])

        if others:
            lines.append("[기타 텍스트]")
            FileIO.append_list_or_unknown(lines, others)
            lines.append("")

        if qr_analysis:
            lines.append("[QR 분석 정보]")
            lines.append(json.dumps(
                FileIO.clean_for_json(qr_analysis),
                ensure_ascii=False,
                indent=2
            ))
            lines.append("")

        raw_section_text = result_data.get("raw_section_text", {})

        if raw_section_text:
            lines.append("[구간별 원문 텍스트]")
            for key, value in raw_section_text.items():
                lines.append(f"\n<{key}>")
                lines.append(str(value or "확인 불가"))
            lines.append("")

        processing_summary = result_data.get("processing_summary", {})

        if processing_summary:
            lines.append("[처리 요약]")
            for key, value in processing_summary.items():
                lines.append(f"{key}: {value}")
            lines.append("")

        section_detection_summary = result_data.get("section_detection_summary", {})

        if section_detection_summary:
            lines.append("[구간 탐지 요약]")
            for key, value in section_detection_summary.items():
                lines.append(f"{key}: {value}")
            lines.append("")

        raw_text = result_data.get("raw_text", "")

        if raw_text:
            lines.append("[OCR RAW TEXT]")
            lines.append(raw_text)
            lines.append("")

        layout_text = result_data.get("layout_text", "")

        if layout_text and layout_text != raw_text:
            lines.append("[OCR LAYOUT TEXT]")
            lines.append(layout_text)
            lines.append("")

    # =========================================================
    # 8. 데이터 정리 유틸
    # =========================================================

    @staticmethod
    def unwrap_result(data):
        if isinstance(data, dict) and "result" in data:
            result = data.get("result")

            if isinstance(result, dict):
                return result

        return data if isinstance(data, dict) else {}

    @staticmethod
    def clean_for_json(data):
        if isinstance(data, dict):
            return {
                str(key): FileIO.clean_for_json(value)
                for key, value in data.items()
            }

        if isinstance(data, list):
            return [
                FileIO.clean_for_json(item)
                for item in data
            ]

        if isinstance(data, tuple):
            return [
                FileIO.clean_for_json(item)
                for item in data
            ]

        if isinstance(data, (str, int, float, bool)):
            return data

        if data is None:
            return None

        return str(data)

    @staticmethod
    def clean_list(items):
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

            item = str(item).strip()

            if not item:
                continue

            if item in [
                "확인 불가",
                "없음",
                "null",
                "None",
                "none",
                "N/A",
                "n/a",
                "unknown",
                "Unknown"
            ]:
                continue

            key = (
                item.lower()
                .replace(" ", "")
                .replace("\n", "")
                .replace("\t", "")
                .replace("\r", "")
            )

            if key not in seen:
                results.append(item)
                seen.add(key)

        return results

    @staticmethod
    def list_or_unknown(items):
        cleaned = FileIO.clean_list(items)
        return cleaned if cleaned else ["확인 불가"]

    @staticmethod
    def merge_fragmented_text(items):
        """
        OCR로 인해 한 문장이 여러 줄로 쪼개진 경우, 공백으로 이어붙여 한 항목으로 합친다.
        가독성 좋은 단일 문자열 형태로 server payload에 내보낸다.
        """
        cleaned = FileIO.clean_list(items)
        if not cleaned:
            return ["확인 불가"]

        merged = " ".join(cleaned)
        merged = re.sub(r"\s+", " ", merged).strip()

        return [merged] if merged else ["확인 불가"]

    @staticmethod
    def compute_accuracy_block(ocr_confidence_avg, ingredient_api_validation):
        """
        server payload 최상단에 들어갈 accuracy 블록 생성.

        - ocr_confidence: OCR 평균 신뢰도 (0.0 ~ 1.0)
        - ingredient_match_rate: 성분 API 검증 성공률 (%)
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

    @staticmethod
    def extract_normalized_ingredients(matched_results):
        normalized = []
        seen = set()

        for result in matched_results or []:
            if not isinstance(result, dict):
                continue

            if not result.get("matched"):
                continue

            matched_name = result.get("matched_name_kr")

            if not matched_name:
                continue

            if isinstance(matched_name, list):
                names = matched_name
            else:
                names = [matched_name]

            for name in names:
                if not name:
                    continue

                key = str(name).replace(" ", "").lower()

                if key not in seen:
                    normalized.append(str(name).strip())
                    seen.add(key)

        return normalized

    @staticmethod
    def append_list_or_unknown(lines, data_list):
        cleaned = FileIO.clean_list(data_list)

        if cleaned:
            for idx, item in enumerate(cleaned, start=1):
                lines.append(f"{idx}. {item}")
        else:
            lines.append("확인 불가")

    @staticmethod
    def value_or_unknown(value):
        if value is None:
            return "확인 불가"

        if isinstance(value, list):
            cleaned = FileIO.clean_list(value)
            return cleaned[0] if cleaned else "확인 불가"

        if isinstance(value, dict):
            if "url" in value:
                return FileIO.value_or_unknown(value.get("url"))

            if "value" in value:
                return FileIO.value_or_unknown(value.get("value"))

            return "확인 불가"

        text = str(value).strip()

        if not text:
            return "확인 불가"

        if text in [
            "확인 불가",
            "없음",
            "null",
            "None",
            "none",
            "N/A",
            "n/a",
            "unknown",
            "Unknown"
        ]:
            return "확인 불가"

        return text