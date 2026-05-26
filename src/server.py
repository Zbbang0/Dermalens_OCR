"""
Dermalens OCR FastAPI 서버.

엔드포인트
----------
GET  /health
    헬스 체크.

POST /ocr   (application/json)   ← B 플랜: URL 방식
    body:
      - user_id   : 앱(프론트)이 전달하는 사용자 식별자
      - image_url : 화장품 라벨 이미지 URL (단일 문자열 또는 URL 배열)

    처리 흐름:
      1. image_url 의 이미지를 인터넷에서 다운로드 → 임시 폴더 저장
      2. OCR 파이프라인 실행 (src.pipeline.OCRPipeline)
      3. 백엔드 스펙 payload 생성
         (user_id / ingredients=after_api 배열 / raw_text)
      4. 백엔드(BACKEND_OCR_RESULT_URL)로 결과 POST 전송
      5. 전송한 payload + 백엔드 응답을 호출자에게 반환

    구조 (B 플랜):
      프론트 촬영 → 이미지 업로드(스토리지) → URL 이 DB 에 저장
        → 백엔드가 그 URL 을 읽어 OCR /ocr 로 요청
        → OCR 은 URL 의 이미지를 다운로드해 분석 (OCR 은 DB 를 모름)

기존 main.py / file_io.build_server_payload 등 기존 기능은 변경하지 않는다.
(과거 '파일 직접 업로드' 방식 엔드포인트는 아래에 주석으로 보존.)

실행
----
프로젝트 루트에서:
    uvicorn src.server:app --reload --host 0.0.0.0 --port 8080
"""

import base64
import json
import os
import shutil
import sys
import tempfile
import traceback
from typing import List, Union

# Windows 콘솔 기본 인코딩(cp949)에서는 파이프라인 로그의 유니코드 기호(▸, ✖ 등)를
# 출력할 때 UnicodeEncodeError 가 발생한다. 서버 진입점에서 표준 출력 스트림을
# UTF-8 로 재설정해 방지한다. (기존 main.py 로그 함수는 변경하지 않는다.)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.pipeline import OCRPipeline
from src.api.backend_payload import build_backend_payload

load_dotenv()


def _bootstrap_google_credentials():
    """
    Railway 등 '파일 업로드가 어려운' 배포 환경 지원.

    Google Vision 서비스계정 키는 원래 JSON '파일'이라 클라우드에 올리기 까다롭다.
    그래서 키 JSON '내용'을 환경변수 GOOGLE_CREDENTIALS_JSON 으로 받아
    서버 시작 시 임시 파일로 복원하고, GOOGLE_APPLICATION_CREDENTIALS 가
    그 경로를 가리키게 한다.

    - 일반 JSON 문자열, base64 인코딩 문자열 둘 다 지원.
    - GOOGLE_CREDENTIALS_JSON 이 없으면 아무 것도 하지 않는다
      → 로컬의 기존 '파일 경로' 방식(run_ocr.py)이 그대로 동작한다.
    """
    raw = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if not raw:
        return  # 기존 파일 경로 방식 유지

    content = raw.strip()

    # base64 로 넣은 경우 디코드 (JSON 은 '{' 로 시작)
    if not content.startswith("{"):
        try:
            content = base64.b64decode(content).decode("utf-8")
        except Exception:
            pass

    # 유효한 JSON 인지 가볍게 확인
    try:
        json.loads(content)
    except Exception as error:
        print(f"[경고] GOOGLE_CREDENTIALS_JSON 파싱 실패 — 키 복원 생략: {error}")
        return

    key_path = os.path.join(tempfile.gettempdir(), "gcp-vision-key.json")
    with open(key_path, "w", encoding="utf-8") as out:
        out.write(content)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    print(f"[부트스트랩] GOOGLE_CREDENTIALS_JSON → 임시 키 파일 생성 완료: {key_path}")


# 파이프라인(OCRRunner)이 키를 읽기 전에 미리 복원해 둔다.
_bootstrap_google_credentials()

app = FastAPI(title="Dermalens OCR API", version="1.0.0")

# 파이프라인은 1회만 초기화해 재사용 (Google Vision 클라이언트 등 비용 절감).
_pipeline = None


def get_pipeline() -> OCRPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = OCRPipeline()
    return _pipeline


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================================================
# B 플랜: URL 기반 OCR 엔드포인트
# =========================================================

class OCRRequest(BaseModel):
    """
    POST /ocr 요청 본문.

    user_id   : 앱/백엔드가 전달하는 사용자 식별자 (그대로 백엔드로 전달)
    image_url : 이미지 URL. 단일 문자열 또는 여러 장이면 문자열 배열.
    """
    user_id: Union[str, int]
    image_url: Union[str, List[str]]


@app.post("/ocr")
def ocr(req: OCRRequest):
    # URL 정규화 (단일/배열 모두 list 로)
    urls = req.image_url if isinstance(req.image_url, list) else [req.image_url]
    urls = [str(u).strip() for u in urls if u and str(u).strip()]

    if not urls:
        raise HTTPException(status_code=400, detail="image_url 이 필요합니다.")

    temp_dir = tempfile.mkdtemp(prefix="dermalens_ocr_")

    try:
        # 1. URL 이미지 다운로드
        try:
            saved_paths = _download_images(urls, temp_dir)
        except requests.exceptions.RequestException as error:
            raise HTTPException(
                status_code=400,
                detail=f"이미지 다운로드 실패: {error}",
            )

        if not saved_paths:
            raise HTTPException(status_code=400, detail="다운로드된 이미지가 없습니다.")

        # 2. OCR 파이프라인 실행
        try:
            final_result = get_pipeline().analyze(saved_paths)
        except Exception as error:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"OCR 분석 실패: {error}")

        # 3. 백엔드 스펙 payload 생성 (분석한 이미지 URL 도 함께 전달)
        payload = build_backend_payload(
            final_result,
            user_id=req.user_id,
            image_url=urls[0],
        )

        # 4. 백엔드로 전송
        backend_url = os.getenv("BACKEND_OCR_RESULT_URL")
        backend_response = _send_to_backend(backend_url, payload)

        # 5. 결과 반환
        return JSONResponse(
            content={
                "success": True,
                "sent_payload": payload,
                "backend_response": backend_response,
            }
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _download_images(urls, temp_dir):
    """
    URL 목록의 이미지를 임시 폴더에 다운로드하고 저장 경로 list 를 반환한다.

    - 외부에서 GET 으로 접근 가능한 URL 이어야 한다 (공개 URL 또는 서명 URL).
    - 확장자는 URL 에서 추정, 없으면 .jpg 로 저장 (OCR 은 내용 기반이라 무방).
    """
    saved = []

    for index, url in enumerate(urls, start=1):
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        # URL 의 쿼리스트링 제거 후 확장자 추정
        path_part = url.split("?")[0]
        ext = os.path.splitext(path_part)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
            ext = ".jpg"

        dest = os.path.join(temp_dir, f"{index:02d}{ext}")
        with open(dest, "wb") as out:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    out.write(chunk)

        saved.append(dest)

    return saved


def _send_to_backend(backend_url, payload):
    """백엔드로 payload 를 POST 전송하고 결과 요약을 반환한다."""

    if not backend_url:
        return {"skipped": "BACKEND_OCR_RESULT_URL 미설정 — 전송 생략"}

    try:
        response = requests.post(backend_url, json=payload, timeout=30)
        return {
            "status_code": response.status_code,
            "body": _safe_json(response),
        }
    except requests.exceptions.RequestException as error:
        return {"error": str(error)}


def _safe_json(response):
    try:
        return response.json()
    except Exception:
        return response.text


# =========================================================
# [보존] 과거 방식 — 파일 직접 업로드 (multipart/form-data)
# B 플랜(URL 방식)으로 전환하면서 주석 처리. 필요 시 되살려 쓰면 된다.
# (활성화하려면 위의 URL 기반 @app.post("/ocr") 와 경로가 겹치지 않게
#  경로를 /ocr-upload 등으로 바꾸거나 둘 중 하나만 사용할 것)
# =========================================================
#
# @app.post("/ocr-upload")
# async def ocr_upload(
#     user_id: str = Form(...),
#     files: List[UploadFile] = File(...),
# ):
#     if not files:
#         raise HTTPException(status_code=400, detail="이미지 파일이 필요합니다.")
#
#     temp_dir = tempfile.mkdtemp(prefix="dermalens_ocr_")
#     saved_paths = []
#
#     try:
#         # 1. 업로드 이미지 임시 저장
#         for index, upload in enumerate(files, start=1):
#             base = os.path.basename(upload.filename or f"image_{index}")
#             dest = os.path.join(temp_dir, f"{index:02d}_{base}")
#             with open(dest, "wb") as out:
#                 shutil.copyfileobj(upload.file, out)
#             saved_paths.append(dest)
#
#         # 2. OCR 파이프라인 실행
#         try:
#             final_result = get_pipeline().analyze(saved_paths)
#         except Exception as error:
#             traceback.print_exc()
#             raise HTTPException(status_code=500, detail=f"OCR 분석 실패: {error}")
#
#         # 3. 백엔드 스펙 payload 생성
#         payload = build_backend_payload(final_result, user_id=user_id)
#
#         # 4. 백엔드로 전송
#         backend_url = os.getenv("BACKEND_OCR_RESULT_URL")
#         backend_response = _send_to_backend(backend_url, payload)
#
#         # 5. 결과 반환
#         return JSONResponse(
#             content={
#                 "success": True,
#                 "sent_payload": payload,
#                 "backend_response": backend_response,
#             }
#         )
#
#     finally:
#         shutil.rmtree(temp_dir, ignore_errors=True)
