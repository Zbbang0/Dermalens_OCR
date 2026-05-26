"""
Dermalens OCR FastAPI 서버.

엔드포인트
----------
GET  /health
    헬스 체크.

POST /ocr   (multipart/form-data)
    - user_id : 앱(프론트)이 전달하는 사용자 식별자
    - files   : 화장품 라벨 이미지 (1장 이상)

    처리 흐름:
      1. 업로드 이미지를 임시 폴더에 저장
      2. 기존 OCR 파이프라인 실행 (src.pipeline.OCRPipeline)
      3. 백엔드 스펙 payload 생성
         (user_id / ingredients=after_api 배열 / raw_text)
      4. 백엔드(BACKEND_OCR_RESULT_URL)로 결과 POST 전송
      5. 전송한 payload + 백엔드 응답을 호출자에게 반환

기존 main.py / file_io.build_server_payload 등 기존 기능은 변경하지 않는다.

실행
----
프로젝트 루트에서:
    uvicorn src.server:app --reload --host 0.0.0.0 --port 8080
"""

import os
import shutil
import sys
import tempfile
import traceback
from typing import List

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

from src.pipeline import OCRPipeline
from src.api.backend_payload import build_backend_payload

load_dotenv()

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


@app.post("/ocr")
async def ocr(
    user_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="이미지 파일이 필요합니다.")

    temp_dir = tempfile.mkdtemp(prefix="dermalens_ocr_")
    saved_paths = []

    try:
        # 1. 업로드 이미지 임시 저장
        for index, upload in enumerate(files, start=1):
            base = os.path.basename(upload.filename or f"image_{index}")
            dest = os.path.join(temp_dir, f"{index:02d}_{base}")
            with open(dest, "wb") as out:
                shutil.copyfileobj(upload.file, out)
            saved_paths.append(dest)

        # 2. OCR 파이프라인 실행
        try:
            final_result = get_pipeline().analyze(saved_paths)
        except Exception as error:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"OCR 분석 실패: {error}")

        # 3. 백엔드 스펙 payload 생성
        payload = build_backend_payload(final_result, user_id=user_id)

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
