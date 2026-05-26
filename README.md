# Dermalens OCR

화장품 라벨 이미지를 OCR 분석해 **제품명 · 용량 · 전성분 · 사용방법 · 주의사항 · 효능 · QR/URL** 을 추출하고,
공공데이터포털 성분 API 로 성분을 검증한 뒤 백엔드로 전달하는 파이프라인.

## 파이프라인

```
이미지 → 전처리 → Google Vision OCR → 구간 탐지 → 후처리
      → GPT 정밀 분류/추출 → QR/URL 분석 → 성분 API 검증 → 최종 JSON
```

OCR 엔진은 **Google Cloud Vision** (`DOCUMENT_TEXT_DETECTION`) 을 사용한다.

## 설치

```bash
pip install -r requirements.txt
cp .env.example .env   # Windows: copy .env.example .env
# .env 에 실제 키 값 채우기
```

필요한 환경변수는 [.env.example](.env.example) 참고.

## 실행

### 1) CLI (단발 분석)

```bash
python -m src.main
```

`src/main.py` 의 `image_paths` 에 지정된 이미지를 분석한다.

### 2) API 서버 (FastAPI)

```bash
uvicorn src.server:app --reload --host 0.0.0.0 --port 8080
```

#### `POST /ocr` (multipart/form-data)

| 필드 | 타입 | 설명 |
|------|------|------|
| `user_id` | str | 앱(프론트)이 전달하는 사용자 식별자. OCR 은 그대로 백엔드로 전달 |
| `files` | file[] | 화장품 라벨 이미지 (1장 이상) |

처리: 이미지 OCR → 분석 → 백엔드 스펙 payload 생성 → `BACKEND_OCR_RESULT_URL` 로 POST 전송.

##### 백엔드로 전송하는 payload (`POST /api/analysis/ocr-result/` 스펙)

```json
{
  "user_id": "앱에서 전달한 값 그대로",
  "ingredients": ["검증된 성분(after_api) 배열"],
  "raw_text": "OCR 원문"
}
```

호출 예시:

```bash
curl -X POST http://localhost:8080/ocr \
  -F "user_id=123" \
  -F "files=@images/sample1.jpg"
```

## 프로젝트 구조

```
src/
  main.py                 # CLI 진입점 (단계별 분석 흐름)
  pipeline.py             # 분석 단계를 함수로 호출하는 래퍼 (서버에서 사용)
  server.py               # FastAPI 서버 (POST /ocr)
  ocr/                    # OCR / 전처리 / 구간 탐지 / 후처리 / QR
  ai/                     # GPT·Claude 기반 추출/검증
  api/
    ingredient_api.py     # 공공데이터포털 성분 검증
    qr_analyzer.py        # QR/URL 분석
    backend_payload.py    # 백엔드 스펙 payload 어댑터
  utils/                  # 설정 / 파일 저장 유틸
```
