# Dermalens OCR 서버 실행
# 사용법: 프로젝트 루트에서  .\run_server.ps1
# 끄기: 이 창에서 Ctrl + C  (또는 .\stop_server.ps1)

$env:PYTHONUTF8 = "1"   # 윈도우 콘솔(cp949) 한글/기호 로그 깨짐 방지

Write-Host "Dermalens OCR 서버 시작 → http://127.0.0.1:8080  (문서: /docs)" -ForegroundColor Cyan
Write-Host "끄려면 Ctrl + C" -ForegroundColor DarkGray

.\derma\Scripts\python.exe -m uvicorn src.server:app --host 127.0.0.1 --port 8080 --reload
