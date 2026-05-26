# Dermalens OCR 서버 종료
# 사용법: 프로젝트 루트에서  .\stop_server.ps1
# (서버를 백그라운드로 띄웠거나 Ctrl+C 로 못 끈 경우 사용)

$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*uvicorn*src.server*' }

if ($procs) {
    $procs | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force
        Write-Host "서버 종료: PID $($_.ProcessId)" -ForegroundColor Yellow
    }
} else {
    Write-Host "실행 중인 OCR 서버가 없습니다." -ForegroundColor DarkGray
}
