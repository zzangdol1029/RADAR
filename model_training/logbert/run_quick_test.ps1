# LogBERT Quick Test - Intel XPU
# UTF-8 인코딩 설정
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 로그 파일 이름 생성
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "logs\train_quick_$timestamp.log"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host " LogBERT Quick Test - Intel XPU" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Conda environment: logbert_ipex"
Write-Host "Config: configs/test_quick_xpu.yaml"
Write-Host "Expected duration: 20-30 minutes" -ForegroundColor Yellow
Write-Host "Log file: $logFile" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# 로그 디렉토리 생성
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

# 환경 변수 설정
$env:PYTHONIOENCODING = "utf-8"

# 학습 실행 및 로그 저장
Write-Host "Training started..." -ForegroundColor Green
Start-Transcript -Path $logFile -Append

try {
    & conda run -n logbert_ipex --no-capture-output python scripts/train_intel.py --config configs/test_quick_xpu.yaml
    $exitCode = $LASTEXITCODE
} catch {
    Write-Host "Error occurred: $_" -ForegroundColor Red
    $exitCode = 1
} finally {
    Stop-Transcript
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "End time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
if ($exitCode -eq 0) {
    Write-Host "Status: SUCCESS" -ForegroundColor Green
} else {
    Write-Host "Status: FAILED (Exit code: $exitCode)" -ForegroundColor Red
}
Write-Host "Log saved to: $logFile" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
