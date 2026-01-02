#!/bin/bash
# 백그라운드에서 전처리를 실행하는 스크립트
# 화면보호기나 터미널 종료와 무관하게 실행됩니다

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${1:-logs/date_split}"
OUTPUT_DIR="${2:-preprocessing/output}"
LOG_FILE="${SCRIPT_DIR}/preprocessing_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "백그라운드 전처리 실행"
echo "=========================================="
echo "로그 디렉토리: $PROJECT_ROOT/$LOG_DIR"
echo "출력 디렉토리: $PROJECT_ROOT/$OUTPUT_DIR"
echo "실행 로그: $LOG_FILE"
echo ""

# conda 환경 활성화 및 실행
cd "$SCRIPT_DIR"

# 출력 디렉토리 생성
mkdir -p "$PROJECT_ROOT/$OUTPUT_DIR"

# nohup으로 실행 (터미널 종료와 무관하게 실행)
nohup bash -c "
    cd \"$SCRIPT_DIR\"
    source \$(conda info --base)/etc/profile.d/conda.sh
    conda activate radar
    python log_preprocessor.py --log-dir \"$PROJECT_ROOT/$LOG_DIR\" --output \"$PROJECT_ROOT/$OUTPUT_DIR/preprocessed_logs.json\" 2>&1 | tee \"$LOG_FILE\"
" > "${LOG_FILE}.nohup" 2>&1 &

PROCESS_ID=$!

echo "전처리 프로세스 시작됨 (PID: $PROCESS_ID)"
echo ""
echo "프로세스 확인:"
echo "  ps -p $PROCESS_ID"
echo ""
echo "실행 로그 확인:"
echo "  tail -f $LOG_FILE"
echo ""
echo "프로세스 종료:"
echo "  kill $PROCESS_ID"
echo ""
echo "=========================================="

