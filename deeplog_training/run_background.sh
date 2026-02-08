#!/bin/bash
# DeepLog 백그라운드 학습 실행 스크립트
# nohup을 사용하여 SSH 연결 끊어져도 학습 계속

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/training_${TIMESTAMP}.log"
PID_FILE="${OUTPUT_DIR}/training.pid"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 기존 학습 프로세스 확인
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "WARNING: 기존 학습 프로세스가 실행 중입니다 (PID: $OLD_PID)"
        echo "중단하려면: kill $OLD_PID"
        exit 1
    fi
fi

echo "============================================================"
echo "DeepLog Background Training"
echo "============================================================"
echo ""
echo "Start Time: $(date)"
echo "Log File: $LOG_FILE"
echo ""

# 학습 시작 (백그라운드)
nohup python "${SCRIPT_DIR}/train.py" "$@" > "$LOG_FILE" 2>&1 &

# PID 저장
echo $! > "$PID_FILE"
PID=$(cat "$PID_FILE")

echo "Training started in background!"
echo "  PID: $PID"
echo ""
echo "Useful commands:"
echo "  - 로그 확인:  tail -f $LOG_FILE"
echo "  - 상태 확인:  ps -p $PID"
echo "  - 학습 중단:  kill $PID"
echo ""
echo "============================================================"

# 초기 로그 출력
sleep 3
echo ""
echo "=== Initial Log Output ==="
tail -20 "$LOG_FILE"
