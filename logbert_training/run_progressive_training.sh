#!/bin/bash
# 점진적 학습을 백그라운드로 실행하는 스크립트
# 화면보호기나 터미널 종료와 무관하게 실행됩니다

set -e

# 스크립트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 기본 설정 (발열 감소를 위해 50% 데이터, 배치 크기 8로 설정, 5%씩 10번 학습)
PRETRAINED_MODEL="${1:-bert-base-uncased}"
START_RATIO="${2:-0.05}"
STEP_SIZE="${3:-0.05}"
MAX_RATIO="${4:-0.5}"
EPOCHS_PER_STAGE="${5:-5}"
MAX_MEMORY_MB="${6:-45000}"
MIN_BATCH_SIZE="${7:-8}"
FIXED_BATCH_SIZE="${8:-8}"

# 출력 디렉토리
OUTPUT_DIR="checkpoints_transfer"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUTPUT_DIR}/logs"
MAIN_LOG="${LOG_DIR}/progressive_training_${TIMESTAMP}.log"

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "점진적 학습 백그라운드 실행"
echo "=========================================="
echo "Pre-trained 모델: $PRETRAINED_MODEL"
echo "시작 비율: $START_RATIO ($(echo "$START_RATIO * 100" | bc)%)"
echo "단계 크기: $STEP_SIZE ($(echo "$STEP_SIZE * 100" | bc)%)"
echo "최대 비율: $MAX_RATIO ($(echo "$MAX_RATIO * 100" | bc)%)"
echo "단계당 에폭: $EPOCHS_PER_STAGE"
if [ -n "$MAX_MEMORY_MB" ]; then
    echo "최대 메모리: ${MAX_MEMORY_MB}MB"
fi
echo "최소 배치 크기: $MIN_BATCH_SIZE"
if [ -n "$FIXED_BATCH_SIZE" ]; then
    echo "고정 배치 크기: $FIXED_BATCH_SIZE"
fi
echo "출력 디렉토리: $OUTPUT_DIR"
echo "로그 파일: $MAIN_LOG"
echo ""

# Conda 환경 확인
if command -v conda &> /dev/null; then
    # Conda 환경 활성화
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | grep -q "^radar "; then
        conda activate radar
        echo "Conda 환경 'radar' 활성화됨"
    else
        echo "경고: Conda 환경 'radar'를 찾을 수 없습니다."
        echo "기본 Python 환경을 사용합니다."
    fi
else
    echo "경고: Conda를 찾을 수 없습니다."
    echo "기본 Python 환경을 사용합니다."
fi

# 명령어 구성
CMD="python train_transfer.py"
CMD="$CMD --progressive"
CMD="$CMD --pretrained $PRETRAINED_MODEL"
CMD="$CMD --start-ratio $START_RATIO"
CMD="$CMD --step-size $STEP_SIZE"
CMD="$CMD --max-ratio $MAX_RATIO"
CMD="$CMD --epochs-per-stage $EPOCHS_PER_STAGE"

if [ -n "$MAX_MEMORY_MB" ]; then
    CMD="$CMD --max-memory-mb $MAX_MEMORY_MB"
fi

CMD="$CMD --min-batch-size $MIN_BATCH_SIZE"

if [ -n "$FIXED_BATCH_SIZE" ]; then
    CMD="$CMD --fixed-batch-size $FIXED_BATCH_SIZE"
fi

echo "실행 명령어:"
echo "  $CMD"
echo ""

# 백그라운드 실행
echo "백그라운드에서 학습 시작..."
echo ""

# nohup으로 실행 (터미널 종료와 무관하게 실행)
nohup bash -c "
    cd '$SCRIPT_DIR'
    $CMD 2>&1 | tee '$MAIN_LOG'
" > "${MAIN_LOG}.nohup" 2>&1 &

PROCESS_ID=$!

echo "점진적 학습 프로세스 시작됨 (PID: $PROCESS_ID)"
echo ""
echo "=========================================="
echo "모니터링 명령어"
echo "=========================================="
echo ""
echo "프로세스 확인:"
echo "  ps -p $PROCESS_ID"
echo ""
echo "실시간 로그 확인:"
echo "  tail -f $MAIN_LOG"
echo ""
echo "전체 점진적 학습 로그:"
echo "  tail -f ${OUTPUT_DIR}/logs/progressive_training_*.log"
echo ""
echo "단계별 로그 확인:"
echo "  tail -f ${OUTPUT_DIR}/stage_*_*pct/logs/*.log"
echo ""
echo "프로세스 종료:"
echo "  kill $PROCESS_ID"
echo ""
echo "학습 진행 상황 확인:"
echo "  ls -lh ${OUTPUT_DIR}/stage_*_*pct/checkpoints/"
echo ""
echo "최종 결과 확인:"
echo "  cat ${OUTPUT_DIR}/progressive_training_results.json"
echo ""
echo "=========================================="
echo ""

# PID 파일 저장
echo $PROCESS_ID > "${OUTPUT_DIR}/progressive_training.pid"
echo "PID 파일 저장: ${OUTPUT_DIR}/progressive_training.pid"
echo ""

