#!/bin/bash
# 서버에서 LogBERT 학습을 백그라운드로 실행하는 스크립트
# preprocessing/output 디렉토리의 전처리된 파일을 사용하여 모델 학습

set -e  # 오류 발생 시 스크립트 중단

# 스크립트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "LogBERT 백그라운드 학습 스크립트"
echo "=========================================="
echo ""
echo "스크립트 디렉토리: $SCRIPT_DIR"
echo ""

# Python 명령어 확인 (python3 우선, 없으면 python)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ 오류: Python을 찾을 수 없습니다."
    echo "   python3 또는 python을 설치하세요."
    exit 1
fi

# Python 버전 확인
echo "Python 정보:"
$PYTHON_CMD --version
echo ""

# Python 환경 확인
if command -v conda &> /dev/null; then
    # Conda 환경 활성화
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | grep -q "^radar "; then
        conda activate radar
        echo "✅ Conda 환경 'radar' 활성화됨"
    else
        echo "⚠️  경고: Conda 환경 'radar'를 찾을 수 없습니다."
        echo "   기본 Python 환경을 사용합니다."
    fi
else
    echo "⚠️  경고: Conda를 찾을 수 없습니다."
    echo "   기본 Python 환경을 사용합니다."
fi
echo ""

# 데이터 디렉토리 확인
DATA_DIR="${SCRIPT_DIR}/../preprocessing/output"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 오류: 데이터 디렉토리를 찾을 수 없습니다: $DATA_DIR"
    echo ""
    echo "해결 방법:"
    echo "  1. preprocessing/output 디렉토리가 존재하는지 확인하세요"
    echo "  2. 또는 --data-dir 옵션으로 경로를 지정하세요"
    exit 1
fi

FILE_COUNT=$(find "$DATA_DIR" -name "preprocessed_logs_*.json" | wc -l)
if [ "$FILE_COUNT" -eq 0 ]; then
    echo "❌ 오류: 전처리된 데이터 파일을 찾을 수 없습니다."
    echo "   디렉토리: $DATA_DIR"
    echo "   예상 파일 형식: preprocessed_logs_*.json"
    exit 1
fi

echo "✅ 데이터 디렉토리 확인: $DATA_DIR"
echo "   발견된 데이터 파일: $FILE_COUNT개"
echo ""

# 출력 디렉토리 생성
OUTPUT_DIR="${SCRIPT_DIR}/checkpoints"
mkdir -p "$OUTPUT_DIR"

# 로그 디렉토리 생성
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# 타임스탬프 생성
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/training_${TIMESTAMP}.pid"

echo "출력 디렉토리: $OUTPUT_DIR"
echo "로그 파일: $LOG_FILE"
echo ""

# 학습 명령어 구성
TRAIN_CMD="$PYTHON_CMD train_server.py"

# 명령줄 인자가 있으면 추가
if [ $# -gt 0 ]; then
    TRAIN_CMD="$TRAIN_CMD $@"
else
    # 기본 설정 파일 사용
    TRAIN_CMD="$TRAIN_CMD --config training_config_dgx.yaml"
fi

echo "=========================================="
echo "학습 명령어"
echo "=========================================="
echo "$TRAIN_CMD"
echo ""

# 백그라운드 실행
echo "백그라운드에서 학습 시작..."
echo ""

# nohup으로 실행 (터미널 종료와 무관하게 실행)
nohup bash -c "
    cd '$SCRIPT_DIR'
    $TRAIN_CMD
" > "$LOG_FILE" 2>&1 &

PROCESS_ID=$!

# 프로세스 ID 저장
echo $PROCESS_ID > "$PID_FILE"

echo "=========================================="
echo "✅ 학습 프로세스 시작됨"
echo "=========================================="
echo ""
echo "프로세스 ID: $PROCESS_ID"
echo "로그 파일: $LOG_FILE"
echo "PID 파일: $PID_FILE"
echo ""

# 프로세스가 실제로 실행 중인지 확인
sleep 2
if ps -p $PROCESS_ID > /dev/null 2>&1; then
    echo "✅ 프로세스가 정상적으로 실행 중입니다."
else
    echo "❌ 경고: 프로세스가 시작되지 않았을 수 있습니다."
    echo "   로그 파일을 확인하세요: $LOG_FILE"
    exit 1
fi

echo ""
echo "=========================================="
echo "모니터링 명령어"
echo "=========================================="
echo ""
echo "실시간 로그 확인:"
echo "  tail -f $LOG_FILE"
echo ""
echo "프로세스 확인:"
echo "  ps -p $PROCESS_ID"
echo ""
echo "GPU 사용량 확인:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "프로세스 종료:"
echo "  kill $PROCESS_ID"
echo ""
echo "체크포인트 확인:"
echo "  ls -lh $OUTPUT_DIR/checkpoints/"
echo ""
echo "=========================================="


