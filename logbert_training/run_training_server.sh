#!/bin/bash
# 서버에서 LogBERT 학습 실행 스크립트
# preprocessing/output 디렉토리의 전처리된 파일을 사용하여 모델 학습

set -e  # 오류 발생 시 스크립트 중단

# 스크립트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "LogBERT 서버 학습 스크립트"
echo "=========================================="
echo ""
echo "스크립트 디렉토리: $SCRIPT_DIR"
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
echo ""
echo "Python 정보:"
$PYTHON_CMD --version
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
echo "출력 디렉토리: $OUTPUT_DIR"
echo ""

# 학습 실행
echo "=========================================="
echo "학습 시작"
echo "=========================================="
echo ""

# 명령줄 인자 전달
$PYTHON_CMD train_server.py "$@"

echo ""
echo "=========================================="
echo "학습 완료"
echo "=========================================="
echo ""
echo "체크포인트 위치: $OUTPUT_DIR"
echo ""

