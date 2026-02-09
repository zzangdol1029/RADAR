#!/bin/bash
# JSON → Parquet 변환 실행 스크립트
#
# 사용법:
#   bash scripts/convert_data.sh
#
# 또는 사용자 정의 경로로:
#   bash scripts/convert_data.sh /path/to/json /path/to/parquet

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 기본 경로 (config.yaml과 일치)
INPUT_DIR="${1:-/home/zzangdol/RADAR/preprocessing/output}"
OUTPUT_DIR="${2:-/home/zzangdol/RADAR/preprocessing/output_parquet}"

# 변환 설정
PATTERN="preprocessed_logs_*.json"
NUM_WORKERS=8          # 병렬 워커 수 (CPU 코어 수에 맞춰 조정)
CHUNK_SIZE=10000       # 청크 크기
COMPRESSION="snappy"   # 압축 방식 (snappy: 빠름, gzip: 압축률 높음)

echo "========================================="
echo "JSON → Parquet 데이터 변환"
echo "========================================="
echo "입력 디렉토리: $INPUT_DIR"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "파일 패턴: $PATTERN"
echo "병렬 워커 수: $NUM_WORKERS"
echo "========================================="
echo ""

# 입력 디렉토리 확인
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 오류: 입력 디렉토리가 존재하지 않습니다: $INPUT_DIR"
    exit 1
fi

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# Python 경로 확인
if ! command -v python &> /dev/null; then
    echo "❌ 오류: python이 설치되지 않았습니다."
    exit 1
fi

# 필요한 패키지 확인
echo "필요한 패키지 확인 중..."
python -c "import ijson; import pyarrow" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 오류: 필요한 패키지가 설치되지 않았습니다."
    echo "설치 명령: pip install ijson pyarrow"
    exit 1
fi
echo "✅ 모든 패키지가 설치되어 있습니다."
echo ""

# 변환 실행
echo "변환 시작..."
python "$PROJECT_DIR/convert_to_parquet.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --pattern "$PATTERN" \
    --num-workers "$NUM_WORKERS" \
    --chunk-size "$CHUNK_SIZE" \
    --compression "$COMPRESSION"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ 변환 완료!"
    echo "========================================="
    echo "출력 디렉토리: $OUTPUT_DIR"
    echo ""
    echo "다음 단계:"
    echo "1. config.yaml의 data.preprocessed_dir을 업데이트:"
    echo "   preprocessed_dir: \"$OUTPUT_DIR\""
    echo "2. data.file_pattern 업데이트:"
    echo "   file_pattern: \"*.parquet\""
    echo ""
else
    echo ""
    echo "========================================="
    echo "❌ 변환 실패 (Exit Code: $EXIT_CODE)"
    echo "========================================="
    exit $EXIT_CODE
fi
