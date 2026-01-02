#!/bin/bash
# 로그 파일을 날짜별로 분리한 후 전처리하는 스크립트

set -e

LOG_DIR="${1:-logs/real_logs}"
OUTPUT_DIR="${2:-logs/date_split}"
PREPROCESSING_OUTPUT="${3:-preprocessing/preprocessed_logs.json}"

echo "=========================================="
echo "로그 파일 날짜별 분리 및 전처리"
echo "=========================================="
echo ""

# 1단계: 로그 파일을 날짜별로 분리
echo "1단계: 로그 파일을 날짜별로 분리 중..."
echo "  입력: $LOG_DIR"
echo "  출력: $OUTPUT_DIR"
echo ""

cd preprocessing
python split_logs_by_date.py --input "../$LOG_DIR" --output "../$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "오류: 로그 파일 분리 실패"
    exit 1
fi

echo ""
echo "2단계: 분리된 파일로 전처리 시작..."
echo ""

# 2단계: 분리된 파일로 전처리
# preprocessing_config.yaml 수정 필요: log_directory를 date_split으로 변경
python log_preprocessor.py --log-dir "../$OUTPUT_DIR" --output "../$PREPROCESSING_OUTPUT"

if [ $? -ne 0 ]; then
    echo "오류: 전처리 실패"
    exit 1
fi

echo ""
echo "=========================================="
echo "완료!"
echo "=========================================="
echo "분리된 로그: $OUTPUT_DIR"
echo "전처리 결과: $PREPROCESSING_OUTPUT"
echo ""

