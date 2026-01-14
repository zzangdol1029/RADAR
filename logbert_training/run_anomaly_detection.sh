#!/bin/bash
# 로그 이상 탐지 실행 스크립트

# 기본 설정
CHECKPOINT="checkpoints/best_model.pt"
LOG_DIR="../preprocessing/logs/real_logs"
OUTPUT="results/anomaly_detection.json"
BATCH_SIZE=32
DEVICE="cuda"  # 또는 "cpu"

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        --test)
            MAX_FILES=5
            OUTPUT="results/test_anomaly_detection.json"
            shift
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "사용법: $0 [옵션]"
            echo "옵션:"
            echo "  --checkpoint PATH     체크포인트 파일 경로 (기본: $CHECKPOINT)"
            echo "  --log-dir PATH        로그 디렉토리 (기본: $LOG_DIR)"
            echo "  --output PATH         출력 파일 경로 (기본: $OUTPUT)"
            echo "  --batch-size SIZE     배치 크기 (기본: $BATCH_SIZE)"
            echo "  --device DEVICE       디바이스 (cuda/cpu, 기본: $DEVICE)"
            echo "  --threshold VALUE     이상 임계값 (선택사항)"
            echo "  --max-files NUM       최대 파일 수 (테스트용)"
            echo "  --test                테스트 모드 (5개 파일만)"
            exit 1
            ;;
    esac
done

# 출력 디렉토리 생성
mkdir -p "$(dirname "$OUTPUT")"

# 명령어 구성
CMD="python detect_anomalies.py"
CMD="$CMD --checkpoint $CHECKPOINT"
CMD="$CMD --log-dir $LOG_DIR"
CMD="$CMD --output $OUTPUT"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --device $DEVICE"

if [ -n "$THRESHOLD" ]; then
    CMD="$CMD --threshold $THRESHOLD"
fi

if [ -n "$MAX_FILES" ]; then
    CMD="$CMD --max-files $MAX_FILES"
fi

# 실행
echo "이상 탐지 실행 중..."
echo "체크포인트: $CHECKPOINT"
echo "로그 디렉토리: $LOG_DIR"
echo "출력 파일: $OUTPUT"
echo "배치 크기: $BATCH_SIZE"
echo "디바이스: $DEVICE"
echo ""

$CMD

echo ""
echo "완료! 결과 파일: $OUTPUT"
