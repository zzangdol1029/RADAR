#!/bin/bash
# DeepLog 학습 실행 스크립트
# Tesla V100-DGXS-32GB x 4 환경

set -e

# 스크립트 위치
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 기본 설정
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"
DATA_DIR="/home/zzangdol/RADAR/preprocessing/output"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

# 파라미터 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="--epochs $2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="--batch-size $2"
            shift 2
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --config FILE      설정 파일 경로 (기본: config.yaml)"
            echo "  --data-dir DIR     데이터 디렉토리 경로"
            echo "  --output-dir DIR   출력 디렉토리 경로"
            echo "  --epochs N         학습 에폭 수"
            echo "  --batch-size N     배치 크기"
            echo "  --resume PATH      재개할 체크포인트 경로"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# GPU 정보 출력
echo "============================================================"
echo "DeepLog Training Script"
echo "============================================================"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""
echo "Configuration:"
echo "  - Config File: $CONFIG_FILE"
echo "  - Data Directory: $DATA_DIR"
echo "  - Output Directory: $OUTPUT_DIR"
echo ""

# CUDA 환경 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Python 환경 확인
echo "Python Environment:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
echo ""

# 의존성 체크
echo "Checking dependencies..."
python -c "import ijson; print('ijson: OK')" || { echo "ERROR: ijson not installed. Run: pip install ijson"; exit 1; }
python -c "import yaml; print('pyyaml: OK')" || { echo "ERROR: pyyaml not installed. Run: pip install pyyaml"; exit 1; }
python -c "import tqdm; print('tqdm: OK')" || { echo "ERROR: tqdm not installed. Run: pip install tqdm"; exit 1; }
echo ""

echo "============================================================"
echo "Starting Training..."
echo "Log file: $LOG_FILE"
echo "============================================================"
echo ""

# 학습 실행
python "${SCRIPT_DIR}/train.py" \
    --config "$CONFIG_FILE" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    $EPOCHS $BATCH_SIZE $RESUME \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
