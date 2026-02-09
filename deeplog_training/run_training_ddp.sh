#!/bin/bash
# DeepLog DistributedDataParallel (DDP) 학습 실행 스크립트
#
# 사용법:
#   bash run_training_ddp.sh                  # 기본 설정으로 실행
#   bash run_training_ddp.sh --resume PATH    # 체크포인트에서 재개
#
# torchrun을 사용하여 DDP 학습을 실행합니다.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

# GPU 수 설정
NUM_GPUS=4

# CUDA 환경 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# NCCL 최적화 (A100/H100용)
export NCCL_DEBUG=INFO              # NCCL 디버그 로그 (필요 시 WARN으로 변경)
export NCCL_IB_DISABLE=0            # InfiniBand 사용 (있는 경우)
export NCCL_SOCKET_IFNAME=eth0      # 네트워크 인터페이스 (환경에 맞게 수정)
# export NCCL_P2P_DISABLE=0         # GPU 간 P2P 통신 활성화 (기본값)

# cuDNN 최적화
export CUDNN_BENCHMARK=1            # cuDNN auto-tuner 활성화

echo "========================================="
echo "DeepLog DDP Training"
echo "========================================="
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG_FILE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="
echo ""

# Python 및 PyTorch 확인
if ! command -v python &> /dev/null; then
    echo "❌ 오류: python이 설치되지 않았습니다."
    exit 1
fi

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo "❌ 오류: PyTorch가 제대로 설치되지 않았습니다."
    exit 1
fi

echo ""

# torchrun 확인
if ! command -v torchrun &> /dev/null; then
    echo "❌ 오류: torchrun이 설치되지 않았습니다."
    echo "PyTorch 1.10+ 이상이 필요합니다."
    exit 1
fi

# 로그 파일명 (타임스탬프 포함)
LOG_FILE="training_ddp_$(date +%Y%m%d_%H%M%S).log"

echo "Starting DDP Training with $NUM_GPUS GPUs..."
echo "로그 파일: $LOG_FILE"
echo ""

# torchrun으로 DDP 실행
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    "${SCRIPT_DIR}/train.py" \
    --config "$CONFIG_FILE" \
    "$@" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================="
    echo "✅ 학습 완료!"
    echo "========================================="
    echo "로그 파일: $LOG_FILE"
else
    echo "========================================="
    echo "❌ 학습 실패 (Exit Code: $EXIT_CODE)"
    echo "========================================="
    echo "로그 파일: $LOG_FILE"
    exit $EXIT_CODE
fi
