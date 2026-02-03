# LogBERT Training

BERT 기반 로그 이상 탐지 모델 학습 패키지

## 🎯 개요

이 디렉토리는 전처리된 로그 데이터셋으로부터 LogBERT 모델을 학습하는 통합 패키지입니다.

## 📁 디렉토리 구조

```
logbert/
├── model.py              # LogBERT 모델 정의
├── dataset.py            # 데이터셋 클래스 (전처리된 JSON 로딩)
├── train.py              # 통합 학습 스크립트 (XPU/CUDA/CPU 자동 감지)
├── evaluate.py           # 모델 평가 스크립트 (성능 메트릭)
├── __init__.py           # 패키지 초기화
├── configs/              # 학습 설정 파일들
│   ├── test_quick.yaml   # 빠른 테스트용 (5개 파일)
│   └── full_gpu.yaml     # 전체 학습용 (324개 파일)
├── scripts/              # [DEPRECATED] 레거시 스크립트
│   ├── train_cuda.py     # CUDA 스크립트
│   └── train_intel.py    # Intel 스크립트
├── logs/                 # 학습 로그 저장 위치
├── checkpoints/          # 모델 체크포인트 저장 위치
└── README.md             # 이 파일
```

## 🚀 빠른 시작

### 1. 테스트 실행 (5개 파일, ~10분)

```bash
cd C:\RADAR\RADAR\model_training\logbert

# 기본 실행
python train.py --config configs/test_quick.yaml

# 로그 파일 지정
python train.py --config configs/test_quick.yaml --log-file logs/my_test.log
```

### 2. 전체 학습 (324개 파일, ~수일)

```bash
python train.py --config configs/full_gpu.yaml
```

### 3. 커스텀 설정

```bash
# 데이터 디렉토리 오버라이드
python train.py --config configs/test_quick.yaml --data-dir "D:/other/path"

# 출력 디렉토리 오버라이드
python train.py --config configs/test_quick.yaml --output-dir "./my_checkpoints"
```

## ⚙️ 설정 파일 설명

### test_quick.yaml (빠른 테스트)
```yaml
model:
  vocab_size: 10000        # 어휘 크기
  hidden_size: 768
  num_hidden_layers: 6     # 테스트용으로 감소

training:
  batch_size: 16
  num_epochs: 1
  
data:
  preprocessed_dir: "/RADAR/preprocessing/output"
  limit_files: 5           # 최근 5개 파일만
```

### full_gpu.yaml (전체 학습)
```yaml
model:
  vocab_size: 10000
  num_hidden_layers: 12    # BERT-base 표준

training:
  batch_size: 64           # GPU 메모리에 따라 조정
  num_epochs: 3
  
data:
  limit_files: null        # 전체 324개 파일 사용
```

## 📊 데이터셋 정보

- **위치**: `/home/zzangdol/RADAR/preprocessing/output`
- **파일 수**: 324개 (날짜별 JSON 파일)
- **총 세션 수**: ~수백만 개
- **Vocabulary Size**: 586 (설정값 10000으로 여유 있음)
- **시퀀스 길이**: 최대 512 (평균 3.7, 최대 52)

### 데이터 형식
각 JSON 파일은 세션 배열을 포함:
```json
{
  "session_id": 0,
  "token_ids": [101, 1, 2, 3, ..., 102, 0, 0],      // 길이 512
  "attention_mask": [1, 1, 1, 1, ..., 1, 0, 0],     // 길이 512
  "event_sequence": [1, 2, 3],
  "has_error": false,
  "has_warn": false,
  "service_name": "portal"
}
```

## 💻 디바이스 지원

`train.py`는 자동으로 최적의 디바이스를 감지합니다:

1. **Intel XPU** (Intel Arc Graphics) - IPEX 최적화 적용
2. **NVIDIA CUDA** (GeForce/RTX) - Multi-GPU 지원
3. **CPU** (Fallback)

### 필요 패키지

```bash
# 기본 패키지
pip install torch transformers pyyaml tqdm

# Intel GPU 사용 시 (선택)
pip install intel-extension-for-pytorch

# NVIDIA GPU는 추가 설치 불필요 (PyTorch에 포함)
```

## 📝 학습 과정

### 1. 데이터 로딩
- JSON 파일들을 순차적으로 로드
- `limit_files` 설정으로 파일 수 제한 가능
- 각 세션을 개별 샘플로 처리

### 2. MLM (Masked Language Modeling)
- 15% 토큰 마스킹
  - 80%: [MASK] 토큰으로 교체
  - 10%: 랜덤 토큰으로 교체
  - 10%: 원래 토큰 유지

### 3. 학습 진행
- Epoch별 학습
- 배치 단위 loss 계산 및 역전파
- Cosine Annealing LR 스케줄링
- Gradient Clipping

### 4. 체크포인트 저장
- `save_interval` 마다 중간 저장 (기본: 5000 steps)
- 최고 성능 모델 (`best_model.pt`)
- Epoch별 모델 (`epoch_1.pt`, `epoch_2.pt`, ...)

## 📈 학습 모니터링

### 로그 출력
```
================================================================================
🚀 LogBERT 학습 시작
================================================================================
디바이스: cuda (CUDA)
총 에폭: 3
배치 크기: 64
학습률: 2e-05
================================================================================

Epoch 1/3: 100%|████████| 12345/12345 [1:23:45<00:00, loss=2.3456, avg=2.4567]
[Step 100] Loss=2.3456, Avg=2.4567, LR=1.99e-05
[Step 200] Loss=2.2345, Avg=2.3456, LR=1.98e-05
...
💾 체크포인트 저장: checkpoints/checkpoint_step_5000.pt
```

### 체크포인트 구조
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'global_step': 5000,
    'best_loss': 1.2345,
    'config': {...},
    'device_type': 'cuda'
}
```

## 🔧 문제 해결

### 메모리 부족
- `batch_size` 줄이기 (64 → 32 → 16)
- `num_workers` 줄이기 (8 → 4 → 0)
- `limit_files` 줄이기

### 학습 속도 느림
- `num_workers` 늘리기 (CPU 코어 수만큼)
- `batch_size` 늘리기 (GPU 메모리 허용 시)
- Multi-GPU 사용 (자동 감지)

### 데이터 로딩 오류
```bash
# 데이터 디렉토리 확인
ls /home/zzangdol/RADAR/preprocessing/output

# 설정 파일 경로 확인
cat configs/test_quick.yaml
```

## � 모델 평가

학습이 완료된 후 `evaluate.py`로 모델 성능을 평가할 수 있습니다.

### 평가 실행

```bash
python evaluate.py \
    --checkpoint checkpoints_test/checkpoints/best_model.pt \
    --config configs/test_quick.yaml \
    --validation-data C:/RADAR/RADAR/preprocessing/output/preprocessed_logs_2025-02-24.json \
    --normal-ratio 0.8 \
    --max-samples 10000
```

### 평가 옵션

- `--checkpoint`: 평가할 모델 체크포인트 경로
- `--config`: 학습 시 사용한 설정 파일
- `--validation-data`: 검증용 JSON 파일 (학습에 사용하지 않은 파일 권장)
- `--normal-ratio`: 정상 데이터 비율 (기본: 0.8, 앞 80%를 정상으로 간주)
- `--max-samples`: 빠른 평가를 위한 샘플 수 제한 (선택)
- `--output-dir`: 결과 저장 디렉토리 (기본: `evaluation_results`)

### 평가 결과

평가 완료 후 다음 파일들이 생성됩니다:

```
evaluation_results/
├── evaluation_results.json    # 평가 메트릭 (JSON)
├── score_distribution.png     # 점수 분포 그래프
└── confusion_matrix.png        # 혼동 행렬 히트맵
```

### 평가 메트릭

- **Accuracy (정확도)**: 전체 예측 중 정확한 예측 비율
- **Precision (정밀도)**: 이상으로 예측한 것 중 실제 이상 비율
- **Recall (재현율)**: 실제 이상 중 올바르게 탐지한 비율
- **F1-Score**: Precision과 Recall의 조화평균
- **ROC AUC**: ROC 곡선 아래 면적
- **Confusion Matrix**: 정상/이상 예측 혼동 행렬

### 예상 출력

```
================================================================================
📊 성능 평가 결과
================================================================================
정확도 (Accuracy):  0.8543 (85.43%)
정밀도 (Precision): 0.7823 (78.23%)
재현율 (Recall):    0.8912 (89.12%)
F1-Score:          0.8333 (83.33%)
ROC AUC:           0.9012

혼동 행렬:
  True Negative (TN):  7234 (정상을 정상으로 예측)
  False Positive (FP): 1123 (정상을 이상으로 예측)
  False Negative (FN):  234 (이상을 정상으로 예측)
  True Positive (TP):  1409 (이상을 이상으로 예측)
```

## �📌 다음 단계

1. **테스트 실행**: `python train.py --config configs/test_quick.yaml`
2. **학습 결과 확인**: `logs/` 및 `checkpoints/` 디렉토리 확인
3. **모델 평가**: `python evaluate.py --checkpoint checkpoints_test/checkpoints/best_model.pt ...`
4. **전체 학습**: `configs/full_gpu.yaml` 수정 후 실행

## 📚 참고

- 전처리된 데이터: `/home/zzangdol/RADAR/preprocessing/output`
- Vocab size: 586 (설정: 10000)
- 최대 시퀀스 길이: 512
- BERT 아키텍처: 12 layers, 768 hidden, 12 heads
