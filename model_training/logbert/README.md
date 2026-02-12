# LogBERT Training

BERT 기반 로그 이상 탐지 모델 학습 패키지

## 🎯 개요

이 디렉토리는 전처리된 로그 데이터셋으로부터 LogBERT 모델을 학습하는 통합 패키지입니다.

## 📁 디렉토리 구조

```
logbert/
├── model.py                    # LogBERT 모델 정의
├── dataset.py                  # 데이터셋 클래스 (전처리된 JSON 로딩)
├── train.py                    # 통합 학습 스크립트 (XPU/CUDA/CPU 자동 감지)
├── evaluate.py                 # 모델 평가 스크립트 (성능 메트릭, 시각화)
├── __init__.py                 # 패키지 초기화
│
├── configs/                    # 학습 설정 파일
│   ├── full_gpu.yaml           # ⭐ 전체 학습 (324 files, NVIDIA GPU)
│   ├── test_quick.yaml         # 테스트용 (Intel XPU)
│   ├── test_quick_xpu_small.yaml  # 테스트용 (Intel XPU, small)
│   └── test_xpu.yaml          # 테스트용 (10 files, Intel XPU)
│
├── docs/                       # 문서
│   ├── setup_guide.md          # 환경 설정 가이드
│   ├── quick_start.md          # 빠른 시작 가이드
│   ├── evaluation_guide.md     # 모델 평가 가이드
│   └── 10_file_training_guide.md  # 10개 파일 학습 가이드
│
├── checkpoints_full/           # 전체 학습 (GPU) 체크포인트
├── checkpoints_test/           # 테스트 학습 (XPU) 체크포인트
├── evaluation_results/         # 평가 결과
│
├── logs/                       # 학습/평가 로그
├── train.log                   # 전체 학습 로그
│
├── EVALUATION_RESULTS.md       # 평가 결과 정리
├── PROJECT_STATUS.md           # 프로젝트 진행 상황
├── requirements.txt            # 기본 의존성
├── requirements_cuda.txt       # NVIDIA CUDA 환경 의존성
├── requirements_intel_xpu.txt  # Intel XPU 환경 의존성
└── README.md                   # 이 파일
```

## 🚀 빠른 시작

### 1. 전체 학습 (324개 파일, NVIDIA GPU)

```bash
cd logbert
python train.py --config configs/full_gpu.yaml
```

### 2. 로컬 테스트 (Intel XPU)

```bash
cd logbert
python train.py --config configs/test_quick.yaml
```

### 3. 커스텀 설정

```bash
# 데이터 디렉토리 오버라이드
python train.py --config configs/full_gpu.yaml --data-dir "/path/to/data"

# 출력 디렉토리 오버라이드
python train.py --config configs/full_gpu.yaml --output-dir "./my_checkpoints"
```

## ⚙️ 설정 파일

### full_gpu.yaml (⭐ 전체 학습 - Production)
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

### test_quick.yaml / test_xpu.yaml (로컬 테스트)
```yaml
model:
  vocab_size: 10000
  num_hidden_layers: 6     # 테스트용으로 감소

training:
  batch_size: 16
  num_epochs: 1
  
data:
  limit_files: 5           # 일부 파일만
```

## 📊 데이터셋 정보

- **파일 수**: 324개 (날짜별 JSON 파일)
- **Vocabulary Size**: 586 (설정값 10000으로 여유 있음)
- **시퀀스 길이**: 최대 512

### 데이터 형식
각 JSON 파일은 세션 배열을 포함:
```json
{
  "session_id": 0,
  "token_ids": [101, 1, 2, 3, ..., 102, 0, 0],
  "attention_mask": [1, 1, 1, 1, ..., 1, 0, 0],
  "event_sequence": [1, 2, 3],
  "has_error": false,
  "has_warn": false,
  "service_name": "portal"
}
```

## 💻 디바이스 지원

`train.py`는 자동으로 최적의 디바이스를 감지합니다:

1. **NVIDIA CUDA** (GeForce/RTX) - Multi-GPU 지원
2. **Intel XPU** (Intel Arc Graphics) - IPEX 최적화 적용
3. **CPU** (Fallback)

### 필요 패키지

```bash
# NVIDIA GPU
pip install -r requirements_cuda.txt

# Intel XPU
pip install -r requirements_intel_xpu.txt

# 기본
pip install -r requirements.txt
```

## 📝 학습 과정

### 1. 데이터 로딩
- JSON 파일들을 순차적으로 로드
- `limit_files` 설정으로 파일 수 제한 가능

### 2. MLM (Masked Language Modeling)
- 15% 토큰 마스킹
  - 80%: [MASK] 토큰으로 교체
  - 10%: 랜덤 토큰으로 교체
  - 10%: 원래 토큰 유지

### 3. 학습 진행
- Epoch별 학습, Cosine Annealing LR, Gradient Clipping

### 4. 체크포인트 저장
- `save_interval` 마다 중간 저장
- 최고 성능 모델 (`best_model.pt`)
- Epoch별 모델 (`epoch_1.pt`, `epoch_2.pt`, ...)

## 📊 모델 평가

학습이 완료된 후 `evaluate.py`로 모델 성능을 평가합니다.

### 평가 실행

```bash
python evaluate.py \
    --checkpoint checkpoints_full/checkpoints/best_model.pt \
    --config configs/full_gpu.yaml \
    --validation-data /path/to/validation/data \
    --normal-ratio 0.8 \
    --generate-fake-anomaly
```

### 평가 옵션

| Option | Description | Default |
|--------|-------------|---------|
| `--checkpoint` | 평가할 모델 체크포인트 경로 | (required) |
| `--config` | 학습 시 사용한 설정 파일 | (required) |
| `--validation-data` | 검증용 데이터 경로 (파일 또는 디렉토리) | (required) |
| `--normal-ratio` | 정상 데이터 비율 | 0.8 |
| `--max-samples` | 샘플 수 제한 (빠른 평가용) | None |
| `--output-dir` | 결과 저장 디렉토리 | `evaluation_results` |
| `--generate-fake-anomaly` | Pseudo-Anomaly 생성 모드 | False |
| `--anomaly-ratio` | 토큰 변조 비율 | 0.1 |
| `--batch-size` | 평가 배치 크기 | 32 |

### 평가 결과

평가 완료 후 다음 파일들이 생성됩니다:

```
evaluation_results/<checkpoint_name>/
├── evaluation_results_<name>.json   # 평가 메트릭 (JSON)
├── score_dist_<name>.png            # 점수 분포 그래프
└── confusion_matrix_<name>.png      # 혼동 행렬 히트맵
```

### 평가 메트릭

- **Accuracy**: 전체 예측 중 정확한 예측 비율
- **Precision**: 이상으로 예측한 것 중 실제 이상 비율
- **Recall**: 실제 이상 중 올바르게 탐지한 비율
- **F1-Score**: Precision과 Recall의 조화평균
- **ROC AUC**: ROC 곡선 아래 면적
- **Confusion Matrix**: Normal/Anomaly 예측 혼동 행렬

## 🔧 문제 해결

### 메모리 부족
- `batch_size` 줄이기 (64 → 32 → 16)
- `num_workers` 줄이기
- `limit_files` 줄이기

### 학습 속도 느림
- `num_workers` 늘리기
- `batch_size` 늘리기 (GPU 메모리 허용 시)
- Multi-GPU 사용 (자동 감지)

## 📚 참고

- Vocab size: 586 (설정: 10000)
- 최대 시퀀스 길이: 512
- BERT 아키텍처: 12 layers, 768 hidden, 12 heads
