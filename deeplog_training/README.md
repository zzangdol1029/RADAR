# DeepLog 모델 학습

LSTM 기반 로그 이상 탐지 모델 **DeepLog**의 학습 모듈입니다.

> **논문**: "DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning" (CCS 2017)

## 🌟 주요 기능

### 1. Lazy Loading 기반 대용량 데이터 처리
- **120GB 데이터도 OOM 없이 처리** 가능
- `ijson` 라이브러리를 사용한 스트리밍 JSON 파싱
- 배치 단위 데이터 로드로 메모리 효율 극대화

### 2. 멀티 GPU 지원
- **DataParallel**을 통한 4 GPU 분산 학습
- Tesla V100-DGXS-32GB x 4 환경 최적화

### 3. 상세한 학습 모니터링
- 실시간 GPU 메모리/활용률/온도/전력 추적
- 배치별/에폭별 loss 및 시간 로깅
- 예상 남은 시간(ETA) 표시

### 4. 안정적인 학습
- Mixed Precision Training (FP16) 지원
- Early Stopping
- 체크포인트 저장/복원

---

## 📁 파일 구조

```
deeplog_training/
├── config.yaml          # 학습 설정 파일
├── model.py             # DeepLog 모델 정의
├── dataset.py           # Lazy Loading Dataset
├── train.py             # 메인 학습 스크립트
├── evaluate.py          # 모델 성능 평가 스크립트
├── utils.py             # GPU 모니터링, Early Stopping 등
├── requirements.txt     # 의존성 목록
└── README.md            # 사용 가이드
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 의존성 설치
cd deeplog_training
pip install -r requirements.txt

# ijson 필수 설치 (Lazy Loading의 핵심)
pip install ijson
```

### 2. 설정 확인

`config.yaml` 파일에서 데이터 경로를 확인/수정합니다:

```yaml
data:
  preprocessed_dir: "/home/zzangdol/RADAR/preprocessing/output"
  max_seq_length: 512
  validation_split: 0.1
```

### 3. 학습 실행

```bash
# 기본 실행
python train.py

# 설정 파일 지정
python train.py --config config.yaml

# 데이터 경로 직접 지정
python train.py --data-dir /path/to/data

# 하이퍼파라미터 오버라이드
python train.py --epochs 100 --batch-size 128 --lr 0.0005

# 체크포인트에서 재개
python train.py --resume outputs/checkpoints/epoch_10.pt
```

### 4. 서버에서 백그라운드 실행

```bash
# nohup 사용
nohup python train.py > training.log 2>&1 &

# 또는 screen 사용
screen -S deeplog_training
python train.py
# Ctrl+A, D로 분리
```

---

## ⚙️ 설정 가이드

### 모델 설정 (`model`)

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `vocab_size` | 10000 | 어휘 크기 (Event ID 수) |
| `embedding_dim` | 128 | 임베딩 차원 |
| `hidden_size` | 256 | LSTM 은닉층 크기 |
| `num_layers` | 2 | LSTM 레이어 수 |
| `dropout` | 0.2 | 드롭아웃 확률 |

### 학습 설정 (`training`)

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `batch_size` | 256 | 배치 크기 (4 GPU 기준, GPU당 64) |
| `learning_rate` | 0.001 | 초기 학습률 |
| `num_epochs` | 50 | 총 에폭 수 |
| `max_grad_norm` | 1.0 | Gradient Clipping |
| `use_multi_gpu` | true | 멀티 GPU 사용 |
| `mixed_precision` | true | FP16 학습 |

### Early Stopping 설정

```yaml
training:
  early_stopping:
    enabled: true
    patience: 5        # 개선 없이 지속될 수 있는 에폭 수
    min_delta: 0.0001  # 개선으로 간주되는 최소 변화량
```

### Lazy Loading 설정

```yaml
data:
  lazy_loading:
    enabled: true
    buffer_size: 10000    # 메모리에 유지할 샘플 수
    shuffle_buffer: true  # 버퍼 내 셔플 여부
```

---

## 📊 학습 모니터링

### 콘솔 출력 예시

```
2026-02-08 20:30:15 | INFO | [Step 1000] Loss: 2.3456 (avg: 2.4567) | LR: 9.90e-04 | GPU: [0:12345MB|95%] [1:12340MB|94%] | Time: 10.5m (ETA: 2h 30m)

GPU 상태 (Step 1000):
  GPU 0: 메모리: 12345/32510MB (38.0%) | 활용률: 95% | 온도: 65°C | 전력: 250W
  GPU 1: 메모리: 12340/32510MB (37.9%) | 활용률: 94% | 온도: 64°C | 전력: 248W
  GPU 2: 메모리: 12350/32510MB (38.0%) | 활용률: 96% | 온도: 66°C | 전력: 252W
  GPU 3: 메모리: 12342/32510MB (37.9%) | 활용률: 93% | 온도: 63°C | 전력: 245W
```

### 출력 파일

학습 완료 후 다음 위치에 파일이 저장됩니다:

```
/home/zzangdol/silverw/deeplog/
├── training.log                     # 전체 학습 로그
├── training_history.json            # 학습 이력 (loss, lr 등)
├── evaluation_results_YYYYMMDD.json # 성능 평가 결과 (JSON)
├── evaluation_report_YYYYMMDD.txt   # 성능 평가 리포트 (텍스트)
└── output/
    └── checkpoints/
        ├── best_model.pt            # 최고 성능 모델
        ├── epoch_1.pt               # 에폭별 체크포인트
        ├── epoch_2.pt
        └── step_5000.pt             # 스텝별 체크포인트
```

---

## 🔧 메모리 최적화 가이드

### OOM 발생 시 조치

1. **배치 크기 감소**
   ```bash
   python train.py --batch-size 128
   ```

2. **버퍼 크기 감소** (config.yaml)
   ```yaml
   data:
     lazy_loading:
       buffer_size: 5000
   ```

3. **시퀀스 길이 감소**
   ```yaml
   data:
     max_seq_length: 256
   ```

4. **Mixed Precision 활성화** (기본 활성화)
   ```yaml
   training:
     mixed_precision: true
   ```

---

## 📈 학습 결과 분석

`training_history.json` 파일을 사용하여 학습 곡선을 시각화할 수 있습니다:

```python
import json
import matplotlib.pyplot as plt

with open('outputs/training_history.json', 'r') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
if history['val_loss']:
    plt.plot(history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(history['learning_rate'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')

plt.tight_layout()
plt.savefig('training_curve.png')
plt.show()
```

---

## 📊 모델 성능 평가

학습 완료 후 자동으로 성능 평가가 실행됩니다. 수동으로 실행하려면:

```bash
# 최고 모델 평가
python evaluate.py --checkpoint /home/zzangdol/silverw/deeplog/output/checkpoints/best_model.pt

# 특정 체크포인트 평가
python evaluate.py --checkpoint /path/to/checkpoint.pt --output-dir /home/zzangdol/silverw/deeplog
```

### 평가 지표

1. **Top-k Accuracy**: 다음 로그 예측이 top-k 안에 있는 비율
   - Top-1, Top-5, Top-10, Top-20 정확도 측정

2. **이상 탐지 성능** (이상 데이터가 있는 경우):
   - Precision, Recall, F1 Score
   - False Positive Rate
   - 다양한 임계값(P90, P95, P99)에서의 성능

### 평가 결과 예시

```
================================================================================
DeepLog 모델 성능 평가 리포트
================================================================================
평가 시간: 2026-02-08 22:30:00

[ 다음 로그 예측 정확도 ]
  - Evaluation Loss: 1.2345
  - Total Predictions: 1,000,000
  - Top-1 Accuracy: 0.6523 (65.23%)
  - Top-5 Accuracy: 0.8234 (82.34%)
  - Top-10 Accuracy: 0.8912 (89.12%)
  - Top-20 Accuracy: 0.9234 (92.34%)

[ 이상 점수 통계 (정상 데이터) ]
  - 평균: 0.1234
  - 표준편차: 0.0567
  - 샘플 수: 50,000

[ 추천 임계값 ]
  - p90: 0.2345
  - p95: 0.3456
  - p99: 0.4567
================================================================================
```

---

## 🧪 이상 탐지 (Inference)

학습된 모델을 사용하여 로그 이상을 탐지합니다:

```python
import torch
from model import DeepLog

# 모델 로드
model = DeepLog(vocab_size=10000, hidden_size=256, num_layers=2)
checkpoint = torch.load('outputs/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 이상 점수 계산 (top-k 방식)
input_ids = torch.tensor([[1, 5, 3, 7, 2, 8]])  # 로그 시퀀스
anomaly_scores = model.calculate_anomaly_score(input_ids, top_k=10)
print(f"이상 점수: {anomaly_scores.item():.4f}")

# 이상 점수 > 임계값이면 이상으로 판단
threshold = 0.3
is_anomaly = anomaly_scores.item() > threshold
print(f"이상 여부: {is_anomaly}")
```

---

## 🐛 문제 해결

### ijson 설치 오류

```bash
# yajl 라이브러리 필요
sudo apt-get install libyajl-dev  # Ubuntu
pip install ijson
```

### GPU 메모리 부족

```bash
# 현재 GPU 메모리 확인
nvidia-smi

# Python에서 캐시 정리
import torch
torch.cuda.empty_cache()
```

### 체크포인트 로드 오류

```python
# CPU에서 로드
checkpoint = torch.load('checkpoint.pt', map_location='cpu')
```

---

## 📚 참고 자료

- **DeepLog 논문**: [CCS 2017](https://dl.acm.org/doi/10.1145/3133956.3134015)
- **PyTorch LSTM**: [공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- **ijson**: [GitHub](https://github.com/ICRAR/ijson)
