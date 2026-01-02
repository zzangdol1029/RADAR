# LogBERT 학습 모듈

로그 이상 탐지를 위한 LogBERT 모델 학습 모듈입니다.

## 개요

LogBERT는 **Masked Language Modeling (MLM)** 방식을 사용한 비지도 학습 모델로, 정상적인 로그 패턴을 학습하여 이상 로그를 탐지합니다.

### 주요 특징

- **비지도 학습**: 레이블이 필요 없는 Self-Supervised Learning
- **BERT 기반**: Transformer 아키텍처를 사용한 강력한 표현 학습
- **MLM 방식**: 마스킹된 토큰을 예측하여 로그 패턴 학습
- **DGX 지원**: Multi-GPU 병렬 학습 지원

## 설치

### 1. Conda 환경 활성화

```bash
conda activate radar
```

### 2. 의존성 설치

```bash
cd logbert_training
pip install -r requirements.txt
```

### 3. CUDA 확인 (GPU 사용 시)

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 사용 방법

### 기본 학습

```bash
python train.py
```

### 설정 파일 지정

```bash
python train.py --config training_config.yaml
```

### 데이터 디렉토리 지정

```bash
python train.py --data-dir ../preprocessing/output
```

### 출력 디렉토리 지정

```bash
python train.py --output-dir ./my_checkpoints
```

## 설정 파일

`training_config.yaml` 파일에서 학습 설정을 변경할 수 있습니다:

```yaml
model:
  vocab_size: 10000          # 어휘 크기
  hidden_size: 768           # 은닉층 크기
  num_hidden_layers: 12      # 레이어 수
  num_attention_heads: 12    # 어텐션 헤드 수

training:
  batch_size: 32             # 배치 크기
  learning_rate: 2e-5        # 학습률
  num_epochs: 10             # 에폭 수
  mask_prob: 0.15           # 마스킹 확률
```

## 학습 과정

### 1. 데이터 로드

전처리된 JSON 파일에서 세션 데이터를 로드합니다:
- `token_ids`: BERT 입력 토큰
- `attention_mask`: 패딩 마스크
- `event_sequence`: Event ID 시퀀스

### 2. MLM 마스킹

각 배치에서 토큰의 15%를 마스킹합니다:
- 80%: [MASK] 토큰으로 교체
- 10%: 랜덤 토큰으로 교체
- 10%: 원래 토큰 유지

### 3. 모델 학습

- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: AdamW
- **Learning Rate Schedule**: Cosine Annealing

### 4. 체크포인트 저장

- `checkpoint_step_*`: 주기적 체크포인트
- `best_model`: 최고 성능 모델
- `epoch_*`: 에폭별 체크포인트

## 출력 파일

학습이 완료되면 `checkpoints/` 디렉토리에 다음 파일이 생성됩니다:

```
checkpoints/
├── checkpoint_step_1000.pt
├── checkpoint_step_2000.pt
├── ...
├── best_model.pt
└── epoch_10.pt
```

## 체크포인트 구조

각 체크포인트 파일은 다음 정보를 포함합니다:

```python
{
    'model_state_dict': {...},      # 모델 가중치
    'optimizer_state_dict': {...},  # 옵티마이저 상태
    'scheduler_state_dict': {...},  # 스케줄러 상태
    'global_step': 10000,           # 현재 스텝
    'best_loss': 0.1234,            # 최고 성능 Loss
    'config': {...}                 # 학습 설정
}
```

## 모델 사용 (추론)

학습된 모델을 사용하여 이상 점수를 계산할 수 있습니다:

```python
from model import LogBERT
import torch

# 모델 로드
checkpoint = torch.load('checkpoints/best_model.pt')
model = LogBERT(**checkpoint['config']['model'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 이상 점수 계산
input_ids = torch.tensor([...])  # 토큰 ID
attention_mask = torch.tensor([...])  # 어텐션 마스크

anomaly_scores = model.predict_anomaly_score(input_ids, attention_mask)
```

## 성능 최적화

### GPU 메모리 최적화

- `batch_size`를 줄이거나
- `gradient_accumulation_steps`를 사용하거나
- `mixed_precision` (FP16) 사용

### 학습 속도 최적화

- `num_workers` 증가 (데이터 로딩 병렬화)
- `pin_memory=True` (GPU 전송 최적화)
- Multi-GPU 학습 (DataParallel 또는 DistributedDataParallel)

## 문제 해결

### CUDA Out of Memory

```yaml
training:
  batch_size: 16  # 배치 크기 줄이기
```

### 학습이 너무 느림

```yaml
training:
  num_workers: 8  # 워커 수 증가
  batch_size: 64   # 배치 크기 증가
```

### Loss가 수렴하지 않음

```yaml
training:
  learning_rate: 1e-5  # 학습률 낮추기
  weight_decay: 0.01    # 정규화 강화
```

## 논문 작성 팁

논문에 이 내용을 서술할 때 다음 포인트를 강조하세요:

1. **비지도 학습의 필요성**: "레이블링된 데이터를 구하기 어려운 현실적인 상황을 고려하여, 비지도 학습 방식인 MLM을 채택했다."

2. **BERT의 적합성**: "Transformer 아키텍처를 사용하여 로그 시퀀스의 장거리 의존성을 효과적으로 모델링할 수 있다."

3. **DGX 활용**: "대규모 연산이 필요한 LogBERT 학습을 위해 NVIDIA DGX Station의 Multi-GPU 병렬 처리를 활용하여 실시간성을 확보했다."

## 다음 단계

학습이 완료되면:

1. **모델 평가**: 검증 데이터셋으로 성능 평가
2. **이상 탐지**: 실시간 로그에 모델 적용
3. **임계값 설정**: 이상 점수 기준 설정

## 참고 자료

- [BERT 논문](https://arxiv.org/abs/1810.04805)
- [LogBERT 논문](https://arxiv.org/abs/2001.09244)
- [Transformers 문서](https://huggingface.co/docs/transformers)













