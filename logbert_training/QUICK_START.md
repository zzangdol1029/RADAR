# LogBERT 학습 빠른 시작 가이드

## 1. 환경 설정

```bash
# Conda 환경 활성화
conda activate radar

# 의존성 설치
cd logbert_training
pip install -r requirements.txt
```

## 2. 기본 학습 실행

```bash
python train.py
```

이 명령은 다음을 수행합니다:
- `../preprocessing/output`에서 전처리된 데이터 로드
- `training_config.yaml`의 설정 사용
- `checkpoints/` 디렉토리에 모델 저장

## 3. 설정 커스터마이징

`training_config.yaml` 파일을 수정하여 학습 설정을 변경할 수 있습니다:

```yaml
training:
  batch_size: 64        # 배치 크기 증가
  learning_rate: 1e-5   # 학습률 조정
  num_epochs: 20        # 에폭 수 증가
```

## 4. 학습 모니터링

학습 중 다음 정보가 출력됩니다:
- 현재 Loss
- 평균 Loss
- 학습률
- 진행 상황

## 5. 체크포인트 확인

```bash
ls -lh checkpoints/
```

주요 파일:
- `best_model.pt`: 최고 성능 모델
- `checkpoint_step_*.pt`: 주기적 체크포인트
- `epoch_*.pt`: 에폭별 체크포인트

## 6. 모델 사용 (추론)

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pt \
  --input ../preprocessing/output/preprocessed_logs_2025-02-24.json \
  --output results.json \
  --threshold 2.0
```

## 7. 문제 해결

### CUDA Out of Memory
`training_config.yaml`에서 `batch_size`를 줄이세요:
```yaml
training:
  batch_size: 16
```

### 학습이 너무 느림
- GPU가 사용 중인지 확인: `nvidia-smi`
- `num_workers` 증가
- `batch_size` 증가

### Loss가 수렴하지 않음
- 학습률 낮추기: `learning_rate: 1e-5`
- 에폭 수 증가: `num_epochs: 20`

