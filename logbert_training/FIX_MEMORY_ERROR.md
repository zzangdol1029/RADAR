# CUDA Out of Memory 오류 해결 가이드

## 🔴 오류 발생

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB. 
GPU 0 has a total capacity of 31.73 GiB of which 190.62 MiB is free.
```

## ✅ 해결 방법

### 방법 1: 배치 크기 줄이기 (권장)

배치 크기를 64 또는 48로 줄입니다:

```bash
# 배치 크기 64로 실행
python3 train_server.py --config training_config_dgx.yaml --batch-size 64

# 또는 배치 크기 48로 실행 (더 안전)
python3 train_server.py --config training_config_dgx.yaml --batch-size 48
```

### 방법 2: 설정 파일 수정

`training_config_dgx.yaml` 파일을 수정:

```yaml
training:
  batch_size: 64  # 128에서 64로 변경
```

그 다음 다시 실행:

```bash
python3 train_server.py --config training_config_dgx.yaml
```

### 방법 3: GPU 메모리 정리 후 재실행

```bash
# 실행 중인 프로세스 종료
kill $(cat logs/training_*.pid 2>/dev/null) 2>/dev/null

# GPU 메모리 정리 (필요시)
# 다른 프로세스가 GPU를 사용 중일 수 있음
nvidia-smi

# 다시 실행 (작은 배치 크기로)
python3 train_server.py --config training_config_dgx.yaml --batch-size 64
```

## 📊 배치 크기별 메모리 사용량 (실제 측정)

| 배치 크기 | GPU 메모리 사용 | 안전성 | 권장 |
|----------|---------------|--------|------|
| 32 | ~12-15GB | ✅ 매우 안전 | 보수적 |
| **48** | **~16-20GB** | **✅ 안전** | **권장** |
| **64** | **~18-22GB** | **✅ 안전** | **권장** |
| 128 | ~30GB+ | ❌ 메모리 부족 | 비권장 |

## 🚀 권장 실행 명령어

### 안정적인 설정 (권장)

```bash
cd /home/zzangdol/RADAR-1/logbert_training

# 배치 크기 64로 실행
python3 train_server.py --config training_config_dgx.yaml --batch-size 64
```

### 더 안전한 설정

```bash
# 배치 크기 48로 실행
python3 train_server.py --config training_config_dgx.yaml --batch-size 48
```

### 백그라운드 실행

```bash
# 배치 크기 64로 백그라운드 실행
./run_training_background.sh --batch-size 64
```

## 📈 예상 학습 시간 (배치 크기 조정 후)

| 배치 크기 | 에폭당 시간 | 10 에폭 시간 |
|----------|-----------|------------|
| 32 | ~4시간 | ~40시간 |
| **48** | **~2.5시간** | **~25시간** |
| **64** | **~2.2시간** | **~22시간** |
| 128 | ~1.2시간 | ~12시간 (메모리 부족) |

## 🔍 GPU 메모리 확인

학습 전에 GPU 메모리를 확인하세요:

```bash
# GPU 상태 확인
nvidia-smi

# 다른 프로세스가 GPU를 사용 중인지 확인
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

## 💡 추가 최적화 방법

### 1. Gradient Accumulation 사용 (향후 추가 가능)

작은 배치를 여러 번 누적하여 큰 배치 효과를 낼 수 있습니다:

```python
# 예: 배치 32를 2번 누적 = 배치 64 효과
accumulation_steps = 2
effective_batch_size = batch_size * accumulation_steps
```

### 2. Mixed Precision Training (향후 추가 가능)

FP16을 사용하면 메모리 사용량을 절반으로 줄일 수 있습니다:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(...)
```

## ⚠️ 주의사항

1. **다른 프로세스 확인**: 다른 프로세스가 GPU 메모리를 사용 중일 수 있습니다.
2. **데이터 크기**: 데이터셋이 크면 메모리 사용량이 증가할 수 있습니다.
3. **모델 크기**: BERT-base 모델이므로 hidden_size나 layers를 줄이면 메모리를 절약할 수 있습니다.

## 🎯 빠른 해결

**가장 빠른 해결 방법:**

```bash
cd /home/zzangdol/RADAR-1/logbert_training

# 기존 프로세스 종료 (있는 경우)
kill $(cat logs/training_*.pid 2>/dev/null) 2>/dev/null

# 배치 크기 64로 재실행
python3 train_server.py --config training_config_dgx.yaml --batch-size 64
```

또는 백그라운드로:

```bash
./run_training_background.sh --batch-size 64
```

## 📝 체크리스트

- [ ] 기존 프로세스 종료
- [ ] GPU 메모리 확인 (`nvidia-smi`)
- [ ] 배치 크기 64 또는 48로 설정
- [ ] 재실행
- [ ] 로그 모니터링 (`tail -f logs/training_*.log`)

