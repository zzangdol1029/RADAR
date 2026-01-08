# 멀티 GPU 메모리 오류 해결 가이드

## 🔴 오류 발생

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. 
GPU 0 has a total capacity of 31.73 GiB of which 14.62 MiB is free.
```

멀티 GPU를 사용해도 각 GPU당 배치 크기가 너무 커서 메모리 부족이 발생했습니다.

## ✅ 해결 방법

### 방법 1: 배치 크기 줄이기 (권장)

**현재 설정**: 배치 256 (각 GPU당 64) → 메모리 부족
**권장 설정**: 배치 128 (각 GPU당 32) → 안정적

```bash
cd /home/zzangdol/RADAR-1/logbert_training

# 기존 프로세스 종료
kill $(cat logs/training_*.pid 2>/dev/null) 2>/dev/null

# 배치 크기 128로 재실행
python3 train_server.py --config training_config_dgx.yaml --batch-size 128
```

### 방법 2: 더 보수적인 설정

```bash
# 배치 크기 96 (각 GPU당 24) - 매우 안전
python3 train_server.py --config training_config_dgx.yaml --batch-size 96
```

### 방법 3: 설정 파일 수정

`training_config_dgx.yaml` 파일이 이미 배치 크기 128로 수정되었습니다:

```yaml
training:
  batch_size: 128  # 각 GPU당 32
```

## 📊 배치 크기별 메모리 사용량 (멀티 GPU)

| 총 배치 크기 | 각 GPU당 배치 | GPU 메모리/GPU | 안전성 | 권장 |
|------------|------------|--------------|--------|------|
| 96 | 24 | ~10-12GB | ✅ 매우 안전 | 보수적 |
| **128** | **32** | **~12-15GB** | **✅ 안전** | **권장** |
| 192 | 48 | ~18-22GB | ⚠️ 주의 | 공격적 |
| 256 | 64 | ~25GB+ | ❌ 메모리 부족 | 비권장 |

## 🚀 권장 실행 명령어

### 안정적인 설정 (권장)

```bash
cd /home/zzangdol/RADAR-1/logbert_training

# 기존 프로세스 종료
kill $(cat logs/training_*.pid 2>/dev/null) 2>/dev/null

# 배치 크기 128로 재실행 (각 GPU당 32)
python3 train_server.py --config training_config_dgx.yaml --batch-size 128
```

### 백그라운드 실행

```bash
./run_training_background.sh --config training_config_dgx.yaml --batch-size 128
```

## 📈 예상 학습 시간 (배치 크기 조정 후)

| 총 배치 크기 | 각 GPU당 배치 | 에폭당 시간 | 10 에폭 시간 |
|------------|------------|-----------|------------|
| 96 | 24 | ~1.0시간 | ~10시간 |
| **128** | **32** | **~0.8시간** | **~8시간** |
| 192 | 48 | ~0.6시간 | ~6시간 (메모리 위험) |

## 🔍 GPU 메모리 확인

학습 전에 GPU 메모리를 확인하세요:

```bash
# GPU 상태 확인
nvidia-smi

# 다른 프로세스가 GPU를 사용 중인지 확인
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

## 💡 추가 최적화 방법

### 1. Gradient Accumulation (향후 추가 가능)

작은 배치를 여러 번 누적하여 큰 배치 효과를 낼 수 있습니다:

```python
# 예: 배치 32를 4번 누적 = 배치 128 효과
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps
```

### 2. Mixed Precision Training (FP16)

FP16을 사용하면 메모리 사용량을 절반으로 줄일 수 있습니다:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(...)
```

**효과:**
- 메모리 사용량: 50% 감소
- 배치 크기: 2배 증가 가능 (128 → 256)
- 학습 속도: 1.5-2배 향상

### 3. 환경 변수 설정

메모리 단편화 문제 해결:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 train_server.py --config training_config_dgx.yaml --batch-size 128
```

## ⚠️ 주의사항

1. **DataParallel 동작**: 배치가 자동으로 GPU 개수로 나뉩니다
   - 배치 128 → 각 GPU당 32
   - 배치 256 → 각 GPU당 64 (메모리 부족)

2. **다른 프로세스**: 다른 프로세스가 GPU 메모리를 사용 중일 수 있습니다

3. **데이터 크기**: 데이터셋이 크면 메모리 사용량이 증가할 수 있습니다

## 🎯 빠른 해결

**가장 빠른 해결 방법:**

```bash
cd /home/zzangdol/RADAR-1/logbert_training

# 기존 프로세스 종료
kill $(cat logs/training_*.pid 2>/dev/null) 2>/dev/null

# 배치 크기 128로 재실행
python3 train_server.py --config training_config_dgx.yaml --batch-size 128
```

또는 백그라운드로:

```bash
./run_training_background.sh --config training_config_dgx.yaml --batch-size 128
```

## 📝 체크리스트

- [x] 배치 크기 128로 설정 완료 ✅
- [ ] 기존 프로세스 종료
- [ ] GPU 메모리 확인 (`nvidia-smi`)
- [ ] 재실행
- [ ] 로그 모니터링 (`tail -f logs/training_*.log`)

## 🚀 예상 결과

배치 크기 128 (각 GPU당 32)로 실행 시:

- ✅ 4개 GPU 모두 사용
- ✅ 각 GPU 메모리: ~12-15GB 사용 (안전)
- ✅ 학습 시간: 약 8시간 (10 에폭)
- ✅ 안정적인 학습

**멀티 GPU를 사용하더라도 각 GPU당 배치 크기를 적절히 설정해야 합니다!**



