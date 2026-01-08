# 멀티 GPU 병렬 처리 확인 가이드

## ✅ 코드상 설정 확인

### 현재 코드 설정

`train.py`에서 확인된 설정:

```python
# 멀티 GPU 설정
self.use_multi_gpu = config.get('training', {}).get('use_multi_gpu', True)
self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

if self.num_gpus > 1 and self.use_multi_gpu:
    logger.info(f"멀티 GPU 모드: {self.num_gpus}개 GPU 사용")
    logger.info(f"사용 디바이스: cuda (GPU 0-{self.num_gpus-1})")
    
    # DataParallel로 모델 래핑
    self.model = nn.DataParallel(self.model)
    logger.info("✅ 멀티 GPU 설정 완료")
```

**코드상으로는 병렬 처리가 설정되어 있습니다!** ✅

---

## 🔍 실제 병렬 처리 확인 방법

### 방법 1: 로그 확인

학습 시작 시 로그에서 다음 메시지를 확인하세요:

```bash
# 로그 파일 확인
tail -f logs/training_*.log | grep -E "멀티 GPU|DataParallel|GPU"

# 또는 전체 로그 확인
grep -E "멀티 GPU|DataParallel|각 GPU당" logs/training_*.log
```

**예상 로그:**
```
멀티 GPU 모드: 4개 GPU 사용
사용 디바이스: cuda (GPU 0-3)
DataParallel로 4개 GPU에 모델 배포 중...
✅ 멀티 GPU 설정 완료
배치 크기: 128
  → 각 GPU당 배치 크기: 32 (총 4개 GPU)
멀티 GPU: 4개 GPU 사용 중
```

**이 로그가 나타나면 병렬 처리가 활성화된 것입니다!** ✅

---

### 방법 2: GPU 사용량 확인 (가장 확실한 방법)

**실시간 GPU 모니터링:**

```bash
# 실시간 GPU 사용량 확인
watch -n 1 nvidia-smi

# 또는 한 번만 확인
nvidia-smi
```

**병렬 처리 중일 때 예상 결과:**

```
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-DGXS-32GB  On | 00000000:07:00.0 On |                  N/A |
| N/A   65C    P0   250W / 300W |  15000MiB / 32768MiB |    85%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-DGXS-32GB  On | 00000000:08:00.0 Off|                  N/A |
| N/A   65C    P0   250W / 300W |  15000MiB / 32768MiB |    85%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-DGXS-32GB  On | 00000000:0E:00.0 Off|                  N/A |
| N/A   65C    P0   250W / 300W |  15000MiB / 32768MiB |    85%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-DGXS-32GB  On | 00000000:0F:00.0 Off|                  N/A |
| N/A   65C    P0   250W / 300W |  15000MiB / 32768MiB |    85%      Default |
+-----------------------------------------------------------------------------+
```

**확인 사항:**
- ✅ GPU 0, 1, 2, 3 모두 **GPU-Util이 80-90%**
- ✅ 각 GPU 메모리 사용량이 비슷함 (~12-15GB)
- ✅ 전력 사용량이 높음 (250W/300W)

**만약 GPU 0만 사용 중이면:**
- GPU 1, 2, 3의 GPU-Util이 0%
- 메모리 사용량이 거의 없음
- → 병렬 처리가 안 되고 있음 ❌

---

### 방법 3: Python으로 직접 확인

```bash
# 서버에서 실행
python3 -c "
import torch
print(f'GPU 개수: {torch.cuda.device_count()}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

**예상 출력:**
```
GPU 개수: 4
CUDA 사용 가능: True
GPU 0: Tesla V100-DGXS-32GB
GPU 1: Tesla V100-DGXS-32GB
GPU 2: Tesla V100-DGXS-32GB
GPU 3: Tesla V100-DGXS-32GB
```

---

### 방법 4: 프로세스 확인

```bash
# 학습 프로세스가 사용하는 GPU 확인
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv

# 또는
fuser -v /dev/nvidia*
```

**병렬 처리 중일 때:**
- 같은 PID가 여러 GPU를 사용 중
- 각 GPU에 메모리가 할당됨

---

## 📊 DataParallel 동작 방식

### DataParallel이 하는 일

```
배치 크기 128 입력
  ↓
DataParallel이 자동으로 분산
  ↓
GPU 0: 배치 32개 처리
GPU 1: 배치 32개 처리
GPU 2: 배치 32개 처리
GPU 3: 배치 32개 처리
  ↓
각 GPU에서 Forward Pass
  ↓
결과를 GPU 0으로 수집 (Gather)
  ↓
Loss 계산 및 Backward Pass
  ↓
Gradient를 각 GPU로 분산
  ↓
각 GPU에서 가중치 업데이트
```

**핵심:**
- 배치가 자동으로 GPU 개수로 나뉨
- 각 GPU가 병렬로 처리
- 결과를 수집하여 학습 진행

---

## ⚠️ 병렬 처리가 안 되는 경우

### 확인 사항

1. **설정 파일 확인**
   ```yaml
   training:
     use_multi_gpu: true  # true인지 확인
   ```

2. **GPU 개수 확인**
   ```bash
   python3 -c "import torch; print(torch.cuda.device_count())"
   # 4가 나와야 함
   ```

3. **로그 확인**
   - "멀티 GPU 모드: 4개 GPU 사용" 메시지 확인
   - "DataParallel로 4개 GPU에 모델 배포 중..." 메시지 확인

4. **nvidia-smi 확인**
   - GPU 0만 사용 중이면 병렬 처리 안 됨
   - 모든 GPU가 사용 중이어야 함

---

## 🎯 현재 상태 확인 명령어

### 빠른 확인

```bash
# 1. GPU 사용량 확인 (가장 확실)
nvidia-smi

# 2. 로그 확인
grep -E "멀티 GPU|각 GPU당" logs/training_*.log | head -5

# 3. Python으로 확인
python3 -c "import torch; print(f'GPU: {torch.cuda.device_count()}개')"
```

---

## 💡 예상 결과

### 병렬 처리 중일 때

**로그:**
```
멀티 GPU 모드: 4개 GPU 사용
사용 디바이스: cuda (GPU 0-3)
DataParallel로 4개 GPU에 모델 배포 중...
✅ 멀티 GPU 설정 완료
배치 크기: 128
  → 각 GPU당 배치 크기: 32 (총 4개 GPU)
```

**nvidia-smi:**
- GPU 0, 1, 2, 3 모두 GPU-Util 80-90%
- 각 GPU 메모리 ~12-15GB 사용

**성능:**
- 배치당 시간: ~1.06초
- 에폭당 시간: ~5.67시간

### 병렬 처리 안 될 때

**로그:**
```
사용 디바이스: cuda
(멀티 GPU 관련 메시지 없음)
```

**nvidia-smi:**
- GPU 0만 GPU-Util 80-90%
- GPU 1, 2, 3은 GPU-Util 0%

**성능:**
- 배치당 시간: ~4초 (4배 느림)
- 에폭당 시간: ~22시간 (4배 느림)

---

## 🔧 문제 해결

### 병렬 처리가 안 되는 경우

1. **설정 확인**
   ```yaml
   training:
     use_multi_gpu: true
   ```

2. **재시작**
   ```bash
   # 기존 프로세스 종료
   kill $(cat logs/training_*.pid 2>/dev/null) 2>/dev/null
   
   # 재시작
   python3 train_server.py --config training_config_dgx.yaml
   ```

3. **강제 설정**
   ```bash
   # 환경 변수로 확인
   echo $CUDA_VISIBLE_DEVICES
   # 비어있어야 함 (모든 GPU 사용)
   ```

---

## 📝 요약

### 현재 코드 상태

✅ **코드상으로는 병렬 처리가 설정되어 있습니다!**

- `use_multi_gpu: true` 설정됨
- `DataParallel` 사용
- GPU 개수 자동 감지

### 실제 확인 방법

1. **로그 확인** (가장 쉬움)
   ```bash
   grep "멀티 GPU\|각 GPU당" logs/training_*.log
   ```

2. **nvidia-smi 확인** (가장 확실)
   ```bash
   nvidia-smi
   # GPU 0, 1, 2, 3 모두 사용 중이면 병렬 처리 중
   ```

3. **성능 확인**
   - 배치당 시간 ~1.06초 → 병렬 처리 중 ✅
   - 배치당 시간 ~4초 → 병렬 처리 안 됨 ❌

**지금 바로 확인하세요:**
```bash
nvidia-smi
```

**GPU 0, 1, 2, 3 모두 사용 중이면 병렬 처리가 정상 작동 중입니다!** 🚀

