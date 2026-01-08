# 멀티 GPU 학습 가이드

DGX Station의 4개 GPU를 모두 활용한 병렬 학습 가이드입니다.

## 🚀 멀티 GPU 학습의 장점

### 하드웨어 활용
- **GPU**: Tesla V100-DGXS-32GB × 4개
- **총 GPU 메모리**: 128GB (32GB × 4)
- **병렬 처리**: 4배 빠른 학습

### 성능 향상
- **배치 크기**: 단일 GPU 64 → 멀티 GPU 256 (4배)
- **학습 속도**: 약 3-4배 향상
- **GPU 활용률**: 4개 GPU 모두 활용

## 📊 배치 크기 설정

### 멀티 GPU 배치 크기 계산

| 설정 | 각 GPU당 배치 | 총 배치 크기 | GPU 메모리/GPU | 총 메모리 |
|------|------------|------------|--------------|----------|
| 보수적 | 32 | 128 | ~12-15GB | ~48-60GB |
| **권장** | **64** | **256** | **~18-22GB** | **~72-88GB** |
| 공격적 | 96 | 384 | ~25-28GB | ~100-112GB |

### 설정 파일

`training_config_dgx.yaml`에서:

```yaml
training:
  batch_size: 256            # 총 배치 크기 (4개 GPU × 64)
  use_multi_gpu: true        # 멀티 GPU 활성화
```

**각 GPU당 배치 크기**: 256 ÷ 4 = 64

## 🎯 실행 방법

### 기본 실행 (멀티 GPU 자동 활성화)

```bash
cd /home/zzangdol/RADAR-1/logbert_training

# 멀티 GPU 자동 감지 및 사용
python3 train_server.py --config training_config_dgx.yaml
```

### 커스텀 배치 크기

```bash
# 총 배치 크기 256 (각 GPU당 64)
python3 train_server.py --config training_config_dgx.yaml --batch-size 256

# 총 배치 크기 128 (각 GPU당 32) - 보수적
python3 train_server.py --config training_config_dgx.yaml --batch-size 128

# 총 배치 크기 384 (각 GPU당 96) - 공격적
python3 train_server.py --config training_config_dgx.yaml --batch-size 384
```

### 백그라운드 실행

```bash
# 멀티 GPU로 백그라운드 실행
./run_training_background.sh --config training_config_dgx.yaml --batch-size 256
```

## 📈 예상 성능

### 학습 시간 비교

전체 데이터 (약 430,850개 세션) 기준:

| 설정 | 배치 크기 | 에폭당 시간 | 10 에폭 시간 | GPU 활용 |
|------|----------|-----------|------------|---------|
| 단일 GPU | 64 | ~2.2시간 | ~22시간 | 1개 GPU |
| **멀티 GPU** | **256** | **~0.6시간** | **~6시간** | **4개 GPU** |

**멀티 GPU 사용 시 약 3.7배 빠름!**

### GPU 메모리 사용량

| 총 배치 크기 | 각 GPU당 배치 | GPU 메모리/GPU | 총 메모리 |
|------------|------------|--------------|----------|
| 128 | 32 | ~12-15GB | ~48-60GB |
| **256** | **64** | **~18-22GB** | **~72-88GB** |
| 384 | 96 | ~25-28GB | ~100-112GB |

## 🔍 모니터링

### GPU 사용량 확인

```bash
# 모든 GPU 사용량 확인
nvidia-smi

# 실시간 모니터링
watch -n 1 nvidia-smi

# GPU별 상세 정보
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

**정상 작동 시:**
- GPU 0, 1, 2, 3 모두 사용 중
- 각 GPU 메모리: ~18-22GB 사용
- GPU 활용률: 80-90%

### 로그 확인

```bash
# 실시간 로그 확인
tail -f logs/training_*.log

# 멀티 GPU 관련 로그 확인
grep "멀티 GPU\|DataParallel\|GPU" logs/training_*.log
```

**예상 로그:**
```
멀티 GPU 모드: 4개 GPU 사용
DataParallel로 4개 GPU에 모델 배포 중...
✅ 멀티 GPU 설정 완료
각 GPU당 배치 크기: 64 (총 4개 GPU)
```

## ⚙️ 설정 옵션

### 멀티 GPU 비활성화 (단일 GPU 사용)

```bash
# 설정 파일에서 use_multi_gpu: false로 변경하거나
# 명령줄에서 단일 GPU 강제 사용
CUDA_VISIBLE_DEVICES=0 python3 train_server.py --config training_config_dgx.yaml
```

### 특정 GPU만 사용

```bash
# GPU 0, 1만 사용
CUDA_VISIBLE_DEVICES=0,1 python3 train_server.py --config training_config_dgx.yaml

# GPU 0, 2만 사용
CUDA_VISIBLE_DEVICES=0,2 python3 train_server.py --config training_config_dgx.yaml
```

## 🛠️ 문제 해결

### 멀티 GPU가 작동하지 않음

**확인 사항:**
```bash
# GPU 개수 확인
python3 -c "import torch; print(f'GPU 개수: {torch.cuda.device_count()}')"

# 각 GPU 확인
nvidia-smi
```

**해결 방법:**
- `use_multi_gpu: true` 확인
- GPU가 2개 이상인지 확인
- CUDA가 정상 작동하는지 확인

### GPU 메모리 불균형

DataParallel은 자동으로 배치를 분산하므로 일반적으로 문제없습니다. 만약 문제가 있다면:

```bash
# 배치 크기 줄이기
python3 train_server.py --config training_config_dgx.yaml --batch-size 128
```

### 학습 속도가 느림

**확인 사항:**
- 모든 GPU가 사용 중인지 (`nvidia-smi`)
- 데이터 로딩이 병목인지 (`num_workers` 증가)
- 배치 크기가 충분한지

## 💡 최적화 팁

### 1. 배치 크기 조정

- **시작**: 배치 128 (각 GPU당 32)로 테스트
- **안정적**: 배치 256 (각 GPU당 64) - 권장
- **공격적**: 배치 384 (각 GPU당 96) - 메모리 확인 필요

### 2. 데이터 로딩 최적화

```yaml
training:
  num_workers: 8  # GPU 개수 × 2 권장
```

### 3. Mixed Precision (향후 추가 가능)

FP16을 사용하면:
- 메모리 사용량 절반
- 배치 크기 2배 증가 가능
- 학습 속도 1.5-2배 향상

## 📝 체크리스트

멀티 GPU 학습 시작 전:

- [x] GPU 4개 확인: `nvidia-smi` ✅
- [ ] `training_config_dgx.yaml`에서 `use_multi_gpu: true` 확인
- [ ] 배치 크기 설정 (권장: 256)
- [ ] 데이터 파일 확인
- [ ] 의존성 설치 완료

## 🎯 빠른 시작

```bash
cd /home/zzangdol/RADAR-1/logbert_training

# 멀티 GPU로 학습 시작
python3 train_server.py --config training_config_dgx.yaml

# 또는 백그라운드
./run_training_background.sh --config training_config_dgx.yaml
```

**예상 결과:**
- 4개 GPU 모두 사용
- 각 GPU당 배치 64
- 총 배치 크기 256
- 학습 시간: 약 6시간 (10 에폭)

## 🚀 성능 비교

| 항목 | 단일 GPU (64) | 멀티 GPU (256) | 향상 |
|------|-------------|--------------|------|
| 배치 크기 | 64 | 256 | 4배 |
| 에폭당 시간 | ~2.2시간 | ~0.6시간 | 3.7배 빠름 |
| 10 에폭 시간 | ~22시간 | ~6시간 | 3.7배 빠름 |
| GPU 활용 | 1개 | 4개 | 4배 |

**결론: 멀티 GPU 사용 시 약 3-4배 빠른 학습!** 🎉



