# Tesla V100 32GB 최적화 가이드

Tesla V100-DGXS-32GB 4개를 활용한 최적화 설정 가이드입니다.

## 🔧 하드웨어 사양

- **GPU**: Tesla V100-DGXS-32GB × 4개
- **GPU 메모리**: 각 32GB (총 128GB)
- **CUDA**: 12.2
- **Driver**: 535.247.01

## 📊 배치 크기별 메모리 사용량 (예상)

BERT-base 모델 (hidden_size: 768, layers: 12, seq_len: 512) 기준:

| 배치 크기 | GPU 메모리 사용 | GPU 활용률 | 학습 속도 | 권장 |
|----------|---------------|-----------|----------|------|
| 32 | ~12-15GB | 20-30% | 1x (기준) | ❌ 비권장 |
| 64 | ~18-22GB | 40-50% | 1.8x | ⚠️ 보통 |
| **128** | **~20-25GB** | **80-90%** | **3.5x** | **✅ 권장** |
| 192 | ~28-32GB | 90-95% | 5x | ⚠️ 메모리 여유 없음 |
| 256 | ~35-40GB | 95%+ | 6x | ❌ 메모리 부족 가능 |

## 🎯 권장 설정

### 설정 1: 안정적인 설정 (권장) ✅

```bash
python train_server.py --config training_config_dgx.yaml
# 또는
python train_server.py --batch-size 128 --epochs 10
```

**특징:**
- 배치 크기: 128
- GPU 메모리: ~20-25GB 사용 (32GB 중 약 75%)
- 안정적이고 빠른 학습
- 메모리 여유: ~7GB

### 설정 2: 공격적인 설정 (최대 성능)

```bash
python train_server.py --batch-size 192 --epochs 10
```

**특징:**
- 배치 크기: 192
- GPU 메모리: ~28-32GB 사용 (32GB 중 약 90%)
- 최대 성능
- ⚠️ 메모리 여유 거의 없음 (주의 필요)

### 설정 3: 보수적인 설정 (안전)

```bash
python train_server.py --batch-size 64 --epochs 10
```

**특징:**
- 배치 크기: 64
- GPU 메모리: ~18-22GB 사용
- 매우 안전하지만 상대적으로 느림

## 📈 예상 학습 시간

전체 데이터 (약 430,850개 세션) 기준:

| 배치 크기 | 에폭당 시간 | 10 에폭 시간 | GPU 활용률 |
|----------|-----------|------------|-----------|
| 32 | ~4시간 | ~40시간 | 20-30% |
| 64 | ~2.2시간 | ~22시간 | 40-50% |
| **128** | **~1.2시간** | **~12시간** | **80-90%** |
| 192 | ~0.8시간 | ~8시간 | 90-95% |

## 🚀 실행 방법

### 기본 실행 (권장)

```bash
cd /home/zzangdol/RADAR-1/logbert_training

# DGX 최적화 설정 사용
python train_server.py --config training_config_dgx.yaml
```

### 커스텀 배치 크기

```bash
# 안정적인 설정
python train_server.py --batch-size 128 --epochs 10

# 공격적인 설정 (메모리 확인 필요)
python train_server.py --batch-size 192 --epochs 10
```

### 백그라운드 실행

```bash
# 백그라운드 실행 및 로그 저장
nohup python train_server.py --config training_config_dgx.yaml > training.log 2>&1 &

# 프로세스 ID 확인
echo $!

# 로그 실시간 확인
tail -f training.log
```

## 🔍 GPU 모니터링

### 실시간 모니터링

```bash
# GPU 사용량 실시간 확인
watch -n 1 nvidia-smi

# 또는
nvidia-smi -l 1
```

### 메모리 사용량 확인

학습 시작 후 `nvidia-smi`로 확인:
- **배치 128**: Memory-Usage가 약 20-25GB 정도여야 함
- **배치 192**: Memory-Usage가 약 28-32GB 정도여야 함

## ⚠️ 메모리 부족 시 대응

### CUDA Out of Memory 오류 발생 시

```bash
# 배치 크기 감소
python train_server.py --batch-size 64

# 또는 더 작게
python train_server.py --batch-size 32
```

### 메모리 사용량이 너무 높을 때

```bash
# 보수적인 설정으로 변경
python train_server.py --batch-size 64 --epochs 10
```

## 💡 최적화 팁

### 1. 처음 실행 시 테스트

```bash
# 작은 배치로 먼저 테스트
python train_server.py --batch-size 64 --epochs 1

# GPU 메모리 확인 후 배치 크기 조정
nvidia-smi
```

### 2. Mixed Precision Training (향후 추가 가능)

FP16을 사용하면 메모리 사용량을 절반으로 줄이고 배치 크기를 2배로 늘릴 수 있습니다:
- 배치 128 → 배치 256 가능
- 학습 속도 1.5-2배 향상

### 3. Gradient Accumulation

메모리 부족 시 여러 작은 배치를 누적하여 큰 배치 효과를 낼 수 있습니다.

## 📝 체크리스트

V100에서 학습 시작 전:

- [x] GPU 확인: `nvidia-smi` ✅ (Tesla V100-DGXS-32GB × 4개 확인됨)
- [ ] GPU 메모리 확인: 각 GPU 32GB
- [ ] `training_config_dgx.yaml` 사용 또는 `--batch-size 128` 지정
- [ ] 데이터 파일 확인: `ls ../preprocessing/output/`
- [ ] 의존성 설치: `pip install -r requirements.txt`

## 🎯 결론

**현재 설정 (`batch_size: 128`)은 V100 32GB에 적합합니다!**

- ✅ GPU 메모리: ~20-25GB 사용 (안전한 범위)
- ✅ GPU 활용률: 80-90% (효율적)
- ✅ 학습 속도: 기본 설정 대비 약 3.5배 빠름
- ✅ 예상 학습 시간: 10 에폭 기준 약 12시간

바로 실행 가능합니다:

```bash
cd /home/zzangdol/RADAR-1/logbert_training
python train_server.py --config training_config_dgx.yaml
```


