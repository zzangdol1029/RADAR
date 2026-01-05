# 서버에서 Output 파일로 학습하기 가이드

이 가이드는 서버에 있는 `preprocessing/output` 디렉토리의 전처리된 파일을 사용하여 LogBERT 모델을 학습하는 방법을 설명합니다.

## 📋 목차

1. [필수 파일 확인](#필수-파일-확인)
2. [환경 설정](#환경-설정)
3. [학습 실행](#학습-실행)
4. [모니터링](#모니터링)
5. [문제 해결](#문제-해결)

## 📦 필수 파일 확인

서버에 다음 파일들이 있어야 합니다:

```
/서버/경로/
├── logbert_training/
│   ├── train_server.py          # 서버 학습 스크립트 (새로 생성됨)
│   ├── run_training_server.sh    # 실행 스크립트 (새로 생성됨)
│   ├── train.py                 # 학습 모듈 (필수)
│   ├── dataset.py               # 데이터셋 클래스 (필수)
│   ├── model.py                 # 모델 정의 (필수)
│   ├── __init__.py              # Python 패키지 초기화 (필수)
│   ├── training_config.yaml     # 학습 설정 파일 (선택)
│   └── requirements.txt         # 의존성 패키지 목록 (필수)
└── preprocessing/
    └── output/
        ├── preprocessed_logs_2025-02-24.json
        ├── preprocessed_logs_2025-02-25.json
        └── ... (전처리된 JSON 파일들)
```

## 🚀 환경 설정

### 1. Python 환경 설정

#### Conda 환경 사용 (권장)

```bash
# Conda 환경 생성 (아직 없는 경우)
conda create -n radar python=3.9
conda activate radar

# 의존성 설치
cd logbert_training
pip install -r requirements.txt
```

#### 가상환경 사용

```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

# 의존성 설치
cd logbert_training
pip install -r requirements.txt
```

### 2. 실행 권한 부여

```bash
chmod +x logbert_training/run_training_server.sh
```

### 3. 데이터 경로 확인

기본적으로 다음 경로에서 데이터를 찾습니다:
- `../preprocessing/output/preprocessed_logs_*.json`

데이터가 다른 위치에 있다면 `--data-dir` 옵션을 사용하세요.

## 🎯 학습 실행

### 방법 1: 실행 스크립트 사용 (권장)

```bash
cd logbert_training
./run_training_server.sh
```

### 방법 2: Python 스크립트 직접 실행

```bash
cd logbert_training
python train_server.py
```

### 방법 3: 커스텀 옵션으로 실행

#### 데이터 디렉토리 지정

```bash
python train_server.py --data-dir /path/to/preprocessing/output
```

#### 출력 디렉토리 지정

```bash
python train_server.py --output-dir /path/to/checkpoints
```

#### 배치 크기, 에폭 수, 학습률 지정

```bash
python train_server.py \
    --batch-size 64 \
    --epochs 20 \
    --learning-rate 1e-5
```

#### 모든 옵션 조합

```bash
python train_server.py \
    --data-dir /path/to/data \
    --output-dir /path/to/output \
    --config custom_config.yaml \
    --batch-size 64 \
    --epochs 20 \
    --learning-rate 1e-5
```

### 방법 4: 백그라운드 실행

```bash
# nohup으로 백그라운드 실행
nohup python train_server.py > training.log 2>&1 &

# 프로세스 ID 확인
echo $!

# 로그 확인
tail -f training.log
```

## 📊 모니터링

### 학습 진행 상황 확인

학습 중 다음 정보가 출력됩니다:
- 현재 Loss
- 평균 Loss
- 학습률
- 진행 상황 (에폭, 배치)

### 체크포인트 확인

```bash
# 체크포인트 파일 목록 확인
ls -lh checkpoints/checkpoints/

# 최고 성능 모델 확인
ls -lh checkpoints/checkpoints/best_model.pt

# 에폭별 체크포인트 확인
ls -lh checkpoints/checkpoints/epoch_*.pt
```

### GPU 사용량 확인 (CUDA 사용 시)

```bash
# 실시간 GPU 사용량 모니터링
watch -n 1 nvidia-smi

# 또는
nvidia-smi -l 1
```

### 프로세스 확인

```bash
# Python 학습 프로세스 확인
ps aux | grep train_server.py

# 메모리 사용량 확인
top -p $(pgrep -f train_server.py)
```

## ⚙️ 설정 파일 커스터마이징

`training_config.yaml` 파일을 수정하여 학습 설정을 변경할 수 있습니다:

```yaml
# 모델 설정
model:
  vocab_size: 10000
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12

# 학습 설정
training:
  batch_size: 32              # 배치 크기
  learning_rate: 0.00002      # 학습률
  num_epochs: 10              # 에폭 수
  weight_decay: 0.01          # 가중치 감쇠
  mask_prob: 0.15            # MLM 마스킹 확률

# 데이터 설정
data:
  preprocessed_dir: "../preprocessing/output"
  max_seq_length: 512

# 출력 설정
output_dir: "checkpoints"
```

## 🛠️ 문제 해결

### 1. 데이터 파일을 찾을 수 없음

**증상:**
```
❌ 데이터 디렉토리를 찾을 수 없습니다: /path/to/data
```

**해결 방법:**
```bash
# 데이터 디렉토리 경로 확인
ls -la ../preprocessing/output/

# 올바른 경로로 지정
python train_server.py --data-dir /올바른/경로/preprocessing/output
```

### 2. CUDA Out of Memory

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결 방법:**

`training_config.yaml`에서 배치 크기를 줄이세요:
```yaml
training:
  batch_size: 16  # 32에서 16으로 감소
```

또는 명령줄에서:
```bash
python train_server.py --batch-size 16
```

### 3. 메모리 부족 (CPU 환경)

**해결 방법:**
- 배치 크기 감소: `--batch-size 8`
- `num_workers`를 0으로 설정 (이미 자동으로 처리됨)
- 데이터 일부만 사용 (향후 기능 추가 예정)

### 4. 학습이 너무 느림

**해결 방법:**
- GPU 사용 확인: `nvidia-smi`
- 배치 크기 증가: `--batch-size 64`
- `num_workers` 증가 (GPU 환경에서만)

### 5. Loss가 수렴하지 않음

**해결 방법:**
- 학습률 낮추기: `--learning-rate 1e-5`
- 에폭 수 증가: `--epochs 20`
- 가중치 감쇠 조정: `training_config.yaml`에서 `weight_decay` 수정

### 6. Python 패키지 설치 오류

**해결 방법:**
```bash
# pip 업그레이드
pip install --upgrade pip

# 의존성 재설치
pip install -r requirements.txt --force-reinstall
```

### 7. 권한 오류

**해결 방법:**
```bash
# 실행 권한 부여
chmod +x run_training_server.sh

# 출력 디렉토리 쓰기 권한 확인
mkdir -p checkpoints
chmod 755 checkpoints
```

## 📝 체크리스트

학습 시작 전 확인사항:

- [ ] 모든 필수 Python 파일이 서버에 업로드됨
- [ ] `requirements.txt`의 모든 패키지 설치 완료
- [ ] `preprocessing/output` 디렉토리에 전처리된 JSON 파일 존재
- [ ] 실행 스크립트에 실행 권한 부여 (`chmod +x`)
- [ ] 출력 디렉토리 쓰기 권한 확인
- [ ] GPU 사용 시 CUDA 설치 및 확인 (선택사항)
- [ ] `training_config.yaml` 설정 확인 (선택사항)

## 📞 추가 도움말

- 기본 학습 가이드: `QUICK_START.md`
- 학습 옵션 상세: `TRAINING_OPTIONS.md`
- 모델 평가: `MODEL_EVALUATION.md`

## 💡 팁

1. **첫 실행 시 작은 데이터로 테스트**
   - 일부 파일만 사용하여 설정이 올바른지 확인

2. **체크포인트 주기적 확인**
   - 학습 중간에 체크포인트가 저장되는지 확인

3. **로그 파일 저장**
   - 백그라운드 실행 시 로그를 파일로 저장하여 나중에 확인

4. **리소스 모니터링**
   - 학습 시작 전 메모리와 디스크 공간 확인


