# 서버에서 빠른 시작 가이드

DGX 서버에서 LogBERT 학습을 빠르게 시작하는 가이드입니다.

## 🚀 빠른 실행

### 방법 1: 실행 스크립트 사용 (권장)

```bash
cd /home/zzangdol/RADAR-1/logbert_training

# 실행 권한 부여 (처음 한 번만)
chmod +x run_training_server.sh

# 학습 실행
./run_training_server.sh --config training_config_dgx.yaml
```

### 방법 2: Python3 직접 실행

```bash
cd /home/zzangdol/RADAR-1/logbert_training

# Python3로 실행
python3 train_server.py --config training_config_dgx.yaml
```

## ⚠️ Python 명령어 문제 해결

### 문제: `python` 명령어를 찾을 수 없음

**오류 메시지:**
```
Command 'python' not found, did you mean:
  command 'python3' from deb python3
```

**해결 방법:**

#### 방법 1: `python3` 사용 (권장)

```bash
# python3로 실행
python3 train_server.py --config training_config_dgx.yaml
```

#### 방법 2: 심볼릭 링크 생성 (선택사항)

```bash
# python 심볼릭 링크 생성 (관리자 권한 필요)
sudo apt-get install python-is-python3

# 또는 수동으로
sudo ln -s /usr/bin/python3 /usr/bin/python
```

#### 방법 3: 실행 스크립트 사용

`run_training_server.sh`는 자동으로 `python3`를 찾아서 사용합니다:

```bash
./run_training_server.sh --config training_config_dgx.yaml
```

## 📋 실행 전 체크리스트

```bash
# 1. 현재 위치 확인
pwd
# 출력: /home/zzangdol/RADAR-1/logbert_training

# 2. Python3 확인
python3 --version
# 출력: Python 3.x.x

# 3. GPU 확인
nvidia-smi
# 출력: Tesla V100-DGXS-32GB × 4개 확인

# 4. 데이터 파일 확인
ls ../preprocessing/output/preprocessed_logs_*.json | wc -l

# 5. 의존성 확인
pip3 list | grep torch
```

## 🎯 실행 명령어 요약

### 기본 실행

```bash
cd /home/zzangdol/RADAR-1/logbert_training
python3 train_server.py --config training_config_dgx.yaml
```

### 커스텀 옵션

```bash
python3 train_server.py \
    --config training_config_dgx.yaml \
    --batch-size 128 \
    --epochs 10
```

### 백그라운드 실행

```bash
nohup python3 train_server.py --config training_config_dgx.yaml > training.log 2>&1 &

# 프로세스 ID 확인
echo $!

# 로그 확인
tail -f training.log
```

## 🔍 문제 해결

### Python3를 찾을 수 없음

```bash
# Python3 설치 확인
which python3

# 설치되어 있지 않다면
sudo apt-get update
sudo apt-get install python3 python3-pip
```

### Conda 환경 사용 시

```bash
# Conda 환경 활성화
conda activate radar

# Conda 환경에서는 python 명령어 사용 가능
python train_server.py --config training_config_dgx.yaml
```

### 의존성 설치

```bash
# pip3로 설치
pip3 install -r requirements.txt

# 또는 conda 환경에서
conda activate radar
pip install -r requirements.txt
```

## 💡 팁

1. **항상 `python3` 사용**: Linux 서버에서는 `python3`가 표준입니다.
2. **실행 스크립트 사용**: `run_training_server.sh`가 자동으로 `python3`를 찾습니다.
3. **Conda 환경**: Conda 환경을 사용하면 `python` 명령어도 사용 가능합니다.


