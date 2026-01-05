# SCP를 사용한 서버 업로드 가이드

SCP(Secure Copy)를 사용하여 서버에 파일을 업로드하는 방법을 안내합니다.

## 🚀 빠른 시작

### 방법 1: 자동 업로드 스크립트 사용 (가장 쉬움)

프로젝트 루트에 있는 `upload_to_server.sh` 스크립트를 사용하세요:

```bash
# 기본 사용 (서버 정보를 스크립트 내에서 수정 필요)
./upload_to_server.sh

# 또는 명령줄 인자로 지정
./upload_to_server.sh 사용자명 서버주소 /서버/경로

# 예시
./upload_to_server.sh user 192.168.1.100 /home/user/RADAR
```

## 📦 업로드할 파일 목록

### 필수 파일들

```bash
# Python 스크립트
logbert_training/train_server.py          # 서버 학습 스크립트 (새로 추가됨)
logbert_training/run_training_server.sh   # 실행 스크립트 (새로 추가됨)
logbert_training/train.py                 # 학습 모듈
logbert_training/dataset.py               # 데이터셋 클래스
logbert_training/model.py                 # 모델 정의
logbert_training/__init__.py              # Python 패키지 초기화

# 의존성
logbert_training/requirements.txt
```

### 선택 파일
```bash
logbert_training/training_config.yaml     # 학습 설정 파일
```

### 데이터 파일 (별도 업로드 필요)
```bash
preprocessing/output/preprocessed_logs_*.json
```

## 🚀 SCP 업로드 명령어

### 방법 1: 개별 파일 업로드

```bash
# 로컬에서 RADAR 디렉토리로 이동
cd /Users/zzangdol/RADAR

# 필수 Python 파일들 업로드
scp logbert_training/train_transfer.py 사용자명@서버주소:/서버/경로/logbert_training/
scp logbert_training/dataset.py 사용자명@서버주소:/서버/경로/logbert_training/
scp logbert_training/__init__.py 사용자명@서버주소:/서버/경로/logbert_training/

# 실행 스크립트 업로드
scp logbert_training/run_progressive_training.sh 사용자명@서버주소:/서버/경로/logbert_training/

# 의존성 파일 업로드
scp logbert_training/requirements.txt 사용자명@서버주소:/서버/경로/logbert_training/

# 설정 파일 업로드 (선택)
scp logbert_training/training_config.yaml 사용자명@서버주소:/서버/경로/logbert_training/
```

### 방법 2: 디렉토리 전체 업로드 (권장)

```bash
# 로컬에서 RADAR 디렉토리로 이동
cd /Users/zzangdol/RADAR

# logbert_training 디렉토리 전체 업로드
scp -r logbert_training 사용자명@서버주소:/서버/경로/

# 데이터 파일 업로드 (별도)
scp -r preprocessing/output 사용자명@서버주소:/서버/경로/preprocessing/
```

### 방법 3: 필수 파일만 선택하여 업로드

```bash
# 로컬에서 RADAR 디렉토리로 이동
cd /Users/zzangdol/RADAR

# 서버에 디렉토리 생성 (SSH로 먼저 실행)
ssh 사용자명@서버주소 "mkdir -p /서버/경로/logbert_training"

# 필수 파일들만 업로드
scp logbert_training/train_transfer.py \
    logbert_training/dataset.py \
    logbert_training/__init__.py \
    logbert_training/run_progressive_training.sh \
    logbert_training/requirements.txt \
    사용자명@서버주소:/서버/경로/logbert_training/
```

## 📝 실제 사용 예시

### 예시 1: 기본 업로드
```bash
# 서버 정보
# - 사용자명: user
# - 서버 주소: 192.168.1.100 또는 example.com
# - 서버 경로: /home/user/RADAR

cd /Users/zzangdol/RADAR

# logbert_training 디렉토리 업로드
scp -r logbert_training user@192.168.1.100:/home/user/RADAR/

# 데이터 파일 업로드
scp -r preprocessing/output user@192.168.1.100:/home/user/RADAR/preprocessing/
```

### 예시 2: 포트 지정 (기본 포트가 아닌 경우)
```bash
scp -P 2222 -r logbert_training user@example.com:/home/user/RADAR/
```

### 예시 3: SSH 키 사용
```bash
# SSH 키가 있는 경우 자동으로 사용됩니다
scp -i ~/.ssh/id_rsa -r logbert_training user@example.com:/home/user/RADAR/
```

## 🔧 업로드 후 서버에서 실행할 명령어

서버에 SSH로 접속한 후:

```bash
# 1. 업로드된 디렉토리로 이동
cd /서버/경로/logbert_training

# 2. 실행 권한 부여
chmod +x run_training_server.sh

# 3. 의존성 설치
pip install -r requirements.txt
# 또는 conda 환경 사용 시
conda activate radar
pip install -r requirements.txt

# 4. 데이터 경로 확인
ls -la ../preprocessing/output/preprocessed_logs_*.json

# 5. 학습 실행
./run_training_server.sh
# 또는
python train_server.py
```

### 새로운 서버 학습 스크립트 사용

새로 추가된 `train_server.py`와 `run_training_server.sh`를 사용하면 더 쉽게 학습할 수 있습니다:

```bash
# 기본 실행
./run_training_server.sh

# 커스텀 옵션
python train_server.py --batch-size 64 --epochs 20
```

자세한 내용은 `SERVER_TRAINING_GUIDE.md`를 참고하세요.

## 📋 한 번에 실행하는 스크립트

프로젝트 루트에 `upload_to_server.sh` 스크립트가 있습니다. 이 스크립트는 다음을 수행합니다:

1. 서버 연결 테스트
2. 서버에 필요한 디렉토리 생성
3. `logbert_training` 디렉토리 전체 업로드
4. `preprocessing/output` 디렉토리 업로드

### 스크립트 사용법

```bash
# 실행 권한 부여 (처음 한 번만)
chmod +x upload_to_server.sh

# 기본 사용 (스크립트 내 서버 정보 수정 필요)
./upload_to_server.sh

# 명령줄 인자로 서버 정보 지정
./upload_to_server.sh 사용자명 서버주소 /서버/경로

# 예시
./upload_to_server.sh user 192.168.1.100 /home/user/RADAR
```

### 스크립트 수정하기

스크립트를 열어서 기본 서버 정보를 수정할 수 있습니다:

```bash
# upload_to_server.sh 파일 편집
nano upload_to_server.sh

# 또는
vim upload_to_server.sh
```

다음 부분을 수정하세요:
```bash
SERVER_USER="${1:-user}"           # 기본 사용자명
SERVER_HOST="${2:-192.168.1.100}"  # 기본 서버 주소
SERVER_PATH="${3:-/home/user/RADAR}"  # 기본 서버 경로
```

## ⚠️ 주의사항

1. **대용량 파일**: 데이터 파일이 크면 시간이 오래 걸릴 수 있습니다.
   ```bash
   # 진행률 표시
   scp -v -r logbert_training user@server:/path/
   ```

2. **권한 확인**: 서버의 업로드 경로에 쓰기 권한이 있는지 확인하세요.

3. **디렉토리 구조**: 서버에서도 동일한 디렉토리 구조를 유지하는 것이 좋습니다.

4. **네트워크 안정성**: 대용량 파일 업로드 시 네트워크가 끊기지 않도록 주의하세요.

## 🔍 업로드 확인

서버에 SSH로 접속하여 확인:

```bash
# 파일 확인
ls -la /서버/경로/logbert_training/

# 파일 크기 확인
du -sh /서버/경로/logbert_training/

# 데이터 파일 확인
ls -lh /서버/경로/preprocessing/output/preprocessed_logs_*.json
```

## 💡 팁

1. **압축 후 업로드** (대용량 파일의 경우):
   ```bash
   # 로컬에서 압축
   tar -czf logbert_training.tar.gz logbert_training/
   
   # 업로드
   scp logbert_training.tar.gz user@server:/path/
   
   # 서버에서 압축 해제
   ssh user@server "cd /path && tar -xzf logbert_training.tar.gz"
   ```

2. **rsync 사용** (더 효율적, 변경된 파일만 업로드):
   ```bash
   rsync -avz logbert_training/ user@server:/path/logbert_training/
   ```

3. **백그라운드 업로드** (대용량 파일):
   ```bash
   nohup scp -r logbert_training user@server:/path/ > upload.log 2>&1 &
   ```






