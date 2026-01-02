# 서버 배포 가이드

서버에서 점진적 학습을 실행하기 위해 필요한 파일과 설정 방법을 안내합니다.

## 📦 필수 파일 목록

### 1. Python 스크립트 파일
```
logbert_training/
├── train_transfer.py          # 메인 학습 스크립트 (필수)
├── dataset.py                 # 데이터셋 클래스 (필수)
├── __init__.py                # Python 패키지 초기화 (필수)
└── requirements.txt           # 의존성 패키지 목록 (필수)
```

### 2. 실행 스크립트
```
logbert_training/
└── run_progressive_training.sh  # 백그라운드 실행 스크립트 (필수)
```

### 3. 설정 파일 (선택사항)
```
logbert_training/
└── training_config.yaml       # 학습 설정 파일 (선택, 기본값 사용 가능)
```

### 4. 데이터 파일
```
preprocessing/
└── output/
    ├── preprocessed_logs_*.json  # 전처리된 로그 데이터 파일들 (필수)
    └── ...
```

## 🚀 서버 배포 절차

### 1단계: 파일 업로드

서버에 다음 디렉토리 구조로 파일을 업로드합니다:

```
/your/server/path/
├── logbert_training/
│   ├── train_transfer.py
│   ├── dataset.py
│   ├── __init__.py
│   ├── requirements.txt
│   ├── run_progressive_training.sh
│   └── training_config.yaml (선택)
└── preprocessing/
    └── output/
        ├── preprocessed_logs_*.json
        └── ...
```

### 2단계: Python 환경 설정

#### Conda 환경 사용 (권장)
```bash
# Conda 환경 생성
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
# 또는
venv\Scripts\activate  # Windows

# 의존성 설치
cd logbert_training
pip install -r requirements.txt
```

### 3단계: 실행 권한 부여

```bash
chmod +x logbert_training/run_progressive_training.sh
```

### 4단계: 데이터 경로 확인

`train_transfer.py`는 기본적으로 다음 경로에서 데이터를 찾습니다:
- `../preprocessing/output/preprocessed_logs_*.json`

데이터가 다른 위치에 있다면 `--preprocessed-dir` 옵션을 사용하세요:
```bash
python train_transfer.py --progressive --preprocessed-dir /path/to/your/data
```

### 5단계: 학습 실행

#### 방법 1: 스크립트 사용 (권장)
```bash
cd logbert_training
./run_progressive_training.sh
```

#### 방법 2: 직접 실행
```bash
cd logbert_training
python train_transfer.py --progressive
```

#### 방법 3: 커스텀 파라미터로 실행
```bash
cd logbert_training
./run_progressive_training.sh \
    bert-base-uncased \    # Pre-trained 모델
    0.05 \                 # 시작 비율 (5%)
    0.05 \                 # 단계 크기 (5%)
    0.5 \                  # 최대 비율 (50%)
    5 \                    # 단계당 에폭
    45000 \                # 최대 메모리 (MB)
    8 \                    # 최소 배치 크기
    8                      # 고정 배치 크기
```

## 📋 필수 의존성 패키지

`requirements.txt`에 포함된 패키지:
- `torch>=2.0.0` - PyTorch
- `transformers>=4.30.0` - Hugging Face Transformers (BERT)
- `numpy>=1.24.0` - 수치 연산
- `tqdm>=4.65.0` - 진행률 표시
- `PyYAML>=6.0` - YAML 설정 파일 파싱
- `psutil` - 시스템 리소스 모니터링

## 🔍 모니터링

### 프로세스 확인
```bash
# PID 파일에서 프로세스 ID 확인
cat logbert_training/checkpoints_transfer/progressive_training.pid

# 프로세스 상태 확인
ps -p $(cat logbert_training/checkpoints_transfer/progressive_training.pid)
```

### 로그 확인
```bash
# 실시간 로그 확인
tail -f logbert_training/checkpoints_transfer/logs/progressive_training_*.log

# 단계별 로그 확인
tail -f logbert_training/checkpoints_transfer/stage_*_*pct/logs/*.log
```

### 학습 진행 상황 확인
```bash
# 체크포인트 파일 확인
ls -lh logbert_training/checkpoints_transfer/stage_*_*pct/checkpoints/

# 최종 결과 확인
cat logbert_training/checkpoints_transfer/progressive_training_results.json
```

## ⚙️ 서버별 최적화 설정

### 메모리가 충분한 서버 (예: 64GB+)
```bash
./run_progressive_training.sh \
    bert-base-uncased \
    0.05 \
    0.05 \
    1.0 \      # 100% 데이터까지 학습
    5 \
    60000 \    # 더 높은 메모리 제한
    16 \       # 더 큰 배치 크기
    16
```

### 메모리가 제한적인 서버 (예: 16GB)
```bash
./run_progressive_training.sh \
    bert-base-uncased \
    0.05 \
    0.05 \
    0.3 \      # 30% 데이터만 학습
    3 \        # 더 적은 에폭
    12000 \    # 낮은 메모리 제한
    4 \        # 작은 배치 크기
    4
```

## 🛠️ 문제 해결

### 1. 메모리 부족 오류
- `--max-memory-mb` 값을 줄이세요
- `--fixed-batch-size` 값을 줄이세요
- `--max-ratio` 값을 줄여서 더 적은 데이터만 사용하세요

### 2. 데이터 파일을 찾을 수 없음
- `--preprocessed-dir` 옵션으로 올바른 경로를 지정하세요
- 데이터 파일이 `preprocessed_logs_*.json` 형식인지 확인하세요

### 3. 권한 오류
```bash
chmod +x run_progressive_training.sh
```

### 4. Python 패키지 설치 오류
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📝 체크리스트

배포 전 확인사항:
- [ ] 모든 필수 Python 파일 업로드 완료
- [ ] `requirements.txt`의 모든 패키지 설치 완료
- [ ] 데이터 파일 경로 확인 및 접근 가능
- [ ] 실행 스크립트에 실행 권한 부여
- [ ] 서버 메모리 용량 확인 및 적절한 파라미터 설정
- [ ] 로그 디렉토리 쓰기 권한 확인

## 📞 추가 도움말

- 상세한 학습 옵션: `python train_transfer.py --help`
- 점진적 학습 가이드: `PROGRESSIVE_TRAINING_GUIDE.md`
- 전이 학습 가이드: `TRANSFER_LEARNING_GUIDE.md`






