# 서버 의존성 설치 가이드

DGX 서버에서 필요한 Python 패키지를 설치하는 방법입니다.

## 🚀 빠른 설치

```bash
cd /home/zzangdol/RADAR-1/logbert_training

# pip3로 의존성 설치
pip3 install -r requirements.txt
```

## 📋 필수 패키지

### 기본 설치

```bash
# pip3 업그레이드
pip3 install --upgrade pip

# 의존성 설치
pip3 install -r requirements.txt
```

### 개별 설치 (문제 해결 시)

```bash
# PyTorch (CUDA 12.2 지원)
pip3 install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu121

# Transformers
pip3 install transformers>=4.30.0

# 기타 패키지
pip3 install numpy>=1.24.0 tqdm>=4.65.0 PyYAML>=6.0 psutil>=5.9.0
```

## 🔧 문제 해결

### 문제 1: transformers 버전 문제

**오류:**
```
ImportError: cannot import name 'BertModel' from 'transformers'
```

**해결 방법:**

```bash
# transformers 업그레이드
pip3 install --upgrade transformers

# 특정 버전 설치
pip3 install transformers==4.30.0
```

### 문제 2: PyTorch CUDA 버전 불일치

**확인:**
```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

**CUDA 12.2용 PyTorch 설치:**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 문제 3: 권한 오류

**사용자 디렉토리에 설치:**
```bash
pip3 install --user -r requirements.txt
```

**또는 가상환경 사용:**
```bash
# 가상환경 생성
python3 -m venv venv

# 활성화
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 문제 4: Conda 환경 사용

```bash
# Conda 환경 생성
conda create -n radar python=3.10
conda activate radar

# PyTorch 설치 (Conda)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 기타 패키지
pip install transformers>=4.30.0 numpy tqdm PyYAML psutil
```

## ✅ 설치 확인

```bash
# Python 버전 확인
python3 --version

# 패키지 확인
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import transformers; print('Transformers:', transformers.__version__)"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python3 -c "import torch; print('CUDA version:', torch.version.cuda)"

# GPU 확인
nvidia-smi
```

## 📝 전체 설치 스크립트

```bash
#!/bin/bash
# install_dependencies.sh

cd /home/zzangdol/RADAR-1/logbert_training

echo "의존성 설치 시작..."

# pip 업그레이드
pip3 install --upgrade pip

# PyTorch 설치 (CUDA 12.2)
pip3 install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu121

# Transformers 및 기타 패키지
pip3 install transformers>=4.30.0 numpy>=1.24.0 tqdm>=4.65.0 PyYAML>=6.0 psutil>=5.9.0

echo "설치 완료!"

# 확인
python3 -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python3 -c "import transformers; print('Transformers:', transformers.__version__)"
```

## 💡 팁

1. **가상환경 사용 권장**: 시스템 Python과 분리하여 사용
2. **Conda 사용**: DGX 서버에서는 Conda 환경이 더 안정적일 수 있음
3. **CUDA 버전 확인**: `nvidia-smi`로 CUDA 버전 확인 후 맞는 PyTorch 설치




