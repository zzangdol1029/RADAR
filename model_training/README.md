# RADAR Model Training

딥러닝 모델 학습을 위한 워크스페이스입니다. 로그 기반 anomaly detection 모델(LogBERT)을 학습합니다.

## 📊 Current Status

### ✅ LogBERT
- **환경 구축**: Intel XPU, NVIDIA CUDA 지원 ✅
- **전체 학습 완료**: 324개 파일, NVIDIA GPU, `full_gpu.yaml` 사용
- **평가 완료**: `evaluate.py`로 epoch별 성능 평가

### 🔄 LogRobust
- **Status**: 계획 단계

---

## 🗂️ Project Structure

```
model_training/
├── logbert/                    # LogBERT 모델
│   ├── model.py                # 모델 정의
│   ├── dataset.py              # 데이터셋 클래스
│   ├── train.py                # 통합 학습 스크립트 (XPU/CUDA/CPU)
│   ├── evaluate.py             # 모델 평가
│   ├── configs/                # 학습 설정
│   │   ├── full_gpu.yaml       # ⭐ 전체 학습 (NVIDIA GPU)
│   │   ├── test_quick.yaml     # 테스트용 (Intel XPU)
│   │   ├── test_quick_xpu_small.yaml
│   │   └── test_xpu.yaml
│   ├── docs/                   # 문서
│   ├── checkpoints_full/       # 전체 학습 체크포인트
│   ├── checkpoints_test/       # 테스트 학습 체크포인트
│   ├── evaluation_results/     # 평가 결과 (epoch별)
│   ├── logs/                   # 학습/평가 로그
│   └── README.md               # LogBERT 상세 문서
│
├── logrobust/                   # LogRobust 모델 (계획 중)
│   └── README.md
│
├── plans/                       # 프로젝트 계획 문서
│   ├── PROJECT_PLAN.md
│   ├── README.md
│   └── data_validator.py
│
├── README.md                    # 📍 This file
└── FOLDER_STRUCTURE.md          # 상세 폴더 구조
```

---

## 🚀 Quick Start

### LogBERT Training

#### Prerequisites
- Python 3.10
- Conda
- Intel Arc Graphics (로컬) or NVIDIA GPU (서버)

#### 1. Setup Environment

**For NVIDIA GPU (전체 학습)**:
```bash
conda create -n logbert_cuda python=3.10 -y
conda activate logbert_cuda
cd logbert
pip install -r requirements_cuda.txt
```

**For Intel XPU (로컬 테스트)**:
```bash
conda create -n logbert_ipex python=3.10 -y
conda activate logbert_ipex
cd logbert
pip install -r requirements_intel_xpu.txt
```

#### 2. Run Training

**전체 학습 (324 files, NVIDIA GPU)**:
```bash
cd logbert
python train.py --config configs/full_gpu.yaml
```

**로컬 테스트 (Intel XPU)**:
```bash
cd logbert
python train.py --config configs/test_quick.yaml
```

#### 3. Evaluate Model

```bash
cd logbert
python evaluate.py \
    --checkpoint checkpoints_full/checkpoints/best_model.pt \
    --config configs/full_gpu.yaml \
    --validation-data /path/to/validation/data \
    --generate-fake-anomaly
```

---

## 📚 Documentation

### LogBERT
- **[LogBERT README](logbert/README.md)**: 전체 프로젝트 개요 및 사용법
- **[Setup Guide](logbert/docs/setup_guide.md)**: 환경 설정 상세 가이드
- **[Quick Start](logbert/docs/quick_start.md)**: 빠른 시작
- **[Evaluation Guide](logbert/docs/evaluation_guide.md)**: 모델 평가

### Project Planning
- **[Project Plan](plans/PROJECT_PLAN.md)**: 전체 프로젝트 계획
- **[Folder Structure](FOLDER_STRUCTURE.md)**: 상세 디렉토리 구조

---

## 🔧 Environment Support

| Environment | Hardware | Config File | Purpose |
|------------|----------|-------------|---------|
| **NVIDIA GPU** | RTX 3090+ | `full_gpu.yaml` | ⭐ 전체 학습 (Production) |
| **Intel XPU** | Intel Arc Graphics | `test_*.yaml` | 로컬 테스트 |

---

**Last Updated**: 2026-02-12
**Maintained by**: RADAR Team
