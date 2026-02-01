# RADAR Model Training

딥러닝 모델 학습을 위한 워크스페이스입니다. 로그 기반 anomaly detection 모델(LogBERT, LogRobust)을 학습합니다.

## 📊 Current Status

### ✅ LogBERT
- **환경 구축**: Intel XPU, NVIDIA CUDA 지원 ✅
- **Quick Test**: 1개 파일 학습 완료 (3h 5m, Loss 4.71→0.91) ✅
- **Next**: 10개 파일 테스트 or 전체 학습 (GPU 서버)

### 🔄 LogRobust
- **Status**: 계획 단계
- **Purpose**: LogBERT 대비 robustness 개선

---

## 🗂️ Project Structure

```
model_training/
├── logbert/                    # LogBERT 모델 (진행 중)
│   ├── configs/                 # 학습 설정
│   │   ├── test_quick_xpu_small.yaml  # 1개 파일 (Intel XPU, ~3h)
│   │   ├── test_xpu.yaml              # 10개 파일 (Intel XPU, 3-5h)
│   │   └── full_gpu.yaml              # 324개 파일 (NVIDIA GPU, 3-5d)
│   ├── scripts/                 # 학습 스크립트
│   │   ├── train_intel.py       # Intel XPU 학습
│   │   ├── train_cuda.py        # NVIDIA GPU 학습
│   │   └── evaluate.py          # 모델 평가
│   ├── docs/                    # 문서
│   │   ├── setup_guide.md       # 환경 설정 가이드
│   │   ├── quick_start.md       # 빠른 시작
│   │   └── evaluation_guide.md  # 평가 가이드
│   ├── checkpoints/             # 전체 학습 체크포인트
│   ├── checkpoints_quick/       # Quick test 체크포인트 (best_model.pt 등)
│   ├── logs/                    # 학습 로그
│   ├── requirements_intel_xpu.txt
│   ├── requirements_cuda.txt
│   ├── run_quick_test.ps1       # Quick test 실행 스크립트
│   └── README.md                # LogBERT 상세 문서
│
├── logrobust/                   # LogRobust 모델 (계획 중)
│   ├── configs/
│   ├── scripts/
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

**For Intel XPU (로컬 PC)**:
```bash
conda create -n logbert_ipex python=3.10 -y
conda activate logbert_ipex
cd logbert
pip install -r requirements_intel_xpu.txt
```

**For NVIDIA GPU (서버)**:
```bash
conda create -n logbert_cuda python=3.10 -y
conda activate logbert_cuda
cd logbert
pip install -r requirements_cuda.txt
```

#### 2. Run Training

**Quick Test (1 file, ~3h)**:
```powershell
cd logbert
.\run_quick_test.ps1
```

**Standard Test (10 files, 3-5h)**:
```bash
cd logbert
python scripts/train_intel.py --config configs/test_xpu.yaml
```

**Full Training (324 files, 3-5 days)**:
```bash
cd logbert
python scripts/train_cuda.py --config configs/full_gpu.yaml
```

---

## 📊 Training Results

### LogBERT Quick Test (2026-02-01)
- **Environment**: Intel Arc Graphics (XPU)
- **Data**: 1 file
- **Duration**: 3h 5m
- **Result**: Loss 4.71 → 0.91 (80.7% reduction) ✅
- **Checkpoint**: `logbert/checkpoints_quick/best_model.pt`

**Detailed logs**: `logbert/logs/train_quick_20260201_v2.log`

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

## 🎯 Training Pipeline

```
Data (324 files)
    │
    ├─→ [Quick Test] 1 file → Intel XPU (3h) ✅
    │
    ├─→ [Standard Test] 10 files → Intel XPU (3-5h)
    │
    └─→ [Full Training] 324 files → NVIDIA GPU (3-5 days)
                        │
                        └─→ Best Model → Production
```

---

## 🔧 Environment Support

| Environment | Hardware | Status | Config Files |
|------------|----------|--------|--------------|
| **Intel XPU** | Intel Arc Graphics | ✅ Ready | `test_quick_xpu_small.yaml`, `test_xpu.yaml` |
| **NVIDIA GPU** | RTX 3090/4090+ | ✅ Ready | `full_gpu.yaml` |
| **CPU** | Any | ⚠️ Not recommended | `full.yaml` |

---

## 📝 Next Steps

1. ✅ **Quick Test 완료** (1 file, Intel XPU)
2. **Option A**: 10개 파일 테스트 (로컬 Intel XPU, 3-5시간)
   - 중간 규모 검증
   - 리소스: 로컬 PC
3. **Option B**: 전체 학습 (GPU 서버 대여, 3-5일)
   - 전체 324개 파일
   - 최종 production 모델
   - 리소스: NVIDIA GPU 서버

---

## 🤝 Contributing

프로젝트 구조 및 코드 스타일은 각 하위 프로젝트의 README를 참조하세요.

---

## 📄 License

RADAR Project
