# Model Training Folder Structure

RADAR 프로젝트의 모델 학습 워크스페이스 구조입니다.

Last Updated: 2026-02-12

---

## 📁 Directory Tree

```
model_training/
├── README.md                       # 프로젝트 개요 및 Quick Start
├── FOLDER_STRUCTURE.md             # 📍 This file - 상세 디렉토리 구조
│
├── logbert/                        # LogBERT 모델 학습
│   ├── README.md                   # LogBERT 상세 문서
│   ├── EVALUATION_RESULTS.md       # 평가 결과 정리
│   ├── PROJECT_STATUS.md           # 프로젝트 진행 상황
│   │
│   ├── model.py                    # LogBERT 모델 정의
│   ├── dataset.py                  # 데이터셋 클래스
│   ├── train.py                    # 통합 학습 스크립트 (XPU/CUDA/CPU 자동 감지)
│   ├── evaluate.py                 # 모델 평가 스크립트
│   ├── __init__.py                 # 패키지 초기화
│   │
│   ├── configs/                    # 학습 설정 파일
│   │   ├── full_gpu.yaml           # ⭐ 전체 학습 (324 files, NVIDIA GPU)
│   │   ├── test_quick.yaml         # 테스트용 (Intel XPU)
│   │   ├── test_quick_xpu_small.yaml  # 테스트용 (Intel XPU, small)
│   │   └── test_xpu.yaml          # 테스트용 (10 files, Intel XPU)
│   │
│   ├── docs/                       # 문서
│   │   ├── setup_guide.md          # 환경 설정 가이드
│   │   ├── quick_start.md          # 빠른 시작 가이드
│   │   ├── evaluation_guide.md     # 모델 평가 가이드
│   │   └── 10_file_training_guide.md  # 10개 파일 학습 가이드
│   │
│   ├── checkpoints_full/           # 전체 학습 (GPU) 체크포인트
│   │   └── checkpoints/
│   │
│   ├── checkpoints_test/           # 테스트 학습 (XPU) 체크포인트
│   │   └── checkpoints/
│   │
│   ├── evaluation_results/         # 평가 결과
│   │
│   ├── logs/                       # 학습/평가 로그
│   ├── train.log                   # 전체 학습 로그
│   ├── requirements.txt            # 기본 의존성
│   ├── requirements_cuda.txt       # NVIDIA CUDA 환경 의존성
│   └── requirements_intel_xpu.txt  # Intel XPU 환경 의존성
│
├── logrobust/                      # LogRobust 모델 (계획 단계)
│   └── README.md
│
└── plans/                          # 프로젝트 계획 문서
    ├── PROJECT_PLAN.md             # 전체 프로젝트 계획
    ├── README.md
    └── data_validator.py           # 데이터 검증 스크립트
```

---

## 📝 File Descriptions

### Root Level

| File/Directory | Description | Status |
|---------------|-------------|---------|
| `README.md` | 프로젝트 개요, Quick Start, 결과 요약 | ✅ Updated |
| `FOLDER_STRUCTURE.md` | 상세 디렉토리 구조 문서 | ✅ Updated |
| `logbert/` | LogBERT 모델 학습 디렉토리 | ✅ Active |
| `logrobust/` | LogRobust 모델 디렉토리 | 🔄 Planned |
| `plans/` | 프로젝트 계획 및 검증 스크립트 | ✅ Active |

---

## 🔧 LogBERT Directory Details

### Core Files (학습/평가 코드)

| File | Purpose | Status |
|------|---------|--------|
| `model.py` | LogBERT 모델 아키텍처 정의 | ✅ Production |
| `dataset.py` | 데이터셋 로딩/전처리 클래스 | ✅ Production |
| `train.py` | 통합 학습 스크립트 (XPU/CUDA/CPU 자동 감지) | ✅ Production |
| `evaluate.py` | 모델 평가 (메트릭, 시각화) | ✅ Production |

### Configuration Files (`configs/`)

| File | Purpose | Environment | Status |
|------|---------|-------------|--------|
| `full_gpu.yaml` | ⭐ 전체 학습 (324 files) | NVIDIA GPU | ✅ Production |
| `test_quick.yaml` | 테스트 학습 | Intel XPU | 📌 Test |
| `test_quick_xpu_small.yaml` | 소규모 테스트 | Intel XPU | 📌 Test |
| `test_xpu.yaml` | 중간 테스트 (10 files) | Intel XPU | 📌 Test |

### Documentation (`docs/`)

| Document | Description |
|----------|-------------|
| `setup_guide.md` | 환경 설정 가이드 (Intel XPU + CUDA) |
| `quick_start.md` | 빠른 시작 가이드 |
| `evaluation_guide.md` | 모델 평가 가이드 |
| `10_file_training_guide.md` | 10개 파일 학습 가이드 |

---

## 🎯 Training Workflow

```
1. Training
   ├─→ [Production] full_gpu.yaml + train.py → checkpoints_full/
   └─→ [Test]       test_*.yaml + train.py   → checkpoints_test/

2. Evaluation
   └─→ evaluate.py → evaluation_results/
```

---

**Last Updated**: 2026-02-12
**Maintained by**: RADAR Team
