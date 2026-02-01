# Model Training Folder Structure

RADAR 프로젝트의 모델 학습 워크스페이스 구조입니다.

Last Updated: 2026-02-01

---

## 📁 Directory Tree

```
model_training/
├── README.md                       # 프로젝트 개요 및 Quick Start
├── FOLDER_STRUCTURE.md             # 📍 This file - 상세 디렉토리 구조
│
├── logbert/                        # LogBERT 모델 학습 (진행 중)
│   ├── README.md                   # LogBERT 상세 문서
│   │
│   ├── configs/                    # 학습 설정 파일
│   │   ├── test_quick_xpu_small.yaml  # 빠른 검증 (1 file, Intel XPU, ~3h)
│   │   ├── test_quick_xpu.yaml        # 빠른 검증 대안 설정
│   │   ├── test_xpu.yaml              # 표준 테스트 (10 files, Intel XPU, 3-5h)
│   │   ├── full.yaml                  # 전체 학습 (CPU/XPU)
│   │   └── full_gpu.yaml              # 전체 학습 (324 files, NVIDIA GPU, 3-5d)
│   │
│   ├── scripts/                    # 학습 및 평가 스크립트
│   │   ├── train_intel.py          # Intel XPU 학습 스크립트
│   │   ├── train_cuda.py           # NVIDIA GPU 학습 스크립트
│   │   └── evaluate.py             # 모델 평가 스크립트
│   │
│   ├── docs/                       # 문서
│   │   ├── setup_guide.md          # 환경 설정 가이드 (Intel XPU + CUDA)
│   │   ├── quick_start.md          # 빠른 시작 가이드
│   │   └── evaluation_guide.md     # 모델 평가 가이드
│   │
│   ├── checkpoints/                # 전체 학습 체크포인트 (empty)
│   │
│   ├── checkpoints_quick/          # Quick test 체크포인트
│   │   ├── best_model.pt           # 최고 성능 모델 (~1.1GB)
│   │   ├── epoch_1.pt              # Epoch 1 체크포인트
│   │   └── checkpoint_step_*.pt    # 중간 체크포인트 (500 step 간격)
│   │
│   ├── logs/                       # 학습 로그
│   │   ├── train_quick_20260201_v2.log    # Quick test 학습 로그 ✅
│   │   ├── evaluation_20260201_*.log      # 평가 로그
│   │   └── ...                     # 기타 로그 파일
│   │
│   ├── requirements_intel_xpu.txt  # Intel XPU 환경 의존성
│   ├── requirements_cuda.txt       # NVIDIA CUDA 환경 의존성
│   └── run_quick_test.ps1          # Quick test 실행 스크립트 (Windows)
│
├── logrobust/                      # LogRobust 모델 (계획 단계)
│   ├── README.md
│   ├── configs/
│   └── scripts/
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

### Configuration Files (`configs/`)

| File | Files | Environment | Batch | Time | Purpose | Status |
|------|-------|-------------|-------|------|---------|--------|
| `test_quick_xpu_small.yaml` | 1 | Intel XPU | 16 | ~3h | 코드 검증 | ✅ Tested |
| `test_quick_xpu.yaml` | 1 | Intel XPU | 32 | ~3h | 대안 설정 | ⚠️ Untested |
| `test_xpu.yaml` | 10 | Intel XPU | 32 | 3-5h | 중간 테스트 | 🔄 Ready |
| `full.yaml` | 324 | CPU/XPU | 32 | Days | Legacy | ⚠️ Not recommended |
| `full_gpu.yaml` | 324 | NVIDIA GPU | 64 | 3-5d | 전체 학습 | 🔄 Ready |

### Scripts (`scripts/`)

| Script | Purpose | Environment | Status |
|--------|---------|-------------|--------|
| `train_intel.py` | Intel XPU 학습 | Intel Arc Graphics | ✅ Working |
| `train_cuda.py` | NVIDIA GPU 학습 | CUDA 11.8+ | ✅ Ready |
| `evaluate.py` | 모델 평가 | Both | ✅ Working |

### Documentation (`docs/`)

| Document | Description | Status |
|----------|-------------|--------|
| `setup_guide.md` | 환경 설정 상세 가이드 (Intel XPU + CUDA) | ✅ Updated |
| `quick_start.md` | 빠른 시작 가이드 | ✅ Complete |
| `evaluation_guide.md` | 모델 평가 가이드 | ✅ Complete |

### Checkpoints (`checkpoints_quick/`)

**Quick Test Results (2026-02-01)**:

| Checkpoint | Size | Description | Loss |
|-----------|------|-------------|------|
| `best_model.pt` | ~1.1GB | 최고 성능 모델 | 0.91 |
| `epoch_1.pt` | ~1.1GB | Epoch 1 완료 | 0.91 |
| `checkpoint_step_500.pt` | ~1.1GB | Step 500 | - |
| `checkpoint_step_1000.pt` | ~1.1GB | Step 1000 | - |
| ... | ... | 500 step 간격 | - |
| `checkpoint_step_5000.pt` | ~1.1GB | Step 5000 | - |

**Total**: 12 files, ~13GB

### Logs (`logs/`)

주요 로그 파일:
- `train_quick_20260201_v2.log`: Quick test 학습 로그 (3h 5m, Loss 4.71→0.91)
- `evaluation_20260201_*.log`: 평가 결과 로그
- 모든 로그는 UTF-8 인코딩

---

## 🎯 Training Workflow

```
1. Environment Setup
   ├─→ Intel XPU: conda env + requirements_intel_xpu.txt
   └─→ NVIDIA GPU: conda env + requirements_cuda.txt

2. Training
   ├─→ [Quick] test_quick_xpu_small.yaml → checkpoints_quick/ (✅ Done)
   ├─→ [Standard] test_xpu.yaml → checkpoints_test_xpu/
   └─→ [Full] full_gpu.yaml → checkpoints_full/

3. Evaluation
   └─→ evaluate.py → logs/evaluation_*.log
```

---

## 💾 Storage Requirements

| Stage | Files | Checkpoints | Logs | Total |
|-------|-------|-------------|------|-------|
| Quick Test | 1 | ~13GB (12 files) | ~1MB | ~13GB |
| Standard Test | 10 | ~13GB (estimated) | ~10MB | ~13GB |
| Full Training | 324 | ~50GB (estimated) | ~100MB | ~50GB |

**Note**: 각 체크포인트는 ~1.1GB (BERT-base 모델 크기)

---

## 🔍 Key Files

### Must Read
1. **[README.md](README.md)**: 시작점
2. **[logbert/README.md](logbert/README.md)**: LogBERT 상세
3. **[logbert/docs/setup_guide.md](logbert/docs/setup_guide.md)**: 환경 설정

### Configuration
4. **[configs/test_quick_xpu_small.yaml](logbert/configs/test_quick_xpu_small.yaml)**: Quick test 설정
5. **[configs/test_xpu.yaml](logbert/configs/test_xpu.yaml)**: 10-file test 설정
6. **[configs/full_gpu.yaml](logbert/configs/full_gpu.yaml)**: 전체 학습 설정

### Results
7. **[logs/train_quick_20260201_v2.log](logbert/logs/train_quick_20260201_v2.log)**: 학습 결과
8. **[checkpoints_quick/best_model.pt](logbert/checkpoints_quick/best_model.pt)**: 최고 모델

---

## 🚀 Quick Navigation

**Start here**: [README.md](README.md)  
**Setup environment**: [logbert/docs/setup_guide.md](logbert/docs/setup_guide.md)  
**Run training**: [logbert/README.md](logbert/README.md#-quick-start)  
**View results**: [logbert/logs/](logbert/logs/)  
**Load model**: [logbert/checkpoints_quick/best_model.pt](logbert/checkpoints_quick/best_model.pt)

---

## 📊 Project Status

- ✅ **Phase 1**: Intel XPU 환경 구축 완료
- ✅ **Phase 2**: Quick test (1 file) 완료
- 🔄 **Phase 3**: Standard test (10 files) Ready
- 🔄 **Phase 4**: Full training (324 files, GPU 서버) Ready

---

**Last Updated**: 2026-02-01  
**Maintained by**: RADAR Team
