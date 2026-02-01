# LogBERT Setup Guide

## 📋 Prerequisites

### Required Software
- Python 3.8+
- Conda (Anaconda or Miniconda)
- Git

### Hardware Requirements

#### For Intel XPU (로컬 PC)
- Intel Arc Graphics (16GB VRAM recommended)
- 16GB+ RAM
- Windows 10/11 or Linux

#### For NVIDIA GPU (서버)
- NVIDIA GPU (RTX 3090/4090 recommended, 24GB+ VRAM)
- 32GB+ RAM
- CUDA 11.8+ compatible

---

## 🚀 Quick Start

### 1. Intel XPU Setup (로컬 PC)

#### Step 1: Create Conda Environment
```bash
conda create -n logbert_ipex python=3.10 -y
conda activate logbert_ipex
```

#### Step 2: Install Dependencies
```bash
# Navigate to logbert directory
cd C:\workspace\RADAR\model_training\logbert

# Install Intel XPU requirements
pip install -r requirements_intel_xpu.txt
```

#### Step 3: Verify Installation
```bash
python -c "import torch; import intel_extension_for_pytorch as ipex; print(f'PyTorch: {torch.__version__}'); print(f'IPEX: {ipex.__version__}'); print(f'XPU Available: {torch.xpu.is_available()}')"
```

Expected output:
```
PyTorch: 2.5.1+cxx11.abi
IPEX: 2.5.10+xpu
XPU Available: True
```

### 2. NVIDIA GPU Setup (서버)

#### Step 1: Create Conda Environment
```bash
conda create -n logbert_cuda python=3.10 -y
conda activate logbert_cuda
```

#### Step 2: Install Dependencies
```bash
cd /path/to/RADAR/model_training/logbert
pip install -r requirements_cuda.txt
```

#### Step 3: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## 🏃 Running Training

### Quick Test (1 file) - Intel XPU
```powershell
# Using PowerShell script
.\run_quick_test.ps1
```

Or manually:
```bash
conda activate logbert_ipex
python scripts/train_intel.py --config configs/test_quick_xpu_small.yaml
```

**Expected:**
- Time: ~3 hours
- Final Loss: ~0.91
- Output: `checkpoints_quick/`

### Standard Test (10 files) - Intel XPU
```bash
conda activate logbert_ipex
python scripts/train_intel.py --config configs/test_xpu.yaml
```

**Expected:**
- Time: 3-5 hours
- Output: `checkpoints_test_xpu/`

### Full Training (324 files) - NVIDIA GPU
```bash
conda activate logbert_cuda
python scripts/train_cuda.py --config configs/full_gpu.yaml
```

**Expected:**
- Time: 3-5 days (RTX 4090)
- Output: `checkpoints_full/`

---

## 📊 Evaluation

```bash
# Intel XPU
python scripts/evaluate.py --checkpoint checkpoints_quick/best_model.pt --config configs/test_quick_xpu_small.yaml

# NVIDIA GPU
python scripts/evaluate.py --checkpoint checkpoints_full/best_model.pt --config configs/full_gpu.yaml
```

---

## 🔧 Troubleshooting

### Intel XPU Issues

**Problem: `torch.xpu.is_available()` returns False**
```bash
# Reinstall Intel GPU drivers
# Download from: https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html

# Reinstall PyTorch with XPU support
pip uninstall torch intel-extension-for-pytorch -y
pip install torch==2.5.1 intel-extension-for-pytorch==2.5.10 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

**Problem: Out of Memory Error**
- Reduce `batch_size` in config file (e.g., 32 → 16 → 8)
- Reduce `num_workers` to 0

### NVIDIA GPU Issues

**Problem: CUDA Out of Memory**
- Reduce `batch_size` in config
- Use gradient accumulation

**Problem: CUDA version mismatch**
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
# Refer to: https://pytorch.org/get-started/locally/
```

---

## 📁 Project Structure

```
logbert/
├── configs/              # Configuration files
│   ├── test_quick_xpu_small.yaml  # 1 file, Intel XPU
│   ├── test_xpu.yaml              # 10 files, Intel XPU
│   └── full_gpu.yaml              # 324 files, NVIDIA GPU
├── scripts/              # Training scripts
│   ├── train_intel.py    # Intel XPU training
│   ├── train_cuda.py     # NVIDIA GPU training
│   └── evaluate.py       # Model evaluation
├── docs/                 # Documentation
├── checkpoints/          # Model checkpoints (empty initially)
├── checkpoints_quick/    # Quick test checkpoints
└── logs/                 # Training logs
```

---

## 📝 Configuration Guide

### Key Parameters

**model** section:
- `vocab_size`: 10000 (fixed)
- `hidden_size`: 768 (BERT-base)
- `num_hidden_layers`: 12

**training** section:
- `batch_size`: 8-64 (adjust based on GPU memory)
- `learning_rate`: 2e-5
- `num_epochs`: 1-3
- `num_workers`: 0-8 (parallel data loading)

**data** section:
- `limit_files`: 1, 10, or null (all files)
- `max_seq_length`: 512

---

## 🎯 Next Steps

1. **After Quick Test**: Review logs in `logs/` directory
2. **Check Results**: Use evaluation script
3. **Scale Up**: Run 10-file test or full training
4. **Monitor**: Check GPU utilization with `nvidia-smi` or Intel GPU tools
