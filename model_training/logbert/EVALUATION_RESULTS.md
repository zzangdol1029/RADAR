# LogBERT 모델 평가 결과

## 개요

LogBERT 모델 3개 버전(Epoch 1, 2, 3)에 대한 이상 탐지 성능 평가 결과입니다.
정상 데이터만으로 학습된 모델의 성능을 검증하기 위해 **Pseudo-Anomaly** 방식을 사용했습니다.

### 평가 방법
- **데이터**: 검증 데이터 1,000개 세션
- **방식**: `--generate-fake-anomaly` 옵션으로 정상 데이터의 2%를 무작위 변조
- **측정 지표**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 📊 성능 비교표

| Epoch | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Normal Loss | Anomaly Loss | FP (오탐) | FN (미탐) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **95.85%** | **100.00%** | 91.70% | 0.9567 | **0.9670** | 17.27 | 19.52 | **0** | 83 |
| **2** | 95.65% | 99.78% | 91.50% | 0.9546 | 0.9659 | 14.66 | 16.68 | 2 | 85 |
| **3** | **96.00%** | 99.89% | **92.10%** | **0.9584** | 0.9636 | **12.53** | 14.38 | 1 | **79** |

---

## 🔍 핵심 분석

### 1. Validation Loss 추이 (정상 데이터)

```
Epoch 1: 17.27
Epoch 2: 14.66  (-15%)
Epoch 3: 12.53  (-27%)
```

✅ **Loss가 지속적으로 감소** → 모델이 학습을 거듭할수록 정상 패턴을 더 잘 이해하고 있음을 의미합니다.

### 2. 탐지 성능 비교

#### Epoch 1: 가장 보수적
- **Precision 100%**: 오탐(False Positive) 0건
- **Recall 91.7%**: 미탐(False Negative) 83건
- **특징**: 절대 오탐 금지 환경에 적합

#### Epoch 2: 균형잡힘
- **Precision 99.78%**: 오탐 2건
- **Recall 91.5%**: 미탐 85건
- **특징**: 안정적인 중간 성능

#### Epoch 3: 종합 최고 ⭐
- **Precision 99.89%**: 오탐 1건 (거의 없음)
- **Recall 92.1%**: 미탐 79건 (**가장 적음**)
- **Accuracy 96.0%**: **가장 높은 정확도**
- **Normal Loss 12.53**: **가장 낮은 Loss** (정상 패턴 학습 최고)

---

## 💡 최종 권장사항

| 사용 목적 | 추천 모델 | 이유 |
|:---|:---:|:---|
| **🎯 프로덕션 배포** | **Epoch 3** | 가장 높은 정확도(96%), 가장 낮은 Normal Loss, 최고의 Recall |
| **🔒 절대 오탐 금지** | Epoch 1 | Precision 100% (오탐 0건) |
| **📊 PoC/데모** | Epoch 2 또는 3 | 충분한 성능, 안정적 |

### ✅ 최종 결론

**`checkpoints_full/checkpoints/epoch_3.pt` 모델을 프로덕션 배포용으로 권장합니다.**

**선정 근거:**
1. **학습 수준**: Normal Loss가 12.53으로 가장 낮아 정상 패턴을 가장 잘 이해
2. **균형 성능**: 오탐(1건)과 미탐(79건) 모두 최소화
3. **종합 정확도**: 96.00%로 3개 모델 중 최고
4. **실전 적합성**: Precision과 Recall이 모두 우수하여 실제 운영 환경에 최적

---

## 📁 평가 결과 파일

각 Epoch별 상세 결과는 다음 디렉토리에 저장되어 있습니다:

```
evaluation_results/
├── epoch_1/
│   ├── evaluation_results_epoch_1.json
│   ├── score_dist_epoch_1.png
│   └── confusion_matrix_epoch_1.png
├── epoch_2/
│   ├── evaluation_results_epoch_2.json
│   ├── score_dist_epoch_2.png
│   └── confusion_matrix_epoch_2.png
└── epoch_3/
    ├── evaluation_results_epoch_3.json
    ├── score_dist_epoch_3.png
    └── confusion_matrix_epoch_3.png
```

---

## 🔬 평가 재현 방법

동일한 평가를 재현하려면 다음 명령어를 사용하세요:

```bash
# Epoch 1 평가
python evaluate.py \
    --checkpoint checkpoints_full/checkpoints/epoch_1.pt \
    --config configs/full_gpu.yaml \
    --validation-data ../../preprocessing/output \
    --output-dir evaluation_results \
    --max-samples 1000 \
    --generate-fake-anomaly \
    --anomaly-ratio 0.02

# Epoch 2 평가
python evaluate.py \
    --checkpoint checkpoints_full/checkpoints/epoch_2.pt \
    --config configs/full_gpu.yaml \
    --validation-data ../../preprocessing/output \
    --output-dir evaluation_results \
    --max-samples 1000 \
    --generate-fake-anomaly \
    --anomaly-ratio 0.02

# Epoch 3 평가
python evaluate.py \
    --checkpoint checkpoints_full/checkpoints/epoch_3.pt \
    --config configs/full_gpu.yaml \
    --validation-data ../../preprocessing/output \
    --output-dir evaluation_results \
    --max-samples 1000 \
    --generate-fake-anomaly \
    --anomaly-ratio 0.02
```

---

**평가일**: 2026-02-09  
**평가자**: LogBERT Training Team
