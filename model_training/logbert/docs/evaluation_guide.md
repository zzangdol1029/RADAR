# LogBERT 모델 성능 평가 가이드

## 📋 개요

학습된 LogBERT 모델의 실제 이상 탐지 성능을 평가하는 방법입니다. 정확도, 정밀도, 재현율, F1-Score 등의 메트릭을 계산하고 시각화합니다.

## ⚠️ 중요한 이해

### Loss vs 실제 성능

```
학습 Loss (예: 0.2029) ≠ 실제 이상 탐지 성능

Loss는 학습 진행 상황을 나타내는 지표일 뿐,
실제 이상 탐지 성능은 별도로 측정해야 합니다!
```

**Loss가 낮다는 것은:**
- 모델이 정상 로그 패턴을 잘 학습했다는 의미
- 마스킹된 토큰을 잘 예측한다는 의미
- 하지만 실제 이상을 얼마나 잘 탐지하는지는 별개

**실제 성능 측정이 필요한 이유:**
- 정상/이상을 얼마나 정확히 구분하는지 확인
- 오탐률(False Positive)과 미탐률(False Negative) 파악
- 임계값 최적화
- 프로덕션 배포 결정

## 🚀 사용 방법

### 1. 기본 실행

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints_quick_xpu/checkpoints/best_model.pt \
    --config configs/test_quick_xpu.yaml \
    --validation-data ../output/preprocessed_logs_000.json \
    --normal-ratio 0.8
```

**파라미터 설명:**
- `--checkpoint`: 평가할 모델 체크포인트 경로
- `--config`: 학습 시 사용한 설정 파일
- `--validation-data`: 검증용 전처리된 로그 데이터
- `--normal-ratio`: 데이터 중 정상으로 간주할 비율 (기본값: 0.8)

### 2. 여러 체크포인트 비교

```bash
# 최고 성능 모델
python scripts/evaluate.py \
    --checkpoint checkpoints_quick_xpu/checkpoints/best_model.pt \
    --config configs/test_quick_xpu.yaml \
    --validation-data ../output/preprocessed_logs_000.json \
    --output-dir evaluation_results/best_model

# 마지막 에폭 모델
python scripts/evaluate.py \
    --checkpoint checkpoints_quick_xpu/checkpoints/epoch_1.pt \
    --config configs/test_quick_xpu.yaml \
    --validation-data ../output/preprocessed_logs_000.json \
    --output-dir evaluation_results/epoch_1
```

### 3. 커스텀 로그 파일 지정

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints_quick_xpu/checkpoints/best_model.pt \
    --config configs/test_quick_xpu.yaml \
    --validation-data ../output/preprocessed_logs_000.json \
    --log-file logs/evaluation_experiment1.log
```

## 📊 출력 결과

### 1. 콘솔 출력

```
================================================================================
LogBERT 모델 성능 평가
================================================================================
디바이스: xpu
모델 로딩 중: checkpoints_quick_xpu/checkpoints/best_model.pt
체크포인트 정보:
  Global Step: 100
  Best Loss: 0.2029
✅ 모델 로드 완료

================================================================================
이상 점수 계산 중...
================================================================================
정상 세션 평가 중...
이상 세션 평가 중...
✅ 정상 세션 점수 계산 완료: 800개
✅ 이상 세션 점수 계산 완료: 200개

================================================================================
점수 통계
================================================================================
정상 세션 - 평균: 0.1523, 표준편차: 0.0421
정상 세션 - 최소: 0.0812, 최대: 0.2934
이상 세션 - 평균: 0.8724, 표준편차: 0.1523
이상 세션 - 최소: 0.4521, 최대: 1.5234

================================================================================
최적 임계값 탐색 중...
================================================================================
✅ 최적 임계값: 0.3521

================================================================================
📊 성능 평가 결과
================================================================================
정확도 (Accuracy):  0.9450 (94.50%)
정밀도 (Precision): 0.9123 (91.23%)
재현율 (Recall):    0.8950 (89.50%)
F1-Score:          0.9035 (90.35%)
ROC AUC:           0.9623

혼동 행렬:
  True Negative (TN):   760 (정상을 정상으로 예측)
  False Positive (FP):   40 (정상을 이상으로 예측)
  False Negative (FN):   21 (이상을 정상으로 예측)
  True Positive (TP):   179 (이상을 이상으로 예측)

================================================================================
✅ 평가 완료!
================================================================================
결과 저장 위치: evaluation_results
```

### 2. 저장되는 파일

**`evaluation_results/` 디렉토리:**

1. **`evaluation_results.json`** - 평가 결과 (JSON 형식)
   ```json
   {
     "checkpoint": "checkpoints_quick_xpu/checkpoints/best_model.pt",
     "optimal_threshold": 0.3521,
     "metrics": {
       "accuracy": 0.9450,
       "precision": 0.9123,
       "recall": 0.8950,
       "f1_score": 0.9035,
       "roc_auc": 0.9623
     },
     "confusion_matrix": {
       "true_negative": 760,
       "false_positive": 40,
       "false_negative": 21,
       "true_positive": 179
     },
     "statistics": {
       "normal_mean": 0.1523,
       "normal_std": 0.0421,
       "anomaly_mean": 0.8724,
       "anomaly_std": 0.1523
     }
   }
   ```

2. **`score_distribution.png`** - 점수 분포 그래프
   - 정상 vs 이상 점수 히스토그램
   - 박스플롯
   - 최적 임계값 표시

3. **`confusion_matrix.png`** - 혼동 행렬 시각화
   - True Negative, False Positive
   - False Negative, True Positive

4. **로그 파일** - `logs/evaluation_YYYYMMDD_HHMMSS.log`

## 📈 메트릭 해석

### 1. 정확도 (Accuracy)

```python
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**의미:** 전체 예측 중 맞춘 비율

**예시:**
- Accuracy 0.9450 (94.50%)
- 1000개 세션 중 945개를 정확히 분류

**평가 기준:**
- 90% 이상: 우수 ✅
- 80-90%: 양호 ⚠️
- 80% 미만: 개선 필요 ❌

### 2. 정밀도 (Precision)

```python
Precision = TP / (TP + FP)
```

**의미:** 이상으로 예측한 것 중 실제 이상인 비율

**예시:**
- Precision 0.9123 (91.23%)
- 경고 100개 중 91개가 실제 이상
- 오탐 9개 (매우 낮음)

**중요성:**
- 오탐률을 낮추는 것이 중요한 경우
- 알림이 많으면 신뢰도가 떨어짐

**평가 기준:**
- 90% 이상: 우수 (오탐 10% 미만) ✅
- 80-90%: 양호 (오탐 10-20%) ⚠️
- 80% 미만: 개선 필요 ❌

### 3. 재현율 (Recall)

```python
Recall = TP / (TP + FN)
```

**의미:** 실제 이상 중 탐지한 비율

**예시:**
- Recall 0.8950 (89.50%)
- 실제 이상 100개 중 89개 탐지
- 미탐 11개

**중요성:**
- 이상을 놓치면 안 되는 경우
- 보안, 시스템 장애 등

**평가 기준:**
- 90% 이상: 우수 (미탐 10% 미만) ✅
- 80-90%: 양호 (미탐 10-20%) ⚠️
- 80% 미만: 개선 필요 ❌

### 4. F1-Score

```python
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
```

**의미:** Precision과 Recall의 조화 평균

**예시:**
- F1-Score 0.9035 (90.35%)
- 정밀도와 재현율의 균형적 성능

**평가 기준:**
- 90% 이상: 우수 ✅
- 80-90%: 양호 ⚠️
- 80% 미만: 개선 필요 ❌

### 5. ROC AUC

**의미:** ROC 곡선 아래 면적

**예시:**
- ROC AUC 0.9623
- 매우 우수한 분류 성능

**평가 기준:**
- 0.9 이상: 우수 ✅
- 0.8-0.9: 양호 ⚠️
- 0.8 미만: 개선 필요 ❌

## 🎯 성능 개선 방법

### 성능이 낮은 경우 (정확도 < 80%)

**1. 학습 데이터 증가**
```bash
# limit_files를 늘려서 더 많은 데이터로 학습
python scripts/train_intel.py --config configs/test_xpu.yaml  # 10개 파일
python scripts/train_intel.py --config configs/full.yaml      # 전체 파일
```

**2. 에폭 수 증가**
```yaml
# configs/test_xpu.yaml
training:
  num_epochs: 5  # 1 → 5로 증가
```

**3. 데이터 분리 비율 조정**
```bash
# normal_ratio 조정
python scripts/evaluate.py \
    --normal-ratio 0.7  # 0.8 → 0.7 (더 많은 이상 데이터)
```

**4. 모델 크기 증가**
```yaml
# configs/test_xpu.yaml
model:
  hidden_size: 512      # 256 → 512
  num_layers: 8         # 4 → 8
  num_attention_heads: 8  # 4 → 8
```

### Precision이 낮은 경우 (오탐이 많음)

**임계값을 높게 설정:**
- 더 확실한 경우에만 이상으로 판단
- 오탐은 줄지만 미탐이 증가할 수 있음

### Recall이 낮은 경우 (미탐이 많음)

**임계값을 낮게 설정:**
- 조금이라도 의심되면 이상으로 판단
- 미탐은 줄지만 오탐이 증가할 수 있음

## 📌 주의사항

### 1. 데이터 분리 방법

현재는 `normal_ratio`로 단순 분리합니다:
- 앞 80%를 정상으로 간주
- 뒤 20%를 이상으로 간주

**실제 환경에서는:**
- 실제 레이블이 있는 데이터 사용 권장
- 정상/이상이 명확히 분리된 데이터셋 준비

### 2. 과적합 확인

**학습 Loss가 매우 낮은데 (예: 0.05) 평가 성능이 낮다면:**
- 과적합 발생 가능성
- 학습 데이터에만 과도하게 맞춰진 상태
- 더 많은 다양한 데이터로 재학습 필요

### 3. 임계값 최적화

**평가 스크립트는 F1-Score를 최대화하는 임계값을 찾습니다:**
- Precision과 Recall의 균형
- 실제 사용 시에는 요구사항에 따라 조정 필요

**예시:**
- 보안 시스템: Recall 우선 (미탐 최소화)
- 알림 시스템: Precision 우선 (오탐 최소화)

## 💡 예상 성능 (Loss 기준)

| 학습 Loss | 예상 정확도 | 예상 F1-Score | 평가 |
|----------|-----------|--------------|------|
| 0.9 이상 | < 80% | < 0.75 | 학습 더 필요 ❌ |
| 0.5-0.9 | 80-90% | 0.75-0.85 | 양호 ⚠️ |
| 0.2-0.5 | 90-95% | 0.85-0.92 | 우수 ✅ |
| < 0.2 | 95%+ | 0.92+ | 매우 우수 ⭐ |

**주의:** 이는 일반적인 예상치이며, 실제 성능은 데이터와 환경에 따라 다릅니다!

## 🎉 다음 단계

### 성능이 좋은 경우 (F1-Score > 90%)

1. **프로덕션 배포 준비**
   - 임계값 저장
   - 모델 최적화
   - API 구축

2. **더 많은 데이터로 검증**
   ```bash
   # 여러 파일로 평가
   for file in ../output/preprocessed_logs_*.json; do
       python scripts/evaluate.py \
           --checkpoint checkpoints_quick_xpu/checkpoints/best_model.pt \
           --config configs/test_quick_xpu.yaml \
           --validation-data "$file"
   done
   ```

3. **실시간 이상 탐지 시스템 구축**

### 성능이 부족한 경우

1. **전체 데이터로 재학습**
2. **하이퍼파라미터 튜닝**
3. **모델 구조 개선**

---

**작업 완료일**: 2026-02-01  
**작성자**: Antigravity
