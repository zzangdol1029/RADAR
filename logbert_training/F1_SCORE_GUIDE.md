# F1-Score 가이드

## 📊 F1-Score란?

### 정의

**F1-Score**는 정밀도(Precision)와 재현율(Recall)의 **조화 평균(Harmonic Mean)**입니다.

```
F1-Score = 2 × (정밀도 × 재현율) / (정밀도 + 재현율)
```

### 왜 조화 평균인가?

**조화 평균**은 극단값에 민감하여, 한 지표가 낮으면 전체 점수가 크게 떨어집니다.

**예시:**
```
정밀도: 90%, 재현율: 10%
→ 산술 평균: (90 + 10) / 2 = 50%
→ 조화 평균: 2 × (90 × 10) / (90 + 10) = 18%

정밀도: 50%, 재현율: 50%
→ 산술 평균: 50%
→ 조화 평균: 50%
```

**의미**: F1-Score는 두 지표를 **균형있게** 고려합니다.

---

## 🎯 F1-Score의 의미

### 핵심 개념

**F1-Score는 정밀도와 재현율의 균형을 나타냅니다.**

- **높은 F1-Score**: 정밀도와 재현율이 모두 높음 → 이상적인 모델
- **낮은 F1-Score**: 한 지표가 낮거나 둘 다 낮음 → 개선 필요

### 예시

#### 예시 1: 균형잡힌 모델

```
정밀도: 80%
재현율: 80%

F1-Score = 2 × (0.8 × 0.8) / (0.8 + 0.8)
         = 2 × 0.64 / 1.6
         = 1.28 / 1.6
         = 0.8 (80%)
```

**해석**: 정밀도와 재현율이 균형있게 높음 ✅

#### 예시 2: 정밀도 높음, 재현율 낮음

```
정밀도: 90%
재현율: 50%

F1-Score = 2 × (0.9 × 0.5) / (0.9 + 0.5)
         = 2 × 0.45 / 1.4
         = 0.9 / 1.4
         = 0.643 (64.3%)
```

**해석**: 재현율이 낮아서 F1-Score가 떨어짐 ⚠️

#### 예시 3: 재현율 높음, 정밀도 낮음

```
정밀도: 50%
재현율: 90%

F1-Score = 2 × (0.5 × 0.9) / (0.5 + 0.9)
         = 2 × 0.45 / 1.4
         = 0.9 / 1.4
         = 0.643 (64.3%)
```

**해석**: 정밀도가 낮아서 F1-Score가 떨어짐 ⚠️

---

## 📈 F1-Score 범위 및 해석

### 점수 범위

```
F1-Score 범위: 0 ~ 1 (또는 0% ~ 100%)
```

### 해석 가이드

| F1-Score | 해석 | 모델 상태 |
|----------|------|----------|
| **0.9 ~ 1.0** | 우수 | 매우 좋은 성능 |
| **0.8 ~ 0.9** | 좋음 | 좋은 성능 |
| **0.7 ~ 0.8** | 양호 | 실용적 사용 가능 |
| **0.6 ~ 0.7** | 보통 | 개선 필요 |
| **0.5 ~ 0.6** | 낮음 | 성능 부족 |
| **0.0 ~ 0.5** | 매우 낮음 | 거의 사용 불가 |

### 현재 모델 예상 F1-Score

```
정밀도: 80-88%
재현율: 75-85%

F1-Score = 2 × (0.8 × 0.75) / (0.8 + 0.75) ~ 2 × (0.88 × 0.85) / (0.88 + 0.85)
         = 2 × 0.6 / 1.55 ~ 2 × 0.748 / 1.73
         = 0.774 ~ 0.865
         = 77.4% ~ 86.5%
```

**해석**: 좋은 성능 (실용적 사용 가능) ✅

---

## ✅ 학습에 적용 가능한가?

### 답: 네, 적용 가능합니다!

**F1-Score는 이상 탐지 모델 평가에 매우 유용합니다.**

### 적용 방법

#### 1. 검증 데이터셋 필요

```python
# 정상/이상 레이블이 있는 검증 데이터
validation_data = {
    'normal_sessions': [...],  # 정상 세션 (레이블: 0)
    'anomaly_sessions': [...],  # 이상 세션 (레이블: 1)
}
```

#### 2. 예측 및 평가

```python
from sklearn.metrics import f1_score, precision_score, recall_score

# 이상 점수 계산
normal_scores = model.predict_anomaly_score(normal_sessions)
anomaly_scores = model.predict_anomaly_score(anomaly_sessions)

# 임계값 설정
threshold = 2.0  # 이상 점수 임계값

# 예측 생성
all_scores = np.concatenate([normal_scores, anomaly_scores])
all_labels = np.concatenate([
    np.zeros(len(normal_scores)),  # 정상: 0
    np.ones(len(anomaly_scores))   # 이상: 1
])

predictions = (all_scores >= threshold).astype(int)

# F1-Score 계산
f1 = f1_score(all_labels, predictions)
precision = precision_score(all_labels, predictions)
recall = recall_score(all_labels, predictions)

print(f"F1-Score: {f1:.4f}")
print(f"정밀도: {precision:.4f}")
print(f"재현율: {recall:.4f}")
```

#### 3. 학습 중 평가 (에폭마다)

```python
# 각 에폭 완료 후
def evaluate_model(model, validation_data, threshold):
    """모델 평가"""
    normal_scores = model.predict_anomaly_score(validation_data['normal'])
    anomaly_scores = model.predict_anomaly_score(validation_data['anomaly'])
    
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    all_labels = np.concatenate([
        np.zeros(len(normal_scores)),
        np.ones(len(anomaly_scores))
    ])
    
    predictions = (all_scores >= threshold).astype(int)
    
    f1 = f1_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 에폭 완료 후
metrics = evaluate_model(model, validation_data, threshold=2.0)
logger.info(f"F1-Score: {metrics['f1']:.4f}")
```

---

## 🔧 실제 적용 예시 코드

### 검증 스크립트 예시

```python
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from model import LogBERT
from dataset import LogBERTDataset

def evaluate_checkpoint(checkpoint_path, validation_data, threshold=2.0):
    """체크포인트 평가"""
    
    # 모델 로드
    checkpoint = torch.load(checkpoint_path)
    model = LogBERT(**checkpoint['config']['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 데이터 준비
    normal_sessions = validation_data['normal']
    anomaly_sessions = validation_data['anomaly']
    
    # 이상 점수 계산
    normal_scores = []
    anomaly_scores = []
    
    with torch.no_grad():
        # 정상 세션
        for session in normal_sessions:
            input_ids = torch.tensor([session['token_ids']])
            attention_mask = torch.tensor([session['attention_mask']])
            score = model.predict_anomaly_score(input_ids, attention_mask)
            normal_scores.append(score.item())
        
        # 이상 세션
        for session in anomaly_sessions:
            input_ids = torch.tensor([session['token_ids']])
            attention_mask = torch.tensor([session['attention_mask']])
            score = model.predict_anomaly_score(input_ids, attention_mask)
            anomaly_scores.append(score.item())
    
    # 예측 및 평가
    all_scores = np.array(normal_scores + anomaly_scores)
    all_labels = np.array([0] * len(normal_scores) + [1] * len(anomaly_scores))
    
    predictions = (all_scores >= threshold).astype(int)
    
    # 지표 계산
    accuracy = accuracy_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold
    }

# 사용 예시
metrics = evaluate_checkpoint(
    checkpoint_path='checkpoints/checkpoint_step_1000.pt',
    validation_data={
        'normal': normal_sessions,
        'anomaly': anomaly_sessions
    },
    threshold=2.0
)

print(f"정확도: {metrics['accuracy']:.4f}")
print(f"정밀도: {metrics['precision']:.4f}")
print(f"재현율: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
```

---

## 💡 F1-Score의 장단점

### 장점

1. **균형잡힌 평가**
   - 정밀도와 재현율을 동시에 고려
   - 한 지표에 치우치지 않음

2. **단일 지표**
   - 하나의 숫자로 성능 파악
   - 비교하기 쉬움

3. **임계값 최적화**
   - F1-Score를 최대화하는 임계값 찾기 가능

### 단점

1. **레이블 필요**
   - 정상/이상 레이블이 있는 검증 데이터 필요
   - 비지도 학습에서는 직접 사용 어려움

2. **임계값 의존**
   - 임계값에 따라 F1-Score가 달라짐
   - 최적 임계값 찾기 필요

3. **불균형 데이터**
   - 데이터가 불균형하면 해석이 어려울 수 있음

---

## 🎯 학습 과정에 통합하기

### 방법 1: 에폭마다 평가

```python
# train.py에 추가
def evaluate_epoch(model, validation_loader, threshold=2.0):
    """에폭 완료 후 평가"""
    model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in validation_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']  # 정상: 0, 이상: 1
            
            scores = model.predict_anomaly_score(input_ids, attention_mask)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    predictions = (all_scores >= threshold).astype(int)
    
    f1 = f1_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 에폭 완료 후
if epoch % 1 == 0:  # 매 에폭마다
    metrics = evaluate_epoch(model, validation_loader)
    logger.info(f"F1-Score: {metrics['f1']:.4f}")
    logger.info(f"정밀도: {metrics['precision']:.4f}")
    logger.info(f"재현율: {metrics['recall']:.4f}")
```

### 방법 2: Best Model 선택

```python
# F1-Score가 가장 높은 모델 저장
best_f1 = 0.0

for epoch in range(num_epochs):
    # 학습...
    
    # 평가
    metrics = evaluate_epoch(model, validation_loader)
    
    # Best Model 저장
    if metrics['f1'] > best_f1:
        best_f1 = metrics['f1']
        save_checkpoint(model, f'best_model_f1_{best_f1:.4f}.pt')
        logger.info(f"Best F1-Score: {best_f1:.4f} - 모델 저장")
```

---

## 📊 F1-Score vs 다른 지표

### 비교표

| 지표 | 장점 | 단점 | 사용 시기 |
|------|------|------|----------|
| **정확도** | 이해하기 쉬움 | 불균형 데이터에서 오해 | 균형 데이터 |
| **정밀도** | 오탐 최소화 | 재현율 무시 | 오탐 비용 높을 때 |
| **재현율** | 미탐지 최소화 | 정밀도 무시 | 미탐지 비용 높을 때 |
| **F1-Score** | 균형잡힌 평가 | 레이블 필요 | 종합 평가 |

---

## 🎯 요약

### F1-Score란?

- **정밀도와 재현율의 조화 평균**
- 두 지표를 균형있게 고려하는 종합 지표
- 범위: 0 ~ 1 (또는 0% ~ 100%)

### 학습에 적용 가능한가?

**네, 적용 가능합니다!**

**조건:**
- 정상/이상 레이블이 있는 검증 데이터 필요
- 임계값 설정 필요

**활용 방법:**
1. 에폭마다 F1-Score 계산
2. Best Model 선택 기준으로 사용
3. 임계값 최적화

### 현재 모델 예상 F1-Score

```
정밀도: 80-88%
재현율: 75-85%

예상 F1-Score: 77.4% ~ 86.5%
→ 좋은 성능 (실용적 사용 가능) ✅
```

**F1-Score는 이상 탐지 모델 평가에 매우 유용한 지표입니다!** 🎯

