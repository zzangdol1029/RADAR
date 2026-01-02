# 앙상블 학습 가이드

## 앙상블의 이점

### 성능 향상
- **단일 모델**: 70-75% 정확도
- **앙상블 (3개 모델)**: 73-78% 정확도 (+3-4%)
- **최적 앙상블**: 75-80% 정확도 (+5-6%)

### 안정성 향상
- 한 모델의 오류를 다른 모델이 보완
- 더 일관된 예측
- 오탐률 감소

## 사용 가능한 모델

### 1. DistilBERT (권장 - M4 Pro 최적)
```python
type: "distilbert"
pretrained_model: "distilbert-base-uncased"
```
- **장점**: 빠르고 효율적, 작은 메모리
- **성능**: 70-75%
- **학습 시간**: 1-2시간

### 2. BERT-base
```python
type: "bert"
pretrained_model: "bert-base-uncased"
```
- **장점**: 균형잡힌 성능
- **성능**: 70-75%
- **학습 시간**: 2-3시간

### 3. RoBERTa
```python
type: "roberta"
pretrained_model: "roberta-base"
```
- **장점**: 높은 성능
- **성능**: 75-80%
- **학습 시간**: 3-4시간

### 4. LSTM (시퀀스 특화)
```python
type: "lstm"
```
- **장점**: 시퀀스 패턴에 특화, 빠른 추론
- **성능**: 65-70%
- **학습 시간**: 1-2시간

## 앙상블 방법

### 방법 1: 동시 학습 (train_ensemble.py)

여러 모델을 동시에 학습:

```bash
python train_ensemble.py
```

**장점:**
- ✅ 한 번에 학습
- ✅ 모델 간 상호작용 가능

**단점:**
- ❌ 메모리 많이 사용
- ❌ 학습 시간 길어짐

### 방법 2: 개별 학습 후 결합 (권장)

각 모델을 따로 학습한 후 앙상블:

```bash
# 1. DistilBERT 학습
python train_transfer.py --pretrained distilbert-base-uncased --output-dir checkpoints_distilbert

# 2. BERT 학습
python train_transfer.py --pretrained bert-base-uncased --output-dir checkpoints_bert

# 3. RoBERTa 학습
python train_transfer.py --pretrained roberta-base --output-dir checkpoints_roberta

# 4. 앙상블 추론
python ensemble_simple.py \
  --checkpoints \
    checkpoints_distilbert/checkpoints/best_model.pt \
    checkpoints_bert/checkpoints/best_model.pt \
    checkpoints_roberta/checkpoints/best_model.pt \
  --model-types distilbert bert roberta \
  --input ../preprocessing/output/preprocessed_logs_2025-02-24.json \
  --output ensemble_results.json \
  --method weighted_average \
  --weights 0.4 0.4 0.2
```

**장점:**
- ✅ 메모리 효율적
- ✅ 모델별 최적화 가능
- ✅ 유연한 조합

**단점:**
- ❌ 여러 번 학습 필요

## 권장 앙상블 조합

### 조합 1: M4 Pro 최적화 (권장)

```
1. DistilBERT (빠르고 효율적)
2. BERT-base (균형잡힌 성능)
3. 작은 LSTM (시퀀스 특화)
```

**예상 성능**: 73-78% 정확도
**학습 시간**: 4-6시간 (순차 학습)
**메모리**: 각 모델별로 4-8GB

### 조합 2: 성능 중심

```
1. DistilBERT
2. BERT-base
3. RoBERTa
```

**예상 성능**: 75-80% 정확도
**학습 시간**: 6-9시간
**메모리**: 각 모델별로 4-12GB

### 조합 3: 다양성 중심

```
1. DistilBERT (Transformer)
2. LSTM (RNN)
3. 작은 Transformer (커스텀)
```

**예상 성능**: 72-77% 정확도
**학습 시간**: 4-6시간

## 앙상블 방법 비교

### Weighted Average (가중 평균) - 권장

```python
final_score = w1*score1 + w2*score2 + w3*score3
```

**장점:**
- ✅ 모델별 성능에 따라 가중치 조정 가능
- ✅ 가장 일반적으로 사용
- ✅ 좋은 성능

**가중치 예시:**
- DistilBERT: 0.4
- BERT: 0.4
- RoBERTa: 0.2 (더 좋은 성능이면 높게)

### Average (단순 평균)

```python
final_score = (score1 + score2 + score3) / 3
```

**장점:**
- ✅ 간단함
- ✅ 모든 모델 동등하게 취급

**단점:**
- ❌ 성능 차이 반영 안 됨

### Max (최대값)

```python
final_score = max(score1, score2, score3)
```

**장점:**
- ✅ 보수적 접근 (높은 점수만 선택)

**단점:**
- ❌ 오탐률 높을 수 있음

## 실행 예시

### 1단계: 개별 모델 학습

```bash
# DistilBERT
python train_transfer.py \
  --pretrained distilbert-base-uncased \
  --sample-ratio 0.05 \
  --max-files 10 \
  --output-dir checkpoints_distilbert

# BERT
python train_transfer.py \
  --pretrained bert-base-uncased \
  --sample-ratio 0.05 \
  --max-files 10 \
  --output-dir checkpoints_bert
```

### 2단계: 앙상블 추론

```bash
python ensemble_simple.py \
  --checkpoints \
    checkpoints_distilbert/checkpoints/best_model.pt \
    checkpoints_bert/checkpoints/best_model.pt \
  --model-types distilbert bert \
  --input ../preprocessing/output/preprocessed_logs_2025-02-24.json \
  --output ensemble_results.json \
  --method weighted_average \
  --weights 0.5 0.5 \
  --threshold 2.0
```

## 예상 성능 향상

### 단일 모델
- DistilBERT: 70-75%
- BERT-base: 70-75%
- LSTM: 65-70%

### 앙상블 (2개 모델)
- DistilBERT + BERT: 72-77% (+2-3%)
- DistilBERT + LSTM: 71-76% (+1-2%)

### 앙상블 (3개 모델)
- DistilBERT + BERT + RoBERTa: 75-80% (+4-5%)
- DistilBERT + BERT + LSTM: 73-78% (+3-4%)

## M4 Pro에서의 현실

### 가능한 것
- ✅ 2-3개 모델 순차 학습 (각각 2-4시간)
- ✅ 앙상블 추론 (빠름)
- ✅ 73-78% 정확도 확보

### 제약사항
- ❌ 동시 학습은 메모리 부족
- ❌ 4개 이상 모델은 시간이 너무 오래 걸림

## 권장 워크플로우

```
1단계: DistilBERT 학습 (2시간)
  → 기본 성능 확인

2단계: BERT 학습 (3시간)
  → 성능 비교

3단계: 앙상블 추론 (수 분)
  → 성능 향상 확인

4단계: 필요시 RoBERTa 추가 (4시간)
  → 최고 성능 확보
```

## 결론

**앙상블로 2-5% 성능 향상이 가능합니다.**

**M4 Pro 환경에서의 권장:**
1. DistilBERT + BERT 앙상블 (5시간 학습, 73-77% 성능)
2. 필요시 RoBERTa 추가 (9시간 학습, 75-80% 성능)

**비용 대비 효과:**
- 추가 학습 시간: 3-4시간
- 성능 향상: +3-5%
- **가치 있음**: 프로토타입/검증용으로 충분

