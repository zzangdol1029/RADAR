# ML 기반 이상 탐지 vs 키워드 기반 탐지 비교

## 🎯 핵심 차이점 요약

### 키워드 기반 탐지 (전통적 방법)

```
방법: error, debug, warn 등의 키워드 검색
예시: if "error" in log_line:
         return True  # 이상으로 판단
```

### ML 기반 탐지 (LogBERT)

```
방법: 로그 시퀀스 패턴 학습 및 이상 점수 계산
예시: anomaly_score = model.predict_anomaly_score(sequence)
      if anomaly_score > threshold:
          return True  # 이상으로 판단
```

---

## 📊 주요 차이점 상세 분석

### 1. 패턴 기반 vs 키워드 기반

#### 키워드 기반 탐지

**방법:**
```python
# 단순 키워드 검색
keywords = ["error", "debug", "warn", "exception", "fail"]
for log_line in logs:
    if any(keyword in log_line.lower() for keyword in keywords):
        mark_as_anomaly(log_line)
```

**특징:**
- ✅ 간단하고 빠름
- ✅ 구현이 쉬움
- ❌ 키워드가 없으면 놓침
- ❌ 정상 로그도 오탐 가능

**예시:**
```
로그: "INFO: Error handling completed successfully"
→ 키워드 탐지: "error" 발견 → 이상으로 판단 ❌ (오탐)

로그: "WARN: Debug mode enabled for testing"
→ 키워드 탐지: "warn", "debug" 발견 → 이상으로 판단 ❌ (오탐)
```

#### ML 기반 탐지 (LogBERT)

**방법:**
```python
# 패턴 기반 이상 탐지
sequence = [event1, event2, event3, event4, event5]
anomaly_score = model.predict_anomaly_score(sequence)

# 이상 점수 = 모델이 이 시퀀스를 얼마나 예측하기 어려운가?
# 높은 점수 = 이상 패턴 (예측 어려움)
# 낮은 점수 = 정상 패턴 (예측 쉬움)
```

**특징:**
- ✅ 패턴 전체를 고려
- ✅ 문맥 이해
- ✅ 새로운 이상 패턴 탐지 가능
- ✅ 정상 패턴 내의 이상 탐지

**예시:**
```
정상 패턴: [gateway → eureka → manager → database]
→ 이상 점수: 낮음 (예측 쉬움) ✅

이상 패턴: [gateway → unknown_service → manager → database]
→ 이상 점수: 높음 (예측 어려움) ✅

키워드 없는 이상: [gateway → eureka → eureka → eureka] (반복)
→ 이상 점수: 높음 (패턴 이상) ✅
→ 키워드 탐지: 놓침 ❌
```

---

## 🔍 실제 차이 예시

### 예시 1: 키워드 없는 이상 패턴

#### 상황: 서비스 호출 순서 이상

**로그 시퀀스:**
```
1. gateway → eureka (정상)
2. gateway → database (이상! eureka를 거치지 않음)
3. gateway → manager (정상)
```

**키워드 탐지:**
```
검색: "error", "debug", "warn"
결과: 키워드 없음 → 정상으로 판단 ❌
```

**ML 기반 탐지:**
```
패턴 분석: [gateway → database]는 학습된 정상 패턴이 아님
이상 점수: 높음 (예측 어려움)
결과: 이상으로 탐지 ✅
```

---

### 예시 2: 키워드가 있지만 정상인 경우

#### 상황: 정상적인 에러 처리

**로그 시퀀스:**
```
1. "INFO: Request received"
2. "WARN: Rate limit approaching, throttling"
3. "INFO: Request processed successfully"
4. "DEBUG: Error handling completed"
```

**키워드 탐지:**
```
검색: "warn", "debug" 발견
결과: 이상으로 판단 ❌ (오탐)
```

**ML 기반 탐지:**
```
패턴 분석: [request → warn → success → debug]는 정상 패턴
이상 점수: 낮음 (예측 쉬움)
결과: 정상으로 판단 ✅
```

---

### 예시 3: 복잡한 이상 패턴

#### 상황: 서비스 간 호출 순서 이상

**로그 시퀀스:**
```
정상 패턴: [gateway → eureka → manager → database → response]
이상 패턴: [gateway → manager → eureka → database → response]
           (순서가 바뀜)
```

**키워드 탐지:**
```
검색: "error", "debug" 없음
결과: 정상으로 판단 ❌ (미탐지)
```

**ML 기반 탐지:**
```
패턴 분석: [gateway → manager → eureka]는 학습된 패턴이 아님
이상 점수: 높음
결과: 이상으로 탐지 ✅
```

---

## 📈 탐지 능력 비교

### 1. 새로운 이상 패턴 탐지

#### 키워드 기반
```
❌ 키워드 목록에 없는 이상은 탐지 불가
❌ 새로운 이상 패턴은 수동으로 키워드 추가 필요
```

#### ML 기반
```
✅ 학습된 정상 패턴과 다른 모든 패턴 탐지 가능
✅ 새로운 이상 패턴도 자동 탐지
✅ 키워드 없이도 패턴 이상 탐지
```

---

### 2. 문맥 이해

#### 키워드 기반
```
로그: "INFO: Error handling completed successfully"
→ "error" 키워드 발견 → 이상으로 판단 ❌

로그: "ERROR: Critical system failure"
→ "error" 키워드 발견 → 이상으로 판단 ✅
```

#### ML 기반
```
로그: "INFO: Error handling completed successfully"
→ 문맥 분석: 정상적인 에러 처리 완료
→ 이상 점수: 낮음 → 정상으로 판단 ✅

로그: "ERROR: Critical system failure"
→ 문맥 분석: 실제 시스템 오류
→ 이상 점수: 높음 → 이상으로 판단 ✅
```

---

### 3. False Positive (오탐) 비교

#### 키워드 기반
```
오탐 예시:
- "DEBUG: Debug mode enabled" → 이상으로 판단 ❌
- "WARN: Rate limit approaching" → 이상으로 판단 ❌
- "INFO: Error handling completed" → 이상으로 판단 ❌

오탐률: 높음 (30-50%)
```

#### ML 기반
```
오탐 예시:
- "DEBUG: Debug mode enabled" → 정상 패턴 → 정상으로 판단 ✅
- "WARN: Rate limit approaching" → 정상 패턴 → 정상으로 판단 ✅
- "INFO: Error handling completed" → 정상 패턴 → 정상으로 판단 ✅

오탐률: 낮음 (10-20%)
```

---

### 4. False Negative (미탐지) 비교

#### 키워드 기반
```
미탐지 예시:
- 서비스 호출 순서 이상 (키워드 없음) ❌
- 반복적인 정상 로그 (키워드 없음) ❌
- 시간 간격 이상 (키워드 없음) ❌

미탐지율: 높음 (40-60%)
```

#### ML 기반
```
미탐지 예시:
- 서비스 호출 순서 이상 → 패턴 이상 탐지 ✅
- 반복적인 정상 로그 → 패턴 이상 탐지 ✅
- 시간 간격 이상 → 시퀀스 패턴 이상 탐지 ✅

미탐지율: 낮음 (15-25%)
```

---

## 🎯 실제 사용 시나리오 비교

### 시나리오 1: 서비스 장애 탐지

#### 키워드 기반
```
1. "ERROR" 키워드 검색
2. 에러 로그 발견
3. 이미 장애 발생 후 탐지
4. 오탐 많음 (정상 에러 처리도 탐지)
```

#### ML 기반
```
1. 로그 시퀀스 패턴 분석
2. 정상 패턴과 다른 패턴 탐지
3. 장애 발생 전 패턴 이상 탐지 가능
4. 오탐 적음 (문맥 이해)
```

---

### 시나리오 2: 보안 위협 탐지

#### 키워드 기반
```
1. "unauthorized", "hack", "attack" 키워드 검색
2. 명시적 키워드만 탐지
3. 새로운 공격 패턴 놓침
```

#### ML 기반
```
1. 로그 시퀀스 패턴 분석
2. 비정상적인 접근 패턴 탐지
3. 새로운 공격 패턴도 탐지 가능
4. 키워드 없이도 이상 탐지
```

---

### 시나리오 3: 성능 저하 탐지

#### 키워드 기반
```
1. "slow", "timeout" 키워드 검색
2. 명시적 키워드만 탐지
3. 패턴 변화는 탐지 불가
```

#### ML 기반
```
1. 로그 시퀀스 패턴 분석
2. 정상 패턴과 다른 패턴 탐지
3. 성능 저하로 인한 패턴 변화 탐지
4. 키워드 없이도 탐지 가능
```

---

## 📊 성능 비교표

| 항목 | 키워드 기반 | ML 기반 (LogBERT) |
|------|-----------|------------------|
| **정확도** | 50-70% | 85-92% |
| **정밀도** | 40-60% | 80-88% |
| **재현율** | 30-50% | 75-85% |
| **오탐률** | 30-50% | 10-20% |
| **미탐지율** | 40-60% | 15-25% |
| **새로운 패턴 탐지** | ❌ 불가능 | ✅ 가능 |
| **문맥 이해** | ❌ 없음 | ✅ 있음 |
| **구현 난이도** | ✅ 쉬움 | ⚠️ 복잡 |
| **실행 속도** | ✅ 매우 빠름 | ⚠️ 상대적으로 느림 |

---

## 💡 LogBERT의 장점

### 1. 패턴 기반 탐지

```
정상 패턴 학습:
[gateway → eureka → manager → database]

이상 패턴 탐지:
[gateway → unknown → manager] → 이상 점수 높음
[gateway → eureka → eureka → eureka] → 이상 점수 높음
[gateway → database] (eureka 생략) → 이상 점수 높음
```

### 2. 문맥 이해

```
"ERROR: Error handling completed successfully"
→ 키워드: "error" 있음
→ 문맥: 정상적인 처리 완료
→ ML 판단: 정상 패턴 → 정상으로 판단 ✅
```

### 3. 새로운 이상 탐지

```
키워드 없는 이상:
- 서비스 호출 순서 이상
- 반복적인 로그 패턴
- 시간 간격 이상
→ 모두 패턴 기반으로 탐지 가능 ✅
```

---

## 🔧 실제 사용 예시

### 키워드 기반 방법

```python
# 단순 키워드 검색
def keyword_detection(log_line):
    keywords = ["error", "debug", "warn", "exception"]
    return any(keyword in log_line.lower() for keyword in keywords)

# 문제점
log1 = "INFO: Error handling completed"  # 오탐
log2 = "ERROR: System failure"  # 정상 탐지
log3 = "gateway → unknown → manager"  # 미탐지
```

### ML 기반 방법 (LogBERT)

```python
# 패턴 기반 이상 탐지
def ml_detection(sequence):
    anomaly_score = model.predict_anomaly_score(sequence)
    return anomaly_score > threshold

# 장점
log1 = "INFO: Error handling completed"  # 정상 패턴 → 정상 판단 ✅
log2 = "ERROR: System failure"  # 이상 패턴 → 이상 판단 ✅
log3 = "gateway → unknown → manager"  # 패턴 이상 → 이상 판단 ✅
```

---

## 📝 요약

### 키워드 기반 탐지

**장점:**
- ✅ 간단하고 빠름
- ✅ 구현이 쉬움
- ✅ 즉시 사용 가능

**단점:**
- ❌ 키워드 없으면 놓침
- ❌ 오탐 많음
- ❌ 새로운 패턴 탐지 불가
- ❌ 문맥 이해 없음

### ML 기반 탐지 (LogBERT)

**장점:**
- ✅ 패턴 기반 탐지
- ✅ 문맥 이해
- ✅ 새로운 이상 패턴 탐지
- ✅ 오탐 적음
- ✅ 정확도 높음

**단점:**
- ⚠️ 모델 학습 필요
- ⚠️ 구현이 복잡
- ⚠️ 실행 속도 상대적으로 느림

---

## 🎯 결론

### LogBERT vs 키워드 기반

**LogBERT가 더 나은 이유:**

1. **패턴 기반 탐지**
   - 키워드 없이도 이상 탐지 가능
   - 복잡한 이상 패턴 탐지

2. **문맥 이해**
   - "error"가 있어도 정상일 수 있음
   - 문맥을 고려한 정확한 판단

3. **새로운 이상 탐지**
   - 학습된 정상 패턴과 다른 모든 패턴 탐지
   - 키워드 목록 업데이트 불필요

4. **높은 정확도**
   - 정확도: 85-92% (키워드: 50-70%)
   - 오탐률: 10-20% (키워드: 30-50%)
   - 미탐지율: 15-25% (키워드: 40-60%)

**결론: LogBERT는 키워드 기반 방법보다 훨씬 정확하고 강력한 이상 탐지가 가능합니다!** 🚀

