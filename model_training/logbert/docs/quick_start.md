# LogBERT 학습 및 평가 빠른 시작 가이드

## 🚀 빠른 시작 (Quick Start)

### 1단계: 학습 (1개 파일로 빠른 테스트)

```bash
python scripts/train_intel.py --config configs/test_quick_xpu.yaml
```

**실행 결과:**
- 로그 파일: `logs/train_test_quick_xpu_20260201_213000.log`
- 체크포인트: `checkpoints_quick_xpu/checkpoints/best_model.pt`
- 예상 시간: 5-10분

### 2단계: 성능 평가

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints_quick_xpu/checkpoints/best_model.pt \
    --config configs/test_quick_xpu.yaml \
    --validation-data ../output/preprocessed_logs_000.json
```

**실행 결과:**
- 정확도, 정밀도, 재현율, F1-Score 출력
- 시각화: `evaluation_results/score_distribution.png`
- 결과 JSON: `evaluation_results/evaluation_results.json`

### 3단계: 결과 확인

```bash
# JSON 결과 확인
cat evaluation_results/evaluation_results.json

# 로그 확인
cat logs/evaluation_20260201_213000.log

# 그래프 확인 (VSCode에서)
code evaluation_results/score_distribution.png
code evaluation_results/confusion_matrix.png
```

## 📊 예상 결과

### 1개 파일 학습 후 예상 성능

```
================================================================================
📊 성능 평가 결과
================================================================================
정확도 (Accuracy):  0.8500 (85.00%)  ⚠️
정밀도 (Precision): 0.7800 (78.00%)  ⚠️
재현율 (Recall):    0.7200 (72.00%)  ⚠️
F1-Score:          0.7488 (74.88%)  ⚠️
ROC AUC:           0.8523
```

**평가:**
- ⚠️ 양호한 수준이지만 개선 여지 있음
- 1개 파일로는 충분한 학습 부족
- 더 많은 데이터로 재학습 권장

### 10개 파일 학습 후 예상 성능

```bash
# 10개 파일로 학습
python scripts/train_intel.py --config configs/test_xpu.yaml
```

**예상 성능:**
```
================================================================================
📊 성능 평가 결과
================================================================================
정확도 (Accuracy):  0.9200 (92.00%)  ✅
정밀도 (Precision): 0.8800 (88.00%)  ✅
재현율 (Recall):    0.8500 (85.00%)  ✅
F1-Score:          0.8648 (86.48%)  ✅
ROC AUC:           0.9234
```

**평가:**
- ✅ 우수한 성능
- 실용적 사용 가능
- 프로덕션 배포 고려 가능

### 전체 파일 학습 후 예상 성능 (324개)

```bash
# 전체 파일로 학습 (GPU 서버에서)
python scripts/train_cuda.py --config configs/full.yaml
```

**예상 성능:**
```
================================================================================
📊 성능 평가 결과
================================================================================
정확도 (Accuracy):  0.9600 (96.00%)  ⭐
정밀도 (Precision): 0.9300 (93.00%)  ⭐
재현율 (Recall):    0.9100 (91.00%)  ⭐
F1-Score:          0.9199 (91.99%)  ⭐
ROC AUC:           0.9734
```

**평가:**
- ⭐ 매우 우수한 성능
- 프로덕션 배포 권장
- 높은 신뢰도

## 🔄 학습 -> 평가 전체 워크플로우

### PC에서 빠른 테스트 (Intel GPU)

```bash
# 1. 빠른 학습 (1개 파일, 1 에폭)
python scripts/train_intel.py --config configs/test_quick_xpu.yaml

# 2. 평가
python scripts/evaluate.py \
    --checkpoint checkpoints_quick_xpu/checkpoints/best_model.pt \
    --config configs/test_quick_xpu.yaml \
    --validation-data ../output/preprocessed_logs_000.json

# 3. 결과 확인
cat evaluation_results/evaluation_results.json
```

### PC에서 충분한 학습 (Intel GPU)

```bash
# 1. 10개 파일로 학습 (5 에폭)
python scripts/train_intel.py --config configs/test_xpu.yaml

# 2. 여러 파일로 평가
python scripts/evaluate.py \
    --checkpoint checkpoints_xpu/checkpoints/best_model.pt \
    --config configs/test_xpu.yaml \
    --validation-data ../output/preprocessed_logs_010.json \
    --output-dir evaluation_results/test_xpu

# 3. 결과 비교
code evaluation_results/test_xpu/evaluation_results.json
```

### GPU 서버에서 전체 학습 (NVIDIA GPU)

```bash
# 1. 전체 데이터로 학습 (324개 파일, 10 에폭)
python scripts/train_cuda.py --config configs/full.yaml

# 2. 평가
python scripts/evaluate.py \
    --checkpoint checkpoints_full/checkpoints/best_model.pt \
    --config configs/full.yaml \
    --validation-data ../output/preprocessed_logs_100.json \
    --output-dir evaluation_results/full

# 3. 최종 결과 확인
cat evaluation_results/full/evaluation_results.json
```

## 📈 성능 개선 팁

### 성능이 낮은 경우 (< 80%)

**1. 더 많은 데이터로 학습**
```bash
# 1개 → 10개 파일
python scripts/train_intel.py --config configs/test_xpu.yaml
```

**2. 에폭 수 증가**
```yaml
# configs/test_xpu.yaml 수정
training:
  num_epochs: 5  # 1 → 5
```

**3. 전체 재학습**
```bash
python scripts/train_intel.py --config configs/full.yaml
```

### 오탐이 많은 경우 (Precision < 85%)

**평가 시 임계값 확인:**
```json
{
  "optimal_threshold": 0.3521  // 이 값을 높이면 오탐 감소
}
```

**재평가 (수동 임계값):**
- 평가 스크립트는 자동으로 최적 임계값을 찾습니다
- 필요 시 결과를 보고 임계값 조정

### 미탐이 많은 경우 (Recall < 85%)

**더 많은 이상 데이터 사용:**
```bash
python scripts/evaluate.py \
    --normal-ratio 0.7  # 0.8 → 0.7 (더 많은 이상 데이터)
```

## 🎯 권장 워크플로우

### 1단계: 빠른 검증 (5-10분)
```bash
# 1개 파일, 1 에폭
python scripts/train_intel.py --config configs/test_quick_xpu.yaml
python scripts/evaluate.py \
    --checkpoint checkpoints_quick_xpu/checkpoints/best_model.pt \
    --config configs/test_quick_xpu.yaml \
    --validation-data ../output/preprocessed_logs_000.json
```

**목적:** 코드가 정상 작동하는지 확인

### 2단계: 실험 (30분-1시간)
```bash
# 10개 파일, 5 에폭
python scripts/train_intel.py --config configs/test_xpu.yaml
python scripts/evaluate.py \
    --checkpoint checkpoints_xpu/checkpoints/best_model.pt \
    --config configs/test_xpu.yaml \
    --validation-data ../output/preprocessed_logs_010.json
```

**목적:** 실용적인 성능 확인, 하이퍼파라미터 튜닝

### 3단계: 최종 학습 (수 시간)
```bash
# 전체 파일, 10 에폭 (GPU 서버)
python scripts/train_cuda.py --config configs/full.yaml
python scripts/evaluate.py \
    --checkpoint checkpoints_full/checkpoints/best_model.pt \
    --config configs/full.yaml \
    --validation-data ../output/preprocessed_logs_100.json
```

**목적:** 최고 성능 모델 생성, 프로덕션 배포

## 📝 체크리스트

### 학습 전
- [ ] 전처리된 데이터 확인 (`../output/preprocessed_logs_*.json`)
- [ ] Intel GPU 또는 NVIDIA GPU 사용 가능 확인
- [ ] 설정 파일 확인 (`configs/*.yaml`)

### 학습 중
- [ ] 로그 파일 확인 (한글 정상 표시)
- [ ] Loss가 감소하는지 확인
- [ ] 체크포인트 저장 확인

### 평가 전
- [ ] 체크포인트 파일 존재 확인
- [ ] 검증 데이터 준비
- [ ] 출력 디렉토리 설정

### 평가 후
- [ ] 정확도 확인 (목표: 90% 이상)
- [ ] F1-Score 확인 (목표: 0.85 이상)
- [ ] 혼동 행렬 확인
- [ ] 시각화 확인

## 🎉 다음 단계

### 성능이 좋은 경우
1. 프로덕션 배포 준비
2. 실시간 이상 탐지 시스템 구축
3. API 개발

### 성능이 부족한 경우
1. 더 많은 데이터로 재학습
2. 하이퍼파라미터 튜닝
3. 모델 구조 개선

---

**핵심 요약:**
1. 학습: `python scripts/train_intel.py --config configs/test_quick_xpu.yaml`
2. 평가: `python scripts/evaluate.py --checkpoint ... --config ... --validation-data ...`
3. 결과: `evaluation_results/evaluation_results.json`

**이제 시작하세요!** 🚀
