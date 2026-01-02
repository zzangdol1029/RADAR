# 점진적 학습 가이드

## 점진적 학습이란?

점진적 학습(Progressive Training)은 작은 데이터 비율(10%)부터 시작해서 단계적으로 데이터를 늘려가며 학습하는 방법입니다.

### 장점

1. **조기 검증**: 10% 데이터로도 학습 가능성 확인
2. **시간 절약**: 문제가 있으면 조기에 발견
3. **성능 추이 확인**: 데이터 양에 따른 성능 변화 관찰
4. **체크포인트 관리**: 각 단계별 모델 저장
5. **메모리 효율**: 각 단계 사이에 메모리 정리

## 폴더 구조

점진적 학습은 각 단계별로 별도 폴더를 생성합니다:

```
checkpoints_transfer/
├── logs/
│   └── progressive_training_20250101_120000.log  # 전체 점진적 학습 로그
├── progressive_training_results.json              # 각 단계 요약
├── progressive_training_summary.txt              # 텍스트 요약
├── progressive_training.pid                      # 프로세스 ID (백그라운드 실행 시)
│
├── stage_1_10pct/                                # 10% 단계
│   ├── checkpoints/
│   │   ├── best_model.pt                         # 최고 성능 모델
│   │   ├── epoch_1.pt
│   │   └── epoch_2.pt
│   ├── logs/
│   │   └── stage_1_10pct_20250101_120000.log     # 단계별 상세 로그
│   └── training_metrics.json                      # 학습 메트릭
│
├── stage_2_20pct/                                # 20% 단계
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── ...
│   ├── logs/
│   │   └── stage_2_20pct_20250101_123000.log
│   └── training_metrics.json
│
└── stage_10_100pct/                               # 100% 단계
    ├── checkpoints/
    ├── logs/
    └── training_metrics.json
```

## 사용 방법

### 1. 기본 사용 (10% → 100%)

```bash
python train_transfer.py --progressive
```

**기본 설정:**
- 시작: 10% 데이터
- 단계: 10%씩 증가
- 최대: 100% 데이터
- 단계당 에폭: 2

### 2. 커스텀 설정

```bash
python train_transfer.py \
  --progressive \
  --pretrained distilbert-base-uncased \
  --start-ratio 0.1 \
  --step-size 0.1 \
  --max-ratio 1.0 \
  --epochs-per-stage 2 \
  --max-memory-mb 4000
```

**파라미터:**
- `--pretrained`: Pre-trained 모델명
- `--start-ratio`: 시작 비율 (0.1 = 10%)
- `--step-size`: 단계 크기 (0.1 = 10%)
- `--max-ratio`: 최대 비율 (1.0 = 100%)
- `--epochs-per-stage`: 각 단계당 에폭 수
- `--max-memory-mb`: 최대 메모리 사용량 제한 (MB)

### 3. 백그라운드 실행 (권장)

```bash
./run_progressive_training.sh
```

또는 커스텀 설정:

```bash
./run_progressive_training.sh \
  bert-base-uncased \    # Pre-trained 모델
  0.1 \                  # 시작 비율
  0.1 \                  # 단계 크기
  1.0 \                  # 최대 비율
  2 \                    # 단계당 에폭
  4000                   # 최대 메모리 (MB, 선택사항)
```

## 학습 과정

### 단계별 진행

```
단계 1: 10% 데이터
  → stage_1_10pct/ 폴더 생성
  → 학습 → 체크포인트 저장
  → 메모리 정리

단계 2: 20% 데이터
  → stage_2_20pct/ 폴더 생성
  → 이전 체크포인트 로드 (stage_1_10pct/checkpoints/best_model.pt)
  → 학습 → 체크포인트 저장
  → 메모리 정리

단계 3: 30% 데이터
  → stage_3_30pct/ 폴더 생성
  → 이전 체크포인트 로드 (stage_2_20pct/checkpoints/best_model.pt)
  → 학습 → 체크포인트 저장
  → 메모리 정리

...

단계 10: 100% 데이터
  → stage_10_100pct/ 폴더 생성
  → 이전 체크포인트 로드
  → 학습 → 체크포인트 저장
```

## 결과 확인

### 1. 실시간 로그 확인

```bash
# 전체 점진적 학습 로그
tail -f checkpoints_transfer/logs/progressive_training_*.log

# 특정 단계 로그
tail -f checkpoints_transfer/stage_1_10pct/logs/*.log
```

### 2. 학습 진행 상황 확인

```bash
# 각 단계별 체크포인트 확인
ls -lh checkpoints_transfer/stage_*_*pct/checkpoints/

# 최신 단계 확인
ls -lt checkpoints_transfer/stage_*_*pct/ | head -5
```

### 3. 최종 결과 확인

```bash
# JSON 결과 파일
cat checkpoints_transfer/progressive_training_results.json | jq

# 텍스트 요약
cat checkpoints_transfer/progressive_training_summary.txt
```

### 4. 메트릭 확인

```bash
# 특정 단계 메트릭
cat checkpoints_transfer/stage_1_10pct/training_metrics.json | jq
```

## 백그라운드 실행

### 실행

```bash
./run_progressive_training.sh
```

### 모니터링

```bash
# 프로세스 확인
ps -p $(cat checkpoints_transfer/progressive_training.pid)

# 실시간 로그
tail -f checkpoints_transfer/logs/progressive_training_*.log

# 단계별 로그
tail -f checkpoints_transfer/stage_*_*pct/logs/*.log
```

### 중단

```bash
# 프로세스 종료
kill $(cat checkpoints_transfer/progressive_training.pid)

# 또는
pkill -f "train_transfer.py --progressive"
```

## 메모리 최적화

### 자동 최적화

점진적 학습은 자동으로 메모리를 최적화합니다:

1. **배치 크기 자동 조정**
   - 10% 데이터 → 배치 크기 1-2
   - 20% 데이터 → 배치 크기 2
   - 50% 데이터 → 배치 크기 2-4
   - 100% 데이터 → 배치 크기 4

2. **메모리 정리**
   - 각 단계 사이에 메모리 정리
   - 데이터셋과 모델 객체 삭제
   - Python 가비지 컬렉션

3. **리소스 모니터링**
   - 각 단계마다 메모리/CPU 사용량 로깅
   - 메모리 초과 시 경고

### 메모리 제한 설정

```bash
python train_transfer.py \
  --progressive \
  --max-memory-mb 4000
```

메모리가 4GB를 초과하면 경고 메시지가 출력됩니다.

## 예상 시간

### M4 Pro 환경

- 10% 단계: 약 30분
- 20% 단계: 약 1시간
- 30% 단계: 약 1.5시간
- ...
- 100% 단계: 약 5시간

**전체 점진적 학습**: 약 15-20시간 (10단계)

### DGX Station 환경

- 10% 단계: 약 10분
- 20% 단계: 약 20분
- 30% 단계: 약 30분
- ...
- 100% 단계: 약 1.5시간

**전체 점진적 학습**: 약 5-7시간 (10단계)

## 권장 워크플로우

### 1단계: 빠른 검증 (10%만)

```bash
python train_transfer.py \
  --progressive \
  --max-ratio 0.1 \
  --epochs-per-stage 2
```

**목적**: 학습이 가능한지 빠르게 확인

### 2단계: 점진적 학습 (10% → 50%)

```bash
python train_transfer.py \
  --progressive \
  --max-ratio 0.5 \
  --epochs-per-stage 2
```

**목적**: 성능 추이 확인, 최적 비율 찾기

### 3단계: 전체 학습 (50% → 100%)

```bash
python train_transfer.py \
  --progressive \
  --start-ratio 0.5 \
  --epochs-per-stage 3
```

**목적**: 최종 모델 학습

## 주의사항

1. **디스크 공간**: 각 단계마다 체크포인트가 저장되므로 충분한 공간 필요
2. **학습 시간**: 점진적 학습은 전체 시간이 더 걸릴 수 있음
3. **메모리**: 각 단계에서 데이터셋이 재생성되므로 메모리 사용량 확인
4. **중단 시**: 각 단계는 독립적으로 저장되므로 중단해도 이전 단계는 보존됨

## 문제 해결

### 메모리 부족

```bash
# 배치 크기 줄이기
python train_transfer.py --progressive --max-memory-mb 2000

# 또는 데이터 비율 낮추기
python train_transfer.py --progressive --max-ratio 0.5
```

### 학습 중단

```bash
# 마지막 완료된 단계 확인
ls -lt checkpoints_transfer/stage_*_*pct/ | head -1

# 해당 단계부터 재시작
python train_transfer.py \
  --progressive \
  --start-ratio 0.3  # 마지막 완료된 비율
```

## 결론

점진적 학습은 **학습 가능성을 조기에 확인**하고 **데이터 양에 따른 성능 변화를 관찰**하는 데 유용합니다.

**권장:**
- 처음 학습: 10% → 50% 점진적 학습으로 검증
- 검증 완료 후: 50% → 100% 점진적 학습으로 최종 모델 생성
- 백그라운드 실행: `./run_progressive_training.sh` 사용
