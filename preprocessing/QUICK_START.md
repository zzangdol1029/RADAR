# 전처리 빠른 시작 가이드

## 전제 조건

✅ 날짜별로 로그 파일 분리 완료 (`logs/date_split/` 디렉토리에 파일 존재)

## 전처리 실행 방법

### 방법 1: 기본 실행 (전체 날짜 처리)

```bash
cd preprocessing
conda activate radar
python log_preprocessor.py
```

**결과:**
- `logs/date_split/`의 모든 날짜별 파일 처리
- 각 날짜별로 `preprocessed_logs_YYYY-MM-DD.json` 생성

### 방법 2: 특정 날짜만 처리 (테스트용)

**1단계: 설정 파일 수정**
```yaml
# preprocessing_config.yaml
date_filter: "2025-08-22"  # 처리할 날짜 지정
```

**2단계: 실행**
```bash
cd preprocessing
python log_preprocessor.py
```

### 방법 3: 커스텀 디렉토리 지정

```bash
cd preprocessing
python log_preprocessor.py --log-dir ../logs/date_split --output preprocessed_logs.json
```

## 실행 전 확인사항

### 1. Conda 환경 활성화
```bash
conda activate radar
```

### 2. 설정 파일 확인
```bash
cat preprocessing_config.yaml
```

확인할 항목:
- `log_directory: "logs/date_split"` ✅
- `batch_by_date: true` ✅
- `stream_mode: true` ✅

### 3. 로그 파일 확인
```bash
ls -lh logs/date_split/*.log | head -5
```

## 실행 중 모니터링

### 진행 상황 확인

실행 중 다음과 같은 로그가 출력됩니다:

```
2025-12-29 18:12:58,532 - INFO - 로그 디렉토리 처리 시작: logs/date_split
2025-12-29 18:12:58,532 - INFO - 날짜별 배치 처리 모드 활성화
2025-12-29 18:12:58,533 - INFO - 날짜별 배치 처리 모드로 시작 (메모리 효율적)
2025-12-29 18:13:01,271 - INFO - 날짜별 처리 시작: 2025-08-22 (4개 파일)
2025-12-29 18:13:01,271 - INFO -   로그 수집 중: user_2025-08-22.log
...
2025-12-29 18:13:06,507 - INFO -   2025-08-22: 처리 완료
```

### 예상 소요 시간

- **날짜당**: 약 1-2시간
- **전체**: 날짜 수 × 1-2시간
- **예시**: 30일치 → 약 30-60시간 (1.5-2.5일)

## 출력 파일

### 위치
```
preprocessing/
├── preprocessed_logs_2025-02-25.json
├── preprocessed_logs_2025-02-26.json
├── preprocessed_logs_2025-02-27.json
└── ...
```

### 파일 구조
각 JSON 파일은 다음과 같은 세션 리스트를 포함합니다:

```json
[
  {
    "session_id": 0,
    "event_sequence": [1, 5, 1, 12, 3],
    "token_ids": [101, 1, 2, 1, 3, 4, 102, 0, 0, ...],
    "attention_mask": [1, 1, 1, 1, 1, 1, 1, 0, 0, ...],
    "has_error": false,
    "has_warn": true,
    "service_name": "manager",
    "time_window": "2025-08-22_14_05",
    "related_services": ["manager", "gateway", "eureka"],
    "simplified_text": "[manager] INFO [main] Starting | WARN ..."
  },
  ...
]
```

## 문제 해결

### 메모리 부족

```yaml
# preprocessing_config.yaml
enable_correlation: false  # 관계 추적 비활성화
window_size: 30  # 윈도우 크기 감소
```

### 특정 날짜만 처리

```yaml
# preprocessing_config.yaml
date_filter: "2025-08-22"  # 원하는 날짜
```

### 중단 후 재개

날짜별로 처리되므로:
- 이미 처리된 날짜는 스킵
- 중단된 날짜부터 다시 시작

## 다음 단계

전처리 완료 후:
1. LogBERT 모델 학습
2. RAG 시스템 구축
3. 장애 탐지 시스템 구축














