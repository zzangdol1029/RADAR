# 날짜별 분리 후 전처리 가이드

## 전체 프로세스

### 1단계: 로그 파일 날짜별 분리
```
logs/real_logs/user_250822_15:16:45.log (여러 날짜 포함)
  ↓
logs/date_split/user_2025-08-22.log
logs/date_split/user_2025-08-23.log
logs/date_split/gateway_2025-08-22.log
...
```

### 2단계: 분리된 파일로 전처리
```
logs/date_split/ (날짜별 분리된 파일들)
  ↓
전처리 파이프라인
  ↓
preprocessed_logs_2025-08-22.json
preprocessed_logs_2025-08-23.json
...
```

## 전처리 진행 방식

### 파일명 인식

전처리 파이프라인은 파일명에서 날짜를 자동으로 인식합니다:

1. **날짜별 분리된 파일** (`service_YYYY-MM-DD.log`)
   - 예: `user_2025-08-22.log`
   - 파일명에서 날짜를 직접 추출
   - 하나의 파일 = 하나의 날짜

2. **원본 파일** (`service_YYMMDD_HH:MM:SS.log`)
   - 예: `user_250822_15:16:45.log`
   - 파일명에서 날짜 추출 시도
   - 실패 시 로그 내용에서 날짜 추출

### 처리 흐름

#### 날짜별 배치 처리 모드 (batch_by_date: true)

```
1. logs/date_split/ 디렉토리 스캔
   ↓
2. 파일명에서 날짜 추출
   - user_2025-08-22.log → 2025-08-22
   - gateway_2025-08-22.log → 2025-08-22
   ↓
3. 날짜별로 파일 그룹화
   {
     "2025-08-22": [
       "user_2025-08-22.log",
       "gateway_2025-08-22.log",
       "manager_2025-08-22.log"
     ],
     "2025-08-23": [...]
   }
   ↓
4. 날짜별 순차 처리
   - 2025-08-22 처리
     - 모든 서비스 로그 수집
     - 세션 생성 및 저장
     - 메모리 해제
   - 2025-08-23 처리
     - ...
   ↓
5. 날짜별 출력 파일 생성
   - preprocessed_logs_2025-08-22.json
   - preprocessed_logs_2025-08-23.json
```

## 실행 방법

### 방법 1: 자동 스크립트 (권장)

```bash
cd preprocessing
./preprocess_split_logs.sh
```

이 스크립트는:
1. `logs/real_logs/` → `logs/date_split/` 분리
2. `logs/date_split/` → 전처리 실행

### 방법 2: 수동 실행

#### 1단계: 날짜별 분리

```bash
cd preprocessing
python split_logs_by_date.py --input ../logs/real_logs --output ../logs/date_split
```

#### 2단계: 전처리

```bash
# 기본 실행 (전체 날짜)
python log_preprocessor.py --log-dir ../logs/date_split

# 특정 날짜만 처리
python log_preprocessor.py --log-dir ../logs/date_split --config preprocessing_config.yaml
# preprocessing_config.yaml에서 date_filter: "2025-08-22" 설정
```

## 설정 파일

### preprocessing_config.yaml

```yaml
# 로그 디렉토리 (날짜별 분리된 파일 위치)
log_directory: "logs/date_split"

# 날짜별 배치 처리
batch_by_date: true

# 날짜 필터 (선택사항)
date_filter: null  # 전체 처리
# date_filter: "2025-08-22"  # 특정 날짜만

# 관계 추적 (같은 날짜의 다른 서비스 연결)
enable_correlation: true

# 스트리밍 모드 (메모리 효율적)
stream_mode: true
```

## 처리 예시

### 입력 파일 구조

```
logs/date_split/
├── user_2025-08-22.log
├── user_2025-08-23.log
├── gateway_2025-08-22.log
├── gateway_2025-08-23.log
├── manager_2025-08-22.log
└── manager_2025-08-23.log
```

### 처리 과정

#### 날짜: 2025-08-22

1. **파일 수집**
   - `user_2025-08-22.log`
   - `gateway_2025-08-22.log`
   - `manager_2025-08-22.log`

2. **로그 파싱**
   - 각 파일의 로그를 Drain3로 파싱
   - Event ID 할당

3. **세션화**
   - 시간 윈도우별로 그룹화 (5분 단위)
   - 같은 시간 윈도우의 서비스들을 연결

4. **인코딩 및 저장**
   - BERT 토큰화
   - `preprocessed_logs_2025-08-22.json` 저장

5. **메모리 해제**
   - 해당 날짜 데이터 삭제

#### 날짜: 2025-08-23

동일한 과정 반복 → `preprocessed_logs_2025-08-23.json`

## 장점

### 1. 정확한 날짜 필터링
- 파일명에 날짜가 명확히 표시됨
- 날짜 추출 오류 최소화

### 2. 메모리 효율성
- 날짜별로 처리하여 메모리 사용량 최소화
- 날짜 처리 완료 후 즉시 해제

### 3. 병렬 처리 가능
- 날짜별로 독립적 처리
- 여러 날짜를 동시에 처리 가능

### 4. 중단/재개 용이
- 날짜별로 완료되므로 중단 후 재개 가능
- 이미 처리된 날짜는 스킵 가능

## 출력 파일

### 날짜별 출력

```
preprocessing/
├── preprocessed_logs_2025-08-22.json
├── preprocessed_logs_2025-08-23.json
├── preprocessed_logs_2025-08-24.json
└── ...
```

### 파일 병합 (선택사항)

```python
import json
from pathlib import Path

output_dir = Path('preprocessing')
date_files = sorted(output_dir.glob('preprocessed_logs_*.json'))

all_sessions = []
for date_file in date_files:
    with open(date_file, 'r', encoding='utf-8') as f:
        sessions = json.load(f)
        all_sessions.extend(sessions)

with open('preprocessed_logs_merged.json', 'w', encoding='utf-8') as f:
    json.dump(all_sessions, f, ensure_ascii=False, indent=2)
```

## 성능

### 메모리 사용량
- 날짜당: 약 1-2GB
- 전체: 날짜 수 × 1-2GB (하지만 날짜별로 해제)

### 처리 시간
- 날짜당: 약 1-2시간
- 전체: 날짜 수 × 1-2시간

### 예시
- 30일치 로그
- 날짜당 1.5시간
- 전체: 약 45시간 (약 2일)

## 문제 해결

### 파일명 인식 실패

파일명이 `service_YYYY-MM-DD.log` 형식이 아니면:
- 로그 내용에서 날짜 추출 시도
- 첫 100줄만 확인하여 성능 최적화

### 날짜 필터링이 안 될 때

```yaml
# preprocessing_config.yaml
date_filter: "2025-08-22"  # 정확한 형식: YYYY-MM-DD
```

### 메모리 부족

```yaml
# preprocessing_config.yaml
enable_correlation: false  # 관계 추적 비활성화
window_size: 30  # 윈도우 크기 감소
```

