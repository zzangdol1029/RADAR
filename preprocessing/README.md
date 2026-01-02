# 로그 전처리 파이프라인

LogBERT 및 RAG 시스템을 위한 로그 데이터 전처리 도구입니다.

## 주요 기능

### 1. 로그 파싱 (Log Parsing)
- **Drain3 알고리즘**을 사용하여 비정형 로그를 템플릿과 파라미터로 분리
- 수백만 개의 로그를 수십~수백 개의 이벤트 유형으로 압축

**예시:**
```
Raw: 2025-12-08 17:54:36 ERROR [manager] Connection timeout from 192.168.0.1
Template: ERROR [manager] Connection timeout from <*>
Event ID: 1
Parameters: [192.168.0.1]
```

### 2. 세션화 (Sessionization)
- **Trace ID 기반**: MSA 환경에서 요청 추적 ID를 기준으로 로그 그룹화
- **Sliding Window**: 시간(기본 5분) 또는 로그 개수(기본 20개) 단위로 그룹화

**결과:**
```
[Event_1, Event_5, Event_1, Event_12, Event_3]
```

### 3. 데이터 인코딩 및 토큰화
- Event ID를 정수 토큰으로 변환
- BERT 스타일 special tokens 추가 (`[CLS]`, `[SEP]`, `[MASK]`)
- Padding으로 시퀀스 길이 통일

### 4. 메타데이터 결합
- 로그 레벨 태깅 (ERROR, WARN 우선순위 부여)
- 서비스명 결합 (manager, research, gateway 등)
- SQL 쿼리 간소화 (SELECT, UPDATE 등 키워드와 테이블명만 추출)

## 빠른 시작

### 1. Conda 환경 설정 (권장)

#### 방법 1: 자동 스크립트 사용

**macOS/Linux:**
```bash
cd preprocessing
./setup_conda_env.sh
```

**Windows:**
```cmd
cd preprocessing
setup_conda_env.bat
```

#### 방법 2: 수동 생성

```bash
cd preprocessing
conda env create -f environment.yml
conda activate radar-preprocessing
```

#### 환경 활성화/비활성화

```bash
# 환경 활성화
conda activate radar-preprocessing

# 환경 비활성화
conda deactivate
```

### 2. pip 설치 (Conda 미사용 시)

```bash
cd preprocessing
pip install -r requirements.txt
```

### 2. 기본 사용

```bash
cd preprocessing
python log_preprocessor.py
```

기본 설정으로 상위 디렉토리의 `logs/` 폴더에 있는 모든 `.log` 파일을 처리하고 `preprocessed_logs.json`에 저장합니다.

### 3. 커스텀 설정

```bash
cd preprocessing
python log_preprocessor.py --log-dir ../logs --output output.json
```

### 4. 테스트 실행

```bash
cd preprocessing
python test_preprocessor.py
```

샘플 로그로 전처리 기능을 테스트합니다.

### 5. 예제 실행

```bash
cd preprocessing
python example_usage.py basic    # 기본 사용 예제
python example_usage.py custom   # 커스텀 설정 예제
python example_usage.py analyze  # 결과 분석 예제
```

### 6. 로그 파일 날짜별 분리 및 전처리 (권장)

각 로그 파일이 여러 날짜의 로그를 포함하는 경우, 먼저 날짜별로 분리한 후 전처리:

#### 방법 1: 자동 스크립트 사용 (권장)

```bash
cd preprocessing
./preprocess_split_logs.sh [로그디렉토리] [출력디렉토리] [전처리결과파일]

# 예시
./preprocess_split_logs.sh ../logs ../logs/date_split ../preprocessing/preprocessed_logs.json
```

#### 방법 2: 수동 실행

**1단계: 로그 파일을 날짜별로 분리**
```bash
cd preprocessing

# 전체 디렉토리 분리
python split_logs_by_date.py --input ../logs --output ../logs/date_split
```

**결과:**
- `user_250822_15:16:45.log` (여러 날짜 포함)
  → `user_2025-08-22.log`, `user_2025-08-23.log` 등
- `gateway_250822_10:30:00.log` (여러 날짜 포함)
  → `gateway_2025-08-22.log`, `gateway_2025-08-23.log` 등

**2단계: 분리된 파일로 전처리**
```bash
# preprocessing_config.yaml에서 log_directory를 date_split으로 변경하거나
python log_preprocessor.py --log-dir ../logs/date_split --output preprocessed_logs.json
```

**장점:**
- 날짜별로 정확하게 분리된 파일로 전처리
- 날짜 필터링이 더 정확함
- 메모리 사용량 감소

## 사용법

### 명령줄 인자

```bash
cd preprocessing
python log_preprocessor.py [옵션]

옵션:
  --config CONFIG      설정 파일 경로 (기본: preprocessing/preprocessing_config.yaml)
  --log-dir DIR        로그 디렉토리 경로 (기본: ../logs)
  --output OUTPUT      출력 파일 경로 (기본: preprocessing/preprocessed_logs.json)
```

### Python 코드에서 사용

#### 방법 1: preprocessing 폴더에서 직접 import

```python
import sys
from pathlib import Path

# preprocessing 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent / 'preprocessing'))

from log_preprocessor import LogPreprocessor, load_config

# 설정 로드
config = load_config()

# 전처리기 생성
preprocessor = LogPreprocessor(config)

# 로그 파일 처리
sessions = preprocessor.process_log_file("../logs/manager.log")

# 또는 전체 디렉토리 처리
sessions = preprocessor.process_log_directory("../logs")

# 결과 저장
preprocessor.save_results(sessions, "output.json")
```

#### 방법 2: 패키지로 import

```python
from preprocessing import LogPreprocessor, load_config

# 설정 로드
config = load_config()

# 전처리기 생성 및 실행
preprocessor = LogPreprocessor(config)
sessions = preprocessor.process_log_directory("../logs")
preprocessor.save_results(sessions, "output.json")
```

### 설정 파일 사용

`preprocessing_config.yaml` 파일을 수정하여 세부 설정을 변경할 수 있습니다:

```yaml
log_directory: "logs"
output_path: "preprocessed_logs.json"
sessionization_method: "sliding_window"  # 또는 "trace_id"
window_size: 20
window_time: 300  # 5분
max_seq_length: 512
```

## 출력 형식

전처리된 결과는 JSON 형식으로 저장되며, 각 세션은 다음과 같은 구조를 가집니다:

```json
{
  "session_id": 0,
  "event_sequence": [1, 5, 1, 12, 3],
  "token_ids": [101, 1, 2, 1, 3, 4, 102, 0, 0, ...],
  "attention_mask": [1, 1, 1, 1, 1, 1, 1, 0, 0, ...],
  "has_error": false,
  "has_warn": true,
  "service_name": "manager",
  "original_logs": ["2025-12-08 17:23:47.950 INFO ...", ...],
  "simplified_text": "[manager] INFO [main] Starting Application | WARN ..."
}
```

## 파이프라인 단계

1. **Ingestion**: 로그 파일 읽기
2. **Cleaning**: 배너, 공백, 특수문자 제거
3. **Parsing (Drain3)**: 템플릿 추출 및 Event ID 할당
4. **Grouping**: 세션화 (Trace ID 또는 Sliding Window)
5. **Encoding**: 토큰화 및 숫자 시퀀스 생성
6. **Enrichment**: 메타데이터 결합 및 SQL 쿼리 가공

## 주의사항

- **Drain3 설정**: `drain3_config.yaml`에서 파싱 정확도를 조정할 수 있습니다.
  - `drain_sim_th`: 낮을수록 더 엄격한 매칭 (기본 0.4)
  - `drain_depth`: 로그 구조에 맞게 조정 (기본 4)

- **메모리 사용량**: 대용량 로그 파일의 경우 메모리 사용량을 고려하세요.
  - 필요시 배치 처리로 개선 가능

- **Trace ID 추출**: 현재는 JSON 형식 로그와 일부 HTTP 헤더에서 추출합니다.
  - 커스텀 추출 로직이 필요하면 `Sessionizer.extract_trace_id()` 메서드를 수정하세요.

## 다음 단계

전처리된 데이터는 다음 용도로 사용할 수 있습니다:

1. **LogBERT 학습**: `token_ids`와 `attention_mask`를 사용하여 모델 학습
2. **RAG 검색**: `simplified_text`를 벡터화하여 유사도 검색
3. **장애 탐지**: `has_error` 또는 `has_warn` 플래그로 우선순위 필터링

