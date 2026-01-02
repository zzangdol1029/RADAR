# 데이터 전처리 파이프라인 상세 설명

## 전체 흐름도

```
원본 로그 파일 (.log)
    ↓
[1단계] 로그 정리 (Cleaning)
    ↓
[2단계] 로그 파싱 (Parsing) - Drain3
    ↓
[3단계] 세션화 (Sessionization)
    ↓
[4단계] 인코딩 및 메타데이터 결합
    ↓
전처리된 세션 데이터 (JSON)
```

---

## 1단계: 로그 정리 (Cleaning)

### 목적
의미 없는 데이터를 제거하여 파싱 효율을 높입니다.

### 처리 내용
- **Spring Boot 배너 제거**: 애플리케이션 시작 시 출력되는 ASCII 아트 제거
- **빈 줄 제거**: 공백만 있는 라인 제거
- **특수 패턴 제거**: 
  - `:: Spring Boot ::`
  - `:: File Module Info ::`
  - 구분선 (`---`)

### 코드 위치
```python
# LogCleaner.clean_log_line()
cleaned_line = self.cleaner.clean_log_line(line)
```

### 예시
```
입력: "  .   ____          _            __ _ _"
출력: None (제거됨)

입력: "2025-12-08 17:23:47.950 INFO  kr.re.bbp.Application --- [main] Starting..."
출력: "2025-12-08 17:23:47.950 INFO  kr.re.bbp.Application --- [main] Starting..."
```

---

## 2단계: 로그 파싱 (Parsing) - Drain3

### 목적
비정형 로그를 **템플릿(Template)**과 **파라미터(Parameter)**로 분리하여 구조화합니다.

### Drain3 알고리즘
- **트리 기반 파싱**: 로그 구조를 트리로 표현하여 유사한 로그를 클러스터링
- **동적 템플릿 추출**: 변수 부분을 `<*>`로 마스킹하여 공통 패턴 추출
- **파라미터 추출**: 마스킹된 부분의 실제 값 추출

### 처리 과정

#### Step 1: 템플릿 추출
```
원본: "2025-12-08 17:54:36 ERROR [manager] Connection timeout from 192.168.0.1"
      ↓
템플릿: "ERROR [manager] Connection timeout from <*>"
파라미터: ["192.168.0.1"]
```

#### Step 2: Event ID 할당
- 동일한 템플릿은 같은 Event ID를 받음
- 새로운 템플릿 발견 시 자동으로 다음 ID 할당

```
템플릿: "ERROR [manager] Connection timeout from <*>"
  → Event ID: 1

템플릿: "INFO [gateway] Request received from <*>"
  → Event ID: 2
```

### 코드 위치
```python
# LogParser.parse_log()
parsed_log = self.parser.parse_log(cleaned_line)
# 결과: {
#   'template': "ERROR [manager] Connection timeout from <*>",
#   'event_id': 1,
#   'parameters': ["192.168.0.1"],
#   'original': "원본 로그"
# }
```

### 효과
- **압축률**: 수백만 개의 로그 → 수십~수백 개의 이벤트 유형
- **학습 효율**: 모델이 패턴을 더 빠르게 학습

---

## 3단계: 세션화 (Sessionization)

### 목적
개별 로그를 **의미 있는 그룹(세션)**으로 묶어 시퀀스 데이터를 생성합니다.

### 세션화 방법

#### 방법 A: Sliding Window (현재 기본값)
- **크기 기반**: 로그 개수 기준 (기본: 50개)
- **시간 기반**: 시간 범위 기준 (기본: 180초 = 3분)
- **조건**: 두 조건 중 하나라도 만족하면 세션 완성

```
로그 1 (Event ID: 1)
로그 2 (Event ID: 5)
로그 3 (Event ID: 1)
...
로그 50 (Event ID: 12)
  ↓
세션 완성!
  ↓
[1, 5, 1, ..., 12]  (50개 이벤트 시퀀스)
```

#### 방법 B: Trace ID 기반
- MSA 환경에서 요청 추적 ID를 기준으로 그룹화
- 같은 Trace ID를 가진 로그들을 하나의 세션으로 묶음

### 처리 과정

```python
# Sessionizer.add_log()
completed_sessions = self.sessionizer.add_log(parsed_log)
```

### 세션 생성 조건
1. **크기 도달**: 윈도우에 50개 로그가 쌓임
2. **시간 초과**: 첫 로그부터 3분 경과
3. **파일 종료**: 로그 파일 끝에 도달 (flush)

### 결과
```python
session = [
    {'event_id': 1, 'template': '...', ...},
    {'event_id': 5, 'template': '...', ...},
    {'event_id': 1, 'template': '...', ...},
    ...
]
```

---

## 4단계: 인코딩 및 메타데이터 결합

### 4-1. 메타데이터 결합 (Metadata Enrichment)

#### 로그 레벨 태깅
```python
ERROR 로그 포함 → has_error = True
WARN 로그 포함 → has_warn = True
```

#### 서비스명 추출
```python
파일명: "manager.log" → service_name = "manager"
파일명: "gateway.log" → service_name = "gateway"
```

#### SQL 쿼리 간소화
```
원본 SQL:
  "select attachfile0_.atch_file_sn from bio.cs_atch_file_m 
   where attachfile0_.atch_file_group_sn=?"

간소화:
  "[SQL] SELECT bio.cs_atch_file_m"
```

#### RAG용 텍스트 생성
```python
simplified_text = "[manager] ERROR [manager] Connection timeout from <*> | INFO [main] Starting..."
```

### 4-2. 인코딩 및 토큰화 (Encoding)

#### Step 1: Event ID 시퀀스 추출
```python
event_sequence = [1, 5, 1, 12, 3, ...]
```

#### Step 2: Token ID 매핑
```python
Event ID 1 → Token ID 1
Event ID 5 → Token ID 2
Event ID 12 → Token ID 3
...
```

#### Step 3: Special Tokens 추가
```python
# BERT 스타일
[CLS] + [1, 2, 1, 3, ...] + [SEP]
  ↓
[101, 1, 2, 1, 3, ..., 102]
```

#### Step 4: Padding
```python
# max_seq_length = 256
[101, 1, 2, 1, 3, ..., 102, 0, 0, 0, ...]  # 길이 256으로 맞춤
```

#### Step 5: Attention Mask 생성
```python
attention_mask = [1, 1, 1, ..., 1, 0, 0, 0, ...]
# 실제 토큰: 1, 패딩: 0
```

### 코드 위치
```python
# MetadataEnricher.enrich_session()
enriched = self.metadata_enricher.enrich_session(session, service_name)

# LogEncoder.encode_sequence()
encoded = self.encoder.encode_sequence(session)
```

---

## 최종 출력 형식

### JSON 구조
```json
{
  "session_id": 0,
  "event_sequence": [1, 5, 1, 12, 3],
  "token_ids": [101, 1, 2, 1, 3, 4, 102, 0, 0, ...],
  "attention_mask": [1, 1, 1, 1, 1, 1, 1, 0, 0, ...],
  "has_error": false,
  "has_warn": true,
  "service_name": "manager",
  "original_logs": [
    "2025-12-08 17:23:47.950 INFO ...",
    "2025-12-08 17:23:50.947 WARN ...",
    ...
  ],
  "simplified_text": "[manager] INFO [main] Starting | WARN ..."
}
```

### 필드 설명

| 필드 | 설명 | 용도 |
|------|------|------|
| `session_id` | 세션 고유 ID | 식별자 |
| `event_sequence` | Event ID 시퀀스 | 분석용 |
| `token_ids` | BERT 입력 토큰 | LogBERT 학습 |
| `attention_mask` | 패딩 마스크 | LogBERT 학습 |
| `has_error` | ERROR 로그 포함 여부 | 우선순위 필터링 |
| `has_warn` | WARN 로그 포함 여부 | 우선순위 필터링 |
| `service_name` | 서비스명 | RAG 검색 범위 지정 |
| `original_logs` | 원본 로그 리스트 | 디버깅/분석 |
| `simplified_text` | 간소화된 텍스트 | RAG 벡터화 |

---

## 처리 예시: 전체 파이프라인

### 입력 (원본 로그)
```
2025-12-08 17:23:47.950 INFO  kr.re.bbp.Application --- [main] Starting Application
2025-12-08 17:23:50.947 WARN  org.hibernate.mapping.RootClass --- [main] HHH000038: Composite-id class does not override equals()
2025-12-08 17:23:55.796 DEBUG org.hibernate.SQL --- [scheduling-1] select attachfile0_.atch_file_sn from bio.cs_atch_file_m
2025-12-08 17:24:10.812 ERROR kr.re.bbp.manager.Service --- [worker-1] Connection timeout from 192.168.0.1
```

### 1단계: 정리
```
✓ INFO  kr.re.bbp.Application --- [main] Starting Application
✓ WARN  org.hibernate.mapping.RootClass --- [main] HHH000038: Composite-id class does not override equals()
✓ DEBUG org.hibernate.SQL --- [scheduling-1] select attachfile0_.atch_file_sn from bio.cs_atch_file_m
✓ ERROR kr.re.bbp.manager.Service --- [worker-1] Connection timeout from 192.168.0.1
```

### 2단계: 파싱
```
Event ID 1: "INFO  kr.re.bbp.Application --- [main] Starting Application"
Event ID 2: "WARN  org.hibernate.mapping.RootClass --- [main] HHH000038: Composite-id class does not override equals()"
Event ID 3: "[SQL] SELECT bio.cs_atch_file_m"
Event ID 4: "ERROR kr.re.bbp.manager.Service --- [worker-1] Connection timeout from <*>"
```

### 3단계: 세션화 (50개 또는 3분)
```
Session 1: [1, 2, 3, 4, ...]  (50개 이벤트)
```

### 4단계: 인코딩 및 메타데이터
```json
{
  "session_id": 0,
  "event_sequence": [1, 2, 3, 4, ...],
  "token_ids": [101, 1, 2, 3, 4, ..., 102, 0, 0, ...],
  "attention_mask": [1, 1, 1, 1, ..., 1, 0, 0, ...],
  "has_error": true,
  "has_warn": true,
  "service_name": "manager",
  "simplified_text": "[manager] INFO [main] Starting | WARN ... | [SQL] SELECT ... | ERROR Connection timeout"
}
```

---

## 성능 및 통계

### 처리 통계 예시
```
처리 완료: 176,497줄, 3,529개 세션 생성
발견된 고유 이벤트 수: 245개
```

### 압축률
- 원본 로그: 176,497줄
- 이벤트 유형: 245개
- 압축률: 약 99.86% (176,497 → 245)

### 세션 생성
- 평균 세션 크기: 50개 이벤트
- 생성된 세션 수: 3,529개
- 평균 로그/세션: 약 50개

---

## 설정 파라미터

### preprocessing_config.yaml
```yaml
window_size: 50          # 세션당 로그 개수
window_time: 180         # 세션 시간 범위 (초)
max_seq_length: 256      # 최대 시퀀스 길이
sessionization_method: "sliding_window"  # 세션화 방법
```

### drain3_config.yaml
```yaml
drain_sim_th: 0.4        # 유사도 임계값
drain_depth: 4           # 트리 깊이
drain_max_children: 100  # 최대 자식 노드 수
```

---

## 다음 단계

전처리된 데이터는 다음 용도로 사용됩니다:

1. **LogBERT 학습**: `token_ids`와 `attention_mask` 사용
2. **RAG 검색**: `simplified_text`를 벡터화하여 유사도 검색
3. **장애 탐지**: `has_error` 또는 `has_warn` 플래그로 우선순위 필터링
4. **패턴 분석**: `event_sequence`를 분석하여 이상 패턴 탐지

