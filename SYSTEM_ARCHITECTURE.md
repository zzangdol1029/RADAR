# MSA 로그 이상 탐지 및 RAG 시스템 아키텍처

## 📋 시스템 개요

MSA 환경의 로그를 분석하여 이상을 탐지하고, 치명도를 계산하며, 소스 코드 기반 RAG 가이드를 제공하는 통합 시스템입니다.

### 목표
1. **각 서비스별 이상 탐지**: Gateway, Eureka, User, Research, Manager, Code 등
2. **앙상블 이상 탐지**: LogBERT, DeepLog, LogLSTM, LogTCN 모델 통합
3. **치명도 계산**: 이상의 심각도 평가
4. **소스 코드 기반 RAG 가이드**: 소스 코드를 참조한 오류 해결 가이드 제공

### 팀 구성 (3명)
- **멤버 1**: 로그 수집, 통합 및 전처리 전담
- **멤버 2**: LogBERT 학습, 앙상블 탐지 및 치명도 계산
- **멤버 3**: DeepLog/LogLSTM/LogTCN 학습, 소스 코드 RAG 구축 및 API 서버

---

## 🏗️ 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    로그 수집 레이어                            │
│  Gateway │ Eureka │ User │ Research │ Manager │ Code        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│          전처리 파이프라인 (Preprocessing) - 멤버 1          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ 로그 수집 │→│ 로그 정리 │→│ 로그 파싱 │→│ 복합 키   │    │
│  │ Collection│  │ Cleaning │  │ Parsing  │  │ 생성     │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ 하이브리드│→│ 메타데이터│→│ 인코딩    │→│ 서비스간  │    │
│  │ 세션화    │  │ 추출     │  │ 토큰화    │  │ 연결     │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│                                                              │
│  [배치 전처리] 학습용 데이터 생성                             │
│  [실시간 전처리] 운영용 스트리밍 처리                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
┌───────────────────────┐      ┌───────────────────────┐
│   이상 탐지 모듈        │      │   소스 코드 RAG 모듈    │
│  (Anomaly Detection)  │      │   (Source Code RAG)   │
│  - 멤버 2 담당         │      │   - 멤버 3 담당        │
│                       │      │                       │
│  ┌─────────────────┐  │      │  ┌─────────────────┐ │
│  │ LogBERT 모델    │  │      │  │ 소스 코드 파싱  │ │
│  │ (멤버 2 학습)   │  │      │  │ 코드 청크 분할  │ │
│  └─────────────────┘  │      │  └─────────────────┘ │
│           ↓            │      │           ↓          │
│  ┌─────────────────┐  │      │  ┌─────────────────┐ │
│  │ 앙상블 탐지      │  │      │  │ 코드 벡터화      │ │
│  │ (LogBERT+       │  │      │  │ (CodeBERT 등)    │ │
│  │  DeepLog+       │  │      │  └─────────────────┘ │
│  │  LogLSTM+       │  │      │           ↓          │
│  │  LogTCN)        │  │      │  ┌─────────────────┐ │
│  └─────────────────┘  │      │  │ 벡터 DB 구축    │ │
│           ↓            │      │  │ (로그+코드)     │ │
│  ┌─────────────────┐  │      │  └─────────────────┘ │
│  │ 치명도 계산      │  │      │           ↓          │
│  │ Severity Score  │  │      │  ┌─────────────────┐ │
│  │ (멤버 2 구현)   │  │      │  │ RAG 검색 모듈    │ │
│  └─────────────────┘  │      │  │ 가이드 생성     │ │
└───────────────────────┘      └───────────────────────┘
        ↓                                   ↓
┌─────────────────────────────────────────────────────────────┐
│                    결과 통합 및 저장                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 이상 결과 │  │ 치명도   │  │ 벡터 DB   │  │ 메타데이터│  │
│  │ DB       │  │ DB       │  │ (로그+코드)│  │ DB       │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│          소스 코드 기반 RAG 가이드 시스템 - 멤버 3           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 쿼리 처리 │→│ 벡터 검색 │→│ 코드 검색 │→│ 가이드 생성│  │
│  │ Query    │  │ Retrieval │  │ Code     │  │ Generation│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              API 서버 (FastAPI) - 멤버 3 구현                │
│  POST /api/v1/detect     - 앙상블 이상 탐지                  │
│  GET  /api/v1/severity  - 치명도 조회 (멤버 2 모듈 사용)     │
│  POST /api/v1/guide     - 소스 코드 기반 가이드 조회        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 주요 컴포넌트

### 1. 전처리 파이프라인 (Preprocessing Pipeline) - 멤버 1 담당

**역할:**
- 로그 수집 및 통합
- 로그 파일 읽기 및 정리
- Drain3 기반 파싱
- 하이브리드 복합 키 기반 세션화
- MSA 서비스 간 연결 (IP + URL + 시간 매칭)
- 배치 전처리 (학습용)
- 실시간 전처리 (운영용)

**⚠️ 현재 상황:**
- Trace ID가 로그에 없으므로 하이브리드 복합 키 방식 사용
- Gateway: `client_ip` + `시간(초)` + `url` 기반 복합 키
- Manager: `스레드명` + `시간(초)` 기반 복합 키
- 후처리 연결: IP + URL + 시간 매칭

**입력:**
- 원본 로그 파일들 (서비스별 + 날짜별)

**출력:**
- 배치 전처리된 세션 데이터 (JSON, 학습용)
- 실시간 전처리 스트림 (운영용)
- 서비스 간 연결 정보 (`related_sessions`)

**구현 파일:**
- `preprocessing/log_collector.py` (신규) ⭐
- `preprocessing/batch_preprocessor.py` (배치 전처리)
- `preprocessing/streaming_preprocessor.py` (실시간 전처리) ⭐
- `preprocessing/log_preprocessor.py` (기존 확장)
- `preprocessing/split_logs_by_date.py` (날짜별 분리)

---

### 2. 이상 탐지 모듈 (Anomaly Detection Module) - 멤버 2 담당

**역할:**
- LogBERT 모델 학습 및 추론
- 앙상블 이상 탐지 (LogBERT + DeepLog + LogLSTM + LogTCN)
- 서비스별 이상 탐지
- 세션별 이상 탐지
- 앙상블 점수 통합

**입력:**
- 전처리된 세션 데이터 (멤버 1이 생성)
- 학습된 모델들:
  - LogBERT (멤버 2 학습)
  - DeepLog, LogLSTM, LogTCN (멤버 3 학습)

**출력:**
- 이상 점수 (Anomaly Score)
- 앙상블 이상 점수
- 이상 여부 (Is Anomaly)
- 서비스별 통계

**구현 파일:**
- `logbert_training/train.py` (LogBERT 학습)
- `logbert_training/detect_anomalies.py` (기존)
- `anomaly_detection/ensemble_detector.py` (앙상블 탐지) ⭐
- `anomaly_detection/service_detector.py` (서비스별 탐지)
- `config/ensemble_config.yaml` (앙상블 설정)

---

### 3. 치명도 계산 모듈 (Severity Calculation Module) - 멤버 2 담당

**역할:**
- 이상의 치명도 계산
- 앙상블 점수와 통합
- 서비스 중요도 반영
- 에러 유형별 가중치 적용

**입력:**
- 앙상블 이상 점수
- 에러 정보
- 서비스 정보
- MSA 연결 정보 (`related_sessions`)

**출력:**
- 치명도 점수 (Severity Score)
- 치명도 등급 (Critical, High, Medium, Low)
- 서비스별 치명도

**구현 파일:**
- `severity/severity_calculator.py` (신규)
- `severity/service_weights.py` (신규)
- `severity/error_classifier.py` (신규)

---

### 4. 소스 코드 RAG 구축 모듈 (Source Code RAG Builder) - 멤버 3 담당

**역할:**
- 소스 코드 파싱 및 청크 분할
- 코드 벡터화 (CodeBERT, StarCoder 등)
- 로그 데이터 벡터화
- 벡터 DB에 저장 (로그 + 소스 코드)

**입력:**
- 전처리된 세션 데이터 (멤버 1이 생성)
- 소스 코드 파일들 (Java, Python 등)
- 이상 탐지 결과
- 치명도 정보

**출력:**
- 코드 벡터 임베딩
- 로그 벡터 임베딩
- 메타데이터
- 벡터 DB 인덱스 (로그 + 소스 코드 통합)

**구현 파일:**
- `rag/code_parser.py` (소스 코드 파싱) ⭐
- `rag/code_vectorizer.py` (코드 벡터화) ⭐
- `rag/vectorizer.py` (로그 벡터화)
- `rag/metadata_extractor.py` (메타데이터 추출)
- `rag/db_builder.py` (벡터 DB 구축)

---

### 5. 소스 코드 기반 RAG 가이드 시스템 (Source Code RAG Guide System) - 멤버 3 담당

**역할:**
- 사용자 쿼리 처리
- 벡터 DB에서 유사도 검색 (로그 + 소스 코드)
- 관련 소스 코드 검색
- 소스 코드를 참조한 가이드 생성
- 코드 위치 및 수정 방법 제시

**입력:**
- 사용자 쿼리 (에러 메시지, 세션 ID 등)
- 벡터 DB (로그 + 소스 코드)

**출력:**
- 관련 소스 코드 청크
- 코드 위치 정보 (파일 경로, 라인 번호)
- 해결 방법 제안 (코드 수정 방법 포함)
- 유사 사례

**구현 파일:**
- `rag/code_retriever.py` (코드 검색) ⭐
- `rag/retriever.py` (로그 검색)
- `rag/code_guide_generator.py` (코드 기반 가이드 생성) ⭐
- `rag/llm_integration.py` (LLM 통합)

---

### 6. API 서버 (API Server) - 멤버 3 담당

**역할:**
- RESTful API 제공
- 앙상블 이상 탐지 요청 처리 (멤버 2 모듈 사용)
- 치명도 조회 (멤버 2 모듈 사용)
- 소스 코드 기반 가이드 조회

**구현 파일:**
- `api/main.py` (FastAPI 앱) ⭐
- `api/routes/detection.py` (앙상블 이상 탐지 API)
- `api/routes/severity.py` (치명도 조회 API - 멤버 2와 협업)
- `api/routes/guide.py` (소스 코드 기반 가이드 API) ⭐
- `api/models/schemas.py` (Pydantic 모델)

---

## 📊 데이터 흐름

### 1. 로그 수집 → 전처리 (멤버 1)

**배치 전처리 (학습용):**
```
원본 로그 파일들 (서비스별 + 날짜별)
    ↓
로그 수집 및 통합
    ↓
로그 정리 (Cleaning)
    ↓
로그 파싱 (Drain3)
    ↓
서비스별 복합 키 생성
    ├── Gateway: client_ip + 시간(초) + url
    └── Manager: 스레드명 + 시간(초)
    ↓
하이브리드 세션화 (복합 키 + Sliding Window)
    ↓
메타데이터 추출 및 저장
    ↓
인코딩 및 토큰화
    ↓
후처리 연결 (IP + URL + 시간 매칭)
    ↓
전처리된 세션 데이터 (JSON, 학습용)
```

**실시간 전처리 (운영용):**
```
로그 스트림 (tail -f)
    ↓
실시간 로그 정리 및 파싱
    ↓
복합 키 생성
    ↓
실시간 세션화 (윈도우 기반)
    ↓
즉시 이상 탐지로 전달
```

### 2. 전처리 → 앙상블 이상 탐지 (멤버 2)

```
전처리된 세션 데이터 (멤버 1이 생성)
    ↓
        ┌───────────┬───────────┬───────────┬───────────┐
        ↓           ↓           ↓           ↓
    LogBERT      DeepLog     LogLSTM     LogTCN
    (멤버 2)     (멤버 3)    (멤버 3)    (멤버 3)
        ↓           ↓           ↓           ↓
        └───────────┴───────────┴───────────┘
                    ↓
            앙상블 점수 통합
            (가중 평균/평균/최대값)
                    ↓
            임계값 비교
                    ↓
            이상 여부 판단
                    ↓
            앙상블 이상 탐지 결과
```

### 3. 이상 탐지 → 치명도 계산

```
이상 탐지 결과
    ↓
에러 정보 추출
    ↓
서비스 중요도 조회
    ↓
에러 유형 분류
    ↓
치명도 점수 계산
    ↓
치명도 등급 할당
    ↓
치명도 결과
```

### 4. 전처리 + 소스 코드 → RAG 구축 (멤버 3)

```
전처리된 세션 데이터 (멤버 1)
    ↓
소스 코드 파일들 (Java, Python 등)
    ↓
        ┌───────────────────┬───────────────────┐
        ↓                   ↓
    로그 벡터화          소스 코드 파싱
    (Sentence-BERT)     (함수/클래스 단위)
        ↓                   ↓
    로그 임베딩          코드 벡터화
                        (CodeBERT/StarCoder)
        ↓                   ↓
        └───────────────────┘
                    ↓
            벡터 DB에 저장
            (로그 + 소스 코드 통합)
                    ↓
            RAG 인덱스 완성
```

### 5. RAG → 소스 코드 기반 가이드 생성 (멤버 3)

```
사용자 쿼리 (에러 메시지 등)
    ↓
쿼리 벡터화
    ↓
벡터 DB에서 유사도 검색
    ├── 관련 로그 검색
    └── 관련 소스 코드 검색
    ↓
상위 K개 문서 검색
    ├── 로그 세션
    └── 코드 청크 (파일 경로, 라인 번호 포함)
    ↓
LLM으로 가이드 생성
    ├── 소스 코드 참조
    ├── 코드 위치 정보
    └── 수정 방법 제시
    ↓
소스 코드 기반 가이드 문서 반환
```

---

## 🗄️ 데이터베이스 설계

### 1. 이상 탐지 결과 DB

**테이블: `anomaly_detections`**
```sql
CREATE TABLE anomaly_detections (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    composite_key VARCHAR(500),  -- 복합 키 (Trace ID 대신)
    service_name VARCHAR(50),
    anomaly_score FLOAT,          -- 앙상블 점수
    ensemble_scores JSONB,        -- 각 모델별 점수
    is_anomaly BOOLEAN,
    threshold FLOAT,
    detected_at TIMESTAMP,
    related_sessions JSONB,        -- MSA 연결 정보
    metadata JSONB
);
```

### 2. 치명도 DB

**테이블: `severity_scores`**
```sql
CREATE TABLE severity_scores (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    composite_key VARCHAR(500),  -- 복합 키
    service_name VARCHAR(50),
    ensemble_anomaly_score FLOAT, -- 앙상블 이상 점수
    severity_score FLOAT,
    severity_level VARCHAR(20), -- CRITICAL, HIGH, MEDIUM, LOW
    error_count INTEGER,
    warning_count INTEGER,
    affected_services TEXT[],    -- 영향받은 서비스 목록
    service_weight FLOAT,
    calculated_at TIMESTAMP,
    metadata JSONB
);
```

### 3. RAG 메타데이터 DB

**테이블: `rag_metadata`**
```sql
CREATE TABLE rag_metadata (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    composite_key VARCHAR(500),  -- 복합 키
    service_name VARCHAR(50),
    content_type VARCHAR(20),     -- 'log' 또는 'code'
    text_content TEXT,
    vector_id VARCHAR(255),       -- 벡터 DB의 ID
    metadata JSONB,
    created_at TIMESTAMP
);
```

### 4. 벡터 DB (Chroma/Pinecone)

**컬렉션: `log_and_code_guides`**
- 벡터: 임베딩 벡터 (768차원 또는 1536차원)
- 메타데이터 (로그):
  - session_id
  - composite_key
  - service_name
  - error_type
  - severity_level
  - rag_text
- 메타데이터 (소스 코드):
  - file_path
  - service_name
  - chunk_type (function/class)
  - chunk_name
  - line_start
  - line_end
  - code_snippet

---

## 🔄 API 엔드포인트

### 1. 앙상블 이상 탐지 API

**POST `/api/v1/detect`**
```json
{
  "log_file_path": "/path/to/logs/gateway.log",
  "service_name": "gateway",
  "threshold": 0.5,
  "ensemble_method": "weighted_average"
}
```

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "gateway_12345",
      "composite_key": "gateway_192.168.0.18_12:49:46_/user/api/moduleMng/getModule",
      "anomaly_score": 0.85,
      "ensemble_scores": {
        "logbert": 0.82,
        "deeplog": 0.88,
        "lstm": 0.80,
        "tcn": 0.85
      },
      "is_anomaly": true,
      "service_name": "gateway",
      "related_sessions": [
        {
          "service": "manager",
          "session_id": "manager_67890",
          "match_score": 0.95
        }
      ]
    }
  ],
  "statistics": {
    "total_sessions": 100,
    "anomaly_sessions": 5,
    "anomaly_rate": 5.0
  }
}
```

### 2. 치명도 조회 API

**GET `/api/v1/severity/{session_id}`**
```json
{
  "session_id": "gateway_12345",
  "composite_key": "gateway_192.168.0.18_12:49:46_/user/api/moduleMng/getModule",
  "severity_score": 0.92,
  "severity_level": "CRITICAL",
  "ensemble_anomaly_score": 0.85,
  "services": [
    {
      "service_name": "gateway",
      "severity_score": 0.85,
      "error_count": 2,
      "affected_services": ["gateway", "manager"]
    }
  ]
}
```

### 3. 소스 코드 기반 가이드 조회 API

**POST `/api/v1/guide`**
```json
{
  "query": "Connection timeout error in gateway service",
  "session_id": "gateway_12345",
  "service_name": "gateway"
}
```

**Response:**
```json
{
  "guides": [
    {
      "title": "Connection Timeout 해결 방법",
      "severity": "HIGH",
      "related_code": [
        {
          "file_path": "gateway/src/main/java/com/example/GatewayController.java",
          "method": "handleRequest",
          "line_start": 45,
          "line_end": 78,
          "code_snippet": "public ResponseEntity<?> handleRequest(...) { ... }",
          "explanation": "이 메서드에서 타임아웃 설정을 확인하세요"
        }
      ],
      "steps": [
        "1. GatewayController.java의 handleRequest 메서드 확인 (45-78줄)",
        "2. 타임아웃 설정 값 확인 및 조정",
        "3. 로드 밸런서 상태 확인"
      ],
      "code_fix_suggestion": "timeout 설정을 5000ms에서 10000ms로 증가",
      "similar_cases": [
        {
          "session_id": "gateway_xyz789",
          "solution": "..."
        }
      ]
    }
  ]
}
```

---

## 📁 프로젝트 구조

```
RADAR-1/
├── preprocessing/                    # 전처리 모듈 (멤버 1)
│   ├── log_collector.py            # 로그 수집 ⭐
│   ├── batch_preprocessor.py        # 배치 전처리 (학습용)
│   ├── streaming_preprocessor.py   # 실시간 전처리 ⭐
│   ├── streaming_sessionizer.py    # 실시간 세션화 ⭐
│   ├── log_preprocessor.py         # 기존 전처리 (확장)
│   ├── split_logs_by_date.py       # 날짜별 분리
│   └── ...
│
├── logbert_training/                # 모델 학습 및 추론
│   ├── train.py                    # LogBERT 학습 (멤버 2)
│   ├── train_deeplog.py            # DeepLog 학습 (멤버 3)
│   ├── train_lstm.py               # LogLSTM 학습 (멤버 3)
│   ├── train_tcn.py                # LogTCN 학습 (멤버 3)
│   ├── detect_anomalies.py         # 이상 탐지 (기존)
│   ├── model.py                    # LogBERT 모델
│   ├── model_deeplog.py            # DeepLog 모델
│   ├── model_lstm.py               # LogLSTM 모델
│   ├── model_tcn.py                # LogTCN 모델
│   └── ...
│
├── anomaly_detection/               # 이상 탐지 모듈 (멤버 2)
│   ├── ensemble_detector.py        # 앙상블 탐지 ⭐
│   ├── service_detector.py         # 서비스별 탐지
│   └── __init__.py
│
├── severity/                        # 치명도 계산 모듈 (멤버 2)
│   ├── severity_calculator.py      # 치명도 계산
│   ├── service_weights.py          # 서비스 가중치
│   ├── error_classifier.py         # 에러 분류
│   └── __init__.py
│
├── rag/                            # RAG 모듈 (멤버 3)
│   ├── code_parser.py              # 소스 코드 파싱 ⭐
│   ├── code_vectorizer.py          # 코드 벡터화 ⭐
│   ├── vectorizer.py               # 로그 벡터화
│   ├── metadata_extractor.py       # 메타데이터 추출
│   ├── db_builder.py               # 벡터 DB 구축
│   ├── code_retriever.py           # 코드 검색 ⭐
│   ├── retriever.py                # 로그 검색
│   ├── code_guide_generator.py     # 코드 기반 가이드 생성 ⭐
│   ├── llm_integration.py          # LLM 통합
│   └── __init__.py
│
├── api/                            # API 서버 (멤버 3)
│   ├── main.py                    # FastAPI 앱 ⭐
│   ├── routes/
│   │   ├── detection.py           # 앙상블 이상 탐지 API
│   │   ├── severity.py            # 치명도 API (멤버 2 모듈 사용)
│   │   └── guide.py               # 소스 코드 기반 가이드 API ⭐
│   └── models/
│       └── schemas.py             # Pydantic 모델
│
├── database/                       # 데이터베이스
│   ├── migrations/                 # DB 마이그레이션
│   └── models.py                   # SQLAlchemy 모델
│
└── config/                         # 설정 파일
    ├── ensemble_config.yaml        # 앙상블 설정
    ├── preprocessing_config.yaml   # 전처리 설정
    ├── service_weights.yaml        # 서비스 가중치
    └── rag_config.yaml            # RAG 설정
```

---

## 🔧 기술 스택

### 백엔드
- **Python 3.10+**
- **FastAPI**: API 서버
- **SQLAlchemy**: ORM
- **PostgreSQL**: 관계형 DB
- **Chroma/Pinecone**: 벡터 DB

### 머신러닝
- **PyTorch**: 모델 학습 및 추론
- **Transformers**: BERT 모델
- **Sentence Transformers**: 벡터 임베딩

### LLM 통합
- **OpenAI API**: GPT-4/GPT-3.5
- **LangChain**: RAG 파이프라인
- **LlamaIndex**: 벡터 검색

### 전처리
- **Drain3**: 로그 파싱
- **Pandas**: 데이터 처리
- **NumPy**: 수치 연산

---

## 📝 구현 일정 및 역할 분담

### Phase 1: 로그 수집, 통합 및 전처리 (3-4주) - 멤버 1

**주요 작업:**
- 로그 수집 모듈 구현 ⭐
- 로그 통합 모듈 구현 ⭐
- 배치 전처리 파이프라인 (학습용)
- 실시간 전처리 파이프라인 ⭐ (운영용)
- 배치 전처리 데이터 생성 및 공유 ⭐

**산출물:**
- 전처리된 세션 데이터 (JSON, 학습용)
- 실시간 전처리 파이프라인 (운영용)

---

### Phase 2: 병렬 모델 학습 (2-3주) ⭐⭐⭐

**멤버 2:**
- LogBERT 모델 학습

**멤버 3:**
- DeepLog 모델 학습 (동시에)
- LogLSTM/LogTCN 모델 학습 (선택)

**장점:**
- 전처리 데이터만 공유하면 동시에 학습 가능
- 학습 시간 단축 (순차: 4-6주 → 병렬: 2-3주)

---

### Phase 3: 모듈 구현 (3-4주) - 병렬 작업

**멤버 2:**
- 앙상블 이상 탐지 모듈 구현
- 치명도 계산 모듈 구현

**멤버 3:**
- 소스 코드 파싱 및 벡터화 모듈 구현
- 벡터 DB 구축
- 소스 코드 기반 RAG 검색 모듈 구현
- 가이드 생성 모듈 구현

---

### Phase 4: 통합 및 완성 (1-2주)

**멤버 2:**
- 앙상블 시스템 최종 통합
- 치명도 계산 통합

**멤버 3:**
- API 서버 구현
- 전체 시스템 통합
- 소스 코드 기반 가이드 생성 통합

**멤버 2와 멤버 3 협업:**
- API 서버의 치명도 조회 API 연동

**총 예상 기간: 9-13주**

---

## 🔗 관련 문서

- **`FINAL_PREPROCESSING_GUIDE.md`**: 최종 전처리 프로세스 가이드 ⭐
- **`TEAM_ROLES.md`**: 팀 역할 분담 상세 가이드 ⭐
- **`REALTIME_PREPROCESSING_GUIDE.md`**: 실시간 전처리 가이드
- **`DATA_SHARING_PROTOCOL.md`**: 전처리 데이터 공유 프로토콜
- **`ENSEMBLE_ANOMALY_DETECTION.md`**: 앙상블 이상 탐지 시스템 가이드
- **`SOURCE_CODE_RAG_GUIDE.md`**: 소스 코드 기반 RAG 시스템 상세 가이드 ⭐
- **`SEVERITY_CALCULATION.md`**: 치명도 계산 방법

---

이 아키텍처를 바탕으로 `TEAM_ROLES.md`의 역할 분담에 따라 단계별로 구현을 진행하세요! 🚀
