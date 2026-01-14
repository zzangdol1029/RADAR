# MSA 로그 이상 탐지 및 RAG 시스템 아키텍처

## 📋 시스템 개요

MSA 환경의 로그를 분석하여 이상을 탐지하고, 치명도를 계산하며, RAG 기반 가이드를 제공하는 통합 시스템입니다.

### 목표
1. **각 모듈별 이상 탐지**: Gateway, Eureka, User, Research, Manager, Code 등
2. **치명도 계산**: 이상의 심각도 평가
3. **RAG 기반 가이드**: 오류 내용에 대한 해결 가이드 제공

---

## 🏗️ 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    로그 수집 레이어                            │
│  Gateway │ Eureka │ User │ Research │ Manager │ Code        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  전처리 파이프라인 (Preprocessing)             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ 로그 정리 │→│ 로그 파싱 │→│ Trace ID │→│ 세션화    │    │
│  │ Cleaning │  │ Parsing  │  │ 추출     │  │ Session  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
┌───────────────────────┐      ┌───────────────────────┐
│   이상 탐지 모듈        │      │   RAG 구축 모듈        │
│  (Anomaly Detection)  │      │   (RAG Builder)       │
│                       │      │                       │
│  ┌─────────────────┐  │      │  ┌─────────────────┐ │
│  │ LogBERT 모델    │  │      │  │ 벡터 임베딩      │ │
│  │ 이상 점수 계산   │  │      │  │ 메타데이터 추출  │ │
│  └─────────────────┘  │      │  └─────────────────┘ │
│           ↓            │      │           ↓          │
│  ┌─────────────────┐  │      │  ┌─────────────────┐ │
│  │ 치명도 계산      │  │      │  │ 벡터 DB 저장    │ │
│  │ Severity Score  │  │      │  │ (Chroma/Pinecone)│ │
│  └─────────────────┘  │      │  └─────────────────┘ │
└───────────────────────┘      └───────────────────────┘
        ↓                                   ↓
┌─────────────────────────────────────────────────────────────┐
│                    결과 통합 및 저장                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 이상 결과 │  │ 치명도   │  │ RAG 인덱스│  │ 메타데이터│  │
│  │ DB       │  │ DB       │  │ DB       │  │ DB       │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    RAG 기반 가이드 시스템                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ 쿼리 처리 │→│ 벡터 검색 │→│ 가이드 생성│                │
│  │ Query    │  │ Retrieval │  │ Generation│                │
│  └──────────┘  └──────────┘  └──────────┘                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                      API 서버 (FastAPI)                      │
│  /detect     - 이상 탐지                                     │
│  /severity   - 치명도 조회                                  │
│  /guide      - RAG 가이드 조회                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 주요 컴포넌트

### 1. 전처리 파이프라인 (Preprocessing Pipeline)

**역할:**
- 로그 파일 읽기 및 정리
- Drain3 기반 파싱
- Trace ID 추출 및 연결
- MSA 컨텍스트 결합
- 세션화

**입력:**
- 원본 로그 파일들 (서비스별)

**출력:**
- 전처리된 세션 데이터 (JSON)
- Trace ID별 그룹화된 데이터
- 서비스별 세션 데이터

**구현 파일:**
- `preprocessing/msa_preprocessor.py` (신규 생성 필요)
- `preprocessing/log_preprocessor.py` (기존 확장)

---

### 2. 이상 탐지 모듈 (Anomaly Detection Module)

**역할:**
- LogBERT 모델로 이상 점수 계산
- 서비스별 이상 탐지
- Trace별 이상 탐지

**입력:**
- 전처리된 세션 데이터
- 학습된 LogBERT 모델

**출력:**
- 이상 점수 (Anomaly Score)
- 이상 여부 (Is Anomaly)
- 서비스별 통계

**구현 파일:**
- `logbert_training/detect_anomalies.py` (기존)
- `anomaly_detection/service_detector.py` (신규)
- `anomaly_detection/trace_detector.py` (신규)

---

### 3. 치명도 계산 모듈 (Severity Calculation Module)

**역할:**
- 이상의 치명도 계산
- 서비스 중요도 반영
- 에러 유형별 가중치 적용

**입력:**
- 이상 점수
- 에러 정보
- 서비스 정보
- Trace 정보

**출력:**
- 치명도 점수 (Severity Score)
- 치명도 등급 (Critical, High, Medium, Low)

**구현 파일:**
- `severity/severity_calculator.py` (신규)
- `severity/service_weights.py` (신규)
- `severity/error_classifier.py` (신규)

---

### 4. RAG 구축 모듈 (RAG Builder Module)

**역할:**
- 로그 데이터 벡터화
- 메타데이터 추출
- 벡터 DB에 저장

**입력:**
- 전처리된 세션 데이터
- 이상 탐지 결과
- 치명도 정보

**출력:**
- 벡터 임베딩
- 메타데이터
- 벡터 DB 인덱스

**구현 파일:**
- `rag/vectorizer.py` (신규)
- `rag/metadata_extractor.py` (신규)
- `rag/db_builder.py` (신규)

---

### 5. RAG 기반 가이드 시스템 (RAG Guide System)

**역할:**
- 사용자 쿼리 처리
- 유사도 검색
- 가이드 생성

**입력:**
- 사용자 쿼리 (에러 메시지, Trace ID 등)
- 벡터 DB

**출력:**
- 관련 가이드 문서
- 해결 방법 제안
- 유사 사례

**구현 파일:**
- `rag/retriever.py` (신규)
- `rag/guide_generator.py` (신규)
- `rag/llm_integration.py` (신규)

---

### 6. API 서버 (API Server)

**역할:**
- RESTful API 제공
- 이상 탐지 요청 처리
- 치명도 조회
- 가이드 조회

**구현 파일:**
- `api/main.py` (신규)
- `api/routes/detection.py` (신규)
- `api/routes/severity.py` (신규)
- `api/routes/guide.py` (신규)

---

## 📊 데이터 흐름

### 1. 로그 수집 → 전처리

```
원본 로그 파일들
    ↓
로그 정리 (Cleaning)
    ↓
로그 파싱 (Drain3)
    ↓
Trace ID 추출
    ↓
세션화 (Trace 기반 또는 시간 기반)
    ↓
MSA 컨텍스트 결합
    ↓
전처리된 세션 데이터 (JSON)
```

### 2. 전처리 → 이상 탐지

```
전처리된 세션 데이터
    ↓
서비스별 세션 분리
    ↓
LogBERT 모델로 이상 점수 계산
    ↓
임계값 비교
    ↓
이상 여부 판단
    ↓
이상 탐지 결과
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

### 4. 전처리 → RAG 구축

```
전처리된 세션 데이터
    ↓
RAG용 텍스트 생성
    ↓
벡터 임베딩 생성
    ↓
메타데이터 추출
    ↓
벡터 DB에 저장
    ↓
RAG 인덱스 완성
```

### 5. RAG → 가이드 생성

```
사용자 쿼리
    ↓
쿼리 벡터화
    ↓
벡터 DB에서 유사도 검색
    ↓
상위 K개 문서 검색
    ↓
LLM으로 가이드 생성
    ↓
가이드 문서 반환
```

---

## 🗄️ 데이터베이스 설계

### 1. 이상 탐지 결과 DB

**테이블: `anomaly_detections`**
```sql
CREATE TABLE anomaly_detections (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(255),
    session_id VARCHAR(255),
    service_name VARCHAR(50),
    anomaly_score FLOAT,
    is_anomaly BOOLEAN,
    threshold FLOAT,
    detected_at TIMESTAMP,
    metadata JSONB
);
```

### 2. 치명도 DB

**테이블: `severity_scores`**
```sql
CREATE TABLE severity_scores (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(255),
    session_id VARCHAR(255),
    service_name VARCHAR(50),
    severity_score FLOAT,
    severity_level VARCHAR(20), -- CRITICAL, HIGH, MEDIUM, LOW
    error_count INTEGER,
    warning_count INTEGER,
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
    trace_id VARCHAR(255),
    session_id VARCHAR(255),
    service_name VARCHAR(50),
    text_content TEXT,
    vector_id VARCHAR(255), -- 벡터 DB의 ID
    metadata JSONB,
    created_at TIMESTAMP
);
```

### 4. 벡터 DB (Chroma/Pinecone)

**컬렉션: `log_guides`**
- 벡터: 임베딩 벡터 (768차원 또는 1536차원)
- 메타데이터:
  - trace_id
  - service_name
  - error_type
  - severity_level
  - guide_text
  - solution_steps

---

## 🔄 API 엔드포인트

### 1. 이상 탐지 API

**POST `/api/v1/detect`**
```json
{
  "log_file_path": "/path/to/logs/gateway.log",
  "service_name": "gateway",
  "threshold": 0.5
}
```

**Response:**
```json
{
  "trace_id": "abc123",
  "sessions": [
    {
      "session_id": "gateway_1",
      "anomaly_score": 0.85,
      "is_anomaly": true,
      "service_name": "gateway"
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

**GET `/api/v1/severity/{trace_id}`**
```json
{
  "trace_id": "abc123",
  "severity_score": 0.92,
  "severity_level": "CRITICAL",
  "services": [
    {
      "service_name": "gateway",
      "severity_score": 0.85,
      "error_count": 2
    }
  ]
}
```

### 3. 가이드 조회 API

**POST `/api/v1/guide`**
```json
{
  "query": "Connection timeout error in gateway service",
  "trace_id": "abc123",
  "service_name": "gateway"
}
```

**Response:**
```json
{
  "guides": [
    {
      "title": "Connection Timeout 해결 방법",
      "steps": [
        "1. 네트워크 연결 확인",
        "2. 타임아웃 설정 조정",
        "3. 로드 밸런서 상태 확인"
      ],
      "similar_cases": [
        {
          "trace_id": "xyz789",
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
├── preprocessing/              # 전처리 모듈
│   ├── msa_preprocessor.py    # MSA 전처리 (신규)
│   ├── log_preprocessor.py    # 기존 전처리 (확장)
│   └── ...
│
├── logbert_training/          # 모델 학습 및 추론
│   ├── detect_anomalies.py    # 이상 탐지 (기존)
│   ├── model.py               # LogBERT 모델
│   └── ...
│
├── anomaly_detection/         # 이상 탐지 모듈 (신규)
│   ├── service_detector.py   # 서비스별 탐지
│   ├── trace_detector.py     # Trace별 탐지
│   └── __init__.py
│
├── severity/                  # 치명도 계산 모듈 (신규)
│   ├── severity_calculator.py # 치명도 계산
│   ├── service_weights.py    # 서비스 가중치
│   ├── error_classifier.py   # 에러 분류
│   └── __init__.py
│
├── rag/                      # RAG 모듈 (신규)
│   ├── vectorizer.py         # 벡터화
│   ├── metadata_extractor.py # 메타데이터 추출
│   ├── db_builder.py         # 벡터 DB 구축
│   ├── retriever.py          # 검색
│   ├── guide_generator.py    # 가이드 생성
│   ├── llm_integration.py    # LLM 통합
│   └── __init__.py
│
├── api/                      # API 서버 (신규)
│   ├── main.py              # FastAPI 앱
│   ├── routes/
│   │   ├── detection.py     # 이상 탐지 API
│   │   ├── severity.py      # 치명도 API
│   │   └── guide.py        # 가이드 API
│   └── models/
│       └── schemas.py       # Pydantic 모델
│
├── database/                # 데이터베이스 (신규)
│   ├── migrations/          # DB 마이그레이션
│   └── models.py            # SQLAlchemy 모델
│
└── config/                  # 설정 파일 (신규)
    ├── config.yaml          # 전체 설정
    ├── service_weights.yaml # 서비스 가중치
    └── rag_config.yaml      # RAG 설정
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

## 📝 다음 단계

1. **전처리 모듈 구현** (1주)
2. **이상 탐지 모듈 구현** (1주)
3. **치명도 계산 모듈 구현** (1주)
4. **RAG 구축 모듈 구현** (2주)
5. **API 서버 구현** (1주)
6. **통합 테스트** (1주)

**총 예상 기간: 7주**

---

이 아키텍처를 바탕으로 단계별로 구현을 진행하세요! 🚀
