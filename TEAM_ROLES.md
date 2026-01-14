# 팀 역할 분담 (3명)

## 📋 개요

3명이 동시에 작업할 수 있도록 모듈별로 역할을 분담합니다. 
- **멤버 1**: 전처리 전담
- **멤버 2**: LogBERT 학습, 앙상블 탐지 및 치명도 계산
- **멤버 3**: DeepLog/LogLSTM/LogTCN 학습, 소스 코드 파싱/벡터화, 벡터 DB 구축, RAG 검색 모듈, API 서버
RAG 시스템은 소스 코드를 기반으로 가이드를 생성합니다.

---

## 👥 역할 분담

### 👤 멤버 1: 로그 수집, 통합 및 전처리 전담

**담당 모듈:**
- 로그 수집 파이프라인 (Log Collection Pipeline)
- 로그 통합 (Log Aggregation)
- 전처리 파이프라인 (Preprocessing Pipeline)
- MSA 로그 처리
- Trace ID 추출 및 연결
- 데이터 검증 및 공유

**주요 작업:**

**Phase 1: 로그 수집 및 통합 (1주)**
1. **로그 수집 모듈 구현**
   - `preprocessing/log_collector.py` 작성
   - 여러 서비스 로그 파일 수집 (gateway, eureka, user, research, manager, code)
   - 로그 파일 파싱 및 파싱
   - 날짜 범위 필터링 지원

2. **로그 통합 모듈 구현**
   - 타임스탬프 기준 로그 병합
   - 서비스별 로그 통합
   - 통합 로그 저장 (JSON 형식)
   - 수집 통계 정보 생성

**Phase 2: 배치 전처리 (학습용) (1주)**
3. **MSA 배치 전처리 모듈 구현**
   - `preprocessing/batch_preprocessor.py` 작성
   - `preprocessing/msa_preprocessor.py` 작성
   - Trace ID 추출 로직 구현
   - 서비스별 세션화 구현
   - MSA 컨텍스트 빌더 구현

4. **기존 전처리 모듈 확장**
   - `preprocessing/log_preprocessor.py` 확장
   - Trace ID 기반 세션화 추가
   - 서비스 간 호출 관계 추출

**Phase 3: 실시간 전처리 (운영용) (1주)** ⭐
5. **실시간 스트리밍 전처리 모듈 구현**
   - `preprocessing/streaming_preprocessor.py` 작성 ⭐
   - `preprocessing/streaming_sessionizer.py` 작성 ⭐
   - 로그 스트림 실시간 처리
   - 윈도우 기반 세션화
   - 메모리 효율적 처리

6. **실시간 파이프라인 구축**
   - 실시간 이상 탐지와 연동
   - 로그 파일 tail -f 방식 처리
   - 여러 서비스 병렬 처리

**Phase 4: 데이터 생성 및 공유 (3-5일)**
7. **배치 전처리 데이터 생성 및 공유** ⭐
   - 통합된 로그를 배치 전처리하여 학습 데이터 생성
   - 전처리된 데이터를 JSON 형식으로 저장
   - **다른 멤버들이 사용할 수 있도록 공유**
   - 데이터 형식 문서화
   - 데이터 검증 및 품질 확인

**예상 기간:** 3-4주

**산출물:**
- `preprocessing/log_collector.py` ⭐ (신규)
- `preprocessing/batch_preprocessor.py` (배치 전처리)
- `preprocessing/streaming_preprocessor.py` ⭐ (실시간 전처리)
- `preprocessing/streaming_sessionizer.py` ⭐ (실시간 세션화)
- `preprocessing/msa_preprocessor.py`
- `preprocessing/trace_extractor.py`
- `preprocessing/msa_context_builder.py`
- **통합된 로그 데이터** (logs/merged/)
- **배치 전처리된 데이터** (preprocessing/output/) ⭐ (학습용)
- **실시간 전처리 파이프라인** ⭐ (운영용)
- 데이터 형식 문서
- 수집/전처리 통계 정보
- 테스트 코드 및 문서

---

### 👤 멤버 2: LogBERT 모델 학습, 앙상블 탐지 및 치명도 계산 담당

**담당 모듈:**
- LogBERT 모델 학습
- 앙상블 이상 탐지 모듈
- 서비스별/Trace별 탐지
- 치명도 계산 모듈

**주요 작업:**

**Phase 1: LogBERT 모델 학습 (2주) - 멤버 3과 병렬**
1. **전처리된 데이터 사용**
   - 멤버 1이 생성한 전처리 데이터 활용
   - 데이터 형식 확인 및 검증

2. **LogBERT 모델 학습**
   - `logbert_training/train_logbert.py` 작성/수정
   - 전처리된 데이터로 LogBERT 학습
   - 하이퍼파라미터 튜닝
   - 모델 평가 및 검증
   - 체크포인트 저장

**Phase 2: 앙상블 시스템 구현 (1-2주)**
3. **앙상블 이상 탐지 모듈 구현**
   - `anomaly_detection/ensemble_detector.py` 작성
   - 여러 모델 지원 및 통합 (LogBERT, DeepLog, LogLSTM, LogTCN)
   - 앙상블 방법 구현 (가중 평균, 평균, 최대값)
   - 멤버 3의 모델과 통합

4. **서비스별/Trace별 탐지 구현**
   - `anomaly_detection/service_detector.py` 작성
   - `anomaly_detection/trace_detector.py` 작성
   - 서비스별 이상 탐지 로직
   - Trace별 이상 탐지 로직

**Phase 3: 치명도 계산 모듈 구현 (1-2주)**
5. **치명도 계산 모듈 구현**
   - `severity/severity_calculator.py` 작성
   - `severity/service_weights.py` 작성
   - `severity/error_classifier.py` 작성
   - 치명도 공식 구현
   - 앙상블 점수와 통합

**예상 기간:** 4-6주

**산출물:**
- `logbert_training/checkpoints/logbert/best_model.pt`
- `anomaly_detection/ensemble_detector.py`
- `anomaly_detection/service_detector.py`
- `anomaly_detection/trace_detector.py`
- `severity/` 모듈 전체
- 테스트 코드 및 문서

---

### 👤 멤버 3: DeepLog/LogLSTM/LogTCN 모델 학습, 소스 코드 RAG 구축 및 RAG 검색 모듈 담당

**담당 모듈:**
- DeepLog 모델 학습
- LogLSTM 모델 학습 (선택)
- LogTCN 모델 학습 (선택)
- 소스 코드 파싱 및 벡터화 모듈
- 벡터 DB 구축
- 소스 코드 기반 RAG 검색 모듈
- 가이드 생성 모듈
- API 서버

**주요 작업:**

**Phase 1: 모델 학습 (2-3주) - 멤버 2와 병렬**
1. **전처리된 데이터 사용**
   - 멤버 1이 생성한 전처리 데이터 활용
   - 데이터 형식 확인 및 검증

2. **DeepLog 모델 학습**
   - `logbert_training/train_deeplog.py` 작성/수정
   - 전처리된 데이터로 DeepLog 학습
   - 하이퍼파라미터 튜닝
   - 모델 평가 및 검증
   - 체크포인트 저장

3. **LogLSTM 모델 학습 (선택)**
   - `logbert_training/train_lstm.py` 작성/수정
   - 전처리된 데이터로 LogLSTM 학습
   - 체크포인트 저장

4. **LogTCN 모델 학습 (선택)**
   - `logbert_training/train_tcn.py` 작성/수정
   - 전처리된 데이터로 LogTCN 학습
   - 체크포인트 저장

**Phase 2: 소스 코드 파싱 및 벡터화 모듈 구현 (1-2주)**
5. **소스 코드 파싱 모듈 구현**
   - `rag/code_parser.py` 작성
   - 소스 코드 파일 읽기 및 파싱
   - 코드 청크 분할 (함수/클래스 단위)
   - 코드 메타데이터 추출

6. **소스 코드 벡터화 모듈 구현**
   - `rag/code_vectorizer.py` 작성
   - 코드 전용 임베딩 모델 사용 (CodeBERT, StarCoder 등)
   - 코드 청크별 임베딩 생성

**Phase 3: 벡터 DB 구축 (1주)**
7. **벡터 DB 구축 모듈 구현**
   - `rag/db_builder.py` 작성
   - Chroma/Pinecone 통합
   - 로그 데이터와 소스 코드를 함께 인덱싱

**Phase 4: RAG 검색 모듈 구현 (1-2주)**
8. **소스 코드 기반 RAG 검색 모듈 구현**
   - `rag/code_retriever.py` 작성
   - 로그 에러와 관련된 소스 코드 검색
   - 유사도 기반 코드 청크 검색
   - 상위 K개 관련 코드 검색

9. **소스 코드 기반 가이드 생성 모듈 구현**
   - `rag/code_guide_generator.py` 작성
   - `rag/llm_integration.py` 작성
   - LLM 통합 (OpenAI/LangChain)
   - 소스 코드를 참조한 가이드 생성
   - 코드 위치 및 수정 방법 제시

**Phase 5: API 서버 구현 (1주)**
10. **API 서버 구현**
    - `api/main.py` 작성
    - `api/routes/detection.py` 작성 (앙상블 이상 탐지 API)
    - `api/routes/severity.py` 작성 (치명도 조회 API - 멤버 2와 협업)
    - `api/routes/guide.py` 작성 (소스 코드 기반 가이드 API)
    - FastAPI 통합

11. **통합 및 테스트**
    - 전체 시스템 통합
    - 소스 코드 기반 가이드 생성 테스트
    - API 테스트
    - 문서화

**예상 기간:** 5-7주

**산출물:**
- `logbert_training/checkpoints/deeplog/best_model.pt`
- `logbert_training/checkpoints/lstm/best_model.pt` (선택)
- `logbert_training/checkpoints/tcn/best_model.pt` (선택)
- `rag/code_parser.py`
- `rag/code_vectorizer.py`
- `rag/db_builder.py`
- `rag/code_retriever.py`
- `rag/code_guide_generator.py`
- `rag/llm_integration.py`
- `api/` 모듈 전체
- 벡터 DB 인덱스 (로그 + 소스 코드)
- API 문서 (Swagger)
- 테스트 코드 및 문서

---

## 🔄 작업 순서 및 의존성

### Phase 1: 로그 수집, 통합 및 전처리 (3-4주) - 멤버 1 전담

**멤버 1 (우선 작업):**
- **로그 수집 모듈 구현** ⭐
  - 여러 서비스 로그 파일 수집
  - 로그 파싱 및 파싱
- **로그 통합 모듈 구현** ⭐
  - 타임스탬프 기준 병합
  - 통합 로그 저장
- **배치 전처리 모듈 구현** (학습용)
  - Trace ID 추출 로직 구현
  - 배치 전처리 파이프라인 완성
- **실시간 전처리 모듈 구현** ⭐ (운영용)
  - 스트리밍 전처리 파이프라인
  - 실시간 세션화
  - 메모리 효율적 처리
- **배치 전처리 데이터 생성 및 공유** ⭐
  - 통합된 로그를 배치 전처리하여 학습 데이터 생성
  - 다른 멤버들에게 전달
  - 데이터 형식 문서화

**멤버 2, 3:**
- 각자 모델 학습 스크립트 준비
- 모델 아키텍처 설계
- 하이퍼파라미터 연구
- 추가 모듈 설계 (치명도, RAG, API 등)

---

### Phase 2: 모델 학습 (2주) - 병렬 작업 ⭐⭐⭐

**멤버 2:**
- **전처리된 데이터로 LogBERT 학습** (멤버 3과 동시에)
- 하이퍼파라미터 튜닝
- 모델 평가 및 체크포인트 저장

**멤버 3:**
- **전처리된 데이터로 DeepLog 학습** (멤버 2와 동시에)
- **전처리된 데이터로 LogLSTM/LogTCN 학습** (선택, 순차 또는 병렬)
- 하이퍼파라미터 튜닝
- 모델 평가 및 체크포인트 저장

**멤버 1:**
- 전처리 모듈 개선 및 문서화
- 데이터 품질 검증
- 다음 단계 준비

**장점:**
- ✅ 전처리 데이터만 공유하면 동시에 학습 가능
- ✅ 서로 다른 모델이므로 충돌 없음
- ✅ 학습 시간 단축 (병렬 처리)
- ✅ 각자 최적화 가능

---

### Phase 3: 모듈 구현 (2-3주) - 병렬 작업

**멤버 2:**
- 앙상블 이상 탐지 모듈 구현
- 학습된 모델들 통합 (LogBERT, DeepLog, LogLSTM, LogTCN)
- 서비스별/Trace별 탐지 로직 구현
- 치명도 계산 모듈 구현
- 앙상블 점수와 치명도 통합

**멤버 3:**
- 소스 코드 파싱 및 벡터화 모듈 구현
- 벡터 DB 구축
- 소스 코드 기반 RAG 검색 모듈 구현
- 가이드 생성 모듈 구현

**멤버 1:**
- 전처리 모듈 최종 검증
- 데이터 파이프라인 최적화

---

### Phase 4: 통합 및 완성 (1-2주)

**멤버 2:**
- 앙상블 시스템 최종 통합
- 모든 모델 통합 테스트
- 치명도 계산 통합 테스트
- 앙상블 점수와 치명도 연결

**멤버 3:**
- API 서버 구현
- 전체 시스템 통합
- 소스 코드 기반 가이드 생성 통합 테스트
- API 통합 테스트

**멤버 2와 멤버 3 협업:**
- API 서버의 치명도 조회 API 연동 (멤버 3이 API 구현, 멤버 2가 치명도 모듈 제공)

**모든 멤버:**
- 모듈 간 통합 테스트
- 버그 수정
- 문서화

---

## 📊 작업 병렬화 전략

### 병렬 작업 전략 ⭐⭐⭐

**핵심 아이디어: 전처리 전담 + 모델 학습 병렬화**

1. **로그 수집, 통합 및 전처리 단계 (멤버 1 전담, 3-4주)**
   - 멤버 1이 로그 수집 모듈 구현 ⭐
   - 멤버 1이 로그 통합 모듈 구현 ⭐
   - 멤버 1이 배치 전처리 파이프라인 완성 (학습용)
   - 멤버 1이 실시간 전처리 파이프라인 구현 ⭐ (운영용)
   - 통합된 로그를 배치 전처리하여 학습 데이터 생성
   - 배치 전처리된 데이터를 JSON 형식으로 저장
   - **다른 멤버들에게 공유** ⭐
   - 데이터 형식 문서화

2. **모델 학습 단계 (멤버 2, 3, 2-3주) - 병렬 작업** ⭐⭐⭐
   - **멤버 2**: 전처리된 데이터로 LogBERT 학습
   - **멤버 3**: 전처리된 데이터로 DeepLog 학습 (동시에)
   - **멤버 3**: 전처리된 데이터로 LogLSTM/LogTCN 학습 (선택, 순차 또는 병렬)
   - 서로 다른 모델이므로 충돌 없음
   - 학습 시간 단축 (순차: 4-6주 → 병렬: 2-3주)

3. **모듈 구현 단계 (멤버 2, 3, 3-4주) - 병렬 작업**
   - **멤버 2**: 앙상블 시스템 구현 + 치명도 계산 모듈 구현
   - **멤버 3**: 소스 코드 파싱/벡터화 + 벡터 DB 구축 + RAG 검색 모듈 구현
   - 각자 담당 모듈 독립적으로 구현

4. **통합 단계 (모든 멤버, 1-2주)**
   - 모듈 간 통합
   - 통합 테스트
   - 버그 수정

---

### 인터페이스 정의 (초기 1일)

**모든 멤버가 함께:**
- 데이터 형식 정의 (JSON Schema)
- 함수 시그니처 정의
- API 스펙 정의

**결과물:**
- `interfaces/data_schemas.py`
- `interfaces/api_spec.yaml`
- `interfaces/module_interfaces.md`

---

## 🔗 모듈 간 인터페이스

### 1. 전처리 → 이상 탐지

**입력 형식:**
```json
{
  "trace_id": "abc123",
  "sessions": [
    {
      "session_id": "gateway_1",
      "service_name": "gateway",
      "token_ids": [101, 1, 2, ...],
      "attention_mask": [1, 1, 1, ...]
    }
  ]
}
```

**출력 형식:**
```json
{
  "trace_id": "abc123",
  "sessions": [
    {
      "session_id": "gateway_1",
      "anomaly_score": 0.85,
      "is_anomaly": true
    }
  ]
}
```

---

### 2. 이상 탐지 → 치명도 계산

**입력 형식:**
```json
{
  "anomaly_score": 0.85,
  "error_count": 3,
  "warning_count": 1,
  "service_name": "gateway",
  "affected_services": ["gateway", "research"]
}
```

**출력 형식:**
```json
{
  "severity_score": 0.75,
  "severity_level": "HIGH",
  "components": {
    "anomaly": 0.34,
    "error": 0.09,
    "service": 0.15
  }
}
```

---

### 3. 전처리 + 소스 코드 → RAG 구축

**입력 형식 (로그):**
```json
{
  "trace_id": "abc123",
  "rag_text": "Trace ID: abc123\nServices: gateway, research\n...",
  "metadata": {
    "services": ["gateway", "research"],
    "error_count": 2,
    "error_messages": ["Connection timeout", "Service unavailable"]
  }
}
```

**입력 형식 (소스 코드):**
```json
{
  "file_path": "gateway/src/main/java/com/example/GatewayController.java",
  "service_name": "gateway",
  "code_chunks": [
    {
      "chunk_id": "chunk_1",
      "type": "method",
      "name": "handleRequest",
      "code": "public ResponseEntity<?> handleRequest(...) { ... }",
      "line_start": 45,
      "line_end": 78
    }
  ]
}
```

**출력 형식:**
```json
{
  "vector_id": "vec_123",
  "embedding": [0.1, 0.2, ...],
  "metadata": {
    "type": "log",  // 또는 "code"
    "trace_id": "abc123",
    "service_name": "gateway",
    "related_code": ["chunk_1", "chunk_2"]  // 로그의 경우 관련 코드 청크 ID
  }
}
```

---

### 4. 소스 코드 기반 RAG 검색 → 가이드 생성

**입력 형식:**
```json
{
  "query": "Connection timeout error in gateway service",
  "trace_id": "abc123",
  "service_name": "gateway"
}
```

**출력 형식:**
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
      "code_fix_suggestion": "timeout 설정을 5000ms에서 10000ms로 증가"
    }
  ]
}
```

---

## 📅 일정표

### Week 1-4: 로그 수집, 통합 및 전처리 (멤버 1 전담)
- **Day 1-2**: 인터페이스 정의 (모든 멤버)
- **Day 3-7**: 멤버 1이 로그 수집 및 통합 모듈 구현 ⭐
- **Day 8-14**: 멤버 1이 배치 전처리 파이프라인 완성 (학습용)
- **Day 15-21**: 멤버 1이 실시간 전처리 파이프라인 구현 ⭐ (운영용)
- **Day 22-24**: 배치 전처리된 데이터 생성 및 공유 ⭐
- **다른 멤버들**: 모델 학습 스크립트 준비 및 설계

### Week 5-7: 병렬 모델 학습 ⭐⭐⭐
- **멤버 2**: 전처리된 데이터로 LogBERT 학습 (동시에)
- **멤버 3**: 전처리된 데이터로 DeepLog 학습 (동시에)
- **멤버 3**: 전처리된 데이터로 LogLSTM/LogTCN 학습 (선택, 순차 또는 병렬)
- **멤버 1**: 전처리 모듈 개선 및 문서화
- **결과**: 모든 모델 체크포인트 완성

### Week 8-11: 모듈 구현 (병렬)
- **멤버 2**: 앙상블 이상 탐지 모듈 구현 + 치명도 계산 모듈 구현
- **멤버 3**: 소스 코드 파싱/벡터화 모듈 구현 + 벡터 DB 구축 + RAG 검색 모듈 구현

### Week 12: 통합 및 테스트
- **멤버 2**: 앙상블 시스템 통합 + 치명도 계산 통합
- **멤버 3**: API 서버 구현 및 전체 통합
- **멤버 2와 멤버 3 협업**: API 서버의 치명도 조회 API 연동
- **모든 멤버**: 통합 테스트 및 버그 수정

### Week 13: 최종 완성
- 문서화
- 성능 최적화
- 배포 준비

---

## 💡 협업 팁

### 1. 일일 스탠드업 (15분)
- 각자 진행 상황 공유
- 블로커 확인
- 다음 작업 계획

### 2. 주간 리뷰 (1시간)
- 코드 리뷰
- 인터페이스 조정
- 통합 테스트

### 3. Git 브랜치 전략
- `main`: 메인 브랜치
- `feature/preprocessing`: 전처리 기능
- `feature/anomaly-detection`: 이상 탐지 기능
- `feature/rag`: RAG 기능
- `feature/api`: API 기능

### 4. 문서화
- 각 모듈별 README 작성
- API 문서 자동 생성
- 아키텍처 다이어그램 유지

---

---

## 📚 관련 문서

- `REALTIME_PREPROCESSING_GUIDE.md`: 실시간 전처리 파이프라인 가이드 ⭐
- `LOG_COLLECTION_PIPELINE.md`: 로그 수집 및 통합 파이프라인 가이드 ⭐
- `DATA_SHARING_PROTOCOL.md`: 전처리 데이터 공유 프로토콜 ⭐
- `PARALLEL_TRAINING_GUIDE.md`: 병렬 모델 학습 가이드 ⭐
- `ENSEMBLE_ANOMALY_DETECTION.md`: 앙상블 이상 탐지 시스템 가이드
- `SOURCE_CODE_RAG_GUIDE.md`: 소스 코드 기반 RAG 시스템 상세 가이드
- `MSA_LOG_PREPROCESSING_GUIDE.md`: MSA 로그 전처리 가이드
- `SEVERITY_CALCULATION.md`: 치명도 계산 방법
- `SYSTEM_ARCHITECTURE.md`: 전체 시스템 아키텍처

---

이 역할 분담으로 효율적으로 프로젝트를 진행할 수 있습니다! 🚀
