# 전처리 데이터 검증 가이드

## 개요

전처리된 데이터가 올바르게 생성되었는지 검증하는 방법을 설명합니다.

## 검증 스크립트 사용법

### 기본 사용법

```bash
cd preprocessing
python validate_preprocessed_data.py
```

### 특정 파일만 검증

```bash
python validate_preprocessed_data.py --file preprocessed_logs_2025-02-24.json
```

### 출력 디렉토리 지정

```bash
python validate_preprocessed_data.py --output-dir output
```

## 검증 항목

### 1. 파일 존재 및 JSON 형식 검증
- ✅ 모든 파일이 존재하는지 확인
- ✅ JSON 형식이 올바른지 확인
- ✅ 배열 형식인지 확인

### 2. 데이터 구조 검증
각 세션에 다음 필드가 포함되어 있는지 확인:
- `session_id`: 세션 고유 ID
- `event_sequence`: Event ID 시퀀스 (리스트)
- `token_ids`: BERT 입력 토큰 (리스트)
- `attention_mask`: 패딩 마스크 (리스트)
- `has_error`: ERROR 로그 포함 여부 (불린)
- `has_warn`: WARN 로그 포함 여부 (불린)
- `service_name`: 서비스명 (문자열)
- `original_logs`: 원본 로그 리스트
- `simplified_text`: RAG용 간소화된 텍스트

### 3. 데이터 일관성 검증
- `token_ids`와 `attention_mask`의 길이가 일치하는지 확인
- `token_ids`가 [CLS] 토큰(101)으로 시작하는지 확인
- `event_sequence`가 비어있지 않은지 확인

### 4. 통계 정보 수집
- 총 파일 수
- 총 세션 수
- 총 이벤트 수
- 고유 Event ID 수
- 서비스별 세션 분포
- 에러/경고 포함 세션 수
- 파일 크기 통계

## 검증 결과 해석

### ✅ 정상적인 결과

```
📁 총 파일 수: 286개
   ✅ 유효한 파일: 286개
   ❌ 무효한 파일: 0개

📊 총 세션 수: 1,234,567개
   ⚠️  에러 포함 세션: 12,345개 (1.0%)
   ⚠️  경고 포함 세션: 34,567개 (2.8%)

🔢 총 이벤트 수: 61,728,350개
   고유 Event ID 수: 1,234개
```

### ❌ 문제가 있는 경우

검증 스크립트는 다음과 같은 오류를 감지합니다:

1. **필수 필드 누락**: 세션에 필수 필드가 없을 때
2. **데이터 타입 오류**: 필드의 타입이 예상과 다를 때
3. **길이 불일치**: `token_ids`와 `attention_mask` 길이가 다를 때
4. **빈 시퀀스**: `event_sequence`가 비어있을 때

## 수동 검증 방법

### 1. 파일 크기 확인

```bash
ls -lh preprocessing/output/preprocessed_logs_*.json | head -10
```

### 2. JSON 형식 확인

```bash
python -m json.tool preprocessing/output/preprocessed_logs_2025-02-24.json > /dev/null && echo "✅ JSON 형식 정상"
```

### 3. 세션 수 확인

```bash
python -c "import json; data=json.load(open('preprocessing/output/preprocessed_logs_2025-02-24.json')); print(f'세션 수: {len(data)}개')"
```

### 4. 샘플 데이터 확인

```bash
python -c "import json; data=json.load(open('preprocessing/output/preprocessed_logs_2025-02-24.json')); print(json.dumps(data[0], ensure_ascii=False, indent=2))"
```

## 예상되는 정상 값

### 세션 수
- 날짜별로 수백~수만 개의 세션이 생성될 수 있습니다
- `window_size: 50` 설정 기준으로 약 50개 로그당 1개 세션이 생성됩니다

### Event ID 수
- 서비스와 로그 패턴에 따라 수백~수천 개의 고유 Event ID가 생성됩니다
- Drain3가 로그 템플릿을 추출하여 압축합니다

### 파일 크기
- 세션 수에 비례하여 증가합니다
- 평균적으로 세션당 약 10-20KB 정도입니다

## 문제 해결

### 문제: JSON 파싱 오류
**원인**: 파일이 손상되었거나 전처리 중단
**해결**: 해당 날짜의 전처리를 다시 실행

### 문제: 필수 필드 누락
**원인**: 전처리 코드 버전 불일치
**해결**: 최신 버전의 전처리 스크립트로 재실행

### 문제: token_ids와 attention_mask 길이 불일치
**원인**: 인코딩 과정의 버그
**해결**: 전처리 코드 확인 및 수정 후 재실행

## 다음 단계

검증이 완료되면:

1. **LogBERT 학습**: `token_ids`와 `attention_mask` 사용
2. **RAG 벡터화**: `simplified_text`를 임베딩하여 검색 인덱스 생성
3. **장애 탐지**: `has_error` 또는 `has_warn` 플래그로 우선순위 필터링
4. **패턴 분석**: `event_sequence`를 분석하여 이상 패턴 탐지

