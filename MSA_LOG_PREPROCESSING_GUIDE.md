# MSA 로그 전처리 가이드

## ⚠️ 중요: 현재 로그 환경

**실제 로그 분석 결과:**
- ❌ 로그에 `trace_id` 필드가 **직접 포함되어 있지 않음**
- ✅ Gateway 로그에는 `client_ip`, `access_time`, `method`, `url`, `status` 등 포함
- ✅ Manager 로그는 일반 Spring Boot 로그 형식 (스레드명, 타임스탬프 등)

**따라서 현재는 Trace ID 기반 세션화가 불가능합니다.**

**대안 전략:**
- ✅ **하이브리드 복합 키 기반 세션화** 사용 (현재 적용 중)
- ✅ IP + URL + 시간 매칭으로 서비스 간 연결
- 🔄 향후 Trace ID 추가 시 Trace ID 기반으로 전환 가능

**최종 전처리 프로세스는 `FINAL_PREPROCESSING_GUIDE.md`를 참고하세요!**

---

## 📋 개요

MSA (Microservices Architecture) 환경에서 수집된 로그를 이상 탐지, 치명도 계산, RAG 시스템 구축을 위해 전처리하는 방법을 안내합니다.

**⚠️ 참고:** 이 문서는 **이상적인 MSA 환경(Trace ID가 있는 경우)**을 가정한 가이드입니다. 현재 실제 로그 환경에서는 `FINAL_PREPROCESSING_GUIDE.md`의 하이브리드 복합 키 방식을 사용합니다.

### 대상 서비스
- **Gateway**: API 게이트웨이
- **Eureka**: 서비스 디스커버리
- **User**: 사용자 서비스
- **Research**: 연구 서비스
- **Manager**: 관리 서비스
- **Code**: 코드 서비스

---

## 🏗️ MSA 로그 특성

### 1. 분산 추적 (Distributed Tracing) - 이상적인 경우

**⚠️ 현재 로그에는 Trace ID가 없습니다!**

MSA 환경에서는 **Trace ID**를 통해 여러 서비스의 로그를 연결할 수 있습니다 (이상적인 경우).

```
Gateway → Research → Manager → Code
  ↓         ↓          ↓        ↓
같은 Trace ID로 연결 (Trace ID가 있을 때만 가능)
```

**현재 상황:**
- Trace ID가 없으므로 복합 키 기반 연결 사용
- Gateway: `client_ip` + `시간` + `url`
- Manager: `user_ip_addr` + `url_addr` + `시간` (INSERT 로그에서)

### 2. 서비스 간 의존성

```
Gateway (진입점)
  ├── Research
  │     ├── Manager
  │     └── Code
  └── User
```

### 3. 로그 형식

**Spring Boot 표준 로그:**
```
2025-08-13 17:24:09.631 INFO 4129012 --- [or-http-epoll-2] k.r.b.g.f.CustomLoggingFilter : {"client_ip":"116.125.84.76","trace_id":"abc123","method":"GET","url":"/research/api/...","status":200}
```

**JSON 형식 로그:**
```json
{
  "timestamp": "2025-08-13T17:24:09.631",
  "level": "INFO",
  "service": "gateway",
  "trace_id": "abc123",
  "span_id": "def456",
  "message": "Request processed"
}
```

---

## 🔄 전처리 파이프라인

### 전체 흐름

```
원본 로그 파일들 (서비스별)
    ↓
[1단계] 로그 정리 및 파싱
    ↓
[2단계] Trace ID 추출 및 연결
    ↓
[3단계] 서비스별 세션화
    ↓
[4단계] MSA 컨텍스트 결합
    ↓
[5단계] 이상 탐지용 인코딩
    ↓
[6단계] RAG용 메타데이터 추출
    ↓
전처리된 데이터 (이상 탐지 + RAG)
```

---

## 📝 단계별 전처리 방법

### 1단계: 로그 정리 및 파싱

#### 목적
- 의미 없는 데이터 제거
- 구조화된 로그 추출

#### 처리 내용

**1.1 로그 정리**
```python
# Spring Boot 배너 제거
BANNER_PATTERNS = [
    r'\.\s+____.*?Spring Boot.*?::',
    r':: Spring Boot ::',
    r'-----------------------------------------------------------------------------------------',
]

# 빈 줄 제거
# 특수 패턴 제거
```

**1.2 로그 파싱 (Drain3)**
```python
# 템플릿 추출
원본: "2025-08-13 17:24:09 INFO gateway: GET /research/api/data status=200"
템플릿: "INFO gateway: GET <*> status=<*>"
파라미터: ["/research/api/data", "200"]
Event ID: 123
```

**구현 예시:**
```python
from preprocessing.log_preprocessor import LogCleaner, LogParser

cleaner = LogCleaner()
parser = LogParser()

cleaned_line = cleaner.clean_log_line(raw_line)
if cleaned_line:
    parsed = parser.parse_log(cleaned_line)
```

---

### 2단계: Trace ID 추출 및 연결 (현재 미사용)

#### ⚠️ 현재 상황

**Trace ID가 로그에 없으므로 이 단계는 현재 사용되지 않습니다.**

**대신 사용하는 방식:**
- 복합 키 기반 세션화 (Gateway: `client_ip` + `시간` + `url`)
- 후처리 연결 단계에서 IP + URL + 시간 매칭

**향후 Trace ID가 추가되면 이 방식으로 전환 가능합니다.**

---

#### 목적 (Trace ID가 있을 때)
- MSA 환경에서 분산된 로그를 하나의 요청으로 연결
- 서비스 간 호출 관계 파악

#### Trace ID 추출 방법 (참고용)

**2.1 JSON 로그에서 추출**
```python
import json
import re

def extract_trace_id_from_json(log_line: str) -> Optional[str]:
    """JSON 형식 로그에서 Trace ID 추출"""
    try:
        # JSON 파싱
        data = json.loads(log_line)
        return data.get('trace_id') or data.get('traceId') or data.get('X-Trace-Id')
    except:
        # JSON이 아닌 경우 정규식으로 추출
        match = re.search(r'"trace_id"\s*:\s*"([^"]+)"', log_line)
        if match:
            return match.group(1)
        return None
```

**2.2 HTTP 헤더에서 추출**
```python
def extract_trace_id_from_http(log_line: str) -> Optional[str]:
    """HTTP 로그에서 Trace ID 추출"""
    # X-Trace-Id 헤더 패턴
    patterns = [
        r'X-Trace-Id[:\s]+([a-zA-Z0-9-]+)',
        r'trace_id[:\s]+([a-zA-Z0-9-]+)',
        r'traceId[:\s]+([a-zA-Z0-9-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, log_line, re.IGNORECASE)
        if match:
            return match.group(1)
    return None
```

**2.3 Spring Cloud Sleuth 형식**
```python
def extract_trace_id_sleuth(log_line: str) -> Optional[str]:
    """Spring Cloud Sleuth 형식에서 추출"""
    # 예: [abc123,def456,false]
    match = re.search(r'\[([a-zA-Z0-9-]+),', log_line)
    if match:
        return match.group(1)
    return None
```

**통합 추출 함수:**
```python
def extract_trace_id(log_line: str, service_name: str) -> Optional[str]:
    """모든 방법을 시도하여 Trace ID 추출"""
    # 1. JSON 형식 시도
    trace_id = extract_trace_id_from_json(log_line)
    if trace_id:
        return trace_id
    
    # 2. HTTP 헤더 시도
    trace_id = extract_trace_id_from_http(log_line)
    if trace_id:
        return trace_id
    
    # 3. Spring Cloud Sleuth 시도
    trace_id = extract_trace_id_sleuth(log_line)
    if trace_id:
        return trace_id
    
    return None
```

---

### 3단계: 서비스별 세션화

#### ⚠️ 현재 사용 방식

**Trace ID가 없으므로 하이브리드 복합 키 기반 세션화를 사용합니다.**

**현재 방식:**
- Gateway: `client_ip` + `시간(초)` + `url` 기반 복합 키
- Manager: `스레드명` + `시간(초)` 기반 복합 키
- 각 복합 키별로 Sliding Window 적용

**자세한 내용은 `FINAL_PREPROCESSING_GUIDE.md` 참고!**

---

#### 목적
- 각 서비스의 로그를 세션 단위로 그룹화
- Trace ID 기반 또는 시간 기반 세션화

#### 3.1 Trace ID 기반 세션화 (향후 사용 가능)

**장점:**
- MSA 환경에 최적화
- 서비스 간 호출 관계 파악 가능
- 하나의 요청을 전체적으로 추적

**구현:**
```python
from collections import defaultdict
from datetime import datetime

class MSASessionizer:
    """MSA 환경용 세션화 클래스"""
    
    def __init__(self):
        self.trace_sessions = defaultdict(list)  # trace_id -> logs
        self.service_sessions = defaultdict(lambda: defaultdict(list))  # service -> trace_id -> logs
    
    def add_log(self, parsed_log: Dict, trace_id: Optional[str], service_name: str):
        """로그 추가"""
        if trace_id:
            # Trace ID 기반 세션화
            self.trace_sessions[trace_id].append({
                **parsed_log,
                'service': service_name,
                'trace_id': trace_id
            })
            self.service_sessions[service_name][trace_id].append(parsed_log)
        else:
            # Trace ID가 없으면 시간 기반 세션화
            self._add_time_based_session(parsed_log, service_name)
    
    def get_trace_sessions(self) -> List[Dict]:
        """Trace ID별 세션 반환"""
        sessions = []
        for trace_id, logs in self.trace_sessions.items():
            # 시간순 정렬
            logs.sort(key=lambda x: x.get('timestamp', ''))
            
            # 서비스별 그룹화
            service_groups = defaultdict(list)
            for log in logs:
                service_groups[log['service']].append(log)
            
            sessions.append({
                'trace_id': trace_id,
                'services': dict(service_groups),
                'all_logs': logs,
                'service_count': len(service_groups),
                'total_logs': len(logs)
            })
        
        return sessions
    
    def get_service_sessions(self, service_name: str) -> List[List[Dict]]:
        """특정 서비스의 세션 리스트 반환"""
        sessions = []
        for trace_id, logs in self.service_sessions[service_name].items():
            logs.sort(key=lambda x: x.get('timestamp', ''))
            sessions.append(logs)
        return sessions
```

#### 3.2 시간 기반 세션화 (Fallback)

**Trace ID가 없는 경우:**
```python
class TimeBasedSessionizer:
    """시간 기반 세션화 (기존 Sessionizer 확장)"""
    
    def __init__(self, window_size: int = 20, max_gap_seconds: int = 300):
        self.window_size = window_size
        self.max_gap_seconds = max_gap_seconds
        self.sliding_window = deque(maxlen=window_size)
        self.current_sessions = []
    
    def add_log(self, parsed_log: Dict, service_name: str):
        """로그 추가 (시간 기반)"""
        # 기존 Sessionizer 로직 사용
        # ...
```

---

### 4단계: MSA 컨텍스트 결합

#### 목적
- 서비스 간 호출 관계 파악
- 전체 요청 흐름 추적
- 의존성 정보 추가

#### 4.1 서비스 호출 그래프 생성

```python
class MSAContextBuilder:
    """MSA 컨텍스트 빌더"""
    
    def build_context(self, trace_session: Dict) -> Dict:
        """Trace 세션에 MSA 컨텍스트 추가"""
        services = trace_session['services']
        service_order = self._determine_service_order(services)
        
        context = {
            'trace_id': trace_session['trace_id'],
            'entry_service': service_order[0] if service_order else None,
            'service_chain': service_order,
            'service_count': len(services),
            'total_logs': trace_session['total_logs'],
            'has_error': self._check_errors(trace_session),
            'has_warn': self._check_warnings(trace_session),
            'services': services
        }
        
        return context
    
    def _determine_service_order(self, services: Dict) -> List[str]:
        """서비스 호출 순서 결정"""
        # Gateway가 보통 진입점
        if 'gateway' in services:
            order = ['gateway']
            # Gateway가 호출한 서비스 찾기
            gateway_logs = services['gateway']
            called_services = self._extract_called_services(gateway_logs)
            order.extend(called_services)
            return order
        
        # 시간순 정렬
        all_timestamps = []
        for service, logs in services.items():
            for log in logs:
                timestamp = self._extract_timestamp(log)
                if timestamp:
                    all_timestamps.append((timestamp, service))
        
        all_timestamps.sort()
        return [service for _, service in all_timestamps]
    
    def _extract_called_services(self, logs: List[Dict]) -> List[str]:
        """로그에서 호출된 서비스 추출"""
        called = []
        for log in logs:
            # URL 패턴에서 서비스명 추출
            # 예: /research/api/... -> research
            url = log.get('url', '') or log.get('original', '')
            match = re.search(r'/(research|manager|code|user|eureka)/', url, re.IGNORECASE)
            if match:
                service = match.group(1).lower()
                if service not in called:
                    called.append(service)
        return called
```

#### 4.2 의존성 정보 추가

```python
def add_dependency_info(session: Dict) -> Dict:
    """의존성 정보 추가"""
    services = session['services']
    
    # 서비스 간 호출 관계
    dependencies = {}
    for service, logs in services.items():
        called_services = set()
        for log in logs:
            # 로그에서 다른 서비스 호출 추출
            called = extract_called_services(log)
            called_services.update(called)
        dependencies[service] = list(called_services)
    
    session['dependencies'] = dependencies
    return session
```

---

### 5단계: 이상 탐지용 인코딩

#### 목적
- LogBERT 모델 입력 형식으로 변환
- 서비스별 또는 Trace별 인코딩

#### 5.1 서비스별 인코딩

```python
def encode_service_session(session: List[Dict], encoder: LogEncoder) -> Dict:
    """서비스 세션을 인코딩"""
    # Event ID 시퀀스 추출
    event_ids = [log.get('event_id', 0) for log in session]
    
    # 인코딩
    encoded = encoder.encode_sequence(session)
    
    return {
        'token_ids': encoded['token_ids'],
        'attention_mask': encoded['attention_mask'],
        'event_ids': event_ids,
        'service_name': session[0].get('service', 'unknown'),
        'session_length': len(session)
    }
```

#### 5.2 Trace별 인코딩

```python
def encode_trace_session(trace_session: Dict, encoder: LogEncoder) -> Dict:
    """전체 Trace를 하나의 시퀀스로 인코딩"""
    # 모든 서비스의 로그를 시간순으로 결합
    all_logs = []
    for service, logs in trace_session['services'].items():
        for log in logs:
            all_logs.append({
                **log,
                'service': service
            })
    
    # 시간순 정렬
    all_logs.sort(key=lambda x: x.get('timestamp', ''))
    
    # 인코딩
    encoded = encoder.encode_sequence(all_logs)
    
    return {
        'token_ids': encoded['token_ids'],
        'attention_mask': encoded['attention_mask'],
        'trace_id': trace_session['trace_id'],
        'services': list(trace_session['services'].keys()),
        'service_count': len(trace_session['services']),
        'total_logs': len(all_logs)
    }
```

---

### 6단계: RAG용 메타데이터 추출

#### 목적
- 벡터 DB에 저장할 메타데이터 추출
- 검색 및 가이드 제공을 위한 정보 준비

#### 6.1 에러 정보 추출

```python
def extract_error_info(session: Dict) -> Dict:
    """에러 정보 추출"""
    errors = []
    warnings = []
    
    for service, logs in session.get('services', {}).items():
        for log in logs:
            level = log.get('level', '').upper()
            template = log.get('template', '')
            original = log.get('original', '')
            
            if 'ERROR' in level or 'error' in template.lower():
                errors.append({
                    'service': service,
                    'timestamp': log.get('timestamp'),
                    'template': template,
                    'original': original,
                    'event_id': log.get('event_id')
                })
            
            if 'WARN' in level or 'warn' in template.lower():
                warnings.append({
                    'service': service,
                    'timestamp': log.get('timestamp'),
                    'template': template,
                    'original': original
                })
    
    return {
        'errors': errors,
        'warnings': warnings,
        'error_count': len(errors),
        'warning_count': len(warnings)
    }
```

#### 6.2 RAG용 텍스트 생성

```python
def generate_rag_text(session: Dict, error_info: Dict) -> str:
    """RAG 시스템용 텍스트 생성"""
    parts = []
    
    # Trace 정보
    parts.append(f"Trace ID: {session.get('trace_id', 'N/A')}")
    parts.append(f"Services: {', '.join(session.get('services', {}).keys())}")
    parts.append(f"Service Chain: {' -> '.join(session.get('service_chain', []))}")
    
    # 에러 정보
    if error_info['errors']:
        parts.append("\nErrors:")
        for error in error_info['errors']:
            parts.append(f"  [{error['service']}] {error['template']}")
            parts.append(f"    {error['original']}")
    
    # 경고 정보
    if error_info['warnings']:
        parts.append("\nWarnings:")
        for warn in error_info['warnings']:
            parts.append(f"  [{warn['service']}] {warn['template']}")
    
    # 서비스별 로그 요약
    parts.append("\nService Logs:")
    for service, logs in session.get('services', {}).items():
        parts.append(f"\n[{service}] ({len(logs)} logs):")
        # 주요 로그만 추출 (처음 5개)
        for log in logs[:5]:
            parts.append(f"  {log.get('template', '')}")
    
    return "\n".join(parts)
```

#### 6.3 메타데이터 구조

```python
def create_rag_metadata(session: Dict, error_info: Dict, anomaly_score: float) -> Dict:
    """RAG용 메타데이터 생성"""
    return {
        'trace_id': session.get('trace_id'),
        'services': list(session.get('services', {}).keys()),
        'service_chain': session.get('service_chain', []),
        'entry_service': session.get('entry_service'),
        'error_count': error_info['error_count'],
        'warning_count': error_info['warning_count'],
        'anomaly_score': anomaly_score,
        'severity': calculate_severity(error_info, anomaly_score),
        'timestamp': session.get('timestamp'),
        'dependencies': session.get('dependencies', {}),
        'rag_text': generate_rag_text(session, error_info)
    }
```

---

## 📊 전처리 결과 구조

### 이상 탐지용 데이터

```json
{
  "session_id": "gateway_trace_abc123",
  "token_ids": [101, 1, 2, 3, ..., 102],
  "attention_mask": [1, 1, 1, ..., 1, 0, 0],
  "service_name": "gateway",
  "trace_id": "abc123",
  "has_error": true,
  "has_warn": false
}
```

### RAG용 데이터

```json
{
  "trace_id": "abc123",
  "services": ["gateway", "research", "manager"],
  "service_chain": ["gateway", "research", "manager"],
  "error_count": 2,
  "warning_count": 1,
  "anomaly_score": 0.85,
  "severity": "high",
  "rag_text": "Trace ID: abc123\nServices: gateway, research, manager\n...",
  "errors": [
    {
      "service": "manager",
      "template": "ERROR Connection timeout",
      "original": "..."
    }
  ]
}
```

---

## 🔧 구현 예시

### 통합 전처리 스크립트

```python
from preprocessing.log_preprocessor import LogCleaner, LogParser
from msa_preprocessor import MSASessionizer, MSAContextBuilder

def preprocess_msa_logs(log_dir: Path, output_dir: Path):
    """MSA 로그 전처리"""
    
    # 초기화
    cleaner = LogCleaner()
    parser = LogParser()
    sessionizer = MSASessionizer()
    context_builder = MSAContextBuilder()
    
    # 로그 파일 처리
    for log_file in log_dir.glob('*.log'):
        service_name = extract_service_name(log_file)
        
        with open(log_file, 'r') as f:
            for line in f:
                # 1. 정리 및 파싱
                cleaned = cleaner.clean_log_line(line)
                if not cleaned:
                    continue
                
                parsed = parser.parse_log(cleaned)
                if not parsed:
                    continue
                
                # 2. Trace ID 추출
                trace_id = extract_trace_id(cleaned, service_name)
                
                # 3. 세션화
                sessionizer.add_log(parsed, trace_id, service_name)
    
    # 4. Trace 세션 추출
    trace_sessions = sessionizer.get_trace_sessions()
    
    # 5. MSA 컨텍스트 추가
    processed_sessions = []
    for trace_session in trace_sessions:
        context = context_builder.build_context(trace_session)
        processed_sessions.append(context)
    
    # 6. 저장
    save_preprocessed_data(processed_sessions, output_dir)
```

---

## 📝 현재 적용 방식 vs 이상적인 방식

### 현재 적용 방식 (Trace ID 없음)

**사용 중인 가이드:**
- ✅ `FINAL_PREPROCESSING_GUIDE.md`: 하이브리드 복합 키 기반

**방식:**
1. 서비스별 독립 세션화 (복합 키 + Sliding Window)
2. 메타데이터 추출 및 저장
3. 후처리 연결 (IP + URL + 시간 매칭)

**장점:**
- ✅ 현재 로그 환경에서 즉시 사용 가능
- ✅ 정확한 연결 가능 (IP + URL + 시간)
- ✅ 실용적으로 충분한 수준

**한계:**
- ⚠️ 완벽한 분산 추적은 어려움
- ⚠️ INSERT 로그가 없는 Manager 로그는 연결 어려움

---

### 이상적인 방식 (Trace ID 있음)

**이 가이드에서 설명한 방식:**
- Trace ID 기반 세션화
- 완벽한 분산 추적

**장점:**
- ✅ 완벽한 MSA 분산 추적
- ✅ 서비스 간 호출 관계 정확히 파악
- ✅ 하나의 요청을 전체적으로 추적

**전환 방법:**
1. 애플리케이션에 Trace ID 로깅 추가 (Spring Cloud Sleuth 등)
2. 모든 서비스에 적용
3. 이 가이드의 Trace ID 기반 방식으로 전환

---

## 📝 다음 단계

### 현재 (Trace ID 없음)

1. ✅ **하이브리드 복합 키 기반 전처리 구현** (완료)
2. ✅ **메타데이터 추출 및 저장** (완료)
3. ✅ **후처리 연결 구현** (완료)
4. **연결 정확도 검증 및 최적화**

**참고 문서:** `FINAL_PREPROCESSING_GUIDE.md`

---

### 향후 (Trace ID 추가 시)

1. **Trace ID 추출 로직 구현**
2. **MSA 세션화 클래스 구현**
3. **컨텍스트 빌더 구현**
4. **RAG 메타데이터 추출 구현**

**이 가이드를 바탕으로 Trace ID 기반 전처리로 전환하세요!** 🚀

---

## 🔗 관련 문서

- **`FINAL_PREPROCESSING_GUIDE.md`**: 현재 적용 중인 하이브리드 복합 키 기반 전처리 가이드 ⭐
- **`TRACE_ID_ALTERNATIVE_STRATEGY.md`**: Trace ID 대안 전략 (삭제됨, 내용은 FINAL 가이드에 통합)
- **`ADVANCED_PREPROCESSING_GUIDE.md`**: 고급 전처리 기법
