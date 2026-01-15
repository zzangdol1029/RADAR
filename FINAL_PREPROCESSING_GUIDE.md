# 최종 전처리 프로세스 가이드: 하이브리드 복합 키 기반 MSA 로그 연결

## 📋 개요

MSA 환경에서 Gateway → Manager 등으로 서비스 간 로그를 연결하는 하이브리드 복합 키 기반 전처리 프로세스입니다.

**핵심 전략:**
1. 서비스별 독립 세션화 (복합 키 + Sliding Window)
2. 메타데이터 추출 및 저장
3. 후처리 연결 (공통 식별자 매칭)
4. MSA 전체 흐름 파악

---

## 🏗️ 전체 아키텍처

### 전처리 파이프라인 흐름

```
원본 로그 파일들 (서비스별 + 날짜별)
    ↓
[1단계] 로그 정리 및 파싱
    ↓
[2단계] 서비스별 복합 키 생성
    ├── Gateway: client_ip + 시간(초) + url
    └── Manager: 스레드명 + 시간(초)
    ↓
[3단계] 하이브리드 세션화
    ├── 복합 키별 그룹화
    └── Sliding Window 적용 (크기/시간 제한)
    ↓
[4단계] 메타데이터 추출 및 저장
    ├── Gateway: client_ip, url, access_time
    └── Manager: user_ip_addr, url_addr (INSERT 로그에서)
    ↓
[5단계] 인코딩 및 토큰화
    ↓
[6단계] 후처리 연결 (MSA 서비스 간 연결)
    ├── IP + URL + 시간 매칭
    └── related_sessions 필드 추가
    ↓
전처리된 세션 데이터 (MSA 연결 정보 포함)
```

---

## 📁 로그 파일 조직화

### 디렉토리 구조

```
preprocessing/logs/
├── real_logs/              # 원본 로그 (서비스별)
│   ├── gateway_*.log
│   ├── manager_*.log
│   ├── code_*.log
│   ├── eureka_*.log
│   └── ...
│
└── date_split/            # 날짜별 분리 (서비스별 + 날짜별)
    ├── gateway_2025-08-13.log
    ├── manager_2025-08-13.log
    ├── code_2025-08-13.log
    ├── eureka_2025-08-13.log
    ├── gateway_2025-08-14.log
    ├── manager_2025-08-14.log
    └── ...
```

### 조직화 원칙

1. **서비스별 분리**: 각 서비스의 로그 패턴이 다르므로 분리
2. **날짜별 분리**: 메모리 효율성을 위해 날짜별 처리
3. **날짜별 배치 처리**: 같은 날짜의 모든 서비스 로그를 함께 읽어서 연결

---

## 🔄 단계별 전처리 프로세스

### 1단계: 로그 정리 및 파싱

#### 목적
- 의미 없는 데이터 제거
- 구조화된 로그 추출
- 템플릿 및 Event ID 생성

#### 처리 내용

**1.1 로그 정리 (Cleaning)**
```python
# Spring Boot 배너 제거
# 빈 줄 제거
# 특수 패턴 제거
cleaned_line = LogCleaner.clean_log_line(raw_line)
```

**1.2 로그 파싱 (Parsing)**
```python
# Drain3로 템플릿 추출
parsed_log = LogParser.parse_log(cleaned_line)
# 결과:
# {
#   'template': "ERROR Connection timeout from <*>",
#   'event_id': 1,
#   'parameters': ["192.168.0.1"],
#   'original': "원본 로그"
# }
```

**1.3 서비스명 추출**
```python
# 파일명에서 서비스명 추출
service_name = MetadataEnricher.extract_service_name(log_file_path)
# 예: "gateway_2025-08-13.log" → "gateway"
# 예: "manager_250813_17_32_23.log" → "manager"

parsed_log['service_name'] = service_name
```

---

### 2단계: 서비스별 복합 키 생성

#### 목적
- 서비스별 특성에 맞는 복합 키 생성
- 관련 로그들을 그룹화할 수 있는 식별자 생성

#### Gateway 로그 복합 키

**구성 요소:**
- `client_ip`: 클라이언트 IP 주소
- `시간(초)`: 접근 시간을 초 단위로 정규화
- `url`: 요청 URL

**생성 로직:**
```python
def extract_gateway_composite_key(log_data: Dict[str, Any]) -> Optional[str]:
    """Gateway 로그 복합 키 생성"""
    original = log_data.get('original', '')
    
    # JSON 형식 로그에서 추출
    json_match = re.search(r'\{[^}]+\}', original)
    if json_match:
        try:
            json_data = json.loads(json_match.group())
            
            client_ip = json_data.get('client_ip', 'unknown')
            access_time = json_data.get('access_time', '')
            url = json_data.get('url', 'unknown')
            
            # 시간을 초 단위로 정규화
            if access_time:
                # "2025-08-07T13:52:43.250215074+09:00[Asia/Seoul]" -> "13:52:43"
                time_part = access_time.split('T')[1].split('.')[0]
            else:
                timestamp = extract_timestamp(original)
                time_part = timestamp.strftime('%H:%M:%S') if timestamp else 'unknown'
            
            # Gateway 복합 키: client_ip + 시간 + url
            return f"{client_ip}_{time_part}_{url}"
        except:
            pass
    
    return None
```

**예시:**
```
원본: {"client_ip":"192.168.0.18","url":"/user/api/moduleMng/getModule","access_time":"2026-01-15T12:49:46.250"}
복합 키: "192.168.0.18_12:49:46_/user/api/moduleMng/getModule"
```

#### Manager 로그 복합 키

**구성 요소:**
- `service_name`: 서비스명 ("manager")
- `시간(초)`: 타임스탬프를 초 단위로 정규화
- `스레드명`: 스레드 이름 (예: "XNIO-1 task-1")

**생성 로직:**
```python
def extract_manager_composite_key(log_data: Dict[str, Any]) -> Optional[str]:
    """Manager 로그 복합 키 생성"""
    original = log_data.get('original', '')
    service_name = log_data.get('service_name', 'manager')
    
    timestamp = extract_timestamp(original)
    if not timestamp:
        return None
    
    time_part = timestamp.strftime('%H:%M:%S')
    
    # 스레드명 추출
    # 예: "2026-01-15 12:49:46.729 DEBUG org.hibernate.SQL --- [XNIO-1 task-1]"
    thread_pattern = r'---\s+\[([^\]]+)\]'
    thread_match = re.search(thread_pattern, original)
    thread_name = thread_match.group(1) if thread_match else 'unknown'
    
    # Manager 복합 키: service_name + 시간 + 스레드명
    return f"{service_name}_{time_part}_{thread_name}"
```

**예시:**
```
원본: "2026-01-15 12:49:46.729 DEBUG org.hibernate.SQL --- [XNIO-1 task-1]"
복합 키: "manager_12:49:46_XNIO-1 task-1"
```

#### 다른 서비스 로그 복합 키

**구성 요소:**
- `service_name`: 서비스명
- `시간(초)`: 타임스탬프를 초 단위로 정규화
- `로그_패턴`: 로그 레벨 또는 템플릿 패턴

**생성 로직:**
```python
def extract_other_service_composite_key(log_data: Dict[str, Any]) -> Optional[str]:
    """다른 서비스 로그 복합 키 생성"""
    original = log_data.get('original', '')
    service_name = log_data.get('service_name', 'unknown')
    
    timestamp = extract_timestamp(original)
    if not timestamp:
        return None
    
    time_part = timestamp.strftime('%H:%M:%S')
    
    # 로그 레벨 추출
    level_match = re.search(r'\b(ERROR|WARN|INFO|DEBUG|TRACE)\b', original)
    log_pattern = level_match.group(1) if level_match else 'default'
    
    # 복합 키: service_name + 시간 + 로그_패턴
    return f"{service_name}_{time_part}_{log_pattern}"
```

---

### 3단계: 하이브리드 세션화

#### 목적
- 복합 키로 관련 로그들을 그룹화
- Sliding Window로 세션 크기와 시간 제어

#### 하이브리드 방식 동작 원리

**3.1 복합 키별 그룹화**
```python
# 같은 복합 키를 가진 로그들을 하나의 그룹으로 묶음
composite_sessions = {
    "192.168.0.18_12:49:46_/user/api/moduleMng/getModule": {
        'logs': deque(maxlen=window_size),
        'start_time': timestamp
    },
    "manager_12:49:46_XNIO-1 task-1": {
        'logs': deque(maxlen=window_size),
        'start_time': timestamp
    }
}
```

**3.2 Sliding Window 적용**

**세션 완성 조건:**
1. **복합 키 변경**: 새로운 요청 시작
2. **크기 도달**: 윈도우에 최대 개수(기본 20개) 로그가 쌓임
3. **시간 초과**: 첫 로그부터 최대 시간(기본 300초) 경과

**구현:**
```python
def add_log_hybrid(self, log_data: Dict[str, Any]) -> List[List[Dict]]:
    """하이브리드 방식으로 로그 추가"""
    completed_sessions = []
    
    # 복합 키 추출
    composite_key = self.extract_composite_key(log_data)
    if not composite_key:
        composite_key = 'default'
    
    # 복합 키 변경 감지 (이전 키와 다른 경우 이전 세션 완성)
    if self.last_composite_key is not None and self.last_composite_key != composite_key:
        prev_session_info = self.composite_sessions[self.last_composite_key]
        if len(prev_session_info['logs']) > 0:
            completed_sessions.append(list(prev_session_info['logs']))
            prev_session_info['logs'].clear()
            prev_session_info['start_time'] = None
    
    # 현재 키의 세션 정보 가져오기
    session_info = self.composite_sessions[composite_key]
    
    # 타임스탬프 추출
    timestamp = self.extract_timestamp(log_data.get('original', ''))
    
    # 첫 로그인 경우 시작 시간 설정
    if session_info['start_time'] is None and timestamp:
        session_info['start_time'] = timestamp
    
    # Sliding Window 시간 체크
    if timestamp and session_info['start_time']:
        time_diff = (timestamp - session_info['start_time']).total_seconds()
        if time_diff > self.window_time:
            # 시간 초과로 세션 완성
            if len(session_info['logs']) > 0:
                completed_sessions.append(list(session_info['logs']))
            session_info['logs'].clear()
            session_info['start_time'] = timestamp
    
    # 로그 추가
    session_info['logs'].append(log_data)
    
    # Sliding Window 크기 체크
    if len(session_info['logs']) >= self.window_size:
        # 크기 도달로 세션 완성
        completed_sessions.append(list(session_info['logs']))
        session_info['logs'].clear()
        session_info['start_time'] = None
    
    # 현재 키 저장
    self.last_composite_key = composite_key
    
    return completed_sessions
```

---

### 4단계: 메타데이터 추출 및 저장

#### 목적
- 서비스 간 연결에 필요한 메타데이터 추출
- 세션에 연결 정보 저장

#### Gateway 세션 메타데이터

**추출 필드:**
- `client_ip`: 클라이언트 IP 주소
- `url`: 요청 URL
- `access_time`: 접근 시간
- `method`: HTTP 메서드
- `status`: HTTP 상태 코드

**구현:**
```python
def extract_gateway_metadata(log_data: Dict[str, Any]) -> Dict[str, Any]:
    """Gateway 로그에서 연결 가능한 메타데이터 추출"""
    original = log_data.get('original', '')
    json_match = re.search(r'\{[^}]+\}', original)
    
    if json_match:
        try:
            json_data = json.loads(json_match.group())
            return {
                'client_ip': json_data.get('client_ip'),
                'url': json_data.get('url'),
                'access_time': json_data.get('access_time'),
                'method': json_data.get('method'),
                'status': json_data.get('status')
            }
        except:
            pass
    
    return {}
```

#### Manager 세션 메타데이터

**추출 필드:**
- `user_ip_addr`: 사용자 IP 주소 (INSERT 로그에서)
- `url_addr`: API 엔드포인트 (INSERT 로그에서)
- `user_id`: 사용자 ID (INSERT 로그에서)
- `sys_log_sn`: 시스템 로그 시퀀스 번호 (INSERT 로그에서)
- `thread_name`: 스레드명 (모든 로그에서)

**구현:**
```python
def extract_manager_metadata(log_data: Dict[str, Any]) -> Dict[str, Any]:
    """Manager 로그에서 연결 가능한 메타데이터 추출"""
    original = log_data.get('original', '')
    metadata = {}
    
    # INSERT 로그에서 정보 추출
    # insert into bio.cs_sys_log_l 패턴 확인
    if 'insert' in original.lower() and 'bio.cs_sys_log_l' in original:
        # 세션 내에서 다음 로그들을 읽어서 파라미터 추출
        # binding parameter [9] as [VARCHAR] - [url_addr]
        # binding parameter [11] as [VARCHAR] - [user_id]
        # binding parameter [12] as [VARCHAR] - [user_ip_addr]
        # binding parameter [13] as [BIGINT] - [sys_log_sn]
        pass
    
    # 스레드명 추출 (모든 로그에서)
    thread_pattern = r'---\s+\[([^\]]+)\]'
    thread_match = re.search(thread_pattern, original)
    if thread_match:
        metadata['thread_name'] = thread_match.group(1)
    
    return metadata
```

**세션 메타데이터 저장:**
```python
# 세션화 후 메타데이터 추가
enriched_session = {
    'service_name': 'gateway',
    'composite_key': '192.168.0.18_12:49:46_/user/api/moduleMng/getModule',
    'correlation_metadata': {
        'client_ip': '192.168.0.18',
        'url': '/user/api/moduleMng/getModule',
        'access_time': '2026-01-15T12:49:46.250',
        'method': 'GET',
        'status': 200
    },
    'event_sequence': [1, 5, 12, ...],
    'logs': [...],
    ...
}
```

---

### 5단계: 인코딩 및 토큰화

#### 목적
- 모델 입력 형식으로 변환
- BERT 스타일 토큰화

#### 처리 과정

**5.1 Event ID 시퀀스 추출**
```python
event_sequence = [log['event_id'] for log in session['logs']]
# 예: [1, 5, 1, 12, 3, ...]
```

**5.2 Token ID 매핑**
```python
# Event ID → Token ID 매핑
token_ids = [event_to_token[event_id] for event_id in event_sequence]
```

**5.3 Special Tokens 추가**
```python
# BERT 스타일: [CLS] + tokens + [SEP]
token_ids = [CLS_TOKEN_ID] + token_ids + [SEP_TOKEN_ID]
```

**5.4 Padding**
```python
# max_seq_length = 256
if len(token_ids) < max_seq_length:
    token_ids = token_ids + [PAD_TOKEN_ID] * (max_seq_length - len(token_ids))
else:
    token_ids = token_ids[:max_seq_length]
```

**5.5 Attention Mask 생성**
```python
attention_mask = [1 if token != PAD_TOKEN_ID else 0 for token in token_ids]
```

---

### 6단계: 후처리 연결 (MSA 서비스 간 연결)

#### 목적
- Gateway와 Manager 등 서비스 간 세션 연결
- MSA 전체 흐름 파악

#### 연결 방식: 두 가지 접근법

**⚠️ 중요: Sliding Window와는 다른 방식입니다!**

현재 가이드에서는 **시간 근접성 기반 매칭**을 사용하지만, 실제 구현에서는 **시간 윈도우 기반 그룹화**도 가능합니다.

---

#### 방식 1: 시간 근접성 기반 매칭 (정확한 매칭)

**특징:**
- Sliding Window가 아닌 **특정 시간점 기준 ±5초 범위** 내에서 매칭
- IP + URL + 시간을 모두 일치시켜 정확한 연결

**매칭 조건:**
1. **IP 매칭**: Gateway의 `client_ip` = Manager의 `user_ip_addr`
2. **URL 매칭**: Gateway의 `url` = Manager의 `url_addr`
3. **시간 근접성**: 시간 차이가 5초 이내

**동작 원리:**
```
Gateway 세션:
  access_time: "2026-01-15T12:49:46.250"
  → 시간 범위: 12:49:46.250 ± 5초 = 12:49:41.250 ~ 12:49:51.250

Manager 세션:
  timestamp: "2026-01-15T12:49:46.729"
  → 이 시간이 Gateway의 시간 범위 내에 있으면 매칭!
```

**구현:**
```python
def build_msa_correlation(sessions_by_service: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """MSA 환경에서 서비스 간 세션 연결 (시간 근접성 기반)"""
    correlations = {}
    
    gateway_sessions = sessions_by_service.get('gateway', [])
    manager_sessions = sessions_by_service.get('manager', [])
    
    for gw_session in gateway_sessions:
        gw_meta = gw_session.get('correlation_metadata', {})
        gw_ip = gw_meta.get('client_ip')
        gw_url = gw_meta.get('url')
        gw_time = gw_meta.get('access_time')
        
        if not all([gw_ip, gw_url, gw_time]):
            continue
        
        gw_timestamp = parse_timestamp(gw_time)
        related = []
        
        for mgr_session in manager_sessions:
            # Manager INSERT 로그에서 추출한 메타데이터 확인
            mgr_meta = mgr_session.get('correlation_metadata', {})
            mgr_ip = mgr_meta.get('user_ip_addr')
            mgr_url = mgr_meta.get('url_addr')
            mgr_time = mgr_session.get('timestamp')
            
            if not all([mgr_ip, mgr_url, mgr_time]):
                continue
            
            mgr_timestamp = parse_timestamp(mgr_time)
            time_diff = abs((gw_timestamp - mgr_timestamp).total_seconds())
            
            # 매칭 조건: IP + URL + 시간 (5초 이내)
            if (gw_ip == mgr_ip and 
                gw_url == mgr_url and 
                time_diff < 5):
                match_score = 1.0 - (time_diff / 5.0)  # 시간 차이에 따른 점수
                related.append({
                    'service': 'manager',
                    'session_id': mgr_session['session_id'],
                    'match_score': match_score,
                    'match_reason': f'IP+URL+시간 매칭 (차이: {time_diff:.2f}초)'
                })
        
        if related:
            correlations[gw_session['session_id']] = related
            gw_session['related_sessions'] = related
    
    return correlations
```

---

#### 방식 2: 시간 윈도우 기반 그룹화 (대략적인 매칭)

**특징:**
- **고정된 시간 윈도우(5분 단위)** 내의 모든 세션을 그룹화
- Sliding Window와 유사하지만, 연속적인 슬라이딩이 아닌 **고정 윈도우**

**동작 원리:**
```
시간 윈도우: 2025-12-08_14_05 (14:05 ~ 14:10, 5분 단위)

이 시간 윈도우 내의 모든 세션:
  - Gateway 세션 (14:06:23)
  - Manager 세션 (14:07:15)
  - Code 세션 (14:08:42)
  
→ 모두 같은 시간 윈도우에 속하므로 관련이 있을 가능성이 높음
```

**구현:**
```python
def build_msa_correlation_time_window(sessions_by_service: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """MSA 환경에서 서비스 간 세션 연결 (시간 윈도우 기반)"""
    # 1단계: 시간 윈도우별로 세션 그룹화 (5분 단위)
    time_window_sessions = defaultdict(lambda: defaultdict(list))
    
    for service_name, sessions in sessions_by_service.items():
        for session in sessions:
            timestamp = parse_timestamp(session.get('timestamp') or session.get('start_time'))
            if timestamp:
                # 5분 단위 시간 윈도우 생성
                window_key = f"{timestamp.strftime('%Y-%m-%d')}_{timestamp.hour:02d}_{timestamp.minute // 5:02d}"
                time_window_sessions[window_key][service_name].append(session)
    
    # 2단계: 같은 시간 윈도우의 세션들을 연결
    correlations = {}
    for window_key, services_sessions in time_window_sessions.items():
        # 같은 시간 윈도우 내의 모든 서비스 세션
        all_services = list(services_sessions.keys())
        
        for service_name, sessions in services_sessions.items():
            for session in sessions:
                related = []
                for other_service in all_services:
                    if other_service != service_name:
                        for other_session in services_sessions[other_service]:
                            related.append({
                                'service': other_service,
                                'session_id': other_session['session_id'],
                                'match_reason': f'시간 윈도우 매칭 ({window_key})'
                            })
                
                if related:
                    session['related_sessions'] = related
                    correlations[session['session_id']] = related
    
    return correlations
```

---

#### 두 방식의 비교

| 방식 | 시간 범위 | 정확도 | 적용 시나리오 |
|------|----------|--------|--------------|
| **시간 근접성** | ±5초 | ⭐⭐⭐⭐⭐ | IP+URL+시간이 모두 일치하는 정확한 연결 |
| **시간 윈도우** | 5분 단위 | ⭐⭐⭐ | 같은 시간대의 모든 세션을 그룹화 |

**권장:**
- **정확한 연결**: 시간 근접성 기반 (방식 1) - IP + URL + 시간 매칭
- **대략적인 그룹화**: 시간 윈도우 기반 (방식 2) - 같은 시간대의 모든 세션

---

#### 현재 가이드의 방식

**사용하는 방식: 시간 근접성 기반 매칭**

**이유:**
- ✅ IP + URL + 시간을 모두 일치시켜 정확한 연결
- ✅ Sliding Window와 달리 특정 시간점 기준으로 매칭
- ✅ MSA 환경에서 정확한 요청 추적 가능

**Sliding Window와의 차이:**
- **Sliding Window**: 연속적인 시간 윈도우를 슬라이딩하면서 처리 (세션화 단계)
- **시간 근접성 매칭**: 특정 시간점 기준 ±5초 범위 내에서 매칭 (연결 단계)

---

## 📊 최종 출력 형식

### 세션 데이터 구조

```json
{
  "session_id": 12345,
  "service_name": "gateway",
  "composite_key": "gateway_192.168.0.18_12:49:46_/user/api/moduleMng/getModule",
  "correlation_metadata": {
    "client_ip": "192.168.0.18",
    "url": "/user/api/moduleMng/getModule",
    "access_time": "2026-01-15T12:49:46.250",
    "method": "GET",
    "status": 200
  },
  "related_sessions": [
    {
      "service": "manager",
      "session_id": 67890,
      "match_score": 0.95,
      "match_reason": "IP+URL+시간 매칭 (차이: 0.48초)"
    }
  ],
  "event_sequence": [1, 5, 12, 3, 8],
  "token_ids": [101, 1, 5, 12, 3, 8, 102, 0, 0, ...],
  "attention_mask": [1, 1, 1, 1, 1, 1, 1, 0, 0, ...],
  "logs": [
    {
      "event_id": 1,
      "template": "GET request to <*>",
      "parameters": ["/user/api/moduleMng/getModule"],
      "timestamp": "2026-01-15T12:49:46.250",
      "service_name": "gateway"
    },
    ...
  ],
  "has_error": false,
  "has_warn": false,
  "start_time": "2026-01-15T12:49:46.250",
  "end_time": "2026-01-15T12:49:46.750",
  "duration_seconds": 0.5
}
```

---

## ⚙️ 설정 파일

### preprocessing_config.yaml

```yaml
# 로그 디렉토리 경로 (날짜별 분리된 파일 위치)
log_directory: "logs/date_split"

# 출력 디렉토리
output_directory: "output"

# 출력 파일 경로
output_path: "preprocessed_logs.json"

# 세션화 방법: "hybrid" (하이브리드 방식)
sessionization_method: "hybrid"

# 하이브리드 방식 설정
hybrid_composite_key: true       # 복합 키 사용
hybrid_window_size: 20           # Sliding Window 크기 (로그 개수)
hybrid_window_time: 300          # Sliding Window 시간 (초, 5분)

# 복합 키 구성 요소
composite_key_fields:
  gateway:
    - client_ip
    - access_time_second
    - url
  manager:
    - service_name
    - time_second
    - thread_name
  default:
    - service_name
    - time_second
    - log_pattern

# 인코딩 설정
max_seq_length: 256              # 최대 시퀀스 길이

# Drain3 설정 파일 경로
drain3_config_path: "drain3_config.yaml"

# 메모리 효율성 설정
stream_mode: true                # 스트리밍 모드 (세션 완성 시 즉시 파일에 저장)
enable_correlation: true         # MSA 서비스 간 관계 추적 활성화
batch_by_date: true              # 날짜별 배치 처리 (메모리 효율적)
date_filter: null                 # 날짜 필터 (YYYY-MM-DD 형식, null이면 전체 처리)

# MSA 연결 설정
msa_correlation:
  enabled: true                  # MSA 서비스 간 연결 활성화
  time_window: 5                 # 시간 윈도우 (초, 5초 이내)
  match_fields:                  # 매칭 필드
    gateway_to_manager:
      - client_ip: user_ip_addr
      - url: url_addr
      - access_time: timestamp

# 병렬 처리 설정
parallel: true                   # 병렬 처리 활성화 (날짜별 동시 처리)
max_workers: 4                   # 최대 동시 처리 프로세스 수
```

---

## 🚀 실행 방법

### 1단계: 로그 파일 준비

```bash
# 원본 로그 확인
ls preprocessing/logs/real_logs/

# 날짜별 분리 (이미 완료된 경우 생략)
cd preprocessing
python split_logs_by_date.py \
  --input logs/real_logs \
  --output logs/date_split
```

### 2단계: 설정 파일 확인

```bash
cd preprocessing
# preprocessing_config.yaml 편집
# sessionization_method: "hybrid" 확인
# enable_correlation: true 확인
# msa_correlation.enabled: true 확인
```

### 3단계: 전처리 실행

```bash
cd preprocessing
python log_preprocessor.py \
  --log-dir logs/date_split \
  --config preprocessing_config.yaml
```

### 4단계: 결과 확인

```bash
# 출력 파일 확인
ls output/preprocessed_logs_*.json

# 세션 수 및 연결 정보 확인
python -c "
import json
with open('output/preprocessed_logs_2025-08-13.json', 'r') as f:
    data = json.load(f)
    print(f'총 세션 수: {len(data)}')
    
    # Gateway 세션 중 연결된 세션 확인
    gateway_sessions = [s for s in data if s.get('service_name') == 'gateway']
    connected_sessions = [s for s in gateway_sessions if s.get('related_sessions')]
    
    print(f'Gateway 세션 수: {len(gateway_sessions)}')
    print(f'연결된 Gateway 세션 수: {len(connected_sessions)}')
    
    # 연결 예시 출력
    if connected_sessions:
        print(f'\n연결 예시:')
        session = connected_sessions[0]
        print(f'  Gateway 세션 ID: {session[\"session_id\"]}')
        print(f'  연결된 Manager 세션: {session[\"related_sessions\"]}')
"
```

---

## 📈 성능 및 최적화

### 메모리 사용량

**하이브리드 방식:**
- 복합 키별로 세션 저장
- 각 세션은 최대 20개 로그만 유지
- 메모리 효율적

**예상 메모리:**
- 복합 키 수: 약 1,000개 (동시 요청 수)
- 세션당 로그: 최대 20개
- 총 메모리: 약 100-200MB

### 처리 속도

**하이브리드 방식:**
- 복합 키 생성: O(1)
- 세션 추가: O(1)
- 세션 완성 체크: O(1)
- 전체 처리: O(n) (n = 로그 수)

**후처리 연결:**
- 시간 복잡도: O(m × k) (m = Gateway 세션 수, k = Manager 세션 수)
- 최적화: 시간 윈도우 기반 인덱싱으로 O(m × log(k)) 가능

---

## ✅ 검증 체크리스트

### 전처리 전 확인

- [ ] `logs/date_split/` 폴더에 날짜별 + 서비스별 파일 존재
- [ ] 파일명 형식: `service_YYYY-MM-DD.log`
- [ ] Gateway 로그에 `client_ip`, `url`, `access_time` 필드 확인
- [ ] Manager 로그에 INSERT 로그 패턴 확인

### 설정 확인

- [ ] `sessionization_method: "hybrid"` 설정
- [ ] `enable_correlation: true` 설정
- [ ] `msa_correlation.enabled: true` 설정
- [ ] `batch_by_date: true` 설정
- [ ] `stream_mode: true` 설정 (메모리 효율성)

### 결과 확인

- [ ] 복합 키별 세션이 생성되었는지 확인
- [ ] 세션 크기가 제한되었는지 확인 (최대 20개)
- [ ] 세션 시간이 제한되었는지 확인 (최대 300초)
- [ ] Gateway와 Manager 세션이 연결되었는지 확인
- [ ] `related_sessions` 필드가 올바르게 채워졌는지 확인

---

## 🎯 장점 및 특징

### 1. 서비스별 최적화

**Gateway:**
- `client_ip` + `시간` + `url` 기반 복합 키
- 정확한 요청 단위 세션화

**Manager:**
- `스레드명` + `시간` 기반 복합 키
- 동시 요청 구분 가능

### 2. MSA 환경 지원

**서비스 간 연결:**
- Gateway → Manager 연결 가능
- IP + URL + 시간 매칭
- 전체 요청 흐름 파악

### 3. 메모리 효율성

**하이브리드 방식:**
- 복합 키별 세션 관리
- Sliding Window로 크기 제한
- 스트리밍 모드로 즉시 저장

### 4. 유연한 세션 완성

**여러 조건으로 세션 완성:**
- 복합 키 변경 (새로운 요청 시작)
- 크기 도달 (20개 로그)
- 시간 초과 (300초 경과)

---

## ⚠️ 주의사항 및 한계

### 1. Manager INSERT 로그 의존성

**문제:**
- Manager의 `user_ip_addr`, `url_addr`는 INSERT 로그에서만 추출 가능
- INSERT 로그가 없는 경우 연결 어려움

**해결:**
- INSERT 로그가 있는 세션만 연결
- 나머지는 독립적으로 처리

### 2. 시간 근접성 기반 매칭

**문제:**
- 완벽한 추적은 어려움
- 같은 시간대의 다른 요청과 혼동 가능

**해결:**
- IP + URL + 시간 조합으로 정확도 향상
- 실용적으로 충분한 수준

### 3. 다른 스레드의 로그 분리

**문제:**
- Manager의 `[XNIO-1 task-1]`과 `[audit-1]`이 별도 세션

**해결:**
- 시간이 가까우면 Sliding Window로 연결 가능
- 또는 후처리 연결 단계에서 시간 기반으로 연결

---

## 📝 결론

### 최종 전처리 프로세스

**하이브리드 복합 키 기반 MSA 로그 연결:**

1. ✅ 서비스별 독립 세션화 (복합 키 + Sliding Window)
2. ✅ 메타데이터 추출 및 저장
3. ✅ 후처리 연결 (IP + URL + 시간 매칭)
4. ✅ MSA 전체 흐름 파악

**결과:**
- ✅ 각 서비스의 세션화 로직 유지
- ✅ MSA 환경에서 전체 흐름 파악 가능
- ✅ 이상 탐지 시 서비스 간 영향 분석 가능
- ✅ 메모리 효율적이고 확장 가능

**다음 단계:**
1. 전처리 실행 및 결과 확인
2. 연결 정확도 검증
3. 이상 탐지 모델 학습 준비

**현재 전처리 파이프라인이 이 방식을 완벽하게 지원합니다!** 🚀
