# 고급 전처리 프로세스 가이드

## 📋 개요

이 문서는 **LogBERT, DeepLog, LogLSTM, LogTCN** 모델의 최고 성능을 위한 상세하고 정확도가 높은 전처리 프로세스를 설명합니다.

### 목표
- ✅ **최고 정확도**: 모델 성능 최대화
- ✅ **상세한 특징 추출**: 시간적 패턴, 서비스 의존성, 에러 전파 등
- ✅ **데이터 품질 최우선**: 노이즈 제거, 이상치 처리, 데이터 검증
- ✅ **모델별 최적화**: 각 모델의 특성에 맞는 전처리

### 원칙
- **데이터는 최대한 활용**: 가능한 모든 로그 데이터 사용
- **전처리 시간/자원 무제한**: 정확도 우선
- **복잡도 허용**: 정교한 알고리즘 사용

---

## 🏗️ 전체 전처리 파이프라인

```
원본 로그 파일들 (서비스별)
    ↓
[1단계] 로그 정리 및 고급 파싱
    ↓
[2단계] Trace ID 추출 및 검증
    ↓
[3단계] 다중 세션화 전략
    ↓
[4단계] 고급 특징 추출
    ↓
[5단계] MSA 컨텍스트 빌딩
    ↓
[6단계] 모델별 최적화 인코딩
    ↓
[7단계] 데이터 품질 검증 및 필터링
    ↓
[8단계] 데이터 증강 (선택)
    ↓
전처리된 데이터 (모델별 최적화)
```

---

## 📝 단계별 상세 전처리 방법

### 1단계: 로그 정리 및 고급 파싱

#### 1.1 다층 로그 정리 (Multi-layer Cleaning)

**목적**: 모든 노이즈와 불필요한 데이터 제거

```python
class AdvancedLogCleaner:
    """고급 로그 정리 클래스"""
    
    def __init__(self):
        # Spring Boot 배너 패턴 (확장)
        self.banner_patterns = [
            r'\.\s+____.*?Spring Boot.*?::',
            r':: Spring Boot ::',
            r'-----------------------------------------------------------------------------------------',
            r'Started .*? in \d+\.\d+ seconds',
            r'Running Spring Boot',
            r'Spring Boot version',
        ]
        
        # 빈 줄 및 공백 패턴
        self.empty_patterns = [
            r'^\s*$',  # 빈 줄
            r'^\s+$',  # 공백만
            r'^---+$',  # 구분선
        ]
        
        # 불필요한 로그 패턴
        self.noise_patterns = [
            r'^DEBUG.*?org\.springframework\.',  # Spring 내부 디버그
            r'^TRACE.*?',  # TRACE 레벨 (너무 상세)
            r'^.*?\.(jar|class) loaded$',  # 클래스 로딩
        ]
    
    def clean_log_line(self, line: str) -> Optional[str]:
        """다층 정리"""
        # 1. 인코딩 정규화
        line = self._normalize_encoding(line)
        
        # 2. 배너 제거
        for pattern in self.banner_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return None
        
        # 3. 빈 줄 제거
        for pattern in self.empty_patterns:
            if re.match(pattern, line):
                return None
        
        # 4. 노이즈 패턴 제거
        for pattern in self.noise_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return None
        
        # 5. 앞뒤 공백 제거
        line = line.strip()
        
        # 6. 최소 길이 검증 (너무 짧은 로그 제거)
        if len(line) < 10:
            return None
        
        return line
    
    def _normalize_encoding(self, line: str) -> str:
        """인코딩 정규화"""
        # UTF-8로 통일
        try:
            line = line.encode('utf-8', errors='ignore').decode('utf-8')
        except:
            pass
        
        # 특수 문자 정규화
        line = line.replace('\x00', '')  # NULL 문자 제거
        line = line.replace('\r\n', '\n')  # 줄바꿈 통일
        
        return line
```

#### 1.2 고급 로그 파싱 (Drain3 최적화)

**목적**: 정확한 템플릿 추출 및 파라미터 분리

```python
class AdvancedLogParser:
    """고급 로그 파서 (Drain3 최적화)"""
    
    def __init__(self, drain3_config_path: str):
        # Drain3 설정 최적화
        self.drain3_config = {
            'depth': 4,  # 트리 깊이 증가 (더 정교한 파싱)
            'st': 0.5,  # 유사도 임계값 (낮추면 더 세밀하게)
            'max_children': 100,  # 최대 자식 노드 수 증가
            'max_clusters': None,  # 클러스터 수 제한 없음
        }
        
        self.parser = Drain3(
            config=self.drain3_config,
            persistence_handler=FilePersistenceHandler(drain3_config_path)
        )
        
        # Event ID 매핑
        self.event_id_map = {}  # template -> event_id
        self.next_event_id = 1
    
    def parse_log(self, log_line: str, service_name: str) -> Optional[Dict]:
        """고급 파싱"""
        try:
            # Drain3 파싱
            result = self.parser.parse(log_line)
            
            if not result:
                return None
            
            template = result.get('template', '')
            parameters = result.get('parameters', [])
            
            # Event ID 할당
            if template not in self.event_id_map:
                self.event_id_map[template] = self.next_event_id
                self.next_event_id += 1
            
            event_id = self.event_id_map[template]
            
            # 타임스탬프 추출 (정교하게)
            timestamp = self._extract_timestamp_advanced(log_line)
            
            # 로그 레벨 추출 (정교하게)
            level = self._extract_level_advanced(log_line)
            
            # 추가 메타데이터 추출
            metadata = self._extract_metadata(log_line, service_name)
            
            return {
                'original': log_line,
                'template': template,
                'parameters': parameters,
                'event_id': event_id,
                'timestamp': timestamp,
                'level': level,
                'service_name': service_name,
                'metadata': metadata,
                'parameter_count': len(parameters),
                'template_length': len(template),
            }
        
        except Exception as e:
            logger.debug(f"파싱 실패: {e}")
            return None
    
    def _extract_timestamp_advanced(self, log_line: str) -> Optional[datetime]:
        """고급 타임스탬프 추출"""
        # 다양한 타임스탬프 형식 지원
        timestamp_patterns = [
            # Spring Boot 표준 형식
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})',
            # ISO 8601 형식
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})',
            # Unix 타임스탬프
            r'(\d{10}\.\d{3})',
        ]
        
        for pattern in timestamp_patterns:
            match = re.search(pattern, log_line)
            if match:
                timestamp_str = match.group(1)
                try:
                    # Spring Boot 형식
                    if '.' in timestamp_str and 'T' not in timestamp_str:
                        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                    # ISO 8601 형식
                    elif 'T' in timestamp_str:
                        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    # Unix 타임스탬프
                    else:
                        return datetime.fromtimestamp(float(timestamp_str))
                except:
                    continue
        
        return None
    
    def _extract_level_advanced(self, log_line: str) -> str:
        """고급 로그 레벨 추출"""
        # 정규식 패턴 (순서 중요)
        level_patterns = [
            (r'\bERROR\b', 'ERROR'),
            (r'\bWARN\b', 'WARN'),
            (r'\bWARNING\b', 'WARN'),
            (r'\bINFO\b', 'INFO'),
            (r'\bDEBUG\b', 'DEBUG'),
            (r'\bTRACE\b', 'TRACE'),
        ]
        
        for pattern, level in level_patterns:
            if re.search(pattern, log_line, re.IGNORECASE):
                return level
        
        return 'UNKNOWN'
    
    def _extract_metadata(self, log_line: str, service_name: str) -> Dict:
        """추가 메타데이터 추출"""
        metadata = {}
        
        # 스레드명 추출
        thread_match = re.search(r'\[([^\]]+)\]', log_line)
        if thread_match:
            metadata['thread'] = thread_match.group(1)
        
        # HTTP 메서드 추출
        http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
        for method in http_methods:
            if method in log_line:
                metadata['http_method'] = method
                break
        
        # HTTP 상태 코드 추출
        status_match = re.search(r'status[=:](\d{3})', log_line, re.IGNORECASE)
        if status_match:
            metadata['http_status'] = int(status_match.group(1))
        
        # IP 주소 추출
        ip_match = re.search(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', log_line)
        if ip_match:
            metadata['ip_address'] = ip_match.group(1)
        
        # URL 추출
        url_match = re.search(r'(https?://[^\s]+|/[^\s]+)', log_line)
        if url_match:
            metadata['url'] = url_match.group(1)
        
        return metadata
```

---

### 2단계: Trace ID 추출 및 검증

#### 2.1 다중 방법 Trace ID 추출

**목적**: 모든 가능한 방법으로 Trace ID 추출

```python
class AdvancedTraceExtractor:
    """고급 Trace ID 추출기"""
    
    def __init__(self):
        # Trace ID 패턴 (확장)
        self.trace_patterns = [
            # JSON 형식
            (r'"trace_id"\s*:\s*"([^"]+)"', 'json'),
            (r'"traceId"\s*:\s*"([^"]+)"', 'json'),
            (r'"X-Trace-Id"\s*:\s*"([^"]+)"', 'json'),
            (r'"correlationId"\s*:\s*"([^"]+)"', 'json'),
            
            # HTTP 헤더 형식
            (r'X-Trace-Id[:\s]+([a-zA-Z0-9-]+)', 'http'),
            (r'trace_id[:\s]+([a-zA-Z0-9-]+)', 'http'),
            (r'traceId[:\s]+([a-zA-Z0-9-]+)', 'http'),
            
            # Spring Cloud Sleuth 형식
            (r'\[([a-zA-Z0-9-]+),', 'sleuth'),
            (r'\[([a-zA-Z0-9]{16,32})\]', 'sleuth'),
            
            # UUID 형식
            (r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', 'uuid'),
            
            # 16진수 형식 (32자)
            (r'([0-9a-f]{32})', 'hex'),
            
            # 숫자 형식 (긴 숫자)
            (r'\b(\d{16,32})\b', 'numeric'),
        ]
    
    def extract_trace_id(self, log_line: str, service_name: str) -> Optional[Dict]:
        """다중 방법으로 Trace ID 추출 및 검증"""
        candidates = []
        
        # 모든 패턴 시도
        for pattern, source in self.trace_patterns:
            matches = re.finditer(pattern, log_line, re.IGNORECASE)
            for match in matches:
                trace_id = match.group(1)
                
                # Trace ID 검증
                if self._validate_trace_id(trace_id):
                    candidates.append({
                        'trace_id': trace_id,
                        'source': source,
                        'confidence': self._calculate_confidence(trace_id, source),
                        'position': match.start()
                    })
        
        if not candidates:
            return None
        
        # 가장 신뢰도 높은 Trace ID 선택
        best = max(candidates, key=lambda x: x['confidence'])
        
        return {
            'trace_id': best['trace_id'],
            'source': best['source'],
            'confidence': best['confidence'],
            'all_candidates': candidates  # 모든 후보 저장
        }
    
    def _validate_trace_id(self, trace_id: str) -> bool:
        """Trace ID 유효성 검증"""
        # 너무 짧거나 길면 제외
        if len(trace_id) < 8 or len(trace_id) > 64:
            return False
        
        # 특수 문자 제외 (일부 허용)
        if re.search(r'[^a-zA-Z0-9\-_]', trace_id):
            return False
        
        return True
    
    def _calculate_confidence(self, trace_id: str, source: str) -> float:
        """신뢰도 계산"""
        confidence = 0.5  # 기본값
        
        # 소스별 가중치
        source_weights = {
            'json': 1.0,
            'sleuth': 0.9,
            'http': 0.8,
            'uuid': 0.9,
            'hex': 0.7,
            'numeric': 0.6,
        }
        confidence *= source_weights.get(source, 0.5)
        
        # 길이별 가중치 (16-32자일 때 최고)
        length = len(trace_id)
        if 16 <= length <= 32:
            confidence *= 1.0
        elif 8 <= length < 16 or 32 < length <= 64:
            confidence *= 0.9
        else:
            confidence *= 0.7
        
        # 형식별 가중치 (UUID 형식이면 높음)
        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', trace_id, re.IGNORECASE):
            confidence *= 1.1
        
        return min(confidence, 1.0)
```

#### 2.2 Trace ID 연결 및 검증

```python
class TraceValidator:
    """Trace ID 검증 및 연결 클래스"""
    
    def __init__(self):
        self.trace_registry = {}  # trace_id -> metadata
        self.service_traces = defaultdict(set)  # service -> trace_ids
    
    def register_trace(self, trace_id: str, service_name: str, timestamp: datetime):
        """Trace ID 등록"""
        if trace_id not in self.trace_registry:
            self.trace_registry[trace_id] = {
                'services': set(),
                'first_seen': timestamp,
                'last_seen': timestamp,
                'log_count': 0,
            }
        
        self.trace_registry[trace_id]['services'].add(service_name)
        self.trace_registry[trace_id]['last_seen'] = max(
            self.trace_registry[trace_id]['last_seen'],
            timestamp
        )
        self.trace_registry[trace_id]['log_count'] += 1
        self.service_traces[service_name].add(trace_id)
    
    def validate_trace(self, trace_id: str) -> Dict:
        """Trace ID 검증"""
        if trace_id not in self.trace_registry:
            return {
                'valid': False,
                'reason': 'not_found'
            }
        
        metadata = self.trace_registry[trace_id]
        
        # 검증 규칙
        validations = {
            'valid': True,
            'service_count': len(metadata['services']),
            'log_count': metadata['log_count'],
            'duration': (metadata['last_seen'] - metadata['first_seen']).total_seconds(),
            'services': list(metadata['services']),
        }
        
        # 이상치 검출
        if validations['service_count'] > 10:  # 너무 많은 서비스
            validations['warning'] = 'too_many_services'
        
        if validations['duration'] > 3600:  # 1시간 이상
            validations['warning'] = 'too_long_duration'
        
        return validations
```

---

### 3단계: 다중 세션화 전략

#### 3.1 하이브리드 세션화

**목적**: Trace ID 기반 + 시간 기반 + 서비스 기반 세션화

```python
class HybridSessionizer:
    """하이브리드 세션화 클래스"""
    
    def __init__(
        self,
        trace_window_time: int = 600,  # 10분
        service_window_size: int = 50,
        service_window_time: int = 300,  # 5분
        sliding_window_size: int = 20,
        sliding_window_time: int = 180,  # 3분
    ):
        # Trace ID 기반 세션
        self.trace_sessions = defaultdict(list)  # trace_id -> logs
        
        # 서비스별 세션 (Trace ID 없을 때)
        self.service_sessions = defaultdict(lambda: deque(maxlen=service_window_size))
        
        # Sliding Window 세션
        self.sliding_windows = defaultdict(lambda: deque(maxlen=sliding_window_size))
        
        # 시간 윈도우
        self.trace_window_time = trace_window_time
        self.service_window_time = service_window_time
        self.sliding_window_time = sliding_window_time
    
    def add_log(
        self,
        parsed_log: Dict,
        trace_id: Optional[str],
        service_name: str,
        timestamp: datetime
    ) -> List[Dict]:
        """로그 추가 및 세션 생성"""
        sessions = []
        
        # 1. Trace ID 기반 세션화 (최우선)
        if trace_id:
            trace_session = self._add_to_trace_session(
                parsed_log, trace_id, service_name, timestamp
            )
            if trace_session:
                sessions.append(trace_session)
        
        # 2. 서비스별 세션화
        service_session = self._add_to_service_session(
            parsed_log, service_name, timestamp
        )
        if service_session:
            sessions.append(service_session)
        
        # 3. Sliding Window 세션화
        sliding_session = self._add_to_sliding_window(
            parsed_log, service_name, timestamp
        )
        if sliding_session:
            sessions.append(sliding_session)
        
        return sessions
    
    def _add_to_trace_session(
        self,
        parsed_log: Dict,
        trace_id: str,
        service_name: str,
        timestamp: datetime
    ) -> Optional[Dict]:
        """Trace ID 기반 세션에 추가"""
        key = f"{trace_id}_{service_name}"
        
        # 시간 윈도우 확인
        if key in self.trace_sessions:
            first_log = self.trace_sessions[key][0]
            first_time = first_log.get('timestamp')
            
            if first_time and (timestamp - first_time).total_seconds() > self.trace_window_time:
                # 윈도우 초과 시 세션 완성
                session = self._create_trace_session(self.trace_sessions[key], trace_id, service_name)
                self.trace_sessions[key] = []
                return session
        
        # 로그 추가
        self.trace_sessions[key].append({
            **parsed_log,
            'trace_id': trace_id,
            'service_name': service_name,
            'timestamp': timestamp
        })
        
        # 세션 크기 확인
        if len(self.trace_sessions[key]) >= 100:  # 최대 크기
            session = self._create_trace_session(self.trace_sessions[key], trace_id, service_name)
            self.trace_sessions[key] = []
            return session
        
        return None
    
    def _add_to_service_session(
        self,
        parsed_log: Dict,
        service_name: str,
        timestamp: datetime
    ) -> Optional[Dict]:
        """서비스별 세션에 추가"""
        buffer = self.service_sessions[service_name]
        
        # 시간 윈도우 확인
        if buffer:
            first_time = buffer[0].get('timestamp')
            if first_time and (timestamp - first_time).total_seconds() > self.service_window_time:
                # 윈도우 초과 시 세션 완성
                session = self._create_service_session(list(buffer), service_name)
                buffer.clear()
                return session
        
        # 로그 추가
        buffer.append({
            **parsed_log,
            'service_name': service_name,
            'timestamp': timestamp
        })
        
        # 세션 크기 확인
        if len(buffer) >= 50:  # 최대 크기
            session = self._create_service_session(list(buffer), service_name)
            buffer.clear()
            return session
        
        return None
    
    def _add_to_sliding_window(
        self,
        parsed_log: Dict,
        service_name: str,
        timestamp: datetime
    ) -> Optional[Dict]:
        """Sliding Window에 추가"""
        buffer = self.sliding_windows[service_name]
        
        # 시간 윈도우 확인
        if buffer:
            first_time = buffer[0].get('timestamp')
            if first_time and (timestamp - first_time).total_seconds() > self.sliding_window_time:
                buffer.popleft()
        
        # 로그 추가
        buffer.append({
            **parsed_log,
            'service_name': service_name,
            'timestamp': timestamp
        })
        
        # 윈도우가 가득 찼을 때 세션 생성
        if len(buffer) >= 20:
            session = self._create_sliding_session(list(buffer), service_name)
            return session
        
        return None
    
    def _create_trace_session(self, logs: List[Dict], trace_id: str, service_name: str) -> Dict:
        """Trace 세션 생성"""
        logs.sort(key=lambda x: x.get('timestamp', datetime.min))
        
        return {
            'session_type': 'trace',
            'session_id': f"{trace_id}_{service_name}_{logs[0].get('timestamp', '').timestamp()}",
            'trace_id': trace_id,
            'service_name': service_name,
            'logs': logs,
            'log_count': len(logs),
            'time_span': (logs[-1].get('timestamp') - logs[0].get('timestamp')).total_seconds() if len(logs) > 1 else 0,
            'has_error': any(log.get('level') == 'ERROR' for log in logs),
            'has_warn': any(log.get('level') == 'WARN' for log in logs),
        }
    
    def _create_service_session(self, logs: List[Dict], service_name: str) -> Dict:
        """서비스 세션 생성"""
        logs.sort(key=lambda x: x.get('timestamp', datetime.min))
        
        return {
            'session_type': 'service',
            'session_id': f"{service_name}_{logs[0].get('timestamp', '').timestamp()}",
            'service_name': service_name,
            'logs': logs,
            'log_count': len(logs),
            'time_span': (logs[-1].get('timestamp') - logs[0].get('timestamp')).total_seconds() if len(logs) > 1 else 0,
            'has_error': any(log.get('level') == 'ERROR' for log in logs),
            'has_warn': any(log.get('level') == 'WARN' for log in logs),
        }
    
    def _create_sliding_session(self, logs: List[Dict], service_name: str) -> Dict:
        """Sliding Window 세션 생성"""
        logs.sort(key=lambda x: x.get('timestamp', datetime.min))
        
        return {
            'session_type': 'sliding',
            'session_id': f"{service_name}_sliding_{logs[0].get('timestamp', '').timestamp()}",
            'service_name': service_name,
            'logs': logs,
            'log_count': len(logs),
            'time_span': (logs[-1].get('timestamp') - logs[0].get('timestamp')).total_seconds() if len(logs) > 1 else 0,
            'has_error': any(log.get('level') == 'ERROR' for log in logs),
            'has_warn': any(log.get('level') == 'WARN' for log in logs),
        }
```

---

### 4단계: 고급 특징 추출

#### 4.1 시간적 패턴 특징

```python
class TemporalFeatureExtractor:
    """시간적 패턴 특징 추출기"""
    
    def extract_features(self, session: Dict) -> Dict:
        """시간적 특징 추출"""
        logs = session.get('logs', [])
        
        if not logs:
            return {}
        
        timestamps = [log.get('timestamp') for log in logs if log.get('timestamp')]
        timestamps = [ts for ts in timestamps if ts]
        
        if len(timestamps) < 2:
            return {}
        
        # 시간 간격 계산
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        # 통계 특징
        features = {
            'time_span': (timestamps[-1] - timestamps[0]).total_seconds(),
            'mean_interval': np.mean(intervals) if intervals else 0,
            'std_interval': np.std(intervals) if intervals else 0,
            'min_interval': np.min(intervals) if intervals else 0,
            'max_interval': np.max(intervals) if intervals else 0,
            'median_interval': np.median(intervals) if intervals else 0,
            'interval_variance': np.var(intervals) if intervals else 0,
        }
        
        # 시간대 특징
        first_hour = timestamps[0].hour
        features['hour_of_day'] = first_hour
        features['is_business_hours'] = 9 <= first_hour <= 18
        features['is_night'] = first_hour < 6 or first_hour > 22
        
        # 주기성 특징 (FFT)
        if len(intervals) >= 8:
            fft_values = np.fft.fft(intervals)
            features['fft_dominant_freq'] = np.argmax(np.abs(fft_values[1:len(fft_values)//2])) + 1
            features['fft_power'] = np.sum(np.abs(fft_values)**2)
        
        return features
```

#### 4.2 서비스 의존성 특징

```python
class DependencyFeatureExtractor:
    """서비스 의존성 특징 추출기"""
    
    def extract_features(self, trace_session: Dict) -> Dict:
        """의존성 특징 추출"""
        services = trace_session.get('services', {})
        service_order = list(services.keys())
        
        features = {
            'service_count': len(services),
            'service_diversity': len(set(service_order)),
            'has_gateway': 'gateway' in service_order,
            'has_eureka': 'eureka' in service_order,
        }
        
        # 서비스 호출 순서 특징
        if len(service_order) > 1:
            features['service_chain_length'] = len(service_order)
            features['service_chain'] = '->'.join(service_order)
            
            # Gateway가 첫 번째인지
            features['gateway_first'] = service_order[0] == 'gateway'
            
            # 서비스 깊이 (호출 체인 깊이)
            features['max_depth'] = self._calculate_depth(services)
        
        # 서비스별 로그 수
        service_log_counts = {svc: len(logs) for svc, logs in services.items()}
        features['service_log_distribution'] = service_log_counts
        features['max_service_logs'] = max(service_log_counts.values()) if service_log_counts else 0
        features['min_service_logs'] = min(service_log_counts.values()) if service_log_counts else 0
        
        return features
    
    def _calculate_depth(self, services: Dict) -> int:
        """서비스 호출 깊이 계산"""
        # 간단한 구현 (실제로는 호출 그래프 분석)
        if 'gateway' in services:
            return len(services) - 1  # Gateway 제외
        return len(services)
```

#### 4.3 에러 전파 특징

```python
class ErrorPropagationExtractor:
    """에러 전파 특징 추출기"""
    
    def extract_features(self, trace_session: Dict) -> Dict:
        """에러 전파 특징 추출"""
        services = trace_session.get('services', {})
        service_order = list(services.keys())
        
        features = {
            'error_count': 0,
            'warning_count': 0,
            'error_services': [],
            'warning_services': [],
            'error_propagation': False,
            'error_chain': [],
        }
        
        # 서비스별 에러 추출
        for service in service_order:
            logs = services[service]
            errors = [log for log in logs if log.get('level') == 'ERROR']
            warnings = [log for log in logs if log.get('level') == 'WARN']
            
            if errors:
                features['error_count'] += len(errors)
                features['error_services'].append(service)
                features['error_chain'].append(f"{service}:ERROR")
            
            if warnings:
                features['warning_count'] += len(warnings)
                features['warning_services'].append(service)
                if not errors:  # 에러가 없을 때만 경고 체인에 추가
                    features['error_chain'].append(f"{service}:WARN")
        
        # 에러 전파 확인
        if len(features['error_services']) > 1:
            features['error_propagation'] = True
        
        # 에러 체인 문자열
        features['error_chain_str'] = '->'.join(features['error_chain'])
        
        return features
```

---

### 5단계: MSA 컨텍스트 빌딩

```python
class AdvancedMSAContextBuilder:
    """고급 MSA 컨텍스트 빌더"""
    
    def build_context(self, trace_session: Dict) -> Dict:
        """고급 컨텍스트 빌딩"""
        services = trace_session.get('services', {})
        
        # 기본 컨텍스트
        context = {
            'trace_id': trace_session.get('trace_id'),
            'services': services,
            'service_count': len(services),
            'service_order': list(services.keys()),
        }
        
        # 서비스 호출 그래프 생성
        context['call_graph'] = self._build_call_graph(services)
        
        # 서비스 간 의존성 분석
        context['dependencies'] = self._analyze_dependencies(services)
        
        # 전체 요청 흐름 추적
        context['request_flow'] = self._trace_request_flow(services)
        
        # 성능 메트릭
        context['performance_metrics'] = self._calculate_performance_metrics(services)
        
        # 에러 컨텍스트
        context['error_context'] = self._build_error_context(services)
        
        return context
    
    def _build_call_graph(self, services: Dict) -> Dict:
        """호출 그래프 생성"""
        graph = {}
        
        for service, logs in services.items():
            called_services = set()
            
            for log in logs:
                # URL에서 호출된 서비스 추출
                url = log.get('metadata', {}).get('url', '')
                if url:
                    for target_service in ['research', 'manager', 'code', 'user', 'eureka']:
                        if f'/{target_service}/' in url.lower():
                            called_services.add(target_service)
            
            graph[service] = list(called_services)
        
        return graph
    
    def _analyze_dependencies(self, services: Dict) -> Dict:
        """의존성 분석"""
        dependencies = {}
        
        for service, logs in services.items():
            deps = {
                'depends_on': [],
                'depended_by': [],
                'dependency_count': 0,
            }
            
            # 호출한 서비스 찾기
            for log in logs:
                url = log.get('metadata', {}).get('url', '')
                if url:
                    for target_service in ['research', 'manager', 'code', 'user']:
                        if f'/{target_service}/' in url.lower() and target_service not in deps['depends_on']:
                            deps['depends_on'].append(target_service)
            
            dependencies[service] = deps
            dependencies[service]['dependency_count'] = len(deps['depends_on'])
        
        # 역방향 의존성 계산
        for service, deps in dependencies.items():
            for other_service, other_deps in dependencies.items():
                if service != other_service and service in other_deps['depends_on']:
                    deps['depended_by'].append(other_service)
        
        return dependencies
    
    def _trace_request_flow(self, services: Dict) -> List[Dict]:
        """요청 흐름 추적"""
        flow = []
        
        # 모든 로그를 시간순으로 정렬
        all_logs = []
        for service, logs in services.items():
            for log in logs:
                all_logs.append({
                    **log,
                    'service': service
                })
        
        all_logs.sort(key=lambda x: x.get('timestamp', datetime.min))
        
        # 흐름 생성
        for log in all_logs:
            flow.append({
                'timestamp': log.get('timestamp'),
                'service': log.get('service'),
                'level': log.get('level'),
                'template': log.get('template'),
                'event_id': log.get('event_id'),
            })
        
        return flow
    
    def _calculate_performance_metrics(self, services: Dict) -> Dict:
        """성능 메트릭 계산"""
        metrics = {}
        
        for service, logs in services.items():
            timestamps = [log.get('timestamp') for log in logs if log.get('timestamp')]
            
            if len(timestamps) >= 2:
                duration = (timestamps[-1] - timestamps[0]).total_seconds()
                metrics[service] = {
                    'duration': duration,
                    'log_count': len(logs),
                    'logs_per_second': len(logs) / duration if duration > 0 else 0,
                }
        
        return metrics
    
    def _build_error_context(self, services: Dict) -> Dict:
        """에러 컨텍스트 빌딩"""
        error_context = {
            'has_error': False,
            'error_services': [],
            'error_messages': [],
            'error_templates': [],
            'first_error_service': None,
            'error_propagation_path': [],
        }
        
        for service, logs in services.items():
            errors = [log for log in logs if log.get('level') == 'ERROR']
            
            if errors:
                error_context['has_error'] = True
                error_context['error_services'].append(service)
                
                for error in errors:
                    error_context['error_messages'].append(error.get('original', ''))
                    error_context['error_templates'].append(error.get('template', ''))
        
        if error_context['error_services']:
            error_context['first_error_service'] = error_context['error_services'][0]
            error_context['error_propagation_path'] = error_context['error_services']
        
        return error_context
```

---

### 6단계: 모델별 최적화 인코딩

#### 6.1 LogBERT 인코딩

```python
class LogBERTEncoder:
    """LogBERT 최적화 인코더"""
    
    def __init__(self, vocab_size: int = 20000, max_seq_length: int = 512):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Special Tokens
        self.PAD_TOKEN_ID = 0
        self.CLS_TOKEN_ID = 101
        self.SEP_TOKEN_ID = 102
        self.MASK_TOKEN_ID = 103
        self.UNK_TOKEN_ID = 100
        
        # Event ID -> Token ID 매핑
        self.event_to_token = {}
        self.token_to_event = {}
        self.next_token_id = 1
    
    def encode_session(self, session: Dict) -> Dict:
        """LogBERT용 인코딩"""
        logs = session.get('logs', [])
        
        # Event ID 시퀀스 추출
        event_ids = [log.get('event_id', 0) for log in logs]
        
        # Token ID로 변환
        token_ids = []
        for event_id in event_ids:
            if event_id not in self.event_to_token:
                self.event_to_token[event_id] = self.next_token_id
                self.token_to_event[self.next_token_id] = event_id
                self.next_token_id += 1
            
            token_id = self.event_to_token[event_id]
            token_ids.append(token_id)
        
        # Special Tokens 추가
        token_ids = [self.CLS_TOKEN_ID] + token_ids + [self.SEP_TOKEN_ID]
        
        # Padding
        attention_mask = [1] * len(token_ids)
        
        if len(token_ids) < self.max_seq_length:
            padding_length = self.max_seq_length - len(token_ids)
            token_ids = token_ids + [self.PAD_TOKEN_ID] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        else:
            token_ids = token_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
        
        return {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'event_sequence': event_ids,
            'session_length': len(logs),
            'padded_length': len(token_ids),
        }
```

#### 6.2 DeepLog 인코딩

```python
class DeepLogEncoder:
    """DeepLog 최적화 인코더"""
    
    def __init__(self, vocab_size: int = 10000, window_size: int = 20):
        self.vocab_size = vocab_size
        self.window_size = window_size
        
        # Event ID -> Index 매핑
        self.event_to_index = {}
        self.index_to_event = {}
        self.next_index = 1
    
    def encode_session(self, session: Dict) -> Dict:
        """DeepLog용 인코딩"""
        logs = session.get('logs', [])
        
        # Event ID 시퀀스 추출
        event_ids = [log.get('event_id', 0) for log in logs]
        
        # Index로 변환
        indices = []
        for event_id in event_ids:
            if event_id not in self.event_to_index:
                self.event_to_index[event_id] = self.next_index
                self.index_to_event[self.next_index] = event_id
                self.next_index += 1
            
            index = self.event_to_index[event_id]
            indices.append(index)
        
        # Sliding Window 생성
        windows = []
        labels = []
        
        for i in range(len(indices) - self.window_size):
            window = indices[i:i+self.window_size]
            label = indices[i+self.window_size]
            
            windows.append(window)
            labels.append(label)
        
        return {
            'windows': windows,
            'labels': labels,
            'event_sequence': event_ids,
            'vocab_size': len(self.event_to_index),
        }
```

#### 6.3 LogLSTM/LogTCN 인코딩

```python
class SequenceEncoder:
    """LogLSTM/LogTCN 최적화 인코더"""
    
    def __init__(self, vocab_size: int = 10000, max_seq_length: int = 256):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Event ID -> Index 매핑
        self.event_to_index = {}
        self.index_to_event = {}
        self.next_index = 1
    
    def encode_session(self, session: Dict) -> Dict:
        """시퀀스 모델용 인코딩"""
        logs = session.get('logs', [])
        
        # Event ID 시퀀스 추출
        event_ids = [log.get('event_id', 0) for log in logs]
        
        # Index로 변환
        indices = []
        for event_id in event_ids:
            if event_id not in self.event_to_index:
                self.event_to_index[event_id] = self.next_index
                self.index_to_event[self.next_index] = event_id
                self.next_index += 1
            
            index = self.event_to_index[event_id]
            indices.append(index)
        
        # Padding
        if len(indices) < self.max_seq_length:
            padding_length = self.max_seq_length - len(indices)
            indices = indices + [0] * padding_length
            mask = [1] * len(logs) + [0] * padding_length
        else:
            indices = indices[:self.max_seq_length]
            mask = [1] * self.max_seq_length
        
        return {
            'sequence': indices,
            'mask': mask,
            'event_sequence': event_ids[:self.max_seq_length],
            'sequence_length': len(logs),
            'vocab_size': len(self.event_to_index),
        }
```

---

### 7단계: 데이터 품질 검증 및 필터링

```python
class DataQualityValidator:
    """데이터 품질 검증 클래스"""
    
    def __init__(self):
        self.min_session_length = 5  # 최소 세션 길이
        self.max_session_length = 1000  # 최대 세션 길이
        self.min_unique_events = 2  # 최소 고유 이벤트 수
        self.max_duplicate_ratio = 0.9  # 최대 중복 비율
    
    def validate_session(self, session: Dict) -> Dict:
        """세션 검증"""
        logs = session.get('logs', [])
        
        validations = {
            'valid': True,
            'reasons': [],
            'warnings': [],
        }
        
        # 1. 길이 검증
        if len(logs) < self.min_session_length:
            validations['valid'] = False
            validations['reasons'].append(f'session_too_short: {len(logs)} < {self.min_session_length}')
        
        if len(logs) > self.max_session_length:
            validations['valid'] = False
            validations['reasons'].append(f'session_too_long: {len(logs)} > {self.max_session_length}')
        
        # 2. 고유 이벤트 수 검증
        event_ids = [log.get('event_id', 0) for log in logs]
        unique_events = len(set(event_ids))
        
        if unique_events < self.min_unique_events:
            validations['valid'] = False
            validations['reasons'].append(f'too_few_unique_events: {unique_events} < {self.min_unique_events}')
        
        # 3. 중복 비율 검증
        if len(event_ids) > 0:
            duplicate_ratio = 1 - (unique_events / len(event_ids))
            if duplicate_ratio > self.max_duplicate_ratio:
                validations['warnings'].append(f'high_duplicate_ratio: {duplicate_ratio:.2f}')
        
        # 4. 타임스탬프 검증
        timestamps = [log.get('timestamp') for log in logs if log.get('timestamp')]
        if len(timestamps) < len(logs) * 0.8:  # 80% 이상 타임스탬프 필요
            validations['warnings'].append('missing_timestamps')
        
        # 5. 시간 순서 검증
        if len(timestamps) >= 2:
            sorted_timestamps = sorted(timestamps)
            if timestamps != sorted_timestamps:
                validations['warnings'].append('timestamps_not_sorted')
        
        return validations
    
    def filter_sessions(self, sessions: List[Dict]) -> List[Dict]:
        """세션 필터링"""
        valid_sessions = []
        invalid_count = 0
        
        for session in sessions:
            validation = self.validate_session(session)
            
            if validation['valid']:
                # 경고가 있어도 포함 (로깅만)
                if validation['warnings']:
                    logger.debug(f"Session {session.get('session_id')} warnings: {validation['warnings']}")
                
                valid_sessions.append(session)
            else:
                invalid_count += 1
                logger.debug(f"Session {session.get('session_id')} invalid: {validation['reasons']}")
        
        logger.info(f"필터링 완료: {len(valid_sessions)}/{len(sessions)} 유효 세션 (제거: {invalid_count})")
        
        return valid_sessions
```

---

### 8단계: 데이터 증강 (선택)

```python
class DataAugmenter:
    """데이터 증강 클래스"""
    
    def augment_session(self, session: Dict, methods: List[str] = ['noise', 'shuffle', 'mask']) -> List[Dict]:
        """세션 증강"""
        augmented = [session]  # 원본 포함
        
        logs = session.get('logs', [])
        
        if 'noise' in methods:
            augmented.append(self._add_noise(session))
        
        if 'shuffle' in methods:
            augmented.append(self._shuffle_events(session))
        
        if 'mask' in methods:
            augmented.append(self._mask_events(session))
        
        return augmented
    
    def _add_noise(self, session: Dict) -> Dict:
        """노이즈 추가"""
        logs = session.get('logs', [])
        noisy_logs = logs.copy()
        
        # 일부 이벤트를 랜덤하게 교체 (5% 확률)
        for i in range(len(noisy_logs)):
            if random.random() < 0.05:
                # 랜덤 이벤트 ID로 교체
                noisy_logs[i] = {
                    **noisy_logs[i],
                    'event_id': random.randint(1, 1000)
                }
        
        return {
            **session,
            'logs': noisy_logs,
            'augmented': True,
            'augmentation_method': 'noise'
        }
    
    def _shuffle_events(self, session: Dict) -> Dict:
        """이벤트 순서 섞기 (시간 순서 유지하면서 부분적으로)"""
        logs = session.get('logs', [])
        
        # 작은 윈도우 내에서만 섞기
        window_size = 5
        shuffled_logs = []
        
        for i in range(0, len(logs), window_size):
            window = logs[i:i+window_size]
            random.shuffle(window)
            shuffled_logs.extend(window)
        
        return {
            **session,
            'logs': shuffled_logs,
            'augmented': True,
            'augmentation_method': 'shuffle'
        }
    
    def _mask_events(self, session: Dict) -> Dict:
        """일부 이벤트 마스킹"""
        logs = session.get('logs', [])
        masked_logs = logs.copy()
        
        # 10% 이벤트 마스킹
        mask_count = int(len(masked_logs) * 0.1)
        mask_indices = random.sample(range(len(masked_logs)), mask_count)
        
        for idx in mask_indices:
            masked_logs[idx] = {
                **masked_logs[idx],
                'event_id': 0,  # MASK 토큰
            }
        
        return {
            **session,
            'logs': masked_logs,
            'augmented': True,
            'augmentation_method': 'mask'
        }
```

---

## 📊 최종 출력 형식

### LogBERT용 데이터

```json
{
  "session_id": "gateway_trace_abc123_1234567890",
  "session_type": "trace",
  "token_ids": [101, 1, 2, 3, ..., 102, 0, 0, ...],
  "attention_mask": [1, 1, 1, ..., 1, 0, 0, ...],
  "event_sequence": [1, 5, 1, 12, 3, ...],
  "service_name": "gateway",
  "trace_id": "abc123",
  "has_error": true,
  "has_warn": false,
  "temporal_features": {
    "time_span": 45.2,
    "mean_interval": 2.3,
    "std_interval": 1.5
  },
  "dependency_features": {
    "service_count": 3,
    "service_chain": "gateway->research->manager"
  },
  "error_features": {
    "error_count": 2,
    "error_services": ["manager"],
    "error_propagation": false
  }
}
```

### DeepLog용 데이터

```json
{
  "session_id": "gateway_service_1234567890",
  "session_type": "service",
  "windows": [[1, 2, 3, ...], [2, 3, 4, ...], ...],
  "labels": [4, 5, 6, ...],
  "event_sequence": [1, 2, 3, 4, 5, 6, ...],
  "service_name": "gateway",
  "vocab_size": 500
}
```

### LogLSTM/LogTCN용 데이터

```json
{
  "session_id": "gateway_sliding_1234567890",
  "session_type": "sliding",
  "sequence": [1, 2, 3, 4, 5, ..., 0, 0, ...],
  "mask": [1, 1, 1, ..., 0, 0, ...],
  "event_sequence": [1, 2, 3, 4, 5, ...],
  "service_name": "gateway",
  "sequence_length": 20,
  "vocab_size": 500
}
```

---

## 🔧 통합 전처리 파이프라인

```python
class AdvancedPreprocessingPipeline:
    """고급 전처리 파이프라인"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 컴포넌트 초기화
        self.cleaner = AdvancedLogCleaner()
        self.parser = AdvancedLogParser(config.get('drain3_config_path'))
        self.trace_extractor = AdvancedTraceExtractor()
        self.trace_validator = TraceValidator()
        self.sessionizer = HybridSessionizer(
            trace_window_time=config.get('trace_window_time', 600),
            service_window_size=config.get('service_window_size', 50),
            sliding_window_size=config.get('sliding_window_size', 20),
        )
        
        # 특징 추출기
        self.temporal_extractor = TemporalFeatureExtractor()
        self.dependency_extractor = DependencyFeatureExtractor()
        self.error_extractor = ErrorPropagationExtractor()
        
        # 컨텍스트 빌더
        self.context_builder = AdvancedMSAContextBuilder()
        
        # 인코더
        self.logbert_encoder = LogBERTEncoder(
            vocab_size=config.get('logbert_vocab_size', 20000),
            max_seq_length=config.get('logbert_max_seq_length', 512)
        )
        self.deeplog_encoder = DeepLogEncoder(
            vocab_size=config.get('deeplog_vocab_size', 10000),
            window_size=config.get('deeplog_window_size', 20)
        )
        self.sequence_encoder = SequenceEncoder(
            vocab_size=config.get('sequence_vocab_size', 10000),
            max_seq_length=config.get('sequence_max_seq_length', 256)
        )
        
        # 검증 및 증강
        self.validator = DataQualityValidator()
        self.augmenter = DataAugmenter() if config.get('enable_augmentation', False) else None
    
    def process_logs(self, log_files: List[Path]) -> Dict:
        """전체 전처리 파이프라인 실행"""
        all_sessions = {
            'logbert': [],
            'deeplog': [],
            'lstm': [],
            'tcn': [],
        }
        
        # 1. 로그 파일 처리
        for log_file in log_files:
            service_name = self._extract_service_name(log_file)
            
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # 1-1. 로그 정리
                    cleaned = self.cleaner.clean_log_line(line)
                    if not cleaned:
                        continue
                    
                    # 1-2. 로그 파싱
                    parsed = self.parser.parse_log(cleaned, service_name)
                    if not parsed:
                        continue
                    
                    # 2. Trace ID 추출
                    trace_result = self.trace_extractor.extract_trace_id(cleaned, service_name)
                    trace_id = trace_result['trace_id'] if trace_result else None
                    
                    # 3. Trace ID 검증 및 등록
                    if trace_id:
                        self.trace_validator.register_trace(
                            trace_id, service_name, parsed['timestamp']
                        )
                    
                    # 4. 세션화
                    sessions = self.sessionizer.add_log(
                        parsed, trace_id, service_name, parsed['timestamp']
                    )
                    
                    # 5. 세션 처리
                    for session in sessions:
                        # 5-1. 특징 추출
                        session['temporal_features'] = self.temporal_extractor.extract_features(session)
                        session['dependency_features'] = self.dependency_extractor.extract_features(session)
                        session['error_features'] = self.error_extractor.extract_features(session)
                        
                        # 5-2. MSA 컨텍스트 빌딩
                        if session.get('session_type') == 'trace':
                            session['msa_context'] = self.context_builder.build_context(session)
                        
                        # 5-3. 모델별 인코딩
                        logbert_data = self.logbert_encoder.encode_session(session)
                        deeplog_data = self.deeplog_encoder.encode_session(session)
                        sequence_data = self.sequence_encoder.encode_session(session)
                        
                        # 5-4. 데이터 검증
                        validation = self.validator.validate_session(session)
                        if not validation['valid']:
                            continue
                        
                        # 5-5. 데이터 저장
                        all_sessions['logbert'].append({
                            **session,
                            **logbert_data
                        })
                        all_sessions['deeplog'].append({
                            **session,
                            **deeplog_data
                        })
                        all_sessions['lstm'].append({
                            **session,
                            **sequence_data
                        })
                        all_sessions['tcn'].append({
                            **session,
                            **sequence_data
                        })
                        
                        # 5-6. 데이터 증강 (선택)
                        if self.augmenter and self.config.get('enable_augmentation'):
                            augmented = self.augmenter.augment_session(session)
                            # 증강된 데이터도 추가 (간단히 생략)
        
        # 6. 최종 필터링
        all_sessions['logbert'] = self.validator.filter_sessions(all_sessions['logbert'])
        all_sessions['deeplog'] = self.validator.filter_sessions(all_sessions['deeplog'])
        all_sessions['lstm'] = self.validator.filter_sessions(all_sessions['lstm'])
        all_sessions['tcn'] = self.validator.filter_sessions(all_sessions['tcn'])
        
        return all_sessions
    
    def _extract_service_name(self, log_file: Path) -> str:
        """파일명에서 서비스명 추출"""
        filename = log_file.stem.lower()
        
        for service in ['gateway', 'eureka', 'user', 'research', 'manager', 'code']:
            if service in filename:
                return service
        
        return 'unknown'
```

---

## ✅ 체크리스트

### 전처리 품질 확인

- [ ] 모든 로그가 정확하게 파싱되었는가?
- [ ] Trace ID가 정확하게 추출되고 연결되었는가?
- [ ] 세션이 적절한 크기와 시간 범위를 가지는가?
- [ ] 시간적 특징이 정확하게 추출되었는가?
- [ ] 서비스 의존성이 정확하게 분석되었는가?
- [ ] 에러 전파가 정확하게 추적되었는가?
- [ ] 모델별 인코딩이 올바른 형식인가?
- [ ] 데이터 품질 검증이 통과되었는가?

---

이 가이드를 따라 최고 정확도의 전처리 데이터를 생성할 수 있습니다! 🚀
