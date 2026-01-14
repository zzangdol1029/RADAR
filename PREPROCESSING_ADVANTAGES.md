# 고급 전처리 프로세스의 장점 및 개선 사항

## 📋 개요

이 문서는 `ADVANCED_PREPROCESSING_GUIDE.md`에서 제시한 전처리 프로세스의 장점을 분석하고, 추가 개선 사항을 제안합니다.

---

## ✅ 현재 전처리 프로세스의 주요 장점

### 1. 정확도 향상

#### 1.1 정교한 로그 파싱
- **Drain3 최적화**: 깊이 4, 낮은 유사도 임계값(0.5)으로 더 세밀한 템플릿 추출
- **다층 정리**: 배너, 노이즈, 인코딩 문제를 다단계로 제거
- **메타데이터 추출**: 스레드, HTTP 메서드, 상태 코드, IP, URL 등 상세 정보 추출

**효과:**
- 템플릿 추출 정확도 향상 → Event ID 매핑 정확도 향상
- 노이즈 제거로 모델 학습 품질 향상
- 메타데이터로 추가 특징 추출 가능

#### 1.2 다중 Trace ID 추출 및 검증
- **다중 패턴 지원**: JSON, HTTP, Sleuth, UUID 등 다양한 형식 지원
- **신뢰도 기반 선택**: 여러 후보 중 가장 신뢰도 높은 Trace ID 선택
- **검증 시스템**: Trace ID 유효성 검증 및 이상치 탐지

**효과:**
- Trace ID 추출률 향상 (단일 방법 대비 20-30% 향상 예상)
- 잘못된 Trace ID 연결 방지
- MSA 환경에서 서비스 간 연결 정확도 향상

### 2. 데이터 품질 향상

#### 2.1 다중 세션화 전략
- **하이브리드 접근**: Trace ID 기반 + 서비스별 + Sliding Window
- **유연한 윈도우**: 시간 기반 + 크기 기반 윈도우
- **다양한 세션 타입**: trace, service, sliding 세션 생성

**효과:**
- Trace ID가 없는 로그도 활용 가능
- 다양한 모델에 최적화된 세션 생성
- 데이터 활용률 향상 (기존 대비 30-50% 향상 예상)

#### 2.2 고급 특징 추출
- **시간적 패턴**: 간격 통계, FFT, 주기성 분석
- **서비스 의존성**: 호출 그래프, 깊이 분석, 분포 분석
- **에러 전파**: 에러 체인, 전파 경로 추적

**효과:**
- 모델이 학습할 수 있는 특징 증가
- 시간적 패턴 인식 능력 향상
- MSA 환경의 복잡한 의존성 파악 가능

### 3. 모델별 최적화

#### 3.1 모델 특성에 맞는 인코딩
- **LogBERT**: BERT 스타일 토큰화 (CLS/SEP, Padding)
- **DeepLog**: Sliding Window + Label 생성
- **LogLSTM/LogTCN**: 시퀀스 인코딩

**효과:**
- 각 모델의 최고 성능 달성 가능
- 모델 아키텍처에 맞는 입력 형식 제공
- 학습 효율 향상

#### 3.2 데이터 품질 검증
- **다층 검증**: 길이, 고유 이벤트 수, 중복 비율, 타임스탬프 검증
- **자동 필터링**: 불량 데이터 자동 제거
- **경고 시스템**: 문제가 있어도 경고만 하고 포함

**효과:**
- 학습 데이터 품질 보장
- 모델 성능 안정성 향상
- 디버깅 용이성 향상

### 4. MSA 환경 특화

#### 4.1 MSA 컨텍스트 빌딩
- **호출 그래프 생성**: 서비스 간 호출 관계 시각화
- **의존성 분석**: 정방향/역방향 의존성 분석
- **요청 흐름 추적**: 전체 요청의 시간순 추적
- **성능 메트릭**: 서비스별 성능 지표 계산

**효과:**
- MSA 환경의 복잡한 구조 이해 가능
- 서비스 간 에러 전파 추적 가능
- 성능 병목 지점 식별 가능

#### 4.2 에러 컨텍스트 분석
- **에러 전파 경로**: 어떤 서비스에서 에러가 시작되어 전파되는지 추적
- **에러 체인**: 에러 발생 순서 및 서비스 체인 분석
- **첫 에러 서비스 식별**: 근본 원인 서비스 식별

**효과:**
- 에러 근본 원인 분석 용이
- 서비스 간 의존성 문제 파악 가능
- 빠른 문제 해결 가능

---

## 🚀 추가 개선 사항 제안

### 1. 통계적 특징 추가

#### 1.1 이벤트 빈도 분석
```python
class FrequencyFeatureExtractor:
    """이벤트 빈도 특징 추출기"""
    
    def extract_features(self, session: Dict) -> Dict:
        """빈도 기반 특징 추출"""
        logs = session.get('logs', [])
        event_ids = [log.get('event_id', 0) for log in logs]
        
        # 이벤트 빈도 계산
        from collections import Counter
        event_counts = Counter(event_ids)
        
        features = {
            'total_events': len(event_ids),
            'unique_events': len(event_counts),
            'event_diversity': len(event_counts) / len(event_ids) if event_ids else 0,
            'most_frequent_event': event_counts.most_common(1)[0][0] if event_counts else 0,
            'most_frequent_count': event_counts.most_common(1)[0][1] if event_counts else 0,
            'entropy': self._calculate_entropy(event_counts),
            'gini_coefficient': self._calculate_gini(event_counts),
        }
        
        # 상위 N개 이벤트 빈도
        top_n = 5
        for i, (event_id, count) in enumerate(event_counts.most_common(top_n)):
            features[f'top_{i+1}_event_id'] = event_id
            features[f'top_{i+1}_frequency'] = count / len(event_ids)
        
        return features
    
    def _calculate_entropy(self, event_counts: Counter) -> float:
        """엔트로피 계산 (이벤트 분포의 불확실성)"""
        total = sum(event_counts.values())
        if total == 0:
            return 0
        
        entropy = 0
        for count in event_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_gini(self, event_counts: Counter) -> float:
        """지니 계수 계산 (이벤트 분포의 불평등도)"""
        total = sum(event_counts.values())
        if total == 0:
            return 0
        
        counts = sorted(event_counts.values(), reverse=True)
        n = len(counts)
        
        gini = 0
        for i, count in enumerate(counts):
            gini += (2 * (i + 1) - n - 1) * count
        
        return gini / (n * total) if n > 0 and total > 0 else 0
```

**장점:**
- 이벤트 분포의 특성 파악 (균등한지, 특정 이벤트에 집중되는지)
- 이상 탐지에 유용한 특징 제공
- 엔트로피가 낮으면 특정 패턴 반복, 높으면 다양한 패턴

#### 1.2 N-gram 패턴 분석
```python
class NGramFeatureExtractor:
    """N-gram 패턴 특징 추출기"""
    
    def __init__(self, n_values: List[int] = [2, 3, 4]):
        self.n_values = n_values
    
    def extract_features(self, session: Dict) -> Dict:
        """N-gram 특징 추출"""
        logs = session.get('logs', [])
        event_ids = [log.get('event_id', 0) for log in logs]
        
        features = {}
        
        for n in self.n_values:
            # N-gram 생성
            ngrams = [tuple(event_ids[i:i+n]) for i in range(len(event_ids) - n + 1)]
            
            if not ngrams:
                continue
            
            # N-gram 빈도
            from collections import Counter
            ngram_counts = Counter(ngrams)
            
            features[f'{n}gram_count'] = len(ngrams)
            features[f'{n}gram_unique'] = len(ngram_counts)
            features[f'{n}gram_diversity'] = len(ngram_counts) / len(ngrams) if ngrams else 0
            
            # 가장 빈번한 N-gram
            if ngram_counts:
                most_common = ngram_counts.most_common(1)[0]
                features[f'{n}gram_most_frequent'] = most_common[0]
                features[f'{n}gram_most_frequent_count'] = most_common[1]
                features[f'{n}gram_most_frequent_ratio'] = most_common[1] / len(ngrams)
        
        return features
```

**장점:**
- 이벤트 시퀀스의 패턴 인식
- 반복되는 패턴 탐지
- 순서 정보 보존

### 2. 시계열 특징 강화

#### 2.1 트렌드 분석
```python
class TrendFeatureExtractor:
    """트렌드 분석 특징 추출기"""
    
    def extract_features(self, session: Dict) -> Dict:
        """트렌드 특징 추출"""
        logs = session.get('logs', [])
        timestamps = [log.get('timestamp') for log in logs if log.get('timestamp')]
        event_ids = [log.get('event_id', 0) for log in logs]
        
        if len(timestamps) < 3:
            return {}
        
        # 시간 간격 추출
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        features = {}
        
        # 선형 트렌드 분석
        if len(intervals) >= 3:
            x = np.arange(len(intervals))
            slope, intercept = np.polyfit(x, intervals, 1)
            features['interval_trend_slope'] = slope
            features['interval_trend_intercept'] = intercept
            features['interval_increasing'] = slope > 0
            features['interval_decreasing'] = slope < 0
        
        # 이벤트 ID 트렌드
        if len(event_ids) >= 3:
            x = np.arange(len(event_ids))
            slope, intercept = np.polyfit(x, event_ids, 1)
            features['event_trend_slope'] = slope
            features['event_trend_intercept'] = intercept
        
        # 변화율 분석
        if len(intervals) >= 2:
            changes = [intervals[i] - intervals[i-1] for i in range(1, len(intervals))]
            features['interval_change_mean'] = np.mean(changes) if changes else 0
            features['interval_change_std'] = np.std(changes) if changes else 0
            features['interval_change_max'] = np.max(changes) if changes else 0
            features['interval_change_min'] = np.min(changes) if changes else 0
        
        return features
```

**장점:**
- 시간에 따른 변화 패턴 파악
- 가속/감속 패턴 탐지
- 이상 구간 식별 용이

#### 2.2 계절성 및 주기성 분석
```python
class SeasonalityFeatureExtractor:
    """계절성 및 주기성 특징 추출기"""
    
    def extract_features(self, session: Dict) -> Dict:
        """계절성 특징 추출"""
        logs = session.get('logs', [])
        timestamps = [log.get('timestamp') for log in logs if log.get('timestamp')]
        
        if len(timestamps) < 10:
            return {}
        
        features = {}
        
        # 시간대별 분포
        hours = [ts.hour for ts in timestamps]
        hour_distribution = Counter(hours)
        features['hour_distribution_entropy'] = self._calculate_entropy(hour_distribution)
        features['most_frequent_hour'] = hour_distribution.most_common(1)[0][0] if hour_distribution else 0
        
        # 요일별 분포
        weekdays = [ts.weekday() for ts in timestamps]
        weekday_distribution = Counter(weekdays)
        features['weekday_distribution_entropy'] = self._calculate_entropy(weekday_distribution)
        features['most_frequent_weekday'] = weekday_distribution.most_common(1)[0][0] if weekday_distribution else 0
        
        # 분 단위 주기성 (FFT)
        if len(timestamps) >= 16:
            minutes = [(ts - timestamps[0]).total_seconds() / 60 for ts in timestamps]
            fft_values = np.fft.fft(minutes)
            power_spectrum = np.abs(fft_values) ** 2
            
            # 주요 주파수 찾기
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            features['dominant_period_minutes'] = len(minutes) / dominant_freq_idx if dominant_freq_idx > 0 else 0
            features['period_strength'] = power_spectrum[dominant_freq_idx] / np.sum(power_spectrum)
        
        return features
```

**장점:**
- 시간대별 패턴 인식
- 주기적 패턴 탐지
- 계절성 요인 반영

### 3. 의미론적 특징 추가

#### 3.1 에러 유형 분류
```python
class ErrorTypeClassifier:
    """에러 유형 분류기"""
    
    def __init__(self):
        # 에러 패턴 정의
        self.error_patterns = {
            'connection': [
                r'connection.*?timeout',
                r'connection.*?refused',
                r'connection.*?reset',
                r'unable.*?connect',
            ],
            'authentication': [
                r'authentication.*?failed',
                r'unauthorized',
                r'access.*?denied',
                r'invalid.*?token',
            ],
            'resource': [
                r'out.*?memory',
                r'disk.*?full',
                r'resource.*?exhausted',
                r'too.*?many.*?connections',
            ],
            'timeout': [
                r'timeout',
                r'request.*?timeout',
                r'read.*?timeout',
                r'write.*?timeout',
            ],
            'database': [
                r'sql.*?error',
                r'database.*?error',
                r'query.*?failed',
                r'transaction.*?failed',
            ],
            'network': [
                r'network.*?error',
                r'socket.*?error',
                r'host.*?unreachable',
                r'network.*?unavailable',
            ],
        }
    
    def classify_errors(self, session: Dict) -> Dict:
        """에러 유형 분류"""
        logs = session.get('logs', [])
        error_logs = [log for log in logs if log.get('level') == 'ERROR']
        
        error_types = Counter()
        error_details = []
        
        for error_log in error_logs:
            template = error_log.get('template', '').lower()
            original = error_log.get('original', '').lower()
            text = f"{template} {original}"
            
            # 패턴 매칭
            matched_types = []
            for error_type, patterns in self.error_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        matched_types.append(error_type)
                        break
            
            if matched_types:
                error_types.update(matched_types)
                error_details.append({
                    'error_type': matched_types[0],  # 첫 번째 매칭
                    'template': error_log.get('template'),
                    'service': error_log.get('service_name'),
                })
            else:
                error_types['unknown'] += 1
        
        return {
            'error_type_distribution': dict(error_types),
            'primary_error_type': error_types.most_common(1)[0][0] if error_types else None,
            'error_type_count': len(set([d['error_type'] for d in error_details])),
            'error_details': error_details,
        }
```

**장점:**
- 에러 유형별 특성 파악
- 유형별 대응 전략 수립 가능
- 에러 패턴 학습 향상

#### 3.2 서비스 상태 추론
```python
class ServiceStateInferencer:
    """서비스 상태 추론기"""
    
    def infer_state(self, session: Dict) -> Dict:
        """서비스 상태 추론"""
        logs = session.get('logs', [])
        
        # 로그 레벨 분포
        levels = [log.get('level', 'UNKNOWN') for log in logs]
        level_counts = Counter(levels)
        
        # 상태 추론 규칙
        error_ratio = level_counts.get('ERROR', 0) / len(logs) if logs else 0
        warn_ratio = level_counts.get('WARN', 0) / len(logs) if logs else 0
        
        if error_ratio > 0.1:
            state = 'CRITICAL'
        elif error_ratio > 0.05:
            state = 'ERROR'
        elif warn_ratio > 0.2:
            state = 'WARNING'
        elif warn_ratio > 0.1:
            state = 'DEGRADED'
        else:
            state = 'NORMAL'
        
        # HTTP 상태 코드 분석
        http_statuses = []
        for log in logs:
            status = log.get('metadata', {}).get('http_status')
            if status:
                http_statuses.append(status)
        
        http_status_dist = Counter(http_statuses)
        error_status_ratio = sum([count for status, count in http_status_dist.items() if status >= 400]) / len(http_statuses) if http_statuses else 0
        
        return {
            'inferred_state': state,
            'error_ratio': error_ratio,
            'warn_ratio': warn_ratio,
            'http_error_ratio': error_status_ratio,
            'level_distribution': dict(level_counts),
            'http_status_distribution': dict(http_status_dist),
        }
```

**장점:**
- 서비스 상태 자동 분류
- 상태 변화 추적 가능
- 이상 상태 조기 탐지

### 4. 관계형 특징 강화

#### 4.1 서비스 간 상관관계 분석
```python
class ServiceCorrelationAnalyzer:
    """서비스 간 상관관계 분석기"""
    
    def analyze_correlation(self, trace_sessions: List[Dict]) -> Dict:
        """서비스 간 상관관계 분석"""
        # 서비스 쌍별 공출현 빈도
        service_pairs = Counter()
        service_cooccurrence = defaultdict(set)
        
        for session in trace_sessions:
            services = list(session.get('services', {}).keys())
            
            # 모든 서비스 쌍 생성
            for i in range(len(services)):
                for j in range(i+1, len(services)):
                    pair = tuple(sorted([services[i], services[j]]))
                    service_pairs[pair] += 1
                    service_cooccurrence[services[i]].add(services[j])
                    service_cooccurrence[services[j]].add(services[i])
        
        # 상관관계 계산
        correlations = {}
        total_sessions = len(trace_sessions)
        
        for (svc1, svc2), count in service_pairs.items():
            # Jaccard 유사도
            cooccurring = count
            svc1_total = sum(1 for s in trace_sessions if svc1 in s.get('services', {}))
            svc2_total = sum(1 for s in trace_sessions if svc2 in s.get('services', {}))
            
            jaccard = cooccurring / (svc1_total + svc2_total - cooccurring) if (svc1_total + svc2_total - cooccurring) > 0 else 0
            
            correlations[f"{svc1}_{svc2}"] = {
                'cooccurrence_count': count,
                'cooccurrence_ratio': count / total_sessions,
                'jaccard_similarity': jaccard,
            }
        
        return {
            'service_correlations': correlations,
            'service_cooccurrence_graph': {k: list(v) for k, v in service_cooccurrence.items()},
        }
```

**장점:**
- 서비스 간 관계 정량화
- 강한 의존성 서비스 쌍 식별
- 장애 전파 경로 예측 가능

### 5. 데이터 증강 강화

#### 5.1 시맨틱 보존 증강
```python
class SemanticPreservingAugmenter:
    """의미 보존 데이터 증강기"""
    
    def augment_with_similarity(self, session: Dict, similar_sessions: List[Dict]) -> List[Dict]:
        """유사 세션 기반 증강"""
        augmented = [session]
        
        # 유사한 세션에서 부분 교체
        for similar_session in similar_sessions[:3]:  # 상위 3개만
            # 부분 교체 (50% 확률로 교체)
            mixed_logs = []
            original_logs = session.get('logs', [])
            similar_logs = similar_session.get('logs', [])
            
            for i in range(len(original_logs)):
                if random.random() < 0.5 and i < len(similar_logs):
                    mixed_logs.append(similar_logs[i])
                else:
                    mixed_logs.append(original_logs[i])
            
            augmented.append({
                **session,
                'logs': mixed_logs,
                'augmented': True,
                'augmentation_method': 'similarity_mix',
                'source_session_id': similar_session.get('session_id'),
            })
        
        return augmented
    
    def augment_with_interpolation(self, session1: Dict, session2: Dict) -> Dict:
        """두 세션 간 보간"""
        logs1 = session1.get('logs', [])
        logs2 = session2.get('logs', [])
        
        # 짧은 세션에 맞춤
        min_len = min(len(logs1), len(logs2))
        logs1 = logs1[:min_len]
        logs2 = logs2[:min_len]
        
        # 교차 보간
        interpolated_logs = []
        for i in range(min_len):
            if i % 2 == 0:
                interpolated_logs.append(logs1[i])
            else:
                interpolated_logs.append(logs2[i])
        
        return {
            **session1,
            'logs': interpolated_logs,
            'augmented': True,
            'augmentation_method': 'interpolation',
            'source_session_ids': [session1.get('session_id'), session2.get('session_id')],
        }
```

**장점:**
- 의미를 보존하면서 데이터 증강
- 유사 패턴 학습 강화
- 데이터 다양성 증가

---

## 📊 개선 사항 요약

### 현재 가이드의 강점
1. ✅ **정교한 파싱**: Drain3 최적화로 정확한 템플릿 추출
2. ✅ **다중 Trace ID 추출**: 다양한 형식 지원으로 추출률 향상
3. ✅ **하이브리드 세션화**: 다양한 세션 타입 생성
4. ✅ **고급 특징 추출**: 시간적, 의존성, 에러 전파 특징
5. ✅ **모델별 최적화**: 각 모델에 맞는 인코딩
6. ✅ **품질 검증**: 다층 검증 시스템

### 추가 개선 제안
1. 🔄 **통계적 특징**: 빈도 분석, N-gram 패턴, 엔트로피
2. 🔄 **시계열 강화**: 트렌드 분석, 계절성 분석
3. 🔄 **의미론적 특징**: 에러 유형 분류, 서비스 상태 추론
4. 🔄 **관계형 특징**: 서비스 간 상관관계 분석
5. 🔄 **증강 강화**: 의미 보존 증강, 보간 기반 증강

---

## 💡 권장 사항

### 필수 추가 사항 (높은 우선순위)
1. **통계적 특징 추출**: 이벤트 빈도, 엔트로피, N-gram 패턴
2. **에러 유형 분류**: 에러 패턴 기반 분류
3. **서비스 상태 추론**: 자동 상태 분류

### 선택적 추가 사항 (중간 우선순위)
1. **트렌드 분석**: 시간에 따른 변화 패턴
2. **계절성 분석**: 시간대별, 요일별 패턴
3. **서비스 상관관계**: 서비스 간 관계 분석

### 고급 추가 사항 (낮은 우선순위)
1. **의미 보존 증강**: 유사 세션 기반 증강
2. **보간 기반 증강**: 세션 간 보간

---

## 🎯 결론

현재 전처리 가이드는 **이미 매우 상세하고 정확도가 높은** 프로세스를 제시하고 있습니다. 

**추가 개선 사항은 선택적으로 적용**하면 됩니다:
- **필수**: 통계적 특징, 에러 분류, 상태 추론
- **선택**: 트렌드/계절성 분석, 상관관계 분석
- **고급**: 의미 보존 증강

현재 가이드만으로도 충분히 높은 정확도를 달성할 수 있으며, 추가 개선 사항은 **성능 향상이 필요한 경우**에 점진적으로 적용하는 것을 권장합니다.
