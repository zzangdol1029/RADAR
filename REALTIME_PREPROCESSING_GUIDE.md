# 실시간 전처리 파이프라인 가이드

## 📋 개요

실시간 이상 탐지를 위한 스트리밍 전처리 파이프라인과 배치 전처리 파이프라인을 분리하여 설계합니다.

---

## 🔄 두 가지 전처리 파이프라인

### 1. 배치 전처리 (Batch Preprocessing)
- **목적**: 모델 학습용 데이터 생성
- **처리 방식**: 전체 로그를 통합하여 일괄 처리
- **사용 시점**: 모델 학습 전, 주기적 재학습

### 2. 실시간 전처리 (Real-time Preprocessing)
- **목적**: 실시간 이상 탐지
- **처리 방식**: 로그를 스트리밍으로 실시간 처리
- **사용 시점**: 운영 환경에서 지속적으로 실행

---

## 🏗️ 아키텍처

### 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────┐
│              로그 수집 레이어                              │
│  Gateway │ Eureka │ User │ Research │ Manager │ Code   │
└─────────────────────────────────────────────────────────┘
                    ↓                    ↓
        ┌───────────┴───────────┐       │
        ↓                       ↓       ↓
┌───────────────┐      ┌───────────────┐
│ 배치 전처리     │      │ 실시간 전처리   │
│ (학습용)       │      │ (운영용)       │
└───────────────┘      └───────────────┘
        ↓                       ↓
┌───────────────┐      ┌───────────────┐
│ 학습 데이터     │      │ 실시간 스트림   │
│ 생성           │      │ 처리           │
└───────────────┘      └───────────────┘
        ↓                       ↓
┌───────────────┐      ┌───────────────┐
│ 모델 학습       │      │ 실시간 이상 탐지 │
└───────────────┘      └───────────────┘
```

---

## 📊 배치 전처리 (Batch Preprocessing)

### 특징
- 전체 로그를 통합하여 처리
- 시간이 오래 걸려도 상관없음
- 정확도 우선
- 모델 학습용 데이터 생성

### 사용 시점
- 초기 모델 학습 전
- 주기적 재학습 (예: 주 1회)
- 새로운 서비스 추가 시

### 구현 모듈
- `preprocessing/batch_preprocessor.py`
- `preprocessing/log_collector.py` (기존)
- `preprocessing/msa_preprocessor.py` (기존)

---

## ⚡ 실시간 전처리 (Real-time Preprocessing)

### 특징
- 로그를 스트리밍으로 실시간 처리
- 낮은 지연 시간 (수백 ms ~ 수초)
- 메모리 효율적
- 실시간 이상 탐지용

### 사용 시점
- 운영 환경에서 지속 실행
- 실시간 모니터링
- 즉시 알림 필요 시

### 구현 모듈
- `preprocessing/streaming_preprocessor.py` ⭐ (신규)
- `preprocessing/streaming_sessionizer.py` ⭐ (신규)

---

## 🔧 실시간 전처리 모듈

### `preprocessing/streaming_preprocessor.py`

```python
#!/usr/bin/env python3
"""
실시간 스트리밍 전처리 모듈
로그를 실시간으로 처리하여 이상 탐지에 사용
"""

import json
import re
from typing import Dict, Any, Optional, List, Iterator
from datetime import datetime
from collections import deque
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class StreamingPreprocessor:
    """실시간 스트리밍 전처리 클래스"""
    
    def __init__(
        self,
        window_size: int = 20,
        window_time: int = 300,  # 5분
        max_buffer_size: int = 10000
    ):
        """
        Args:
            window_size: 세션 윈도우 크기 (로그 개수)
            window_time: 세션 윈도우 시간 (초)
            max_buffer_size: 최대 버퍼 크기
        """
        self.window_size = window_size
        self.window_time = window_time
        self.max_buffer_size = max_buffer_size
        
        # 실시간 세션 버퍼
        self.session_buffers = {}  # service_name -> deque
        
        # 로그 정리기
        self.log_cleaner = LogCleaner()
        
        # 로그 파서
        self.log_parser = LogParser()
    
    def process_log_line(
        self,
        log_line: str,
        service_name: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        단일 로그 라인을 실시간으로 처리
        
        Args:
            log_line: 로그 라인 문자열
            service_name: 서비스명
            timestamp: 타임스탬프 (None이면 현재 시간)
        
        Returns:
            전처리된 세션 (윈도우가 채워지면 반환, 아니면 None)
        """
        # 로그 정리
        cleaned = self.log_cleaner.clean(log_line)
        if not cleaned:
            return None
        
        # 로그 파싱
        parsed = self.log_parser.parse(parsed_log, service_name)
        if not parsed:
            return None
        
        # 타임스탬프 설정
        if timestamp is None:
            timestamp = datetime.now()
        parsed['timestamp'] = timestamp
        
        # 서비스별 버퍼에 추가
        if service_name not in self.session_buffers:
            self.session_buffers[service_name] = deque(maxlen=self.max_buffer_size)
        
        buffer = self.session_buffers[service_name]
        buffer.append(parsed)
        
        # 윈도우가 채워졌는지 확인
        if len(buffer) >= self.window_size:
            # 윈도우 추출
            window = list(buffer)[-self.window_size:]
            
            # 세션 생성
            session = self._create_session(window, service_name)
            
            # 버퍼에서 오래된 항목 제거 (메모리 관리)
            self._cleanup_old_logs(service_name, timestamp)
            
            return session
        
        return None
    
    def process_log_stream(
        self,
        log_stream: Iterator[str],
        service_name: str
    ) -> Iterator[Dict[str, Any]]:
        """
        로그 스트림을 실시간으로 처리
        
        Args:
            log_stream: 로그 라인 이터레이터
            service_name: 서비스명
        
        Yields:
            전처리된 세션
        """
        for log_line in log_stream:
            session = self.process_log_line(log_line, service_name)
            if session:
                yield session
    
    def process_file_stream(
        self,
        log_file: Path,
        service_name: str,
        follow: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """
        로그 파일을 실시간으로 읽어서 처리 (tail -f 방식)
        
        Args:
            log_file: 로그 파일 경로
            service_name: 서비스명
            follow: 파일 끝에 도달해도 계속 읽기 (tail -f)
        
        Yields:
            전처리된 세션
        """
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            # 파일 끝으로 이동 (새로운 로그만 읽기)
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                
                if line:
                    session = self.process_log_line(line.strip(), service_name)
                    if session:
                        yield session
                elif not follow:
                    break
                else:
                    # 파일 끝에 도달했지만 follow 모드면 대기
                    import time
                    time.sleep(0.1)
    
    def _create_session(
        self,
        window: List[Dict[str, Any]],
        service_name: str
    ) -> Dict[str, Any]:
        """세션 생성"""
        # Trace ID 추출
        trace_id = self._extract_trace_id(window)
        
        # 이벤트 시퀀스 생성
        event_sequence = [log.get('event_id', 0) for log in window]
        
        # 토큰 인코딩
        token_ids = self._encode_tokens(window)
        attention_mask = [1] * len(token_ids)
        
        # 패딩 (최대 길이 512)
        max_len = 512
        if len(token_ids) < max_len:
            padding_len = max_len - len(token_ids)
            token_ids = token_ids + [0] * padding_len
            attention_mask = attention_mask + [0] * padding_len
        else:
            token_ids = token_ids[:max_len]
            attention_mask = attention_mask[:max_len]
        
        return {
            'session_id': f"{service_name}_{datetime.now().timestamp()}",
            'event_sequence': event_sequence,
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'service_name': service_name,
            'trace_id': trace_id,
            'timestamp': window[-1].get('timestamp'),
            'has_error': any(log.get('level') == 'ERROR' for log in window),
            'has_warn': any(log.get('level') == 'WARN' for log in window),
            'window_size': len(window)
        }
    
    def _extract_trace_id(self, window: List[Dict[str, Any]]) -> Optional[str]:
        """윈도우에서 Trace ID 추출"""
        for log in window:
            trace_id = log.get('trace_id')
            if trace_id:
                return trace_id
        return None
    
    def _encode_tokens(self, window: List[Dict[str, Any]]) -> List[int]:
        """토큰 인코딩"""
        # 간단한 인코딩 (실제로는 vocab 사용)
        tokens = []
        for log in window:
            # 이벤트 ID를 토큰으로 사용
            event_id = log.get('event_id', 0)
            tokens.append(event_id)
        return tokens
    
    def _cleanup_old_logs(self, service_name: str, current_time: datetime):
        """오래된 로그 제거 (메모리 관리)"""
        buffer = self.session_buffers.get(service_name)
        if not buffer:
            return
        
        # 윈도우 시간보다 오래된 로그 제거
        while buffer:
            oldest_log = buffer[0]
            oldest_time = oldest_log.get('timestamp')
            
            if oldest_time and (current_time - oldest_time).total_seconds() > self.window_time:
                buffer.popleft()
            else:
                break


class LogCleaner:
    """로그 정리 클래스"""
    
    def clean(self, log_line: str) -> Optional[str]:
        """로그 라인 정리"""
        # 빈 라인 제거
        if not log_line.strip():
            return None
        
        # Spring Boot 배너 제거
        banner_patterns = [
            r'\.\s+____.*?Spring Boot.*?::',
            r':: Spring Boot ::',
        ]
        
        for pattern in banner_patterns:
            if re.search(pattern, log_line):
                return None
        
        return log_line.strip()


class LogParser:
    """로그 파서 클래스"""
    
    def parse(self, log_line: str, service_name: str) -> Optional[Dict[str, Any]]:
        """로그 라인 파싱"""
        # 기본 파싱 (실제로는 더 정교하게)
        try:
            # 타임스탬프 추출
            timestamp_str = log_line[:23]
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
            
            # 레벨 추출
            level = None
            for log_level in ['ERROR', 'WARN', 'INFO', 'DEBUG']:
                if log_level in log_line:
                    level = log_level
                    break
            
            # Trace ID 추출
            trace_id = None
            trace_match = re.search(r'"trace_id":"([^"]+)"', log_line)
            if trace_match:
                trace_id = trace_match.group(1)
            
            # 이벤트 ID (간단한 해시)
            event_id = hash(log_line) % 10000
            
            return {
                'timestamp': timestamp,
                'level': level,
                'message': log_line,
                'service_name': service_name,
                'trace_id': trace_id,
                'event_id': event_id
            }
        except Exception as e:
            logger.debug(f"로그 파싱 실패: {e}")
            return None
```

---

## 🔄 실시간 이상 탐지 파이프라인

### 전체 흐름

```python
# 실시간 이상 탐지 파이프라인
from preprocessing.streaming_preprocessor import StreamingPreprocessor
from anomaly_detection.ensemble_detector import EnsembleAnomalyDetector

# 전처리기 초기화
preprocessor = StreamingPreprocessor(
    window_size=20,
    window_time=300
)

# 이상 탐지 모델 로드
detector = EnsembleAnomalyDetector(...)

# 로그 파일 실시간 읽기
log_file = Path('logs/gateway/gateway.log')

for session in preprocessor.process_file_stream(log_file, 'gateway', follow=True):
    # 실시간 이상 탐지
    result = detector.predict_batch([session], threshold=0.5)
    
    if result[0]['is_anomaly']:
        # 이상 탐지 알림
        print(f"🚨 이상 탐지: {session['session_id']}")
        print(f"   점수: {result[0]['ensemble_score']:.4f}")
        print(f"   서비스: {session['service_name']}")
```

---

## 📊 배치 vs 실시간 비교

| 항목 | 배치 전처리 | 실시간 전처리 |
|------|------------|--------------|
| **목적** | 모델 학습용 데이터 | 실시간 이상 탐지 |
| **처리 방식** | 전체 로그 통합 | 스트리밍 처리 |
| **지연 시간** | 수 시간 ~ 수일 | 수백 ms ~ 수초 |
| **메모리 사용** | 높음 (전체 로그) | 낮음 (윈도우만) |
| **정확도** | 높음 | 보통 |
| **사용 시점** | 학습 전, 주기적 | 운영 중 지속적 |

---

## 🏗️ 통합 아키텍처

### 멤버 1의 역할 확장

**배치 전처리 (학습용):**
- 전체 로그 수집 및 통합
- 배치 전처리 파이프라인
- 학습 데이터 생성

**실시간 전처리 (운영용):**
- 스트리밍 전처리 모듈 구현
- 실시간 파이프라인 구축
- 실시간 이상 탐지와 연동

---

## 💻 사용 예시

### 배치 전처리 (학습용)

```bash
# 전체 로그 수집 및 전처리
python preprocessing/batch_preprocessor.py \
    --log-dirs "gateway:logs/gateway,eureka:logs/eureka,..." \
    --output-dir preprocessing/output \
    --start-date 2025-02-24 \
    --end-date 2025-12-08
```

### 실시간 전처리 (운영용)

```bash
# 실시간 로그 처리 및 이상 탐지
python preprocessing/streaming_preprocessor.py \
    --log-file logs/gateway/gateway.log \
    --service-name gateway \
    --follow \
    --detector-config config/ensemble_config.yaml
```

---

## 🔧 실시간 전처리 최적화

### 1. 메모리 관리

```python
# 버퍼 크기 제한
max_buffer_size = 10000

# 오래된 로그 자동 제거
cleanup_interval = 60  # 60초마다
```

### 2. 병렬 처리

```python
# 여러 서비스 로그를 병렬로 처리
from concurrent.futures import ThreadPoolExecutor

services = ['gateway', 'eureka', 'user', 'research', 'manager', 'code']

with ThreadPoolExecutor(max_workers=6) as executor:
    for service in services:
        log_file = Path(f'logs/{service}/{service}.log')
        executor.submit(
            process_service_stream,
            log_file,
            service
        )
```

### 3. 배치 처리

```python
# 작은 배치로 묶어서 처리 (성능 향상)
batch_size = 32

sessions_batch = []
for session in preprocessor.process_file_stream(log_file, service_name):
    sessions_batch.append(session)
    
    if len(sessions_batch) >= batch_size:
        # 배치 단위로 이상 탐지
        results = detector.predict_batch(sessions_batch)
        sessions_batch = []
```

---

## ✅ 체크리스트

### 배치 전처리
- [ ] 로그 수집 모듈 구현
- [ ] 로그 통합 모듈 구현
- [ ] 배치 전처리 파이프라인 구현
- [ ] 학습 데이터 생성

### 실시간 전처리
- [ ] 스트리밍 전처리 모듈 구현 ⭐
- [ ] 실시간 세션화 구현 ⭐
- [ ] 메모리 관리 로직 구현
- [ ] 실시간 이상 탐지와 연동

---

이 가이드를 따라 배치와 실시간 전처리를 분리하여 효율적인 시스템을 구축할 수 있습니다! 🚀
