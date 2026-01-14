# 로그 수집 및 통합 파이프라인 가이드

## 📋 개요

MSA 환경에서 여러 서비스의 로그를 수집하고 통합하여 전처리하는 전체 프로세스를 안내합니다.

---

## 🔄 전체 파이프라인

```
로그 수집 → 로그 통합 → 전처리 → 모델 학습
   ↓           ↓          ↓         ↓
 여러 서비스   하나로 합치기  정제/변환   학습 데이터
```

---

## 📊 로그 수집 단계

### 1. 로그 소스

**MSA 서비스:**
- `gateway`: API Gateway 로그
- `eureka`: Service Discovery 로그
- `user`: User Service 로그
- `research`: Research Service 로그
- `manager`: Manager Service 로그
- `code`: Code Service 로그

**로그 위치:**
```
logs/
├── gateway/
│   ├── gateway_2025-02-24.log
│   ├── gateway_2025-02-25.log
│   └── ...
├── eureka/
│   ├── eureka_2025-02-24.log
│   └── ...
├── user/
│   └── ...
├── research/
│   └── ...
├── manager/
│   └── ...
└── code/
    └── ...
```

---

## 🔧 로그 수집 모듈

### `preprocessing/log_collector.py`

```python
#!/usr/bin/env python3
"""
로그 수집 모듈
여러 서비스의 로그 파일을 수집하고 통합
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class LogCollector:
    """로그 수집 클래스"""
    
    def __init__(
        self,
        log_dirs: Dict[str, Path],
        output_dir: Path,
        date_range: Optional[tuple] = None
    ):
        """
        Args:
            log_dirs: 서비스별 로그 디렉토리
                {
                    'gateway': Path('logs/gateway'),
                    'eureka': Path('logs/eureka'),
                    ...
                }
            output_dir: 통합된 로그 저장 디렉토리
            date_range: 날짜 범위 (start_date, end_date) 또는 None (전체)
        """
        self.log_dirs = log_dirs
        self.output_dir = output_dir
        self.date_range = date_range
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_logs(
        self,
        service_name: str,
        log_pattern: str = "*.log"
    ) -> List[Dict[str, Any]]:
        """
        특정 서비스의 로그 수집
        
        Args:
            service_name: 서비스명
            log_pattern: 로그 파일 패턴
        
        Returns:
            로그 엔트리 리스트
        """
        log_dir = self.log_dirs.get(service_name)
        if not log_dir or not log_dir.exists():
            logger.warning(f"로그 디렉토리를 찾을 수 없습니다: {service_name}")
            return []
        
        log_files = sorted(log_dir.glob(log_pattern))
        all_logs = []
        
        for log_file in log_files:
            # 날짜 필터링
            if self.date_range:
                file_date = self._extract_date_from_filename(log_file)
                if file_date and not self._is_in_range(file_date):
                    continue
            
            logs = self._read_log_file(log_file, service_name)
            all_logs.extend(logs)
            logger.info(f"{service_name}: {log_file.name} - {len(logs)}개 로그 수집")
        
        return all_logs
    
    def _read_log_file(
        self,
        log_file: Path,
        service_name: str
    ) -> List[Dict[str, Any]]:
        """로그 파일 읽기"""
        logs = []
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    log_entry = self._parse_log_line(line, service_name, log_file.name, line_num)
                    if log_entry:
                        logs.append(log_entry)
        except Exception as e:
            logger.error(f"로그 파일 읽기 실패 ({log_file}): {e}")
        
        return logs
    
    def _parse_log_line(
        self,
        line: str,
        service_name: str,
        filename: str,
        line_num: int
    ) -> Optional[Dict[str, Any]]:
        """로그 라인 파싱"""
        # 기본 로그 형식: "2025-12-08 17:23:47.950 INFO ..."
        try:
            # 타임스탬프 추출
            timestamp_str = line[:23]  # "2025-12-08 17:23:47.950"
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
            
            # 레벨 추출
            level = None
            for log_level in ['ERROR', 'WARN', 'INFO', 'DEBUG', 'TRACE']:
                if log_level in line:
                    level = log_level
                    break
            
            # 메시지 추출
            message = line[24:].strip()
            
            return {
                'timestamp': timestamp,
                'level': level,
                'message': message,
                'service_name': service_name,
                'source_file': filename,
                'line_number': line_num,
                'raw_line': line
            }
        except Exception as e:
            logger.debug(f"로그 라인 파싱 실패 (line {line_num}): {e}")
            # 파싱 실패해도 기본 정보는 저장
            return {
                'timestamp': None,
                'level': None,
                'message': line,
                'service_name': service_name,
                'source_file': filename,
                'line_number': line_num,
                'raw_line': line
            }
    
    def _extract_date_from_filename(self, filepath: Path) -> Optional[datetime]:
        """파일명에서 날짜 추출"""
        # 예: "gateway_2025-02-24.log" -> 2025-02-24
        try:
            filename = filepath.stem
            date_str = filename.split('_')[-1]  # 마지막 부분
            return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            return None
    
    def _is_in_range(self, date: datetime) -> bool:
        """날짜가 범위 내에 있는지 확인"""
        if not self.date_range:
            return True
        
        start_date, end_date = self.date_range
        return start_date <= date <= end_date
    
    def collect_all_services(self) -> Dict[str, List[Dict[str, Any]]]:
        """모든 서비스의 로그 수집"""
        all_logs = {}
        
        for service_name in self.log_dirs.keys():
            logs = self.collect_logs(service_name)
            all_logs[service_name] = logs
            logger.info(f"{service_name}: 총 {len(logs):,}개 로그 수집 완료")
        
        return all_logs
    
    def merge_logs(
        self,
        all_logs: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        모든 서비스의 로그를 타임스탬프 기준으로 병합
        
        Args:
            all_logs: 서비스별 로그 딕셔너리
        
        Returns:
            타임스탬프 순으로 정렬된 통합 로그 리스트
        """
        merged_logs = []
        
        for service_name, logs in all_logs.items():
            merged_logs.extend(logs)
        
        # 타임스탬프 기준 정렬
        merged_logs.sort(key=lambda x: x.get('timestamp') or datetime.min)
        
        logger.info(f"통합된 로그 수: {len(merged_logs):,}개")
        
        return merged_logs
    
    def save_merged_logs(
        self,
        merged_logs: List[Dict[str, Any]],
        output_filename: str = "merged_logs.json"
    ):
        """통합된 로그 저장"""
        import json
        
        output_path = self.output_dir / output_filename
        
        # JSON으로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_logs, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"통합 로그 저장 완료: {output_path}")
        
        # 통계 정보 저장
        stats = self._calculate_statistics(merged_logs)
        stats_path = self.output_dir / "collection_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"통계 정보 저장 완료: {stats_path}")
    
    def _calculate_statistics(
        self,
        merged_logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """수집 통계 계산"""
        stats = {
            'total_logs': len(merged_logs),
            'by_service': defaultdict(int),
            'by_level': defaultdict(int),
            'date_range': {
                'start': None,
                'end': None
            }
        }
        
        timestamps = []
        
        for log in merged_logs:
            service = log.get('service_name', 'unknown')
            level = log.get('level', 'unknown')
            
            stats['by_service'][service] += 1
            stats['by_level'][level] += 1
            
            timestamp = log.get('timestamp')
            if timestamp:
                timestamps.append(timestamp)
        
        if timestamps:
            stats['date_range']['start'] = min(timestamps).isoformat()
            stats['date_range']['end'] = max(timestamps).isoformat()
        
        return dict(stats)


def main():
    """메인 함수"""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='로그 수집 및 통합')
    parser.add_argument('--log-dirs', type=str, required=True,
                       help='로그 디렉토리 (서비스:경로 형식, 쉼표로 구분)')
    parser.add_argument('--output-dir', type=str, default='logs/merged',
                       help='출력 디렉토리')
    parser.add_argument('--start-date', type=str, default=None,
                       help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='종료 날짜 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # 로그 디렉토리 파싱
    log_dirs = {}
    for item in args.log_dirs.split(','):
        service, path = item.split(':')
        log_dirs[service.strip()] = Path(path.strip())
    
    # 날짜 범위 설정
    date_range = None
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        date_range = (start_date, end_date)
    
    # 로그 수집
    collector = LogCollector(
        log_dirs=log_dirs,
        output_dir=Path(args.output_dir),
        date_range=date_range
    )
    
    # 모든 서비스 로그 수집
    all_logs = collector.collect_all_services()
    
    # 로그 병합
    merged_logs = collector.merge_logs(all_logs)
    
    # 저장
    collector.save_merged_logs(merged_logs)


if __name__ == '__main__':
    main()
```

---

## 🔄 로그 통합 프로세스

### 단계별 처리

1. **로그 수집**
   - 각 서비스 디렉토리에서 로그 파일 읽기
   - 날짜 범위 필터링 (선택사항)

2. **로그 파싱**
   - 타임스탬프 추출
   - 로그 레벨 추출 (ERROR, WARN, INFO, DEBUG)
   - 메시지 추출
   - 서비스명 태깅

3. **로그 병합**
   - 타임스탬프 기준 정렬
   - 서비스별 로그 통합
   - Trace ID 연결 (가능한 경우)

4. **통합 로그 저장**
   - JSON 형식으로 저장
   - 통계 정보 생성

---

## 📊 통합 로그 형식

### 저장 형식

```json
[
  {
    "timestamp": "2025-12-08T17:23:47.950000",
    "level": "INFO",
    "message": "Starting Application",
    "service_name": "gateway",
    "source_file": "gateway_2025-12-08.log",
    "line_number": 1,
    "raw_line": "2025-12-08 17:23:47.950 INFO ..."
  },
  {
    "timestamp": "2025-12-08T17:23:48.123000",
    "level": "WARN",
    "message": "Connection timeout",
    "service_name": "gateway",
    "source_file": "gateway_2025-12-08.log",
    "line_number": 2,
    "raw_line": "2025-12-08 17:23:48.123 WARN ..."
  },
  ...
]
```

---

## 🚀 사용 예시

### 기본 사용법

```bash
python preprocessing/log_collector.py \
    --log-dirs "gateway:logs/gateway,eureka:logs/eureka,user:logs/user,research:logs/research,manager:logs/manager,code:logs/code" \
    --output-dir logs/merged \
    --start-date 2025-02-24 \
    --end-date 2025-12-08
```

### Python 코드로 사용

```python
from preprocessing.log_collector import LogCollector
from pathlib import Path
from datetime import datetime

# 로그 디렉토리 설정
log_dirs = {
    'gateway': Path('logs/gateway'),
    'eureka': Path('logs/eureka'),
    'user': Path('logs/user'),
    'research': Path('logs/research'),
    'manager': Path('logs/manager'),
    'code': Path('logs/code')
}

# 날짜 범위 설정
date_range = (
    datetime(2025, 2, 24),
    datetime(2025, 12, 8)
)

# 로그 수집
collector = LogCollector(
    log_dirs=log_dirs,
    output_dir=Path('logs/merged'),
    date_range=date_range
)

# 모든 서비스 로그 수집
all_logs = collector.collect_all_services()

# 로그 병합
merged_logs = collector.merge_logs(all_logs)

# 저장
collector.save_merged_logs(merged_logs, 'merged_logs_2025-02-24_to_2025-12-08.json')
```

---

## 📈 통계 정보

### 수집 통계 예시

```json
{
  "total_logs": 2464136,
  "by_service": {
    "gateway": 500000,
    "eureka": 200000,
    "user": 400000,
    "research": 500000,
    "manager": 400000,
    "code": 464136
  },
  "by_level": {
    "INFO": 2000000,
    "WARN": 300000,
    "ERROR": 100000,
    "DEBUG": 64136
  },
  "date_range": {
    "start": "2025-02-24T00:00:00",
    "end": "2025-12-08T23:59:59"
  }
}
```

---

## 🔗 전처리 파이프라인과의 연결

### 전체 프로세스

```
1. 로그 수집 (log_collector.py)
   ↓
2. 로그 통합 (merge_logs)
   ↓
3. 전처리 (msa_preprocessor.py)
   ↓
4. 모델 학습 데이터 생성
```

### 통합 스크립트

```python
# 전체 파이프라인 실행
from preprocessing.log_collector import LogCollector
from preprocessing.msa_preprocessor import MSAPreprocessor

# 1. 로그 수집 및 통합
collector = LogCollector(...)
all_logs = collector.collect_all_services()
merged_logs = collector.merge_logs(all_logs)
collector.save_merged_logs(merged_logs)

# 2. 전처리
preprocessor = MSAPreprocessor(...)
preprocessed_data = preprocessor.preprocess(merged_logs)
preprocessor.save_preprocessed_data(preprocessed_data)
```

---

## 💡 최적화 팁

### 1. 증분 수집

```python
# 새로운 로그만 수집
collector = LogCollector(
    log_dirs=log_dirs,
    output_dir=output_dir,
    date_range=(last_collected_date, today)
)
```

### 2. 병렬 처리

```python
from concurrent.futures import ThreadPoolExecutor

# 여러 서비스 로그를 병렬로 수집
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {
        executor.submit(collector.collect_logs, service): service
        for service in log_dirs.keys()
    }
```

### 3. 압축 저장

```python
import gzip
import json

# 대용량 로그는 압축하여 저장
with gzip.open('merged_logs.json.gz', 'wt', encoding='utf-8') as f:
    json.dump(merged_logs, f, ensure_ascii=False, indent=2)
```

---

## ✅ 체크리스트

### 로그 수집 단계

- [ ] 각 서비스 로그 디렉토리 확인
- [ ] 로그 파일 형식 확인
- [ ] 날짜 범위 결정
- [ ] 로그 수집 스크립트 실행
- [ ] 통합 로그 검증
- [ ] 통계 정보 확인

### 전처리 단계

- [ ] 통합 로그 로드
- [ ] 전처리 파이프라인 실행
- [ ] 전처리된 데이터 검증
- [ ] 다른 멤버들에게 공유

---

이 가이드를 따라 로그 수집부터 전처리까지 전체 파이프라인을 구축할 수 있습니다! 🚀
