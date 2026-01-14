# 전처리 데이터 공유 프로토콜

## 📋 개요

멤버 1이 생성한 전처리된 데이터를 다른 멤버들이 사용할 수 있도록 공유하는 방법을 안내합니다.

---

## 📁 데이터 저장 위치

### 공유 디렉토리 구조

```
preprocessing/output/
├── preprocessed_logs_2025-02-24.json
├── preprocessed_logs_2025-02-25.json
├── preprocessed_logs_2025-02-26.json
└── ...
```

### 메타데이터 파일

```
preprocessing/output/
├── metadata.json          # 데이터 메타데이터
├── data_schema.json       # 데이터 스키마 정의
└── README.md              # 사용 가이드
```

---

## 📊 데이터 형식

### 세션 데이터 구조

```json
{
  "session_id": 0,
  "event_sequence": [1, 5, 1, 12, 3],
  "token_ids": [101, 1, 2, 1, 3, 4, 102, 0, 0, ...],
  "attention_mask": [1, 1, 1, 1, 1, 1, 1, 0, 0, ...],
  "has_error": false,
  "has_warn": true,
  "service_name": "gateway",
  "trace_id": "abc123",
  "original_logs": [
    "2025-12-08 17:23:47.950 INFO ...",
    "2025-12-08 17:23:48.123 WARN ..."
  ],
  "simplified_text": "[gateway] INFO [main] Starting Application | WARN ...",
  "timestamp": "2025-12-08T17:23:47",
  "session_length": 25
}
```

### 필수 필드

- `session_id`: 세션 고유 ID
- `token_ids`: 토큰 ID 시퀀스 (List[int])
- `attention_mask`: 어텐션 마스크 (List[int])
- `service_name`: 서비스명 (str)

### 선택 필드

- `event_sequence`: 이벤트 ID 시퀀스
- `has_error`: 에러 포함 여부
- `has_warn`: 경고 포함 여부
- `trace_id`: Trace ID
- `original_logs`: 원본 로그 리스트
- `simplified_text`: 간소화된 텍스트
- `timestamp`: 타임스탬프

---

## 🔄 데이터 공유 방법

### 방법 1: 공유 디렉토리 (서버 환경)

**멤버 1:**
```bash
# 전처리된 데이터를 공유 디렉토리에 저장
preprocessing/output/  # 모든 멤버가 접근 가능
```

**다른 멤버들:**
```bash
# 동일한 데이터 디렉토리 사용
python train_logbert.py --data-dir ../preprocessing/output
```

---

### 방법 2: Git LFS (대용량 파일)

**설정:**
```bash
# Git LFS 설치 및 설정
git lfs install
git lfs track "preprocessing/output/*.json"

# 전처리된 데이터 커밋
git add preprocessing/output/
git commit -m "전처리된 데이터 추가"
git push
```

**다른 멤버들:**
```bash
# Git LFS 파일 다운로드
git lfs pull
```

---

### 방법 3: 압축 파일 공유

**멤버 1:**
```bash
# 전처리된 데이터 압축
tar -czf preprocessed_data.tar.gz preprocessing/output/

# 또는 ZIP 형식
zip -r preprocessed_data.zip preprocessing/output/
```

**다른 멤버들:**
```bash
# 압축 해제
tar -xzf preprocessed_data.tar.gz
# 또는
unzip preprocessed_data.zip
```

---

### 방법 4: 클라우드 스토리지

**멤버 1:**
```bash
# AWS S3, Google Cloud Storage 등에 업로드
aws s3 cp preprocessing/output/ s3://bucket/preprocessed_data/ --recursive
```

**다른 멤버들:**
```bash
# 다운로드
aws s3 cp s3://bucket/preprocessed_data/ preprocessing/output/ --recursive
```

---

## 📝 메타데이터 파일

### `metadata.json`

```json
{
  "version": "1.0",
  "created_at": "2026-01-10T10:00:00",
  "created_by": "member1",
  "total_sessions": 2464136,
  "date_range": {
    "start": "2025-02-24",
    "end": "2025-12-08"
  },
  "services": ["gateway", "eureka", "research", "manager", "code"],
  "data_files": [
    "preprocessed_logs_2025-02-24.json",
    "preprocessed_logs_2025-02-25.json",
    ...
  ],
  "statistics": {
    "total_sessions": 2464136,
    "sessions_with_errors": 123456,
    "sessions_with_warnings": 234567,
    "sessions_with_trace_id": 2000000
  },
  "preprocessing_config": {
    "window_size": 20,
    "max_seq_length": 512,
    "vocab_size": 10000
  }
}
```

### `data_schema.json`

```json
{
  "session_schema": {
    "session_id": {
      "type": "integer",
      "required": true,
      "description": "세션 고유 ID"
    },
    "token_ids": {
      "type": "array",
      "items": {"type": "integer"},
      "required": true,
      "description": "토큰 ID 시퀀스",
      "max_length": 512
    },
    "attention_mask": {
      "type": "array",
      "items": {"type": "integer"},
      "required": true,
      "description": "어텐션 마스크",
      "max_length": 512
    },
    "service_name": {
      "type": "string",
      "required": true,
      "description": "서비스명",
      "enum": ["gateway", "eureka", "research", "manager", "code"]
    },
    "trace_id": {
      "type": "string",
      "required": false,
      "description": "Trace ID"
    }
  }
}
```

---

## ✅ 데이터 검증

### 멤버 1이 수행할 검증

```python
def validate_preprocessed_data(data_dir: Path):
    """전처리된 데이터 검증"""
    issues = []
    
    for json_file in data_dir.glob('preprocessed_*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            sessions = json.load(f)
        
        for i, session in enumerate(sessions):
            # 필수 필드 확인
            if 'token_ids' not in session:
                issues.append(f"{json_file.name}:{i} - token_ids 없음")
            if 'attention_mask' not in session:
                issues.append(f"{json_file.name}:{i} - attention_mask 없음")
            
            # 길이 일치 확인
            if len(session['token_ids']) != len(session['attention_mask']):
                issues.append(f"{json_file.name}:{i} - 길이 불일치")
            
            # 최대 길이 확인
            if len(session['token_ids']) > 512:
                issues.append(f"{json_file.name}:{i} - 길이 초과")
    
    if issues:
        print(f"경고: {len(issues)}개 문제 발견")
        for issue in issues[:10]:  # 처음 10개만 출력
            print(f"  - {issue}")
    else:
        print("✅ 데이터 검증 통과")
    
    return len(issues) == 0
```

---

## 📖 사용 가이드 (다른 멤버들을 위한)

### 데이터 로드 예시

```python
import json
from pathlib import Path
from typing import List, Dict, Any

def load_preprocessed_data(data_dir: Path) -> List[Dict[str, Any]]:
    """전처리된 데이터 로드"""
    all_sessions = []
    
    # 메타데이터 확인
    metadata_path = data_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            print(f"데이터 버전: {metadata['version']}")
            print(f"총 세션 수: {metadata['total_sessions']:,}")
    
    # 데이터 파일 로드
    json_files = sorted(data_dir.glob('preprocessed_*.json'))
    print(f"발견된 데이터 파일: {len(json_files)}개")
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            sessions = json.load(f)
            all_sessions.extend(sessions)
            print(f"  - {json_file.name}: {len(sessions):,}개 세션")
    
    print(f"총 로드된 세션: {len(all_sessions):,}개")
    return all_sessions

# 사용 예시
data_dir = Path('../preprocessing/output')
sessions = load_preprocessed_data(data_dir)
```

---

## 🔍 데이터 품질 확인

### 통계 정보 출력

```python
def print_data_statistics(sessions: List[Dict[str, Any]]):
    """데이터 통계 출력"""
    total = len(sessions)
    
    # 서비스별 통계
    service_counts = {}
    error_counts = {}
    trace_id_counts = {}
    
    for session in sessions:
        service = session.get('service_name', 'unknown')
        service_counts[service] = service_counts.get(service, 0) + 1
        
        if session.get('has_error', False):
            error_counts[service] = error_counts.get(service, 0) + 1
        
        if session.get('trace_id'):
            trace_id_counts[service] = trace_id_counts.get(service, 0) + 1
    
    print("=" * 80)
    print("데이터 통계")
    print("=" * 80)
    print(f"총 세션 수: {total:,}")
    print(f"\n서비스별 세션 수:")
    for service, count in sorted(service_counts.items()):
        error_rate = error_counts.get(service, 0) / count * 100
        trace_rate = trace_id_counts.get(service, 0) / count * 100
        print(f"  {service}: {count:,}개 (에러: {error_rate:.1f}%, Trace ID: {trace_rate:.1f}%)")
```

---

## 📅 데이터 공유 체크리스트

### 멤버 1 (전처리 담당)

- [ ] 전처리 파이프라인 완성
- [ ] 전처리된 데이터 생성
- [ ] 데이터 검증 수행
- [ ] 메타데이터 파일 생성 (`metadata.json`)
- [ ] 데이터 스키마 문서화 (`data_schema.json`)
- [ ] README 작성 (`README.md`)
- [ ] 데이터 공유 (공유 디렉토리/Git/압축 파일)
- [ ] 다른 멤버들에게 공유 완료 알림

### 멤버 2, 3, 4 (모델 학습 담당)

- [ ] 전처리된 데이터 확인
- [ ] 데이터 형식 검증
- [ ] 데이터 로드 스크립트 작성
- [ ] 모델 학습 스크립트 준비
- [ ] 학습 시작

---

## 💡 팁

### 1. 데이터 크기 관리

```python
# 대용량 데이터는 날짜별로 분할
# 각 파일이 100MB 이하가 되도록 조정
```

### 2. 버전 관리

```python
# 데이터 버전을 명시
metadata = {
    "version": "1.0",
    "created_at": "2026-01-10",
    "preprocessing_config": {...}
}
```

### 3. 증분 업데이트

```python
# 새로운 데이터만 추가
# 기존 데이터는 유지
```

---

이 프로토콜을 따라 효율적으로 데이터를 공유할 수 있습니다! 🚀
