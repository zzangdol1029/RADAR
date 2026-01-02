# 날짜별 배치 처리 가이드

## 문제점

### 메모리 사용량 문제
- **현재 상황**: 29GB 메모리 사용
- **원인**: 관계 추적 모드에서 모든 로그를 메모리에 수집
- **결과**: 대용량 로그 처리 시 메모리 부족 또는 매우 느린 처리

## 해결 방법: 날짜별 배치 처리

### 개선 사항

1. **날짜별 분할 처리**
   - 각 날짜별로 로그를 수집하고 처리
   - 날짜 처리 완료 후 메모리 해제
   - 메모리 사용량을 날짜 단위로 제한

2. **스트리밍 모드 유지**
   - 세션 완성 시 즉시 파일에 저장
   - 메모리에서 즉시 해제

3. **날짜 필터링**
   - 특정 날짜만 처리 가능
   - 점진적 처리 가능

## 사용 방법

### 1. 전체 로그를 날짜별로 처리 (권장)

```yaml
# preprocessing_config.yaml
batch_by_date: true       # 날짜별 배치 처리 활성화
date_filter: null         # 전체 날짜 처리
```

**실행:**
```bash
python log_preprocessor.py
```

**결과:**
- 각 날짜별로 별도 파일 생성: `preprocessed_logs_2025-12-08.json`
- 메모리 사용량: 날짜당 약 1-2GB (날짜별로 해제)

### 2. 특정 날짜만 처리

```yaml
# preprocessing_config.yaml
batch_by_date: true
date_filter: "2025-12-08"  # 특정 날짜만 처리
```

**실행:**
```bash
python log_preprocessor.py
```

**결과:**
- 해당 날짜의 로그만 처리
- 빠른 테스트 및 검증 가능

### 3. 날짜별 순차 처리 (스크립트)

```bash
# 날짜 목록 추출
python -c "
from pathlib import Path
from preprocessing.log_preprocessor import LogPreprocessor, load_config
import yaml

config = load_config()
preprocessor = LogPreprocessor(config)
log_files = list(Path('logs').glob('*.log'))
dates = set()
for f in log_files[:10]:  # 샘플
    dates.update(preprocessor._extract_dates_from_file(f))
for d in sorted(dates):
    print(d)
"

# 각 날짜별로 처리
for date in $(python extract_dates.py); do
    echo "Processing $date..."
    # 설정 파일 수정
    python -c "
import yaml
with open('preprocessing_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['date_filter'] = '$date'
with open('preprocessing_config.yaml', 'w') as f:
    yaml.dump(config, f)
"
    python log_preprocessor.py
done
```

## 메모리 사용량 비교

### 기존 방식 (관계 추적, 전체 메모리)
```
메모리 사용량: 29GB
처리 시간: 매우 느림 (2-4일 예상)
```

### 날짜별 배치 처리
```
메모리 사용량: 날짜당 1-2GB (최대)
처리 시간: 날짜당 1-2시간
전체 시간: 날짜 수 × 1-2시간
```

**메모리 절약률: 약 90-95%**

## 처리 흐름

### 날짜별 배치 처리 흐름

```
1. 로그 파일 스캔
   ↓
2. 날짜별로 파일 분류
   ↓
3. 날짜 1 처리
   - 해당 날짜 로그만 수집
   - 세션 생성 및 저장
   - 메모리 해제
   ↓
4. 날짜 2 처리
   - 해당 날짜 로그만 수집
   - 세션 생성 및 저장
   - 메모리 해제
   ↓
5. 반복...
```

## 출력 파일 구조

### 날짜별 배치 처리 시

```
preprocessed_logs_2025-12-01.json
preprocessed_logs_2025-12-02.json
preprocessed_logs_2025-12-03.json
...
```

### 파일 병합 (선택사항)

```python
import json
from pathlib import Path

output_dir = Path('preprocessing')
date_files = sorted(output_dir.glob('preprocessed_logs_*.json'))

all_sessions = []
for date_file in date_files:
    with open(date_file, 'r', encoding='utf-8') as f:
        sessions = json.load(f)
        all_sessions.extend(sessions)

with open('preprocessed_logs_merged.json', 'w', encoding='utf-8') as f:
    json.dump(all_sessions, f, ensure_ascii=False, indent=2)
```

## 성능 최적화 팁

### 1. 날짜별 병렬 처리

여러 날짜를 동시에 처리 (CPU 코어 수만큼):

```python
from multiprocessing import Pool
from preprocessing.log_preprocessor import LogPreprocessor, load_config

def process_date(date):
    config = load_config()
    config['date_filter'] = date
    preprocessor = LogPreprocessor(config)
    return preprocessor.process_log_directory(
        config['log_directory'],
        output_file=f"preprocessed_logs_{date}.json",
        date_filter=date
    )

dates = ['2025-12-01', '2025-12-02', '2025-12-03', '2025-12-04']
with Pool(processes=4) as pool:
    pool.map(process_date, dates)
```

### 2. 관계 추적 비활성화 (더 빠름)

```yaml
enable_correlation: false  # 관계 추적 비활성화
batch_by_date: true
```

**효과**: 약 30-50% 속도 향상

### 3. 최근 날짜만 처리

```yaml
date_filter: "2025-12-29"  # 최근 날짜만
```

## 권장 설정

### 대용량 로그 처리 (현재 상황)

```yaml
# preprocessing_config.yaml
stream_mode: true
enable_correlation: true
batch_by_date: true        # 필수!
date_filter: null          # 또는 특정 날짜
```

### 빠른 테스트

```yaml
stream_mode: true
enable_correlation: false  # 관계 추적 비활성화로 더 빠름
batch_by_date: true
date_filter: "2025-12-29"  # 최근 1일만
```

## 예상 처리 시간

### 날짜별 처리 시간

- **소규모 날짜** (100만 줄 이하): 약 30분-1시간
- **중규모 날짜** (100만-500만 줄): 약 1-2시간
- **대규모 날짜** (500만 줄 이상): 약 2-4시간

### 전체 처리 시간

- **날짜 수**: 예를 들어 30일
- **날짜당 평균**: 1.5시간
- **전체 예상**: 30 × 1.5 = 45시간 (약 2일)

**기존 방식 대비 약 50% 시간 단축 + 메모리 90% 절약**

## 주의사항

1. **날짜별 파일 생성**: 각 날짜마다 별도 파일이 생성됨
2. **디스크 공간**: 날짜별 파일이 필요하므로 충분한 공간 확보
3. **중단/재개**: 날짜별로 처리되므로 중단 후 재개 가능

## 문제 해결

### 메모리 여전히 높은 경우

1. **날짜 필터 사용**: 특정 날짜만 처리
2. **관계 추적 비활성화**: `enable_correlation: false`
3. **윈도우 크기 감소**: `window_size: 30` (50 → 30)

### 처리 속도가 느린 경우

1. **관계 추적 비활성화**: 가장 큰 효과
2. **Drain3 설정 조정**: `drain_sim_th: 0.5`
3. **병렬 처리**: 여러 날짜 동시 처리

