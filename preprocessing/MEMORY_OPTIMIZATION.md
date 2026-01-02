# 메모리 최적화 및 관계 추적 기능

## 개선 사항

### 1. 스트리밍 처리 모드 (메모리 효율성)

#### 문제점
- 기존 방식: 모든 세션을 메모리에 저장 후 한 번에 파일에 쓰기
- 대용량 로그 파일 처리 시 메모리 부족 발생 가능

#### 해결 방법
- **스트리밍 모드**: 세션이 완성되는 즉시 파일에 저장하고 메모리에서 해제
- 메모리 사용량을 최소화하여 대용량 파일도 안정적으로 처리

#### 사용 방법

```yaml
# preprocessing_config.yaml
stream_mode: true  # 스트리밍 모드 활성화
```

#### 동작 방식
```
로그 읽기 → 파싱 → 세션 완성 → 즉시 파일 저장 → 메모리 해제
```

### 2. 날짜/시스템명 기반 관계 추적

#### 목적
같은 시간대에 발생한 다른 서비스(gateway, eureka, manager 등)의 로그를 연결하여 전체적인 흐름을 파악

#### 동작 원리
1. **시간 윈도우 그룹화**: 5분 단위로 로그를 그룹화
2. **서비스별 수집**: 각 서비스의 로그를 시간 윈도우별로 수집
3. **관계 정보 추가**: 같은 시간 윈도우의 서비스들을 `related_services`에 기록

#### 예시

```
시간 윈도우: 2025-12-08_14_05 (2025-12-08 14:05~14:10)

Gateway 로그:
  - Request received from 192.168.0.1
  - Routing to manager service

Manager 로그:
  - Processing request
  - Database query executed

Eureka 로그:
  - Service discovery update

→ 이 세 로그는 같은 시간 윈도우에 발생하여 관계가 있음
```

#### 출력 형식

```json
{
  "session_id": 0,
  "service_name": "gateway",
  "time_window": "2025-12-08_14_05",
  "related_services": ["gateway", "manager", "eureka"],
  "correlation_id": "2025-12-08_14_05_gateway",
  ...
}
```

#### 사용 방법

```yaml
# preprocessing_config.yaml
enable_correlation: true  # 관계 추적 활성화
```

## 설정 옵션

### preprocessing_config.yaml

```yaml
# 메모리 효율성 설정
stream_mode: true          # 스트리밍 모드 (기본: true)
enable_correlation: true  # 관계 추적 (기본: true)
```

### 옵션 설명

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `stream_mode` | `true` | 세션 완성 시 즉시 파일에 저장 (메모리 절약) |
| `enable_correlation` | `true` | 날짜/시스템명 기반 관계 추적 활성화 |

## 메모리 사용량 비교

### 기존 방식 (stream_mode: false)
```
메모리 사용량 = 모든 세션 데이터 크기
예: 10,000개 세션 × 50KB = 약 500MB
```

### 스트리밍 방식 (stream_mode: true)
```
메모리 사용량 = 현재 처리 중인 세션만 유지
예: 약 1-2개 세션 × 50KB = 약 50-100KB
```

**메모리 절약률: 약 99%**

## 관계 추적 활용 예시

### 장애 분석 시나리오

1. **Gateway에서 에러 발생**
   ```json
   {
     "service_name": "gateway",
     "has_error": true,
     "time_window": "2025-12-08_14_05",
     "related_services": ["gateway", "manager", "eureka"]
   }
   ```

2. **같은 시간 윈도우의 다른 서비스 로그 확인**
   - Manager: 정상 동작
   - Eureka: 서비스 등록 문제 발견

3. **원인 분석**
   - Eureka의 서비스 등록 문제가 Gateway 에러의 원인일 가능성

## 성능 영향

### 스트리밍 모드
- **메모리**: 99% 절약
- **처리 속도**: 거의 동일 (파일 I/O 오버헤드 미미)
- **디스크 사용**: 동일

### 관계 추적 모드
- **메모리**: 약간 증가 (시간 윈도우별 버퍼링)
- **처리 속도**: 약간 느려짐 (그룹화 작업)
- **추가 정보**: 관계 정보 제공

## 권장 설정

### 대용량 로그 파일
```yaml
stream_mode: true
enable_correlation: false  # 메모리 절약 우선
```

### 관계 분석이 중요한 경우
```yaml
stream_mode: true
enable_correlation: true  # 관계 추적 활성화
```

### 메모리가 충분한 경우
```yaml
stream_mode: false
enable_correlation: true
```

## 주의사항

1. **스트리밍 모드**: 출력 파일이 이미 존재하면 덮어쓰기
2. **관계 추적**: 시간 윈도우 단위(5분)로 그룹화되므로 정확한 시간 매칭이 아닐 수 있음
3. **메모리**: 관계 추적 모드도 시간 윈도우별 버퍼링이 필요하므로 완전한 스트리밍은 아님

