# 치명도 계산 방법

## 📋 개요

MSA 환경에서 탐지된 이상의 치명도를 계산하여 우선순위를 결정합니다.

---

## 🎯 치명도 계산 공식

### 기본 공식

```
Severity Score = (Anomaly Score × Anomaly Weight) 
               + (Error Count × Error Weight)
               + (Warning Count × Warning Weight)
               + (Service Importance × Service Weight)
               + (Trace Impact × Impact Weight)
```

### 가중치 설정

```python
WEIGHTS = {
    'anomaly': 0.4,      # 이상 점수 가중치 (40%)
    'error': 0.3,        # 에러 개수 가중치 (30%)
    'warning': 0.1,     # 경고 개수 가중치 (10%)
    'service': 0.15,     # 서비스 중요도 가중치 (15%)
    'impact': 0.05      # Trace 영향도 가중치 (5%)
}
```

---

## 📊 치명도 등급

### 등급 분류

| 등급 | 점수 범위 | 설명 | 조치 |
|------|----------|------|------|
| **CRITICAL** | 0.8 ~ 1.0 | 즉시 조치 필요 | 즉시 알림, 긴급 대응 |
| **HIGH** | 0.6 ~ 0.8 | 빠른 조치 필요 | 1시간 내 대응 |
| **MEDIUM** | 0.4 ~ 0.6 | 일반 조치 필요 | 당일 대응 |
| **LOW** | 0.0 ~ 0.4 | 모니터링 필요 | 주기적 확인 |

---

## 🔧 세부 계산 방법

### 1. 이상 점수 (Anomaly Score)

**정규화:**
```python
normalized_anomaly = min(anomaly_score / max_anomaly_score, 1.0)
```

**가중치 적용:**
```python
anomaly_component = normalized_anomaly * WEIGHTS['anomaly']
```

---

### 2. 에러 개수 (Error Count)

**정규화:**
```python
# 최대 에러 개수 기준 (예: 10개)
normalized_errors = min(error_count / 10.0, 1.0)
```

**가중치 적용:**
```python
error_component = normalized_errors * WEIGHTS['error']
```

---

### 3. 경고 개수 (Warning Count)

**정규화:**
```python
# 최대 경고 개수 기준 (예: 20개)
normalized_warnings = min(warning_count / 20.0, 1.0)
```

**가중치 적용:**
```python
warning_component = normalized_warnings * WEIGHTS['warning']
```

---

### 4. 서비스 중요도 (Service Importance)

**서비스별 가중치:**

```yaml
service_weights:
  gateway: 1.0      # 가장 중요 (진입점)
  eureka: 0.9       # 서비스 디스커버리
  manager: 0.8     # 관리 서비스
  research: 0.7    # 연구 서비스
  user: 0.7        # 사용자 서비스
  code: 0.6        # 코드 서비스
```

**계산:**
```python
service_component = service_weight * WEIGHTS['service']
```

---

### 5. Trace 영향도 (Trace Impact)

**계산 방법:**
```python
# 관련된 서비스 수에 비례
impact_score = min(affected_services_count / 6.0, 1.0)
```

**가중치 적용:**
```python
impact_component = impact_score * WEIGHTS['impact']
```

---

## 💻 구현 예시

```python
class SeverityCalculator:
    """치명도 계산 클래스"""
    
    WEIGHTS = {
        'anomaly': 0.4,
        'error': 0.3,
        'warning': 0.1,
        'service': 0.15,
        'impact': 0.05
    }
    
    SERVICE_WEIGHTS = {
        'gateway': 1.0,
        'eureka': 0.9,
        'manager': 0.8,
        'research': 0.7,
        'user': 0.7,
        'code': 0.6
    }
    
    def calculate_severity(
        self,
        anomaly_score: float,
        error_count: int,
        warning_count: int,
        service_name: str,
        affected_services: List[str]
    ) -> Dict[str, Any]:
        """치명도 계산"""
        
        # 1. 이상 점수 정규화
        normalized_anomaly = min(anomaly_score / 1.0, 1.0)
        anomaly_component = normalized_anomaly * self.WEIGHTS['anomaly']
        
        # 2. 에러 개수 정규화
        normalized_errors = min(error_count / 10.0, 1.0)
        error_component = normalized_errors * self.WEIGHTS['error']
        
        # 3. 경고 개수 정규화
        normalized_warnings = min(warning_count / 20.0, 1.0)
        warning_component = normalized_warnings * self.WEIGHTS['warning']
        
        # 4. 서비스 중요도
        service_weight = self.SERVICE_WEIGHTS.get(service_name, 0.5)
        service_component = service_weight * self.WEIGHTS['service']
        
        # 5. Trace 영향도
        impact_score = min(len(affected_services) / 6.0, 1.0)
        impact_component = impact_score * self.WEIGHTS['impact']
        
        # 최종 치명도 점수
        severity_score = (
            anomaly_component +
            error_component +
            warning_component +
            service_component +
            impact_component
        )
        
        # 등급 결정
        severity_level = self._determine_level(severity_score)
        
        return {
            'severity_score': round(severity_score, 4),
            'severity_level': severity_level,
            'components': {
                'anomaly': round(anomaly_component, 4),
                'error': round(error_component, 4),
                'warning': round(warning_component, 4),
                'service': round(service_component, 4),
                'impact': round(impact_component, 4)
            }
        }
    
    def _determine_level(self, score: float) -> str:
        """치명도 등급 결정"""
        if score >= 0.8:
            return 'CRITICAL'
        elif score >= 0.6:
            return 'HIGH'
        elif score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
```

---

## 📈 예시 계산

### 예시 1: Gateway 서비스 이상

```python
anomaly_score = 0.85
error_count = 3
warning_count = 1
service_name = 'gateway'
affected_services = ['gateway', 'research', 'manager']

# 계산
anomaly_component = 0.85 * 0.4 = 0.34
error_component = 0.3 * 0.3 = 0.09
warning_component = 0.05 * 0.1 = 0.005
service_component = 1.0 * 0.15 = 0.15
impact_component = 0.5 * 0.05 = 0.025

severity_score = 0.34 + 0.09 + 0.005 + 0.15 + 0.025 = 0.61
severity_level = 'HIGH'
```

### 예시 2: Code 서비스 이상

```python
anomaly_score = 0.6
error_count = 1
warning_count = 0
service_name = 'code'
affected_services = ['code']

# 계산
anomaly_component = 0.6 * 0.4 = 0.24
error_component = 0.1 * 0.3 = 0.03
warning_component = 0.0 * 0.1 = 0.0
service_component = 0.6 * 0.15 = 0.09
impact_component = 0.17 * 0.05 = 0.0085

severity_score = 0.24 + 0.03 + 0.0 + 0.09 + 0.0085 = 0.3685
severity_level = 'LOW'
```

---

## 🔧 고급 기능

### 1. 시간 가중치

최근 발생한 이상에 더 높은 가중치 부여:

```python
time_weight = 1.0 - (hours_ago / 24.0)  # 24시간 내
severity_score *= time_weight
```

### 2. 반복 발생 가중치

같은 이상이 반복 발생하면 가중치 증가:

```python
if repeat_count > 1:
    repeat_weight = 1.0 + (repeat_count - 1) * 0.1
    severity_score *= repeat_weight
```

### 3. 서비스 의존성 가중치

다른 중요한 서비스에 영향을 주는 경우:

```python
if 'gateway' in affected_services:
    dependency_weight = 1.2
    severity_score *= dependency_weight
```

---

## 📝 설정 파일

### `config/severity_config.yaml`

```yaml
weights:
  anomaly: 0.4
  error: 0.3
  warning: 0.1
  service: 0.15
  impact: 0.05

service_weights:
  gateway: 1.0
  eureka: 0.9
  manager: 0.8
  research: 0.7
  user: 0.7
  code: 0.6

thresholds:
  critical: 0.8
  high: 0.6
  medium: 0.4
  low: 0.0

normalization:
  max_errors: 10
  max_warnings: 20
  max_services: 6
```

---

이 방법으로 치명도를 계산하여 우선순위를 결정할 수 있습니다! 🎯
