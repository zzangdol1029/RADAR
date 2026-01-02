# 백그라운드 실행 가이드

## 화면보호기와 전처리 실행

### ✅ 화면보호기가 켜져도 동작합니다

- **Python 프로세스는 화면보호기와 무관하게 실행됩니다**
- 화면이 꺼져도 프로세스는 계속 실행됩니다
- 컴퓨터가 절전 모드로 들어가지 않는 한 계속 실행됩니다

### ⚠️ 주의사항

1. **터미널 종료**: 터미널을 닫으면 프로세스가 종료될 수 있습니다
2. **절전 모드**: 컴퓨터가 절전 모드로 들어가면 일시 중지됩니다
3. **로그인 세션**: 로그아웃하면 프로세스가 종료될 수 있습니다

## 안전한 백그라운드 실행 방법

### 방법 1: nohup 사용 (권장)

```bash
cd preprocessing
conda activate radar
nohup python log_preprocessor.py > preprocessing.log 2>&1 &
```

**장점:**
- 터미널을 닫아도 계속 실행
- 로그 파일에 출력 저장
- 프로세스 ID 확인 가능

**실행 로그 확인:**
```bash
tail -f preprocessing.log
```

**프로세스 확인:**
```bash
ps aux | grep log_preprocessor
```

### 방법 2: screen 사용

```bash
# screen 세션 시작
screen -S preprocessing

# 전처리 실행
cd preprocessing
conda activate radar
python log_preprocessor.py

# 세션 분리: Ctrl+A, D
# 세션 재접속: screen -r preprocessing
```

### 방법 3: tmux 사용

```bash
# tmux 세션 시작
tmux new -s preprocessing

# 전처리 실행
cd preprocessing
conda activate radar
python log_preprocessor.py

# 세션 분리: Ctrl+B, D
# 세션 재접속: tmux attach -t preprocessing
```

### 방법 4: 자동 스크립트 사용

```bash
cd preprocessing
./run_preprocessing_background.sh
```

이 스크립트는:
- nohup으로 실행
- 로그 파일 자동 생성
- 프로세스 ID 출력

## 프로세스 관리

### 실행 중인 프로세스 확인

```bash
# Python 프로세스 확인
ps aux | grep log_preprocessor

# 또는
pgrep -f log_preprocessor
```

### 프로세스 종료

```bash
# 프로세스 ID로 종료
kill <PID>

# 강제 종료
kill -9 <PID>

# 프로세스 이름으로 종료
pkill -f log_preprocessor
```

### 실행 상태 확인

```bash
# 로그 파일 실시간 확인
tail -f preprocessing/preprocessing_*.log

# 또는
tail -f preprocessing.log
```

## macOS 절전 모드 방지

전처리 중 컴퓨터가 절전 모드로 들어가지 않도록:

```bash
# 절전 모드 방지 (터미널에서)
caffeinate -d

# 특정 프로세스가 실행 중일 때만
caffeinate -w <PID>
```

## 권장 실행 방법

### 장기 실행 (수 시간 ~ 수 일)

```bash
cd preprocessing
conda activate radar

# nohup으로 실행
nohup python log_preprocessor.py > preprocessing_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 절전 모드 방지
caffeinate -w $!
```

### 단기 실행 (수 분 ~ 수 시간)

```bash
cd preprocessing
conda activate radar
python log_preprocessor.py
```

화면보호기는 괜찮지만, 터미널을 닫지 마세요.

## 로그 확인

### 실시간 로그 확인

```bash
tail -f preprocessing/preprocessing_*.log
```

### 특정 날짜 처리 확인

```bash
grep "날짜별 처리 시작" preprocessing_*.log
```

### 오류 확인

```bash
grep -i error preprocessing_*.log
```

## 문제 해결

### 프로세스가 멈춘 경우

```bash
# 프로세스 상태 확인
ps aux | grep log_preprocessor

# 로그 확인
tail -100 preprocessing_*.log
```

### 메모리 부족

```bash
# 메모리 사용량 확인
top -pid $(pgrep -f log_preprocessor)
```

### 재시작

```bash
# 기존 프로세스 종료
pkill -f log_preprocessor

# 다시 실행
cd preprocessing
conda activate radar
nohup python log_preprocessor.py > preprocessing_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```














