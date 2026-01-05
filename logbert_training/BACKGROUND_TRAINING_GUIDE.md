# 백그라운드 학습 가이드

DGX 서버에서 LogBERT 학습을 백그라운드로 실행하는 방법입니다.

## 🚀 빠른 시작

### 백그라운드 실행 스크립트 사용 (권장)

```bash
cd /home/zzangdol/RADAR-1/logbert_training

# 실행 권한 부여 (처음 한 번만)
chmod +x run_training_background.sh

# 백그라운드 실행 (기본 설정: training_config_dgx.yaml)
./run_training_background.sh

# 또는 커스텀 설정
./run_training_background.sh --config training_config_dgx.yaml --batch-size 128
```

## 📋 실행 방법

### 방법 1: 백그라운드 실행 스크립트 (가장 쉬움)

```bash
./run_training_background.sh
```

**특징:**
- 자동으로 로그 파일 생성
- 프로세스 ID 저장
- 터미널 종료와 무관하게 실행
- 모니터링 명령어 자동 출력

### 방법 2: nohup 직접 사용

```bash
# 기본 실행
nohup python3 train_server.py --config training_config_dgx.yaml > training.log 2>&1 &

# 프로세스 ID 확인
echo $!

# 로그 확인
tail -f training.log
```

### 방법 3: screen 사용 (추천)

```bash
# screen 세션 시작
screen -S training

# 학습 실행
python3 train_server.py --config training_config_dgx.yaml

# 세션 분리: Ctrl+A, D
# 세션 재접속: screen -r training
# 세션 목록: screen -ls
```

### 방법 4: tmux 사용

```bash
# tmux 세션 시작
tmux new -s training

# 학습 실행
python3 train_server.py --config training_config_dgx.yaml

# 세션 분리: Ctrl+B, D
# 세션 재접속: tmux attach -t training
# 세션 목록: tmux ls
```

## 📊 모니터링

### 로그 확인

```bash
# 실시간 로그 확인
tail -f logs/training_*.log

# 최신 로그 확인
tail -n 100 logs/training_*.log

# 전체 로그 확인
cat logs/training_*.log
```

### 프로세스 확인

```bash
# PID 파일에서 프로세스 ID 확인
cat logs/training_*.pid

# 프로세스 상태 확인
ps -p $(cat logs/training_*.pid)

# 프로세스 상세 정보
ps aux | grep train_server.py
```

### GPU 사용량 확인

```bash
# 실시간 GPU 사용량 모니터링
watch -n 1 nvidia-smi

# 또는
nvidia-smi -l 1

# 특정 프로세스의 GPU 사용량
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### 학습 진행 상황 확인

```bash
# 체크포인트 확인
ls -lh checkpoints/checkpoints/

# 최신 체크포인트 확인
ls -lt checkpoints/checkpoints/ | head -5

# 체크포인트 크기 확인
du -sh checkpoints/checkpoints/
```

## 🛠️ 프로세스 관리

### 프로세스 종료

```bash
# 정상 종료 (권장)
kill $(cat logs/training_*.pid)

# 강제 종료 (필요시)
kill -9 $(cat logs/training_*.pid)

# 또는 프로세스 ID 직접 지정
kill <PID>
```

### 프로세스 일시 중지/재개

```bash
# 일시 중지
kill -STOP $(cat logs/training_*.pid)

# 재개
kill -CONT $(cat logs/training_*.pid)
```

### 프로세스 우선순위 조정

```bash
# 낮은 우선순위로 실행 (다른 작업에 영향 최소화)
nice -n 19 python3 train_server.py --config training_config_dgx.yaml

# 또는 실행 중인 프로세스 우선순위 변경
renice -n 19 -p $(cat logs/training_*.pid)
```

## 📁 파일 구조

백그라운드 실행 후 생성되는 파일:

```
logbert_training/
├── logs/
│   ├── training_20260102_182211.log    # 학습 로그
│   └── training_20260102_182211.pid    # 프로세스 ID
├── checkpoints/
│   └── checkpoints/
│       ├── best_model.pt              # 최고 성능 모델
│       ├── epoch_*.pt                  # 에폭별 체크포인트
│       └── checkpoint_step_*.pt       # 스텝별 체크포인트
└── ...
```

## 💡 유용한 명령어

### 한 번에 확인

```bash
# 프로세스, GPU, 로그 한 번에 확인
echo "=== 프로세스 ===" && \
ps -p $(cat logs/training_*.pid 2>/dev/null) && \
echo "" && \
echo "=== GPU ===" && \
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv && \
echo "" && \
echo "=== 최신 로그 (마지막 10줄) ===" && \
tail -n 10 logs/training_*.log
```

### 로그에서 중요한 정보 추출

```bash
# Loss 값만 확인
grep "loss" logs/training_*.log | tail -20

# 에러만 확인
grep -i "error\|exception\|traceback" logs/training_*.log

# 체크포인트 저장 확인
grep "체크포인트 저장" logs/training_*.log
```

### 학습 시간 추정

```bash
# 시작 시간 확인
grep "학습 시작" logs/training_*.log

# 현재 진행 상황 확인
grep "Epoch\|Step\|loss" logs/training_*.log | tail -5
```

## ⚠️ 주의사항

1. **터미널 종료**: `nohup`이나 `screen`/`tmux`를 사용하지 않으면 터미널 종료 시 프로세스가 종료됩니다.

2. **로그 파일 크기**: 장시간 실행 시 로그 파일이 커질 수 있습니다. 주기적으로 확인하세요.

3. **디스크 공간**: 체크포인트 파일이 많이 쌓이면 디스크 공간을 차지합니다. 주기적으로 정리하세요.

4. **네트워크 연결**: SSH 연결이 끊겨도 `nohup`으로 실행한 프로세스는 계속 실행됩니다.

## 🔍 문제 해결

### 프로세스가 실행되지 않음

```bash
# 로그 확인
cat logs/training_*.log

# Python 경로 확인
which python3

# 의존성 확인
python3 -c "import torch; import transformers; print('OK')"
```

### GPU를 사용하지 않음

```bash
# CUDA 확인
python3 -c "import torch; print(torch.cuda.is_available())"

# GPU 확인
nvidia-smi
```

### 메모리 부족

```bash
# 메모리 사용량 확인
free -h

# GPU 메모리 확인
nvidia-smi

# 배치 크기 줄이기
# 로그 파일에서 배치 크기 확인 후 재실행
```

## 📝 예시: 전체 워크플로우

```bash
# 1. 디렉토리 이동
cd /home/zzangdol/RADAR-1/logbert_training

# 2. 실행 권한 부여
chmod +x run_training_background.sh

# 3. 백그라운드 실행
./run_training_background.sh

# 4. 다른 터미널에서 모니터링
# 터미널 2: 로그 확인
tail -f logs/training_*.log

# 터미널 3: GPU 확인
watch -n 1 nvidia-smi

# 5. 학습 완료 후 확인
ls -lh checkpoints/checkpoints/
```

## 🎯 결론

**가장 간단한 방법:**

```bash
cd /home/zzangdol/RADAR-1/logbert_training
chmod +x run_training_background.sh
./run_training_background.sh
```

그 다음 다른 터미널에서:

```bash
# 로그 확인
tail -f logs/training_*.log

# GPU 확인
watch -n 1 nvidia-smi
```

이제 학습이 백그라운드에서 실행됩니다! 🚀


