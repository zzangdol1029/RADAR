# 10개 파일 학습 실행 가이드

**설정 파일**: `configs/test_xpu.yaml`  
**환경**: Intel Arc Graphics (XPU)  
**데이터**: 10개 파일  
**예상 시간**: ~20시간

---

## ⚙️ 최적화된 설정

### 주요 변경사항 (1개 파일 테스트 기반)

| 설정 | 원본 | 최적화 | 이유 |
|------|------|--------|------|
| `batch_size` | 32 | **16** ✅ | Quick test에서 안정성 검증 |
| `num_workers` | 4 | **0** ✅ | 메모리 절약 (검증됨) |
| `num_epochs` | 3 | **2** ⚖️ | 학습량 균형 (10파일 × 2 = 20배) |
| `save_interval` | 500 | **1000** | 체크포인트 관리 효율화 |

### 예상 학습량 비교

```
Quick Test (완료):
  1개 파일 × 1 epoch = 5,156 배치
  시간: 3시간 5분

10-File Test (예정):
  10개 파일 × 2 epoch = 약 103,120 배치
  시간: 약 20시간 (3.05h × 10파일 × 2epoch / 1.5 효율)
```

---

## 🚀 실행 방법

### Option 1: PowerShell 스크립트 (권장)
```powershell
cd C:\workspace\RADAR\model_training\logbert
conda activate logbert_ipex

# 실행
python scripts/train_intel.py --config configs/test_xpu.yaml
```

### Option 2: 백그라운드 실행 (장시간 학습용)
```powershell
# nohup 대신 PowerShell Job으로 실행
cd C:\workspace\RADAR\model_training\logbert
conda activate logbert_ipex

# 백그라운드 실행
Start-Job -ScriptBlock {
    cd C:\workspace\RADAR\model_training\logbert
    conda activate logbert_ipex
    python scripts/train_intel.py --config configs/test_xpu.yaml
}

# Job 상태 확인
Get-Job

# 로그 확인
Get-Job -Id <ID> | Receive-Job
```

### Option 3: tmux/screen 사용 (WSL 환경)
```bash
# tmux 세션 시작
tmux new -s logbert_10files

# 학습 실행
cd /mnt/c/workspace/RADAR/model_training/logbert
conda activate logbert_ipex
python scripts/train_intel.py --config configs/test_xpu.yaml

# Detach: Ctrl+B, D
# Reattach: tmux attach -t logbert_10files
```

---

## 📊 모니터링

### 로그 파일 확인
```powershell
# 실시간 로그 모니터링
Get-Content logs\train_*.log -Wait -Tail 50

# 최근 로그 확인
Get-Content logs\train_*.log -Tail 100
```

### GPU 사용량 모니터링
```powershell
# Intel GPU 모니터링
# 작업 관리자 > 성능 > GPU 0 (Intel Arc Graphics)

# 또는 PowerShell에서
while ($true) {
    Get-Counter "\GPU Engine(*)\Utilization Percentage"
    Start-Sleep -Seconds 5
}
```

---

## 📈 예상 진행 상황

### Phase 1: Epoch 1 (0-10시간)
```
10개 파일을 1번 학습
약 51,560 배치 처리
Loss: 4.7 → 1.5 (예상)
체크포인트: checkpoint_step_1000.pt ~ checkpoint_step_51000.pt (1000 간격)
```

### Phase 2: Epoch 2 (10-20시간)
```
같은 10개 파일을 다시 학습
약 51,560 배치 추가 처리
Loss: 1.5 → 0.7~0.9 (예상)
최종 체크포인트: best_model.pt, epoch_2.pt
```

---

## 💾 체크포인트 관리

### 예상 저장 용량
```
체크포인트: 1000 step마다 저장
총 103개 체크포인트 (약 113GB)
+ best_model.pt (~1.1GB)
+ epoch_1.pt, epoch_2.pt (~2.2GB)

총 저장 공간: ~120GB
```

### 용량 절약 팁
```powershell
# 학습 완료 후 중간 체크포인트 정리 (선택)
cd checkpoints_test_xpu
# best_model.pt와 epoch_*.pt만 남기고 삭제
Remove-Item checkpoint_step_*.pt
```

---

## ⚠️ 주의사항

### 1. 메모리 모니터링
- 학습 시작 후 처음 1시간 동안 **메모리 사용량** 확인
- GPU 메모리 16GB 중 14~15GB 사용 예상
- OOM 에러 발생 시 → `batch_size: 8`로 재시작

### 2. 디스크 공간
- 최소 **150GB 여유 공간** 필요
- 체크포인트 + 로그 + 임시 파일

### 3. 전원 관리
- **절전 모드 비활성화**
- **화면 보호기 비활성화**
- 20시간 연속 실행 보장

### 4. 중단 대비
```
학습 중단 시 복구 방법:
python scripts/train_intel.py \
  --config configs/test_xpu.yaml \
  --resume checkpoints_test_xpu/checkpoint_step_XXXXX.pt
```

---

## 🎯 성공 기준

### Loss 목표
- **Epoch 1 종료**: Loss < 1.5
- **Epoch 2 종료**: Loss < 0.9
- **최종 목표**: Loss ≈ 0.7~0.8 (Quick test 0.91보다 개선)

### 학습 완료 확인
```
로그에서 확인:
✅ 학습 완료!
최고 Loss: 0.XX
================================================================================
```

---

## 📋 체크리스트

학습 시작 전:
- [ ] `conda activate logbert_ipex` 확인
- [ ] XPU 사용 가능 확인: `torch.xpu.is_available()` = True
- [ ] 디스크 여유 공간 150GB+ 확인
- [ ] 절전 모드 비활성화
- [ ] `test_xpu.yaml` 설정 확인

학습 중:
- [ ] 첫 1시간 메모리 모니터링
- [ ] 로그 정상 출력 확인
- [ ] GPU 사용률 70%+ 유지

학습 완료 후:
- [ ] `best_model.pt` 생성 확인
- [ ] Loss 값 기록
- [ ] 평가 스크립트 실행
- [ ] 결과 문서화

---

## 🔄 Quick Test와 비교

| 항목 | Quick Test (1 file) | 10-File Test | 비율 |
|------|-------------------|-------------|------|
| 파일 수 | 1 | 10 | 10× |
| Epoch | 1 | 2 | 2× |
| 총 배치 | 5,156 | ~103,120 | 20× |
| 시간 | 3h 5m | ~20h | 6.5× |
| 최종 Loss | 0.91 | 0.7~0.8 (목표) | 개선 |
| 체크포인트 | 12개 (~13GB) | ~103개 (~120GB) | 9× |

---

## 📞 트러블슈팅

### OOM (Out of Memory) 에러
```yaml
# test_xpu.yaml 수정
batch_size: 8  # 16 → 8
```

### 학습 속도 너무 느림
- GPU 사용률 확인
- 다른 프로그램 종료
- Intel Arc 드라이버 최신 버전 확인

### 체크포인트 저장 실패
- 디스크 공간 확인
- 쓰기 권한 확인

---

**준비되셨으면 학습을 시작하세요!** 🚀

```powershell
cd C:\workspace\RADAR\model_training\logbert
conda activate logbert_ipex
python scripts/train_intel.py --config configs/test_xpu.yaml
```
