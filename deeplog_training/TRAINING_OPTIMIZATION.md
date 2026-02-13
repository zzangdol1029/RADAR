# DeepLog 학습 시간 분석 및 최적화 가이드

## 1. 현재 학습 시간 분석 (training.log 기준)

### 요약
| 항목 | 값 |
|------|-----|
| **예상 총 스텝** | 5,702,600 |
| **현재 진행** | Step ~843,200 (Epoch 9/50) |
| **로그 ETA** | **233시간 59분 (약 9.75일)** |
| **배치당 시간** | ~145ms |
| **처리량** | ~6.8 batch/s |

### 결론
**전체 학습이 약 10일 소요될 것으로 예상되며, 3일을 크게 초과합니다.**

---

## 2. GPU 활용 현황 (로그에서 확인된 문제)

- **GPU 0**: 메모리 2,599MB / 32,490MB (약 8%), 활용률 34~42%
- **GPU 1~3**: 메모리 22MB / 32,499MB (거의 미사용), 활용률 36~59%
- 4개 GPU 모두 **활용률 60% 미만**, **메모리도 매우 낮게** 사용 중
- DataParallel 사용 시 GPU 0이 gradient 수집/분배 담당으로 병목이 됨

---

## 3. 권장 개선 사항 (우선순위 순)

### 3-1. DistributedDataParallel(DDP) 전환 (가장 효과적)
**예상 효과: 2~3배 속도 향상**

- **현재**: `nn.DataParallel` → 단일 프로세스, GPU 0이 병목
- **개선**: `torch.nn.parallel.DistributedDataParallel` + `torchrun` (또는 `python -m torch.distributed.launch`)
- DDP는 GPU별로 독립 forward/backward 후 gradient만 동기화하므로 확장성이 좋음
- `train.py`에서 `use_multi_gpu`일 때 DDP 초기화 및 `DistributedSampler` 적용 필요

### 3-2. 배치 크기 증가
**예상 효과: GPU 활용률 상승, 1.2~1.5배 속도**

- **현재**: 전체 256 (GPU당 64), V100 32GB 기준으로 여유가 많음
- **권장**: 512 또는 768로 증가 후 OOM 여부 확인. 가능하면 1024까지 시도
- `config.yaml`의 `training.batch_size` 수정

### 3-3. 데이터 로딩 강화
**예상 효과: 데이터 병목 완화**

- **num_workers**: 4 → **8** (CPU 코어 여유 있으면 12~16도 시도)
- **prefetch_factor**: 2 → **4**
- 디스크 I/O가 병목이면 데이터를 **SSD**에 두거나, 일부 구간을 메모리 캐시하는 방식 검토
- `config.yaml`의 `training.num_workers`, `prefetch_factor` 수정

### 3-4. PyTorch 연산 최적화
**예상 효과: 5~15% 속도**

- 학습 루프 시작 전 추가:
  ```python
  torch.backends.cudnn.benchmark = True  # 연산 최적 알고리즘 자동 선택
  ```
- PyTorch 2.0+ 사용 시:
  ```python
  self.model = torch.compile(self.model)  # JIT 컴파일 (LSTM에서도 일부 이득)
  ```
- `torch.compile`은 처음 몇 스텝은 느릴 수 있으므로, 짧은 실행으로 먼저 검증 권장

### 3-5. 로깅/저장 간격 조정 (오버헤드 감소)
**예상 효과: 수 % 이내, 안정성·가독성 향상**

- **log_interval**: 100 → **500**
- **gpu_log_interval**: 50 → **200**
- **save_interval**: 5000 → **10000** (체크포인트 I/O 스파이크 완화)
- `config.yaml` 및 `train.py` 내 해당 값 수정

### 3-6. Gradient Accumulation (배치 크기 증가 시 OOM 대비)
- effective batch size를 키우고 싶지만 한 번에 큰 배치를 넣기 어려울 때 사용
- 예: `batch_size=256`, `accumulation_steps=2` → effective 512
- 옵션으로 구현해 두고, 배치 크기를 올린 뒤 OOM 시 활성화

---

## 4. 적용 순서 제안

1. **즉시 적용 (설정만)**  
   - 배치 크기 512, num_workers 8, prefetch_factor 4  
   - log_interval 500, save_interval 10000  
   - `cudnn.benchmark = True` 추가  

2. **코드 수정 후 적용**  
   - DDP 전환 (스크립트 실행 방식: `torchrun --nproc_per_node=4 train.py ...`)  
   - 필요 시 `torch.compile` 실험  

3. **인프라**  
   - 데이터 경로를 SSD로 이동하거나, I/O 병목이 확인되면 캐싱 전략 검토  

위 순서로 적용하면 **목표 3일 이내**로 줄이는 것이 현실적입니다.  
(예: DDP + 배치 512 + 데이터 로딩 최적화만으로도 약 4~5일 → 3일 근처까지 단축 가능)

---

## 6. 실행 방법 (DDP 적용 후)

### DDP로 4 GPU 학습 (권장)
```bash
cd deeplog_training
torchrun --nproc_per_node=4 train.py --config config.yaml
# 재개 시
torchrun --nproc_per_node=4 train.py --config config.yaml --resume /path/to/checkpoint.pt
```

### 기존 방식 (단일 프로세스, DataParallel)
```bash
python train.py --config config.yaml
```
- `RANK`/`WORLD_SIZE` 환경 변수가 없으면 자동으로 단일 프로세스 + DataParallel(멀티 GPU 시)로 동작합니다.

---

## 5. 참고: 현재 로그 타임라인

- 학습 시작(재개): 2026-02-12 21:42:42 (step 805,000)
- Epoch 9 시작: 2026-02-12 22:17:47
- Step 843,200 로그: 2026-02-12 23:52:27 (경과 약 1h 34m, 약 38,100 step)
- 이 구간 ETA: **233h 59m** (약 9.75일)
