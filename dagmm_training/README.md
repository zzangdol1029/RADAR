# DAGMM & DeepLog Training

H100 GPU에 최적화된 로그 이상 탐지 모델 학습 패키지입니다.

## 모델

### 1. DAGMM (Deep Autoencoding Gaussian Mixture Model)
- **파일**: `model_dagmm.py`, `train_dagmm.py`
- 오토인코더와 GMM을 결합한 비지도 이상 탐지 모델
- 재구성 오류와 에너지 점수를 이용한 이상 탐지

### 2. DeepLog
- **파일**: `model_deeplog.py`, `train_deeplog.py`
- LSTM 기반의 시퀀스 예측 모델
- 다음 로그 이벤트를 예측하여 이상 탐지 수행

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### DAGMM 학습
```bash
python train_dagmm.py \
    --data-dir ../preprocessing/output \
    --output-dir checkpoints/dagmm \
    --epochs 50 \
    --batch-size 256
```

### DeepLog 학습
```bash
python train_deeplog.py \
    --data-dir ../preprocessing/output \
    --output-dir checkpoints/deeplog \
    --epochs 50 \
    --batch-size 64 \
    --window-size 10
```

## 파일 구조

```
dagmm_training/
├── __init__.py           # 패키지 초기화
├── dataset.py            # 데이터 로딩 및 전처리
├── model_dagmm.py        # DAGMM 모델 정의
├── model_deeplog.py      # DeepLog 모델 정의
├── train_dagmm.py        # DAGMM 학습 스크립트
├── train_deeplog.py      # DeepLog 학습 스크립트
├── requirements.txt      # 의존성 패키지
└── README.md             # 문서
```

## H100 GPU 최적화

- 멀티 GPU 지원 (DataParallel)
- 대용량 배치 크기 지원
- GPU 메모리 최적화
- Mixed Precision 지원 가능

## 출력

학습 후 생성되는 파일:
- `best_model.pt` / `best_deeplog_model.pt`: 최고 성능 모델
- `final_model.pt` / `final_deeplog_model.pt`: 최종 모델
- `event_id_mapping.json`: 이벤트 ID 매핑 정보
