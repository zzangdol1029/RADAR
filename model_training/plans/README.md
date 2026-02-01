# 모델 학습 계획 📋

RADAR 프로젝트의 모델 학습을 위한 계획 및 문서 폴더입니다.

## 📁 폴더 구조

```
model_training/
├── plans/          # 📋 이 폴더 - 학습 계획 및 분석
├── logbert/        # 🤖 LogBERT 모델 학습
└── logrobust/      # 🤖 LogRobust 모델 학습 (향후)
```

## 📄 파일 목록

- **[PROJECT_PLAN.md](PROJECT_PLAN.md)** - 전체 학습 프로젝트 계획
- **[data_validator.py](data_validator.py)** - 데이터 검증 스크립트

## 🎯 목표

1. **LogBERT 학습**: MSA 로그 이상 탐지를 위한 BERT 기반 모델
2. **LogRobust 학습**: 강건한 로그 분석 모델 (향후)
3. **체계적 관리**: 모델별 독립적인 학습 환경

## 🚀 빠른 시작

### 데이터 검증
```bash
cd model_training/plans
python data_validator.py
```

### LogBERT 학습
```bash
cd model_training/logbert/scripts
python train.py --config ../configs/test.yaml
```

## 📊 데이터 현황

- **총 데이터**: 324개 파일 (141 GB)
- **기간**: 2025-02-24 ~ 2026-01-15
- **위치**: `RADAR/output/`

## 💡 학습 단계

1. **테스트** (10 파일) → 설정 검증
2. **중규모** (100 파일) → 성능 평가  
3. **전체** (324 파일) → 최종 모델
