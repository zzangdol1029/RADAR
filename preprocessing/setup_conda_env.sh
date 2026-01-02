#!/bin/bash
# RADAR 로그 전처리 프로젝트용 Conda 환경 생성 스크립트

set -e

ENV_NAME="radar-preprocessing"
ENV_FILE="environment.yml"

echo "=========================================="
echo "RADAR 전처리 프로젝트 Conda 환경 생성"
echo "=========================================="
echo ""

# Conda 설치 확인
if ! command -v conda &> /dev/null; then
    echo "❌ 오류: conda가 설치되어 있지 않습니다."
    echo "   Anaconda 또는 Miniconda를 먼저 설치해주세요."
    exit 1
fi

echo "✓ Conda 확인 완료"
echo ""

# 환경 파일 확인
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ 오류: $ENV_FILE 파일을 찾을 수 없습니다."
    exit 1
fi

echo "✓ 환경 파일 확인: $ENV_FILE"
echo ""

# 기존 환경 확인
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  경고: '$ENV_NAME' 환경이 이미 존재합니다."
    read -p "기존 환경을 삭제하고 새로 만들까요? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "기존 환경 삭제 중..."
        conda env remove -n "$ENV_NAME" -y
        echo "✓ 기존 환경 삭제 완료"
    else
        echo "작업을 취소했습니다."
        exit 0
    fi
fi

# 환경 생성
echo "환경 생성 중: $ENV_NAME"
echo "이 작업은 몇 분이 걸릴 수 있습니다..."
echo ""

conda env create -f "$ENV_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 환경 생성 완료!"
    echo "=========================================="
    echo ""
    echo "환경 활성화 방법:"
    echo "  conda activate $ENV_NAME"
    echo ""
    echo "환경 비활성화 방법:"
    echo "  conda deactivate"
    echo ""
    echo "환경 삭제 방법:"
    echo "  conda env remove -n $ENV_NAME"
    echo ""
else
    echo ""
    echo "❌ 환경 생성 실패"
    exit 1
fi

