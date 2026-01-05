#!/bin/bash
# SCP를 사용하여 서버에 파일 업로드하는 스크립트
# 사용법: ./upload_to_server.sh [서버사용자명] [서버주소] [서버경로]

set -e  # 오류 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 서버 정보 설정 (명령줄 인자 또는 기본값)
SERVER_USER="${1:-user}"
SERVER_HOST="${2:-192.168.1.100}"
SERVER_PATH="${3:-/home/user/RADAR}"

# 로컬 경로 (현재 스크립트 위치 기준)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "서버 파일 업로드 스크립트"
echo "=========================================="
echo ""
echo "서버 정보:"
echo "  사용자: $SERVER_USER"
echo "  주소: $SERVER_HOST"
echo "  경로: $SERVER_PATH"
echo ""
echo "로컬 경로: $SCRIPT_DIR"
echo ""

# 서버 연결 테스트
echo "서버 연결 테스트 중..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes "${SERVER_USER}@${SERVER_HOST}" exit 2>/dev/null; then
    echo -e "${GREEN}✅ 서버 연결 성공${NC}"
else
    echo -e "${YELLOW}⚠️  서버 연결 테스트 실패 (계속 진행)${NC}"
    echo "   비밀번호 입력이 필요할 수 있습니다."
fi
echo ""

# 서버에 디렉토리 생성
echo "서버에 디렉토리 생성 중..."
ssh "${SERVER_USER}@${SERVER_HOST}" "mkdir -p ${SERVER_PATH}/logbert_training"
ssh "${SERVER_USER}@${SERVER_HOST}" "mkdir -p ${SERVER_PATH}/preprocessing/output"
echo -e "${GREEN}✅ 디렉토리 생성 완료${NC}"
echo ""

# 1. logbert_training 디렉토리 업로드
echo "=========================================="
echo "1. logbert_training 디렉토리 업로드 중..."
echo "=========================================="
scp -r "${SCRIPT_DIR}/logbert_training/" "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/logbert_training/"
echo -e "${GREEN}✅ logbert_training 업로드 완료${NC}"
echo ""

# 2. preprocessing/output 디렉토리 업로드
echo "=========================================="
echo "2. preprocessing/output 디렉토리 업로드 중..."
echo "=========================================="
if [ -d "${SCRIPT_DIR}/preprocessing/output" ]; then
    FILE_COUNT=$(find "${SCRIPT_DIR}/preprocessing/output" -name "preprocessed_logs_*.json" | wc -l)
    echo "   발견된 데이터 파일: $FILE_COUNT개"
    
    if [ "$FILE_COUNT" -gt 0 ]; then
        scp -r "${SCRIPT_DIR}/preprocessing/output/" "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/preprocessing/"
        echo -e "${GREEN}✅ preprocessing/output 업로드 완료${NC}"
    else
        echo -e "${YELLOW}⚠️  데이터 파일을 찾을 수 없습니다.${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  preprocessing/output 디렉토리를 찾을 수 없습니다.${NC}"
fi
echo ""

# 업로드 완료
echo "=========================================="
echo -e "${GREEN}✅ 업로드 완료!${NC}"
echo "=========================================="
echo ""
echo "서버에서 다음 명령어를 실행하세요:"
echo ""
echo "  # SSH 접속"
echo "  ssh ${SERVER_USER}@${SERVER_HOST}"
echo ""
echo "  # 디렉토리 이동"
echo "  cd ${SERVER_PATH}/logbert_training"
echo ""
echo "  # 실행 권한 부여"
echo "  chmod +x run_training_server.sh"
echo ""
echo "  # 의존성 설치"
echo "  pip install -r requirements.txt"
echo "  # 또는 conda 환경 사용 시"
echo "  conda activate radar"
echo "  pip install -r requirements.txt"
echo ""
echo "  # 학습 실행"
echo "  ./run_training_server.sh"
echo "  # 또는"
echo "  python train_server.py"
echo ""


