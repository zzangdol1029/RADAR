#!/usr/bin/env python3
"""
서버에서 실행하는 LogBERT 학습 스크립트
preprocessing/output 디렉토리의 전처리된 파일을 사용하여 모델 학습
"""

import os
import sys
import argparse
from pathlib import Path

# 현재 스크립트의 디렉토리를 Python 경로에 추가
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from train import main as train_main, load_config, get_data_files
from dataset import LogBERTDataset, create_dataloader
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """서버에서 실행하는 메인 함수"""
    parser = argparse.ArgumentParser(
        description='서버에서 LogBERT 모델 학습 실행',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 실행 (training_config.yaml 사용)
  python train_server.py

  # 데이터 디렉토리 지정
  python train_server.py --data-dir /path/to/preprocessing/output

  # 출력 디렉토리 지정
  python train_server.py --output-dir /path/to/checkpoints

  # 설정 파일 지정
  python train_server.py --config custom_config.yaml

  # 모든 옵션 조합
  python train_server.py \\
      --data-dir /path/to/data \\
      --output-dir /path/to/output \\
      --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='전처리된 데이터 디렉토리 경로 (기본값: ../preprocessing/output)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='체크포인트 저장 디렉토리 (기본값: checkpoints)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='설정 파일 경로 (기본값: training_config.yaml)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='배치 크기 (설정 파일보다 우선)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='에폭 수 (설정 파일보다 우선)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='학습률 (설정 파일보다 우선)'
    )
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 명령줄 인자로 설정 덮어쓰기
    if args.data_dir:
        config.setdefault('data', {})
        config['data']['preprocessed_dir'] = args.data_dir
    elif 'data' not in config or 'preprocessed_dir' not in config['data']:
        # 기본값 설정
        config.setdefault('data', {})
        config['data']['preprocessed_dir'] = str(script_dir.parent / 'preprocessing' / 'output')
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
    elif 'output_dir' not in config:
        config['output_dir'] = 'checkpoints'
    
    # 학습 파라미터 덮어쓰기
    if args.batch_size:
        config.setdefault('training', {})
        config['training']['batch_size'] = args.batch_size
    
    if args.epochs:
        config.setdefault('training', {})
        config['training']['num_epochs'] = args.epochs
    
    if args.learning_rate:
        config.setdefault('training', {})
        config['training']['learning_rate'] = args.learning_rate
    
    # 데이터 디렉토리 확인
    data_dir = Path(config['data']['preprocessed_dir'])
    logger.info(f"데이터 디렉토리: {data_dir}")
    logger.info(f"데이터 디렉토리 존재 여부: {data_dir.exists()}")
    
    if not data_dir.exists():
        logger.error(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        logger.error("다음 중 하나를 확인하세요:")
        logger.error("  1. --data-dir 옵션으로 올바른 경로를 지정하세요")
        logger.error("  2. preprocessing/output 디렉토리가 존재하는지 확인하세요")
        sys.exit(1)
    
    # 데이터 파일 확인
    try:
        data_files = get_data_files(config['data']['preprocessed_dir'])
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    
    if len(data_files) == 0:
        logger.error("❌ 전처리된 데이터 파일을 찾을 수 없습니다.")
        logger.error(f"   디렉토리: {data_dir}")
        logger.error("   예상 파일 형식: preprocessed_logs_*.json")
        sys.exit(1)
    
    logger.info(f"✅ 발견된 데이터 파일: {len(data_files)}개")
    logger.info(f"   첫 번째 파일: {Path(data_files[0]).name}")
    if len(data_files) > 1:
        logger.info(f"   마지막 파일: {Path(data_files[-1]).name}")
    
    # 출력 디렉토리 확인
    output_dir = Path(config['output_dir'])
    logger.info(f"출력 디렉토리: {output_dir.absolute()}")
    
    # 학습 시작
    logger.info("=" * 80)
    logger.info("서버에서 LogBERT 학습 시작")
    logger.info("=" * 80)
    
    # train.py의 main 함수를 직접 호출하지 않고 여기서 학습 실행
    # (train.py의 main 함수를 재사용하기 위해)
    import torch
    from train import LogBERTTrainer
    
    # 데이터셋 생성
    logger.info("데이터셋 생성 중...")
    dataset = LogBERTDataset(
        data_files=data_files,
        max_seq_length=config['data'].get('max_seq_length', 512),
        mask_prob=config['training'].get('mask_prob', 0.15),
        vocab_size=config['model']['vocab_size'],
    )
    
    # DataLoader 생성
    num_workers = config['training'].get('num_workers', 4)
    if not torch.cuda.is_available():
        num_workers = 0
        logger.info("CUDA가 사용 불가능하므로 num_workers를 0으로 설정")
    
    dataloader = create_dataloader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    # 학습기 생성
    trainer = LogBERTTrainer(config)
    
    # 학습 시작
    trainer.train(
        train_dataloader=dataloader,
        num_epochs=config['training']['num_epochs'],
    )
    
    logger.info("=" * 80)
    logger.info("✅ 학습 완료!")
    logger.info(f"   체크포인트 저장 위치: {output_dir.absolute()}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()





