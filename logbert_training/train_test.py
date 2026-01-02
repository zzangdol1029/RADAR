#!/usr/bin/env python3
"""
LogBERT 테스트용 학습 스크립트
CPU/메모리 사용량을 최소화하여 빠른 테스트 학습
"""

import os
import json
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
import random

from model import create_logbert_model
from dataset import LogBERTDataset, create_dataloader, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogBERTTestTrainer:
    """LogBERT 테스트용 학습 클래스 (메모리 최적화)"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 학습 설정 딕셔너리
        """
        self.config = config
        self.device = torch.device('cpu')  # 테스트는 CPU만 사용
        logger.info(f"사용 디바이스: {self.device} (테스트 모드)")
        
        # 출력 디렉토리
        self.output_dir = Path(config.get('output_dir', 'checkpoints_test'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 초기화 (작은 모델)
        self.model = create_logbert_model(config['model'])
        self.model.to(self.device)
        
        # 옵티마이저 (학습률을 float로 변환)
        learning_rate = float(config['training']['learning_rate'])
        weight_decay = float(config['training'].get('weight_decay', 0.01))
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # 학습률 스케줄러
        total_steps = int(config['training'].get('total_steps', 10000))
        min_lr = float(config['training'].get('min_lr', 1e-6))
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=min_lr,
        )
        
        # 학습 상태
        self.global_step = 0
        self.best_loss = float('inf')
        
        # 체크포인트 저장 경로
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, dataloader, epoch: int) -> float:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 전체 배치 수 계산
        total_batches = len(dataloader)
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}",
            total=total_batches,
            unit="batch",
            leave=True,
            ncols=100
        )
        
        for batch in progress_bar:
            # 배치를 디바이스로 이동
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training'].get('max_grad_norm', 1.0)
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 통계
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 진행 상황 업데이트
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{total_loss / num_batches:.4f}',
                'lr': f'{current_lr:.2e}',
                'step': f'{self.global_step}',
            })
            
            # 로깅
            if self.global_step % self.config['training'].get('log_interval', 50) == 0:
                logger.info(
                    f"[Step {self.global_step}] "
                    f"Loss={loss.item():.4f}, "
                    f"Avg Loss={total_loss/num_batches:.4f}, "
                    f"LR={current_lr:.2e}, "
                    f"Progress={num_batches}/{total_batches} batches"
                )
            
            # 체크포인트 저장
            if self.global_step % self.config['training'].get('save_interval', 500) == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}')
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def save_checkpoint(self, name: str):
        """체크포인트 저장"""
        checkpoint_path = self.checkpoint_dir / f'{name}.pt'
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': self.config,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"체크포인트 저장: {checkpoint_path}")
    
    def train(self, train_dataloader, num_epochs: int):
        """전체 학습 과정"""
        total_batches = len(train_dataloader)
        estimated_total_steps = total_batches * num_epochs
        
        logger.info("=" * 80)
        logger.info("LogBERT 테스트 학습 시작 (메모리 최적화 모드)")
        logger.info("=" * 80)
        logger.info(f"총 에폭: {num_epochs}")
        logger.info(f"에폭당 배치 수: {total_batches:,}")
        logger.info(f"예상 총 스텝: {estimated_total_steps:,}")
        logger.info(f"배치 크기: {self.config['training']['batch_size']}")
        logger.info(f"학습률: {self.config['training']['learning_rate']}")
        logger.info(f"디바이스: {self.device} (CPU 전용)")
        logger.info(f"체크포인트 저장 경로: {self.checkpoint_dir}")
        logger.info("=" * 80)
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"에폭 {epoch}/{num_epochs} 시작")
            logger.info(f"{'='*80}")
            
            avg_loss = self.train_epoch(train_dataloader, epoch)
            
            logger.info(f"\n에폭 {epoch}/{num_epochs} 완료")
            logger.info(f"  평균 Loss: {avg_loss:.4f}")
            logger.info(f"  현재 스텝: {self.global_step:,}")
            logger.info(f"  현재 학습률: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 최고 성능 모델 저장
            if avg_loss < self.best_loss:
                improvement = self.best_loss - avg_loss
                self.best_loss = avg_loss
                self.save_checkpoint('best_model')
                logger.info(f"  ✅ 최고 성능 모델 저장! (Loss: {avg_loss:.4f}, 개선: {improvement:.4f})")
            else:
                logger.info(f"  현재 최고 Loss: {self.best_loss:.4f}")
            
            # 에폭별 체크포인트 저장
            self.save_checkpoint(f'epoch_{epoch}')
            logger.info(f"  체크포인트 저장: epoch_{epoch}.pt")
        
        logger.info("=" * 80)
        logger.info("테스트 학습 완료!")
        logger.info(f"최고 Loss: {self.best_loss:.4f}")
        logger.info("=" * 80)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """설정 파일 로드 (테스트용 기본값)"""
    if config_path is None:
        base_dir = Path(__file__).parent
        config_path = str(base_dir / 'training_config.yaml')
    
    # 기본 테스트 설정
    default_config = {
        'model': {
            'vocab_size': 10000,
            'hidden_size': 256,  # 작은 모델 (768 -> 256)
            'num_hidden_layers': 4,  # 적은 레이어 (12 -> 4)
            'num_attention_heads': 4,  # 적은 헤드 (12 -> 4)
            'intermediate_size': 1024,  # 작은 중간 크기 (3072 -> 1024)
            'max_position_embeddings': 256,  # 짧은 시퀀스 (512 -> 256)
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
        },
        'training': {
            'batch_size': 8,  # 작은 배치 (32 -> 8)
            'learning_rate': 0.00002,
            'weight_decay': 0.01,
            'num_epochs': 2,  # 적은 에폭 (10 -> 2)
            'total_steps': 5000,  # 적은 스텝
            'min_lr': 0.000001,
            'max_grad_norm': 1.0,
            'mask_prob': 0.15,
            'log_interval': 50,  # 더 자주 로그
            'save_interval': 500,
            'num_workers': 0,  # 단일 프로세스
        },
        'data': {
            'preprocessed_dir': '../preprocessing/output',
            'max_seq_length': 256,  # 짧은 시퀀스
            'sample_ratio': 0.01,  # 데이터의 1%만 사용
            'max_files': 5,  # 최대 5개 파일만 사용
        },
        'output_dir': 'checkpoints_test',
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 기본값으로 덮어쓰기
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
            elif isinstance(default_config[key], dict):
                for sub_key in default_config[key]:
                    if sub_key not in config[key]:
                        config[key][sub_key] = default_config[key][sub_key]
        
        # 숫자 값들을 올바르게 변환
        if 'training' in config:
            for key in ['learning_rate', 'weight_decay', 'min_lr', 'max_grad_norm', 'mask_prob']:
                if key in config['training'] and isinstance(config['training'][key], str):
                    config['training'][key] = float(config['training'][key])
            for key in ['batch_size', 'num_epochs', 'total_steps', 'log_interval', 'save_interval', 'num_workers']:
                if key in config['training'] and isinstance(config['training'][key], str):
                    config['training'][key] = int(config['training'][key])
        
        if 'model' in config:
            for key in config['model']:
                if isinstance(config['model'][key], str):
                    try:
                        if '.' in config['model'][key]:
                            config['model'][key] = float(config['model'][key])
                        else:
                            config['model'][key] = int(config['model'][key])
                    except ValueError:
                        pass
        
        if 'data' in config:
            if 'max_seq_length' in config['data'] and isinstance(config['data']['max_seq_length'], str):
                config['data']['max_seq_length'] = int(config['data']['max_seq_length'])
            if 'sample_ratio' not in config['data']:
                config['data']['sample_ratio'] = default_config['data']['sample_ratio']
            if 'max_files' not in config['data']:
                config['data']['max_files'] = default_config['data']['max_files']
    else:
        config = default_config
    
    return config


class SampledLogBERTDataset(LogBERTDataset):
    """샘플링된 데이터셋 (메모리 절약)"""
    
    def __init__(
        self,
        data_files: List[str],
        max_seq_length: int = 256,
        mask_prob: float = 0.15,
        random_mask_prob: float = 0.1,
        keep_mask_prob: float = 0.1,
        vocab_size: int = 10000,
        sample_ratio: float = 0.01,
        max_files: int = 5,
    ):
        """
        Args:
            sample_ratio: 사용할 데이터 비율 (0.01 = 1%)
            max_files: 최대 사용할 파일 수
        """
        # 파일 수 제한
        if len(data_files) > max_files:
            logger.info(f"파일 수 제한: {len(data_files)}개 중 {max_files}개만 사용")
            data_files = random.sample(data_files, max_files)
        
        # 부모 클래스 초기화
        super().__init__(
            data_files,
            max_seq_length,
            mask_prob,
            random_mask_prob,
            keep_mask_prob,
            vocab_size,
        )
        
        # 데이터 샘플링
        if sample_ratio < 1.0:
            original_size = len(self.sessions)
            sample_size = int(len(self.sessions) * sample_ratio)
            self.sessions = random.sample(self.sessions, sample_size)
            logger.info(f"데이터 샘플링: {original_size:,}개 중 {len(self.sessions):,}개 사용 ({sample_ratio*100:.1f}%)")


def get_data_files(preprocessed_dir: str, max_files: int = 5) -> list:
    """전처리된 파일 목록 가져오기 (제한된 수)"""
    data_dir = Path(preprocessed_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"전처리된 데이터 디렉토리를 찾을 수 없습니다: {preprocessed_dir}")
    
    files = sorted(data_dir.glob("preprocessed_logs_*.json"))
    
    # 파일 수 제한
    if len(files) > max_files:
        logger.info(f"파일 수 제한: {len(files)}개 중 {max_files}개만 사용")
        files = random.sample(files, max_files)
    
    logger.info(f"사용할 데이터 파일: {len(files)}개")
    for f in files:
        logger.info(f"  - {f.name}")
    
    return [str(f) for f in files]


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LogBERT 테스트 학습 (메모리 최적화)')
    parser.add_argument('--config', type=str, default=None,
                       help='설정 파일 경로')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='전처리된 데이터 디렉토리')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='출력 디렉토리')
    parser.add_argument('--sample-ratio', type=float, default=0.01,
                       help='데이터 샘플링 비율 (기본값: 0.01 = 1%%)')
    parser.add_argument('--max-files', type=int, default=5,
                       help='최대 사용할 파일 수 (기본값: 5)')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 명령줄 인자로 설정 덮어쓰기
    if args.data_dir:
        config['data']['preprocessed_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.sample_ratio:
        config['data']['sample_ratio'] = args.sample_ratio
    if args.max_files:
        config['data']['max_files'] = args.max_files
    
    # 데이터 파일 목록
    data_files = get_data_files(
        config['data']['preprocessed_dir'],
        max_files=config['data'].get('max_files', 5)
    )
    
    if len(data_files) == 0:
        logger.error("전처리된 데이터 파일을 찾을 수 없습니다.")
        return
    
    # 샘플링된 데이터셋 생성
    logger.info("테스트용 데이터셋 생성 중...")
    logger.info(f"  샘플링 비율: {config['data'].get('sample_ratio', 0.01)*100:.1f}%")
    logger.info(f"  최대 파일 수: {config['data'].get('max_files', 5)}개")
    
    dataset = SampledLogBERTDataset(
        data_files=data_files,
        max_seq_length=config['data']['max_seq_length'],
        mask_prob=config['training'].get('mask_prob', 0.15),
        vocab_size=config['model']['vocab_size'],
        sample_ratio=config['data'].get('sample_ratio', 0.01),
        max_files=config['data'].get('max_files', 5),
    )
    
    # DataLoader 생성 (단일 프로세스, 작은 배치)
    logger.info("DataLoader 생성 중...")
    dataloader = create_dataloader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # 단일 프로세스
        pin_memory=False,  # CPU에서는 불필요
    )
    
    # 학습기 생성
    trainer = LogBERTTestTrainer(config)
    
    # 학습 시작
    trainer.train(
        train_dataloader=dataloader,
        num_epochs=config['training']['num_epochs'],
    )


if __name__ == '__main__':
    main()













