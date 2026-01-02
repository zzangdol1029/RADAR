#!/usr/bin/env python3
"""
LogBERT 모델 학습 스크립트
Masked Language Modeling (MLM) 방식의 비지도 학습
"""

import os
import json
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime

from model import create_logbert_model
from dataset import LogBERTDataset, create_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogBERTTrainer:
    """LogBERT 모델 학습 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 학습 설정 딕셔너리
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"사용 디바이스: {self.device}")
        
        # 출력 디렉토리
        self.output_dir = Path(config.get('output_dir', 'checkpoints'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 초기화
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
        total_steps = int(config['training'].get('total_steps', 100000))
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
            current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{total_loss / num_batches:.4f}',
                'lr': f'{current_lr:.2e}',
                'step': f'{self.global_step}',
            })
            
            # 로깅
            if self.global_step % self.config['training'].get('log_interval', 100) == 0:
                current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"[Step {self.global_step}/{self.config['training'].get('total_steps', 'N/A')}] "
                    f"Loss={loss.item():.4f}, "
                    f"Avg Loss={total_loss/num_batches:.4f}, "
                    f"LR={current_lr:.2e}, "
                    f"Progress={num_batches}/{total_batches} batches"
                )
            
            # 체크포인트 저장
            if self.global_step % self.config['training'].get('save_interval', 1000) == 0:
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
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"체크포인트 로드: {checkpoint_path}")
    
    def train(self, train_dataloader, num_epochs: int):
        """전체 학습 과정"""
        total_batches = len(train_dataloader)
        estimated_total_steps = total_batches * num_epochs
        
        logger.info("=" * 80)
        logger.info("LogBERT 학습 시작")
        logger.info("=" * 80)
        logger.info(f"총 에폭: {num_epochs}")
        logger.info(f"에폭당 배치 수: {total_batches:,}")
        logger.info(f"예상 총 스텝: {estimated_total_steps:,}")
        logger.info(f"배치 크기: {self.config['training']['batch_size']}")
        logger.info(f"학습률: {self.config['training']['learning_rate']}")
        logger.info(f"디바이스: {self.device}")
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
        logger.info("학습 완료!")
        logger.info(f"최고 Loss: {self.best_loss:.4f}")
        logger.info("=" * 80)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """설정 파일 로드"""
    if config_path is None:
        base_dir = Path(__file__).parent
        config_path = str(base_dir / 'training_config.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 숫자 값들을 올바르게 변환
        if 'training' in config:
            # 학습률을 float로 변환
            if 'learning_rate' in config['training']:
                if isinstance(config['training']['learning_rate'], str):
                    config['training']['learning_rate'] = float(config['training']['learning_rate'])
            # 기타 숫자 값들도 변환
            for key in ['weight_decay', 'min_lr', 'max_grad_norm', 'mask_prob']:
                if key in config['training'] and isinstance(config['training'][key], str):
                    config['training'][key] = float(config['training'][key])
            for key in ['batch_size', 'num_epochs', 'total_steps', 'log_interval', 'save_interval', 'num_workers']:
                if key in config['training'] and isinstance(config['training'][key], str):
                    config['training'][key] = int(config['training'][key])
        
        if 'model' in config:
            # 모델 설정의 숫자 값들도 변환
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
        
        return config
    else:
        # 기본 설정
        return {
            'model': {
                'vocab_size': 10000,
                'hidden_size': 768,
                'num_hidden_layers': 12,
                'num_attention_heads': 12,
                'max_position_embeddings': 512,
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 2e-5,
                'num_epochs': 10,
                'total_steps': 100000,
            },
            'data': {
                'preprocessed_dir': '../preprocessing/output',
                'max_seq_length': 512,
            },
        }


def get_data_files(preprocessed_dir: str) -> list:
    """전처리된 파일 목록 가져오기"""
    data_dir = Path(preprocessed_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"전처리된 데이터 디렉토리를 찾을 수 없습니다: {preprocessed_dir}")
    
    files = sorted(data_dir.glob("preprocessed_logs_*.json"))
    logger.info(f"발견된 데이터 파일: {len(files)}개")
    
    return [str(f) for f in files]


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LogBERT 모델 학습')
    parser.add_argument('--config', type=str, default=None,
                       help='설정 파일 경로')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='전처리된 데이터 디렉토리')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='출력 디렉토리')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 명령줄 인자로 설정 덮어쓰기
    if args.data_dir:
        config['data']['preprocessed_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # 데이터 파일 목록
    data_files = get_data_files(config['data']['preprocessed_dir'])
    
    if len(data_files) == 0:
        logger.error("전처리된 데이터 파일을 찾을 수 없습니다.")
        return
    
    # 데이터셋 생성
    logger.info("데이터셋 생성 중...")
    dataset = LogBERTDataset(
        data_files=data_files,
        max_seq_length=config['data']['max_seq_length'],
        mask_prob=config['training'].get('mask_prob', 0.15),
        vocab_size=config['model']['vocab_size'],
    )
    
    # DataLoader 생성
    # macOS나 CPU 환경에서는 num_workers를 0으로 설정 (multiprocessing 문제 방지)
    num_workers = config['training'].get('num_workers', 4)
    if not torch.cuda.is_available():
        num_workers = 0
        logger.info("CUDA가 사용 불가능하므로 num_workers를 0으로 설정 (단일 프로세스 모드)")
    
    dataloader = create_dataloader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # CUDA가 있을 때만 pin_memory 사용
    )
    
    # 학습기 생성
    trainer = LogBERTTrainer(config)
    
    # 학습 시작
    trainer.train(
        train_dataloader=dataloader,
        num_epochs=config['training']['num_epochs'],
    )


if __name__ == '__main__':
    main()

