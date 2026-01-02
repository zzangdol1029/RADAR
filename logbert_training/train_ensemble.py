#!/usr/bin/env python3
"""
LogBERT 앙상블 학습 스크립트
여러 모델을 학습하고 앙상블로 성능 향상
"""

import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
import random

from transformers import (
    BertForMaskedLM,
    DistilBertForMaskedLM,
    RobertaForMaskedLM,
    AutoModelForMaskedLM
)
from dataset import LogBERTDataset, create_dataloader, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleModel(nn.Module):
    """앙상블 모델 (여러 모델의 예측 결합)"""
    
    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        ensemble_method: str = 'weighted_average',
        weights: Optional[List[float]] = None,
    ):
        """
        Args:
            model_configs: 모델 설정 리스트
            ensemble_method: 'weighted_average', 'voting', 'max'
            weights: 각 모델의 가중치 (None이면 균등)
        """
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList()
        self.ensemble_method = ensemble_method
        
        # 모델 로드
        for config in model_configs:
            model = self._create_model(config)
            self.models.append(model)
        
        # 가중치 설정
        if weights is None:
            self.weights = [1.0 / len(model_configs)] * len(model_configs)
        else:
            assert len(weights) == len(model_configs), "가중치 수가 모델 수와 일치해야 합니다"
            total = sum(weights)
            self.weights = [w / total for w in weights]  # 정규화
        
        logger.info(f"앙상블 모델 초기화 완료: {len(self.models)}개 모델")
        logger.info(f"앙상블 방법: {ensemble_method}")
        logger.info(f"가중치: {self.weights}")
    
    def _create_model(self, config: Dict[str, Any]) -> nn.Module:
        """모델 생성"""
        model_type = config['type']
        model_name = config.get('pretrained_model', None)
        vocab_size = config.get('vocab_size', 10000)
        
        if model_type == 'distilbert':
            model = DistilBertForMaskedLM.from_pretrained(
                model_name or 'distilbert-base-uncased'
            )
        elif model_type == 'bert':
            model = BertForMaskedLM.from_pretrained(
                model_name or 'bert-base-uncased'
            )
        elif model_type == 'roberta':
            model = RobertaForMaskedLM.from_pretrained(
                model_name or 'roberta-base'
            )
        elif model_type == 'auto':
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        elif model_type == 'lstm':
            from model_lstm import LogLSTM
            model = LogLSTM(
                vocab_size=vocab_size,
                embedding_dim=config.get('embedding_dim', 128),
                hidden_size=config.get('hidden_size', 256),
                num_layers=config.get('num_layers', 2),
            )
        elif model_type == 'deeplog':
            from model_deeplog import DeepLog
            model = DeepLog(
                vocab_size=vocab_size,
                embedding_dim=config.get('embedding_dim', 128),
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
            )
        elif model_type == 'tcn':
            from model_tcn import LogTCN
            model = LogTCN(
                vocab_size=vocab_size,
                embedding_dim=config.get('embedding_dim', 128),
                num_channels=config.get('num_channels', [128, 128, 128]),
                kernel_size=config.get('kernel_size', 3),
            )
        else:
            raise ValueError(f"알 수 없는 모델 타입: {model_type}")
        
        # 어휘 크기 조정 (필요한 경우)
        if vocab_size != model.config.vocab_size:
            logger.info(f"어휘 크기 조정: {model.config.vocab_size} -> {vocab_size}")
            # 임베딩 레이어 재생성
            if hasattr(model, 'bert'):
                old_embeddings = model.bert.embeddings.word_embeddings
            elif hasattr(model, 'distilbert'):
                old_embeddings = model.distilbert.embeddings.word_embeddings
            else:
                old_embeddings = model.roberta.embeddings.word_embeddings
            
            new_embeddings = nn.Embedding(vocab_size, old_embeddings.embedding_dim)
            min_size = min(vocab_size, old_embeddings.num_embeddings)
            new_embeddings.weight.data[:min_size] = old_embeddings.weight.data[:min_size]
            
            if hasattr(model, 'bert'):
                model.bert.embeddings.word_embeddings = new_embeddings
            elif hasattr(model, 'distilbert'):
                model.distilbert.embeddings.word_embeddings = new_embeddings
            else:
                model.roberta.embeddings.word_embeddings = new_embeddings
            
            model.config.vocab_size = vocab_size
        
        return model
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """앙상블 Forward pass"""
        losses = []
        logits_list = []
        
        for model in self.models:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            losses.append(outputs.loss)
            logits_list.append(outputs.logits)
        
        # 앙상블 방법에 따라 결합
        if self.ensemble_method == 'weighted_average':
            # 가중 평균
            ensemble_loss = sum(w * l for w, l in zip(self.weights, losses))
            ensemble_logits = sum(w * logits for w, logits in zip(self.weights, logits_list))
        elif self.ensemble_method == 'average':
            # 단순 평균
            ensemble_loss = sum(losses) / len(losses)
            ensemble_logits = sum(logits_list) / len(logits_list)
        elif self.ensemble_method == 'max':
            # 최대값
            ensemble_loss = max(losses)
            ensemble_logits = torch.stack(logits_list).max(dim=0)[0]
        else:
            raise ValueError(f"알 수 없는 앙상블 방법: {self.ensemble_method}")
        
        return {
            'loss': ensemble_loss,
            'logits': ensemble_logits,
            'individual_losses': losses,
        }
    
    def predict_anomaly_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """앙상블 이상 점수 계산"""
        self.eval()
        scores_list = []
        
        with torch.no_grad():
            for model in self.models:
                # 각 모델의 이상 점수 계산
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                
                # Softmax로 확률 계산
                probs = torch.softmax(logits, dim=-1)
                
                # 실제 토큰의 확률 추출
                batch_size, seq_len = input_ids.shape
                token_probs = probs[torch.arange(batch_size).unsqueeze(1), 
                                    torch.arange(seq_len).unsqueeze(0), 
                                    input_ids]
                
                # 음의 로그 확률
                scores = -torch.log(token_probs + 1e-10)
                
                # 패딩 처리
                if attention_mask is not None:
                    scores = scores * attention_mask.float()
                
                # 시퀀스별 평균
                if attention_mask is not None:
                    seq_scores = scores.sum(dim=1) / attention_mask.sum(dim=1).float()
                else:
                    seq_scores = scores.mean(dim=1)
                
                scores_list.append(seq_scores)
        
        # 앙상블 결합
        if self.ensemble_method == 'weighted_average':
            ensemble_scores = sum(w * s for w, s in zip(self.weights, scores_list))
        elif self.ensemble_method == 'average':
            ensemble_scores = sum(scores_list) / len(scores_list)
        elif self.ensemble_method == 'max':
            ensemble_scores = torch.stack(scores_list).max(dim=0)[0]
        else:
            ensemble_scores = sum(scores_list) / len(scores_list)
        
        return ensemble_scores


class EnsembleTrainer:
    """앙상블 학습 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cpu')
        logger.info(f"사용 디바이스: {self.device}")
        
        # 출력 디렉토리
        self.output_dir = Path(config.get('output_dir', 'checkpoints_ensemble'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 앙상블 모델 초기화
        self.model = EnsembleModel(
            model_configs=config['ensemble']['models'],
            ensemble_method=config['ensemble'].get('method', 'weighted_average'),
            weights=config['ensemble'].get('weights', None),
        )
        self.model.to(self.device)
        
        # 옵티마이저
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
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs['loss']
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training'].get('max_grad_norm', 1.0)
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{total_loss / num_batches:.4f}',
                'lr': f'{current_lr:.2e}',
            })
            
            if self.global_step % self.config['training'].get('log_interval', 50) == 0:
                individual_losses = outputs.get('individual_losses', [])
                loss_str = ', '.join([f'{l.item():.4f}' for l in individual_losses])
                logger.info(
                    f"[Step {self.global_step}] "
                    f"Ensemble Loss={loss.item():.4f}, "
                    f"Individual=[{loss_str}], "
                    f"Avg={total_loss/num_batches:.4f}"
                )
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
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
        logger.info("=" * 80)
        logger.info("LogBERT 앙상블 학습 시작")
        logger.info("=" * 80)
        logger.info(f"앙상블 모델 수: {len(self.model.models)}")
        for i, model in enumerate(self.model.models):
            logger.info(f"  모델 {i+1}: {type(model).__name__}")
        logger.info(f"앙상블 방법: {self.model.ensemble_method}")
        logger.info(f"가중치: {self.model.weights}")
        logger.info(f"총 에폭: {num_epochs}")
        logger.info(f"배치 크기: {self.config['training']['batch_size']}")
        logger.info("=" * 80)
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n에폭 {epoch}/{num_epochs} 시작")
            
            avg_loss = self.train_epoch(train_dataloader, epoch)
            
            logger.info(f"에폭 {epoch} 완료 - 평균 Loss: {avg_loss:.4f}")
            
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint('best_model')
                logger.info(f"✅ 최고 성능 모델 저장 (Loss: {avg_loss:.4f})")
            
            self.save_checkpoint(f'epoch_{epoch}')
        
        logger.info("=" * 80)
        logger.info("앙상블 학습 완료!")
        logger.info(f"최고 Loss: {self.best_loss:.4f}")
        logger.info("=" * 80)


def load_config() -> Dict[str, Any]:
    """앙상블 학습용 설정"""
    return {
        'ensemble': {
            'method': 'weighted_average',  # 'weighted_average', 'average', 'max'
            'weights': [0.4, 0.4, 0.2],  # 각 모델의 가중치
            'models': [
                {
                    'type': 'distilbert',
                    'pretrained_model': 'distilbert-base-uncased',
                    'vocab_size': 10000,
                },
                {
                    'type': 'bert',
                    'pretrained_model': 'bert-base-uncased',
                    'vocab_size': 10000,
                },
                {
                    'type': 'roberta',
                    'pretrained_model': 'roberta-base',
                    'vocab_size': 10000,
                },
            ],
        },
        'training': {
            'batch_size': 4,  # 작은 배치 (여러 모델 동시 학습)
            'learning_rate': 0.00001,
            'weight_decay': 0.01,
            'num_epochs': 3,
            'total_steps': 5000,
            'min_lr': 0.000001,
            'max_grad_norm': 1.0,
            'mask_prob': 0.15,
            'log_interval': 50,
            'save_interval': 500,
            'num_workers': 0,
        },
        'data': {
            'preprocessed_dir': '../preprocessing/output',
            'max_seq_length': 256,
            'sample_ratio': 0.05,
            'max_files': 10,
        },
        'output_dir': 'checkpoints_ensemble',
    }


def get_data_files(preprocessed_dir: str, max_files: int = 10) -> list:
    """데이터 파일 목록"""
    data_dir = Path(preprocessed_dir)
    files = sorted(data_dir.glob("preprocessed_logs_*.json"))
    
    if len(files) > max_files:
        files = random.sample(files, max_files)
    
    logger.info(f"사용할 데이터 파일: {len(files)}개")
    return [str(f) for f in files]


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LogBERT 앙상블 학습')
    parser.add_argument('--config', type=str, default=None,
                       help='설정 파일 경로')
    parser.add_argument('--method', type=str, default='weighted_average',
                       choices=['weighted_average', 'average', 'max'],
                       help='앙상블 방법')
    parser.add_argument('--sample-ratio', type=float, default=0.05,
                       help='데이터 샘플링 비율')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config()
    if args.method:
        config['ensemble']['method'] = args.method
    if args.sample_ratio:
        config['data']['sample_ratio'] = args.sample_ratio
    
    # 데이터 파일
    data_files = get_data_files(
        config['data']['preprocessed_dir'],
        max_files=config['data'].get('max_files', 10)
    )
    
    if len(data_files) == 0:
        logger.error("데이터 파일을 찾을 수 없습니다.")
        return
    
    # 데이터셋 생성
    logger.info("데이터셋 생성 중...")
    from train_test import SampledLogBERTDataset
    
    dataset = SampledLogBERTDataset(
        data_files=data_files,
        max_seq_length=config['data']['max_seq_length'],
        mask_prob=config['training'].get('mask_prob', 0.15),
        vocab_size=10000,  # 기본값
        sample_ratio=config['data'].get('sample_ratio', 0.05),
        max_files=config['data'].get('max_files', 10),
    )
    
    # DataLoader
    dataloader = create_dataloader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    
    # 학습기
    trainer = EnsembleTrainer(config)
    
    # 학습 시작
    trainer.train(
        train_dataloader=dataloader,
        num_epochs=config['training']['num_epochs'],
    )


if __name__ == '__main__':
    main()

