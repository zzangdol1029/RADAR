#!/usr/bin/env python3
"""
LogBERT 전이 학습 스크립트
Pre-trained BERT 모델을 파인튜닝하여 M4 Pro에서도 학습 가능
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
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
import numpy as np
from datetime import datetime
import random
import gc
import psutil
import sys

from transformers import BertForMaskedLM, BertConfig
from dataset import LogBERTDataset, create_dataloader, collate_fn

# 로거 설정 (기본)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_memory_usage() -> Dict[str, float]:
    """현재 메모리 사용량 반환 (MB)"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss': mem_info.rss / 1024 / 1024,  # Resident Set Size (실제 메모리)
        'vms': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent(),  # 시스템 메모리 대비 비율
    }


def get_cpu_usage() -> float:
    """현재 CPU 사용률 반환 (%)"""
    return psutil.cpu_percent(interval=0.1)


def log_resource_usage(logger_instance: logging.Logger, prefix: str = ""):
    """리소스 사용량 로깅"""
    mem = get_memory_usage()
    cpu = get_cpu_usage()
    msg = (
        f"{prefix}리소스 사용량 - "
        f"메모리: {mem['rss']:.1f}MB (시스템 {mem['percent']:.1f}%), "
        f"CPU: {cpu:.1f}%"
    )
    logger_instance.info(msg)


def cleanup_memory():
    """메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Python의 메모리 정리
    if hasattr(gc, 'collect'):
        gc.collect()


def setup_file_logger(log_dir: Path, log_name: str = 'training') -> logging.Logger:
    """
    파일 로거 설정
    
    Args:
        log_dir: 로그 파일 저장 디렉토리
        log_name: 로그 파일명
    
    Returns:
        설정된 로거
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로그 파일 경로
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{log_name}_{timestamp}.log'
    
    # 파일 핸들러 생성
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    
    # 로거에 핸들러 추가
    file_logger = logging.getLogger(f'{__name__}.{log_name}')
    file_logger.setLevel(logging.INFO)
    file_logger.addHandler(file_handler)
    file_logger.propagate = False  # 루트 로거로 전파 방지
    
    logger.info(f"로그 파일 생성: {log_file}")
    return file_logger


class TransferLogBERT(nn.Module):
    """
    전이 학습용 LogBERT 모델
    Pre-trained BERT를 로드하여 파인튜닝
    """
    
    def __init__(
        self,
        pretrained_model_name: str = 'bert-base-uncased',
        vocab_size: int = 10000,
    ):
        """
        Args:
            pretrained_model_name: Hugging Face 모델명
            vocab_size: 로그 데이터의 어휘 크기
        """
        super(TransferLogBERT, self).__init__()
        
        # Pre-trained BERT 로드
        logger.info(f"Pre-trained BERT 로드 중: {pretrained_model_name}")
        self.bert = BertForMaskedLM.from_pretrained(pretrained_model_name)
        
        # 어휘 크기가 다르면 임베딩 레이어 재초기화
        if vocab_size != self.bert.config.vocab_size:
            logger.info(f"어휘 크기 조정: {self.bert.config.vocab_size} -> {vocab_size}")
            # 임베딩 레이어 재생성
            old_embeddings = self.bert.bert.embeddings.word_embeddings
            new_embeddings = nn.Embedding(vocab_size, old_embeddings.embedding_dim)
            # 기존 가중치 복사 (가능한 범위만)
            min_size = min(vocab_size, old_embeddings.num_embeddings)
            new_embeddings.weight.data[:min_size] = old_embeddings.weight.data[:min_size]
            self.bert.bert.embeddings.word_embeddings = new_embeddings
            self.bert.config.vocab_size = vocab_size
            # MLM 헤드도 재생성
            self.bert.cls.predictions.decoder = nn.Linear(
                self.bert.config.hidden_size,
                vocab_size
            )
            self.bert.cls.predictions.bias = nn.Parameter(torch.zeros(vocab_size))
        
        logger.info("전이 학습 모델 초기화 완료")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
        }
    
    def predict_anomaly_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """이상 점수 계산"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits']
            
            # Softmax로 확률 계산
            probs = torch.softmax(logits, dim=-1)
            
            # 실제 토큰의 확률 추출
            batch_size, seq_len = input_ids.shape
            token_probs = probs[torch.arange(batch_size).unsqueeze(1), 
                                torch.arange(seq_len).unsqueeze(0), 
                                input_ids]
            
            # 음의 로그 확률
            anomaly_scores = -torch.log(token_probs + 1e-10)
            
            # 패딩 위치는 0으로 설정
            if attention_mask is not None:
                anomaly_scores = anomaly_scores * attention_mask.float()
            
            # 시퀀스별 평균 이상 점수
            if attention_mask is not None:
                seq_scores = anomaly_scores.sum(dim=1) / attention_mask.sum(dim=1).float()
            else:
                seq_scores = anomaly_scores.mean(dim=1)
            
            return seq_scores


class TransferTrainer:
    """전이 학습용 트레이너"""
    
    def __init__(self, config: Dict[str, Any], load_checkpoint: Optional[str] = None, file_logger: Optional[logging.Logger] = None):
        self.config = config
        self.device = torch.device('cpu')  # M4 Pro는 CPU만
        self.file_logger = file_logger or logger  # 파일 로거 또는 기본 로거
        
        logger.info(f"사용 디바이스: {self.device}")
        self.file_logger.info(f"사용 디바이스: {self.device}")
        
        # 출력 디렉토리
        self.output_dir = Path(config.get('output_dir', 'checkpoints_transfer'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 초기화 (전이 학습)
        self.model = TransferLogBERT(
            pretrained_model_name=config.get('pretrained_model', 'bert-base-uncased'),
            vocab_size=config['model']['vocab_size'],
        )
        self.model.to(self.device)
        
        # 옵티마이저 (작은 학습률로 파인튜닝)
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
        
        # 학습 메트릭 저장
        self.training_metrics = {
            'epochs': [],
            'losses': [],
            'learning_rates': [],
            'steps': [],
        }
        
        # 체크포인트 저장 경로
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 이전 체크포인트 로드 (점진적 학습용)
        if load_checkpoint:
            self.load_checkpoint(load_checkpoint)
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        logger.info(f"체크포인트 로드 완료: {checkpoint_path}")
        logger.info(f"  - Global Step: {self.global_step}")
        logger.info(f"  - Best Loss: {self.best_loss:.4f}")
    
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
                log_msg = (
                    f"[Step {self.global_step}] "
                    f"Loss={loss.item():.4f}, "
                    f"Avg Loss={total_loss/num_batches:.4f}, "
                    f"LR={current_lr:.2e}"
                )
                logger.info(log_msg)
                self.file_logger.info(log_msg)
        
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
    
    def train(self, train_dataloader, num_epochs: int, stage_name: str = ""):
        """전체 학습 과정"""
        stage_prefix = f"[{stage_name}] " if stage_name else ""
        
        # 학습 시작 로그
        start_msg = f"{stage_prefix}LogBERT 전이 학습 시작 (Pre-trained BERT 파인튜닝)"
        logger.info("=" * 80)
        logger.info(start_msg)
        logger.info("=" * 80)
        logger.info(f"총 에폭: {num_epochs}")
        logger.info(f"배치 크기: {self.config['training']['batch_size']}")
        logger.info(f"학습률: {self.config['training']['learning_rate']}")
        logger.info("=" * 80)
        
        self.file_logger.info("=" * 80)
        self.file_logger.info(start_msg)
        self.file_logger.info("=" * 80)
        self.file_logger.info(f"총 에폭: {num_epochs}")
        self.file_logger.info(f"배치 크기: {self.config['training']['batch_size']}")
        self.file_logger.info(f"학습률: {self.config['training']['learning_rate']}")
        self.file_logger.info("=" * 80)
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_msg = f"{stage_prefix}에폭 {epoch}/{num_epochs} 시작"
            logger.info(f"\n{epoch_start_msg}")
            self.file_logger.info(f"\n{epoch_start_msg}")
            
            avg_loss = self.train_epoch(train_dataloader, epoch)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 메트릭 저장
            self.training_metrics['epochs'].append(epoch)
            self.training_metrics['losses'].append(float(avg_loss))
            self.training_metrics['learning_rates'].append(float(current_lr))
            self.training_metrics['steps'].append(self.global_step)
            
            epoch_end_msg = f"{stage_prefix}에폭 {epoch} 완료 - 평균 Loss: {avg_loss:.4f}, LR: {current_lr:.2e}"
            logger.info(epoch_end_msg)
            self.file_logger.info(epoch_end_msg)
            
            if avg_loss < self.best_loss:
                improvement = self.best_loss - avg_loss
                self.best_loss = avg_loss
                checkpoint_name = f'best_model_{stage_name}' if stage_name else 'best_model'
                self.save_checkpoint(checkpoint_name)
                best_msg = f"{stage_prefix}✅ 최고 성능 모델 저장 (Loss: {avg_loss:.4f}, 개선: {improvement:.4f})"
                logger.info(best_msg)
                self.file_logger.info(best_msg)
            
            epoch_name = f'epoch_{epoch}_{stage_name}' if stage_name else f'epoch_{epoch}'
            self.save_checkpoint(epoch_name)
        
        # 학습 완료 로그
        logger.info("=" * 80)
        logger.info(f"{stage_prefix}전이 학습 완료!")
        logger.info(f"최고 Loss: {self.best_loss:.4f}")
        logger.info("=" * 80)
        
        self.file_logger.info("=" * 80)
        self.file_logger.info(f"{stage_prefix}전이 학습 완료!")
        self.file_logger.info(f"최고 Loss: {self.best_loss:.4f}")
        self.file_logger.info("=" * 80)
        
        return self.best_loss
    
    def save_metrics(self, stage_name: str = ""):
        """학습 메트릭을 JSON 파일로 저장"""
        metrics_file = self.output_dir / f'training_metrics_{stage_name}.json' if stage_name else self.output_dir / 'training_metrics.json'
        
        metrics_data = {
            'stage': stage_name,
            'best_loss': float(self.best_loss),
            'global_step': self.global_step,
            'metrics': self.training_metrics,
            'config': self.config,
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"학습 메트릭 저장: {metrics_file}")
        self.file_logger.info(f"학습 메트릭 저장: {metrics_file}")


def load_config() -> Dict[str, Any]:
    """전이 학습용 설정"""
    return {
        'pretrained_model': 'bert-base-uncased',  # 또는 'distilbert-base-uncased' (더 작음)
        'model': {
            'vocab_size': 10000,
        },
        'training': {
            'batch_size': 8,  # 발열 감소를 위해 배치 크기 축소 (기본값)
            'learning_rate': 0.00001,  # 작은 학습률 (파인튜닝)
            'weight_decay': 0.01,
            'num_epochs': 5,  # 더 많은 에폭 (48GB 메모리로 가능)
            'total_steps': 10000,  # 더 많은 스텝
            'min_lr': 0.000001,
            'max_grad_norm': 1.0,
            'mask_prob': 0.15,
            'log_interval': 50,
            'save_interval': 500,
            'num_workers': 0,
        },
        'data': {
            'preprocessed_dir': '../preprocessing/output',
            'max_seq_length': 512,  # 더 긴 시퀀스 (48GB 메모리로 가능)
            'sample_ratio': 0.1,  # 10% 데이터 (48GB 메모리로 더 많은 데이터 가능)
            'max_files': 50,  # 더 많은 파일 (48GB 메모리로 가능)
        },
        'output_dir': 'checkpoints_transfer',
    }


def get_data_files(preprocessed_dir: str, max_files: int = 10) -> list:
    """데이터 파일 목록"""
    data_dir = Path(preprocessed_dir)
    files = sorted(data_dir.glob("preprocessed_logs_*.json"))
    
    if len(files) > max_files:
        files = random.sample(files, max_files)
    
    logger.info(f"사용할 데이터 파일: {len(files)}개")
    return [str(f) for f in files]


def train_progressive(
    config: Dict[str, Any],
    data_files: List[str],
    start_ratio: float = 0.1,
    step_size: float = 0.1,
    max_ratio: float = 1.0,
    epochs_per_stage: int = 2,
    auto_batch_size: bool = True,
    max_memory_mb: Optional[float] = None,
    min_batch_size: int = 1,
    fixed_batch_size: Optional[int] = None,
):
    """
    점진적 학습: 10%부터 시작해서 단계적으로 데이터 증가
    
    Args:
        config: 학습 설정
        data_files: 데이터 파일 리스트
        start_ratio: 시작 비율 (0.1 = 10%)
        step_size: 각 단계 증가량 (0.1 = 10%)
        max_ratio: 최대 비율 (1.0 = 100%)
        epochs_per_stage: 각 단계당 에폭 수
        auto_batch_size: 데이터 비율에 따라 배치 크기 자동 조정
        max_memory_mb: 최대 메모리 사용량 제한 (MB, None이면 제한 없음)
        min_batch_size: 자동 조정 시 최소 배치 크기
        fixed_batch_size: 고정 배치 크기 (None이면 자동 조정)
    """
    from train_test import SampledLogBERTDataset
    
    # 출력 디렉토리
    output_dir = Path(config.get('output_dir', 'checkpoints_transfer'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 전체 점진적 학습 로그 파일 설정
    progressive_logger = setup_file_logger(output_dir, 'progressive_training')
    
    # 초기 리소스 사용량 확인
    initial_mem = get_memory_usage()
    initial_cpu = get_cpu_usage()
    
    logger.info("=" * 80)
    logger.info("점진적 학습 시작")
    logger.info("=" * 80)
    logger.info(f"시작 비율: {start_ratio*100:.0f}%")
    logger.info(f"단계 크기: {step_size*100:.0f}%")
    logger.info(f"최대 비율: {max_ratio*100:.0f}%")
    logger.info(f"단계당 에폭: {epochs_per_stage}")
    logger.info(f"자동 배치 크기 조정: {auto_batch_size}")
    if max_memory_mb:
        logger.info(f"최대 메모리 제한: {max_memory_mb:.0f}MB")
    logger.info(f"초기 메모리: {initial_mem['rss']:.1f}MB ({initial_mem['percent']:.1f}%)")
    logger.info(f"초기 CPU: {initial_cpu:.1f}%")
    logger.info("=" * 80)
    
    progressive_logger.info("=" * 80)
    progressive_logger.info("점진적 학습 시작")
    progressive_logger.info("=" * 80)
    progressive_logger.info(f"시작 비율: {start_ratio*100:.0f}%")
    progressive_logger.info(f"단계 크기: {step_size*100:.0f}%")
    progressive_logger.info(f"최대 비율: {max_ratio*100:.0f}%")
    progressive_logger.info(f"단계당 에폭: {epochs_per_stage}")
    progressive_logger.info(f"초기 메모리: {initial_mem['rss']:.1f}MB")
    progressive_logger.info("=" * 80)
    
    # 결과 저장
    results = []
    previous_checkpoint = None
    base_batch_size = config['training']['batch_size']
    
    # 누적 데이터 인덱스 저장 (점진적 학습을 위해)
    cumulative_indices = None
    
    # 각 단계별 학습
    current_ratio = start_ratio
    stage_num = 1
    
    while current_ratio <= max_ratio:
        stage_name = f"stage_{stage_num}_{int(current_ratio*100)}pct"
        
        # 이전 단계 정리
        if stage_num > 1:
            logger.info("\n이전 단계 메모리 정리 중...")
            cleanup_memory()
            log_resource_usage(logger, "정리 후 ")
            log_resource_usage(progressive_logger, "정리 후 ")
        
        logger.info("\n" + "=" * 80)
        logger.info(f"단계 {stage_num}: {current_ratio*100:.0f}% 데이터로 학습")
        logger.info("=" * 80)
        
        # 배치 크기 설정
        if fixed_batch_size is not None:
            # 고정 배치 크기 사용
            config['training']['batch_size'] = fixed_batch_size
            logger.info(f"배치 크기 고정: {fixed_batch_size}")
            progressive_logger.info(f"배치 크기: {fixed_batch_size} (고정)")
        elif auto_batch_size:
            # 데이터 비율에 따라 자동 조정
            # 하지만 최소 배치 사이즈는 유지 (성능을 위해)
            adjusted_batch_size = max(min_batch_size, int(base_batch_size * current_ratio))
            config['training']['batch_size'] = adjusted_batch_size
            logger.info(f"배치 크기 자동 조정: {base_batch_size} → {adjusted_batch_size} (비율: {current_ratio*100:.0f}%, 최소: {min_batch_size})")
            progressive_logger.info(f"배치 크기: {adjusted_batch_size}")
        # else: auto_batch_size가 False면 base_batch_size 사용
        
        # 메모리 사용량 확인
        mem_before = get_memory_usage()
        log_resource_usage(logger, f"[단계 {stage_num}] 시작 전 ")
        log_resource_usage(progressive_logger, f"[단계 {stage_num}] 시작 전 ")
        
        # 데이터셋 생성 (각 단계마다 새로운 10% 데이터만 사용)
        # 10% 단계: 0~10% 데이터
        # 20% 단계: 10~20% 데이터 (새로운 10%)
        # 30% 단계: 20~30% 데이터 (새로운 10%)
        # ...
        # 100% 단계: 90~100% 데이터 (새로운 10%)
        # 각 단계마다 항상 전체 데이터의 10%만 사용 (메모리 일정)
        
        prev_ratio = current_ratio - step_size if stage_num > 1 else 0.0
        
        if stage_num == 1:
            # 첫 단계: 0~10% 데이터 사용
            logger.info(f"데이터셋 생성 중... (비율: 0~{current_ratio*100:.0f}%, {step_size*100:.0f}% 사용)")
        else:
            # 이후 단계: 새로운 10% 데이터만 사용
            logger.info(f"데이터셋 생성 중... (비율: {prev_ratio*100:.0f}~{current_ratio*100:.0f}%, 새로운 {step_size*100:.0f}% 사용)")
        
        # 전체 데이터셋 로드 (한 번만, 첫 단계에서)
        if stage_num == 1:
            # 첫 단계: 전체 데이터셋 로드 및 저장
            full_dataset = SampledLogBERTDataset(
                data_files=data_files,
                max_seq_length=config['data']['max_seq_length'],
                mask_prob=config['training'].get('mask_prob', 0.15),
                vocab_size=config['model']['vocab_size'],
                sample_ratio=1.0,  # 전체 데이터 로드
                max_files=config['data'].get('max_files', 10),
            )
            # 전체 세션 저장 (다음 단계에서 재사용)
            cumulative_indices = full_dataset.sessions.copy()
            total_size = len(cumulative_indices)
        
        # 현재 단계에 사용할 데이터 범위 계산
        start_idx = int(total_size * prev_ratio)
        end_idx = int(total_size * current_ratio)
        
        # 새로운 10% 데이터만 사용
        dataset = SampledLogBERTDataset(
            data_files=data_files,
            max_seq_length=config['data']['max_seq_length'],
            mask_prob=config['training'].get('mask_prob', 0.15),
            vocab_size=config['model']['vocab_size'],
            sample_ratio=1.0,
            max_files=config['data'].get('max_files', 10),
        )
        # 해당 범위의 데이터만 사용 (항상 10%)
        dataset.sessions = cumulative_indices[start_idx:end_idx]
        
        logger.info(f"데이터셋 크기: {len(dataset):,}개 샘플 (전체의 {step_size*100:.0f}%, 범위: {prev_ratio*100:.0f}~{current_ratio*100:.0f}%)")
        
        # 데이터셋 생성 후 메모리 확인
        mem_after_dataset = get_memory_usage()
        dataset_mem = mem_after_dataset['rss'] - mem_before['rss']
        logger.info(f"데이터셋 메모리 사용: +{dataset_mem:.1f}MB")
        
        # 메모리 제한 확인 및 배치 크기 자동 조정
        current_batch_size = config['training']['batch_size']
        if max_memory_mb and mem_after_dataset['rss'] > max_memory_mb:
            logger.warning(
                f"⚠️ 메모리 사용량이 제한을 초과했습니다: "
                f"{mem_after_dataset['rss']:.1f}MB > {max_memory_mb:.0f}MB"
            )
            
            # 배치 크기 자동 조정 (메모리 초과 시)
            if fixed_batch_size is None:  # 고정 배치 크기가 아닐 때만 자동 조정
                # 예상 메모리 사용량 계산 (데이터 비율에 비례)
                estimated_memory_ratio = mem_after_dataset['rss'] / max_memory_mb
                # 배치 크기를 줄여서 메모리 사용량 감소
                new_batch_size = max(min_batch_size, int(current_batch_size / estimated_memory_ratio))
                
                if new_batch_size < current_batch_size:
                    config['training']['batch_size'] = new_batch_size
                    logger.warning(
                        f"⚠️ 배치 크기를 자동으로 줄입니다: {current_batch_size} → {new_batch_size} "
                        f"(메모리 초과 방지, 최소: {min_batch_size})"
                    )
                    progressive_logger.warning(
                        f"배치 크기 자동 조정: {current_batch_size} → {new_batch_size} "
                        f"(메모리: {mem_after_dataset['rss']:.1f}MB > {max_memory_mb:.0f}MB)"
                    )
                else:
                    logger.error(
                        f"❌ 메모리 사용량이 너무 높습니다. "
                        f"최소 배치 크기({min_batch_size})로도 메모리 제한을 초과합니다. "
                        f"데이터 비율을 줄이거나 메모리 제한을 높이세요."
                    )
                    progressive_logger.error(
                        f"메모리 초과: {mem_after_dataset['rss']:.1f}MB > {max_memory_mb:.0f}MB, "
                        f"배치 크기 조정 불가 (이미 최소값: {min_batch_size})"
                    )
            else:
                logger.warning(
                    f"⚠️ 고정 배치 크기({fixed_batch_size})를 사용 중입니다. "
                    f"메모리 초과를 방지하려면 --fixed-batch-size를 제거하거나 더 작은 값으로 설정하세요."
                )
                progressive_logger.warning(
                    f"메모리 초과: {mem_after_dataset['rss']:.1f}MB > {max_memory_mb:.0f}MB, "
                    f"고정 배치 크기로 인해 자동 조정 불가"
                )
        
        # 100% 데이터일 때 추가 메모리 체크 및 배치 크기 조정
        if current_ratio >= 0.9 and max_memory_mb:  # 90% 이상일 때
            # 예상 학습 시 메모리 사용량 (배치 크기에 비례)
            estimated_training_memory = mem_after_dataset['rss'] * (1.5 + current_batch_size / 16)
            
            if estimated_training_memory > max_memory_mb * 0.9:  # 90% 이상 사용 예상
                if fixed_batch_size is None:  # 고정 배치 크기가 아닐 때만
                    # 배치 크기를 더 보수적으로 조정
                    safe_batch_size = max(min_batch_size, int(current_batch_size * 0.75))
                    if safe_batch_size < current_batch_size:
                        config['training']['batch_size'] = safe_batch_size
                        logger.info(
                            f"💡 100% 데이터 단계를 위해 배치 크기를 조정: {current_batch_size} → {safe_batch_size} "
                            f"(예상 메모리: {estimated_training_memory:.1f}MB)"
                        )
                        progressive_logger.info(
                            f"100% 데이터 단계 배치 크기 조정: {current_batch_size} → {safe_batch_size}"
                        )
        
        # DataLoader
        dataloader = create_dataloader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        
        # 단계별 폴더 생성
        stage_dir = output_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_checkpoint_dir = stage_dir / 'checkpoints'
        stage_logs_dir = stage_dir / 'logs'
        stage_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 단계별 로그 파일 설정 (단계별 폴더에 저장)
        stage_logger = setup_file_logger(stage_logs_dir, f'stage_{stage_num}_{int(current_ratio*100)}pct')
        
        # 단계별 설정 업데이트 (출력 디렉토리 변경)
        stage_config = config.copy()
        stage_config['output_dir'] = str(stage_dir)
        
        # 학습기 생성 (이전 체크포인트에서 로드)
        if previous_checkpoint:
            load_msg = f"이전 단계 체크포인트에서 로드: {previous_checkpoint}"
            logger.info(load_msg)
            progressive_logger.info(load_msg)
            stage_logger.info(load_msg)
            trainer = TransferTrainer(stage_config, load_checkpoint=previous_checkpoint, file_logger=stage_logger)
        else:
            trainer = TransferTrainer(stage_config, file_logger=stage_logger)
        
        # 학습 전 메모리 확인
        mem_before_train = get_memory_usage()
        log_resource_usage(logger, f"[단계 {stage_num}] 학습 전 ")
        
        # 학습
        best_loss = trainer.train(
            train_dataloader=dataloader,
            num_epochs=epochs_per_stage,
            stage_name=stage_name,
        )
        
        # 학습 후 메모리 확인
        mem_after_train = get_memory_usage()
        train_mem = mem_after_train['rss'] - mem_before_train['rss']
        log_resource_usage(logger, f"[단계 {stage_num}] 학습 후 ")
        log_resource_usage(progressive_logger, f"[단계 {stage_num}] 학습 후 ")
        
        # 학습 메트릭 저장
        trainer.save_metrics(stage_name)
        
        # 결과 저장 (단계별 폴더에 저장)
        checkpoint_path = trainer.checkpoint_dir / 'best_model.pt'
        metrics_path = stage_dir / 'training_metrics.json'
        
        # 체크포인트 이름 변경 (best_model로 통일)
        if (trainer.checkpoint_dir / f'best_model_{stage_name}.pt').exists():
            import shutil
            shutil.move(
                str(trainer.checkpoint_dir / f'best_model_{stage_name}.pt'),
                str(checkpoint_path)
            )
        
        results.append({
            'stage': stage_num,
            'ratio': current_ratio,
            'data_size': len(dataset),
            'best_loss': best_loss,
            'stage_dir': str(stage_dir),
            'checkpoint': str(checkpoint_path),
            'metrics_file': str(metrics_path),
            'log_file': str(stage_logger.handlers[0].baseFilename) if stage_logger.handlers else None,
            'memory_usage': {
                'before_dataset': mem_before['rss'],
                'after_dataset': mem_after_dataset['rss'],
                'after_training': mem_after_train['rss'],
                'dataset_memory': dataset_mem,
                'training_memory': train_mem,
            },
            'batch_size': config['training']['batch_size'],
        })
        
        # 데이터셋과 DataLoader 정리
        del dataset
        del dataloader
        del trainer
        cleanup_memory()
        
        mem_after_cleanup = get_memory_usage()
        logger.info(f"정리 후 메모리: {mem_after_cleanup['rss']:.1f}MB (절약: {mem_after_train['rss'] - mem_after_cleanup['rss']:.1f}MB)")
        
        stage_summary = (
            f"\n단계 {stage_num} 완료:\n"
            f"  - 데이터 비율: {current_ratio*100:.0f}%\n"
            f"  - 데이터 크기: {len(dataset)}개\n"
            f"  - 최고 Loss: {best_loss:.4f}\n"
            f"  - 단계 폴더: {stage_dir}\n"
            f"  - 체크포인트: {checkpoint_path}\n"
            f"  - 메트릭 파일: {metrics_path}\n"
            f"  - 로그 파일: {stage_logger.handlers[0].baseFilename if stage_logger.handlers else 'N/A'}"
        )
        
        logger.info(stage_summary)
        progressive_logger.info(stage_summary)
        
        # 다음 단계를 위한 체크포인트 경로 업데이트
        previous_checkpoint = str(checkpoint_path)
        
        # 다음 단계를 위한 체크포인트 경로 저장
        previous_checkpoint = str(checkpoint_path)
        
        # 다음 단계로
        current_ratio += step_size
        stage_num += 1
        
        # 결과 요약 출력
        logger.info("\n현재까지 결과 요약:")
        for r in results:
            logger.info(
                f"  단계 {r['stage']}: {r['ratio']*100:.0f}% "
                f"(데이터: {r['data_size']}, Loss: {r['best_loss']:.4f})"
            )
    
    # 최종 결과 저장
    results_file = output_dir / 'progressive_training_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    final_summary = (
        "\n" + "=" * 80 + "\n"
        "점진적 학습 완료!\n"
        "=" * 80 + "\n"
        f"결과 저장: {results_file}\n"
        "\n최종 결과:\n"
    )
    
    for r in results:
        final_summary += (
            f"  {r['ratio']*100:.0f}%: Loss={r['best_loss']:.4f}, "
            f"데이터={r['data_size']}개\n"
        )
    
    logger.info(final_summary)
    progressive_logger.info(final_summary)
    
    # 최종 요약을 파일로도 저장
    summary_file = output_dir / 'progressive_training_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(final_summary)
        f.write(f"\n상세 로그:\n")
        f.write(f"  - 전체 로그: {progressive_logger.handlers[0].baseFilename if progressive_logger.handlers else 'N/A'}\n")
        for r in results:
            if r.get('log_file'):
                f.write(f"  - 단계 {r['stage']} ({r['ratio']*100:.0f}%): {r['log_file']}\n")
    
    logger.info(f"요약 파일 저장: {summary_file}")
    progressive_logger.info(f"요약 파일 저장: {summary_file}")
    
    return results


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LogBERT 전이 학습 (M4 Pro 최적화)')
    parser.add_argument('--pretrained', type=str, default='bert-base-uncased',
                       help='Pre-trained 모델명 (bert-base-uncased 또는 distilbert-base-uncased)')
    parser.add_argument('--sample-ratio', type=float, default=0.1,
                       help='데이터 샘플링 비율 (점진적 학습 비활성화 시 사용, 기본: 0.1 = 10%%, M4 Pro 48GB 기준)')
    parser.add_argument('--max-files', type=int, default=50,
                       help='최대 파일 수 (기본: 50, M4 Pro 48GB 기준)')
    parser.add_argument('--progressive', action='store_true',
                       help='점진적 학습 활성화 (10%%부터 시작)')
    parser.add_argument('--start-ratio', type=float, default=0.05,
                       help='점진적 학습 시작 비율 (기본: 0.05 = 5%%)')
    parser.add_argument('--step-size', type=float, default=0.05,
                       help='점진적 학습 단계 크기 (기본: 0.05 = 5%%, 5%씩 10번으로 50%까지 학습)')
    parser.add_argument('--max-ratio', type=float, default=0.5,
                       help='점진적 학습 최대 비율 (기본: 0.5 = 50%%, 발열 감소를 위해 50%로 설정)')
    parser.add_argument('--epochs-per-stage', type=int, default=5,
                       help='점진적 학습 각 단계당 에폭 수 (기본: 5, M4 Pro 48GB 기준)')
    parser.add_argument('--no-auto-batch-size', action='store_true',
                       help='배치 크기 자동 조정 비활성화')
    parser.add_argument('--max-memory-mb', type=float, default=45000,
                       help='최대 메모리 사용량 제한 (MB, 기본: 45000MB = 45GB, M4 Pro 48GB 기준, 10% 데이터에서 16GB 사용 시 더 공격적 설정 가능)')
    parser.add_argument('--min-batch-size', type=int, default=8,
                       help='자동 조정 시 최소 배치 크기 (기본: 8, 발열 감소를 위해 줄임)')
    parser.add_argument('--fixed-batch-size', type=int, default=8,
                       help='고정 배치 크기 (자동 조정 무시, 기본: 8, 발열 감소를 위해 줄임)')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config()
    config['pretrained_model'] = args.pretrained
    config['data']['max_files'] = args.max_files
    
    # 데이터 파일
    data_files = get_data_files(
        config['data']['preprocessed_dir'],
        max_files=config['data']['max_files']
    )
    
    if len(data_files) == 0:
        logger.error("데이터 파일을 찾을 수 없습니다.")
        return
    
    # 점진적 학습 모드
    if args.progressive:
        logger.info("점진적 학습 모드 활성화")
        results = train_progressive(
            config=config,
            data_files=data_files,
            start_ratio=args.start_ratio,
            step_size=args.step_size,
            max_ratio=args.max_ratio,
            epochs_per_stage=args.epochs_per_stage,
            auto_batch_size=not args.no_auto_batch_size,
            max_memory_mb=args.max_memory_mb,
            min_batch_size=args.min_batch_size,
            fixed_batch_size=args.fixed_batch_size,
        )
        logger.info("점진적 학습 완료!")
    else:
        # 기존 방식 (단일 비율)
        logger.info("일반 학습 모드")
        config['data']['sample_ratio'] = args.sample_ratio
        
        # 데이터셋 생성
        logger.info("데이터셋 생성 중...")
        from train_test import SampledLogBERTDataset
        
        dataset = SampledLogBERTDataset(
            data_files=data_files,
            max_seq_length=config['data']['max_seq_length'],
            mask_prob=config['training'].get('mask_prob', 0.15),
            vocab_size=config['model']['vocab_size'],
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
        trainer = TransferTrainer(config)
        
        # 학습 시작
        trainer.train(
            train_dataloader=dataloader,
            num_epochs=config['training']['num_epochs'],
        )


if __name__ == '__main__':
    main()

