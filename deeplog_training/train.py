#!/usr/bin/env python3
"""
DeepLog 모델 학습 스크립트 (메인)
Tesla V100-DGXS-32GB x 4 환경에 최적화

핵심 기능:
1. Lazy Loading으로 120GB 데이터 처리 (OOM 방지)
2. 멀티 GPU 학습 (DataParallel)
3. Mixed Precision Training (FP16)
4. 상세한 학습 모니터링 (GPU 상태, 시간, loss 등)
5. 체크포인트 저장/복원
6. Early Stopping
"""

import os
import sys
import json
import yaml
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from model import DeepLog, create_deeplog_model
from dataset import LazyLogDataset, InMemoryLogDataset, create_dataloaders, collate_fn
from utils import (
    GPUMonitor, TrainingTimer, EarlyStopping, AverageMeter,
    get_lr_scheduler, setup_logging, print_training_banner
)

# 경고 억제
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.parallel._functions')

logger = logging.getLogger(__name__)


class DeepLogTrainer:
    """DeepLog 모델 학습 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 학습 설정 딕셔너리
        """
        self.config = config
        self.training_config = config.get('training', {})
        self.model_config = config.get('model', {})
        self.data_config = config.get('data', {})
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count()
        self.use_multi_gpu = self.training_config.get('use_multi_gpu', True) and self.num_gpus > 1
        
        logger.info(f"디바이스: {self.device}")
        logger.info(f"GPU 수: {self.num_gpus}")
        
        # 출력 디렉토리 설정
        output_config = config.get('output', {})
        self.base_dir = Path(output_config.get('base_dir', '/home/zzangdol/silverw/deeplog'))
        self.output_dir = Path(output_config.get('dir', '/home/zzangdol/silverw/deeplog/output'))
        self.checkpoint_dir = self.output_dir / output_config.get('checkpoint_dir', 'checkpoints')
        self.eval_dir = Path(output_config.get('eval_dir', '/home/zzangdol/silverw/deeplog'))
        
        # 디렉토리 생성
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 생성
        self.model = create_deeplog_model(self.model_config)
        self.model.to(self.device)
        
        # 멀티 GPU 래핑
        if self.use_multi_gpu:
            logger.info(f"DataParallel로 {self.num_gpus}개 GPU에 모델 배포")
            self.model = nn.DataParallel(self.model)
        
        # 옵티마이저
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.training_config.get('learning_rate', 0.001)),
            weight_decay=float(self.training_config.get('weight_decay', 0.0001)),
        )
        
        # Mixed Precision
        self.use_amp = self.training_config.get('mixed_precision', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        logger.info(f"Mixed Precision (FP16): {self.use_amp}")
        
        # 학습 상태
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # 모니터링 도구
        monitoring_config = config.get('monitoring', {})
        self.gpu_monitor = GPUMonitor(log_interval=monitoring_config.get('gpu_log_interval', 50))
        self.timer = TrainingTimer()
        
        # Early Stopping 설정
        es_config = self.training_config.get('early_stopping', {})
        if es_config.get('enabled', False):
            self.early_stopping = EarlyStopping(
                patience=es_config.get('patience', 10),
                min_delta=es_config.get('min_delta', 0.0001),
                mode='min',
                restore_best=True
            )
        else:
            self.early_stopping = None
        
        # 학습 로그
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': [],
        }
        
        # 스케줄러 초기화 (train에서 설정됨)
        self.scheduler = None
    
    def train_epoch(self, train_loader, epoch: int) -> float:
        """한 에폭 학습"""
        self.model.train()
        
        loss_meter = AverageMeter('Loss')
        batch_time_meter = AverageMeter('BatchTime')
        
        self.timer.start_epoch(epoch)
        
        pbar = tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch}/{self.training_config.get('num_epochs', 50)}",
            unit='batch',
            leave=True,
            ncols=120,
        )
        
        batch_start = datetime.now()
        
        # ReduceLROnPlateau 여부 확인
        is_plateau_scheduler = self.training_config.get('scheduler_type') == 'reduce_on_plateau'
        
        for batch_idx, batch in pbar:
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs['loss']
                    
                    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                        loss = loss.mean()
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.get('max_grad_norm', 1.0)
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs['loss']
                
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.get('max_grad_norm', 1.0)
                )
                self.optimizer.step()
            
            # 스케줄러 업데이트 (ReduceLROnPlateau 제외 - 에폭 단위로만 업데이트)
            if self.scheduler is not None and not is_plateau_scheduler:
                self.scheduler.step()
            
            # 통계 업데이트
            batch_time = (datetime.now() - batch_start).total_seconds()
            batch_start = datetime.now()
            
            loss_meter.update(loss.item())
            batch_time_meter.update(batch_time)
            
            self.global_step += 1
            self.timer.step()
            
            # Progress bar 업데이트
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{loss_meter.avg:.4f}',
                'lr': f'{current_lr:.2e}',
                'ms/batch': f'{batch_time*1000:.0f}',
            })
            
            # 상세 로깅
            log_interval = self.training_config.get('log_interval', 100)
            if self.global_step % log_interval == 0:
                gpu_summary = self.gpu_monitor.get_summary()
                time_summary = self.timer.get_summary()
                
                logger.info(
                    f"[Step {self.global_step}] "
                    f"Loss: {loss.item():.4f} (avg: {loss_meter.avg:.4f}) | "
                    f"LR: {current_lr:.2e} | "
                    f"GPU: {gpu_summary} | "
                    f"Time: {time_summary['elapsed']} (ETA: {time_summary['eta']})"
                )
            
            # GPU 상태 로깅
            self.gpu_monitor.log_gpu_status(self.global_step)
            
            # 체크포인트 저장
            save_interval = self.training_config.get('save_interval', 5000)
            if self.global_step % save_interval == 0:
                self.save_checkpoint(f'step_{self.global_step}')
        
        pbar.close()
        
        epoch_time = self.timer.end_epoch()
        
        logger.info(f"Epoch {epoch} 완료: Loss={loss_meter.avg:.4f}, 시간={epoch_time:.1f}s")
        
        return loss_meter.avg
    
    @torch.no_grad()
    def validate(self, val_loader, calculate_topg: bool = True) -> Dict[str, float]:
        """검증 (Loss + Top-g Accuracy)
        
        Args:
            val_loader: 검증 데이터 로더
            calculate_topg: Top-g Accuracy 계산 여부
        
        Returns:
            {'val_loss': float, 'topg_accuracy': float}
        """
        self.model.eval()
        
        loss_meter = AverageMeter('ValLoss')
        
        # Top-g Accuracy 계산용
        eval_config = self.config.get('evaluation', {})
        g = eval_config.get('top_g', 9)
        topg_correct = 0
        topg_total = 0
        
        pbar = tqdm(val_loader, desc="Validation", leave=False, ncols=120)
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            
            loss = outputs['loss']
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            
            loss_meter.update(loss.item())
            
            # Top-g Accuracy 계산
            if calculate_topg:
                logits = outputs['logits']  # [batch, seq_len, vocab_size]
                
                # 각 위치에서 상위 g개 예측
                _, top_indices = torch.topk(logits, k=g, dim=-1)  # [batch, seq_len, g]
                
                # 정답이 상위 g개 안에 있는지 확인
                labels_expanded = labels.unsqueeze(-1).expand_as(top_indices)
                matches = (top_indices == labels_expanded).any(dim=-1)  # [batch, seq_len]
                
                # padding 제외 (-100은 padding 토큰)
                valid_mask = (labels != -100)
                topg_correct += (matches & valid_mask).sum().item()
                topg_total += valid_mask.sum().item()
            
            # Progress bar 업데이트
            if calculate_topg and topg_total > 0:
                current_acc = topg_correct / topg_total
                pbar.set_postfix({
                    'val_loss': f'{loss_meter.avg:.4f}',
                    f'top{g}_acc': f'{current_acc:.4f}'
                })
            else:
                pbar.set_postfix({'val_loss': f'{loss_meter.avg:.4f}'})
        
        pbar.close()
        
        result = {'val_loss': loss_meter.avg}
        
        if calculate_topg and topg_total > 0:
            topg_accuracy = topg_correct / topg_total
            result['topg_accuracy'] = topg_accuracy
            logger.info(f"Top-{g} Accuracy: {topg_accuracy:.4f} ({topg_correct:,}/{topg_total:,})")
        
        return result
    
    @torch.no_grad()
    def calculate_topg_accuracy(self, val_loader, g: int = 9) -> float:
        """DeepLog Top-g Accuracy 계산
        
        Args:
            val_loader: 검증 데이터 로더
            g: 상위 g개 예측 후보 (논문에서는 g=9)
        
        Returns:
            Top-g Accuracy (0.0 ~ 1.0)
        """
        self.model.eval()
        correct = 0
        total = 0
        
        for batch in tqdm(val_loader, desc="Top-g Accuracy", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids=input_ids)
            logits = outputs['logits']  # [batch, seq_len, vocab_size]
            
            # 각 위치에서 상위 g개 예측
            _, top_indices = torch.topk(logits, k=g, dim=-1)  # [batch, seq_len, g]
            
            # 정답이 상위 g개 안에 있는지 확인
            labels_expanded = labels.unsqueeze(-1).expand_as(top_indices)
            matches = (top_indices == labels_expanded).any(dim=-1)  # [batch, seq_len]
            
            # padding 제외
            valid_mask = (labels != -100)
            correct += (matches & valid_mask).sum().item()
            total += valid_mask.sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def save_checkpoint(self, name: str):
        """체크포인트 저장"""
        checkpoint_path = self.checkpoint_dir / f'{name}.pt'
        
        # DataParallel 모델인 경우 module을 통해 접근
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'config': self.config,
            'training_history': self.training_history,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"체크포인트 저장: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 모델 가중치 로드
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 옵티마이저 로드
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 상태 복원
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        # Scaler 로드
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"체크포인트 로드: {checkpoint_path}")
        logger.info(f"  Step: {self.global_step}, Epoch: {self.current_epoch}, Best Loss: {self.best_loss:.4f}")
    
    def train(self, train_loader, val_loader=None, num_epochs: Optional[int] = None):
        """전체 학습 과정"""
        if num_epochs is None:
            num_epochs = self.training_config.get('num_epochs', 50)
        
        # 스케줄러 설정
        try:
            total_steps = len(train_loader) * num_epochs
        except TypeError:
            batch_size = self.training_config.get('batch_size', 64)
            num_files = len(train_loader.dataset.data_files) if hasattr(train_loader.dataset, 'data_files') else 50
            estimated_samples_per_file = 1500000
            estimated_samples = num_files * estimated_samples_per_file
            total_steps = (estimated_samples // batch_size) * num_epochs
            logger.info(f"총 스텝 추정치 사용: {total_steps:,} (파일 {num_files}개 기준)")
        
        self.timer.total_steps = total_steps
        self.timer.total_epochs = num_epochs
        
        self.scheduler = get_lr_scheduler(self.optimizer, self.training_config, total_steps)
        
        # ReduceLROnPlateau 여부 확인
        is_plateau_scheduler = self.training_config.get('scheduler_type') == 'reduce_on_plateau'
        
        # 학습 시작 로그
        print_training_banner(self.config)
        logger.info(f"총 에폭: {num_epochs}")
        logger.info(f"예상 총 스텝: {total_steps:,}")
        logger.info(f"스케줄러 타입: {self.training_config.get('scheduler_type', 'cosine')}")
        
        self.timer.start()
        
        # 에폭 루프
        start_epoch = self.current_epoch + 1
        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch}/{num_epochs} 시작")
            logger.info(f"{'='*80}")
            
            # 학습
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 검증
            val_result = None
            val_loss = None
            topg_accuracy = None
            
            if val_loader is not None:
                eval_config = self.config.get('evaluation', {})
                eval_interval = eval_config.get('eval_interval', 1)
                
                # Top-g Accuracy는 매 에폭마다 또는 설정된 간격마다 계산
                calculate_topg = (epoch % eval_interval == 0)
                
                val_result = self.validate(val_loader, calculate_topg=calculate_topg)
                val_loss = val_result['val_loss']
                topg_accuracy = val_result.get('topg_accuracy', None)
                
                logger.info(f"Validation Loss: {val_loss:.4f}")
                if topg_accuracy is not None:
                    logger.info(f"Top-g Accuracy: {topg_accuracy:.4f}")
            
            # ReduceLROnPlateau 스케줄러 업데이트 (validation loss 기준)
            if self.scheduler is not None and is_plateau_scheduler:
                if val_loss is not None:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step(val_loss)
                    new_lr = self.optimizer.param_groups[0]['lr']
                    
                    if new_lr != old_lr:
                        logger.warning(f"🔽 학습률 감소: {old_lr:.2e} → {new_lr:.2e}")
            
            # 학습 이력 저장
            self.training_history['train_loss'].append(train_loss)
            if val_loss is not None:
                self.training_history['val_loss'].append(val_loss)
            if topg_accuracy is not None:
                if 'topg_accuracy' not in self.training_history:
                    self.training_history['topg_accuracy'] = []
                self.training_history['topg_accuracy'].append(topg_accuracy)
            
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.training_history['epoch_times'].append(self.timer.epoch_times[-1] if self.timer.epoch_times else 0)
            
            # 최고 모델 저장 (validation loss 기준)
            current_loss = val_loss if val_loss is not None else train_loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint('best_model')
                logger.info(f"✅ 새로운 최고 모델 저장! Loss: {current_loss:.4f}")
            
            # 에폭 체크포인트 저장
            self.save_checkpoint(f'epoch_{epoch}')
            
            # Early Stopping 체크 (validation loss 기준)
            if self.early_stopping is not None:
                if self.early_stopping(current_loss, self.model, epoch):
                    logger.warning("🛑 Early Stopping 발동!")
                    break
            
            # 에폭 요약
            time_summary = self.timer.get_summary()
            summary_lines = [
                f"\nEpoch {epoch} 요약:",
                f"  - Train Loss: {train_loss:.4f}",
                f"  - Val Loss: {val_loss:.4f if val_loss else 'N/A'}",
            ]
            
            if topg_accuracy is not None:
                summary_lines.append(f"  - Top-g Accuracy: {topg_accuracy:.4f}")
            
            summary_lines.extend([
                f"  - Best Loss: {self.best_loss:.4f}",
                f"  - Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}",
                f"  - 경과 시간: {time_summary['elapsed']}",
                f"  - 예상 남은 시간: {time_summary['eta']}"
            ])
            
            logger.info("\n".join(summary_lines))
        
        # 학습 완료
        if self.early_stopping is not None and self.early_stopping.stopped:
            logger.info("Early Stopping으로 인한 조기 종료 - 최고 모델 가중치 복원 중...")
            self.early_stopping.restore_best_weights(self.model)
        
        total_time = self.timer.get_elapsed_time()
        logger.info(f"\n{'='*80}")
        logger.info(f"학습 완료!")
        logger.info(f"최고 Loss: {self.best_loss:.4f}")
        logger.info(f"총 학습 시간: {self.timer.format_time(total_time)}")
        logger.info(f"{'='*80}")
        
        # 학습 이력 저장
        history_path = self.base_dir / 'training_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        logger.info(f"학습 이력 저장: {history_path}")
        
        # 평가 자동 실행
        self._run_evaluation()
    
    def _run_evaluation(self):
        """학습 완료 후 모델 성능 평가 자동 실행"""
        logger.info("\n" + "=" * 80)
        logger.info("모델 성능 평가 시작...")
        logger.info("=" * 80)
        
        try:
            from evaluate import DeepLogEvaluator, evaluate_model
            
            # 최고 모델 체크포인트 경로
            best_checkpoint = self.checkpoint_dir / 'best_model.pt'
            
            if not best_checkpoint.exists():
                logger.warning(f"최고 모델 체크포인트를 찾을 수 없습니다: {best_checkpoint}")
                return
            
            # 데이터 파일 목록
            data_dir = self.data_config.get('preprocessed_dir', '')
            file_pattern = self.data_config.get('file_pattern', 'preprocessed_logs_*.json')
            
            data_path = Path(data_dir)
            data_files = sorted(data_path.glob(file_pattern))
            
            if not data_files:
                data_files = sorted(data_path.glob('*.json'))
            
            if not data_files:
                logger.warning(f"평가 데이터 파일을 찾을 수 없습니다: {data_dir}")
                return
            
            # 평가 실행
            results = evaluate_model(
                checkpoint_path=str(best_checkpoint),
                config=self.config,
                data_files=[str(f) for f in data_files],
                output_dir=str(self.eval_dir),
                max_samples=50000,
            )
            
            logger.info("모델 성능 평가 완료!")
            logger.info(f"평가 결과 저장 위치: {self.eval_dir}")
            
        except ImportError as e:
            logger.warning(f"평가 모듈 로드 실패: {e}")
            logger.info("수동으로 평가하려면: python evaluate.py --checkpoint <checkpoint_path>")
        except Exception as e:
            logger.error(f"평가 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()


def get_data_files(data_dir: str, pattern: str = "preprocessed_logs_*.json") -> List[str]:
    """데이터 파일 목록 가져오기"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
    
    files = sorted(data_path.glob(pattern))
    
    if not files:
        # 대안: 모든 JSON 파일 검색
        files = sorted(data_path.glob("*.json"))
    
    logger.info(f"발견된 데이터 파일: {len(files)}개")
    
    return [str(f) for f in files]


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """설정 파일 로드"""
    if config_path is None:
        base_dir = Path(__file__).parent
        config_path = str(base_dir / 'config.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"설정 로드: {config_path}")
        return config
    else:
        logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
        # 기본 설정 반환
        return {
            'model': {
                'vocab_size': 10000,
                'embedding_dim': 128,
                'hidden_size': 256,
                'num_layers': 2,
                'dropout': 0.2,
            },
            'training': {
                'batch_size': 64,
                'learning_rate': 0.001,
                'num_epochs': 50,
                'max_grad_norm': 1.0,
            },
            'data': {
                'preprocessed_dir': '/home/zzangdol/RADAR/preprocessing/output',
                'max_seq_length': 512,
            },
            'output': {
                'base_dir': '/home/zzangdol/silverw/deeplog',
                'dir': '/home/zzangdol/silverw/deeplog/output',
                'checkpoint_dir': 'checkpoints',
                'eval_dir': '/home/zzangdol/silverw/deeplog',
            },
        }


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='DeepLog 모델 학습')
    parser.add_argument('--config', type=str, default=None,
                       help='설정 파일 경로')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='전처리된 데이터 디렉토리')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='출력 디렉토리')
    parser.add_argument('--resume', type=str, default=None,
                       help='재개할 체크포인트 경로')
    parser.add_argument('--epochs', type=int, default=None,
                       help='학습 에폭 수')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='배치 크기')
    parser.add_argument('--lr', type=float, default=None,
                       help='학습률')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 명령줄 인자로 설정 덮어쓰기
    if args.data_dir:
        config['data']['preprocessed_dir'] = args.data_dir
    if args.output_dir:
        config['output']['dir'] = args.output_dir
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # 로깅 설정 (base_dir에 로그 저장)
    output_config = config.get('output', {})
    base_dir = Path(output_config.get('base_dir', '/home/zzangdol/silverw/deeplog'))
    output_dir = Path(output_config.get('dir', '/home/zzangdol/silverw/deeplog/output'))
    base_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = base_dir / output_config.get('log_file', 'training.log')
    setup_logging(str(log_file))
    
    # 데이터 파일 목록
    data_dir = config['data']['preprocessed_dir']
    file_pattern = config['data'].get('file_pattern', 'preprocessed_logs_*.json')
    data_files = get_data_files(data_dir, file_pattern)
    
    if not data_files:
        logger.error("데이터 파일을 찾을 수 없습니다!")
        return
    
    # DataLoader 생성
    logger.info("데이터 로더 생성 중...")
    train_loader, val_loader = create_dataloaders(
        data_files=data_files,
        config=config,
        validation_split=config['data'].get('validation_split', 0.1),
    )
    
    # 학습기 생성
    trainer = DeepLogTrainer(config)
    
    # 체크포인트에서 재개
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 학습 시작
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs or config['training']['num_epochs'],
    )


if __name__ == '__main__':
    main()
