#!/usr/bin/env python3
"""
DeepLog 모델 학습 스크립트 (메인)
Tesla V100-DGXS-32GB x 4 환경에 최적화

핵심 기능:
1. Lazy Loading으로 120GB 데이터 처리 (OOM 방지)
2. 멀티 GPU: DistributedDataParallel(DDP) 또는 DataParallel(DP)
   - DDP 사용 시: torchrun --nproc_per_node=4 train.py ...
3. Mixed Precision Training (FP16)
4. Gradient Accumulation (선택)
5. torch.compile (PyTorch 2+, 선택)
6. 체크포인트 저장/복원, Early Stopping
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
import torch.distributed as dist
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
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


def is_distributed() -> bool:
    """DDP 환경 여부 (torchrun 등으로 실행된 경우)"""
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1


def setup_distributed():
    """DDP 초기화. torchrun으로 실행된 경우에만 호출."""
    if not is_distributed():
        return 0, 1, 0
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return rank, world_size, local_rank


class DeepLogTrainer:
    """DeepLog 모델 학습 클래스 (DDP / DP / 단일 GPU 지원)"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
    ):
        """
        Args:
            config: 학습 설정 딕셔너리
            rank: DDP 시 전역 rank (0 ~ world_size-1)
            world_size: DDP 시 총 프로세스 수
            local_rank: DDP 시 현재 프로세스의 GPU 인덱스
        """
        self.config = config
        self.training_config = config.get('training', {})
        self.model_config = config.get('model', {})
        self.data_config = config.get('data', {})
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main_process = rank == 0

        # 디바이스 설정 (DDP 시 해당 프로세스의 GPU 사용)
        if torch.cuda.is_available():
            if world_size > 1:
                self.device = torch.device(f'cuda:{local_rank}')
            else:
                self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True  # 연산 최적화 (LSTM 등)
        else:
            self.device = torch.device('cpu')

        self.num_gpus = torch.cuda.device_count()
        self.use_ddp = world_size > 1
        self.use_multi_gpu = self.training_config.get('use_multi_gpu', True) and (
            self.use_ddp or (self.num_gpus > 1 and not self.use_ddp)
        )

        if self.is_main_process:
            logger.info(f"디바이스: {self.device}")
            logger.info(f"GPU 수: {self.num_gpus} (DDP: {self.use_ddp}, world_size={world_size})")

        # 출력 디렉토리 설정
        output_config = config.get('output', {})
        self.base_dir = Path(output_config.get('base_dir', '/home/zzangdol/silverw/deeplog'))
        self.output_dir = Path(output_config.get('dir', '/home/zzangdol/silverw/deeplog/output'))
        self.checkpoint_dir = self.output_dir / output_config.get('checkpoint_dir', 'checkpoints')
        self.eval_dir = Path(output_config.get('eval_dir', '/home/zzangdol/silverw/deeplog'))

        # 디렉토리 생성 (rank 0만 또는 모든 rank에서)
        if self.is_main_process:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.eval_dir.mkdir(parents=True, exist_ok=True)

        # 모델 생성
        self.model = create_deeplog_model(self.model_config)
        self.model.to(self.device)

        # 멀티 GPU: DDP 우선, 아니면 DataParallel
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[local_rank])
            if self.is_main_process:
                logger.info(f"DDP로 {world_size}개 프로세스에 모델 배포")
        elif self.use_multi_gpu:
            self.model = nn.DataParallel(self.model)
            if self.is_main_process:
                logger.info(f"DataParallel로 {self.num_gpus}개 GPU에 모델 배포")

        # torch.compile (PyTorch 2+, 선택)
        self.use_compile = self.training_config.get('use_compile', False)
        if self.use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                if self.is_main_process:
                    logger.info("torch.compile 적용")
            except Exception as e:
                self.use_compile = False
                if self.is_main_process:
                    logger.warning(f"torch.compile 적용 실패, 비활성화: {e}")

        # Gradient accumulation
        self.gradient_accumulation_steps = self.training_config.get('gradient_accumulation_steps', 1)

        # 옵티마이저
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.training_config.get('learning_rate', 0.001)),
            weight_decay=float(self.training_config.get('weight_decay', 0.0001)),
        )
        
        # Mixed Precision
        self.use_amp = self.training_config.get('mixed_precision', True) and torch.cuda.is_available()
        self.scaler = GradScaler(device='cuda') if self.use_amp else None
        if self.is_main_process:
            logger.info(f"Mixed Precision (FP16): {self.use_amp}")
            logger.info(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        
        # 학습 상태
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # 모니터링 도구
        monitoring_config = config.get('monitoring', {})
        self.gpu_monitor = GPUMonitor(log_interval=monitoring_config.get('gpu_log_interval', 50))
        self.timer = TrainingTimer()
        
        # Early Stopping (현재 활성화됨)
        es_config = self.training_config.get('early_stopping', {})
        self.early_stopping = None
        # es_config가 dict 형태인지 한 번 더 체크하여 안전성 확보
        if isinstance(es_config, dict) and es_config.get('enabled', True):
            self.early_stopping = EarlyStopping(
                patience=es_config.get('patience', 5),
                min_delta=es_config.get('min_delta', 0.0001),
                mode='min',
                restore_best=True
            )
        
        # 학습 로그
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': [],
        }
    
    def train_epoch(self, train_loader, epoch: int) -> float:
        """한 에폭 학습"""
        self.model.train()
        
        loss_meter = AverageMeter('Loss')
        batch_time_meter = AverageMeter('BatchTime')
        
        self.timer.start_epoch(epoch)
        
        acc_steps = self.gradient_accumulation_steps
        pbar = tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch}/{self.training_config.get('num_epochs', 50)}",
            unit='batch',
            leave=True,
            ncols=120,
            disable=not self.is_main_process,
        )
        batch_start = datetime.now()

        for batch_idx, batch in pbar:
            # 배치를 GPU로 이동
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Gradient accumulation: 매 acc_steps마다만 step
            is_accum_step = (batch_idx + 1) % acc_steps == 0
            if batch_idx % acc_steps == 0:
                self.optimizer.zero_grad()

            if self.use_amp:
                with autocast(device_type='cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs['loss']
                    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                        loss = loss.mean()
                    loss = loss / acc_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs['loss']
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                loss = loss / acc_steps
                loss.backward()

            if is_accum_step:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.get('max_grad_norm', 1.0),
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.get('max_grad_norm', 1.0),
                    )
                    self.optimizer.step()
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    if not getattr(self, 'scheduler_needs_metrics', False):
                        self.scheduler.step()

            if is_accum_step:
                self.global_step += 1
                self.timer.step()

            batch_time = (datetime.now() - batch_start).total_seconds()
            batch_start = datetime.now()
            loss_meter.update(loss.item() * acc_steps)
            batch_time_meter.update(batch_time)

            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item() * acc_steps:.4f}',
                'avg': f'{loss_meter.avg:.4f}',
                'lr': f'{current_lr:.2e}',
                'ms/batch': f'{batch_time * 1000:.0f}',
            })

            log_interval = self.training_config.get('log_interval', 100)
            if self.is_main_process and is_accum_step and self.global_step % log_interval == 0:
                gpu_summary = self.gpu_monitor.get_summary()
                time_summary = self.timer.get_summary()
                logger.info(
                    f"[Step {self.global_step}] "
                    f"Loss: {loss.item() * acc_steps:.4f} (avg: {loss_meter.avg:.4f}) | "
                    f"LR: {current_lr:.2e} | "
                    f"GPU: {gpu_summary} | "
                    f"Time: {time_summary['elapsed']} (ETA: {time_summary['eta']})"
                )
            if self.is_main_process and is_accum_step:
                self.gpu_monitor.log_gpu_status(self.global_step)

            save_interval = self.training_config.get('save_interval', 5000)
            if self.is_main_process and is_accum_step and self.global_step % save_interval == 0:
                self.save_checkpoint(f'step_{self.global_step}')

        pbar.close()
        epoch_time = self.timer.end_epoch()
        if self.is_main_process:
            logger.info(f"Epoch {epoch} 완료: Loss={loss_meter.avg:.4f}, 시간={epoch_time:.1f}s")
        return loss_meter.avg
    
    @torch.no_grad()
    def validate(self, val_loader) -> float:
        """검증 (DDP 시 rank별 loss 평균)"""
        self.model.eval()
        loss_meter = AverageMeter('ValLoss')
        pbar = tqdm(val_loader, desc="Validation", leave=False, ncols=100, disable=not self.is_main_process)

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            if self.use_amp:
                with autocast(device_type='cuda'):
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
            pbar.set_postfix({'val_loss': f'{loss_meter.avg:.4f}'})

        pbar.close()
        if self.use_ddp:
            total_loss = torch.tensor([loss_meter.avg], device=self.device)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            loss_meter.avg = (total_loss.item() / self.world_size)
        return loss_meter.avg
    
    def _get_model_for_save_load(self):
        """저장/로드 시 사용할 실제 모델 (DDP/DP 래퍼 제외)"""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    def save_checkpoint(self, name: str):
        """체크포인트 저장 (rank 0만)"""
        if not self.is_main_process:
            return
        checkpoint_path = self.checkpoint_dir / f'{name}.pt'
        model_state = self._get_model_for_save_load().state_dict()
        
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
        
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"체크포인트 저장: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._get_model_for_save_load().load_state_dict(checkpoint['model_state_dict'])
        
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
        
        if self.is_main_process:
            logger.info(f"체크포인트 로드: {checkpoint_path}")
            logger.info(f"  Step: {self.global_step}, Epoch: {self.current_epoch}, Best Loss: {self.best_loss:.4f}")
    
    def train(self, train_loader, val_loader=None, num_epochs: Optional[int] = None):
        """전체 학습 과정"""
        if num_epochs is None:
            num_epochs = self.training_config.get('num_epochs', 50)
        
        # 스케줄러 설정 (IterableDataset은 len() 미지원 → config 추정값 사용, get_total_samples() 호출 안 함)
        acc_steps = self.gradient_accumulation_steps
        batch_size = self.training_config.get('batch_size', 64)
        effective_batch = batch_size * self.world_size * acc_steps
        try:
            steps_per_epoch = len(train_loader) // acc_steps
            total_steps = steps_per_epoch * num_epochs
        except TypeError:
            # LazyLogDataset 등: 파일 전체 스캔(get_total_samples)은 수십 분 걸리므로 사용 안 함
            estimated_samples = self.data_config.get('estimated_total_samples', 29_200_000)
            total_steps = (estimated_samples // effective_batch) * num_epochs
            if self.is_main_process:
                logger.info(f"예상 총 샘플 수(설정): {estimated_samples:,} → 총 스텝: {total_steps:,}")
        self.timer.total_steps = total_steps
        self.timer.total_epochs = num_epochs
        
        self.scheduler = get_lr_scheduler(self.optimizer, self.training_config, total_steps)
        # ReduceLROnPlateau는 에폭 단위로 metrics와 함께 호출해야 함
        self.scheduler_needs_metrics = (
            self.training_config.get('scheduler_type', 'cosine') == 'reduce_on_plateau'
        )
        
        if self.is_main_process:
            print_training_banner(self.config)
            logger.info(f"총 에폭: {num_epochs}")
            logger.info(f"예상 총 스텝: {total_steps:,}")
        self.timer.start()
        if self.use_ddp:
            dist.barrier()

        start_epoch = self.current_epoch + 1
        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            if self.is_main_process:
                logger.info(f"\n{'='*80}")
                logger.info(f"Epoch {epoch}/{num_epochs} 시작")
                logger.info(f"{'='*80}")

            train_loss = self.train_epoch(train_loader, epoch)

            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                if self.is_main_process:
                    logger.info(f"Validation Loss: {val_loss:.4f}")

            self.training_history['train_loss'].append(train_loss)
            if val_loss is not None:
                self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.training_history['epoch_times'].append(self.timer.epoch_times[-1] if self.timer.epoch_times else 0)

            current_loss = val_loss if val_loss is not None else train_loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint('best_model')
                if self.is_main_process:
                    logger.info(f"✅ 새로운 최고 모델 저장! Loss: {current_loss:.4f}")

            self.save_checkpoint(f'epoch_{epoch}')

            if hasattr(self, 'scheduler') and self.scheduler is not None:
                if getattr(self, 'scheduler_needs_metrics', False):
                    self.scheduler.step(current_loss)

            if self.early_stopping is not None:
                if self.early_stopping(current_loss, self.model, epoch):
                    if self.is_main_process:
                        logger.warning("Early Stopping 발동!")
                    break

            if self.is_main_process:
                time_summary = self.timer.get_summary()
                logger.info(
                    f"\nEpoch {epoch} 요약:\n"
                    f"  - Train Loss: {train_loss:.4f}\n"
                    f"  - Val Loss: {f'{val_loss:.4f}' if val_loss is not None else 'N/A'}\n"
                    f"  - Best Loss: {self.best_loss:.4f}\n"
                    f"  - 경과 시간: {time_summary['elapsed']}\n"
                    f"  - 예상 남은 시간: {time_summary['eta']}"
                )
            if self.use_ddp:
                dist.barrier()

        if self.early_stopping is not None and self.early_stopping.stopped:
            self.early_stopping.restore_best_weights(self.model)

        total_time = self.timer.get_elapsed_time()
        if self.is_main_process:
            logger.info(f"\n{'='*80}")
            logger.info(f"학습 완료!")
            logger.info(f"최고 Loss: {self.best_loss:.4f}")
            logger.info(f"총 학습 시간: {self.timer.format_time(total_time)}")
            logger.info(f"{'='*80}")
            history_path = self.base_dir / 'training_history.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2)
            logger.info(f"학습 이력 저장: {history_path}")
            self._run_evaluation()
        if self.use_ddp:
            dist.destroy_process_group()
    
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
    """메인 함수 (단일 프로세스 또는 torchrun DDP 지원)"""
    parser = argparse.ArgumentParser(description='DeepLog 모델 학습')
    parser.add_argument('--config', type=str, default=None, help='설정 파일 경로')
    parser.add_argument('--data-dir', type=str, default=None, help='전처리된 데이터 디렉토리')
    parser.add_argument('--output-dir', type=str, default=None, help='출력 디렉토리')
    parser.add_argument('--resume', type=str, default=None, help='재개할 체크포인트 경로')
    parser.add_argument('--epochs', type=int, default=None, help='학습 에폭 수')
    parser.add_argument('--batch-size', type=int, default=None, help='배치 크기')
    parser.add_argument('--lr', type=float, default=None, help='학습률')
    args = parser.parse_args()

    # DDP 초기화 (torchrun으로 실행된 경우)
    rank, world_size, local_rank = setup_distributed()

    config = load_config(args.config)
    if args.data_dir:
        config['data']['preprocessed_dir'] = args.data_dir
    if args.output_dir:
        config['output']['dir'] = args.output_dir
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr

    output_config = config.get('output', {})
    base_dir = Path(output_config.get('base_dir', '/home/zzangdol/silverw/deeplog'))
    output_dir = Path(output_config.get('dir', '/home/zzangdol/silverw/deeplog/output'))
    base_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = base_dir / output_config.get('log_file', 'training.log')
    setup_logging(str(log_file))

    data_dir = config['data']['preprocessed_dir']
    file_pattern = config['data'].get('file_pattern', 'preprocessed_logs_*.json')
    data_files = get_data_files(data_dir, file_pattern)
    if not data_files:
        logger.error("데이터 파일을 찾을 수 없습니다!")
        return

    if rank == 0:
        logger.info(f"데이터 로더 생성 중... (학습 파일: {len(data_files)}개)")
        logger.info("학습/검증 데이터셋 초기화 시작...")
    train_loader, val_loader = create_dataloaders(
        data_files=data_files,
        config=config,
        validation_split=config['data'].get('validation_split', 0.1),
        rank=rank,
        world_size=world_size,
    )
    if rank == 0:
        logger.info("데이터 로더 생성 완료!")

    if rank == 0:
        logger.info("DeepLogTrainer 초기화 중...")
    trainer = DeepLogTrainer(config, rank=rank, world_size=world_size, local_rank=local_rank)
    if rank == 0:
        logger.info("DeepLogTrainer 초기화 완료!")

    if args.resume:
        if rank == 0:
            logger.info(f"체크포인트에서 재개: {args.resume}")
        trainer.load_checkpoint(args.resume)

    if rank == 0:
        logger.info("학습 시작 준비 완료, trainer.train() 호출...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs or config['training']['num_epochs'],
    )
    if rank == 0:
        logger.info("학습 완료!")


if __name__ == '__main__':
    main()
