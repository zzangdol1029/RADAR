#!/usr/bin/env python3
"""
GPU 모니터링 및 학습 유틸리티
Tesla V100-DGXS-32GB x 4 환경 최적화
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import subprocess

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GPUMonitor:
    """
    GPU 상태 모니터링 클래스
    
    메모리 사용량, 활용률, 온도 등을 추적합니다.
    """
    
    def __init__(self, log_interval: int = 50):
        """
        Args:
            log_interval: 로깅 간격 (배치 수)
        """
        self.log_interval = log_interval
        self.has_nvidia_smi = self._check_nvidia_smi()
        self.num_gpus = torch.cuda.device_count()
        
        logger.info(f"GPU 모니터 초기화: {self.num_gpus}개 GPU 감지")
    
    def _check_nvidia_smi(self) -> bool:
        """nvidia-smi 사용 가능 여부 확인"""
        try:
            subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def get_gpu_memory_info(self) -> Dict[int, Dict[str, float]]:
        """
        각 GPU의 메모리 정보 반환
        
        Returns:
            {gpu_id: {'used': MB, 'total': MB, 'free': MB, 'percent': %}}
        """
        if not torch.cuda.is_available():
            return {}
        
        info = {}
        for i in range(self.num_gpus):
            try:
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**2  # MB
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**2  # MB
                
                info[i] = {
                    'allocated': mem_allocated,
                    'reserved': mem_reserved,
                    'total': mem_total,
                    'free': mem_total - mem_reserved,
                    'percent': (mem_reserved / mem_total) * 100,
                }
            except Exception as e:
                logger.warning(f"GPU {i} 메모리 정보 획득 실패: {e}")
        
        return info
    
    def get_gpu_utilization(self) -> Dict[int, Dict[str, Any]]:
        """
        nvidia-smi를 사용하여 GPU 활용률 조회
        
        Returns:
            {gpu_id: {'utilization': %, 'temperature': °C, 'power': W}}
        """
        if not self.has_nvidia_smi:
            return {}
        
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,utilization.gpu,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            info = {}
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpu_id = int(parts[0])
                        info[gpu_id] = {
                            'utilization': float(parts[1]) if parts[1] != '[N/A]' else 0,
                            'temperature': float(parts[2]) if parts[2] != '[N/A]' else 0,
                            'power': float(parts[3]) if parts[3] != '[N/A]' else 0,
                        }
            
            return info
            
        except Exception as e:
            logger.debug(f"GPU 활용률 조회 실패: {e}")
            return {}
    
    def get_gpu_names(self) -> List[str]:
        """GPU 이름 목록 반환"""
        names = []
        for i in range(self.num_gpus):
            try:
                name = torch.cuda.get_device_name(i)
                names.append(name)
            except Exception:
                names.append(f"GPU {i}")
        return names
    
    def log_gpu_status(self, step: int = 0, prefix: str = ""):
        """GPU 상태 로깅"""
        if step % self.log_interval != 0:
            return
        
        mem_info = self.get_gpu_memory_info()
        util_info = self.get_gpu_utilization()
        
        log_lines = [f"{prefix}GPU 상태 (Step {step}):"]
        
        for gpu_id in range(self.num_gpus):
            mem = mem_info.get(gpu_id, {})
            util = util_info.get(gpu_id, {})
            
            mem_str = f"메모리: {mem.get('allocated', 0):.0f}/{mem.get('total', 0):.0f}MB ({mem.get('percent', 0):.1f}%)"
            util_str = f"활용률: {util.get('utilization', 0):.0f}%"
            temp_str = f"온도: {util.get('temperature', 0):.0f}°C"
            power_str = f"전력: {util.get('power', 0):.0f}W"
            
            log_lines.append(f"  GPU {gpu_id}: {mem_str} | {util_str} | {temp_str} | {power_str}")
        
        logger.info("\n".join(log_lines))
    
    def get_summary(self) -> str:
        """GPU 상태 요약 문자열 반환"""
        mem_info = self.get_gpu_memory_info()
        util_info = self.get_gpu_utilization()
        
        parts = []
        for gpu_id in range(self.num_gpus):
            mem = mem_info.get(gpu_id, {})
            util = util_info.get(gpu_id, {})
            parts.append(
                f"[{gpu_id}:{mem.get('allocated', 0):.0f}MB|{util.get('utilization', 0):.0f}%]"
            )
        
        return " ".join(parts)


class TrainingTimer:
    """
    학습 시간 측정 및 예측 클래스
    """
    
    def __init__(self, total_steps: Optional[int] = None, total_epochs: Optional[int] = None):
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        
        self.start_time = None
        self.epoch_start_time = None
        self.batch_times = []
        self.epoch_times = []
        
        self._current_step = 0
        self._current_epoch = 0
    
    def start(self):
        """전체 학습 시작"""
        self.start_time = time.time()
    
    def start_epoch(self, epoch: int):
        """에폭 시작"""
        self._current_epoch = epoch
        self.epoch_start_time = time.time()
        self.batch_times = []
    
    def end_epoch(self) -> float:
        """에폭 종료, 소요 시간 반환"""
        elapsed = time.time() - self.epoch_start_time
        self.epoch_times.append(elapsed)
        return elapsed
    
    def step(self):
        """배치 처리 완료"""
        self._current_step += 1
        if self.batch_times:
            self.batch_times.append(time.time())
        else:
            self.batch_times = [time.time()]
    
    def get_batch_time(self) -> float:
        """평균 배치 처리 시간"""
        if len(self.batch_times) < 2:
            return 0.0
        
        times = []
        for i in range(1, len(self.batch_times)):
            times.append(self.batch_times[i] - self.batch_times[i-1])
        
        return sum(times) / len(times) if times else 0.0
    
    def get_elapsed_time(self) -> float:
        """전체 경과 시간"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_eta(self) -> Optional[float]:
        """예상 남은 시간 (초)"""
        if self.total_steps is None or self._current_step == 0:
            return None
        
        elapsed = self.get_elapsed_time()
        rate = self._current_step / elapsed
        remaining_steps = self.total_steps - self._current_step
        
        return remaining_steps / rate if rate > 0 else None
    
    def format_time(self, seconds: float) -> str:
        """초를 읽기 쉬운 형식으로 변환"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def get_summary(self) -> Dict[str, str]:
        """시간 정보 요약"""
        elapsed = self.get_elapsed_time()
        eta = self.get_eta()
        batch_time = self.get_batch_time()
        
        return {
            'elapsed': self.format_time(elapsed),
            'eta': self.format_time(eta) if eta else 'N/A',
            'batch_time': f"{batch_time*1000:.1f}ms" if batch_time > 0 else 'N/A',
            'throughput': f"{1/batch_time:.1f} batch/s" if batch_time > 0 else 'N/A',
        }


class EarlyStopping:
    """
    Early Stopping 구현
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0001,
        mode: str = 'min',
        restore_best: bool = True,
    ):
        """
        Args:
            patience: 개선 없이 지속될 수 있는 에폭 수
            min_delta: 개선으로 간주되는 최소 변화량
            mode: 'min' (loss 감소) 또는 'max' (accuracy 증가)
            restore_best: 최고 모델 복원 여부
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.counter = 0
        self.best_state_dict = None
        self.stopped = False
    
    def __call__(self, score: float, model: nn.Module, epoch: int) -> bool:
        """
        Early Stopping 체크
        
        Args:
            score: 현재 점수 (loss 또는 metric)
            model: 모델
            epoch: 현재 에폭
        
        Returns:
            True if 학습 중단 필요
        """
        if self.mode == 'min':
            is_better = score < (self.best_score - self.min_delta)
        else:
            is_better = score > (self.best_score + self.min_delta)
        
        if is_better:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
            if self.restore_best:
                # 최고 모델 상태 저장
                if isinstance(model, nn.DataParallel):
                    self.best_state_dict = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
                else:
                    self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            logger.info(f"✅ 새로운 최고 성능: {score:.6f} (Epoch {epoch})")
        else:
            self.counter += 1
            logger.info(f"⚠️ 개선 없음: {self.counter}/{self.patience} (최고: {self.best_score:.6f} @ Epoch {self.best_epoch})")
            
            if self.counter >= self.patience:
                self.stopped = True
                logger.warning(f"🛑 Early Stopping! {self.patience} 에폭 동안 개선 없음")
                return True
        
        return False
    
    def restore_best_weights(self, model: nn.Module):
        """최고 모델 가중치 복원"""
        if self.best_state_dict is not None:
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(self.best_state_dict)
            else:
                model.load_state_dict(self.best_state_dict)
            logger.info(f"최고 모델 복원: Epoch {self.best_epoch}, Score: {self.best_score:.6f}")


class AverageMeter:
    """평균값 추적"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def get_lr_scheduler(optimizer, config: Dict[str, Any], total_steps: int):
    """
    학습률 스케줄러 생성
    
    Args:
        optimizer: 옵티마이저
        config: 학습 설정
        total_steps: 총 학습 스텝 수
    
    Returns:
        학습률 스케줄러
    """
    scheduler_type = config.get('scheduler_type', 'cosine')
    min_lr = float(config.get('min_lr', 1e-6))
    warmup_steps = config.get('warmup_steps', 1000)
    
    if scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=min_lr
        )
    elif scheduler_type == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(
            optimizer,
            step_size=total_steps // 10,
            gamma=0.5
        )
    elif scheduler_type == 'reduce_on_plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=min_lr
        )
    else:
        scheduler = None
    
    # 워밍업 스케줄러 (Linear warmup)
    if warmup_steps > 0 and scheduler is not None:
        from torch.optim.lr_scheduler import LambdaLR, SequentialLR
        
        def warmup_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        
        warmup_scheduler = LambdaLR(optimizer, warmup_lambda)
        
        # PyTorch 2.0+ SequentialLR 사용
        try:
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_steps]
            )
        except AttributeError:
            # PyTorch 이전 버전 호환
            logger.warning("SequentialLR 미지원, 워밍업 없이 진행")
    
    return scheduler


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    로깅 설정
    
    Args:
        log_file: 로그 파일 경로
        level: 로깅 레벨
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )


def print_training_banner(config: Dict[str, Any], num_samples: int = 0):
    """학습 시작 배너 출력"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         DeepLog Training Started                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
"""
    
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    info_lines = [
        f"║  Model: DeepLog (LSTM)                                                       ║",
        f"║  - Vocab Size: {model_config.get('vocab_size', 10000):,}                                                          ║",
        f"║  - Hidden Size: {model_config.get('hidden_size', 256)}                                                            ║",
        f"║  - LSTM Layers: {model_config.get('num_layers', 2)}                                                              ║",
        f"║  - Embedding Dim: {model_config.get('embedding_dim', 128)}                                                          ║",
        f"╠══════════════════════════════════════════════════════════════════════════════╣",
        f"║  Training Configuration:                                                      ║",
        f"║  - Batch Size: {training_config.get('batch_size', 64)}                                                            ║",
        f"║  - Learning Rate: {training_config.get('learning_rate', 0.001)}                                                       ║",
        f"║  - Epochs: {training_config.get('num_epochs', 50)}                                                                 ║",
        f"║  - Samples: {num_samples:,}                                                            ║",
        f"╠══════════════════════════════════════════════════════════════════════════════╣",
        f"║  GPU Configuration:                                                           ║",
        f"║  - Device Count: {torch.cuda.device_count()}                                                              ║",
    ]
    
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        info_lines.append(f"║  - GPU {i}: {name[:40]:40s} ({mem:.0f}GB)     ║")
    
    info_lines.append("╚══════════════════════════════════════════════════════════════════════════════╝")
    
    print(banner + "\n".join(info_lines))
