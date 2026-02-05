#!/usr/bin/env python3
"""
LogBERT 통합 학습 스크립트
CUDA GPU, Intel XPU, CPU를 자동으로 감지하여 학습합니다.
"""

import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Intel Extension for PyTorch (선택적)
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

# 로컬 모듈 import
from model import create_logbert_model
from dataset import LogBERTDataset, create_dataloader, collate_fn

logger = logging.getLogger(__name__)


def setup_logging(log_file: Path = None):
    """로깅 설정 - UTF-8 인코딩 지원"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Windows에서 콘솔 인코딩 설정
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # 파일 핸들러
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)
        logger.info(f"📝 로그 파일: {log_file}")
    
    return root_logger


def get_device():
    """최적의 디바이스 자동 감지 (XPU > CUDA > CPU)"""
    # Intel XPU 확인
    if IPEX_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        logger.info(f"🚀 Intel GPU 사용: {torch.xpu.get_device_name(0)}")
        logger.info(f"   XPU 디바이스 수: {torch.xpu.device_count()}")
        return device, 'xpu'
    
    # NVIDIA CUDA 확인
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"🚀 NVIDIA GPU 사용: {torch.cuda.get_device_name(0)}")
        logger.info(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info(f"   GPU 개수: {torch.cuda.device_count()}")
        return device, 'cuda'
    
    # CPU fallback
    else:
        device = torch.device('cpu')
        logger.warning("⚠️  CPU 모드 (GPU를 사용할 수 없습니다)")
        return device, 'cpu'


class LogBERTTrainer:
    """LogBERT 학습 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device, self.device_type = get_device()
        
        # 출력 디렉토리
        self.output_dir = Path(config.get('output_dir', 'checkpoints'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 초기화
        logger.info("모델 초기화 중...")
        self.model = create_logbert_model(config['model'])
        self.model.to(self.device)
        
        # Multi-GPU 지원
        self.use_multi_gpu = False
        if self.device_type == 'cuda' and torch.cuda.device_count() > 1:
            logger.info(f"🔧 Multi-GPU 사용: {torch.cuda.device_count()}개 GPU")
            self.model = torch.nn.DataParallel(self.model)
            self.use_multi_gpu = True

        # Mixed Precision (AMP) 설정
        self.use_amp = config['training'].get('use_amp', True) and self.device_type == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("✅ Mixed Precision (AMP) 활성화")    
        
        # 옵티마이저
        learning_rate = float(config['training']['learning_rate'])
        weight_decay = float(config['training'].get('weight_decay', 0.01))
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Intel GPU 최적화 (IPEX)
        if self.device_type == 'xpu' and IPEX_AVAILABLE:
            logger.info("🔧 Intel GPU 최적화 적용 중...")
            self.model, self.optimizer = ipex.optimize(
                self.model, 
                optimizer=self.optimizer,
                dtype=torch.float32
            )
            logger.info("✅ IPEX 최적화 완료!")
        
        # 학습률 스케줄러
        from torch.optim.lr_scheduler import CosineAnnealingLR
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
        
        logger.info(f"🔄 [Epoch {epoch}] train_epoch 함수 진입 성공")
        
        from tqdm import tqdm
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}",
            total=len(dataloader),
            unit="batch",
            leave=True,
            ncols=100
        )
        
        logger.info(f"⏳ [Epoch {epoch}] 첫 번째 배치를 로드하는 중...")
        
        for i, batch in enumerate(progress_bar):
            if i == 0:
                logger.info(f"✅ [Epoch {epoch}] 첫 번째 배치 로드 완료! GPU 연산 시작")
        
            # 배치를 디바이스로 이동
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 옵티마이저 초기화
            self.optimizer.zero_grad()
            
            # Mixed Precision (AMP) 적용 Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs['loss']
                    
                    # Multi-GPU 사용 시 벡터로 반환된 Loss를 스칼라로 평균화
                    if self.use_multi_gpu:
                        loss = loss.mean()
                
                # 가중치 업데이트 (GradScaler 활용)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer) # Clipping 전 unscale 필수
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training'].get('max_grad_norm', 1.0))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                # 일반 정밀도 학습 (CPU/XPU/기본 CUDA 환경)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs['loss']

                if self.use_multi_gpu:
                    loss = loss.mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training'].get('max_grad_norm', 1.0))
                self.optimizer.step()
            
            # 스케줄러 업데이트 및 통계 기록
            self.scheduler.step()
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 진행 상황 업데이트 및 로깅
            if self.global_step % self.config['training'].get('log_interval', 100) == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                avg_loss_val = total_loss / num_batches

                # 화면에 보이는 tqdm 업데이트
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg': f'{avg_loss_val:.4f}',
                    'lr': f'{current_lr:.2e}',
                })

                # 파일에 기록되는 로거 업데이트 (나중에 분석용)
                logger.info(
                    f"[Step {self.global_step}] Loss={loss.item():.4f}, Avg={avg_loss_val:.4f}, LR={current_lr:.2e}"
                )

            # 체크포인트 저장
            if self.global_step % self.config['training'].get('save_interval', 5000) == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}')
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def save_checkpoint(self, name: str):
        """체크포인트 저장"""
        checkpoint_path = self.checkpoint_dir / f'{name}.pt'
        
        # Multi-GPU 모델 처리
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': self.config,
            'device_type': self.device_type,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 체크포인트 저장: {checkpoint_path}")
    
    def train(self, train_dataloader, num_epochs: int):
        """전체 학습 과정"""
        logger.info("=" * 80)
        logger.info("🚀 LogBERT 학습 시작")
        logger.info("=" * 80)
        logger.info(f"디바이스: {self.device} ({self.device_type.upper()})")
        logger.info(f"총 에폭: {num_epochs}")
        logger.info(f"배치 크기: {self.config['training']['batch_size']}")
        logger.info(f"학습률: {self.config['training']['learning_rate']}")
        logger.info("=" * 80)
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"에폭 {epoch}/{num_epochs} 시작")
            logger.info(f"{'='*80}")
            
            avg_loss = self.train_epoch(train_dataloader, epoch)
            
            logger.info(f"\n에폭 {epoch}/{num_epochs} 완료")
            logger.info(f"  평균 Loss: {avg_loss:.4f}")
            
            # 최고 성능 모델 저장
            if avg_loss < self.best_loss:
                improvement = self.best_loss - avg_loss
                self.best_loss = avg_loss
                self.save_checkpoint('best_model')
                logger.info(f"  ✅ 최고 성능! (개선: {improvement:.4f})")
            
            # 에폭별 체크포인트
            self.save_checkpoint(f'epoch_{epoch}')
        
        logger.info("=" * 80)
        logger.info("✅ 학습 완료!")
        logger.info(f"최고 Loss: {self.best_loss:.4f}")
        logger.info("=" * 80)


def load_config(config_path: str) -> Dict[str, Any]:
    """설정 파일 로드"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 숫자 값 변환
    if 'training' in config:
        for key in ['learning_rate', 'weight_decay', 'min_lr', 'max_grad_norm', 'mask_prob']:
            if key in config['training']:
                config['training'][key] = float(config['training'][key])
        for key in ['batch_size', 'num_epochs', 'total_steps', 'log_interval', 'save_interval', 'num_workers']:
            if key in config['training']:
                config['training'][key] = int(config['training'][key])
    
    if 'model' in config:
        for key in config['model']:
            if isinstance(config['model'][key], (int, float)):
                continue
            try:
                if '.' in str(config['model'][key]):
                    config['model'][key] = float(config['model'][key])
                else:
                    config['model'][key] = int(config['model'][key])
            except (ValueError, TypeError):
                pass
    
    if 'data' in config:
        if 'max_seq_length' in config['data']:
            config['data']['max_seq_length'] = int(config['data']['max_seq_length'])
        if 'limit_files' in config['data'] and config['data']['limit_files'] is not None:
            config['data']['limit_files'] = int(config['data']['limit_files'])
    
    return config


def get_data_files(preprocessed_dir: str, limit_files: int = None) -> list:
    """전처리된 파일 목록 가져오기"""
    script_dir = Path(__file__).parent
    
    if not Path(preprocessed_dir).is_absolute():
        data_dir = (script_dir / preprocessed_dir).resolve()
    else:
        data_dir = Path(preprocessed_dir)
    
    logger.info(f"데이터 디렉토리: {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
    
    files = sorted(data_dir.glob("preprocessed_logs_*.json"))
    logger.info(f"발견된 전체 데이터 파일: {len(files)}개")
    
    if limit_files is not None and limit_files > 0:
        if len(files) > limit_files:
            files = files[-limit_files:]
            logger.info(f"⚙️  limit_files: 최근 {limit_files}개 파일만 사용")
    
    logger.info(f"✅ 사용할 파일 수: {len(files)}개")
    
    return [str(f) for f in files]


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LogBERT 통합 학습 스크립트')
    parser.add_argument('--config', type=str, required=True,
                       help='설정 파일 경로 (예: configs/test_quick.yaml)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='데이터 디렉토리 (기본값: configs 파일 설정 사용)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='출력 디렉토리 (기본값: configs 파일 설정 사용)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='로그 파일 경로 (기본값: logs/train_YYYYMMDD_HHMMSS.log)')
    
    args = parser.parse_args()
    
    # 로그 파일 경로 설정
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        script_dir = Path(__file__).parent
        logs_dir = script_dir / 'logs'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_name = Path(args.config).stem if args.config else 'train'
        log_file = logs_dir / f'train_{config_name}_{timestamp}.log'
    
    # 로깅 초기화
    setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info("LogBERT 통합 학습 스크립트")
    logger.info("=" * 80)
    
    # 설정 로드
    config = load_config(args.config)
    logger.info(f"설정 파일 로드: {args.config}")
    
    # 명령행 인수로 오버라이드
    if args.data_dir:
        config['data']['preprocessed_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # 기본 출력 디렉토리
    if 'output_dir' not in config:
        script_dir = Path(__file__).parent
        config['output_dir'] = str(script_dir / 'checkpoints')
    
    # 데이터 파일
    limit_files = config['data'].get('limit_files')
    data_files = get_data_files(
        config['data']['preprocessed_dir'],
        limit_files=limit_files
    )
    
    if len(data_files) == 0:
        logger.error("❌ 데이터 파일을 찾을 수 없습니다.")
        return
    
    # 데이터셋 생성
    logger.info("\n" + "=" * 80)
    logger.info("데이터셋 생성 중...")
    logger.info("=" * 80)
    
    dataset = LogBERTDataset(
        data_files=data_files,
        max_seq_length=config['data']['max_seq_length'],
        mask_prob=config['training'].get('mask_prob', 0.15),
        vocab_size=config['model']['vocab_size'],
    )
    
    logger.info(f"✅ 총 세션 수: {len(dataset):,}개")
    
    # DataLoader
    num_workers = config['training'].get('num_workers', 4)
    
    # 디바이스에 따라 pin_memory 설정
    _, device_type = get_device()
    pin_memory = (device_type == 'cuda')
    
    dataloader = create_dataloader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,     # 일꾼 유지
        prefetch_factor=4,           # 데이터 미리 가져오기
        collate_fn=collate_fn
    )
    
    logger.info(f"✅ DataLoader 생성 완료 (배치 수: {len(dataloader):,})")
    
    # 학습기 생성
    logger.info("\n" + "=" * 80)
    logger.info("학습기 초기화...")
    logger.info("=" * 80)
    
    trainer = LogBERTTrainer(config)
    
    # 학습 시작
    trainer.train(
        train_dataloader=dataloader,
        num_epochs=config['training']['num_epochs'],
    )
    
    logger.info("\n✅ 모든 학습이 완료되었습니다!")


if __name__ == '__main__':
    main()
