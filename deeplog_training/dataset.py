#!/usr/bin/env python3
"""
DeepLog 학습용 Lazy Loading Dataset
120GB 대용량 데이터를 메모리 효율적으로 처리

핵심 기능:
1. 파일 단위 Lazy Loading - 전체 데이터를 메모리에 로드하지 않음
2. Generator 패턴을 통한 메모리 효율적인 데이터 스트리밍
3. 배치 단위 데이터 로드를 통한 OOM 방지
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Dict, Any, Optional, Iterator, Generator
from pathlib import Path
import logging
import random
import os
import mmap
from collections import deque
import pickle
import time
import ijson  # 스트리밍 JSON 파싱을 위해

logger = logging.getLogger(__name__)


class LazyLogDataset(IterableDataset):
    """
    Lazy Loading 기반 로그 데이터셋
    
    파일을 순차적으로 읽어들여 메모리 사용량을 최소화합니다.
    120GB 데이터도 적은 메모리로 처리 가능합니다.
    """
    
    # Special Tokens
    PAD_TOKEN_ID = 0
    CLS_TOKEN_ID = 101
    SEP_TOKEN_ID = 102
    
    def __init__(
        self,
        data_files: List[str],
        max_seq_length: int = 512,
        buffer_size: int = 10000,
        shuffle_buffer: bool = True,
        vocab_size: int = 10000,
        world_size: int = 1,
        rank: int = 0,
    ):
        """
        Args:
            data_files: 전처리된 JSON 파일 경로 리스트
            max_seq_length: 최대 시퀀스 길이
            buffer_size: 메모리에 유지할 샘플 수 (셔플용)
            shuffle_buffer: 버퍼 내 셔플 여부
            vocab_size: 어휘 크기
            world_size: 분산 학습 시 전체 프로세스 수
            rank: 현재 프로세스 순위
        """
        self.data_files = sorted(data_files)
        self.max_seq_length = max_seq_length
        self.buffer_size = buffer_size
        self.shuffle_buffer = shuffle_buffer
        self.vocab_size = vocab_size
        self.world_size = world_size
        self.rank = rank
        
        # 파일별 샘플 수 캐싱 (전체 크기 추정용)
        self._file_sample_counts = {}
        self._total_samples = None
        
        logger.info(f"LazyLogDataset 초기화")
        logger.info(f"  - 데이터 파일 수: {len(data_files)}")
        logger.info(f"  - 최대 시퀀스 길이: {max_seq_length}")
        logger.info(f"  - 버퍼 크기: {buffer_size:,}")
        logger.info(f"  - 버퍼 셔플: {shuffle_buffer}")
    
    def _count_samples_in_file(self, file_path: str) -> int:
        """파일 내 샘플 수 카운트 (빠른 파일 크기 기반 추정)"""
        if file_path in self._file_sample_counts:
            return self._file_sample_counts[file_path]
        
        # 캐시 파일 경로
        cache_file = f"{file_path}.sample_count.cache"
        
        # 1. 캐시가 있고 최신이면 사용
        if os.path.exists(cache_file):
            try:
                file_mtime = os.path.getmtime(file_path)
                cache_mtime = os.path.getmtime(cache_file)
                
                if cache_mtime >= file_mtime:
                    with open(cache_file, 'r') as f:
                        count = int(f.read().strip())
                        self._file_sample_counts[file_path] = count
                        logger.debug(f"캐시에서 샘플 수 로드: {file_path} → {count:,}")
                        return count
            except Exception as e:
                logger.debug(f"캐시 로드 실패: {e}")
        
        # 2. 파일 크기 기반 빠른 추정 (기본 전략)
        file_size = os.path.getsize(file_path)
        
        # 평균 샘플 크기 추정 (JSON 포맷 기준)
        # 실제 프로젝트에서 측정한 값으로 보정 가능
        avg_sample_size = 800  # bytes (보수적 추정)
        
        count = max(1, file_size // avg_sample_size)
        
        logger.debug(f"파일 크기 기반 추정: {file_path} → {count:,} 샘플 ({file_size:,} bytes)")
        
        # 3. 캐시 저장 (비동기적으로 나중에 저장 가능)
        try:
            with open(cache_file, 'w') as f:
                f.write(str(count))
        except Exception as e:
            logger.debug(f"캐시 저장 실패: {e}")
        
        self._file_sample_counts[file_path] = count
        return count
    
    def get_total_samples(self, force_estimate: bool = True) -> int:
        """전체 샘플 수 반환 (추정치)
        
        Args:
            force_estimate: True면 정확한 카운트 대신 빠른 추정치 사용
        """
        if self._total_samples is None:
            start_time = time.time()
            total = 0
            
            logger.info(f"샘플 수 추정 중... ({len(self.data_files)} 파일)")
            
            for i, file_path in enumerate(self.data_files):
                total += self._count_samples_in_file(file_path)
                
                # 진행 상황 로그 (파일이 많을 때)
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"  진행: {i+1}/{len(self.data_files)} 파일 ({elapsed:.1f}s)")
            
            self._total_samples = total
            elapsed = time.time() - start_time
            logger.info(f"전체 샘플 수 (추정): {self._total_samples:,} (소요 시간: {elapsed:.2f}s)")
        
        return self._total_samples
    
    def _stream_samples_from_file(self, file_path: str) -> Generator[Dict, None, None]:
        """파일에서 샘플을 스트리밍으로 읽기"""
        try:
            with open(file_path, 'rb') as f:
                # ijson을 사용한 메모리 효율적인 JSON 스트리밍
                for session in ijson.items(f, 'item'):
                    # 필수 필드 확인
                    if 'token_ids' not in session:
                        continue
                    
                    yield session
                    
        except Exception as e:
            logger.error(f"파일 스트리밍 오류: {file_path} - {e}")
    
    def _process_sample(self, session: Dict) -> Dict[str, torch.Tensor]:
        """샘플을 텐서로 변환"""
        token_ids = session.get('token_ids', [])
        attention_mask = session.get('attention_mask', [1] * len(token_ids))
        
        # 길이 확인 및 조정
        if len(token_ids) != len(attention_mask):
            attention_mask = [1] * len(token_ids)
        
        # 최대 길이 제한
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
        
        # 패딩
        seq_len = len(token_ids)
        if seq_len < self.max_seq_length:
            padding_len = self.max_seq_length - seq_len
            token_ids = token_ids + [self.PAD_TOKEN_ID] * padding_len
            attention_mask = attention_mask + [0] * padding_len
        
        # vocab_size 범위 제한
        token_ids = [min(tid, self.vocab_size - 1) for tid in token_ids]
        
        # 텐서 변환
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # labels는 input_ids와 동일 (다음 토큰 예측 학습)
        labels = input_ids.clone()
        # 패딩 위치는 -100으로 설정 (loss에서 무시)
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """데이터 이터레이터"""
        worker_info = torch.utils.data.get_worker_info()
        
        # 멀티 워커 환경에서 파일 분배
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0
        
        # 분산 학습 + 멀티 워커 고려
        total_workers = self.world_size * num_workers
        global_worker_id = self.rank * num_workers + worker_id
        
        # 파일을 워커들에게 분배
        files_for_worker = [
            f for i, f in enumerate(self.data_files)
            if i % total_workers == global_worker_id
        ]
        
        # 셔플 버퍼
        buffer = deque(maxlen=self.buffer_size)
        
        for file_path in files_for_worker:
            for session in self._stream_samples_from_file(file_path):
                sample = self._process_sample(session)
                
                if self.shuffle_buffer:
                    buffer.append(sample)
                    
                    # 버퍼가 가득 차면 랜덤하게 하나 꺼내서 yield
                    if len(buffer) >= self.buffer_size:
                        idx = random.randint(0, len(buffer) - 1)
                        yield buffer[idx]
                        del buffer[idx]
                else:
                    yield sample
        
        # 버퍼에 남은 샘플 처리
        if self.shuffle_buffer:
            indices = list(range(len(buffer)))
            random.shuffle(indices)
            for idx in indices:
                yield buffer[idx]


class StreamingLogDataset(IterableDataset):
    """
    더욱 메모리 효율적인 스트리밍 데이터셋
    
    파일을 한 줄씩 읽으며 배치를 생성합니다.
    ijson이 없는 환경을 위한 대안입니다.
    """
    
    PAD_TOKEN_ID = 0
    
    def __init__(
        self,
        data_files: List[str],
        max_seq_length: int = 512,
        vocab_size: int = 10000,
        samples_per_file: int = -1,  # -1 means all
    ):
        self.data_files = sorted(data_files)
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.samples_per_file = samples_per_file
        
        logger.info(f"StreamingLogDataset 초기화: {len(data_files)} 파일")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            files_per_worker = len(self.data_files) // worker_info.num_workers
            start_idx = worker_info.id * files_per_worker
            end_idx = start_idx + files_per_worker
            if worker_info.id == worker_info.num_workers - 1:
                end_idx = len(self.data_files)
            files_to_process = self.data_files[start_idx:end_idx]
        else:
            files_to_process = self.data_files
        
        for file_path in files_to_process:
            try:
                yield from self._process_file(file_path)
            except Exception as e:
                logger.error(f"파일 처리 오류: {file_path} - {e}")
                continue
    
    def _process_file(self, file_path: str) -> Generator[Dict[str, torch.Tensor], None, None]:
        """파일에서 샘플 생성"""
        count = 0
        
        try:
            # 파일을 통째로 읽지 않고 스트리밍으로 처리
            with open(file_path, 'rb') as f:
                try:
                    import ijson
                    parser = ijson.items(f, 'item')
                except ImportError:
                    # ijson이 없으면 일반 json 사용 (메모리 주의)
                    logger.warning("ijson 미설치, 일반 JSON 로드 사용 (메모리 주의)")
                    f.seek(0)
                    import json
                    data = json.load(f)
                    parser = iter(data)
                
                for session in parser:
                    if self.samples_per_file > 0 and count >= self.samples_per_file:
                        break
                    
                    token_ids = session.get('token_ids', [])
                    if not token_ids:
                        continue
                    
                    attention_mask = session.get('attention_mask', [1] * len(token_ids))
                    
                    # 길이 조정
                    if len(token_ids) > self.max_seq_length:
                        token_ids = token_ids[:self.max_seq_length]
                        attention_mask = attention_mask[:self.max_seq_length]
                    elif len(token_ids) < self.max_seq_length:
                        pad_len = self.max_seq_length - len(token_ids)
                        token_ids = token_ids + [self.PAD_TOKEN_ID] * pad_len
                        attention_mask = attention_mask + [0] * pad_len
                    
                    # Vocab 범위 제한
                    token_ids = [min(t, self.vocab_size - 1) for t in token_ids]
                    
                    input_ids = torch.tensor(token_ids, dtype=torch.long)
                    attn_mask = torch.tensor(attention_mask, dtype=torch.long)
                    labels = input_ids.clone()
                    labels[attn_mask == 0] = -100
                    
                    yield {
                        'input_ids': input_ids,
                        'attention_mask': attn_mask,
                        'labels': labels,
                    }
                    count += 1
                    
        except Exception as e:
            logger.error(f"파일 처리 중 오류: {file_path} - {e}")


class InMemoryLogDataset(Dataset):
    """
    메모리 내 데이터셋 (작은 데이터셋 또는 검증용)
    
    전체 데이터를 메모리에 로드합니다.
    검증 데이터셋처럼 작은 데이터에 적합합니다.
    """
    
    PAD_TOKEN_ID = 0
    
    def __init__(
        self,
        data_files: List[str],
        max_seq_length: int = 512,
        vocab_size: int = 10000,
        max_samples: Optional[int] = None,
    ):
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.samples = []
        
        logger.info(f"InMemoryLogDataset 로드 중...")
        self._load_data(data_files, max_samples)
        logger.info(f"로드 완료: {len(self.samples):,} 샘플")
    
    def _load_data(self, data_files: List[str], max_samples: Optional[int] = None):
        """데이터 로드"""
        total_loaded = 0
        
        for file_path in data_files:
            if max_samples and total_loaded >= max_samples:
                break
            
            try:
                with open(file_path, 'rb') as f:
                    try:
                        import ijson
                        for session in ijson.items(f, 'item'):
                            if max_samples and total_loaded >= max_samples:
                                break
                            
                            token_ids = session.get('token_ids', [])
                            if token_ids:
                                self.samples.append(session)
                                total_loaded += 1
                    except ImportError:
                        f.seek(0)
                        import json
                        data = json.load(f)
                        for session in data:
                            if max_samples and total_loaded >= max_samples:
                                break
                            token_ids = session.get('token_ids', [])
                            if token_ids:
                                self.samples.append(session)
                                total_loaded += 1
            except Exception as e:
                logger.error(f"파일 로드 오류: {file_path} - {e}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        session = self.samples[idx]
        
        token_ids = session.get('token_ids', [])
        attention_mask = session.get('attention_mask', [1] * len(token_ids))
        
        # 길이 조정
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
        elif len(token_ids) < self.max_seq_length:
            pad_len = self.max_seq_length - len(token_ids)
            token_ids = token_ids + [self.PAD_TOKEN_ID] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        
        # Vocab 범위 제한
        token_ids = [min(t, self.vocab_size - 1) for t in token_ids]
        
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attn_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'labels': labels,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    배치 데이터 결합 함수
    
    Args:
        batch: 배치 샘플 리스트
    
    Returns:
        배치 딕셔너리
    """
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }


def create_dataloaders(
    data_files: List[str],
    config: Dict[str, Any],
    validation_split: float = 0.1,
) -> tuple:
    """
    학습/검증 DataLoader 생성
    
    Args:
        data_files: 데이터 파일 리스트
        config: 설정 딕셔너리
        validation_split: 검증 데이터 비율
    
    Returns:
        (train_dataloader, val_dataloader) 튜플
    """
    # 설정 추출
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    max_seq_length = data_config.get('max_seq_length', 512)
    vocab_size = config.get('model', {}).get('vocab_size', 10000)
    batch_size = training_config.get('batch_size', 64)
    num_workers = training_config.get('num_workers', 4)
    
    lazy_config = data_config.get('lazy_loading', {})
    buffer_size = lazy_config.get('buffer_size', 10000)
    shuffle_buffer = lazy_config.get('shuffle_buffer', True)
    
    # 파일을 학습/검증으로 분리
    random.shuffle(data_files)
    val_count = max(1, int(len(data_files) * validation_split))
    val_files = data_files[:val_count]
    train_files = data_files[val_count:]
    
    logger.info(f"학습 파일: {len(train_files)}, 검증 파일: {len(val_files)}")
    
    # 학습 데이터셋 (Lazy Loading)
    train_dataset = LazyLogDataset(
        data_files=train_files,
        max_seq_length=max_seq_length,
        buffer_size=buffer_size,
        shuffle_buffer=shuffle_buffer,
        vocab_size=vocab_size,
    )
    
    # 검증 데이터셋 (메모리 로드 - 검증 데이터는 보통 작음)
    val_dataset = InMemoryLogDataset(
        data_files=val_files,
        max_seq_length=max_seq_length,
        vocab_size=vocab_size,
        max_samples=50000,  # 검증용 최대 샘플 수 제한
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=training_config.get('prefetch_factor', 2) if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader
