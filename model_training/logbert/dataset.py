"""
LogBERT 학습용 데이터셋
전처리된 로그 데이터를 PyTorch Dataset으로 변환
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class LogBERTDataset(Dataset):
    """
    LogBERT 학습용 데이터셋
    
    전처리된 JSON 파일에서 세션 데이터를 로드하고,
    MLM 학습을 위한 마스킹을 적용합니다.
    """
    
    # Special Tokens
    PAD_TOKEN_ID = 0
    CLS_TOKEN_ID = 101
    SEP_TOKEN_ID = 102
    MASK_TOKEN_ID = 103
    UNK_TOKEN_ID = 100
    
    def __init__(
        self,
        data_files: List[str],
        max_seq_length: int = 512,
        mask_prob: float = 0.15,
        random_mask_prob: float = 0.1,
        keep_mask_prob: float = 0.1,
        vocab_size: int = 10000,
    ):
        """
        Args:
            data_files: 전처리된 JSON 파일 경로 리스트
            max_seq_length: 최대 시퀀스 길이
            mask_prob: 마스킹 확률 (전체 토큰 중)
            random_mask_prob: 랜덤 토큰으로 교체할 확률 (마스킹된 토큰 중)
            keep_mask_prob: 원래 토큰 유지할 확률 (마스킹된 토큰 중)
            vocab_size: 어휘 크기
        """
        self.max_seq_length = max_seq_length
        self.mask_prob = mask_prob
        self.random_mask_prob = random_mask_prob
        self.keep_mask_prob = keep_mask_prob
        self.vocab_size = vocab_size
        
        # 데이터 로드
        self.sessions = []
        self._load_data(data_files)
        
        logger.info(f"데이터셋 로드 완료: {len(self.sessions)}개 세션")
    
    def _load_data(self, data_files: List[str]):
        """전처리된 파일에서 데이터 로드"""
        for file_path in data_files:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning(f"파일을 찾을 수 없습니다: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    logger.warning(f"데이터 형식이 올바르지 않습니다: {file_path}")
                    continue
                
                for session in data:
                    # 필수 필드 확인
                    if 'token_ids' not in session or 'attention_mask' not in session:
                        continue
                    
                    token_ids = session['token_ids']
                    attention_mask = session['attention_mask']
                    
                    # 길이 확인
                    if len(token_ids) != len(attention_mask):
                        continue
                    
                    # 최대 길이 제한
                    if len(token_ids) > self.max_seq_length:
                        token_ids = token_ids[:self.max_seq_length]
                        attention_mask = attention_mask[:self.max_seq_length]
                    
                    self.sessions.append({
                        'token_ids': token_ids,
                        'attention_mask': attention_mask,
                        'event_sequence': session.get('event_sequence', []),
                        'session_id': session.get('session_id', 0),
                    })
            
            except Exception as e:
                logger.error(f"파일 로드 중 오류 발생: {file_path} - {e}")
                continue
    
    def __len__(self) -> int:
        return len(self.sessions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        데이터 샘플 반환
        
        MLM 학습을 위해 토큰을 마스킹합니다.
        """
        session = self.sessions[idx]
        token_ids = session['token_ids'].copy()
        attention_mask = session['attention_mask'].copy()
        
        # 패딩
        seq_len = len(token_ids)
        if seq_len < self.max_seq_length:
            padding_len = self.max_seq_length - seq_len
            token_ids.extend([self.PAD_TOKEN_ID] * padding_len)
            attention_mask.extend([0] * padding_len)
        
        # Tensor로 변환
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # MLM을 위한 마스킹
        labels = input_ids.clone()
        masked_indices = self._create_masked_lm_predictions(input_ids, attention_mask)
        
        # 마스킹된 위치만 레이블로 사용 (나머지는 -100)
        labels[~masked_indices] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    
    def _create_masked_lm_predictions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        MLM을 위한 마스킹 생성
        
        BERT의 마스킹 전략:
        - 80%: [MASK] 토큰으로 교체
        - 10%: 랜덤 토큰으로 교체
        - 10%: 원래 토큰 유지
        """
        # 실제 토큰 위치 (패딩 제외)
        valid_positions = (attention_mask == 1) & (input_ids != self.CLS_TOKEN_ID) & (input_ids != self.SEP_TOKEN_ID)
        
        # 마스킹할 위치 선택
        num_to_mask = int(valid_positions.sum().item() * self.mask_prob)
        if num_to_mask == 0:
            num_to_mask = 1
        
        # 랜덤하게 선택
        valid_indices = torch.where(valid_positions)[0]
        if len(valid_indices) < num_to_mask:
            num_to_mask = len(valid_indices)
        
        masked_indices = torch.tensor(
            random.sample(valid_indices.tolist(), num_to_mask),
            dtype=torch.long
        )
        
        # 마스킹 전략 적용
        for idx in masked_indices:
            rand = random.random()
            if rand < 0.8:
                # 80%: [MASK] 토큰으로 교체
                input_ids[idx] = self.MASK_TOKEN_ID
            elif rand < 0.9:
                # 10%: 랜덤 토큰으로 교체
                input_ids[idx] = random.randint(1, self.vocab_size - 1)
            # 10%: 원래 토큰 유지
        
        # 마스킹된 위치 마스크 생성
        masked_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        masked_mask[masked_indices] = True
        
        return masked_mask


def collate_fn(batch):
    """
    배치 데이터를 묶는 함수 (multiprocessing을 위해 모듈 레벨에 정의)
    
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


def create_dataloader(
    dataset: LogBERTDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    DataLoader 생성
    
    Args:
        dataset: LogBERTDataset 인스턴스
        batch_size: 배치 크기
        shuffle: 셔플 여부
        num_workers: 데이터 로딩 워커 수
        pin_memory: GPU 전송 최적화 (macOS MPS에서는 자동으로 비활성화)
    
    Returns:
        DataLoader 인스턴스
    """
    # macOS MPS에서는 pin_memory가 지원되지 않음
    import torch
    if torch.backends.mps.is_available() and pin_memory:
        pin_memory = False
        logger.warning("pin_memory is not supported on MPS, setting to False")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

