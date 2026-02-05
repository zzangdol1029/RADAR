import json
import torch
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

class LogBERTDataset(Dataset):
    """
    LogBERT 학습용 데이터셋 클래스 (최종 최적화 버전)
    
    1. Lazy Loading: 파일 경로만 인덱싱하여 RAM 점유율 99% 절감
    2. Single File Caching: 동일 파일 접근 시 재로드 방지로 I/O 속도 극대화
    """
    
    # BERT 표준 특수 토큰 ID 정의
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
        self.data_files = [str(f) for f in data_files]
        self.max_seq_length = max_seq_length
        self.mask_prob = mask_prob
        self.random_mask_prob = random_mask_prob
        self.keep_mask_prob = keep_mask_prob
        self.vocab_size = vocab_size
        
        # [캐싱 최적화] 현재 메모리에 로드된 파일 정보를 저장
        self.current_file_path = None
        self.current_data = None
        
        # 실제 세션 데이터 대신 (파일_인덱스, 세션_인덱스) 위치 지도 생성
        self.index_map = []
        self._build_index()
        
        logger.info(f"✅ 데이터셋 준비 완료: {len(self.index_map):,}개 세션 인덱싱됨")

    def _build_index(self):
        """파일별 세션 개수를 파악하여 위치 지도를 만듭니다."""
        logger.info("🔍 데이터 위치 인덱싱 중... (RAM 점유 방지 모드)")
        
        random.shuffle(self.data_files)
        
        for file_idx, file_path in enumerate(self.data_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for session_idx in range(len(data)):
                            self.index_map.append((file_idx, session_idx))
                    del data # 메모리 즉시 반환
            except Exception as e:
                logger.error(f"❌ 파일 인덱싱 오류 ({file_path}): {e}")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """DataLoader가 요청할 때 캐시를 확인하여 데이터를 반환합니다."""
        file_idx, session_idx = self.index_map[idx]
        file_path = self.data_files[file_idx]
        
        try:
            # [수정 핵심] 요청 파일이 현재 캐시된 파일과 다를 때만 새로 로드
            if self.current_file_path != file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.current_data = json.load(f)
                self.current_file_path = file_path

            session = self.current_data[session_idx]
            
            # 토큰 및 마스크 복사
            token_ids = list(session['token_ids'])
            attention_mask = list(session['attention_mask'])
            
            # 1. Truncation
            if len(token_ids) > self.max_seq_length:
                token_ids = token_ids[:self.max_seq_length]
                attention_mask = attention_mask[:self.max_seq_length]
            
            # 2. Padding
            seq_len = len(token_ids)
            if seq_len < self.max_seq_length:
                padding_len = self.max_seq_length - seq_len
                token_ids.extend([self.PAD_TOKEN_ID] * padding_len)
                attention_mask.extend([0] * padding_len)
            
            # 3. Tensor 변환
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            
            # 4. MLM 레이블 생성
            labels = input_ids.clone()
            masked_indices = self._create_masked_lm_predictions(input_ids, attention_mask)
            labels[~masked_indices] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
        except Exception as e:
            logger.error(f"❌ 데이터 로드 오류 (Index {idx}): {e}")
            return self.__getitem__(0)

    def _create_masked_lm_predictions(self, input_ids, attention_mask):
        """BERT 스타일 마스킹 전략"""
        valid_positions = (attention_mask == 1) & \
                         (input_ids != self.CLS_TOKEN_ID) & \
                         (input_ids != self.SEP_TOKEN_ID)
        
        valid_indices = torch.where(valid_positions)[0]
        if len(valid_indices) == 0:
            return torch.zeros_like(input_ids, dtype=torch.bool)
            
        num_to_mask = max(1, int(len(valid_indices) * self.mask_prob))
        masked_indices = random.sample(valid_indices.tolist(), min(num_to_mask, len(valid_indices)))
        
        for idx in masked_indices:
            rand = random.random()
            if rand < 0.8: # [MASK]
                input_ids[idx] = self.MASK_TOKEN_ID
            elif rand < 0.9: # Random token
                input_ids[idx] = random.randint(1, self.vocab_size - 1)
        
        masked_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        masked_mask[masked_indices] = True
        return masked_mask

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }

def create_dataloader(
    dataset: LogBERTDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True, 
    prefetch_factor: int = 4,
    collate_fn: callable = None 
) -> DataLoader:
    """대규모 데이터 로딩 최적화 버전"""
    
    if torch.backends.mps.is_available():
        pin_memory = False

    # num_workers가 0일 때는 persistent_workers 등을 사용할 수 없으므로 예외 처리
    if num_workers == 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers, 
        prefetch_factor=prefetch_factor, 
        collate_fn=collate_fn
    )