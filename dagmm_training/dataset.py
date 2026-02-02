#!/usr/bin/env python3
"""
DAGMM & DeepLog 학습용 데이터셋
전처리된 로그 데이터를 PyTorch Dataset으로 변환
H100 GPU 최적화
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import numpy as np
import zipfile
import glob
import os

logger = logging.getLogger(__name__)


class LogDataset(Dataset):
    """
    DAGMM/DeepLog 학습용 데이터셋
    
    전처리된 JSON 파일에서 세션 데이터를 로드하고,
    슬라이딩 윈도우를 적용하여 학습 데이터를 생성합니다.
    """
    
    def __init__(
        self,
        sequences: List[List[int]],
        labels: Optional[List[int]] = None,
        window_size: int = 10,
    ):
        """
        Args:
            sequences: 이벤트 시퀀스 리스트
            labels: 레이블 리스트 (DeepLog용, None이면 DAGMM용)
            window_size: 윈도우 크기
        """
        self.window_size = window_size
        self.sequences = sequences
        self.labels = labels
        
        logger.info(f"데이터셋 생성: {len(sequences)}개 샘플")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """데이터 샘플 반환"""
        seq = self.sequences[idx]
        
        result = {
            'input_ids': torch.LongTensor(seq),
        }
        
        if self.labels is not None:
            result['labels'] = torch.LongTensor([self.labels[idx]])
        
        return result


def extract_and_load_log_data(
    dataset_path: str,
    zip_file_name: str = 'logFile.zip',
    extract_path: str = './extracted_logs'
) -> List[Dict]:
    """
    압축 파일 해제 및 로그 데이터 로드
    
    Args:
        dataset_path: 데이터셋 경로
        zip_file_name: 압축 파일 이름
        extract_path: 압축 해제 위치
    
    Returns:
        sessions: 로그 세션 리스트
    """
    print("=" * 50)
    print("압축 파일 해제 및 데이터 로드")
    print("=" * 50)
    
    # 1. 압축 파일 경로
    zip_file_path = os.path.join(dataset_path, zip_file_name)
    print(f"압축 파일: {zip_file_path}")
    
    # 2. 압축 해제 디렉토리 생성
    os.makedirs(extract_path, exist_ok=True)
    
    # 3. 압축 해제
    print(f"압축 해제 중... → {extract_path}")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # 4. 해제된 파일 목록 확인
    all_files = glob.glob(os.path.join(extract_path, '**/*'), recursive=True)
    json_files = [f for f in all_files if f.endswith('.json')]
    
    if json_files:
        print("\nJSON 파일 목록:")
        for f in json_files[:5]:
            print(f"  - {f}")
    
    # 5. 데이터 로드
    sessions = []
    
    for json_file in json_files:
        print(f"\n로드 중: {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                sessions.extend(data)
                print(f"  ✅ {len(data)}개 세션 로드")
            else:
                sessions.append(data)
                print(f"  ✅ 1개 세션 로드")
        except Exception as e:
            print(f"  ❌ 에러: {e}")
    
    print(f"\n총 로드된 세션 수: {len(sessions)}")
    
    return sessions


def load_json_files(data_files: List[str]) -> List[Dict]:
    """
    JSON 파일에서 직접 세션 데이터 로드
    
    Args:
        data_files: JSON 파일 경로 리스트
    
    Returns:
        sessions: 로그 세션 리스트
    """
    sessions = []
    
    for file_path in data_files:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"파일을 찾을 수 없습니다: {file_path}")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                sessions.extend(data)
            else:
                sessions.append(data)
                
            logger.info(f"파일 로드: {file_path} ({len(data) if isinstance(data, list) else 1}개 세션)")
        
        except Exception as e:
            logger.error(f"파일 로드 중 오류 발생: {file_path} - {e}")
            continue
    
    logger.info(f"총 {len(sessions)}개 세션 로드 완료")
    return sessions


def prepare_dagmm_data(
    sessions: List[Dict],
    window_size: int = 10,
) -> Tuple[List[List[int]], int, Dict[int, int]]:
    """
    DAGMM 학습용 데이터 준비
    
    Args:
        sessions: 로그 세션 리스트
        window_size: 슬라이딩 윈도우 크기
    
    Returns:
        samples: 시퀀스 샘플 리스트
        num_classes: 이벤트 클래스 수
        event_id_map: 이벤트 ID 매핑
    """
    samples = []
    all_event_ids = set()
    
    # 세션별 통계
    total_sessions = len(sessions)
    valid_sessions = 0
    
    for session in sessions:
        seq = session.get('event_sequence', [])
        
        if len(seq) < window_size:
            # 짧은 시퀀스는 패딩
            padded_seq = seq + [0] * (window_size - len(seq))
            samples.append(padded_seq)
            all_event_ids.update(seq)
            valid_sessions += 1
        else:
            valid_sessions += 1
            all_event_ids.update(seq)
            
            # Sliding Window 생성
            for i in range(len(seq) - window_size + 1):
                window = seq[i:i + window_size]
                samples.append(window)
    
    print(f"\n세션 통계:")
    print(f"  - 전체 세션: {total_sessions}개")
    print(f"  - 유효 세션: {valid_sessions}개")
    
    # 이벤트 ID 재매핑
    event_id_map = {eid: idx + 1 for idx, eid in enumerate(sorted(all_event_ids))}
    event_id_map[0] = 0  # 패딩용
    num_classes = len(event_id_map)
    
    # 재매핑 적용
    samples_remapped = [[event_id_map.get(eid, 0) for eid in seq] for seq in samples]
    
    print(f"\n데이터 통계:")
    print(f"  - 총 샘플 수: {len(samples_remapped):,}개")
    print(f"  - 윈도우 크기: {window_size}")
    print(f"  - 이벤트 클래스 수: {num_classes}")
    
    return samples_remapped, num_classes, event_id_map


def prepare_deeplog_data(
    sessions: List[Dict],
    window_size: int = 10,
    stride: int = 1,
    min_seq_length: int = 5,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[int, int]]:
    """
    DeepLog 학습용 데이터 준비
    
    Args:
        sessions: 로그 세션 리스트
        window_size: 슬라이딩 윈도우 크기
        stride: 윈도우 이동 크기
        min_seq_length: 최소 시퀀스 길이
    
    Returns:
        sequences: 입력 시퀀스 배열
        labels: 레이블 배열
        num_classes: 이벤트 클래스 수
        event_id_map: 이벤트 ID 매핑
    """
    sequences = []
    labels = []
    all_event_ids = set()
    
    # 세션별 통계
    total_sessions = len(sessions)
    valid_sessions = 0
    skipped_sessions = 0
    
    for session in sessions:
        seq = session.get('event_sequence', [])
        
        # 빈 시퀀스 또는 너무 짧은 시퀀스 스킵
        if len(seq) < min_seq_length:
            skipped_sessions += 1
            continue
        
        valid_sessions += 1
        all_event_ids.update(seq)
        
        # 윈도우보다 작거나 같으면 스킵
        if len(seq) <= window_size:
            continue
        
        # Sliding Window 생성
        for i in range(0, len(seq) - window_size, stride):
            window = seq[i:i + window_size]
            target = seq[i + window_size]
            
            sequences.append(window)
            labels.append(target)
    
    print(f"\n세션 통계:")
    print(f"  - 전체 세션: {total_sessions}개")
    print(f"  - 유효 세션: {valid_sessions}개")
    print(f"  - 스킵된 세션: {skipped_sessions}개")
    
    # 이벤트 ID 재매핑
    event_id_map = {eid: idx + 1 for idx, eid in enumerate(sorted(all_event_ids))}
    event_id_map[0] = 0  # 패딩용
    num_classes = len(event_id_map)
    
    # 재매핑 적용
    sequences_remapped = [[event_id_map.get(eid, 0) for eid in seq] for seq in sequences]
    labels_remapped = [event_id_map.get(label, 0) for label in labels]
    
    print(f"\n데이터 통계:")
    print(f"  - 총 샘플 수: {len(sequences):,}개")
    print(f"  - 윈도우 크기: {window_size}")
    print(f"  - 이벤트 클래스 수: {num_classes}")
    if all_event_ids:
        print(f"  - 이벤트 ID 범위: {min(all_event_ids)} ~ {max(all_event_ids)}")
    
    return np.array(sequences_remapped), np.array(labels_remapped), num_classes, event_id_map


def create_dataloader(
    dataset: LogDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    DataLoader 생성 (H100 GPU 최적화)
    
    Args:
        dataset: LogDataset 인스턴스
        batch_size: 배치 크기
        shuffle: 셔플 여부
        num_workers: 데이터 로딩 워커 수
        pin_memory: GPU 전송 최적화
    
    Returns:
        DataLoader 인스턴스
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
