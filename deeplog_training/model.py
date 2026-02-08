#!/usr/bin/env python3
"""
DeepLog 모델 구현
LSTM 기반 로그 이상 탐지 모델
논문: "DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning"
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class DeepLog(nn.Module):
    """
    DeepLog 모델
    
    LSTM을 사용하여 로그 시퀀스의 정상 패턴을 학습하고,
    다음 로그를 예측하여 이상을 탐지합니다.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Args:
            vocab_size: 어휘 크기 (Event ID 수)
            embedding_dim: 임베딩 차원
            hidden_size: LSTM 은닉층 크기
            num_layers: LSTM 레이어 수
            dropout: 드롭아웃 확률
        """
        super(DeepLog, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False,  # DeepLog는 단방향 LSTM 사용
        )
        
        # Layer Normalization (안정적인 학습을 위해)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 출력 레이어 (다음 로그 예측)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 가중치 초기화
        self._init_weights()
        
        logger.info(f"DeepLog 모델 초기화 완료")
        logger.info(f"  - Vocab Size: {vocab_size:,}")
        logger.info(f"  - Embedding Dim: {embedding_dim}")
        logger.info(f"  - Hidden Size: {hidden_size}")
        logger.info(f"  - LSTM Layers: {num_layers}")
        logger.info(f"  - Dropout: {dropout}")
        logger.info(f"  - Total Parameters: {self.count_parameters():,}")
    
    def _init_weights(self):
        """가중치 초기화"""
        # 임베딩 초기화
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        if self.embedding.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.embedding.padding_idx], 0)
        
        # LSTM 가중치 초기화
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                # Forget gate bias를 1로 설정 (long-term dependency 개선)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        # FC 레이어 초기화
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def count_parameters(self) -> int:
        """학습 가능한 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: 입력 토큰 ID [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len]
            labels: 레이블 (다음 토큰 예측용) [batch_size, seq_len]
            hidden: LSTM 히든 스테이트 (옵션)
        
        Returns:
            Dict containing:
                - loss: 손실값 (labels가 주어진 경우)
                - logits: 출력 로짓 [batch_size, seq_len, vocab_size]
                - hidden: 최종 히든 스테이트
        """
        batch_size, seq_len = input_ids.shape
        
        # 임베딩
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # LSTM
        if hidden is None:
            lstm_out, (h_n, c_n) = self.lstm(embedded)
        else:
            lstm_out, (h_n, c_n) = self.lstm(embedded, hidden)
        
        # Layer Normalization
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        # 다음 로그 예측
        logits = self.fc(lstm_out)
        
        # Loss 계산 (다음 토큰 예측)
        loss = None
        if labels is not None:
            # CrossEntropyLoss (padding token 무시)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
            
            if labels.dim() == 2:
                # 시퀀스의 각 위치에서 다음 토큰 예측
                # shift logits and labels (t번째에서 t+1번째 예측)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                
                # Flatten for CrossEntropyLoss
                loss = loss_fn(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1)
                )
            else:
                loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden': (h_n, c_n),
        }
    
    def predict_next_log(
        self,
        input_ids: torch.Tensor,
        top_k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        다음 로그 예측
        
        Args:
            input_ids: 입력 시퀀스 [batch_size, seq_len]
            top_k: 반환할 상위 k개 예측
        
        Returns:
            top_k_indices: 상위 k개 예측 인덱스 [batch_size, top_k]
            top_k_probs: 상위 k개 예측 확률 [batch_size, top_k]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids)
            logits = outputs['logits']
            
            # 마지막 위치의 예측 사용
            last_logits = logits[:, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            
            # Top-k 예측
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            
            return top_k_indices, top_k_probs
    
    def calculate_anomaly_score(
        self,
        input_ids: torch.Tensor,
        top_k: int = 10,
    ) -> torch.Tensor:
        """
        이상 점수 계산
        
        각 위치에서 다음 로그를 예측하고, 실제 로그가 top-k 안에 없으면
        이상으로 간주합니다.
        
        Args:
            input_ids: 입력 시퀀스 [batch_size, seq_len]
            top_k: 정상으로 간주할 top-k 예측 수
        
        Returns:
            anomaly_scores: 각 시퀀스의 이상 점수 [batch_size]
        """
        self.eval()
        batch_size, seq_len = input_ids.shape
        
        with torch.no_grad():
            outputs = self.forward(input_ids)
            logits = outputs['logits']
            
            # 각 위치에서 다음 토큰 예측
            anomaly_counts = torch.zeros(batch_size, device=input_ids.device)
            
            for i in range(seq_len - 1):
                # 현재 위치에서 top-k 예측
                current_logits = logits[:, i, :]
                _, top_k_indices = torch.topk(current_logits, k=top_k, dim=-1)
                
                # 실제 다음 토큰
                actual_next = input_ids[:, i + 1].unsqueeze(1)
                
                # 실제 토큰이 top-k 안에 있는지 확인
                is_in_top_k = (top_k_indices == actual_next).any(dim=1)
                
                # 이상 카운트 (top-k 안에 없으면 1 추가)
                anomaly_counts += (~is_in_top_k).float()
            
            # 시퀀스 길이로 정규화
            anomaly_scores = anomaly_counts / (seq_len - 1)
            
            return anomaly_scores


def create_deeplog_model(config: Dict[str, Any]) -> DeepLog:
    """
    설정 기반 DeepLog 모델 생성
    
    Args:
        config: 모델 설정 딕셔너리
    
    Returns:
        DeepLog 모델 인스턴스
    """
    return DeepLog(
        vocab_size=config.get('vocab_size', 10000),
        embedding_dim=config.get('embedding_dim', 128),
        hidden_size=config.get('hidden_size', 256),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.2),
    )
