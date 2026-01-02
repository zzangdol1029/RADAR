"""
DeepLog 모델 구현
LSTM 기반 로그 이상 탐지 모델
논문: "DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning"
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
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
        hidden_size: int = 128,
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
        
        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # 출력 레이어 (다음 로그 예측)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"DeepLog 모델 초기화 완료")
        logger.info(f"  - Vocab Size: {vocab_size}")
        logger.info(f"  - Embedding Dim: {embedding_dim}")
        logger.info(f"  - Hidden Size: {hidden_size}")
        logger.info(f"  - Layers: {num_layers}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        DeepLog은 다음 로그를 예측하는 방식으로 학습합니다.
        """
        # 임베딩
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        
        # 다음 로그 예측
        logits = self.fc(lstm_out)
        
        # Loss 계산 (다음 토큰 예측)
        loss = None
        if labels is not None:
            # labels는 다음 토큰 (shifted)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            # 마지막 위치의 예측을 사용
            if labels.dim() == 2:
                # 시퀀스의 각 위치에서 다음 토큰 예측
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = loss_fn(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1)
                )
            else:
                loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
        }
    
    def predict_anomaly_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        이상 점수 계산
        
        각 위치에서 다음 로그를 예측하고, 실제 로그와의 확률 차이를 계산합니다.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits']
            
            # Softmax로 확률 계산
            probs = torch.softmax(logits, dim=-1)
            
            # 다음 토큰의 확률 추출
            batch_size, seq_len = input_ids.shape
            anomaly_scores = []
            
            for i in range(seq_len - 1):
                # 현재 위치에서 다음 토큰 예측
                predicted_probs = probs[:, i, :]
                # 실제 다음 토큰
                actual_token = input_ids[:, i + 1]
                # 실제 토큰의 확률
                token_probs = predicted_probs[torch.arange(batch_size), actual_token]
                # 음의 로그 확률
                scores = -torch.log(token_probs + 1e-10)
                anomaly_scores.append(scores)
            
            # 시퀀스별 평균 이상 점수
            if len(anomaly_scores) > 0:
                seq_scores = torch.stack(anomaly_scores, dim=1).mean(dim=1)
            else:
                seq_scores = torch.zeros(batch_size, device=input_ids.device)
            
            return seq_scores













