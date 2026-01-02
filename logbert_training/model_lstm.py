"""
LSTM 기반 로그 이상 탐지 모델
시퀀스 패턴에 특화된 RNN 모델
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LogLSTM(nn.Module):
    """
    LSTM 기반 로그 이상 탐지 모델
    
    RNN 아키텍처를 사용하여 로그 시퀀스의 시간적 패턴을 학습합니다.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        """
        Args:
            vocab_size: 어휘 크기
            embedding_dim: 임베딩 차원
            hidden_size: LSTM 은닉층 크기
            num_layers: LSTM 레이어 수
            dropout: 드롭아웃 확률
            bidirectional: 양방향 LSTM 사용 여부
        """
        super(LogLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # 출력 레이어
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, vocab_size)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"LogLSTM 모델 초기화 완료")
        logger.info(f"  - Vocab Size: {vocab_size}")
        logger.info(f"  - Embedding Dim: {embedding_dim}")
        logger.info(f"  - Hidden Size: {hidden_size}")
        logger.info(f"  - Layers: {num_layers}")
        logger.info(f"  - Bidirectional: {bidirectional}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: 입력 토큰 ID [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len]
            labels: 레이블 (MLM용) [batch_size, seq_len]
        
        Returns:
            loss: 손실값 (labels가 제공된 경우)
            logits: 예측 로그its [batch_size, seq_len, vocab_size]
        """
        # 임베딩
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        
        # 출력
        logits = self.fc(lstm_out)
        
        # Loss 계산
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
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
        
        각 위치에서 예측 확률의 음의 로그를 계산합니다.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits']
            
            # Softmax로 확률 계산
            probs = torch.softmax(logits, dim=-1)
            
            # 실제 토큰의 확률 추출
            batch_size, seq_len = input_ids.shape
            token_probs = probs[torch.arange(batch_size).unsqueeze(1), 
                                torch.arange(seq_len).unsqueeze(0), 
                                input_ids]
            
            # 음의 로그 확률
            anomaly_scores = -torch.log(token_probs + 1e-10)
            
            # 패딩 처리
            if attention_mask is not None:
                anomaly_scores = anomaly_scores * attention_mask.float()
            
            # 시퀀스별 평균 이상 점수
            if attention_mask is not None:
                seq_scores = anomaly_scores.sum(dim=1) / attention_mask.sum(dim=1).float()
            else:
                seq_scores = anomaly_scores.mean(dim=1)
            
            return seq_scores

