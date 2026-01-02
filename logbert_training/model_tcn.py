"""
TCN (Temporal Convolutional Network) 모델
시계열 데이터에 특화된 컨볼루션 모델
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TemporalBlock(nn.Module):
    """TCN의 기본 블록"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d((kernel_size - 1) * dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d((kernel_size - 1) * dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """TCN을 위한 패딩 제거"""
    
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size != 0 else x


class LogTCN(nn.Module):
    """
    TCN 기반 로그 이상 탐지 모델
    
    Temporal Convolutional Network를 사용하여
    로그 시퀀스의 시간적 패턴을 학습합니다.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        num_channels: list = [128, 128, 128],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        """
        Args:
            vocab_size: 어휘 크기
            embedding_dim: 임베딩 차원
            num_channels: 각 레이어의 채널 수 리스트
            kernel_size: 컨볼루션 커널 크기
            dropout: 드롭아웃 확률
        """
        super(LogTCN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # TCN 레이어
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = embedding_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )
        
        self.tcn = nn.Sequential(*layers)
        
        # 출력 레이어
        self.fc = nn.Linear(num_channels[-1], vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"LogTCN 모델 초기화 완료")
        logger.info(f"  - Vocab Size: {vocab_size}")
        logger.info(f"  - Embedding Dim: {embedding_dim}")
        logger.info(f"  - Channels: {num_channels}")
        logger.info(f"  - Kernel Size: {kernel_size}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len]
        """
        # 임베딩
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # TCN은 [batch_size, channels, seq_len] 형식 필요
        embedded = embedded.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        
        # TCN
        tcn_out = self.tcn(embedded)  # [batch_size, channels, seq_len]
        tcn_out = self.dropout(tcn_out)
        
        # 다시 [batch_size, seq_len, channels]로 변환
        tcn_out = tcn_out.transpose(1, 2)  # [batch_size, seq_len, channels]
        
        # 출력
        logits = self.fc(tcn_out)  # [batch_size, seq_len, vocab_size]
        
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
        """이상 점수 계산"""
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













