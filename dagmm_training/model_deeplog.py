#!/usr/bin/env python3
"""
DeepLog 모델
H100 GPU 최적화된 로그 이상 탐지 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DeepLog(nn.Module):
    """
    DeepLog 모델
    
    LSTM 기반의 로그 이상 탐지 모델입니다.
    시퀀스의 다음 로그 이벤트를 예측하여, 예측과 다른 이벤트를 이상으로 탐지합니다.
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 32,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Args:
            num_classes: 이벤트 클래스 수 (어휘 크기)
            embedding_dim: 임베딩 차원
            hidden_size: LSTM 은닉층 크기
            num_layers: LSTM 레이어 수
            dropout: 드롭아웃 비율
        """
        super(DeepLog, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Embedding Layer
        self.embedding = nn.Embedding(num_classes, embedding_dim, padding_idx=0)
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output Layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
        logger.info(f"DeepLog 모델 초기화 완료")
        logger.info(f"  - Vocab Size: {num_classes}")
        logger.info(f"  - Embedding Dim: {embedding_dim}")
        logger.info(f"  - Hidden Size: {hidden_size}")
        logger.info(f"  - Num Layers: {num_layers}")
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: 입력 텐서 (batch, seq_len) - 정수 인덱스
            hidden: LSTM 초기 상태 (optional)
        
        Returns:
            logits: 출력 로짓
            hidden: LSTM 최종 상태
        """
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        out, hidden = self.lstm(embedded, hidden)
        
        # Output (마지막 시점만)
        out = self.fc(out[:, -1, :])
        
        return {
            'logits': out,
            'hidden': hidden,
        }
    
    def predict_next_k(
        self,
        x: torch.Tensor,
        k: int = 5,
    ) -> torch.Tensor:
        """
        상위 k개의 다음 이벤트 예측
        
        Args:
            x: 입력 시퀀스
            k: 상위 예측 수
        
        Returns:
            top_k_indices: 상위 k개 예측 인덱스
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            logits = outputs['logits']
            probs = F.softmax(logits, dim=-1)
            _, top_k_indices = probs.topk(k, dim=-1)
        
        return top_k_indices
    
    def compute_anomaly_score(
        self,
        x: torch.Tensor,
        true_labels: torch.Tensor,
        k: int = 5,
    ) -> torch.Tensor:
        """
        이상 점수 계산
        
        실제 다음 이벤트가 예측 상위 k개에 포함되지 않으면 이상으로 간주
        
        Args:
            x: 입력 시퀀스
            true_labels: 실제 다음 이벤트
            k: 상위 예측 수
        
        Returns:
            is_anomaly: 이상 여부 (1: 이상, 0: 정상)
        """
        self.eval()
        with torch.no_grad():
            top_k_preds = self.predict_next_k(x, k)
            
            # 실제 레이블이 상위 k에 포함되는지 확인
            true_labels = true_labels.view(-1, 1)
            is_in_top_k = (top_k_preds == true_labels).any(dim=-1)
            
            # 포함되지 않으면 이상 (1)
            is_anomaly = (~is_in_top_k).float()
        
        return is_anomaly


class DeepLogTrainer:
    """DeepLog 모델 학습기"""
    
    def __init__(
        self,
        model: DeepLog,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader) -> tuple:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in dataloader:
            sequences = batch['input_ids'].to(self.device)
            labels = batch['labels'].squeeze(-1).to(self.device)
            
            outputs = self.model(sequences)
            logits = outputs['logits']
            loss = self.criterion(logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def evaluate(self, dataloader) -> tuple:
        """모델 평가"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                sequences = batch['input_ids'].to(self.device)
                labels = batch['labels'].squeeze(-1).to(self.device)
                
                outputs = self.model(sequences)
                logits = outputs['logits']
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy


def create_deeplog_model(config: Dict[str, Any]) -> DeepLog:
    """
    설정에 따라 DeepLog 모델 생성
    
    Args:
        config: 모델 설정 딕셔너리
    
    Returns:
        DeepLog 모델 인스턴스
    """
    model = DeepLog(
        num_classes=config.get('num_classes', 150),
        embedding_dim=config.get('embedding_dim', 32),
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
    )
    
    return model
