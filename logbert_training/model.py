"""
LogBERT 모델 정의
BERT 기반의 로그 이상 탐지 모델
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertForMaskedLM
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LogBERT(nn.Module):
    """
    LogBERT 모델
    
    BERT 기반의 로그 이상 탐지 모델로, Masked Language Modeling (MLM)을 통해
    정상적인 로그 패턴을 학습합니다.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
    ):
        """
        Args:
            vocab_size: 어휘 크기 (Event ID + Special Tokens)
            hidden_size: 은닉층 크기
            num_hidden_layers: Transformer 레이어 수
            num_attention_heads: 어텐션 헤드 수
            intermediate_size: Feed-forward 네트워크 중간 크기
            max_position_embeddings: 최대 시퀀스 길이
            hidden_dropout_prob: 은닉층 드롭아웃 확률
            attention_probs_dropout_prob: 어텐션 드롭아웃 확률
            layer_norm_eps: Layer Normalization epsilon
            initializer_range: 가중치 초기화 범위
        """
        super(LogBERT, self).__init__()
        
        # BERT 설정
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            initializer_range=initializer_range,
        )
        
        # BERT 모델 (MLM 헤드 포함)
        self.bert = BertForMaskedLM(config)
        
        logger.info(f"LogBERT 모델 초기화 완료")
        logger.info(f"  - Vocab Size: {vocab_size}")
        logger.info(f"  - Hidden Size: {hidden_size}")
        logger.info(f"  - Layers: {num_hidden_layers}")
        logger.info(f"  - Attention Heads: {num_attention_heads}")
        logger.info(f"  - Max Sequence Length: {max_position_embeddings}")
    
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
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
        }
    
    def predict_anomaly_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        이상 점수 계산
        
        각 위치에서 예측 확률의 음의 로그를 계산하여 이상 점수로 사용합니다.
        점수가 높을수록 이상 가능성이 높습니다.
        
        Args:
            input_ids: 입력 토큰 ID [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len]
        
        Returns:
            anomaly_scores: 이상 점수 [batch_size, seq_len]
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
            
            # 음의 로그 확률 (높을수록 이상)
            anomaly_scores = -torch.log(token_probs + 1e-10)
            
            # 패딩 위치는 0으로 설정
            if attention_mask is not None:
                anomaly_scores = anomaly_scores * attention_mask.float()
            
            # 시퀀스별 평균 이상 점수
            if attention_mask is not None:
                seq_scores = anomaly_scores.sum(dim=1) / attention_mask.sum(dim=1).float()
            else:
                seq_scores = anomaly_scores.mean(dim=1)
            
            return seq_scores


def create_logbert_model(config: Dict[str, Any]) -> LogBERT:
    """
    설정에 따라 LogBERT 모델 생성
    
    Args:
        config: 모델 설정 딕셔너리
    
    Returns:
        LogBERT 모델 인스턴스
    """
    model = LogBERT(
        vocab_size=config.get('vocab_size', 10000),
        hidden_size=config.get('hidden_size', 768),
        num_hidden_layers=config.get('num_hidden_layers', 12),
        num_attention_heads=config.get('num_attention_heads', 12),
        intermediate_size=config.get('intermediate_size', 3072),
        max_position_embeddings=config.get('max_position_embeddings', 512),
        hidden_dropout_prob=config.get('hidden_dropout_prob', 0.1),
        attention_probs_dropout_prob=config.get('attention_probs_dropout_prob', 0.1),
    )
    
    return model

