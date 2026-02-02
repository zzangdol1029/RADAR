#!/usr/bin/env python3
"""
DAGMM (Deep Autoencoding Gaussian Mixture Model) 모델
H100 GPU 최적화된 로그 이상 탐지 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DAGMM(nn.Module):
    """
    Embedding을 사용한 DAGMM (Deep Autoencoding Gaussian Mixture Model)
    
    DAGMM은 오토인코더와 GMM을 결합한 이상 탐지 모델입니다.
    - Encoder: 입력을 저차원 잠재 공간으로 인코딩
    - Decoder: 잠재 표현을 원본 차원으로 복원
    - Estimation Network: GMM 구성요소에 대한 소프트 할당 추정
    """
    
    def __init__(
        self,
        num_classes: int,
        window_size: int = 10,
        embedding_dim: int = 32,
        hidden_dims: list = [64, 32],
        latent_dim: int = 2,
        n_gmm: int = 4,
        dropout: float = 0.3,
    ):
        """
        Args:
            num_classes: 이벤트 클래스 수 (어휘 크기)
            window_size: 입력 시퀀스 길이
            embedding_dim: 임베딩 차원
            hidden_dims: 인코더/디코더 은닉층 차원 리스트
            latent_dim: 잠재 공간 차원
            n_gmm: GMM 구성요소 수
            dropout: 드롭아웃 비율
        """
        super(DAGMM, self).__init__()
        
        self.num_classes = num_classes
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.n_gmm = n_gmm
        
        # Embedding Layer
        self.embedding = nn.Embedding(num_classes, embedding_dim, padding_idx=0)
        
        input_dim = window_size * embedding_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout)
            ])
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout)
            ])
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Estimation Network
        self.estimation_dim = latent_dim + 2  # z + reconstruction errors
        self.estimation = nn.Sequential(
            nn.Linear(self.estimation_dim, 10),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(10, n_gmm),
            nn.Softmax(dim=1)
        )
        
        # GMM 파라미터 (학습 가능한 버퍼)
        self.register_buffer('phi', torch.ones(n_gmm) / n_gmm)
        self.register_buffer('mu', torch.zeros(n_gmm, self.estimation_dim))
        self.register_buffer('sigma', torch.eye(self.estimation_dim).unsqueeze(0).repeat(n_gmm, 1, 1))
        
        logger.info(f"DAGMM 모델 초기화 완료")
        logger.info(f"  - Vocab Size: {num_classes}")
        logger.info(f"  - Window Size: {window_size}")
        logger.info(f"  - Embedding Dim: {embedding_dim}")
        logger.info(f"  - Latent Dim: {latent_dim}")
        logger.info(f"  - GMM Components: {n_gmm}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass
        
        Args:
            x: 입력 텐서 (batch, window_size) - 정수 인덱스
        
        Returns:
            z: 잠재 표현
            x_hat: 재구성된 임베딩
            z_c: 확장된 잠재 표현 (z + 재구성 오차)
            gamma: GMM 소프트 할당
            x_flat: 평탄화된 원본 임베딩
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, window_size, embedding_dim)
        x_flat = embedded.view(x.size(0), -1)  # (batch, window_size * embedding_dim)
        
        # Autoencoder
        z = self.encoder(x_flat)
        x_hat = self.decoder(z)
        
        # 재구성 오차
        rec_euc = torch.norm(x_flat - x_hat, p=2, dim=1, keepdim=True) / (torch.norm(x_flat, p=2, dim=1, keepdim=True) + 1e-6)
        rec_cos = (1 - F.cosine_similarity(x_flat, x_hat, dim=1, eps=1e-6)).unsqueeze(1)
        
        # 클리핑 (수치 안정성)
        rec_euc = torch.clamp(rec_euc, 0, 10)
        rec_cos = torch.clamp(rec_cos, 0, 2)
        
        z_c = torch.cat([z, rec_euc, rec_cos], dim=1)
        gamma = self.estimation(z_c)
        
        return z, x_hat, z_c, gamma, x_flat
    
    def compute_gmm_params(
        self,
        z_c: torch.Tensor,
        gamma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        GMM 파라미터 계산
        
        Args:
            z_c: 확장된 잠재 표현
            gamma: 소프트 할당
        
        Returns:
            phi: 혼합 가중치
            mu: 평균
            sigma: 공분산
        """
        N = z_c.shape[0]
        sum_gamma = torch.sum(gamma, dim=0) + 1e-6
        
        phi = sum_gamma / N
        mu = torch.sum(gamma.unsqueeze(-1) * z_c.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        
        z_centered = z_c.unsqueeze(1) - mu.unsqueeze(0)
        z_exp = z_centered.unsqueeze(-1)
        z_exp_T = z_centered.unsqueeze(-2)
        
        sigma = torch.sum(
            gamma.unsqueeze(-1).unsqueeze(-1) * (z_exp @ z_exp_T), dim=0
        ) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        
        return phi, mu, sigma
    
    def compute_energy(
        self,
        z_c: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        에너지 점수 계산 (이상 점수)
        
        Args:
            z_c: 확장된 잠재 표현
            phi: 혼합 가중치 (None이면 저장된 값 사용)
            mu: 평균 (None이면 저장된 값 사용)
            sigma: 공분산 (None이면 저장된 값 사용)
        
        Returns:
            energy: 에너지 점수 (높을수록 이상)
        """
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        
        N = z_c.shape[0]
        eps = 1e-6
        
        z_centered = z_c.unsqueeze(1) - mu.unsqueeze(0)
        
        # 정규화된 공분산
        sigma_reg = sigma + eps * torch.eye(self.estimation_dim, device=sigma.device).unsqueeze(0)
        
        # 역행렬 계산
        try:
            sigma_inv = torch.linalg.inv(sigma_reg)
            log_det = torch.logdet(sigma_reg)
        except:
            sigma_diag = torch.diagonal(sigma_reg, dim1=-2, dim2=-1)
            sigma_inv = torch.diag_embed(1.0 / (sigma_diag + eps))
            log_det = torch.sum(torch.log(sigma_diag + eps), dim=-1)
        
        # Mahalanobis distance
        mahal = torch.zeros(N, self.n_gmm, device=z_c.device)
        for k in range(self.n_gmm):
            diff = z_centered[:, k, :]
            mahal[:, k] = torch.sum(diff @ sigma_inv[k] * diff, dim=1)
        
        # 클리핑
        mahal = torch.clamp(mahal, 0, 1000)
        
        d = self.estimation_dim
        log_probs = -0.5 * (d * np.log(2 * np.pi) + log_det.unsqueeze(0) + mahal)
        
        log_phi = torch.log(phi.unsqueeze(0) + eps)
        log_weighted = log_phi + log_probs
        
        # Clipping before exp
        log_weighted = torch.clamp(log_weighted, -50, 50)
        
        max_log = torch.max(log_weighted, dim=1, keepdim=True)[0]
        log_sum_exp = max_log + torch.log(torch.sum(torch.exp(log_weighted - max_log), dim=1, keepdim=True) + eps)
        
        energy = -log_sum_exp.squeeze(1)
        
        # 최종 클리핑
        energy = torch.clamp(energy, -10, 100)
        
        return energy


class DAGMMLoss(nn.Module):
    """DAGMM 손실 함수"""
    
    def __init__(self, lambda_energy: float = 0.1, lambda_diag: float = 0.005):
        """
        Args:
            lambda_energy: 에너지 손실 가중치
            lambda_diag: 대각 정규화 가중치
        """
        super().__init__()
        self.lambda_energy = lambda_energy
        self.lambda_diag = lambda_diag
    
    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        energy: torch.Tensor,
        sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        손실 계산
        
        Args:
            x: 원본 입력
            x_hat: 재구성된 입력
            energy: 에너지 점수
            sigma: 공분산 행렬
        
        Returns:
            total: 총 손실
            rec: 재구성 손실
            eng: 에너지 손실
            diag: 대각 정규화 손실
        """
        rec = F.mse_loss(x_hat, x)
        eng = torch.mean(energy)
        
        # Diagonal penalty
        diag = torch.sum(sigma ** 2) - torch.sum(torch.diagonal(sigma, dim1=-2, dim2=-1) ** 2)
        
        total = rec + self.lambda_energy * eng + self.lambda_diag * diag
        return total, rec, eng, diag


def create_dagmm_model(config: Dict[str, Any]) -> DAGMM:
    """
    설정에 따라 DAGMM 모델 생성
    
    Args:
        config: 모델 설정 딕셔너리
    
    Returns:
        DAGMM 모델 인스턴스
    """
    model = DAGMM(
        num_classes=config.get('num_classes', 150),
        window_size=config.get('window_size', 10),
        embedding_dim=config.get('embedding_dim', 32),
        hidden_dims=config.get('hidden_dims', [64, 32]),
        latent_dim=config.get('latent_dim', 2),
        n_gmm=config.get('n_gmm', 4),
        dropout=config.get('dropout', 0.3),
    )
    
    return model
