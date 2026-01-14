#!/usr/bin/env python3
"""
앙상블 이상 탐지 모듈
여러 이상 탐지 모델을 결합하여 더 정확한 이상 탐지 수행
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import sys

# 상위 디렉토리 추가
sys.path.insert(0, str(Path(__file__).parent.parent / 'logbert_training'))

from model import LogBERT
from model_deeplog import DeepLog
from model_lstm import LogLSTM
from model_tcn import LogTCN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleAnomalyDetector:
    """앙상블 이상 탐지 클래스"""
    
    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        ensemble_method: str = 'weighted_average',
        weights: Optional[List[float]] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            model_configs: 모델 설정 리스트
                [
                    {
                        'type': 'logbert',
                        'checkpoint': 'path/to/logbert.pt',
                        'weight': 0.4,
                        'enabled': True
                    },
                    ...
                ]
            ensemble_method: 앙상블 방법
                - 'weighted_average': 가중 평균 (권장)
                - 'average': 단순 평균
                - 'max': 최대값
                - 'min': 최소값
            weights: 가중치 리스트 (None이면 model_configs의 weight 사용)
            device: 디바이스 ('cuda' or 'cpu', None이면 자동)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.ensemble_method = ensemble_method
        self.models = []
        self.model_types = []
        self.model_names = []
        
        # 활성화된 모델만 필터링
        active_configs = [c for c in model_configs if c.get('enabled', True)]
        
        if not active_configs:
            raise ValueError("활성화된 모델이 없습니다")
        
        # 가중치 설정
        if weights:
            self.weights = weights[:len(active_configs)]
        else:
            self.weights = [config.get('weight', 1.0) for config in active_configs]
        
        # 가중치 정규화
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
        else:
            # 가중치가 모두 0이면 균등 분배
            self.weights = [1.0 / len(active_configs)] * len(active_configs)
        
        # 모델 로드
        for config in active_configs:
            try:
                model = self._load_model(config)
                self.models.append(model)
                self.model_types.append(config['type'])
                self.model_names.append(config.get('name', config['type']))
                logger.info(f"모델 로드 완료: {config['type']} ({config.get('name', 'N/A')})")
            except Exception as e:
                logger.error(f"모델 로드 실패 ({config['type']}): {e}")
                # 모델 로드 실패 시 가중치 재조정
                continue
        
        if not self.models:
            raise ValueError("로드된 모델이 없습니다")
        
        # 가중치 재정규화 (로드 실패한 모델 제외)
        if len(self.weights) > len(self.models):
            self.weights = self.weights[:len(self.models)]
            total_weight = sum(self.weights)
            if total_weight > 0:
                self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"앙상블 모델 구성 완료:")
        logger.info(f"  - 모델 수: {len(self.models)}개")
        logger.info(f"  - 모델 타입: {self.model_types}")
        logger.info(f"  - 앙상블 방법: {ensemble_method}")
        logger.info(f"  - 가중치: {self.weights}")
        logger.info(f"  - 디바이스: {self.device}")
    
    def _load_model(self, config: Dict[str, Any]) -> nn.Module:
        """모델 로드"""
        model_type = config['type'].lower()
        checkpoint_path = Path(config['checkpoint'])
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_config = checkpoint.get('config', {})
        
        # 모델 생성
        if model_type == 'logbert':
            model_params = model_config.get('model', {})
            model = LogBERT(**model_params)
        elif model_type == 'deeplog':
            model_params = model_config.get('model', {})
            model = DeepLog(**model_params)
        elif model_type == 'lstm':
            model_params = model_config.get('model', {})
            model = LogLSTM(**model_params)
        elif model_type == 'tcn':
            model_params = model_config.get('model', {})
            model = LogTCN(**model_params)
        else:
            raise ValueError(f"알 수 없는 모델 타입: {model_type}")
        
        # 가중치 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.warning(f"체크포인트에 model_state_dict가 없습니다: {checkpoint_path}")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict_anomaly_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        앙상블 이상 점수 계산
        
        Args:
            input_ids: 입력 토큰 ID [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len]
        
        Returns:
            {
                'ensemble_score': torch.Tensor,  # [batch_size]
                'individual_scores': List[torch.Tensor],  # 각 모델의 점수
                'model_types': List[str],
                'model_names': List[str],
                'ensemble_method': str
            }
        """
        scores_list = []
        
        # 각 모델의 이상 점수 계산
        for model, model_type, model_name in zip(self.models, self.model_types, self.model_names):
            try:
                with torch.no_grad():
                    if hasattr(model, 'predict_anomaly_score'):
                        scores = model.predict_anomaly_score(input_ids, attention_mask)
                    else:
                        # 기본 이상 점수 계산
                        scores = self._calculate_default_score(model, input_ids, attention_mask)
                    
                    scores_list.append(scores)
            except Exception as e:
                logger.error(f"모델 {model_name} ({model_type}) 추론 실패: {e}")
                # 실패한 모델은 0 점수로 처리
                batch_size = input_ids.shape[0]
                scores_list.append(torch.zeros(batch_size, device=self.device))
        
        if not scores_list:
            raise ValueError("모든 모델의 추론이 실패했습니다")
        
        # 앙상블 결합
        ensemble_score = self._combine_scores(scores_list)
        
        return {
            'ensemble_score': ensemble_score,
            'individual_scores': scores_list,
            'model_types': self.model_types,
            'model_names': self.model_names,
            'ensemble_method': self.ensemble_method
        }
    
    def _calculate_default_score(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """기본 이상 점수 계산 (predict_anomaly_score가 없는 경우)"""
        # Forward pass
        if attention_mask is not None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids)
        
        if isinstance(outputs, dict):
            if 'loss' in outputs:
                # Loss를 점수로 사용 (스칼라인 경우)
                loss = outputs['loss']
                if isinstance(loss, torch.Tensor) and loss.dim() == 0:
                    # 배치 크기에 맞춰 확장
                    batch_size = input_ids.shape[0]
                    return loss.expand(batch_size)
                return loss
            elif 'logits' in outputs:
                # Logits에서 확률 계산
                logits = outputs['logits']
                return self._calculate_score_from_logits(logits, input_ids, attention_mask)
        elif isinstance(outputs, torch.Tensor):
            # Tensor 출력인 경우
            if outputs.dim() == 0:
                batch_size = input_ids.shape[0]
                return outputs.expand(batch_size)
            return outputs
        
        raise ValueError("모델 출력에서 점수를 계산할 수 없습니다")
    
    def _calculate_score_from_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Logits에서 이상 점수 계산"""
        # Softmax로 확률 계산
        probs = torch.softmax(logits, dim=-1)
        
        # 실제 토큰의 확률 추출
        batch_size, seq_len = input_ids.shape
        token_probs = probs[torch.arange(batch_size).unsqueeze(1),
                           torch.arange(seq_len).unsqueeze(0),
                           input_ids]
        
        # 음의 로그 확률 (높을수록 이상)
        scores = -torch.log(token_probs + 1e-10)
        
        # 패딩 처리
        if attention_mask is not None:
            scores = scores * attention_mask.float()
            seq_scores = scores.sum(dim=1) / attention_mask.sum(dim=1).float()
        else:
            seq_scores = scores.mean(dim=1)
        
        return seq_scores
    
    def _combine_scores(self, scores_list: List[torch.Tensor]) -> torch.Tensor:
        """점수 결합"""
        if self.ensemble_method == 'weighted_average':
            # 가중 평균
            weighted_scores = [w * s for w, s in zip(self.weights, scores_list)]
            return sum(weighted_scores)
        
        elif self.ensemble_method == 'average':
            # 단순 평균
            return sum(scores_list) / len(scores_list)
        
        elif self.ensemble_method == 'max':
            # 최대값
            stacked = torch.stack(scores_list)
            return stacked.max(dim=0)[0]
        
        elif self.ensemble_method == 'min':
            # 최소값
            stacked = torch.stack(scores_list)
            return stacked.min(dim=0)[0]
        
        else:
            raise ValueError(f"알 수 없는 앙상블 방법: {self.ensemble_method}")
    
    def predict_batch(
        self,
        sessions: List[Dict[str, Any]],
        batch_size: int = 32,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        배치 단위로 앙상블 이상 탐지
        
        Args:
            sessions: 세션 리스트 (각 세션은 'token_ids'와 'attention_mask' 포함)
            batch_size: 배치 크기
            threshold: 이상 임계값
        
        Returns:
            결과 리스트 (각 결과는 세션 정보 + 앙상블 점수 포함)
        """
        results = []
        
        from tqdm import tqdm
        
        for i in tqdm(range(0, len(sessions), batch_size), desc="앙상블 이상 탐지 중"):
            batch = sessions[i:i+batch_size]
            
            # 배치 구성
            max_len = max(len(s['token_ids']) for s in batch)
            max_len = min(max_len, 512)  # 최대 길이 제한
            
            input_ids_list = []
            attention_mask_list = []
            
            for session in batch:
                token_ids = session['token_ids'][:max_len]
                attention_mask = session['attention_mask'][:max_len]
                
                # 패딩
                if len(token_ids) < max_len:
                    padding_len = max_len - len(token_ids)
                    token_ids = token_ids + [0] * padding_len
                    attention_mask = attention_mask + [0] * padding_len
                
                input_ids_list.append(token_ids)
                attention_mask_list.append(attention_mask)
            
            # Tensor로 변환
            input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(self.device)
            attention_mask = torch.tensor(attention_mask_list, dtype=torch.long).to(self.device)
            
            # 앙상블 추론
            ensemble_result = self.predict_anomaly_score(input_ids, attention_mask)
            
            # 결과 정리
            ensemble_scores = ensemble_result['ensemble_score']
            individual_scores = ensemble_result['individual_scores']
            
            for j, session in enumerate(batch):
                # 개별 모델 점수 딕셔너리 생성
                individual_dict = {}
                for model_name, scores in zip(ensemble_result['model_names'], individual_scores):
                    if isinstance(scores, torch.Tensor):
                        individual_dict[model_name] = float(scores[j].item())
                    else:
                        individual_dict[model_name] = float(scores[j])
                
                result = {
                    **session,
                    'ensemble_score': float(ensemble_scores[j].item() if isinstance(ensemble_scores, torch.Tensor) else ensemble_scores[j]),
                    'individual_scores': individual_dict,
                    'model_types': ensemble_result['model_types'],
                    'ensemble_method': ensemble_result['ensemble_method'],
                    'is_anomaly': float(ensemble_scores[j].item() if isinstance(ensemble_scores, torch.Tensor) else ensemble_scores[j]) >= threshold if threshold else None,
                    'threshold': threshold
                }
                results.append(result)
        
        return results


def load_ensemble_from_config(config_path: str) -> EnsembleAnomalyDetector:
    """설정 파일에서 앙상블 모델 로드"""
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    ensemble_config = config.get('ensemble', {})
    
    return EnsembleAnomalyDetector(
        model_configs=ensemble_config.get('models', []),
        ensemble_method=ensemble_config.get('method', 'weighted_average'),
        device=ensemble_config.get('device', None)
    )
