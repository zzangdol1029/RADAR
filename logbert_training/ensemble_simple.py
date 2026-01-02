#!/usr/bin/env python3
"""
간단한 앙상블 추론 스크립트
개별 모델을 학습한 후 앙상블로 결합
"""

import torch
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleEnsemble:
    """
    간단한 앙상블 클래스
    개별 모델의 체크포인트를 로드하여 앙상블
    """
    
    def __init__(
        self,
        checkpoint_paths: List[str],
        model_types: List[str],
        ensemble_method: str = 'weighted_average',
        weights: Optional[List[float]] = None,
        device: str = None,
    ):
        """
        Args:
            checkpoint_paths: 각 모델의 체크포인트 경로 리스트
            model_types: 각 모델의 타입 리스트 ('bert', 'distilbert', 'roberta', 'lstm')
            ensemble_method: 'weighted_average', 'average', 'max'
            weights: 각 모델의 가중치
            device: 사용할 디바이스
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.ensemble_method = ensemble_method
        self.models = []
        
        # 각 모델 로드
        for checkpoint_path, model_type in zip(checkpoint_paths, model_types):
            model = self._load_model(checkpoint_path, model_type)
            model.to(self.device)
            model.eval()
            self.models.append(model)
            logger.info(f"모델 로드 완료: {checkpoint_path} ({model_type})")
        
        # 가중치 설정
        if weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            assert len(weights) == len(self.models), "가중치 수가 모델 수와 일치해야 합니다"
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        logger.info(f"앙상블 초기화 완료: {len(self.models)}개 모델")
        logger.info(f"앙상블 방법: {ensemble_method}")
        logger.info(f"가중치: {self.weights}")
    
    def _load_model(self, checkpoint_path: str, model_type: str):
        """모델 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if model_type == 'bert':
            from transformers import BertForMaskedLM
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        elif model_type == 'distilbert':
            from transformers import DistilBertForMaskedLM
            model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        elif model_type == 'roberta':
            from transformers import RobertaForMaskedLM
            model = RobertaForMaskedLM.from_pretrained('roberta-base')
        elif model_type == 'lstm':
            from model_lstm import LogLSTM
            config = checkpoint.get('config', {})
            model = LogLSTM(
                vocab_size=config.get('model', {}).get('vocab_size', 10000),
                embedding_dim=config.get('model', {}).get('embedding_dim', 128),
                hidden_size=config.get('model', {}).get('hidden_size', 256),
            )
        elif model_type == 'deeplog':
            from model_deeplog import DeepLog
            config = checkpoint.get('config', {})
            model = DeepLog(
                vocab_size=config.get('model', {}).get('vocab_size', 10000),
                embedding_dim=config.get('model', {}).get('embedding_dim', 128),
                hidden_size=config.get('model', {}).get('hidden_size', 128),
                num_layers=config.get('model', {}).get('num_layers', 2),
            )
        elif model_type == 'tcn':
            from model_tcn import LogTCN
            config = checkpoint.get('config', {})
            model = LogTCN(
                vocab_size=config.get('model', {}).get('vocab_size', 10000),
                embedding_dim=config.get('model', {}).get('embedding_dim', 128),
                num_channels=config.get('model', {}).get('num_channels', [128, 128, 128]),
                kernel_size=config.get('model', {}).get('kernel_size', 3),
            )
        else:
            raise ValueError(f"알 수 없는 모델 타입: {model_type}")
        
        # 가중치 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        
        return model
    
    def predict_anomaly_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """앙상블 이상 점수 계산"""
        scores_list = []
        
        for model in self.models:
            with torch.no_grad():
                # 각 모델의 이상 점수 계산
                if hasattr(model, 'predict_anomaly_score'):
                    scores = model.predict_anomaly_score(input_ids, attention_mask)
                else:
                    # BERT 계열 모델
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # Softmax로 확률 계산
                    probs = torch.softmax(logits, dim=-1)
                    
                    # 실제 토큰의 확률 추출
                    batch_size, seq_len = input_ids.shape
                    token_probs = probs[torch.arange(batch_size).unsqueeze(1), 
                                        torch.arange(seq_len).unsqueeze(0), 
                                        input_ids]
                    
                    # 음의 로그 확률
                    scores = -torch.log(token_probs + 1e-10)
                    
                    # 패딩 처리
                    if attention_mask is not None:
                        scores = scores * attention_mask.float()
                    
                    # 시퀀스별 평균
                    if attention_mask is not None:
                        seq_scores = scores.sum(dim=1) / attention_mask.sum(dim=1).float()
                    else:
                        seq_scores = scores.mean(dim=1)
                    
                    scores = seq_scores
                
                scores_list.append(scores)
        
        # 앙상블 결합
        if self.ensemble_method == 'weighted_average':
            ensemble_scores = sum(w * s for w, s in zip(self.weights, scores_list))
        elif self.ensemble_method == 'average':
            ensemble_scores = sum(scores_list) / len(scores_list)
        elif self.ensemble_method == 'max':
            ensemble_scores = torch.stack(scores_list).max(dim=0)[0]
        else:
            ensemble_scores = sum(scores_list) / len(scores_list)
        
        return ensemble_scores
    
    def predict_batch(
        self,
        sessions: List[Dict[str, Any]],
        batch_size: int = 16,
    ) -> List[float]:
        """배치 단위로 이상 점수 계산"""
        scores = []
        
        for i in tqdm(range(0, len(sessions), batch_size), desc="앙상블 추론 중"):
            batch = sessions[i:i+batch_size]
            
            # 배치 구성
            max_len = max(len(s['token_ids']) for s in batch)
            input_ids_list = []
            attention_mask_list = []
            
            for session in batch:
                token_ids = session['token_ids']
                attention_mask = session['attention_mask']
                
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
            batch_scores = self.predict_anomaly_score(input_ids, attention_mask)
            scores.extend(batch_scores.cpu().tolist())
        
        return scores


def load_preprocessed_data(file_path: str) -> List[Dict[str, Any]]:
    """전처리된 JSON 파일 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='간단한 앙상블 추론')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                       help='체크포인트 파일 경로들 (여러 개)')
    parser.add_argument('--model-types', type=str, nargs='+', required=True,
                       choices=['bert', 'distilbert', 'roberta', 'lstm', 'deeplog', 'tcn'],
                       help='각 모델의 타입')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                       help='각 모델의 가중치 (선택사항)')
    parser.add_argument('--method', type=str, default='weighted_average',
                       choices=['weighted_average', 'average', 'max'],
                       help='앙상블 방법')
    parser.add_argument('--input', type=str, required=True,
                       help='입력 파일 경로')
    parser.add_argument('--output', type=str, default=None,
                       help='출력 파일 경로')
    parser.add_argument('--threshold', type=float, default=None,
                       help='이상 임계값')
    
    args = parser.parse_args()
    
    # 체크포인트와 모델 타입 수 확인
    if len(args.checkpoints) != len(args.model_types):
        logger.error("체크포인트 수와 모델 타입 수가 일치해야 합니다")
        return
    
    # 앙상블 생성
    logger.info("앙상블 모델 로드 중...")
    ensemble = SimpleEnsemble(
        checkpoint_paths=args.checkpoints,
        model_types=args.model_types,
        ensemble_method=args.method,
        weights=args.weights,
    )
    
    # 데이터 로드
    logger.info(f"데이터 로드 중: {args.input}")
    sessions = load_preprocessed_data(args.input)
    logger.info(f"세션 수: {len(sessions)}개")
    
    # 이상 점수 계산
    logger.info("앙상블 이상 점수 계산 중...")
    anomaly_scores = ensemble.predict_batch(sessions, batch_size=16)
    
    # 결과 정리
    results = []
    for i, (session, score) in enumerate(zip(sessions, anomaly_scores)):
        result = {
            'session_id': session.get('session_id', i),
            'anomaly_score': float(score),
            'is_anomaly': score >= args.threshold if args.threshold else None,
            'has_error': session.get('has_error', False),
            'has_warn': session.get('has_warn', False),
            'service_name': session.get('service_name', 'unknown'),
        }
        results.append(result)
    
    # 통계 출력
    scores_array = np.array(anomaly_scores)
    logger.info("=" * 80)
    logger.info("앙상블 추론 결과 통계")
    logger.info("=" * 80)
    logger.info(f"평균 이상 점수: {scores_array.mean():.4f}")
    logger.info(f"표준편차: {scores_array.std():.4f}")
    logger.info(f"최소값: {scores_array.min():.4f}")
    logger.info(f"최대값: {scores_array.max():.4f}")
    
    if args.threshold:
        num_anomalies = sum(1 for r in results if r['is_anomaly'])
        logger.info(f"임계값 ({args.threshold}) 이상: {num_anomalies}개")
    
    # 결과 저장
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"결과 저장: {args.output}")


if __name__ == '__main__':
    main()

