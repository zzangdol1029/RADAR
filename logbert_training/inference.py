#!/usr/bin/env python3
"""
LogBERT 추론 스크립트
학습된 모델을 사용하여 로그 이상 점수 계산
"""

import torch
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from tqdm import tqdm

from model import LogBERT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogBERTInference:
    """LogBERT 추론 클래스"""
    
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
            device: 사용할 디바이스 ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"디바이스: {self.device}")
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # 모델 생성
        self.model = LogBERT(**config['model'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"모델 로드 완료: {checkpoint_path}")
    
    def predict_anomaly_score(
        self,
        token_ids: List[int],
        attention_mask: List[int],
    ) -> float:
        """
        단일 세션의 이상 점수 계산
        
        Args:
            token_ids: 토큰 ID 리스트
            attention_mask: 어텐션 마스크 리스트
        
        Returns:
            anomaly_score: 이상 점수 (높을수록 이상 가능성 높음)
        """
        # Tensor로 변환
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        # 이상 점수 계산
        with torch.no_grad():
            scores = self.model.predict_anomaly_score(input_ids, attention_mask_tensor)
        
        return scores.item()
    
    def predict_batch(
        self,
        sessions: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> List[float]:
        """
        배치 단위로 이상 점수 계산
        
        Args:
            sessions: 세션 리스트 (각 세션은 'token_ids'와 'attention_mask' 포함)
            batch_size: 배치 크기
        
        Returns:
            anomaly_scores: 이상 점수 리스트
        """
        scores = []
        
        for i in tqdm(range(0, len(sessions), batch_size), desc="추론 중"):
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
            
            # 추론
            with torch.no_grad():
                batch_scores = self.model.predict_anomaly_score(input_ids, attention_mask)
            
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
    
    parser = argparse.ArgumentParser(description='LogBERT 추론')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='체크포인트 파일 경로')
    parser.add_argument('--input', type=str, required=True,
                       help='입력 파일 경로 (전처리된 JSON)')
    parser.add_argument('--output', type=str, default=None,
                       help='출력 파일 경로 (JSON)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='이상 임계값 (이 값 이상이면 이상으로 판단)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--device', type=str, default=None,
                       help='디바이스 (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 추론기 생성
    logger.info("모델 로드 중...")
    inference = LogBERTInference(args.checkpoint, args.device)
    
    # 데이터 로드
    logger.info(f"데이터 로드 중: {args.input}")
    sessions = load_preprocessed_data(args.input)
    logger.info(f"세션 수: {len(sessions)}개")
    
    # 이상 점수 계산
    logger.info("이상 점수 계산 중...")
    anomaly_scores = inference.predict_batch(sessions, args.batch_size)
    
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
    logger.info("추론 결과 통계")
    logger.info("=" * 80)
    logger.info(f"평균 이상 점수: {scores_array.mean():.4f}")
    logger.info(f"표준편차: {scores_array.std():.4f}")
    logger.info(f"최소값: {scores_array.min():.4f}")
    logger.info(f"최대값: {scores_array.max():.4f}")
    logger.info(f"중앙값: {np.median(scores_array):.4f}")
    
    if args.threshold:
        num_anomalies = sum(1 for r in results if r['is_anomaly'])
        logger.info(f"임계값 ({args.threshold}) 이상: {num_anomalies}개 ({num_anomalies/len(results)*100:.2f}%)")
    
    # 결과 저장
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"결과 저장: {args.output}")
    else:
        # 상위 10개 이상 세션 출력
        logger.info("\n상위 10개 이상 점수:")
        sorted_results = sorted(results, key=lambda x: x['anomaly_score'], reverse=True)
        for i, result in enumerate(sorted_results[:10], 1):
            logger.info(
                f"{i}. Session {result['session_id']}: "
                f"Score={result['anomaly_score']:.4f}, "
                f"Service={result['service_name']}, "
                f"Error={result['has_error']}, "
                f"Warn={result['has_warn']}"
            )


if __name__ == '__main__':
    main()

