#!/usr/bin/env python3
"""
DeepLog 모델 성능 평가 스크립트

평가 지표:
1. Top-k Accuracy: 다음 로그 예측이 top-k 안에 있는 비율
2. Precision, Recall, F1-Score: 이상 탐지 성능
3. Anomaly Detection Rate: 이상 탐지율
4. False Positive Rate: 오탐율
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import DeepLog, create_deeplog_model
from dataset import InMemoryLogDataset, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class DeepLogEvaluator:
    """DeepLog 모델 평가 클래스"""
    
    def __init__(
        self,
        model: DeepLog,
        device: torch.device,
        top_k_values: List[int] = [1, 5, 10, 20],
    ):
        """
        Args:
            model: 평가할 DeepLog 모델
            device: 사용할 디바이스
            top_k_values: 평가할 top-k 값들
        """
        self.model = model
        self.device = device
        self.top_k_values = top_k_values
        
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_prediction_accuracy(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        다음 로그 예측 정확도 평가
        
        Args:
            dataloader: 평가 데이터 로더
        
        Returns:
            각 top-k에 대한 정확도
        """
        correct_counts = {k: 0 for k in self.top_k_values}
        total_predictions = 0
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Evaluating Prediction Accuracy", leave=False)
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs['loss']
            if loss is not None:
                total_loss += loss.item()
                num_batches += 1
            
            logits = outputs['logits']
            
            # 각 위치에서 다음 토큰 예측 정확도 계산
            batch_size, seq_len, vocab_size = logits.shape
            
            for i in range(seq_len - 1):
                # 현재 위치의 예측
                current_logits = logits[:, i, :]  # [batch_size, vocab_size]
                
                # 실제 다음 토큰
                actual_next = labels[:, i + 1]  # [batch_size]
                
                # 유효한 레이블만 (padding 제외)
                valid_mask = actual_next != -100
                if not valid_mask.any():
                    continue
                
                valid_logits = current_logits[valid_mask]
                valid_labels = actual_next[valid_mask]
                
                # Top-k 예측
                for k in self.top_k_values:
                    _, top_k_indices = torch.topk(valid_logits, k=min(k, vocab_size), dim=-1)
                    correct = (top_k_indices == valid_labels.unsqueeze(1)).any(dim=1)
                    correct_counts[k] += correct.sum().item()
                
                total_predictions += valid_mask.sum().item()
        
        # 정확도 계산
        accuracies = {}
        for k in self.top_k_values:
            acc = correct_counts[k] / total_predictions if total_predictions > 0 else 0.0
            accuracies[f'top_{k}_accuracy'] = acc
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracies['eval_loss'] = avg_loss
        accuracies['total_predictions'] = total_predictions
        
        return accuracies
    
    @torch.no_grad()
    def evaluate_anomaly_detection(
        self,
        normal_dataloader: DataLoader,
        anomaly_dataloader: Optional[DataLoader] = None,
        top_k: int = 10,
        threshold_percentiles: List[int] = [90, 95, 99],
    ) -> Dict[str, Any]:
        """
        이상 탐지 성능 평가
        
        Args:
            normal_dataloader: 정상 데이터 로더
            anomaly_dataloader: 이상 데이터 로더 (없으면 정상 데이터만 평가)
            top_k: 이상 판단에 사용할 top-k 값
            threshold_percentiles: 임계값 결정에 사용할 백분위수
        
        Returns:
            이상 탐지 성능 지표
        """
        # 정상 데이터의 이상 점수 계산
        normal_scores = []
        
        pbar = tqdm(normal_dataloader, desc="Computing Normal Scores", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            
            # 배치별 이상 점수 계산
            scores = self._compute_anomaly_scores(input_ids, top_k)
            normal_scores.extend(scores.cpu().numpy().tolist())
        
        normal_scores = np.array(normal_scores)
        
        results = {
            'normal_score_stats': {
                'mean': float(np.mean(normal_scores)),
                'std': float(np.std(normal_scores)),
                'min': float(np.min(normal_scores)),
                'max': float(np.max(normal_scores)),
                'median': float(np.median(normal_scores)),
            },
            'thresholds': {},
            'top_k': top_k,
            'num_normal_samples': len(normal_scores),
        }
        
        # 백분위수 기반 임계값 계산
        for percentile in threshold_percentiles:
            threshold = np.percentile(normal_scores, percentile)
            results['thresholds'][f'p{percentile}'] = float(threshold)
        
        # 이상 데이터가 있으면 탐지 성능 평가
        if anomaly_dataloader is not None:
            anomaly_scores = []
            
            pbar = tqdm(anomaly_dataloader, desc="Computing Anomaly Scores", leave=False)
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                scores = self._compute_anomaly_scores(input_ids, top_k)
                anomaly_scores.extend(scores.cpu().numpy().tolist())
            
            anomaly_scores = np.array(anomaly_scores)
            
            results['anomaly_score_stats'] = {
                'mean': float(np.mean(anomaly_scores)),
                'std': float(np.std(anomaly_scores)),
                'min': float(np.min(anomaly_scores)),
                'max': float(np.max(anomaly_scores)),
                'median': float(np.median(anomaly_scores)),
            }
            results['num_anomaly_samples'] = len(anomaly_scores)
            
            # 각 임계값에 대해 성능 지표 계산
            for percentile in threshold_percentiles:
                threshold = results['thresholds'][f'p{percentile}']
                
                # True labels (0: normal, 1: anomaly)
                y_true = np.concatenate([
                    np.zeros(len(normal_scores)),
                    np.ones(len(anomaly_scores))
                ])
                
                # Predicted labels
                y_scores = np.concatenate([normal_scores, anomaly_scores])
                y_pred = (y_scores > threshold).astype(int)
                
                # 성능 지표 계산
                tp = np.sum((y_pred == 1) & (y_true == 1))
                tn = np.sum((y_pred == 0) & (y_true == 0))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                accuracy = (tp + tn) / len(y_true)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                
                results[f'metrics_p{percentile}'] = {
                    'threshold': float(threshold),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'accuracy': float(accuracy),
                    'false_positive_rate': float(fpr),
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                }
        
        return results
    
    def _compute_anomaly_scores(
        self,
        input_ids: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """배치별 이상 점수 계산"""
        batch_size, seq_len = input_ids.shape
        
        outputs = self.model(input_ids)
        logits = outputs['logits']
        
        anomaly_counts = torch.zeros(batch_size, device=self.device)
        valid_counts = torch.zeros(batch_size, device=self.device)
        
        for i in range(seq_len - 1):
            current_logits = logits[:, i, :]
            _, top_k_indices = torch.topk(current_logits, k=top_k, dim=-1)
            
            actual_next = input_ids[:, i + 1].unsqueeze(1)
            
            # 패딩이 아닌 경우만 계산
            valid_mask = actual_next.squeeze() != 0
            
            is_in_top_k = (top_k_indices == actual_next).any(dim=1)
            anomaly_counts += (~is_in_top_k).float() * valid_mask.float()
            valid_counts += valid_mask.float()
        
        # 유효한 예측 수로 정규화
        anomaly_scores = anomaly_counts / (valid_counts + 1e-10)
        
        return anomaly_scores
    
    def generate_report(
        self,
        prediction_results: Dict[str, float],
        anomaly_results: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """평가 결과 리포트 생성"""
        lines = []
        lines.append("=" * 80)
        lines.append("DeepLog 모델 성능 평가 리포트")
        lines.append("=" * 80)
        lines.append(f"평가 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 모델 설정
        if config:
            model_config = config.get('model', {})
            lines.append("[ 모델 설정 ]")
            lines.append(f"  - Vocab Size: {model_config.get('vocab_size', 'N/A')}")
            lines.append(f"  - Hidden Size: {model_config.get('hidden_size', 'N/A')}")
            lines.append(f"  - LSTM Layers: {model_config.get('num_layers', 'N/A')}")
            lines.append(f"  - Embedding Dim: {model_config.get('embedding_dim', 'N/A')}")
            lines.append("")
        
        # 예측 정확도
        lines.append("[ 다음 로그 예측 정확도 ]")
        lines.append(f"  - Evaluation Loss: {prediction_results.get('eval_loss', 0):.4f}")
        lines.append(f"  - Total Predictions: {prediction_results.get('total_predictions', 0):,}")
        for k in self.top_k_values:
            acc = prediction_results.get(f'top_{k}_accuracy', 0)
            lines.append(f"  - Top-{k} Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        lines.append("")
        
        # 이상 탐지 결과
        if anomaly_results:
            lines.append("[ 이상 점수 통계 (정상 데이터) ]")
            stats = anomaly_results.get('normal_score_stats', {})
            lines.append(f"  - 평균: {stats.get('mean', 0):.4f}")
            lines.append(f"  - 표준편차: {stats.get('std', 0):.4f}")
            lines.append(f"  - 최소: {stats.get('min', 0):.4f}")
            lines.append(f"  - 최대: {stats.get('max', 0):.4f}")
            lines.append(f"  - 중앙값: {stats.get('median', 0):.4f}")
            lines.append(f"  - 샘플 수: {anomaly_results.get('num_normal_samples', 0):,}")
            lines.append("")
            
            lines.append("[ 추천 임계값 ]")
            for key, value in anomaly_results.get('thresholds', {}).items():
                lines.append(f"  - {key}: {value:.4f}")
            lines.append("")
            
            # 이상 데이터가 있는 경우
            if 'anomaly_score_stats' in anomaly_results:
                lines.append("[ 이상 점수 통계 (이상 데이터) ]")
                stats = anomaly_results.get('anomaly_score_stats', {})
                lines.append(f"  - 평균: {stats.get('mean', 0):.4f}")
                lines.append(f"  - 표준편차: {stats.get('std', 0):.4f}")
                lines.append(f"  - 샘플 수: {anomaly_results.get('num_anomaly_samples', 0):,}")
                lines.append("")
                
                lines.append("[ 이상 탐지 성능 (임계값별) ]")
                for key, metrics in anomaly_results.items():
                    if key.startswith('metrics_p'):
                        percentile = key.replace('metrics_p', '')
                        lines.append(f"\n  임계값 P{percentile} = {metrics['threshold']:.4f}:")
                        lines.append(f"    - Precision: {metrics['precision']:.4f}")
                        lines.append(f"    - Recall: {metrics['recall']:.4f}")
                        lines.append(f"    - F1 Score: {metrics['f1_score']:.4f}")
                        lines.append(f"    - Accuracy: {metrics['accuracy']:.4f}")
                        lines.append(f"    - False Positive Rate: {metrics['false_positive_rate']:.4f}")
                        lines.append(f"    - TP: {metrics['true_positives']}, TN: {metrics['true_negatives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def load_model(checkpoint_path: str, config: Dict[str, Any], device: torch.device) -> DeepLog:
    """체크포인트에서 모델 로드"""
    model = create_deeplog_model(config.get('model', {}))
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # DataParallel로 저장된 경우 처리
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 'module.' 접두사 제거
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    
    return model


def evaluate_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    data_files: List[str],
    output_dir: str,
    max_samples: int = 50000,
) -> Dict[str, Any]:
    """
    모델 평가 실행
    
    Args:
        checkpoint_path: 체크포인트 경로
        config: 설정 딕셔너리
        data_files: 평가 데이터 파일 리스트
        output_dir: 결과 저장 디렉토리
        max_samples: 평가에 사용할 최대 샘플 수
    
    Returns:
        평가 결과
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"평가 디바이스: {device}")
    
    # 모델 로드
    logger.info(f"모델 로드: {checkpoint_path}")
    model = load_model(checkpoint_path, config, device)
    
    # 평가 데이터 로드
    logger.info("평가 데이터 로드 중...")
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    
    eval_dataset = InMemoryLogDataset(
        data_files=data_files,
        max_seq_length=data_config.get('max_seq_length', 512),
        vocab_size=model_config.get('vocab_size', 10000),
        max_samples=max_samples,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.get('training', {}).get('batch_size', 64),
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    
    logger.info(f"평가 샘플 수: {len(eval_dataset):,}")
    
    # 평가 수행
    evaluator = DeepLogEvaluator(model, device)
    
    # 예측 정확도 평가
    logger.info("예측 정확도 평가 중...")
    prediction_results = evaluator.evaluate_prediction_accuracy(eval_loader)
    
    # 이상 탐지 평가 (정상 데이터만 사용)
    logger.info("이상 탐지 임계값 계산 중...")
    anomaly_results = evaluator.evaluate_anomaly_detection(eval_loader, top_k=10)
    
    # 리포트 생성
    report = evaluator.generate_report(prediction_results, anomaly_results, config)
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON 결과 저장
    results = {
        'timestamp': timestamp,
        'checkpoint': checkpoint_path,
        'prediction_accuracy': prediction_results,
        'anomaly_detection': anomaly_results,
        'config': config,
    }
    
    json_path = os.path.join(output_dir, f'evaluation_results_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"결과 저장: {json_path}")
    
    # 리포트 저장
    report_path = os.path.join(output_dir, f'evaluation_report_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"리포트 저장: {report_path}")
    
    # 콘솔 출력
    print("\n" + report)
    
    return results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='DeepLog 모델 성능 평가')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='평가할 체크포인트 경로')
    parser.add_argument('--config', type=str, default=None,
                       help='설정 파일 경로')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='평가 데이터 디렉토리')
    parser.add_argument('--output-dir', type=str, default='/home/zzangdol/silverw/deeplog',
                       help='결과 저장 디렉토리')
    parser.add_argument('--max-samples', type=int, default=50000,
                       help='평가에 사용할 최대 샘플 수')
    
    args = parser.parse_args()
    
    # 설정 로드
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config_path = Path(__file__).parent / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
    
    # 데이터 파일
    data_dir = args.data_dir or config.get('data', {}).get('preprocessed_dir', '')
    if not data_dir:
        logger.error("데이터 디렉토리를 지정해주세요.")
        return
    
    data_path = Path(data_dir)
    pattern = config.get('data', {}).get('file_pattern', '*.json')
    data_files = sorted(data_path.glob(pattern))
    
    if not data_files:
        data_files = sorted(data_path.glob('*.json'))
    
    if not data_files:
        logger.error(f"데이터 파일을 찾을 수 없습니다: {data_dir}")
        return
    
    # 평가 실행
    evaluate_model(
        checkpoint_path=args.checkpoint,
        config=config,
        data_files=[str(f) for f in data_files],
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )


if __name__ == '__main__':
    main()
