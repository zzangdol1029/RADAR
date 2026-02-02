#!/usr/bin/env python3
"""
DeepLog 모델 성능 평가 스크립트
- 정확도, Top-k 정확도
- Precision, Recall, F1
- 혼동 행렬
- 이상 탐지 성능
- 시각화
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import Counter
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import argparse

sys.path.insert(0, str(Path(__file__).parent))
from model_deeplog import DeepLog, create_deeplog_model

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class DeepLogEvaluator:
    """DeepLog 모델 평가 클래스"""
    
    def __init__(self, model: DeepLog, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """
        모든 샘플에 대해 예측 수행
        """
        all_probs = []
        all_preds = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for batch in dataloader:
                sequences = batch[0].to(self.device)
                labels = batch[1].cpu().numpy()
                
                outputs = self.model(sequences)
                logits = outputs['logits']
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1).cpu().numpy()
                
                all_logits.append(logits.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        return {
            'logits': np.vstack(all_logits),
            'probabilities': np.vstack(all_probs),
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
        }
    
    def compute_accuracy_metrics(
        self,
        results: Dict[str, np.ndarray],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, float]:
        """
        정확도 및 Top-k 정확도 계산
        """
        labels = results['labels']
        preds = results['predictions']
        probs = results['probabilities']
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
        }
        
        # Top-k 정확도
        for k in k_values:
            if k <= probs.shape[1]:
                try:
                    top_k_acc = top_k_accuracy_score(labels, probs, k=k)
                    metrics[f'top_{k}_accuracy'] = top_k_acc
                except:
                    # 수동 계산
                    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
                    correct = [1 if label in top_k_pred else 0 
                              for label, top_k_pred in zip(labels, top_k_preds)]
                    metrics[f'top_{k}_accuracy'] = np.mean(correct)
        
        return metrics
    
    def compute_classification_metrics(
        self,
        results: Dict[str, np.ndarray],
        average: str = 'weighted',
    ) -> Dict[str, float]:
        """
        분류 메트릭 계산 (Precision, Recall, F1)
        """
        labels = results['labels']
        preds = results['predictions']
        
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=average, zero_division=0
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    def compute_anomaly_detection_metrics(
        self,
        results: Dict[str, np.ndarray],
        anomaly_labels: np.ndarray,
        k: int = 5,
    ) -> Dict[str, float]:
        """
        이상 탐지 메트릭 계산
        
        예측이 상위 k개에 포함되지 않으면 이상으로 판정
        """
        labels = results['labels']
        probs = results['probabilities']
        
        # Top-k 예측
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        
        # 이상 판정: 실제 레이블이 top-k에 없으면 이상
        predicted_anomaly = np.array([
            0 if label in top_k else 1
            for label, top_k in zip(labels, top_k_preds)
        ])
        
        # 메트릭 계산
        precision, recall, f1, _ = precision_recall_fscore_support(
            anomaly_labels, predicted_anomaly, average='binary', zero_division=0
        )
        
        accuracy = accuracy_score(anomaly_labels, predicted_anomaly)
        
        # 혼동 행렬
        cm = confusion_matrix(anomaly_labels, predicted_anomaly)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'predicted_anomaly': predicted_anomaly,
        }
    
    def analyze_prediction_confidence(
        self,
        results: Dict[str, np.ndarray],
    ) -> Dict[str, any]:
        """
        예측 신뢰도 분석
        """
        probs = results['probabilities']
        labels = results['labels']
        preds = results['predictions']
        
        # 예측 확률
        pred_probs = np.max(probs, axis=1)
        
        # 정답 확률
        true_probs = probs[np.arange(len(labels)), labels]
        
        # 정답/오답별 신뢰도
        correct_mask = preds == labels
        
        return {
            'mean_confidence': np.mean(pred_probs),
            'std_confidence': np.std(pred_probs),
            'mean_true_prob': np.mean(true_probs),
            'correct_confidence': np.mean(pred_probs[correct_mask]) if correct_mask.any() else 0,
            'wrong_confidence': np.mean(pred_probs[~correct_mask]) if (~correct_mask).any() else 0,
        }
    
    def plot_confusion_matrix(
        self,
        results: Dict[str, np.ndarray],
        top_n: int = 20,
        save_path: str = None,
    ):
        """혼동 행렬 시각화 (상위 N개 클래스)"""
        labels = results['labels']
        preds = results['predictions']
        
        # 가장 빈번한 클래스 선택
        label_counts = Counter(labels)
        top_classes = [c for c, _ in label_counts.most_common(top_n)]
        
        # 필터링
        mask = np.isin(labels, top_classes) & np.isin(preds, top_classes)
        filtered_labels = labels[mask]
        filtered_preds = preds[mask]
        
        if len(filtered_labels) == 0:
            print("시각화할 데이터가 없습니다.")
            return
        
        cm = confusion_matrix(filtered_labels, filtered_preds, labels=top_classes)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=top_classes, yticklabels=top_classes)
        plt.xlabel('예측')
        plt.ylabel('실제')
        plt.title(f'혼동 행렬 (상위 {top_n}개 클래스)')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"혼동 행렬 저장: {save_path}")
        
        plt.show()
    
    def plot_confidence_distribution(
        self,
        results: Dict[str, np.ndarray],
        save_path: str = None,
    ):
        """예측 신뢰도 분포 시각화"""
        probs = results['probabilities']
        labels = results['labels']
        preds = results['predictions']
        
        pred_probs = np.max(probs, axis=1)
        correct_mask = preds == labels
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 전체 신뢰도 분포
        axes[0].hist(pred_probs, bins=50, alpha=0.7, color='blue')
        axes[0].axvline(np.mean(pred_probs), color='red', linestyle='--', label=f'평균: {np.mean(pred_probs):.3f}')
        axes[0].set_xlabel('예측 신뢰도')
        axes[0].set_ylabel('빈도')
        axes[0].set_title('전체 예측 신뢰도 분포')
        axes[0].legend()
        
        # 정답/오답별 분포
        if correct_mask.any():
            axes[1].hist(pred_probs[correct_mask], bins=50, alpha=0.7, label='정답', color='green')
        if (~correct_mask).any():
            axes[1].hist(pred_probs[~correct_mask], bins=50, alpha=0.7, label='오답', color='red')
        axes[1].set_xlabel('예측 신뢰도')
        axes[1].set_ylabel('빈도')
        axes[1].set_title('정답/오답별 신뢰도 분포')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_top_k_accuracy(
        self,
        results: Dict[str, np.ndarray],
        k_values: List[int] = [1, 2, 3, 5, 10, 15, 20],
        save_path: str = None,
    ):
        """Top-k 정확도 그래프"""
        labels = results['labels']
        probs = results['probabilities']
        
        accuracies = []
        valid_k = []
        
        for k in k_values:
            if k <= probs.shape[1]:
                top_k_preds = np.argsort(probs, axis=1)[:, -k:]
                correct = [1 if label in top_k_pred else 0 
                          for label, top_k_pred in zip(labels, top_k_preds)]
                accuracies.append(np.mean(correct) * 100)
                valid_k.append(k)
        
        plt.figure(figsize=(10, 6))
        plt.plot(valid_k, accuracies, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('k')
        plt.ylabel('Top-k 정확도 (%)')
        plt.title('DeepLog Top-k 정확도')
        plt.grid(True, alpha=0.3)
        plt.xticks(valid_k)
        
        for k, acc in zip(valid_k, accuracies):
            plt.annotate(f'{acc:.1f}%', (k, acc), textcoords="offset points", 
                        xytext=(0, 10), ha='center')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(
        self,
        results: Dict[str, np.ndarray],
        anomaly_labels: Optional[np.ndarray] = None,
        output_dir: str = None,
    ) -> str:
        """종합 평가 리포트 생성"""
        report = []
        report.append("=" * 60)
        report.append("DeepLog 모델 성능 평가 리포트")
        report.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")
        
        # 기본 통계
        report.append("📊 데이터 통계")
        report.append("-" * 40)
        report.append(f"  총 샘플 수: {len(results['labels']):,}")
        report.append(f"  클래스 수: {results['probabilities'].shape[1]}")
        report.append("")
        
        # 정확도 메트릭
        acc_metrics = self.compute_accuracy_metrics(results)
        report.append("📈 정확도 메트릭")
        report.append("-" * 40)
        report.append(f"  Top-1 정확도: {acc_metrics['accuracy']*100:.2f}%")
        for k in [3, 5, 10]:
            key = f'top_{k}_accuracy'
            if key in acc_metrics:
                report.append(f"  Top-{k} 정확도: {acc_metrics[key]*100:.2f}%")
        report.append("")
        
        # 분류 메트릭
        cls_metrics = self.compute_classification_metrics(results)
        report.append("📈 분류 메트릭 (가중 평균)")
        report.append("-" * 40)
        report.append(f"  Precision: {cls_metrics['precision']:.4f}")
        report.append(f"  Recall: {cls_metrics['recall']:.4f}")
        report.append(f"  F1 Score: {cls_metrics['f1']:.4f}")
        report.append("")
        
        # 신뢰도 분석
        conf_metrics = self.analyze_prediction_confidence(results)
        report.append("🔍 예측 신뢰도 분석")
        report.append("-" * 40)
        report.append(f"  평균 신뢰도: {conf_metrics['mean_confidence']:.4f}")
        report.append(f"  정답 예측 시 신뢰도: {conf_metrics['correct_confidence']:.4f}")
        report.append(f"  오답 예측 시 신뢰도: {conf_metrics['wrong_confidence']:.4f}")
        report.append("")
        
        # 이상 탐지 (레이블이 있는 경우)
        if anomaly_labels is not None:
            report.append("🚨 이상 탐지 성능 (Top-5 기준)")
            report.append("-" * 40)
            
            for k in [5, 10]:
                anom_metrics = self.compute_anomaly_detection_metrics(results, anomaly_labels, k=k)
                report.append(f"  [Top-{k}]")
                report.append(f"    정확도: {anom_metrics['accuracy']*100:.2f}%")
                report.append(f"    Precision: {anom_metrics['precision']:.4f}")
                report.append(f"    Recall: {anom_metrics['recall']:.4f}")
                report.append(f"    F1: {anom_metrics['f1']:.4f}")
            report.append("")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        print(report_text)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, 'deeplog_evaluation_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n리포트 저장: {report_path}")
        
        return report_text


def load_model(checkpoint_path: str) -> DeepLog:
    """체크포인트에서 모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})
    
    model = create_deeplog_model(config.get('model', {}))
    model.load_state_dict(checkpoint['model'])
    
    return model


def main():
    parser = argparse.ArgumentParser(description='DeepLog 모델 성능 평가')
    parser.add_argument('--checkpoint', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--data-dir', type=str, required=True, help='테스트 데이터 디렉토리')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='결과 저장 디렉토리')
    parser.add_argument('--batch-size', type=int, default=64, help='배치 크기')
    args = parser.parse_args()
    
    # 모델 로드
    print("모델 로딩 중...")
    model = load_model(args.checkpoint)
    evaluator = DeepLogEvaluator(model)
    
    # 데이터 로드 (여기서는 예시)
    print("데이터 로딩 중...")
    # 실제 데이터 로드 로직 필요
    
    print("\n평가가 완료되었습니다!")


if __name__ == '__main__':
    main()
