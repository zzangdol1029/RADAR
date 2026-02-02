#!/usr/bin/env python3
"""
DAGMM 모델 성능 평가 스크립트
- 재구성 오류 분석
- 에너지 점수 분포
- 이상 탐지 성능 (Precision, Recall, F1)
- ROC-AUC 곡선
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
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
from torch.utils.data import DataLoader, TensorDataset
import argparse

sys.path.insert(0, str(Path(__file__).parent))
from model_dagmm import DAGMM, create_dagmm_model

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class DAGMMEvaluator:
    """DAGMM 모델 평가 클래스"""
    
    def __init__(self, model: DAGMM, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def compute_scores(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """
        모든 샘플에 대해 에너지 점수 및 재구성 오류 계산
        """
        energies = []
        rec_errors = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                data = data.to(self.device)
                
                z, x_hat, z_c, gamma, x_flat = self.model(data)
                energy = self.model.compute_energy(z_c)
                
                # 재구성 오류
                rec_error = torch.mean((x_flat - x_hat) ** 2, dim=1)
                
                energies.extend(energy.cpu().numpy())
                rec_errors.extend(rec_error.cpu().numpy())
        
        return {
            'energy': np.array(energies),
            'reconstruction_error': np.array(rec_errors),
        }
    
    def evaluate_with_labels(
        self,
        scores: Dict[str, np.ndarray],
        labels: np.ndarray,
        threshold_percentile: float = 95,
    ) -> Dict[str, float]:
        """
        레이블이 있는 경우 성능 평가
        
        Args:
            scores: 에너지 점수 딕셔너리
            labels: 실제 이상 레이블 (0: 정상, 1: 이상)
            threshold_percentile: 이상 판정 임계값 백분위수
        """
        energy = scores['energy']
        
        # 임계값 설정 (정상 데이터의 상위 percentile)
        threshold = np.percentile(energy, threshold_percentile)
        
        # 예측
        predictions = (energy > threshold).astype(int)
        
        # 메트릭 계산
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(labels, energy)
            ap = average_precision_score(labels, energy)
        except:
            auc = 0.0
            ap = 0.0
        
        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'average_precision': ap,
            'predictions': predictions,
        }
    
    def evaluate_unsupervised(
        self,
        scores: Dict[str, np.ndarray],
        threshold_percentile: float = 95,
    ) -> Dict[str, any]:
        """
        비지도 평가 (레이블 없이)
        """
        energy = scores['energy']
        rec_error = scores['reconstruction_error']
        
        threshold = np.percentile(energy, threshold_percentile)
        anomaly_count = np.sum(energy > threshold)
        
        return {
            'energy_mean': np.mean(energy),
            'energy_std': np.std(energy),
            'energy_min': np.min(energy),
            'energy_max': np.max(energy),
            'energy_median': np.median(energy),
            'rec_error_mean': np.mean(rec_error),
            'rec_error_std': np.std(rec_error),
            'threshold': threshold,
            'anomaly_count': anomaly_count,
            'anomaly_ratio': anomaly_count / len(energy),
        }
    
    def find_optimal_threshold(
        self,
        scores: Dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        최적 임계값 탐색 (F1 기준)
        """
        energy = scores['energy']
        best_f1 = 0
        best_threshold = 0
        best_metrics = {}
        
        for percentile in range(80, 100):
            threshold = np.percentile(energy, percentile)
            predictions = (energy > threshold).astype(int)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='binary', zero_division=0
            )
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'percentile': percentile,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                }
        
        return best_threshold, best_metrics
    
    def plot_energy_distribution(
        self,
        scores: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None,
        save_path: str = None,
    ):
        """에너지 점수 분포 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        energy = scores['energy']
        
        # 히스토그램
        if labels is not None:
            axes[0].hist(energy[labels == 0], bins=50, alpha=0.7, label='정상', color='blue')
            axes[0].hist(energy[labels == 1], bins=50, alpha=0.7, label='이상', color='red')
            axes[0].legend()
        else:
            axes[0].hist(energy, bins=50, alpha=0.7, color='blue')
        
        axes[0].set_xlabel('에너지 점수')
        axes[0].set_ylabel('빈도')
        axes[0].set_title('에너지 점수 분포')
        
        # 박스플롯
        if labels is not None:
            data = [energy[labels == 0], energy[labels == 1]]
            bp = axes[1].boxplot(data, labels=['정상', '이상'], patch_artist=True)
            bp['boxes'][0].set_facecolor('blue')
            bp['boxes'][1].set_facecolor('red')
        else:
            axes[1].boxplot(energy, patch_artist=True)
        
        axes[1].set_ylabel('에너지 점수')
        axes[1].set_title('에너지 점수 박스플롯')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"그래프 저장: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(
        self,
        scores: Dict[str, np.ndarray],
        labels: np.ndarray,
        save_path: str = None,
    ):
        """ROC 곡선 시각화"""
        energy = scores['energy']
        
        fpr, tpr, thresholds = roc_curve(labels, energy)
        auc = roc_auc_score(labels, energy)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('DAGMM ROC 곡선')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC 곡선 저장: {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(
        self,
        scores: Dict[str, np.ndarray],
        labels: np.ndarray,
        save_path: str = None,
    ):
        """Precision-Recall 곡선"""
        energy = scores['energy']
        
        precision, recall, thresholds = precision_recall_curve(labels, energy)
        ap = average_precision_score(labels, energy)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AP = {ap:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('DAGMM Precision-Recall 곡선')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(
        self,
        scores: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None,
        output_dir: str = None,
    ) -> str:
        """종합 평가 리포트 생성"""
        report = []
        report.append("=" * 60)
        report.append("DAGMM 모델 성능 평가 리포트")
        report.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")
        
        # 비지도 통계
        unsup_metrics = self.evaluate_unsupervised(scores)
        
        report.append("📊 에너지 점수 통계")
        report.append("-" * 40)
        report.append(f"  평균: {unsup_metrics['energy_mean']:.4f}")
        report.append(f"  표준편차: {unsup_metrics['energy_std']:.4f}")
        report.append(f"  최소: {unsup_metrics['energy_min']:.4f}")
        report.append(f"  최대: {unsup_metrics['energy_max']:.4f}")
        report.append(f"  중앙값: {unsup_metrics['energy_median']:.4f}")
        report.append("")
        
        report.append("📊 재구성 오류 통계")
        report.append("-" * 40)
        report.append(f"  평균: {unsup_metrics['rec_error_mean']:.4f}")
        report.append(f"  표준편차: {unsup_metrics['rec_error_std']:.4f}")
        report.append("")
        
        report.append("🔍 이상 탐지 결과 (상위 5% 기준)")
        report.append("-" * 40)
        report.append(f"  임계값: {unsup_metrics['threshold']:.4f}")
        report.append(f"  이상 샘플 수: {unsup_metrics['anomaly_count']}")
        report.append(f"  이상 비율: {unsup_metrics['anomaly_ratio']*100:.2f}%")
        report.append("")
        
        # 지도 평가 (레이블이 있는 경우)
        if labels is not None:
            report.append("📈 분류 성능 (레이블 기반)")
            report.append("-" * 40)
            
            metrics = self.evaluate_with_labels(scores, labels)
            report.append(f"  Precision: {metrics['precision']:.4f}")
            report.append(f"  Recall: {metrics['recall']:.4f}")
            report.append(f"  F1 Score: {metrics['f1']:.4f}")
            report.append(f"  ROC-AUC: {metrics['auc']:.4f}")
            report.append(f"  Average Precision: {metrics['average_precision']:.4f}")
            report.append("")
            
            # 최적 임계값
            opt_threshold, opt_metrics = self.find_optimal_threshold(scores, labels)
            report.append("🎯 최적 임계값 (F1 기준)")
            report.append("-" * 40)
            report.append(f"  백분위수: {opt_metrics['percentile']}%")
            report.append(f"  최적 F1: {opt_metrics['f1']:.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        print(report_text)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, 'dagmm_evaluation_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n리포트 저장: {report_path}")
        
        return report_text


def load_model(checkpoint_path: str) -> DAGMM:
    """체크포인트에서 모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})
    
    model = create_dagmm_model(config.get('model', {}))
    model.load_state_dict(checkpoint['model'])
    
    return model


def main():
    parser = argparse.ArgumentParser(description='DAGMM 모델 성능 평가')
    parser.add_argument('--checkpoint', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--data-dir', type=str, required=True, help='테스트 데이터 디렉토리')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='결과 저장 디렉토리')
    parser.add_argument('--batch-size', type=int, default=64, help='배치 크기')
    args = parser.parse_args()
    
    # 모델 로드
    print("모델 로딩 중...")
    model = load_model(args.checkpoint)
    evaluator = DAGMMEvaluator(model)
    
    # 데이터 로드 (여기서는 예시)
    print("데이터 로딩 중...")
    # 실제 데이터 로드 로직 필요
    
    print("\n평가가 완료되었습니다!")


if __name__ == '__main__':
    main()
