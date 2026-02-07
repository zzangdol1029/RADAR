#!/usr/bin/env python3
"""
LogBERT 모델 성능 평가 스크립트
정확도, 정밀도, 재현율, F1-Score 계산

사용법:
python evaluate.py \
    --checkpoint checkpoints_test/checkpoints/best_model.pt \
    --config configs/test_quick.yaml \
    --validation-data /home/zzangdol/RADAR/preprocessing/output/preprocessed_logs_2025-02-24.json \
    --normal-ratio 0.8
"""

import os
import sys
import json
import yaml
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

import platform

# 한글 폰트 설정 (Windows 기준)
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# 로컬 모듈 import
from model import create_logbert_model
from dataset import LogBERTDataset

logger = logging.getLogger(__name__)


def setup_logging(log_file: Path = None):
    """로깅 설정 - UTF-8 인코딩 지원"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    root_logger.addHandler(console_handler)
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)
        logger.info(f"📝 로그 파일: {log_file}")
    
    return root_logger


def load_config(config_path: str) -> Dict[str, Any]:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 숫자 값 변환
    if 'model' in config:
        for key in config['model']:
            if isinstance(config['model'][key], str):
                try:
                    config['model'][key] = int(config['model'][key]) if '.' not in config['model'][key] else float(config['model'][key])
                except ValueError:
                    pass
    
    if 'data' in config:
        if 'max_seq_length' in config['data']:
            config['data']['max_seq_length'] = int(config['data']['max_seq_length'])
    
    return config


def load_model(checkpoint_path: str, config: Dict[str, Any], device: torch.device):
    """학습된 모델 로드"""
    logger.info(f"모델 로딩 중: {checkpoint_path}")
    
    # 모델 생성
    model = create_logbert_model(config['model'])
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # state_dict 로드 (strict=False로 설정하여 position_ids 등 불일치 허용)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info(f"체크포인트 정보:")
        logger.info(f"  Global Step: {checkpoint.get('global_step', 'N/A')}")
        logger.info(f"  Best Loss: {checkpoint.get('best_loss', 'N/A'):.4f}")
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    logger.info("✅ 모델 로드 완료")
    return model


def calculate_anomaly_scores(
    model: torch.nn.Module,
    sessions: List[Dict],
    device: torch.device,
    max_seq_length: int,
    vocab_size: int,
    batch_size: int = 32
) -> List[float]:
    model.eval()
    anomaly_scores = []
    
    # 개별 토큰별 손실을 구하기 위한 함수 (평균내지 않음)
    # 0번(PAD)은 계산에서 제외(ignore_index=0)
    criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)

    for i in tqdm(range(0, len(sessions), batch_size), desc="점수 계산 중"):
        batch_sessions = sessions[i : i + batch_size]
        
        batch_input_ids = []
        for s in batch_sessions:
            tokens = s.get('token_ids', [])[:max_seq_length]
            tokens += [0] * (max_seq_length - len(tokens)) 
            batch_input_ids.append(tokens)
            
        input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(device)
        attention_mask = (input_ids != 0).long().to(device)
        labels = input_ids.clone()

        with torch.no_grad():
            # 모델로부터 로짓(Logits)을 직접 가져옴
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits'] # [Batch, Seq, Vocab]

            # CrossEntropy 계산을 위해 차원 변경
            # logits: [Batch * Seq, Vocab], labels: [Batch * Seq]
            flat_logits = logits.view(-1, vocab_size)
            flat_labels = labels.view(-1)
            
            # 1. 모든 토큰의 개별 Loss 계산
            token_losses = criterion(flat_logits, flat_labels) 
            token_losses = token_losses.view(input_ids.size(0), -1) # [Batch, Seq]
            
            # 2. 세션별 평균 Loss 계산 (패딩 제외)
            # 실제 토큰 개수로 나누어 정확한 세션별 점수 산출
            actual_counts = attention_mask.sum(dim=1)
            session_losses = token_losses.sum(dim=1) / actual_counts
            
            anomaly_scores.extend(session_losses.cpu().numpy().tolist())
                
    return anomaly_scores

def load_validation_data(data_path: str, normal_ratio: float = 0.8, max_samples: int = None) -> Tuple[List[Dict], List[Dict]]:
    data_path = Path(data_path)
    sessions = []
    
    if data_path.is_dir():
        logger.info(f"📁 검증 데이터 디렉토리 로드 중: {data_path}")
        files = sorted(data_path.glob("preprocessed_logs_*.json"))
        if not files:
            logger.warning(f"❌ 해당 디렉토리에 JSON 파일이 없습니다: {data_path}")
            return [], []
            
        logger.info(f"   발견된 파일 수: {len(files)}개")
        
        for file_path in files:
            # 목표 샘플 수를 이미 다 채웠다면 더 이상 파일을 열지 않음
            if max_samples is not None and len(sessions) >= max_samples:
                break
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    new_sessions = data if isinstance(data, list) else data.get('sessions', [])
                    sessions.extend(new_sessions)
                    # 파일 하나 읽을 때마다 현재 수집 현황 로그 출력
                    logger.info(f"   로드 중... 현재 {len(sessions):,}개 수집됨 ({file_path.name})")
            except Exception as e:
                logger.error(f"❌ 파일 로드 실패 ({file_path.name}): {e}")
                
    elif data_path.is_file():
        # (파일 하나일 때는 기존과 동일)
        logger.info(f"📄 검증 데이터 파일 로드 중: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            sessions = data if isinstance(data, list) else data.get('sessions', [])
            
    else:
        logger.error(f"❌ 경로를 찾을 수 없습니다: {data_path}")
        return [], []

    # 수집된 데이터가 너무 많으면 max_samples에 맞춰 자르기
    if max_samples is not None and len(sessions) > max_samples:
        import random
        random.seed(42)
        sessions = random.sample(sessions, max_samples)
        logger.info(f"⚡ 샘플링 완료: {max_samples}개 선택됨")

    total_sessions = len(sessions)
    if total_sessions == 0:
        return [], []

    # 정상/이상 분리
    split_idx = int(total_sessions * normal_ratio)
    normal_sessions = sessions[:split_idx]
    anomaly_sessions = sessions[split_idx:]
    
    logger.info(f"총 세션 수: {total_sessions}")
    logger.info(f"정상 세션: {len(normal_sessions)} ({len(normal_sessions)/total_sessions*100:.1f}%)")
    logger.info(f"이상 세션: {len(anomaly_sessions)} ({len(anomaly_sessions)/total_sessions*100:.1f}%)")
    
    return normal_sessions, anomaly_sessions


def generate_pseudo_anomalies(
    sessions: List[Dict],
    vocab_size: int,
    max_seq_len: int,
    ratio: float = 0.1,  # 10% 토큰 변조
    num_anomalies: int = None
) -> List[Dict]:
    """정상 세션을 변조하여 가짜 이상 데이터 생성"""
    import random
    import copy
    
    if num_anomalies is None:
        num_anomalies = len(sessions)
    
    # 원본 데이터에서 샘플링 (복원 추출 허용)
    sampled_sessions = random.choices(sessions, k=num_anomalies)
    anomaly_sessions = copy.deepcopy(sampled_sessions)
    
    for session in anomaly_sessions:
        tokens = session.get('token_ids', session.get('tokens', []))
        if not tokens:
            continue
            
        # 변조할 토큰 개수
        num_mod = max(1, int(len(tokens) * ratio))
        
        # 랜덤하게 인덱스 선택
        indices = random.sample(range(len(tokens)), min(num_mod, len(tokens)))
        
        for idx in indices:
            # 랜덤 토큰으로 교체 (단, [PAD], [CLS], [SEP], [MASK] 등은 피하는 것이 좋지만 간단히 구현)
            # 여기서는 5번부터 vocab_size-1 사이의 랜덤 정수로 교체 (0~4는 특수 토큰 가정)
            tokens[idx] = random.randint(5, vocab_size - 1)
            
        session['label'] = 1  # 이상 레이블
    
    return anomaly_sessions


def calculate_metrics(
    normal_scores: List[float],
    anomaly_scores: List[float],
    threshold: float
) -> Dict[str, float]:
    """성능 메트릭 계산"""
    # 레이블 생성 (0: 정상, 1: 이상)
    y_true = [0] * len(normal_scores) + [1] * len(anomaly_scores)
    
    # 예측 생성 (임계값 기준)
    y_pred = [int(score >= threshold) for score in normal_scores] + \
             [int(score >= threshold) for score in anomaly_scores]
    
    # 점수 (ROC AUC용)
    y_scores = normal_scores + anomaly_scores
    
    # 메트릭 계산
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        roc_auc = 0.0
    
    # 혼동 행렬
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp),
        'y_true': y_true,
        'y_pred': y_pred,
        'y_scores': y_scores
    }


def find_optimal_threshold(
    normal_scores: List[float],
    anomaly_scores: List[float],
    num_thresholds: int = 100
) -> Tuple[float, Dict[str, float]]:
    """최적 임계값 찾기 (F1-Score 최대화)"""
    min_score = min(min(normal_scores), min(anomaly_scores))
    max_score = max(max(normal_scores), max(anomaly_scores))
    
    thresholds = np.linspace(min_score, max_score, num_thresholds)
    best_threshold = None
    best_metrics = None
    best_f1 = 0.0
    
    for threshold in thresholds:
        metrics = calculate_metrics(normal_scores, anomaly_scores, threshold)
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_threshold = threshold
            best_metrics = metrics
    
    return best_threshold, best_metrics


def plot_score_distribution(
    normal_scores: List[float],
    anomaly_scores: List[float],
    threshold: float,
    output_path: Path
):
    """점수 분포 시각화"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(normal_scores, bins=50, alpha=0.7, label='정상', color='blue', edgecolor='black')
    plt.hist(anomaly_scores, bins=50, alpha=0.7, label='이상', color='red', edgecolor='black')
    plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'임계값: {threshold:.4f}')
    plt.xlabel('이상 점수 (Loss)')
    plt.ylabel('빈도')
    plt.title('정상 vs 이상 점수 분포')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([normal_scores, anomaly_scores], tick_labels=['정상', '이상'])
    plt.axhline(threshold, color='green', linestyle='--', linewidth=2, label=f'임계값: {threshold:.4f}')
    plt.ylabel('이상 점수 (Loss)')
    plt.title('정상 vs 이상 점수 박스플롯')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"📊 점수 분포 그래프 저장: {output_path}")
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, output_path: Path):
    """혼동 행렬 시각화"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['정상', '이상'],
        yticklabels=['정상', '이상'],
        cbar_kws={'label': '개수'}
    )
    plt.xlabel('예측')
    plt.ylabel('실제')
    plt.title('혼동 행렬 (Confusion Matrix)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"📊 혼동 행렬 저장: {output_path}")
    plt.close()


def save_evaluation_results(
    results: Dict[str, Any],
    output_path: Path
):
    """평가 결과 저장"""
    # NumPy 배열을 리스트로 변환
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable_results[key] = float(value)
        else:
            serializable_results[key] = value
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 평가 결과 저장: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LogBERT 모델 성능 평가')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='모델 체크포인트 경로')
    parser.add_argument('--config', type=str, required=True,
                        help='설정 파일 경로')
    parser.add_argument('--validation-data', type=str, required=True,
                        help='검증 데이터 파일 경로 또는 디렉토리 경로')
    parser.add_argument('--normal-ratio', type=float, default=0.8,
                        help='정상 데이터 비율 (기본값: 0.8)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='최대 샘플 수 (빠른 평가용, 예: 1000)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--log-file', type=str, default=None,
                        help='로그 파일 경로')
    parser.add_argument('--generate-fake-anomaly', action='store_true',
                        help='가짜 이상 데이터 생성하여 평가 (이상 데이터가 없을 때 유용)')
    parser.add_argument('--anomaly-ratio', type=float, default=0.1,
                        help='가짜 이상 데이터 생성 시 토큰 변조 비율 (기본값: 0.1)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='평가 시 배치 크기 (기본값: 32)')
    
    args = parser.parse_args()
    
    # 1. 체크포인트 이름 추출 및 출력 경로 설정 (ratio 포함)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_name = checkpoint_path.stem 
    output_dir = Path(args.output_dir) / f"{checkpoint_name}_ratio_{args.anomaly_ratio}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 로그 파일 설정
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        logs_dir = Path(__file__).parent / 'logs'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f'evaluation_{checkpoint_name}_{timestamp}.log'
    
    setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info(f"LogBERT 모델 성능 평가: {checkpoint_name}")
    logger.info(f"변조 비율: {args.anomaly_ratio}")
    logger.info(f"결과 저장 위치: {output_dir}")
    logger.info("=" * 80)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"디바이스: {device}")
    
    # 설정 및 모델 로드
    config = load_config(args.config)
    model = load_model(args.checkpoint, config, device)
    
    # 검증 데이터 로드
    normal_sessions, anomaly_sessions = load_validation_data(
        args.validation_data,
        args.normal_ratio,
        args.max_samples
    )

    if not normal_sessions and not anomaly_sessions:
        logger.error("❌ 검증용 세션을 찾지 못했습니다. 경로를 확인해주세요.")
        return
    
    # 3. 가짜 이상 데이터 생성 (옵션)
    if args.generate_fake_anomaly:
        logger.info("\n" + "=" * 80)
        logger.info("⚠️  가짜 이상 데이터(Pseudo-Anomaly) 생성 모드")
        logger.info("=" * 80)
        
        all_normal = normal_sessions + anomaly_sessions
        normal_sessions = all_normal # 변조되지 않은 원본을 모두 정상군으로 설정
        
        # 사용자가 입력한 anomaly-ratio를 사용하여 이상 데이터 생성
        anomaly_sessions = generate_pseudo_anomalies(
            all_normal,
            vocab_size=config['model']['vocab_size'],
            max_seq_len=config['data']['max_seq_length'],
            ratio=args.anomaly_ratio
        )
        logger.info(f"✅ 데이터 재구성 완료: {len(normal_sessions)}개 정상 / {len(anomaly_sessions)}개 이상")

    # 4. 이상 점수 계산
    logger.info("\n" + "=" * 80)
    logger.info("이상 점수 계산 중...")
    logger.info("=" * 80)
    
    max_seq_length = config['data']['max_seq_length']
    vocab_size = config['model']['vocab_size']
    
    logger.info("정상 세션 평가 중...")
    normal_scores = calculate_anomaly_scores(
        model, normal_sessions, device, max_seq_length, vocab_size, batch_size=args.batch_size
    )
    
    logger.info("이상 세션 평가 중...")
    anomaly_scores = calculate_anomaly_scores(
        model, anomaly_sessions, device, max_seq_length, vocab_size, batch_size=args.batch_size
    )
    
    logger.info(f"✅ 정상 세션 점수 계산 완료: {len(normal_scores)}개")
    logger.info(f"✅ 이상 세션 점수 계산 완료: {len(anomaly_scores)}개")
    
    # 점수 통계
    logger.info("\n" + "=" * 80)
    logger.info("점수 통계")
    logger.info("=" * 80)
    logger.info(f"정상 세션 - 평균: {np.mean(normal_scores):.4f}, 표준편차: {np.std(normal_scores):.4f}")
    logger.info(f"이상 세션 - 평균: {np.mean(anomaly_scores):.4f}, 표준편차: {np.std(anomaly_scores):.4f}")
    
    # 최적 임계값 찾기 및 메트릭 계산
    logger.info("\n" + "=" * 80)
    logger.info("최적 임계값 탐색 및 결과 산출")
    logger.info("=" * 80)
    
    best_threshold, best_metrics = find_optimal_threshold(normal_scores, anomaly_scores)
    
    logger.info(f"✅ 최적 임계값: {best_threshold:.4f}")
    
    # 성능 메트릭 출력
    logger.info("\n" + "=" * 80)
    logger.info("📊 성능 평가 결과")
    logger.info("=" * 80)
    logger.info(f"정확도 (Accuracy):  {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
    logger.info(f"정밀도 (Precision): {best_metrics['precision']:.4f} ({best_metrics['precision']*100:.2f}%)")
    logger.info(f"재현율 (Recall):    {best_metrics['recall']:.4f} ({best_metrics['recall']*100:.2f}%)")
    logger.info(f"F1-Score:          {best_metrics['f1_score']:.4f} ({best_metrics['f1_score']*100:.2f}%)")
    logger.info(f"ROC AUC:           {best_metrics['roc_auc']:.4f}")
    
    logger.info("\n혼동 행렬:")
    logger.info(f"  True Negative (TN):  {best_metrics['true_negative']:4d} (정상을 정상으로 예측)")
    logger.info(f"  False Positive (FP): {best_metrics['false_positive']:4d} (정상을 이상으로 예측)")
    logger.info(f"  False Negative (FN): {best_metrics['false_negative']:4d} (이상을 정상으로 예측)")
    logger.info(f"  True Positive (TP):  {best_metrics['true_positive']:4d} (이상을 이상으로 예측)")
    
    # 5. 시각화 및 결과 저장
    plot_score_distribution(
        normal_scores, anomaly_scores, best_threshold,
        output_dir / f'score_dist_{checkpoint_name}.png'
    )
    
    plot_confusion_matrix(
        best_metrics['confusion_matrix'],
        output_dir / f'confusion_matrix_{checkpoint_name}.png'
    )
    
    # 결과 JSON 저장용 데이터 구성
    results = {
        'checkpoint': args.checkpoint,
        'anomaly_ratio': args.anomaly_ratio,
        'optimal_threshold': float(best_threshold),
        'metrics': {
            'accuracy': float(best_metrics['accuracy']),
            'precision': float(best_metrics['precision']),
            'recall': float(best_metrics['recall']),
            'f1_score': float(best_metrics['f1_score']),
            'roc_auc': float(best_metrics['roc_auc']),
        },
        'confusion_matrix': {
            'true_negative': int(best_metrics['true_negative']),
            'false_positive': int(best_metrics['false_positive']),
            'false_negative': int(best_metrics['false_negative']),
            'true_positive': int(best_metrics['true_positive']),
        },
        'statistics': {
            'normal_mean': float(np.mean(normal_scores)),
            'normal_std': float(np.std(normal_scores)),
            'normal_min': float(np.min(normal_scores)),
            'normal_max': float(np.max(normal_scores)),
            'anomaly_mean': float(np.mean(anomaly_scores)),
            'anomaly_std': float(np.std(anomaly_scores)),
            'anomaly_min': float(np.min(anomaly_scores)),
            'anomaly_max': float(np.max(anomaly_scores)),
        }
    }
    
    save_evaluation_results(results, output_dir / f'evaluation_results_{checkpoint_name}.json')
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ 평가 완료! 결과 저장: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
