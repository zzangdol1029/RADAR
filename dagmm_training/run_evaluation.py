#!/usr/bin/env python3
"""
DAGMM & DeepLog 통합 테스트 및 평가 스크립트
CPU 환경 최적화 (Intel 7세대, 16GB RAM)
"""

import os
import sys
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from model_dagmm import DAGMM, DAGMMLoss
from model_deeplog import DeepLog
from evaluate_dagmm import DAGMMEvaluator
from evaluate_deeplog import DeepLogEvaluator

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def load_data_streaming(file_path: str, max_sessions: int = 5000):
    """스트리밍 방식으로 데이터 로드"""
    try:
        import ijson
        sessions = []
        with open(file_path, 'rb') as f:
            parser = ijson.items(f, 'item')
            for i, session in enumerate(parser):
                if i >= max_sessions:
                    break
                event_seq = session.get('event_sequence', [])
                if event_seq:
                    sessions.append({'event_sequence': event_seq})
                if (i + 1) % 1000 == 0:
                    print(f"  로드 중... {i+1:,}개")
        return sessions
    except ImportError:
        print("ijson 패키지가 필요합니다. pip install ijson")
        return []


def prepare_data(sessions, window_size=10, max_samples=20000):
    """데이터 전처리"""
    dagmm_samples = []
    deeplog_sequences = []
    deeplog_labels = []
    all_event_ids = set()
    
    for session in sessions:
        seq = session.get('event_sequence', [])
        if not seq:
            continue
        
        all_event_ids.update(seq)
        
        # DAGMM
        if len(seq) < window_size:
            padded = seq + [0] * (window_size - len(seq))
            dagmm_samples.append(padded)
        else:
            for i in range(len(seq) - window_size + 1):
                dagmm_samples.append(seq[i:i + window_size])
        
        # DeepLog
        if len(seq) > window_size:
            for i in range(len(seq) - window_size):
                deeplog_sequences.append(seq[i:i + window_size])
                deeplog_labels.append(seq[i + window_size])
    
    # 이벤트 ID 매핑
    event_id_map = {eid: idx + 1 for idx, eid in enumerate(sorted(all_event_ids))}
    event_id_map[0] = 0
    num_classes = len(event_id_map)
    
    # 제한 및 매핑
    dagmm_samples = [[event_id_map.get(x, 0) for x in seq] for seq in dagmm_samples[:max_samples]]
    deeplog_sequences = [[event_id_map.get(x, 0) for x in seq] for seq in deeplog_sequences[:max_samples]]
    deeplog_labels = [event_id_map.get(x, 0) for x in deeplog_labels[:max_samples]]
    
    return dagmm_samples, deeplog_sequences, deeplog_labels, num_classes, event_id_map


def train_and_evaluate_dagmm(
    train_samples, test_samples, num_classes, window_size, 
    epochs=3, batch_size=32, output_dir='results'
):
    """DAGMM 학습 및 평가"""
    print("\n" + "=" * 60)
    print("DAGMM 학습 및 평가")
    print("=" * 60)
    
    # DataLoader
    train_loader = DataLoader(TensorDataset(torch.LongTensor(train_samples)), 
                             batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.LongTensor(test_samples)), 
                            batch_size=batch_size)
    
    # 모델
    model = DAGMM(num_classes=num_classes, window_size=window_size)
    criterion = DAGMMLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"학습 샘플: {len(train_samples):,}, 테스트 샘플: {len(test_samples):,}")
    
    # 학습
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for batch_idx, (data,) in enumerate(train_loader):
            z, x_hat, z_c, gamma, x_flat = model(data)
            phi, mu, sigma = model.compute_gmm_params(z_c, gamma)
            
            if batch_idx == 0 and epoch == 1:
                model.phi = phi
                model.mu = mu
                model.sigma = sigma
            else:
                m = 0.9
                model.phi = m * model.phi.detach() + (1-m) * phi
                model.mu = m * model.mu.detach() + (1-m) * mu
                model.sigma = m * model.sigma.detach() + (1-m) * sigma
            
            energy = model.compute_energy(z_c, phi, mu, sigma)
            loss, _, _, _ = criterion(x_flat, x_hat, energy, sigma)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
    
    train_time = time.time() - start_time
    print(f"\n✅ 학습 완료! 소요 시간: {train_time/60:.1f}분")
    
    # 평가
    print("\n📊 DAGMM 성능 평가 중...")
    evaluator = DAGMMEvaluator(model)
    scores = evaluator.compute_scores(test_loader)
    metrics = evaluator.evaluate_unsupervised(scores)
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    
    # 에너지 분포 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(scores['energy'], bins=50, alpha=0.7, color='blue')
    plt.axvline(metrics['threshold'], color='red', linestyle='--', label=f"임계값 (95%): {metrics['threshold']:.2f}")
    plt.xlabel('에너지 점수')
    plt.ylabel('빈도')
    plt.title('DAGMM 에너지 점수 분포')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(scores['reconstruction_error'], bins=50, alpha=0.7, color='green')
    plt.xlabel('재구성 오류')
    plt.ylabel('빈도')
    plt.title('DAGMM 재구성 오류 분포')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dagmm_distribution.png'), dpi=150)
    print(f"  그래프 저장: {output_dir}/dagmm_distribution.png")
    plt.close()
    
    # 리포트
    evaluator.generate_report(scores, output_dir=output_dir)
    
    # 모델 저장
    torch.save({
        'model': model.state_dict(),
        'config': {'num_classes': num_classes, 'window_size': window_size},
        'metrics': metrics,
    }, os.path.join(output_dir, 'dagmm_model.pt'))
    print(f"  모델 저장: {output_dir}/dagmm_model.pt")
    
    return model, metrics


def train_and_evaluate_deeplog(
    train_sequences, train_labels, test_sequences, test_labels,
    num_classes, epochs=3, batch_size=32, output_dir='results'
):
    """DeepLog 학습 및 평가"""
    print("\n" + "=" * 60)
    print("DeepLog 학습 및 평가")
    print("=" * 60)
    
    if len(train_sequences) == 0:
        print("⚠️ 학습 데이터가 없습니다. (시퀀스 길이가 window_size보다 짧음)")
        return None, {}
    
    # DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.LongTensor(train_sequences), torch.LongTensor(train_labels)),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.LongTensor(test_sequences), torch.LongTensor(test_labels)),
        batch_size=batch_size
    )
    
    # 모델
    model = DeepLog(num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"학습 샘플: {len(train_sequences):,}, 테스트 샘플: {len(test_sequences):,}")
    
    # 학습
    start_time = time.time()
    best_acc = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for seq, lbl in train_loader:
            outputs = model(seq)
            logits = outputs['logits']
            loss = criterion(logits, lbl)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += lbl.size(0)
            correct += predicted.eq(lbl).sum().item()
        
        acc = 100. * correct / total
        print(f"  Epoch {epoch}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
    
    train_time = time.time() - start_time
    print(f"\n✅ 학습 완료! 소요 시간: {train_time/60:.1f}분, 최고 정확도: {best_acc:.2f}%")
    
    # 평가
    print("\n📊 DeepLog 성능 평가 중...")
    evaluator = DeepLogEvaluator(model)
    results = evaluator.predict(test_loader)
    
    # 메트릭 계산
    acc_metrics = evaluator.compute_accuracy_metrics(results)
    cls_metrics = evaluator.compute_classification_metrics(results)
    conf_metrics = evaluator.analyze_prediction_confidence(results)
    
    # 시각화
    os.makedirs(output_dir, exist_ok=True)
    
    # Top-k 정확도
    k_values = [1, 2, 3, 5, 10]
    accuracies = []
    valid_k = []
    
    for k in k_values:
        if k <= results['probabilities'].shape[1]:
            top_k_preds = np.argsort(results['probabilities'], axis=1)[:, -k:]
            correct = [1 if label in top_k else 0 
                      for label, top_k in zip(results['labels'], top_k_preds)]
            accuracies.append(np.mean(correct) * 100)
            valid_k.append(k)
    
    if valid_k:
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(valid_k)), accuracies, color='steelblue')
        plt.xticks(range(len(valid_k)), [f'Top-{k}' for k in valid_k])
        plt.ylabel('정확도 (%)')
        plt.title('DeepLog Top-k 정확도')
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 1, f'{acc:.1f}%', ha='center')
        plt.savefig(os.path.join(output_dir, 'deeplog_topk_accuracy.png'), dpi=150)
        print(f"  그래프 저장: {output_dir}/deeplog_topk_accuracy.png")
        plt.close()
    
    # 신뢰도 분포
    pred_probs = np.max(results['probabilities'], axis=1)
    correct_mask = results['predictions'] == results['labels']
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(pred_probs, bins=50, alpha=0.7, color='blue')
    plt.axvline(np.mean(pred_probs), color='red', linestyle='--', label=f'평균: {np.mean(pred_probs):.3f}')
    plt.xlabel('예측 신뢰도')
    plt.ylabel('빈도')
    plt.title('예측 신뢰도 분포')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if correct_mask.any():
        plt.hist(pred_probs[correct_mask], bins=30, alpha=0.7, label='정답', color='green')
    if (~correct_mask).any():
        plt.hist(pred_probs[~correct_mask], bins=30, alpha=0.7, label='오답', color='red')
    plt.xlabel('예측 신뢰도')
    plt.ylabel('빈도')
    plt.title('정답/오답별 신뢰도')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deeplog_confidence.png'), dpi=150)
    print(f"  그래프 저장: {output_dir}/deeplog_confidence.png")
    plt.close()
    
    # 리포트
    evaluator.generate_report(results, output_dir=output_dir)
    
    # 모델 저장
    torch.save({
        'model': model.state_dict(),
        'config': {'num_classes': num_classes},
        'metrics': {**acc_metrics, **cls_metrics},
    }, os.path.join(output_dir, 'deeplog_model.pt'))
    print(f"  모델 저장: {output_dir}/deeplog_model.pt")
    
    return model, {**acc_metrics, **cls_metrics, **conf_metrics}


def main():
    parser = argparse.ArgumentParser(description='DAGMM & DeepLog 통합 테스트 및 평가')
    parser.add_argument('--data-file', type=str, 
                       default=r'C:\Users\ssoo2\Downloads\logFile\logFile\preprocessed_logs_2025-05-01.json',
                       help='데이터 파일 경로')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='결과 저장 디렉토리')
    parser.add_argument('--max-sessions', type=int, default=5000, help='최대 세션 수')
    parser.add_argument('--window-size', type=int, default=5, help='윈도우 크기')
    parser.add_argument('--epochs', type=int, default=3, help='학습 에폭')
    parser.add_argument('--batch-size', type=int, default=32, help='배치 크기')
    args = parser.parse_args()
    
    print("=" * 60)
    print("DAGMM & DeepLog 통합 테스트 및 평가")
    print("=" * 60)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"데이터 파일: {args.data_file}")
    print(f"윈도우 크기: {args.window_size}")
    print()
    
    # 데이터 로드
    print("📂 데이터 로드 중...")
    sessions = load_data_streaming(args.data_file, args.max_sessions)
    print(f"로드 완료: {len(sessions):,}개 세션")
    
    # 데이터 전처리
    print("\n🔧 데이터 전처리 중...")
    dagmm_samples, deeplog_sequences, deeplog_labels, num_classes, _ = prepare_data(
        sessions, window_size=args.window_size
    )
    print(f"DAGMM 샘플: {len(dagmm_samples):,}개")
    print(f"DeepLog 샘플: {len(deeplog_sequences):,}개")
    print(f"클래스 수: {num_classes}")
    
    # 데이터 분할
    n_dagmm = len(dagmm_samples)
    n_deeplog = len(deeplog_sequences)
    
    idx_dagmm = np.random.permutation(n_dagmm)
    split_dagmm = int(n_dagmm * 0.2)
    train_dagmm = [dagmm_samples[i] for i in idx_dagmm[split_dagmm:]]
    test_dagmm = [dagmm_samples[i] for i in idx_dagmm[:split_dagmm]]
    
    if n_deeplog > 0:
        idx_deeplog = np.random.permutation(n_deeplog)
        split_deeplog = int(n_deeplog * 0.2)
        train_seq = [deeplog_sequences[i] for i in idx_deeplog[split_deeplog:]]
        train_lbl = [deeplog_labels[i] for i in idx_deeplog[split_deeplog:]]
        test_seq = [deeplog_sequences[i] for i in idx_deeplog[:split_deeplog]]
        test_lbl = [deeplog_labels[i] for i in idx_deeplog[:split_deeplog]]
    else:
        train_seq, train_lbl = [], []
        test_seq, test_lbl = [], []
    
    # DAGMM 학습 및 평가
    dagmm_model, dagmm_metrics = train_and_evaluate_dagmm(
        train_dagmm, test_dagmm, num_classes, args.window_size,
        epochs=args.epochs, batch_size=args.batch_size, output_dir=args.output_dir
    )
    
    # DeepLog 학습 및 평가
    deeplog_model, deeplog_metrics = train_and_evaluate_deeplog(
        train_seq, train_lbl, test_seq, test_lbl,
        num_classes, epochs=args.epochs, batch_size=args.batch_size, output_dir=args.output_dir
    )
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("🎉 최종 결과 요약")
    print("=" * 60)
    
    print("\n📊 DAGMM 결과:")
    print(f"  - 에너지 점수 평균: {dagmm_metrics.get('energy_mean', 0):.4f}")
    print(f"  - 이상 비율 (상위 5%): {dagmm_metrics.get('anomaly_ratio', 0)*100:.2f}%")
    
    if deeplog_metrics:
        print("\n📊 DeepLog 결과:")
        print(f"  - Top-1 정확도: {deeplog_metrics.get('accuracy', 0)*100:.2f}%")
        print(f"  - F1 Score: {deeplog_metrics.get('f1', 0):.4f}")
    
    print(f"\n📁 결과 저장 위치: {args.output_dir}")
    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
