#!/usr/bin/env python3
"""
CPU 환경 테스트 스크립트 - 메모리 효율 버전
Intel 7세대, 16GB RAM 최적화
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import ijson  # 스트리밍 JSON 파서

# 현재 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from model_dagmm import DAGMM, DAGMMLoss
from model_deeplog import DeepLog
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("DAGMM & DeepLog CPU 테스트 (메모리 효율 버전)")
print("=" * 60)
print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CPU 사용")
print()

# ========== 설정 ==========
DATA_FILE = r"C:\Users\ssoo2\Downloads\logFile\logFile\preprocessed_logs_2025-05-01.json"
WINDOW_SIZE = 10
BATCH_SIZE = 32  # CPU용 작은 배치
TEST_EPOCHS = 3  # 테스트용 적은 에폭
MAX_SESSIONS = 5000  # 메모리 절약을 위해 5000개만 사용

# ========== 1. 데이터 로드 (스트리밍) ==========
print("=" * 60)
print("1단계: 데이터 로드 (스트리밍 방식)")
print("=" * 60)

start_time = time.time()
print(f"파일: {DATA_FILE}")
print(f"최대 {MAX_SESSIONS:,}개 세션만 로드합니다...")

sessions = []
try:
    with open(DATA_FILE, 'rb') as f:
        parser = ijson.items(f, 'item')
        for i, session in enumerate(parser):
            if i >= MAX_SESSIONS:
                break
            # event_sequence만 추출
            event_seq = session.get('event_sequence', [])
            if event_seq:
                sessions.append({'event_sequence': event_seq})
            
            if (i + 1) % 1000 == 0:
                print(f"  로드 중... {i+1:,}개")
    
    load_time = time.time() - start_time
    print(f"✅ 로드 완료: {len(sessions):,}개 세션 ({load_time:.1f}초)")
    
except ImportError:
    print("ijson 패키지가 없습니다. 설치합니다...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ijson'])
    print("ijson 설치 완료. 다시 실행해주세요.")
    sys.exit(0)
except Exception as e:
    print(f"스트리밍 로드 실패: {e}")
    print("일반 JSON 로드를 시도합니다...")
    
    # 일반 방식으로 일부만 로드
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        # 파일의 처음 부분만 읽기
        content = f.read(50_000_000)  # 50MB만 읽기
        # JSON 배열에서 완전한 객체들만 추출
        content = content.rsplit('}, {', 1)[0] + '}]'
        if not content.startswith('['):
            content = '[' + content
        
        all_sessions = json.loads(content)[:MAX_SESSIONS]
        sessions = [{'event_sequence': s.get('event_sequence', [])} for s in all_sessions if s.get('event_sequence')]
    
    load_time = time.time() - start_time
    print(f"✅ 로드 완료: {len(sessions):,}개 세션 ({load_time:.1f}초)")

# ========== 2. 데이터 전처리 ==========
print()
print("=" * 60)
print("2단계: 데이터 전처리")
print("=" * 60)

# DAGMM용 데이터
dagmm_samples = []
all_event_ids = set()

# DeepLog용 데이터
deeplog_sequences = []
deeplog_labels = []

for session in sessions:
    seq = session.get('event_sequence', [])
    if not seq:
        continue
    
    all_event_ids.update(seq)
    
    # DAGMM: 윈도우 생성
    if len(seq) < WINDOW_SIZE:
        padded = seq + [0] * (WINDOW_SIZE - len(seq))
        dagmm_samples.append(padded)
    else:
        for i in range(len(seq) - WINDOW_SIZE + 1):
            dagmm_samples.append(seq[i:i + WINDOW_SIZE])
    
    # DeepLog: 입력 + 레이블 생성
    if len(seq) > WINDOW_SIZE:
        for i in range(len(seq) - WINDOW_SIZE):
            deeplog_sequences.append(seq[i:i + WINDOW_SIZE])
            deeplog_labels.append(seq[i + WINDOW_SIZE])

del sessions  # 메모리 해제

# 이벤트 ID 재매핑
event_id_map = {eid: idx + 1 for idx, eid in enumerate(sorted(all_event_ids))}
event_id_map[0] = 0
num_classes = len(event_id_map)

# 데이터 재매핑
dagmm_samples = [[event_id_map.get(x, 0) for x in seq] for seq in dagmm_samples[:20000]]  # 2만개로 제한
deeplog_sequences = [[event_id_map.get(x, 0) for x in seq] for seq in deeplog_sequences[:20000]]
deeplog_labels = [event_id_map.get(x, 0) for x in deeplog_labels[:20000]]

print(f"✅ 전처리 완료")
print(f"  - DAGMM 샘플: {len(dagmm_samples):,}개")
print(f"  - DeepLog 샘플: {len(deeplog_sequences):,}개")
print(f"  - 이벤트 클래스 수: {num_classes}")

# ========== 3. 시간 추정 ==========
print()
print("=" * 60)
print("3단계: 학습 시간 추정")
print("=" * 60)

# 작은 샘플로 속도 테스트
print("속도 테스트 중...")

# DAGMM 속도 테스트
dagmm_model = DAGMM(num_classes=num_classes, window_size=WINDOW_SIZE)
sample_size = min(100, len(dagmm_samples))
dagmm_test_data = torch.LongTensor(dagmm_samples[:sample_size])

start = time.time()
for _ in range(5):
    with torch.no_grad():
        _ = dagmm_model(dagmm_test_data)
dagmm_batch_time = (time.time() - start) / 5

# DeepLog 속도 테스트
deeplog_model = DeepLog(num_classes=num_classes)
sample_size = min(100, len(deeplog_sequences)) if deeplog_sequences else 100
deeplog_test_data = torch.LongTensor(deeplog_sequences[:sample_size] if deeplog_sequences else [[0]*WINDOW_SIZE]*100)

start = time.time()
for _ in range(5):
    with torch.no_grad():
        _ = deeplog_model(deeplog_test_data)
deeplog_batch_time = (time.time() - start) / 5

# 추정 계산
dagmm_batches = len(dagmm_samples) // BATCH_SIZE + 1
deeplog_batches = len(deeplog_sequences) // BATCH_SIZE + 1 if deeplog_sequences else 0

dagmm_epoch_time = dagmm_batches * dagmm_batch_time * 3
deeplog_epoch_time = deeplog_batches * deeplog_batch_time * 3

dagmm_total_time = dagmm_epoch_time * TEST_EPOCHS
deeplog_total_time = deeplog_epoch_time * TEST_EPOCHS

print()
print("📊 예상 학습 시간:")
print(f"  ┌─────────────────────────────────────────────")
print(f"  │ DAGMM ({TEST_EPOCHS} epochs, {len(dagmm_samples):,} 샘플)")
print(f"  │  - 배치 수: {dagmm_batches:,}")
print(f"  │  - 총 예상: 약 {dagmm_total_time/60:.1f}분")
print(f"  ├─────────────────────────────────────────────")
print(f"  │ DeepLog ({TEST_EPOCHS} epochs, {len(deeplog_sequences):,} 샘플)")
print(f"  │  - 배치 수: {deeplog_batches:,}")
print(f"  │  - 총 예상: 약 {deeplog_total_time/60:.1f}분")
print(f"  └─────────────────────────────────────────────")
print(f"  총 예상 시간: 약 {(dagmm_total_time + deeplog_total_time)/60:.1f}분")
print()

# ========== 4. DAGMM 학습 ==========
print("=" * 60)
print("4단계: DAGMM 학습 시작")
print("=" * 60)

# 데이터 분할
n = len(dagmm_samples)
idx = np.random.permutation(n)
split = int(n * 0.2)
train_dagmm = [dagmm_samples[i] for i in idx[split:]]
test_dagmm = [dagmm_samples[i] for i in idx[:split]]

train_loader = DataLoader(TensorDataset(torch.LongTensor(train_dagmm)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.LongTensor(test_dagmm)), batch_size=BATCH_SIZE)

print(f"학습: {len(train_dagmm):,}개, 테스트: {len(test_dagmm):,}개")

dagmm_model = DAGMM(num_classes=num_classes, window_size=WINDOW_SIZE)
criterion = DAGMMLoss()
optimizer = torch.optim.Adam(dagmm_model.parameters(), lr=0.001)

dagmm_start = time.time()
best_loss = float('inf')

for epoch in range(1, TEST_EPOCHS + 1):
    dagmm_model.train()
    total_loss = 0
    
    for batch_idx, (data,) in enumerate(train_loader):
        z, x_hat, z_c, gamma, x_flat = dagmm_model(data)
        phi, mu, sigma = dagmm_model.compute_gmm_params(z_c, gamma)
        
        if batch_idx == 0 and epoch == 1:
            dagmm_model.phi = phi
            dagmm_model.mu = mu
            dagmm_model.sigma = sigma
        else:
            m = 0.9
            dagmm_model.phi = m * dagmm_model.phi.detach() + (1-m) * phi
            dagmm_model.mu = m * dagmm_model.mu.detach() + (1-m) * mu
            dagmm_model.sigma = m * dagmm_model.sigma.detach() + (1-m) * sigma
        
        energy = dagmm_model.compute_energy(z_c, phi, mu, sigma)
        loss, _, _, _ = criterion(x_flat, x_hat, energy, sigma)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dagmm_model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 200 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{TEST_EPOCHS} - Loss: {avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss

dagmm_time = time.time() - dagmm_start
print(f"\n✅ DAGMM 완료! 시간: {dagmm_time/60:.1f}분, 최저 Loss: {best_loss:.4f}")

# ========== 5. DeepLog 학습 ==========
print()
print("=" * 60)
print("5단계: DeepLog 학습 시작")
print("=" * 60)

if not deeplog_sequences:
    print("⚠️ DeepLog 학습 데이터가 없습니다.")
else:
    n = len(deeplog_sequences)
    idx = np.random.permutation(n)
    split = int(n * 0.2)
    
    train_seq = [deeplog_sequences[i] for i in idx[split:]]
    train_lbl = [deeplog_labels[i] for i in idx[split:]]
    test_seq = [deeplog_sequences[i] for i in idx[:split]]
    test_lbl = [deeplog_labels[i] for i in idx[:split]]
    
    train_loader = DataLoader(TensorDataset(torch.LongTensor(train_seq), torch.LongTensor(train_lbl)), batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"학습: {len(train_seq):,}개, 테스트: {len(test_seq):,}개")
    
    deeplog_model = DeepLog(num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(deeplog_model.parameters(), lr=0.001)
    
    deeplog_start = time.time()
    best_acc = 0
    
    for epoch in range(1, TEST_EPOCHS + 1):
        deeplog_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (seq, lbl) in enumerate(train_loader):
            outputs = deeplog_model(seq)
            logits = outputs['logits']
            loss = criterion(logits, lbl)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += lbl.size(0)
            correct += predicted.eq(lbl).sum().item()
            
            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx+1}/{len(train_loader)}")
        
        acc = 100. * correct / total
        print(f"Epoch {epoch}/{TEST_EPOCHS} - Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
    
    deeplog_time = time.time() - deeplog_start
    print(f"\n✅ DeepLog 완료! 시간: {deeplog_time/60:.1f}분, 최고 Acc: {best_acc:.2f}%")

# ========== 완료 ==========
print()
print("=" * 60)
print("🎉 테스트 완료!")
print("=" * 60)
total_time = time.time() - start_time
print(f"총 소요 시간: {total_time/60:.1f}분")
print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
