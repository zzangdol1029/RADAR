#!/usr/bin/env python3
"""
학습 로그에서 Loss 데이터를 추출하고 학습 곡선을 시각화하는 스크립트
"""

import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from datetime import datetime

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# 한글 폰트가 없을 경우를 대비한 설정
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
    plt.rcParams['axes.unicode_minus'] = False
except:
    try:
        plt.rcParams['font.family'] = 'AppleGothic'  # macOS
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # 한글 폰트가 없으면 영어로 표시
        pass


def parse_log_file(log_path: Path) -> Dict:
    """로그 파일에서 Loss 데이터 추출"""
    
    print(f"로그 파일 파싱 중: {log_path}")
    
    step_losses = []  # (step, loss, avg_loss, epoch)
    epoch_losses = []  # (epoch, avg_loss)
    learning_rates = []  # (step, lr)
    
    current_epoch = 1
    step_pattern = re.compile(r'step=(\d+)')
    loss_pattern = re.compile(r'loss=([\d.]+)')
    avg_pattern = re.compile(r'avg=([\d.]+)')
    lr_pattern = re.compile(r'lr=([\d.e-]+)')
    
    # 에폭별 평균 Loss 패턴
    epoch_avg_pattern = re.compile(r'평균 Loss:\s*([\d.]+)', re.IGNORECASE)
    epoch_complete_pattern = re.compile(r'에폭\s*(\d+)/\d+\s*완료', re.IGNORECASE)
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 에폭 시작 감지
            if '에폭' in line and '시작' in line:
                epoch_match = re.search(r'에폭\s*(\d+)/', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
            
            # 에폭 완료 및 평균 Loss 추출
            epoch_complete_match = epoch_complete_pattern.search(line)
            if epoch_complete_match:
                epoch_num = int(epoch_complete_match.group(1))
                # 다음 줄에서 평균 Loss 찾기
                continue
            
            # 평균 Loss 추출 (에폭 완료 후)
            avg_match = epoch_avg_pattern.search(line)
            if avg_match:
                avg_loss = float(avg_match.group(1))
                epoch_losses.append((current_epoch, avg_loss))
                continue
            
            # Step별 Loss 추출 (progress bar 형식)
            if 'loss=' in line and 'avg=' in line:
                step_match = step_pattern.search(line)
                loss_match = loss_pattern.search(line)
                avg_match = avg_pattern.search(line)
                lr_match = lr_pattern.search(line)
                
                if step_match and loss_match and avg_match:
                    step = int(step_match.group(1))
                    loss = float(loss_match.group(1))
                    avg_loss = float(avg_match.group(1))
                    lr = float(lr_match.group(1)) if lr_match else None
                    
                    step_losses.append((step, loss, avg_loss, current_epoch))
                    if lr:
                        learning_rates.append((step, lr))
            
            # INFO 로그 형식: [Step 12200/100000] Loss=0.6761, Avg Loss=1.2427
            info_pattern = re.compile(r'\[Step\s+(\d+)/\d+\]\s+Loss=([\d.]+),\s+Avg\s+Loss=([\d.]+)')
            info_match = info_pattern.search(line)
            if info_match:
                step = int(info_match.group(1))
                loss = float(info_match.group(2))
                avg_loss = float(info_match.group(3))
                step_losses.append((step, loss, avg_loss, current_epoch))
    
    print(f"추출된 데이터:")
    print(f"  - Step별 Loss: {len(step_losses)}개")
    print(f"  - 에폭별 평균 Loss: {len(epoch_losses)}개")
    print(f"  - 학습률: {len(learning_rates)}개")
    
    return {
        'step_losses': step_losses,
        'epoch_losses': epoch_losses,
        'learning_rates': learning_rates
    }


def plot_training_curves(data: Dict, output_dir: Path):
    """학습 곡선 시각화"""
    
    step_losses = data['step_losses']
    epoch_losses = data['epoch_losses']
    learning_rates = data['learning_rates']
    
    if not step_losses:
        print("경고: Step별 Loss 데이터가 없습니다.")
        return
    
    # 데이터 정렬
    step_losses.sort(key=lambda x: x[0])
    epoch_losses.sort(key=lambda x: x[0])
    learning_rates.sort(key=lambda x: x[0])
    
    # 데이터 추출
    steps = [x[0] for x in step_losses]
    losses = [x[1] for x in step_losses]
    avg_losses = [x[2] for x in step_losses]
    epochs = [x[3] for x in step_losses]
    
    epoch_nums = [x[0] for x in epoch_losses]
    epoch_avg_losses = [x[1] for x in epoch_losses]
    
    lr_steps = [x[0] for x in learning_rates]
    lr_values = [x[1] for x in learning_rates]
    
    # 그래프 생성
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Step별 Loss (전체)
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(steps, losses, 'b-', alpha=0.3, linewidth=0.5, label='Batch Loss')
    ax1.plot(steps, avg_losses, 'r-', linewidth=2, label='Moving Average Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss (All Steps)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 로그 스케일로 표시
    
    # 2. Step별 Loss (스무딩)
    ax2 = plt.subplot(3, 2, 2)
    if len(avg_losses) > 100:
        # 이동 평균으로 스무딩
        window = min(100, len(avg_losses) // 10)
        smoothed = np.convolve(avg_losses, np.ones(window)/window, mode='valid')
        smoothed_steps = steps[window-1:]
        ax2.plot(smoothed_steps, smoothed, 'g-', linewidth=2, label=f'Smoothed (window={window})')
    ax2.plot(steps, avg_losses, 'r-', alpha=0.5, linewidth=1, label='Moving Average')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss (Smoothed)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 에폭별 평균 Loss
    ax3 = plt.subplot(3, 2, 3)
    if epoch_losses:
        ax3.plot(epoch_nums, epoch_avg_losses, 'o-', linewidth=2, markersize=8, color='purple')
        for i, (epoch, loss) in enumerate(zip(epoch_nums, epoch_avg_losses)):
            ax3.annotate(f'{loss:.4f}', (epoch, loss), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average Loss')
    ax3.set_title('Average Loss per Epoch')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(epoch_nums)
    
    # 4. 학습률 변화
    ax4 = plt.subplot(3, 2, 4)
    if learning_rates:
        ax4.plot(lr_steps, lr_values, 'g-', linewidth=2)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    else:
        ax4.text(0.5, 0.5, 'No learning rate data', 
                ha='center', va='center', transform=ax4.transAxes)
    
    # 5. Loss 감소율
    ax5 = plt.subplot(3, 2, 5)
    if len(avg_losses) > 1:
        # 초기 Loss 대비 감소율 계산
        initial_loss = avg_losses[0]
        reduction_rates = [(1 - loss/initial_loss) * 100 for loss in avg_losses]
        ax5.plot(steps, reduction_rates, 'orange', linewidth=2)
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Loss Reduction (%)')
        ax5.set_title(f'Loss Reduction Rate (Initial: {initial_loss:.4f})')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% reduction')
        ax5.axhline(y=75, color='g', linestyle='--', alpha=0.5, label='75% reduction')
        ax5.axhline(y=90, color='b', linestyle='--', alpha=0.5, label='90% reduction')
        ax5.legend()
    
    # 6. 에폭별 Loss 분포 (박스플롯)
    ax6 = plt.subplot(3, 2, 6)
    if epoch_losses:
        # 각 에폭의 Loss 값들 추출
        epoch_loss_dict = {}
        for step, loss, avg_loss, epoch in step_losses:
            if epoch not in epoch_loss_dict:
                epoch_loss_dict[epoch] = []
            epoch_loss_dict[epoch].append(loss)
        
        if epoch_loss_dict:
            epochs_list = sorted(epoch_loss_dict.keys())
            loss_data = [epoch_loss_dict[ep] for ep in epochs_list]
            bp = ax6.boxplot(loss_data, tick_labels=[f'Epoch {ep}' for ep in epochs_list], 
                           patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Loss Distribution')
            ax6.set_title('Loss Distribution per Epoch')
            ax6.grid(True, alpha=0.3, axis='y')
            plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 저장
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n그래프 저장: {output_path}")
    
    # 통계 정보 출력
    print("\n" + "="*80)
    print("학습 통계")
    print("="*80)
    if step_losses:
        print(f"총 Step 수: {len(step_losses):,}")
        print(f"초기 Loss: {avg_losses[0]:.4f}")
        print(f"최종 Loss: {avg_losses[-1]:.4f}")
        print(f"Loss 감소율: {(1 - avg_losses[-1]/avg_losses[0])*100:.2f}%")
        print(f"최소 Loss: {min(avg_losses):.4f} (Step {steps[avg_losses.index(min(avg_losses))]})")
    
    if epoch_losses:
        print(f"\n에폭별 평균 Loss:")
        for epoch, loss in epoch_losses:
            print(f"  Epoch {epoch}: {loss:.4f}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='학습 로그에서 학습 곡선 시각화')
    parser.add_argument('--log', type=str, default=None, 
                       help='로그 파일 경로 (지정하지 않으면 logs 폴더에서 가장 최근 파일 사용)')
    parser.add_argument('--output', type=str, default='.', 
                       help='출력 디렉토리 (기본값: 현재 디렉토리)')
    
    args = parser.parse_args()
    
    # 로그 파일 찾기
    if args.log:
        log_path = Path(args.log)
    else:
        logs_dir = Path(__file__).parent / 'logs'
        log_files = sorted(logs_dir.glob('training_*.log'), key=lambda x: x.stat().st_mtime, reverse=True)
        if not log_files:
            print("오류: 로그 파일을 찾을 수 없습니다.")
            return
        log_path = log_files[0]
        print(f"가장 최근 로그 파일 사용: {log_path}")
    
    if not log_path.exists():
        print(f"오류: 로그 파일을 찾을 수 없습니다: {log_path}")
        return
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 로그 파싱
    data = parse_log_file(log_path)
    
    # 그래프 생성
    plot_training_curves(data, output_dir)


if __name__ == '__main__':
    main()

