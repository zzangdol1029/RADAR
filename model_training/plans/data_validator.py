#!/usr/bin/env python3
"""
전처리 데이터 검증 스크립트
목적: output 폴더의 JSON 파일들을 검증하고 통계 출력
"""

import json
import os
from pathlib import Path
from collections import Counter
import statistics

def format_size(bytes_size):
    """바이트를 읽기 쉬운 형식으로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def validate_data():
    """전처리 데이터 검증"""
    # 스크립트 파일 위치 기준으로 경로 설정
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "output"
    
    if not output_dir.exists():
        print(f"[ERROR] 출력 디렉토리가 존재하지 않습니다: {output_dir}")
        return
    
    files = sorted(output_dir.glob("preprocessed_logs_*.json"))
    
    if not files:
        print(f"[ERROR] JSON 파일이 없습니다: {output_dir}")
        return
    
    print("=" * 80)
    print("[DATA VALIDATION] 전처리 데이터 검증")
    print("=" * 80)
    print(f"\n총 파일 수: {len(files)}")
    
    # 파일 크기 통계
    file_sizes = []
    for file in files:
        file_sizes.append(file.stat().st_size)
    
    total_size = sum(file_sizes)
    print(f"총 데이터 크기: {format_size(total_size)}")
    print(f"평균 파일 크기: {format_size(statistics.mean(file_sizes))}")
    print(f"최대 파일 크기: {format_size(max(file_sizes))}")
    print(f"최소 파일 크기: {format_size(min(file_sizes))}")
    
    # 샘플 파일 검증
    print("\n" + "=" * 80)
    print("[SAMPLE ANALYSIS] 샘플 파일 분석")
    print("=" * 80)
    
    sample_file = files[0]
    print(f"\n샘플 파일: {sample_file.name}")
    print(f"파일 크기: {format_size(sample_file.stat().st_size)}")
    
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"세션 수: {len(data)}")
        
        # 첫 세션 구조 확인
        if data:
            session = data[0]
            print(f"\n첫 세션 구조:")
            for key in session.keys():
                value = session[key]
                value_type = type(value).__name__
                
                if isinstance(value, list):
                    print(f"  - {key}: {value_type} (길이: {len(value)})")
                elif isinstance(value, str):
                    preview = value[:50] + "..." if len(value) > 50 else value
                    print(f"  - {key}: {value_type} ('{preview}')")
                else:
                    print(f"  - {key}: {value_type} ({value})")
            
            # 시퀀스 길이 분포
            seq_lengths = [len(s.get('event_sequence', [])) for s in data if 'event_sequence' in s]
            if seq_lengths:
                print(f"\n시퀀스 길이 통계:")
                print(f"  평균: {statistics.mean(seq_lengths):.1f}")
                print(f"  중앙값: {statistics.median(seq_lengths):.1f}")
                print(f"  최소: {min(seq_lengths)}")
                print(f"  최대: {max(seq_lengths)}")
                print(f"  표준편차: {statistics.stdev(seq_lengths):.1f}")
            
            # 서비스 분포
            services = [s.get('service_name', 'unknown') for s in data]
            service_counts = Counter(services)
            print(f"\n서비스 분포:")
            for service, count in service_counts.most_common():
                percentage = (count / len(data)) * 100
                print(f"  - {service}: {count} ({percentage:.1f}%)")
            
            # 에러 통계
            error_count = sum(1 for s in data if s.get('has_error', False))
            warn_count = sum(1 for s in data if s.get('has_warn', False))
            print(f"\n로그 레벨 통계:")
            print(f"  - ERROR 포함 세션: {error_count} ({error_count/len(data)*100:.1f}%)")
            print(f"  - WARN 포함 세션: {warn_count} ({warn_count/len(data)*100:.1f}%)")
            
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 파싱 실패: {e}")
    except Exception as e:
        print(f"[ERROR] {e}")
    
    # 날짜별 파일 개수
    print("\n" + "=" * 80)
    print("[MONTHLY STATS] 월별 파일 수")
    print("=" * 80)
    
    month_counts = Counter()
    for file in files:
        # preprocessed_logs_2025-02-24.json 형식에서 년월 추출
        parts = file.stem.split('_')
        if len(parts) >= 3:
            date_str = parts[2]  # 2025-02-24
            year_month = date_str[:7]  # 2025-02
            month_counts[year_month] += 1
    
    for month, count in sorted(month_counts.items()):
        print(f"  {month}: {count}개")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] 검증 완료")
    print("=" * 80)
    
    # 권장사항
    print("\n[RECOMMENDATION] 권장사항:")
    print("  1. 테스트 학습: 최근 10개 파일 사용")
    print("  2. 중규모 학습: 최근 100개 파일 사용")
    print("  3. 전체 학습: 324개 전체 파일 사용")
    print()

if __name__ == '__main__':
    validate_data()
