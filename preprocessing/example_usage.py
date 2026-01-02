"""
로그 전처리 파이프라인 사용 예제
"""

import sys
from pathlib import Path

# preprocessing 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from log_preprocessor import LogPreprocessor, load_config
import json

def example_basic_usage():
    """기본 사용 예제"""
    print("=" * 60)
    print("기본 사용 예제")
    print("=" * 60)
    
    # 설정 로드
    config_path = Path(__file__).parent / "preprocessing_config.yaml"
    config = load_config(str(config_path))
    
    # 전처리기 생성
    preprocessor = LogPreprocessor(config)
    
    # 단일 로그 파일 처리 (상위 디렉토리의 logs 폴더)
    log_path = Path(__file__).parent.parent / "logs" / "manager.log"
    sessions = preprocessor.process_log_file(str(log_path))
    
    # 결과 확인
    print(f"\n처리된 세션 수: {len(sessions)}")
    if sessions:
        print(f"\n첫 번째 세션:")
        print(json.dumps(sessions[0], indent=2, ensure_ascii=False))
    
    # 결과 저장
    preprocessor.save_results(sessions, "output_example.json")
    print(f"\n결과가 output_example.json에 저장되었습니다.")


def example_custom_config():
    """커스텀 설정 예제"""
    print("=" * 60)
    print("커스텀 설정 예제")
    print("=" * 60)
    
    # 커스텀 설정
    base_dir = Path(__file__).parent
    custom_config = {
        'log_directory': str(base_dir.parent / 'logs'),
        'output_path': str(base_dir / 'custom_output.json'),
        'sessionization_method': 'trace_id',  # Trace ID 기반 세션화
        'window_size': 30,  # 더 큰 윈도우
        'window_time': 600,  # 10분
        'max_seq_length': 256,  # 더 짧은 시퀀스
        'drain3_config_path': str(base_dir / 'drain3_config.yaml')
    }
    
    # 전처리기 생성
    preprocessor = LogPreprocessor(custom_config)
    
    # 전체 디렉토리 처리
    sessions = preprocessor.process_log_directory(custom_config['log_directory'])
    
    # ERROR 또는 WARN이 포함된 세션만 필터링
    error_sessions = [s for s in sessions if s.get('has_error') or s.get('has_warn')]
    
    print(f"\n전체 세션 수: {len(sessions)}")
    print(f"ERROR/WARN 포함 세션 수: {len(error_sessions)}")
    
    # 결과 저장
    preprocessor.save_results(sessions, custom_config['output_path'])
    print(f"\n결과가 {custom_config['output_path']}에 저장되었습니다.")


def example_analyze_results():
    """결과 분석 예제"""
    print("=" * 60)
    print("결과 분석 예제")
    print("=" * 60)
    
    # 저장된 결과 로드
    result_path = Path(__file__).parent / "preprocessed_logs.json"
    with open(result_path, 'r', encoding='utf-8') as f:
        sessions = json.load(f)
    
    # 통계 정보
    total_sessions = len(sessions)
    error_sessions = sum(1 for s in sessions if s.get('has_error'))
    warn_sessions = sum(1 for s in sessions if s.get('has_warn'))
    
    # 서비스별 통계
    service_stats = {}
    for session in sessions:
        service = session.get('service_name', 'unknown')
        if service not in service_stats:
            service_stats[service] = {'total': 0, 'error': 0, 'warn': 0}
        service_stats[service]['total'] += 1
        if session.get('has_error'):
            service_stats[service]['error'] += 1
        if session.get('has_warn'):
            service_stats[service]['warn'] += 1
    
    print(f"\n전체 통계:")
    print(f"  총 세션 수: {total_sessions}")
    print(f"  ERROR 포함: {error_sessions} ({error_sessions/total_sessions*100:.1f}%)")
    print(f"  WARN 포함: {warn_sessions} ({warn_sessions/total_sessions*100:.1f}%)")
    
    print(f"\n서비스별 통계:")
    for service, stats in service_stats.items():
        print(f"  {service}:")
        print(f"    총 세션: {stats['total']}")
        print(f"    ERROR: {stats['error']}")
        print(f"    WARN: {stats['warn']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_type = sys.argv[1]
        if example_type == "basic":
            example_basic_usage()
        elif example_type == "custom":
            example_custom_config()
        elif example_type == "analyze":
            example_analyze_results()
        else:
            print("사용법: python example_usage.py [basic|custom|analyze]")
    else:
        print("사용 가능한 예제:")
        print("  python example_usage.py basic    - 기본 사용 예제")
        print("  python example_usage.py custom   - 커스텀 설정 예제")
        print("  python example_usage.py analyze  - 결과 분석 예제")

