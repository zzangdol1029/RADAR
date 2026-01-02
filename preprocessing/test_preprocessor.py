"""
로그 전처리 파이프라인 테스트 스크립트
작은 샘플 로그로 전처리 기능을 테스트합니다.
"""

import json
import tempfile
import os
import sys
from pathlib import Path

# preprocessing 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from log_preprocessor import LogPreprocessor, load_config

def create_sample_log_file():
    """테스트용 샘플 로그 파일 생성"""
    sample_logs = """2025-12-08 17:23:47.950 INFO  kr.re.bbp.Application --- [main] Starting Application using Java 17.0.16
2025-12-08 17:23:47.956 INFO  kr.re.bbp.Application --- [main] The following 1 profile is active: "test"
2025-12-08 17:23:50.947 WARN  org.hibernate.mapping.RootClass --- [main] HHH000038: Composite-id class does not override equals()
2025-12-08 17:23:55.796 DEBUG org.hibernate.SQL --- [scheduling-1] select attachfile0_.atch_file_sn from bio.cs_atch_file_m attachfile0_ where attachfile0_.atch_file_group_sn=?
2025-12-08 17:23:55.804 DEBUG org.hibernate.SQL --- [scheduling-2] select attachfile0_.atch_file_sn from bio.cs_atch_file_m attachfile0_ where attachfile0_.reg_dt>? and attachfile0_.atch_file_tmpr_strg_yn=?
2025-12-08 17:24:00.807 DEBUG org.hibernate.SQL --- [scheduling-2] select attachfile0_.atch_file_sn from bio.cs_atch_file_m attachfile0_ where attachfile0_.reg_dt>? and attachfile0_.atch_file_tmpr_strg_yn=?
2025-12-08 17:24:05.810 DEBUG org.hibernate.SQL --- [scheduling-2] select attachfile0_.atch_file_sn from bio.cs_atch_file_m attachfile0_ where attachfile0_.reg_dt>? and attachfile0_.atch_file_tmpr_strg_yn=?
2025-12-08 17:24:10.812 ERROR kr.re.bbp.manager.Service --- [worker-1] Connection timeout from 192.168.0.1
2025-12-08 17:24:15.814 WARN  kr.re.bbp.manager.Service --- [worker-1] Retrying connection to 192.168.0.2
2025-12-08 17:24:20.816 INFO  kr.re.bbp.manager.Service --- [worker-1] Connection established successfully
2025-12-08 17:24:25.818 INFO  kr.re.bbp.Application --- [main] Started Application in 8.153 seconds
"""
    
    # 임시 파일 생성
    fd, temp_path = tempfile.mkstemp(suffix='.log', text=True)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(sample_logs)
        return temp_path
    except:
        os.close(fd)
        raise

def test_preprocessing():
    """전처리 테스트"""
    print("=" * 60)
    print("로그 전처리 파이프라인 테스트")
    print("=" * 60)
    
    # 샘플 로그 파일 생성
    sample_log_path = create_sample_log_file()
    print(f"\n1. 샘플 로그 파일 생성: {sample_log_path}")
    
    try:
        # 설정 로드
        config = {
            'log_directory': os.path.dirname(sample_log_path),
            'output_path': 'test_output.json',
            'sessionization_method': 'sliding_window',
            'window_size': 5,  # 작은 윈도우로 테스트
            'window_time': 60,
            'max_seq_length': 128,
            'drain3_config_path': 'drain3_config.yaml'
        }
        
        print("\n2. 전처리 파이프라인 초기화...")
        preprocessor = LogPreprocessor(config)
        
        print("\n3. 로그 파일 처리 중...")
        sessions = preprocessor.process_log_file(sample_log_path)
        
        print(f"\n4. 처리 결과:")
        print(f"   - 생성된 세션 수: {len(sessions)}")
        print(f"   - 발견된 고유 이벤트 수: {preprocessor.parser.get_event_id_count()}")
        
        if sessions:
            print(f"\n5. 첫 번째 세션 예시:")
            first_session = sessions[0]
            print(f"   - Session ID: {first_session.get('session_id')}")
            print(f"   - Event Sequence: {first_session.get('event_sequence', [])[:10]}...")
            print(f"   - Service Name: {first_session.get('service_name')}")
            print(f"   - Has Error: {first_session.get('has_error')}")
            print(f"   - Has Warn: {first_session.get('has_warn')}")
            print(f"   - Token IDs 길이: {len(first_session.get('token_ids', []))}")
            print(f"   - Simplified Text: {first_session.get('simplified_text', '')[:100]}...")
        
        # 결과 저장
        print("\n6. 결과 저장 중...")
        preprocessor.save_results(sessions, config['output_path'])
        print(f"   저장 완료: {config['output_path']}")
        
        print("\n" + "=" * 60)
        print("테스트 완료!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 임시 파일 정리
        if os.path.exists(sample_log_path):
            os.remove(sample_log_path)
            print(f"\n임시 파일 삭제: {sample_log_path}")

if __name__ == "__main__":
    test_preprocessing()

