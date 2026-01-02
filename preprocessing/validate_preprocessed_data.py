#!/usr/bin/env python3
"""
전처리된 데이터 검증 스크립트

전처리된 JSON 파일들의 품질을 검증합니다:
1. 파일 존재 여부 및 JSON 형식 검증
2. 데이터 구조 검증
3. 통계 정보 수집
4. 샘플 데이터 확인
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import sys


class PreprocessedDataValidator:
    """전처리된 데이터 검증 클래스"""
    
    REQUIRED_FIELDS = [
        'session_id',
        'event_sequence',
        'token_ids',
        'attention_mask',
        'has_error',
        'has_warn',
        'service_name',
        'original_logs',
        'simplified_text'
    ]
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.errors = []
        self.warnings = []
        self.stats = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'total_sessions': 0,
            'total_events': 0,
            'unique_event_ids': set(),
            'service_names': Counter(),
            'dates': [],
            'error_sessions': 0,
            'warn_sessions': 0,
            'file_sizes': {},
            'sample_sessions': []
        }
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """단일 파일 검증"""
        result = {
            'file': str(file_path),
            'valid': False,
            'errors': [],
            'warnings': [],
            'sessions': 0,
            'file_size_mb': 0
        }
        
        # 파일 존재 확인
        if not file_path.exists():
            result['errors'].append(f"파일이 존재하지 않습니다: {file_path}")
            return result
        
        # 파일 크기
        file_size = file_path.stat().st_size
        result['file_size_mb'] = file_size / (1024 * 1024)
        self.stats['file_sizes'][str(file_path)] = result['file_size_mb']
        
        # JSON 형식 검증
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            result['errors'].append(f"JSON 파싱 오류: {e}")
            return result
        except Exception as e:
            result['errors'].append(f"파일 읽기 오류: {e}")
            return result
        
        # 배열 형식 확인
        if not isinstance(data, list):
            result['errors'].append(f"데이터가 배열 형식이 아닙니다: {type(data)}")
            return result
        
        result['sessions'] = len(data)
        self.stats['total_sessions'] += len(data)
        
        # 각 세션 검증
        for idx, session in enumerate(data):
            session_errors = self._validate_session(session, idx)
            result['errors'].extend(session_errors)
            
            if not session_errors:
                # 통계 수집
                self._collect_stats(session)
        
        # 샘플 세션 저장 (처음 3개)
        if len(data) > 0 and len(self.stats['sample_sessions']) < 3:
            self.stats['sample_sessions'].append(data[0])
        
        result['valid'] = len(result['errors']) == 0
        return result
    
    def _validate_session(self, session: Dict[str, Any], index: int) -> List[str]:
        """단일 세션 검증"""
        errors = []
        
        # 필수 필드 확인
        for field in self.REQUIRED_FIELDS:
            if field not in session:
                errors.append(f"세션 {index}: 필수 필드 '{field}'가 없습니다")
        
        # 데이터 타입 검증
        if 'event_sequence' in session:
            if not isinstance(session['event_sequence'], list):
                errors.append(f"세션 {index}: 'event_sequence'가 리스트가 아닙니다")
            elif len(session['event_sequence']) == 0:
                errors.append(f"세션 {index}: 'event_sequence'가 비어있습니다")
        
        if 'token_ids' in session:
            if not isinstance(session['token_ids'], list):
                errors.append(f"세션 {index}: 'token_ids'가 리스트가 아닙니다")
            elif len(session['token_ids']) == 0:
                errors.append(f"세션 {index}: 'token_ids'가 비어있습니다")
        
        if 'attention_mask' in session:
            if not isinstance(session['attention_mask'], list):
                errors.append(f"세션 {index}: 'attention_mask'가 리스트가 아닙니다")
            elif 'token_ids' in session and len(session['attention_mask']) != len(session['token_ids']):
                errors.append(f"세션 {index}: 'attention_mask' 길이가 'token_ids'와 일치하지 않습니다")
        
        # 값 범위 검증
        if 'token_ids' in session and isinstance(session['token_ids'], list):
            if session['token_ids'][0] != 101:  # [CLS] 토큰
                errors.append(f"세션 {index}: 'token_ids'가 [CLS] 토큰(101)으로 시작하지 않습니다")
            if session['token_ids'][-1] != 102 and 102 in session['token_ids']:  # [SEP] 토큰
                sep_idx = session['token_ids'].index(102) if 102 in session['token_ids'] else -1
                if sep_idx < len(session['token_ids']) - 1:
                    # [SEP] 이후에만 패딩이 있어야 함
                    pass
        
        return errors
    
    def _collect_stats(self, session: Dict[str, Any]):
        """통계 정보 수집"""
        # Event ID 수집
        if 'event_sequence' in session:
            for event_id in session['event_sequence']:
                self.stats['unique_event_ids'].add(event_id)
                self.stats['total_events'] += 1
        
        # 서비스명 수집
        if 'service_name' in session:
            self.stats['service_names'][session['service_name']] += 1
        
        # 에러/경고 세션 수집
        if session.get('has_error', False):
            self.stats['error_sessions'] += 1
        if session.get('has_warn', False):
            self.stats['warn_sessions'] += 1
    
    def validate_all(self) -> Dict[str, Any]:
        """모든 전처리 파일 검증"""
        print("=" * 80)
        print("전처리된 데이터 검증 시작")
        print("=" * 80)
        print()
        
        # 출력 디렉토리 확인
        if not self.output_dir.exists():
            print(f"❌ 출력 디렉토리가 존재하지 않습니다: {self.output_dir}")
            return {'valid': False, 'error': 'Output directory not found'}
        
        # JSON 파일 찾기
        json_files = sorted(self.output_dir.glob("preprocessed_logs_*.json"))
        self.stats['total_files'] = len(json_files)
        
        if len(json_files) == 0:
            print(f"❌ 전처리된 파일을 찾을 수 없습니다: {self.output_dir}")
            return {'valid': False, 'error': 'No preprocessed files found'}
        
        print(f"발견된 파일 수: {len(json_files)}개")
        print()
        
        # 각 파일 검증
        results = []
        for json_file in json_files:
            print(f"검증 중: {json_file.name}...", end=' ', flush=True)
            result = self.validate_file(json_file)
            results.append(result)
            
            if result['valid']:
                print(f"✅ ({result['sessions']}개 세션, {result['file_size_mb']:.2f}MB)")
                self.stats['valid_files'] += 1
            else:
                print(f"❌ ({len(result['errors'])}개 오류)")
                self.stats['invalid_files'] += 1
                self.errors.extend(result['errors'])
        
        print()
        print("=" * 80)
        print("검증 결과 요약")
        print("=" * 80)
        print()
        
        # 통계 출력
        self._print_summary()
        
        # 샘플 데이터 출력
        if self.stats['sample_sessions']:
            print()
            print("=" * 80)
            print("샘플 세션 데이터 (첫 번째 파일의 첫 번째 세션)")
            print("=" * 80)
            sample = self.stats['sample_sessions'][0]
            print(json.dumps(sample, ensure_ascii=False, indent=2)[:1000] + "...")
        
        return {
            'valid': self.stats['invalid_files'] == 0,
            'stats': self.stats,
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    def _print_summary(self):
        """통계 요약 출력"""
        print(f"📁 총 파일 수: {self.stats['total_files']}개")
        print(f"   ✅ 유효한 파일: {self.stats['valid_files']}개")
        print(f"   ❌ 무효한 파일: {self.stats['invalid_files']}개")
        print()
        
        print(f"📊 총 세션 수: {self.stats['total_sessions']:,}개")
        print(f"   ⚠️  에러 포함 세션: {self.stats['error_sessions']:,}개 ({self.stats['error_sessions']/max(self.stats['total_sessions'],1)*100:.1f}%)")
        print(f"   ⚠️  경고 포함 세션: {self.stats['warn_sessions']:,}개 ({self.stats['warn_sessions']/max(self.stats['total_sessions'],1)*100:.1f}%)")
        print()
        
        print(f"🔢 총 이벤트 수: {self.stats['total_events']:,}개")
        print(f"   고유 Event ID 수: {len(self.stats['unique_event_ids']):,}개")
        print()
        
        print(f"🏷️  서비스별 세션 수:")
        for service, count in self.stats['service_names'].most_common(10):
            print(f"   - {service}: {count:,}개")
        if len(self.stats['service_names']) > 10:
            print(f"   ... 외 {len(self.stats['service_names']) - 10}개 서비스")
        print()
        
        # 파일 크기 통계
        if self.stats['file_sizes']:
            total_size = sum(self.stats['file_sizes'].values())
            avg_size = total_size / len(self.stats['file_sizes'])
            max_size = max(self.stats['file_sizes'].values())
            min_size = min(self.stats['file_sizes'].values())
            
            print(f"💾 파일 크기 통계:")
            print(f"   총 크기: {total_size:.2f}MB")
            print(f"   평균 크기: {avg_size:.2f}MB")
            print(f"   최대 크기: {max_size:.2f}MB")
            print(f"   최소 크기: {min_size:.2f}MB")
            print()
        
        # 오류 출력
        if self.errors:
            print("=" * 80)
            print("❌ 발견된 오류:")
            print("=" * 80)
            for error in self.errors[:20]:  # 최대 20개만 출력
                print(f"  - {error}")
            if len(self.errors) > 20:
                print(f"  ... 외 {len(self.errors) - 20}개 오류")
            print()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='전처리된 데이터 검증')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='전처리된 파일이 있는 디렉토리 (기본값: output)')
    parser.add_argument('--file', type=str, default=None,
                       help='특정 파일만 검증 (선택사항)')
    
    args = parser.parse_args()
    
    # 기본 경로 설정
    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir
    
    validator = PreprocessedDataValidator(output_dir=str(output_dir))
    
    if args.file:
        # 특정 파일만 검증
        file_path = output_dir / args.file
        result = validator.validate_file(file_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # 모든 파일 검증
        result = validator.validate_all()
        
        # 종료 코드
        sys.exit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()

