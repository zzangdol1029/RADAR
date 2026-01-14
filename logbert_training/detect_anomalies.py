#!/usr/bin/env python3
"""
실제 로그 파일에서 이상 탐지 수행
학습된 LogBERT 모델을 사용하여 로그 파일의 이상을 탐지합니다.
"""

import torch
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from tqdm import tqdm
import sys
import re
from collections import defaultdict

# 전처리 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent / 'preprocessing'))
from log_preprocessor import LogParser, Sessionizer, LogEncoder, LogCleaner

from model import LogBERT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """이상 탐지 클래스"""
    
    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: Optional[str] = None,
        device: str = None,
        max_seq_length: int = 512
    ):
        """
        Args:
            checkpoint_path: 학습된 모델 체크포인트 경로
            vocab_path: 학습 시 사용한 vocab 파일 경로 (선택사항)
            device: 사용할 디바이스 ('cuda' or 'cpu')
            max_seq_length: 최대 시퀀스 길이
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"디바이스: {self.device}")
        
        # 체크포인트 로드
        logger.info(f"모델 로드 중: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # 모델 생성
        self.model = LogBERT(**config['model'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"모델 로드 완료")
        logger.info(f"  - Vocab Size: {config['model']['vocab_size']}")
        logger.info(f"  - Hidden Size: {config['model']['hidden_size']}")
        logger.info(f"  - Max Seq Length: {max_seq_length}")
        
        # Vocab 로드 (학습 시 사용한 vocab)
        self.vocab_size = config['model']['vocab_size']
        self.max_seq_length = max_seq_length
        
        # 전처리 컴포넌트 초기화
        self.log_cleaner = LogCleaner()
        self.log_parser = LogParser()
        self.sessionizer = Sessionizer(window_size=20, max_gap_seconds=300)
        
        # Vocab 로드 시도
        self.event_to_token = {}
        self.token_to_event = {}
        if vocab_path and Path(vocab_path).exists():
            self._load_vocab(vocab_path)
        else:
            logger.warning("Vocab 파일을 찾을 수 없습니다. 새로 생성합니다.")
            logger.warning("학습 시 사용한 vocab과 다를 수 있습니다.")
        
        # 인코더 초기화
        self.encoder = LogEncoder(max_seq_length=max_seq_length)
        
        # 학습 시 사용한 vocab이 있으면 적용
        if self.event_to_token:
            self.encoder.event_to_id = self.event_to_token
            self.encoder.id_to_event = self.token_to_event
            self.encoder.next_token_id = max(self.event_to_token.values()) + 1 if self.event_to_token else 1
    
    def _load_vocab(self, vocab_path: str):
        """학습 시 사용한 vocab 로드"""
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            if isinstance(vocab_data, dict):
                # event_to_token 형식
                self.event_to_token = {int(k): int(v) for k, v in vocab_data.items()}
                self.token_to_event = {v: k for k, v in self.event_to_token.items()}
            elif isinstance(vocab_data, list):
                # 리스트 형식 (event_id가 인덱스)
                self.event_to_token = {i: i+1 for i in range(len(vocab_data))}
                self.token_to_event = {i+1: i for i in range(len(vocab_data))}
            
            logger.info(f"Vocab 로드 완료: {len(self.event_to_token)}개 이벤트")
        except Exception as e:
            logger.warning(f"Vocab 로드 실패: {e}")
            logger.warning("새로운 vocab을 생성합니다.")
    
    def process_log_file(self, log_file_path: Path) -> List[Dict[str, Any]]:
        """
        로그 파일을 읽고 전처리하여 세션 리스트 반환
        
        Args:
            log_file_path: 로그 파일 경로
        
        Returns:
            전처리된 세션 리스트
        """
        logger.info(f"로그 파일 처리 중: {log_file_path}")
        
        sessions = []
        parsed_logs = []
        line_count = 0
        
        # 파일명에서 서비스명 추출
        service_name = log_file_path.stem.split('_')[0] if '_' in log_file_path.stem else 'unknown'
        
        # 로그 파일 읽기
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                
                # 로그 정리
                cleaned_line = self.log_cleaner.clean_log_line(line.strip())
                if not cleaned_line:
                    continue
                
                # 로그 파싱
                parsed = self.log_parser.parse_log(cleaned_line)
                if parsed:
                    parsed['line_number'] = line_num
                    parsed['service_name'] = service_name
                    parsed_logs.append(parsed)
        
        logger.info(f"  - 총 라인 수: {line_count:,}")
        logger.info(f"  - 파싱된 로그: {len(parsed_logs):,}")
        
        # 세션화
        for parsed_log in parsed_logs:
            self.sessionizer.add_log(parsed_log)
        
        # 세션 추출
        file_sessions = self.sessionizer.flush_sessions()
        logger.info(f"  - 생성된 세션: {len(file_sessions):,}개")
        
        # 세션 인코딩
        for session_idx, session in enumerate(file_sessions):
            # 메타데이터 추출
            has_error = any('ERROR' in str(log.get('template', '')) or 
                          'error' in str(log.get('original', '')).lower() 
                          for log in session)
            has_warn = any('WARN' in str(log.get('template', '')) or 
                          'warn' in str(log.get('original', '')).lower() 
                          for log in session)
            
            # 인코딩
            encoded = self.encoder.encode_sequence(session)
            
            # 세션 정보 구성
            session_data = {
                'session_id': f"{service_name}_{log_file_path.stem}_{session_idx}",
                'token_ids': encoded['token_ids'],
                'attention_mask': encoded['attention_mask'],
                'event_ids': encoded.get('event_ids', []),
                'has_error': has_error,
                'has_warn': has_warn,
                'service_name': service_name,
                'file_name': log_file_path.name,
                'session_length': len(session),
                'original_logs': [log.get('original', '') for log in session[:5]]  # 처음 5개만
            }
            
            sessions.append(session_data)
        
        return sessions
    
    def detect_anomalies(
        self,
        sessions: List[Dict[str, Any]],
        batch_size: int = 32,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        세션 리스트에서 이상 탐지 수행
        
        Args:
            sessions: 전처리된 세션 리스트
            batch_size: 배치 크기
            threshold: 이상 임계값 (None이면 자동 계산)
        
        Returns:
            이상 탐지 결과 리스트
        """
        logger.info(f"이상 탐지 수행 중: {len(sessions)}개 세션")
        
        # 임계값 자동 계산 (없는 경우)
        if threshold is None:
            # 먼저 모든 점수 계산
            all_scores = []
            for i in tqdm(range(0, len(sessions), batch_size), desc="점수 계산 중"):
                batch = sessions[i:i+batch_size]
                scores = self._predict_batch(batch)
                all_scores.extend(scores)
            
            # 통계 기반 임계값 계산 (평균 + 2*표준편차)
            scores_array = np.array(all_scores)
            threshold = float(scores_array.mean() + 2 * scores_array.std())
            logger.info(f"자동 계산된 임계값: {threshold:.4f}")
            logger.info(f"  - 평균: {scores_array.mean():.4f}")
            logger.info(f"  - 표준편차: {scores_array.std():.4f}")
        
        # 배치 단위로 이상 점수 계산
        results = []
        for i in tqdm(range(0, len(sessions), batch_size), desc="이상 탐지 중"):
            batch = sessions[i:i+batch_size]
            scores = self._predict_batch(batch)
            
            for session, score in zip(batch, scores):
                result = {
                    **session,
                    'anomaly_score': float(score),
                    'is_anomaly': score >= threshold,
                    'threshold': threshold
                }
                results.append(result)
        
        return results
    
    def _predict_batch(self, sessions: List[Dict[str, Any]]) -> List[float]:
        """배치 단위로 이상 점수 예측"""
        # 배치 구성
        max_len = max(len(s['token_ids']) for s in sessions)
        max_len = min(max_len, self.max_seq_length)  # 최대 길이 제한
        
        input_ids_list = []
        attention_mask_list = []
        
        for session in sessions:
            token_ids = session['token_ids'][:max_len]
            attention_mask = session['attention_mask'][:max_len]
            
            # 패딩
            if len(token_ids) < max_len:
                padding_len = max_len - len(token_ids)
                token_ids = token_ids + [0] * padding_len
                attention_mask = attention_mask + [0] * padding_len
            
            input_ids_list.append(token_ids)
            attention_mask_list.append(attention_mask)
        
        # Tensor로 변환
        input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long).to(self.device)
        
        # 추론
        with torch.no_grad():
            batch_scores = self.model.predict_anomaly_score(input_ids, attention_mask)
        
        return batch_scores.cpu().tolist()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='로그 파일 이상 탐지')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='학습된 모델 체크포인트 경로')
    parser.add_argument('--log-dir', type=str, required=True,
                       help='로그 파일 디렉토리')
    parser.add_argument('--output', type=str, default='anomaly_results.json',
                       help='결과 출력 파일 경로')
    parser.add_argument('--threshold', type=float, default=None,
                       help='이상 임계값 (None이면 자동 계산)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--device', type=str, default=None,
                       help='디바이스 (cuda or cpu)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='처리할 최대 파일 수 (테스트용)')
    parser.add_argument('--vocab', type=str, default=None,
                       help='학습 시 사용한 vocab 파일 경로')
    
    args = parser.parse_args()
    
    # 로그 디렉토리 확인
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        logger.error(f"로그 디렉토리를 찾을 수 없습니다: {log_dir}")
        return
    
    # 로그 파일 찾기
    log_files = list(log_dir.glob('*.log'))
    if not log_files:
        logger.error(f"로그 파일을 찾을 수 없습니다: {log_dir}")
        return
    
    if args.max_files:
        log_files = log_files[:args.max_files]
    
    logger.info(f"발견된 로그 파일: {len(log_files)}개")
    
    # 이상 탐지기 생성
    detector = AnomalyDetector(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device=args.device
    )
    
    # 모든 로그 파일 처리
    all_sessions = []
    for log_file in tqdm(log_files, desc="로그 파일 처리"):
        try:
            sessions = detector.process_log_file(log_file)
            all_sessions.extend(sessions)
        except Exception as e:
            logger.error(f"파일 처리 실패 ({log_file.name}): {e}")
            continue
    
    logger.info(f"총 세션 수: {len(all_sessions):,}개")
    
    if not all_sessions:
        logger.error("처리된 세션이 없습니다.")
        return
    
    # 이상 탐지 수행
    results = detector.detect_anomalies(
        sessions=all_sessions,
        batch_size=args.batch_size,
        threshold=args.threshold
    )
    
    # 통계 출력
    scores_array = np.array([r['anomaly_score'] for r in results])
    anomalies = [r for r in results if r['is_anomaly']]
    
    logger.info("=" * 80)
    logger.info("이상 탐지 결과 통계")
    logger.info("=" * 80)
    logger.info(f"총 세션 수: {len(results):,}")
    logger.info(f"이상 세션 수: {len(anomalies):,} ({len(anomalies)/len(results)*100:.2f}%)")
    logger.info(f"평균 이상 점수: {scores_array.mean():.4f}")
    logger.info(f"표준편차: {scores_array.std():.4f}")
    logger.info(f"최소값: {scores_array.min():.4f}")
    logger.info(f"최대값: {scores_array.max():.4f}")
    logger.info(f"중앙값: {np.median(scores_array):.4f}")
    logger.info(f"사용된 임계값: {results[0]['threshold']:.4f}")
    
    # 서비스별 통계
    service_stats = defaultdict(lambda: {'total': 0, 'anomalies': 0, 'scores': []})
    for result in results:
        service = result['service_name']
        service_stats[service]['total'] += 1
        service_stats[service]['scores'].append(result['anomaly_score'])
        if result['is_anomaly']:
            service_stats[service]['anomalies'] += 1
    
    logger.info("\n서비스별 통계:")
    for service, stats in sorted(service_stats.items()):
        avg_score = np.mean(stats['scores'])
        anomaly_rate = stats['anomalies'] / stats['total'] * 100
        logger.info(
            f"  {service}: "
            f"총 {stats['total']}개, "
            f"이상 {stats['anomalies']}개 ({anomaly_rate:.2f}%), "
            f"평균 점수 {avg_score:.4f}"
        )
    
    # 상위 이상 세션 출력
    logger.info("\n상위 10개 이상 점수:")
    sorted_results = sorted(results, key=lambda x: x['anomaly_score'], reverse=True)
    for i, result in enumerate(sorted_results[:10], 1):
        logger.info(
            f"{i}. {result['session_id']}: "
            f"Score={result['anomaly_score']:.4f}, "
            f"Service={result['service_name']}, "
            f"Error={result['has_error']}, "
            f"Warn={result['has_warn']}, "
            f"Length={result['session_length']}"
        )
    
    # 결과 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # JSON으로 저장 (일부 필드만)
    output_data = {
        'summary': {
            'total_sessions': len(results),
            'anomaly_sessions': len(anomalies),
            'anomaly_rate': len(anomalies) / len(results) * 100,
            'threshold': results[0]['threshold'],
            'statistics': {
                'mean': float(scores_array.mean()),
                'std': float(scores_array.std()),
                'min': float(scores_array.min()),
                'max': float(scores_array.max()),
                'median': float(np.median(scores_array))
            }
        },
        'results': [
            {
                'session_id': r['session_id'],
                'anomaly_score': r['anomaly_score'],
                'is_anomaly': r['is_anomaly'],
                'service_name': r['service_name'],
                'file_name': r['file_name'],
                'has_error': r['has_error'],
                'has_warn': r['has_warn'],
                'session_length': r['session_length']
            }
            for r in results
        ],
        'top_anomalies': [
            {
                'session_id': r['session_id'],
                'anomaly_score': r['anomaly_score'],
                'service_name': r['service_name'],
                'file_name': r['file_name'],
                'has_error': r['has_error'],
                'has_warn': r['has_warn'],
                'original_logs': r.get('original_logs', [])
            }
            for r in sorted_results[:50]  # 상위 50개
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n결과 저장: {output_path}")
    logger.info(f"  - 총 세션: {len(results):,}개")
    logger.info(f"  - 이상 세션: {len(anomalies):,}개")
    logger.info(f"  - 상위 50개 이상 세션 상세 정보 포함")


if __name__ == '__main__':
    main()
