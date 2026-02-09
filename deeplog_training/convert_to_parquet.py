#!/usr/bin/env python3
"""
JSON 로그 데이터를 Parquet 형식으로 변환

목적:
- 학습 데이터 I/O 속도 10-20배 향상
- 저장 공간 3-5배 압축
- PyTorch DataLoader와의 더 나은 통합

주요 기능:
- ijson을 사용한 메모리 효율적인 스트리밍 읽기
- PyArrow를 사용한 고속 Parquet 작성
- 멀티프로세스 병렬 변환
- 체크포인트 및 재개 기능
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from multiprocessing import Pool, cpu_count
import glob

try:
    import ijson
except ImportError:
    print("ERROR: ijson이 설치되지 않았습니다.")
    print("설치 명령: pip install ijson")
    sys.exit(1)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow가 설치되지 않았습니다.")
    print("설치 명령: pip install pyarrow")
    sys.exit(1)

from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def convert_json_to_parquet(
    json_path: str,
    parquet_path: str,
    chunk_size: int = 10000,
    compression: str = 'snappy',
) -> Dict[str, Any]:
    """
    단일 JSON 파일을 Parquet로 변환

    Args:
        json_path: 입력 JSON 파일 경로
        parquet_path: 출력 Parquet 파일 경로
        chunk_size: 한 번에 처리할 샘플 수
        compression: 압축 알고리즘 (snappy, gzip, brotli, zstd)

    Returns:
        변환 통계 (샘플 수, 파일 크기, 소요 시간 등)
    """
    start_time = datetime.now()

    try:
        # Parquet 스키마 정의
        schema = pa.schema([
            ('session_id', pa.string()),
            ('token_ids', pa.list_(pa.int32())),
            ('attention_mask', pa.list_(pa.int8())),
        ])

        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

        total_samples = 0
        writer = None

        # 청크 버퍼
        session_ids = []
        token_ids_list = []
        attention_mask_list = []

        # JSON 파일 스트리밍 읽기
        with open(json_path, 'rb') as f:
            # ijson을 사용하여 메모리 효율적으로 파싱
            for session in ijson.items(f, 'item'):
                # 필수 필드 확인
                if 'token_ids' not in session:
                    continue

                # 데이터 추출
                session_id = session.get('session_id', f'session_{total_samples}')
                token_ids = session.get('token_ids', [])
                attention_mask = session.get('attention_mask', [1] * len(token_ids))

                # 버퍼에 추가
                session_ids.append(session_id)
                token_ids_list.append(token_ids)
                attention_mask_list.append(attention_mask)

                total_samples += 1

                # 청크 크기에 도달하면 쓰기
                if len(session_ids) >= chunk_size:
                    table = pa.table({
                        'session_id': session_ids,
                        'token_ids': token_ids_list,
                        'attention_mask': attention_mask_list,
                    }, schema=schema)

                    if writer is None:
                        writer = pq.ParquetWriter(
                            parquet_path,
                            schema,
                            compression=compression,
                            version='2.6',
                        )

                    writer.write_table(table)

                    # 버퍼 초기화
                    session_ids = []
                    token_ids_list = []
                    attention_mask_list = []

        # 남은 데이터 쓰기
        if session_ids:
            table = pa.table({
                'session_id': session_ids,
                'token_ids': token_ids_list,
                'attention_mask': attention_mask_list,
            }, schema=schema)

            if writer is None:
                writer = pq.ParquetWriter(
                    parquet_path,
                    schema,
                    compression=compression,
                    version='2.6',
                )

            writer.write_table(table)

        # Writer 종료
        if writer is not None:
            writer.close()

        # 통계 계산
        elapsed = (datetime.now() - start_time).total_seconds()
        json_size = os.path.getsize(json_path)
        parquet_size = os.path.getsize(parquet_path) if os.path.exists(parquet_path) else 0
        compression_ratio = json_size / parquet_size if parquet_size > 0 else 0

        return {
            'success': True,
            'json_path': json_path,
            'parquet_path': parquet_path,
            'total_samples': total_samples,
            'json_size_mb': json_size / (1024 ** 2),
            'parquet_size_mb': parquet_size / (1024 ** 2),
            'compression_ratio': compression_ratio,
            'elapsed_seconds': elapsed,
            'samples_per_second': total_samples / elapsed if elapsed > 0 else 0,
        }

    except Exception as e:
        logger.error(f"변환 실패: {json_path} - {e}")
        return {
            'success': False,
            'json_path': json_path,
            'error': str(e),
        }


def convert_file_wrapper(args):
    """멀티프로세스용 래퍼 함수"""
    json_path, parquet_path, chunk_size, compression = args
    return convert_json_to_parquet(json_path, parquet_path, chunk_size, compression)


def batch_convert(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.json",
    num_workers: int = 4,
    chunk_size: int = 10000,
    compression: str = 'snappy',
    resume: bool = True,
) -> Dict[str, Any]:
    """
    여러 JSON 파일을 병렬로 Parquet로 변환

    Args:
        input_dir: 입력 디렉토리
        output_dir: 출력 디렉토리
        pattern: 파일 패턴 (예: "preprocessed_logs_*.json")
        num_workers: 병렬 워커 수
        chunk_size: 청크 크기
        compression: 압축 알고리즘
        resume: 이미 변환된 파일 건너뛰기

    Returns:
        전체 변환 통계
    """
    logger.info("=" * 80)
    logger.info("JSON → Parquet 배치 변환 시작")
    logger.info("=" * 80)

    # 입력 파일 찾기
    input_pattern = os.path.join(input_dir, pattern)
    json_files = sorted(glob.glob(input_pattern))

    if not json_files:
        logger.error(f"입력 파일을 찾을 수 없습니다: {input_pattern}")
        return {'success': False, 'error': 'No input files found'}

    logger.info(f"입력 디렉토리: {input_dir}")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"발견된 파일 수: {len(json_files)}")
    logger.info(f"병렬 워커 수: {num_workers}")
    logger.info(f"청크 크기: {chunk_size:,}")
    logger.info(f"압축 방식: {compression}")

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 변환 작업 준비
    tasks = []
    for json_path in json_files:
        filename = os.path.basename(json_path)
        parquet_filename = filename.replace('.json', '.parquet')
        parquet_path = os.path.join(output_dir, parquet_filename)

        # 재개 모드: 이미 존재하는 파일 건너뛰기
        if resume and os.path.exists(parquet_path):
            logger.info(f"⏭️  건너뛰기 (이미 존재): {parquet_filename}")
            continue

        tasks.append((json_path, parquet_path, chunk_size, compression))

    if not tasks:
        logger.info("변환할 파일이 없습니다 (모든 파일이 이미 변환됨)")
        return {'success': True, 'converted': 0, 'skipped': len(json_files)}

    logger.info(f"\n변환할 파일 수: {len(tasks)}")
    logger.info("-" * 80)

    # 병렬 변환 실행
    start_time = datetime.now()

    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(convert_file_wrapper, tasks),
                total=len(tasks),
                desc="변환 진행",
                unit="파일"
            ))
    else:
        # 단일 프로세스 (디버깅용)
        results = []
        for task in tqdm(tasks, desc="변환 진행", unit="파일"):
            results.append(convert_file_wrapper(task))

    # 통계 집계
    total_samples = 0
    total_json_size = 0
    total_parquet_size = 0
    success_count = 0
    fail_count = 0

    for result in results:
        if result['success']:
            success_count += 1
            total_samples += result['total_samples']
            total_json_size += result['json_size_mb']
            total_parquet_size += result['parquet_size_mb']
        else:
            fail_count += 1
            logger.error(f"실패: {result['json_path']} - {result.get('error', 'Unknown error')}")

    elapsed = (datetime.now() - start_time).total_seconds()

    # 결과 출력
    logger.info("\n" + "=" * 80)
    logger.info("변환 완료!")
    logger.info("=" * 80)
    logger.info(f"총 파일 수: {len(tasks)}")
    logger.info(f"성공: {success_count} | 실패: {fail_count}")
    logger.info(f"총 샘플 수: {total_samples:,}")
    logger.info(f"JSON 크기: {total_json_size:.2f} MB")
    logger.info(f"Parquet 크기: {total_parquet_size:.2f} MB")
    logger.info(f"압축률: {total_json_size / total_parquet_size:.2f}x" if total_parquet_size > 0 else "N/A")
    logger.info(f"소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    logger.info(f"처리 속도: {total_samples / elapsed:.0f} samples/sec" if elapsed > 0 else "N/A")
    logger.info("=" * 80)

    return {
        'success': True,
        'total_files': len(tasks),
        'success_count': success_count,
        'fail_count': fail_count,
        'total_samples': total_samples,
        'total_json_size_mb': total_json_size,
        'total_parquet_size_mb': total_parquet_size,
        'compression_ratio': total_json_size / total_parquet_size if total_parquet_size > 0 else 0,
        'elapsed_seconds': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description='JSON 로그 파일을 Parquet 형식으로 변환'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='입력 JSON 디렉토리'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='출력 Parquet 디렉토리'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.json',
        help='파일 패턴 (기본값: *.json)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=max(1, cpu_count() // 2),
        help=f'병렬 워커 수 (기본값: {max(1, cpu_count() // 2)})'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        help='청크 크기 (기본값: 10000)'
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='snappy',
        choices=['snappy', 'gzip', 'brotli', 'zstd', 'none'],
        help='압축 알고리즘 (기본값: snappy)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='이미 변환된 파일도 다시 변환'
    )

    args = parser.parse_args()

    # 변환 실행
    result = batch_convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        compression=args.compression if args.compression != 'none' else None,
        resume=not args.no_resume,
    )

    if result['success']:
        logger.info("\n✅ 모든 변환이 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        logger.error("\n❌ 변환 중 오류가 발생했습니다.")
        sys.exit(1)


if __name__ == '__main__':
    main()
