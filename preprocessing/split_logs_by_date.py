"""
로그 파일을 날짜별로 분리하는 스크립트

하나의 로그 파일에 여러 날짜의 로그가 있을 때, 날짜별로 분리하여 저장합니다.
"""

import os
import re
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LogDateSplitter:
    """로그 파일을 날짜별로 분리하는 클래스"""
    
    @staticmethod
    def extract_timestamp(line: str) -> Optional[datetime]:
        """로그 라인에서 타임스탬프 추출"""
        # Spring Boot 로그 형식: 2025-12-08 17:23:47.950
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.?\d*)'
        match = re.search(timestamp_pattern, line)
        if match:
            try:
                ts_str = match.group(1)
                # 밀리초 처리
                if '.' in ts_str:
                    return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
                else:
                    return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
            except:
                pass
        return None
    
    @staticmethod
    def is_valid_log_line(line: str) -> bool:
        """유효한 로그 라인인지 확인"""
        line = line.strip()
        if not line:
            return False
        
        # Spring Boot 배너 제거
        banner_patterns = [
            r'\.\s+____.*?Spring Boot.*?::',
            r':: Spring Boot ::',
            r'-----------------------------------------------------------------------------------------',
            r':: File Module Info ::',
        ]
        
        for pattern in banner_patterns:
            if re.search(pattern, line, re.DOTALL):
                return False
        
        return True
    
    def split_log_file(self, log_file_path: str, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        로그 파일을 날짜별로 분리
        
        Args:
            log_file_path: 입력 로그 파일 경로
            output_dir: 출력 디렉토리 (None이면 입력 파일과 같은 디렉토리)
        
        Returns:
            {날짜: 출력파일경로} 딕셔너리
        """
        log_file = Path(log_file_path)
        if not log_file.exists():
            raise FileNotFoundError(f"로그 파일을 찾을 수 없습니다: {log_file_path}")
        
        if output_dir is None:
            output_dir = log_file.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"로그 파일 분리 시작: {log_file.name}")
        
        # 날짜별로 로그 라인 수집
        date_logs = defaultdict(list)  # {date: [lines]}
        line_count = 0
        processed_count = 0
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_count += 1
                
                if not self.is_valid_log_line(line):
                    continue
                
                timestamp = self.extract_timestamp(line)
                if timestamp:
                    date_key = timestamp.strftime('%Y-%m-%d')
                    date_logs[date_key].append(line)
                    processed_count += 1
                else:
                    # 타임스탬프가 없는 라인은 마지막 날짜에 추가
                    if date_logs:
                        last_date = sorted(date_logs.keys())[-1]
                        date_logs[last_date].append(line)
        
        logger.info(f"  전체 {line_count}줄 중 {processed_count}줄 처리")
        logger.info(f"  발견된 날짜: {sorted(date_logs.keys())}")
        
        # 날짜별로 파일 저장
        # 파일명 형식: service_YYYY-MM-DD.log (프로그램명_날짜.log)
        output_files = {}
        
        # 서비스명 추출 (파일명에서)
        # 예: user_250822_15:16:45.log -> user
        service_name = log_file.stem.split('_')[0] if '_' in log_file.stem else log_file.stem
        
        for date in sorted(date_logs.keys()):
            # 출력 파일명: service_YYYY-MM-DD.log
            output_file = output_dir / f"{service_name}_{date}.log"
            
            # 같은 날짜의 다른 파일이 이미 있으면 append 모드로 추가
            mode = 'a' if output_file.exists() else 'w'
            
            with open(output_file, mode, encoding='utf-8') as f:
                if mode == 'a':
                    # 기존 파일에 추가할 때 구분선 추가
                    f.write(f"\n# === Merged from {log_file.name} ===\n")
                f.writelines(date_logs[date])
            
            output_files[date] = str(output_file)
            logger.info(f"  {date}: {len(date_logs[date])}줄 → {output_file.name} ({mode} 모드)")
        
        logger.info(f"분리 완료: {len(output_files)}개 파일 생성")
        return output_files
    
    def split_log_directory(self, log_dir: str, output_dir: Optional[str] = None, 
                           clean_output: bool = True) -> Dict[str, Dict[str, str]]:
        """
        로그 디렉토리의 모든 파일을 날짜별로 분리
        
        Args:
            log_dir: 로그 디렉토리 경로
            output_dir: 출력 디렉토리 (None이면 log_dir/date_split)
            clean_output: 출력 디렉토리를 먼저 정리할지 여부
        
        Returns:
            {파일명: {날짜: 출력파일경로}} 딕셔너리
        """
        log_dir_path = Path(log_dir)
        if not log_dir_path.exists():
            raise FileNotFoundError(f"로그 디렉토리를 찾을 수 없습니다: {log_dir}")
        
        if output_dir is None:
            output_dir = log_dir_path / "date_split"
        else:
            output_dir = Path(output_dir)
        
        # 출력 디렉토리 정리
        if clean_output and output_dir.exists():
            logger.info(f"기존 출력 디렉토리 정리: {output_dir}")
            import shutil
            shutil.rmtree(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"로그 디렉토리 분리 시작: {log_dir}")
        logger.info(f"출력 디렉토리: {output_dir}")
        logger.info(f"출력 파일 형식: service_YYYY-MM-DD.log")
        
        log_files = list(log_dir_path.glob("*.log"))
        log_files = [f for f in log_files if not f.name.startswith('.')]
        # 이미 분리된 파일은 제외 (service_YYYY-MM-DD.log 형식)
        log_files = [f for f in log_files if not re.match(r'^[a-z]+_\d{4}-\d{2}-\d{2}\.log$', f.name)]
        
        logger.info(f"처리할 파일 수: {len(log_files)}개")
        
        all_results = {}
        
        for log_file in log_files:
            try:
                # 각 파일을 날짜별로 분리
                output_files = self.split_log_file(str(log_file), str(output_dir))
                all_results[log_file.name] = output_files
            except Exception as e:
                logger.error(f"  {log_file.name} 처리 중 오류: {e}")
                continue
        
        # 생성된 파일 요약
        date_files = defaultdict(list)
        for file_results in all_results.values():
            for date, file_path in file_results.items():
                date_files[date].append(file_path)
        
        logger.info(f"\n전체 분리 완료: {len(all_results)}개 파일 처리")
        logger.info(f"생성된 날짜별 파일:")
        for date in sorted(date_files.keys()):
            logger.info(f"  {date}: {len(date_files[date])}개 파일 → {date_files[date][0].split('/')[-1]}")
        
        return all_results


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='로그 파일을 날짜별로 분리')
    parser.add_argument('--input', type=str, required=True,
                       help='입력 로그 파일 또는 디렉토리 경로')
    parser.add_argument('--output', type=str, default=None,
                       help='출력 디렉토리 경로 (기본: 입력과 같은 디렉토리 또는 date_split)')
    
    args = parser.parse_args()
    
    splitter = LogDateSplitter()
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 단일 파일 처리
        output_files = splitter.split_log_file(str(input_path), args.output)
        print(f"\n생성된 파일:")
        for date, file_path in output_files.items():
            print(f"  {date}: {file_path}")
    elif input_path.is_dir():
        # 디렉토리 처리
        results = splitter.split_log_directory(str(input_path), args.output)
        print(f"\n처리 완료: {len(results)}개 파일")
        print(f"출력 디렉토리: {args.output or str(input_path / 'date_split')}")
    else:
        print(f"오류: 입력 경로를 찾을 수 없습니다: {args.input}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

