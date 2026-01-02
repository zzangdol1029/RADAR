"""
로그 전처리 파이프라인
LogBERT 및 RAG 시스템을 위한 로그 데이터 가공 모듈

주요 기능:
1. 로그 파싱 (Drain3 기반)
2. 세션화 (Trace ID 또는 Sliding Window)
3. 데이터 인코딩 및 토큰화
4. RAG를 위한 메타데이터 결합
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, deque
from pathlib import Path
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    DRAIN3_AVAILABLE = True
except ImportError:
    DRAIN3_AVAILABLE = False
    print("경고: drain3가 설치되지 않았습니다. pip install drain3를 실행하세요.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 이 함수는 클래스 정의 이후에 배치되어야 합니다 (아래로 이동)


class LogCleaner:
    """로그 정리 클래스 - 의미 없는 데이터 제거"""
    
    # Spring Boot 배너 패턴
    BANNER_PATTERNS = [
        r'\.\s+____.*?Spring Boot.*?::',
        r':: Spring Boot ::',
        r'-----------------------------------------------------------------------------------------',
        r':: File Module Info ::',
    ]
    
    @staticmethod
    def clean_log_line(line: str) -> Optional[str]:
        """로그 라인 정리"""
        line = line.strip()
        if not line:
            return None
        
        # 배너 제거
        for pattern in LogCleaner.BANNER_PATTERNS:
            if re.search(pattern, line, re.DOTALL):
                return None
        
        # 빈 줄 제거
        if not line or line.isspace():
            return None
        
        return line


class LogParser:
    """Drain3 기반 로그 파서"""
    
    def __init__(self, config_path: str = "drain3_config.yaml"):
        """Drain3 템플릿 마이너 초기화"""
        if not DRAIN3_AVAILABLE:
            raise ImportError("drain3가 설치되지 않았습니다. pip install drain3를 실행하세요.")
        
        self.config = TemplateMinerConfig()
        
        # 설정 파일이 있으면 로드, 없으면 기본값 사용
        if os.path.exists(config_path):
            try:
                # YAML 파일인 경우 직접 파싱하여 설정
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        yaml_config = yaml.safe_load(f)
                    
                    # YAML 설정을 Drain3 설정 객체에 직접 할당
                    if yaml_config:
                        if 'profiling_enabled' in yaml_config:
                            self.config.profiling_enabled = yaml_config['profiling_enabled']
                        if 'snapshot_interval_minutes' in yaml_config:
                            self.config.snapshot_interval_minutes = yaml_config['snapshot_interval_minutes']
                        if 'drain_sim_th' in yaml_config:
                            self.config.drain_sim_th = yaml_config['drain_sim_th']
                        if 'drain_depth' in yaml_config:
                            self.config.drain_depth = yaml_config['drain_depth']
                        if 'drain_max_children' in yaml_config:
                            self.config.drain_max_children = yaml_config['drain_max_children']
                        if 'drain_max_clusters' in yaml_config:
                            self.config.drain_max_clusters = yaml_config['drain_max_clusters']
                        if 'param_str' in yaml_config:
                            self.config.param_str = yaml_config['param_str']
                    
                    logger.info(f"Drain3 설정 파일 로드 (YAML): {config_path}")
                else:
                    # INI 형식 파일인 경우 기존 방식 사용
                    self.config.load(config_path)
                    logger.info(f"Drain3 설정 파일 로드 (INI): {config_path}")
            except Exception as e:
                logger.warning(f"Drain3 설정 파일 로드 실패, 기본값 사용: {e}")
                self._set_default_config()
        else:
            logger.info("Drain3 설정 파일이 없어 기본값 사용")
            self._set_default_config()
        
        self.template_miner = TemplateMiner(config=self.config)
        self.event_id_map = {}  # Template -> Event ID 매핑
        self.next_event_id = 1
    
    def _set_default_config(self):
        """기본 Drain3 설정"""
        self.config.profiling_enabled = False
        self.config.snapshot_interval_minutes = 1
        self.config.drain_sim_th = 0.4
        self.config.drain_depth = 4
        self.config.drain_max_children = 100
        self.config.drain_max_clusters = None
    
    def parse_log(self, log_line: str) -> Dict[str, Any]:
        """
        로그 라인 파싱
        
        Returns:
            {
                'template': str,      # 템플릿 (예: "ERROR [manager] Connection timeout from <*>")
                'event_id': int,      # 이벤트 ID
                'parameters': List[str],  # 추출된 파라미터
                'original': str       # 원본 로그
            }
        """
        result = self.template_miner.add_log_message(log_line)
        
        if result:
            # Drain3 결과에서 템플릿 추출
            # result가 딕셔너리인 경우와 객체인 경우 모두 처리
            if isinstance(result, dict):
                template = result.get('template_mined', log_line)
                parameters = result.get('parameter_list', [])
            else:
                # 객체인 경우
                if hasattr(result, 'get_template_mined'):
                    template = result.get_template_mined()
                elif hasattr(result, 'template_mined'):
                    template = result.template_mined
                else:
                    template = log_line
                
                # 파라미터 추출
                if hasattr(result, 'get_parameter_list'):
                    parameters = result.get_parameter_list()
                elif hasattr(result, 'parameter_list'):
                    parameters = result.parameter_list
                else:
                    parameters = []
            
            # Event ID 할당
            if template not in self.event_id_map:
                self.event_id_map[template] = self.next_event_id
                self.next_event_id += 1
            
            event_id = self.event_id_map[template]
            
            return {
                'template': template,
                'event_id': event_id,
                'parameters': parameters,
                'original': log_line
            }
        
        return {
            'template': log_line,  # 파싱 실패 시 원본 반환
            'event_id': 0,  # 알 수 없는 이벤트
            'parameters': [],
            'original': log_line
        }
    
    def get_event_id_count(self) -> int:
        """발견된 고유 이벤트 수 반환"""
        return len(self.event_id_map)


class SQLQueryProcessor:
    """SQL 쿼리 가공 클래스 - Hibernate 쿼리 간소화"""
    
    SQL_KEYWORDS = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']
    
    @staticmethod
    def simplify_sql(sql_text: str) -> str:
        """
        SQL 쿼리 간소화
        - 핵심 키워드와 테이블명만 추출
        """
        if not sql_text:
            return ""
        
        sql_upper = sql_text.upper().strip()
        
        # SQL 키워드 추출
        keywords_found = []
        for keyword in SQLQueryProcessor.SQL_KEYWORDS:
            if keyword in sql_upper:
                keywords_found.append(keyword)
        
        # 테이블명 추출 (FROM, JOIN, INTO, UPDATE 뒤)
        table_pattern = r'(?:FROM|JOIN|INTO|UPDATE)\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
        tables = re.findall(table_pattern, sql_upper, re.IGNORECASE)
        
        # 간소화된 SQL 생성
        simplified = " ".join(keywords_found)
        if tables:
            simplified += f" {', '.join(set(tables))}"
        
        return simplified if simplified else "SQL_QUERY"
    
    @staticmethod
    def is_sql_log(log_line: str) -> bool:
        """SQL 로그인지 확인"""
        return 'org.hibernate.SQL' in log_line or 'DEBUG org.hibernate.SQL' in log_line


class Sessionizer:
    """로그 세션화 클래스"""
    
    def __init__(self, method: str = "sliding_window", window_size: int = 20, window_time: int = 300):
        """
        Args:
            method: "trace_id" 또는 "sliding_window"
            window_size: Sliding Window의 로그 개수
            window_time: Sliding Window의 시간(초)
        """
        self.method = method
        self.window_size = window_size
        self.window_time = window_time
        self.trace_sessions = defaultdict(list)  # trace_id -> logs
        self.sliding_window = deque(maxlen=window_size)
        self.window_start_time = None
    
    def extract_trace_id(self, log_line: str) -> Optional[str]:
        """로그에서 Trace ID 추출"""
        # JSON 형식 로그에서 trace_id 추출 시도
        json_match = re.search(r'\{[^}]+\}', log_line)
        if json_match:
            try:
                json_data = json.loads(json_match.group())
                if 'trace_id' in json_data:
                    return json_data['trace_id']
                if 'client_ip' in json_data:
                    # client_ip를 trace_id 대신 사용
                    return json_data['client_ip']
            except:
                pass
        
        # HTTP 헤더나 다른 형식에서 추출 시도
        trace_patterns = [
            r'trace[_-]?id["\']?\s*[:=]\s*["\']?([a-zA-Z0-9-]+)',
            r'X-Trace-Id["\']?\s*[:=]\s*["\']?([a-zA-Z0-9-]+)',
        ]
        
        for pattern in trace_patterns:
            match = re.search(pattern, log_line, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def extract_timestamp(self, log_line: str) -> Optional[datetime]:
        """로그에서 타임스탬프 추출"""
        # Spring Boot 로그 형식: 2025-12-08 17:23:47.950
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.?\d*)'
        match = re.search(timestamp_pattern, log_line)
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
    
    def add_log(self, log_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        로그를 세션에 추가하고, 완성된 세션이 있으면 반환
        
        Returns:
            완성된 세션 리스트 (각 세션은 event_id 시퀀스)
        """
        completed_sessions = []
        
        if self.method == "trace_id":
            trace_id = self.extract_trace_id(log_data.get('original', ''))
            if trace_id:
                self.trace_sessions[trace_id].append(log_data)
            else:
                # Trace ID가 없으면 기본 세션에 추가
                self.trace_sessions['default'].append(log_data)
        
        elif self.method == "sliding_window":
            timestamp = self.extract_timestamp(log_data.get('original', ''))
            
            if timestamp:
                if self.window_start_time is None:
                    self.window_start_time = timestamp
                
                # 시간 윈도우 체크
                time_diff = (timestamp - self.window_start_time).total_seconds()
                if time_diff > self.window_time:
                    # 윈도우 완성
                    if len(self.sliding_window) > 0:
                        completed_sessions.append(list(self.sliding_window))
                    self.sliding_window.clear()
                    self.window_start_time = timestamp
                
                self.sliding_window.append(log_data)
                
                # 크기 윈도우 체크
                if len(self.sliding_window) >= self.window_size:
                    completed_sessions.append(list(self.sliding_window))
                    self.sliding_window.clear()
                    self.window_start_time = None
            else:
                # 타임스탬프가 없으면 크기만 체크
                self.sliding_window.append(log_data)
                if len(self.sliding_window) >= self.window_size:
                    completed_sessions.append(list(self.sliding_window))
                    self.sliding_window.clear()
        
        return completed_sessions
    
    def flush_sessions(self) -> List[List[Dict[str, Any]]]:
        """남은 세션들을 반환"""
        sessions = []
        
        if self.method == "trace_id":
            for trace_id, logs in self.trace_sessions.items():
                if len(logs) > 0:
                    sessions.append(logs)
            self.trace_sessions.clear()
        
        elif self.method == "sliding_window":
            if len(self.sliding_window) > 0:
                sessions.append(list(self.sliding_window))
            self.sliding_window.clear()
        
        return sessions


class LogEncoder:
    """로그 인코딩 및 토큰화 클래스"""
    
    # Special tokens
    CLS_TOKEN = 101  # [CLS]
    SEP_TOKEN = 102  # [SEP]
    MASK_TOKEN = 103  # [MASK]
    PAD_TOKEN = 0     # [PAD]
    UNK_TOKEN = 0     # 알 수 없는 이벤트
    
    def __init__(self, max_seq_length: int = 512):
        self.max_seq_length = max_seq_length
        self.event_to_id = {}  # Event ID -> Token ID 매핑
        self.id_to_event = {}  # Token ID -> Event ID 매핑
        self.next_token_id = 1  # 0은 PAD/UNK용
    
    def encode_sequence(self, session: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        이벤트 시퀀스를 토큰 ID 시퀀스로 변환
        
        Args:
            session: 세션화된 로그 리스트 (각 항목은 event_id 포함)
        
        Returns:
            {
                'token_ids': List[int],
                'event_ids': List[int],
                'attention_mask': List[int]
            }
        """
        # Event ID 추출
        event_ids = [log.get('event_id', 0) for log in session]
        
        # Token ID 매핑
        token_ids = []
        for event_id in event_ids:
            if event_id not in self.event_to_id:
                self.event_to_id[event_id] = self.next_token_id
                self.id_to_event[self.next_token_id] = event_id
                self.next_token_id += 1
            token_ids.append(self.event_to_id[event_id])
        
        # Special tokens 추가
        token_ids = [self.CLS_TOKEN] + token_ids + [self.SEP_TOKEN]
        
        # Padding
        attention_mask = [1] * len(token_ids)
        if len(token_ids) < self.max_seq_length:
            pad_length = self.max_seq_length - len(token_ids)
            token_ids.extend([self.PAD_TOKEN] * pad_length)
            attention_mask.extend([0] * pad_length)
        else:
            token_ids = token_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            token_ids[-1] = self.SEP_TOKEN  # 마지막을 SEP으로
        
        return {
            'token_ids': token_ids,
            'event_ids': event_ids,
            'attention_mask': attention_mask
        }


class MetadataEnricher:
    """메타데이터 결합 클래스"""
    
    LOG_LEVELS = ['ERROR', 'WARN', 'INFO', 'DEBUG', 'TRACE']
    
    @staticmethod
    def extract_log_level(log_line: str) -> str:
        """로그 레벨 추출"""
        for level in MetadataEnricher.LOG_LEVELS:
            if f' {level} ' in log_line or log_line.startswith(level):
                return level
        return 'UNKNOWN'
    
    @staticmethod
    def extract_service_name(log_file_path: str) -> str:
        """로그 파일 경로에서 서비스명 추출"""
        filename = os.path.basename(log_file_path)
        service_name = filename.replace('.log', '')
        return service_name
    
    @staticmethod
    def enrich_session(session: List[Dict[str, Any]], service_name: str) -> Dict[str, Any]:
        """
        세션에 메타데이터 추가
        
        Returns:
            {
                'event_sequence': List[int],
                'has_error': bool,
                'has_warn': bool,
                'service_name': str,
                'original_logs': List[str],
                'simplified_text': str  # RAG용 간소화된 텍스트
            }
        """
        has_error = False
        has_warn = False
        original_logs = []
        simplified_texts = []
        
        for log_data in session:
            original = log_data.get('original', '')
            original_logs.append(original)
            
            # 로그 레벨 체크
            log_level = MetadataEnricher.extract_log_level(original)
            if log_level == 'ERROR':
                has_error = True
            elif log_level == 'WARN':
                has_warn = True
            
            # SQL 쿼리 가공
            if SQLQueryProcessor.is_sql_log(original):
                simplified = SQLQueryProcessor.simplify_sql(original)
                simplified_texts.append(f"[SQL] {simplified}")
            else:
                # 일반 로그는 템플릿 사용
                template = log_data.get('template', original)
                simplified_texts.append(template)
        
        # RAG용 간소화된 텍스트 생성
        simplified_text = f"[{service_name}] " + " | ".join(simplified_texts[:10])  # 최대 10개만
        
        return {
            'event_sequence': [log.get('event_id', 0) for log in session],
            'has_error': has_error,
            'has_warn': has_warn,
            'service_name': service_name,
            'original_logs': original_logs,
            'simplified_text': simplified_text
        }


class LogPreprocessor:
    """메인 로그 전처리 파이프라인"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.cleaner = LogCleaner()
        self.parser = LogParser(config.get('drain3_config_path', 'drain3_config.yaml'))
        self.sessionizer = Sessionizer(
            method=config.get('sessionization_method', 'sliding_window'),
            window_size=config.get('window_size', 20),
            window_time=config.get('window_time', 300)
        )
        self.encoder = LogEncoder(max_seq_length=config.get('max_seq_length', 512))
        self.metadata_enricher = MetadataEnricher()
    
    def process_log_file(self, log_file_path: str, output_file: Optional[str] = None, 
                        stream_mode: bool = True) -> List[Dict[str, Any]]:
        """
        로그 파일 처리 (메모리 효율적 스트리밍 방식)
        
        Args:
            log_file_path: 로그 파일 경로
            output_file: 스트리밍 모드일 때 출력 파일 경로 (None이면 메모리에 저장)
            stream_mode: True면 세션 완성 시 즉시 파일에 저장, False면 메모리에 모두 저장
        
        Returns:
            전처리된 세션 리스트 (stream_mode=False일 때만 반환)
        """
        logger.info(f"로그 파일 처리 시작: {log_file_path}")
        service_name = MetadataEnricher.extract_service_name(log_file_path)
        
        processed_sessions = []
        line_count = 0
        session_count = 0
        
        # 스트리밍 모드: 파일에 직접 쓰기
        output_f = None
        if stream_mode and output_file:
            output_f = open(output_file, 'w', encoding='utf-8')
            output_f.write('[\n')  # JSON 배열 시작
            first_item = True
        
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line_count += 1
                    
                    # 로그 정리
                    cleaned_line = self.cleaner.clean_log_line(line)
                    if not cleaned_line:
                        continue
                    
                    # 로그 파싱
                    parsed_log = self.parser.parse_log(cleaned_line)
                    
                    # 타임스탬프와 서비스명 추가 (관계 추적용)
                    timestamp = self.sessionizer.extract_timestamp(cleaned_line)
                    parsed_log['timestamp'] = timestamp.isoformat() if timestamp else None
                    parsed_log['service_name'] = service_name
                    
                    # 세션화
                    completed_sessions = self.sessionizer.add_log(parsed_log)
                    
                    # 완성된 세션 처리
                    for session in completed_sessions:
                        # 메타데이터 결합
                        enriched = self.metadata_enricher.enrich_session(session, service_name)
                        
                        # 인코딩
                        encoded = self.encoder.encode_sequence(session)
                        
                        # 최종 결과 결합
                        final_result = {
                            **enriched,
                            **encoded,
                            'session_id': session_count
                        }
                        session_count += 1
                        
                        if stream_mode and output_f:
                            # 스트리밍: 즉시 파일에 저장
                            if not first_item:
                                output_f.write(',\n')
                            json.dump(final_result, output_f, ensure_ascii=False, indent=2)
                            first_item = False
                        else:
                            # 메모리 모드: 리스트에 저장
                            processed_sessions.append(final_result)
            
            # 남은 세션 처리
            remaining_sessions = self.sessionizer.flush_sessions()
            for session in remaining_sessions:
                enriched = self.metadata_enricher.enrich_session(session, service_name)
                encoded = self.encoder.encode_sequence(session)
                final_result = {
                    **enriched,
                    **encoded,
                    'session_id': session_count
                }
                session_count += 1
                
                if stream_mode and output_f:
                    if not first_item:
                        output_f.write(',\n')
                    json.dump(final_result, output_f, ensure_ascii=False, indent=2)
                    first_item = False
                else:
                    processed_sessions.append(final_result)
            
            if stream_mode and output_f:
                output_f.write('\n]')  # JSON 배열 종료
                output_f.close()
                logger.info(f"처리 완료: {line_count}줄, {session_count}개 세션 생성 (스트리밍 모드)")
            else:
                logger.info(f"처리 완료: {line_count}줄, {len(processed_sessions)}개 세션 생성")
            
            logger.info(f"발견된 고유 이벤트 수: {self.parser.get_event_id_count()}")
            
        except Exception as e:
            logger.error(f"로그 파일 처리 중 오류 발생: {e}")
            if output_f:
                output_f.close()
            raise
        
        return processed_sessions
    
    def process_log_directory(self, log_dir: str, output_file: Optional[str] = None,
                             stream_mode: bool = True, enable_correlation: bool = True,
                             date_filter: Optional[str] = None, batch_by_date: bool = True,
                             parallel: bool = False, max_workers: int = 8) -> List[Dict[str, Any]]:
        """
        로그 디렉토리 전체 처리 (날짜별 배치 처리 지원)
        
        Args:
            log_dir: 로그 디렉토리 경로
            output_file: 출력 파일 경로 (스트리밍 모드)
            stream_mode: 스트리밍 모드 사용 여부
            enable_correlation: 날짜/시스템명 기반 관계 추적 활성화
            date_filter: 날짜 필터 (YYYY-MM-DD 형식, None이면 전체)
            batch_by_date: 날짜별로 나누어 처리 (메모리 효율적)
        
        Returns:
            전처리된 세션 리스트
        """
        logger.info(f"로그 디렉토리 처리 시작: {log_dir}")
        if date_filter:
            logger.info(f"날짜 필터: {date_filter}")
        if batch_by_date:
            logger.info("날짜별 배치 처리 모드 활성화")
        
        log_dir_path = Path(log_dir)
        if not log_dir_path.exists():
            logger.error(f"로그 디렉토리를 찾을 수 없습니다: {log_dir}")
            logger.error(f"절대 경로: {log_dir_path.absolute()}")
            return []
        
        log_files = list(log_dir_path.glob("*.log"))
        log_files = [f for f in log_files if not f.name.startswith('.')]
        logger.info(f"발견된 로그 파일 수: {len(log_files)}개")
        
        if not log_files:
            logger.warning(f"로그 파일을 찾을 수 없습니다: {log_dir}")
            logger.warning(f"확인할 경로: {log_dir_path.absolute()}")
            return []
        
        # 날짜별 배치 처리
        if batch_by_date and enable_correlation:
            return self._process_by_date_batch(
                log_files, output_file, stream_mode, date_filter, parallel, max_workers
            )
        elif enable_correlation:
            # 관계 추적 모드 (메모리 집약적 - 비권장)
            logger.warning("관계 추적 모드가 메모리를 많이 사용합니다. batch_by_date=True를 권장합니다.")
            return self._process_with_correlation(log_files, output_file, stream_mode)
        else:
            # 기존 방식: 파일별 순차 처리
            all_sessions = []
            for log_file in log_files:
                if stream_mode and output_file:
                    # 스트리밍 모드: 각 파일별로 처리 (메모리 효율적)
                    file_output = str(Path(output_file).parent / f"{log_file.stem}_preprocessed.json")
                    sessions = self.process_log_file(str(log_file), file_output, stream_mode=True)
                else:
                    # 메모리 모드
                    sessions = self.process_log_file(str(log_file), None, stream_mode=False)
                    all_sessions.extend(sessions)
            
            if not stream_mode and output_file:
                self.save_results(all_sessions, output_file)
            
            logger.info(f"전체 처리 완료: {len(all_sessions) if not stream_mode else '스트리밍 모드'}개 세션 생성")
            return all_sessions
    
    def _process_by_date_batch(self, log_files: List[Path], output_file: Optional[str] = None,
                               stream_mode: bool = True, date_filter: Optional[str] = None,
                               parallel: bool = False, max_workers: int = 8) -> List[Dict[str, Any]]:
        """
        날짜별 배치 처리 (메모리 효율적)
        
        각 날짜별로 로그를 수집하고 처리하여 메모리 사용량을 최소화
        parallel=True일 경우 여러 날짜를 동시에 병렬 처리
        """
        logger.info("날짜별 배치 처리 모드로 시작 (메모리 효율적)")
        logger.info(f"처리할 로그 파일 수: {len(log_files)}개")
        
        if not log_files:
            logger.warning("처리할 로그 파일이 없습니다. 로그 디렉토리 경로를 확인하세요.")
            return []
        
        # 날짜별로 로그 파일 분류
        date_files = defaultdict(list)  # {date: [log_files]}
        
        for log_file in log_files:
            # 로그 파일에서 날짜 추출 시도
            dates_in_file = self._extract_dates_from_file(log_file)
            
            # date_filter가 있으면 파일명 날짜를 먼저 확인
            if date_filter:
                # 파일명에서 날짜 추출 (빠른 필터링)
                import re
                file_date = None
                match = re.search(r'(\d{6})', log_file.name)
                if match:
                    try:
                        date_str = match.group(1)
                        date_obj = datetime.strptime(date_str, '%y%m%d')
                        file_date = date_obj.strftime('%Y-%m-%d')
                    except:
                        pass
                
                # 파일명 날짜가 필터와 일치하지 않으면 스킵
                if file_date and file_date != date_filter:
                    continue
            
            # 날짜별로 파일 분류
            for date in dates_in_file:
                if date_filter is None or date == date_filter:
                    date_files[date].append(log_file)
        
        # 출력 디렉토리 생성
        if output_file:
            output_path = Path(output_file)
            output_base = output_path.parent
            output_name = output_path.stem
        else:
            output_base = Path('.')
            output_name = 'preprocessed'
        
        # 출력 디렉토리 생성 (없으면 자동 생성)
        output_base.mkdir(parents=True, exist_ok=True)
        logger.info(f"출력 디렉토리: {output_base.absolute()}")
        
        # 날짜 목록
        dates = sorted(date_files.keys())
        logger.info(f"처리할 날짜 수: {len(dates)}개")
        
        if parallel and len(dates) > 1:
            # 병렬 처리 모드
            logger.info(f"병렬 처리 모드 활성화 (최대 {max_workers}개 프로세스)")
            return self._process_dates_parallel(
                dates, date_files, output_base, output_name, 
                stream_mode, max_workers
            )
        else:
            # 순차 처리 모드
            all_sessions = []
            
            for date in dates:
                logger.info(f"날짜별 처리 시작: {date} ({len(date_files[date])}개 파일)")
                
                # 날짜별 출력 파일
                date_output = str(output_base / f"{output_name}_{date}.json")
                
                # 해당 날짜의 로그만 처리
                date_logs = self._collect_logs_by_date(date_files[date], date)
                
                if not date_logs:
                    logger.info(f"  {date}: 처리할 로그 없음")
                    continue
                
                # 날짜별 세션 생성 및 저장
                sessions = self._process_date_logs(date_logs, date_output, stream_mode, date)
                
                if not stream_mode:
                    all_sessions.extend(sessions)
                
                # 메모리 정리
                del date_logs
                import gc
                gc.collect()
                
                logger.info(f"  {date}: 처리 완료")
            
            logger.info(f"날짜별 배치 처리 완료")
            return all_sessions
    
    def _process_dates_parallel(self, dates: List[str], date_files: Dict[str, List[Path]],
                               output_base: Path, output_name: str, stream_mode: bool,
                               max_workers: int = 8) -> List[Dict[str, Any]]:
        """
        날짜별 병렬 처리
        
        여러 날짜를 동시에 처리하여 속도 향상
        """
        # 병렬 처리 실행
        all_sessions = []
        completed_dates = []
        failed_dates = []
        
        # 작업 인자 준비
        tasks = []
        for date in dates:
            tasks.append((
                date,
                date_files[date],
                str(output_base),
                output_name,
                stream_mode,
                self.config.copy()  # 설정 복사
            ))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 모든 날짜 작업 제출
            future_to_date = {
                executor.submit(_process_single_date_parallel, task): task[0] 
                for task in tasks
            }
            
            # 완료된 작업 처리
            for future in as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    result_date, success, result = future.result()
                    if success:
                        completed_dates.append(result_date)
                        logger.info(f"✅ {result_date}: 완료")
                    else:
                        failed_dates.append((result_date, result))
                        logger.error(f"❌ {result_date}: 실패 - {result}")
                except Exception as e:
                    failed_dates.append((date, str(e)))
                    logger.error(f"❌ {date}: 예외 발생 - {e}")
        
        logger.info(f"병렬 처리 완료: 성공 {len(completed_dates)}개, 실패 {len(failed_dates)}개")
        if failed_dates:
            logger.warning(f"실패한 날짜: {[d[0] for d in failed_dates]}")
        
        return all_sessions
    
    def _extract_dates_from_file(self, log_file: Path) -> List[str]:
        """로그 파일에서 날짜 목록 추출 (파일명 우선)"""
        dates = set()
        
        # 1단계: 파일명에서 날짜 추출 (우선)
        import re
        # 파일명 패턴들:
        # - service_YYYY-MM-DD.log (날짜별 분리된 파일)
        # - service_YYMMDD_HH:MM:SS.log (원본 파일)
        # - service_YYYYMMDD.log
        
        # 패턴 1: YYYY-MM-DD 형식 (날짜별 분리된 파일)
        match = re.search(r'(\d{4}-\d{2}-\d{2})', log_file.name)
        if match:
            date_str = match.group(1)
            try:
                # 형식 검증
                datetime.strptime(date_str, '%Y-%m-%d')
                dates.add(date_str)
                # 날짜별 분리된 파일은 하나의 날짜만 포함
                return list(dates)
            except:
                pass
        
        # 패턴 2: YYMMDD 또는 YYYYMMDD 형식 (원본 파일)
        patterns = [
            r'(\d{6})',  # YYMMDD (250822)
            r'(\d{8})',  # YYYYMMDD (20250822)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, log_file.name)
            if match:
                date_str = match.group(1)
                try:
                    if len(date_str) == 6:
                        # YYMMDD 형식
                        date_obj = datetime.strptime(date_str, '%y%m%d')
                    else:
                        # YYYYMMDD 형식
                        date_obj = datetime.strptime(date_str, '%Y%m%d')
                    dates.add(date_obj.strftime('%Y-%m-%d'))
                    # 파일명에서 날짜를 찾았으면 그것을 우선 사용
                    return list(dates)
                except:
                    pass
        
        # 2단계: 파일명에서 날짜를 찾지 못한 경우에만 로그 내용에서 추출
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= 100:  # 처음 100줄만 확인 (성능 향상)
                        break
                    timestamp = self.sessionizer.extract_timestamp(line)
                    if timestamp:
                        dates.add(timestamp.strftime('%Y-%m-%d'))
                        # 첫 번째 유효한 날짜를 찾으면 중단 (성능 향상)
                        if len(dates) >= 1:
                            break
        except:
            pass
        
        return list(dates) if dates else ['unknown']
    
    def _collect_logs_by_date(self, log_files: List[Path], target_date: str) -> Dict[str, List[Dict[str, Any]]]:
        """특정 날짜의 로그만 수집 (메모리 효율적)"""
        date_logs = defaultdict(list)  # {service: [logs]}
        
        for log_file in log_files:
            service_name = MetadataEnricher.extract_service_name(str(log_file))
            logger.info(f"  로그 수집 중: {log_file.name}")
            
            line_count = 0
            collected_count = 0
            
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line_count += 1
                    cleaned_line = self.cleaner.clean_log_line(line)
                    if not cleaned_line:
                        continue
                    
                    # 타임스탬프 추출
                    timestamp = self.sessionizer.extract_timestamp(cleaned_line)
                    if not timestamp:
                        continue
                    
                    # 날짜 필터링
                    if timestamp.strftime('%Y-%m-%d') != target_date:
                        continue
                    
                    # 로그 파싱
                    parsed_log = self.parser.parse_log(cleaned_line)
                    parsed_log['timestamp'] = timestamp.isoformat()
                    parsed_log['service_name'] = service_name
                    parsed_log['original'] = cleaned_line
                    
                    date_logs[service_name].append(parsed_log)
                    collected_count += 1
                    
                    # 메모리 제한: 날짜별 최대 로그 수 제한
                    if collected_count > 1000000:  # 100만 줄 제한
                        logger.warning(f"  {log_file.name}: 날짜별 로그 수 제한 도달 (100만 줄)")
                        break
            
            logger.info(f"    {log_file.name}: {line_count}줄 중 {collected_count}줄 수집")
        
        return date_logs
    
    def _process_date_logs(self, date_logs: Dict[str, List[Dict[str, Any]]], 
                          output_file: str, stream_mode: bool, date: str) -> List[Dict[str, Any]]:
        """날짜별 로그 처리"""
        all_sessions = []
        output_f = None
        first_item = True
        
        if stream_mode and output_file:
            output_f = open(output_file, 'w', encoding='utf-8')
            output_f.write('[\n')
        
        session_count = 0
        
        # 시간 윈도우별로 그룹화 (5분 단위)
        time_windows = defaultdict(lambda: defaultdict(list))
        
        for service_name, logs in date_logs.items():
            for log_data in logs:
                timestamp_str = log_data.get('timestamp')
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        time_window = (timestamp.hour, timestamp.minute // 5)
                        window_key = f"{date}_{time_window[0]:02d}_{time_window[1]:02d}"
                        time_windows[window_key][service_name].append(log_data)
                    except:
                        pass
        
        # 시간 윈도우별로 세션 생성
        for window_key in sorted(time_windows.keys()):
            services_logs = time_windows[window_key]
            
            for service_name, logs in services_logs.items():
                temp_sessionizer = Sessionizer(
                    method=self.config.get('sessionization_method', 'sliding_window'),
                    window_size=self.config.get('window_size', 50),
                    window_time=self.config.get('window_time', 180)
                )
                
                for log_data in logs:
                    completed_sessions = temp_sessionizer.add_log(log_data)
                    
                    for session in completed_sessions:
                        enriched = self.metadata_enricher.enrich_session(session, service_name)
                        encoded = self.encoder.encode_sequence(session)
                        
                        enriched['time_window'] = window_key
                        enriched['related_services'] = list(services_logs.keys())
                        enriched['correlation_id'] = f"{window_key}_{service_name}"
                        
                        final_result = {
                            **enriched,
                            **encoded,
                            'session_id': session_count
                        }
                        session_count += 1
                        
                        if stream_mode and output_f:
                            if not first_item:
                                output_f.write(',\n')
                            json.dump(final_result, output_f, ensure_ascii=False, indent=2)
                            first_item = False
                        else:
                            all_sessions.append(final_result)
                
                remaining = temp_sessionizer.flush_sessions()
                for session in remaining:
                    enriched = self.metadata_enricher.enrich_session(session, service_name)
                    encoded = self.encoder.encode_sequence(session)
                    enriched['time_window'] = window_key
                    enriched['related_services'] = list(services_logs.keys())
                    enriched['correlation_id'] = f"{window_key}_{service_name}"
                    
                    final_result = {
                        **enriched,
                        **encoded,
                        'session_id': session_count
                    }
                    session_count += 1
                    
                    if stream_mode and output_f:
                        if not first_item:
                            output_f.write(',\n')
                        json.dump(final_result, output_f, ensure_ascii=False, indent=2)
                        first_item = False
                    else:
                        all_sessions.append(final_result)
        
        if stream_mode and output_f:
            output_f.write('\n]')
            output_f.close()
        
        return all_sessions
    
    def _process_with_correlation(self, log_files: List[Path], output_file: Optional[str] = None,
                                  stream_mode: bool = True) -> List[Dict[str, Any]]:
        """
        날짜와 시스템명을 활용한 관계 추적 처리
        
        같은 시간대의 다른 서비스(gateway, eureka, manager 등) 로그를 연결
        """
        logger.info("날짜/시스템명 기반 관계 추적 모드로 처리 시작")
        
        # 1단계: 모든 로그를 날짜/시간 기준으로 수집 (메모리 효율적)
        time_based_logs = defaultdict(lambda: defaultdict(list))  # {date: {service: [logs]}}
        
        for log_file in log_files:
            service_name = MetadataEnricher.extract_service_name(str(log_file))
            logger.info(f"로그 수집 중: {log_file.name} ({service_name})")
            
            line_count = 0
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line_count += 1
                    cleaned_line = self.cleaner.clean_log_line(line)
                    if not cleaned_line:
                        continue
                    
                    # 타임스탬프 추출
                    timestamp = self.sessionizer.extract_timestamp(cleaned_line)
                    if not timestamp:
                        continue
                    
                    # 날짜 키 생성 (시간 윈도우 단위로 그룹화)
                    date_key = timestamp.strftime('%Y-%m-%d')
                    time_window = (timestamp.hour, timestamp.minute // 5)  # 5분 단위
                    
                    # 로그 파싱
                    parsed_log = self.parser.parse_log(cleaned_line)
                    parsed_log['timestamp'] = timestamp.isoformat()
                    parsed_log['service_name'] = service_name
                    parsed_log['original'] = cleaned_line
                    
                    # 시간 윈도우별로 그룹화
                    window_key = f"{date_key}_{time_window[0]:02d}_{time_window[1]:02d}"
                    time_based_logs[window_key][service_name].append(parsed_log)
            
            logger.info(f"  {log_file.name}: {line_count}줄 처리 완료")
        
        # 2단계: 시간 윈도우별로 세션 생성 및 관계 추적
        all_sessions = []
        output_f = None
        first_item = True
        
        if stream_mode and output_file:
            output_f = open(output_file, 'w', encoding='utf-8')
            output_f.write('[\n')
        
        session_count = 0
        
        for window_key in sorted(time_based_logs.keys()):
            services_logs = time_based_logs[window_key]
            
            # 각 서비스별로 세션 생성
            for service_name, logs in services_logs.items():
                # 세션화
                temp_sessionizer = Sessionizer(
                    method=self.config.get('sessionization_method', 'sliding_window'),
                    window_size=self.config.get('window_size', 50),
                    window_time=self.config.get('window_time', 180)
                )
                
                for log_data in logs:
                    completed_sessions = temp_sessionizer.add_log(log_data)
                    
                    for session in completed_sessions:
                        enriched = self.metadata_enricher.enrich_session(session, service_name)
                        encoded = self.encoder.encode_sequence(session)
                        
                        # 관계 정보 추가
                        enriched['time_window'] = window_key
                        enriched['related_services'] = list(services_logs.keys())
                        enriched['correlation_id'] = f"{window_key}_{service_name}"
                        
                        final_result = {
                            **enriched,
                            **encoded,
                            'session_id': session_count
                        }
                        session_count += 1
                        
                        if stream_mode and output_f:
                            if not first_item:
                                output_f.write(',\n')
                            json.dump(final_result, output_f, ensure_ascii=False, indent=2)
                            first_item = False
                        else:
                            all_sessions.append(final_result)
                
                # 남은 세션 처리
                remaining = temp_sessionizer.flush_sessions()
                for session in remaining:
                    enriched = self.metadata_enricher.enrich_session(session, service_name)
                    encoded = self.encoder.encode_sequence(session)
                    enriched['time_window'] = window_key
                    enriched['related_services'] = list(services_logs.keys())
                    enriched['correlation_id'] = f"{window_key}_{service_name}"
                    
                    final_result = {
                        **enriched,
                        **encoded,
                        'session_id': session_count
                    }
                    session_count += 1
                    
                    if stream_mode and output_f:
                        if not first_item:
                            output_f.write(',\n')
                        json.dump(final_result, output_f, ensure_ascii=False, indent=2)
                        first_item = False
                    else:
                        all_sessions.append(final_result)
        
        if stream_mode and output_f:
            output_f.write('\n]')
            output_f.close()
            logger.info(f"관계 추적 처리 완료: {session_count}개 세션 생성 (스트리밍 모드)")
        else:
            logger.info(f"관계 추적 처리 완료: {len(all_sessions)}개 세션 생성")
        
        return all_sessions
    
    def save_results(self, sessions: List[Dict[str, Any]], output_path: str):
        """결과 저장"""
        # 출력 디렉토리 생성
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"결과 저장 중: {output_path}")
        
        # JSON 형식으로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"저장 완료: {len(sessions)}개 세션 → {output_file.absolute()}")


def _process_single_date_parallel(args: Tuple[str, List[Path], str, str, bool, Dict[str, Any]]) -> Tuple[str, bool, Optional[str]]:
    """
    단일 날짜 처리 함수 (병렬 실행용)
    
    이 함수는 모듈 레벨에 있어야 ProcessPoolExecutor가 pickle할 수 있습니다.
    클래스 정의 이후에 배치되어 LogPreprocessor를 참조할 수 있습니다.
    """
    date, file_list, output_base_str, output_name, stream_mode, config = args
    try:
        # 각 프로세스에서 독립적인 전처리기 생성
        temp_preprocessor = LogPreprocessor(config)
        
        logger.info(f"[병렬] 날짜별 처리 시작: {date} ({len(file_list)}개 파일)")
        
        # 날짜별 출력 파일
        output_base = Path(output_base_str)
        date_output = str(output_base / f"{output_name}_{date}.json")
        
        # 해당 날짜의 로그만 처리
        date_logs = temp_preprocessor._collect_logs_by_date(file_list, date)
        
        if not date_logs:
            logger.info(f"  {date}: 처리할 로그 없음")
            return (date, True, None)
        
        # 날짜별 세션 생성 및 저장
        sessions = temp_preprocessor._process_date_logs(date_logs, date_output, stream_mode, date)
        
        logger.info(f"  {date}: 처리 완료")
        return (date, True, date_output)
        
    except Exception as e:
        logger.error(f"  {date}: 처리 중 오류 발생 - {e}")
        import traceback
        traceback.print_exc()
        return (date, False, str(e))


def load_config(config_path: str = None) -> Dict[str, Any]:
    """설정 파일 로드"""
    if config_path is None:
        # 기본 경로: preprocessing 폴더 내의 설정 파일
        base_dir = Path(__file__).parent
        config_path = str(base_dir / "preprocessing_config.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            # 상대 경로를 절대 경로로 변환
            base_dir = Path(config_path).parent
            project_root = base_dir.parent  # RADAR 프로젝트 루트
            
            # drain3_config_path 변환
            if 'drain3_config_path' in config and not os.path.isabs(config['drain3_config_path']):
                config['drain3_config_path'] = str(base_dir / config['drain3_config_path'])
            
            # log_directory 변환 (프로젝트 루트 기준)
            if 'log_directory' in config and not os.path.isabs(config['log_directory']):
                config['log_directory'] = str(project_root / config['log_directory'])
            
            # output_directory 설정 (기본값: output)
            if 'output_directory' not in config:
                config['output_directory'] = 'output'
            
            # output_path 변환 (output_directory 기준)
            if 'output_path' in config:
                if os.path.isabs(config['output_path']):
                    # 절대 경로인 경우 그대로 사용
                    pass
                else:
                    # 상대 경로인 경우 output_directory 기준으로 변환
                    output_dir = base_dir / config.get('output_directory', 'output')
                    config['output_path'] = str(output_dir / config['output_path'])
            
            return config
    else:
        # 기본 설정
        base_dir = Path(__file__).parent
        output_dir = base_dir / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            'log_directory': str(base_dir.parent / 'logs' / 'date_split'),
            'output_directory': 'output',
            'output_path': str(output_dir / 'preprocessed_logs.json'),
            'sessionization_method': 'sliding_window',  # 'trace_id' or 'sliding_window'
            'window_size': 20,
            'window_time': 300,  # 5분
            'max_seq_length': 512,
            'drain3_config_path': str(base_dir / 'drain3_config.yaml')
        }


def main():
    """메인 실행 함수"""
    import argparse
    
    # 기본 경로 설정 (preprocessing 폴더 기준)
    base_dir = Path(__file__).parent
    project_root = base_dir.parent
    default_config = str(base_dir / 'preprocessing_config.yaml')
    default_log_dir = str(project_root / 'logs' / 'date_split')  # 날짜별 분리된 파일 위치
    default_output_dir = base_dir / 'output'  # 출력 디렉토리
    default_output_dir.mkdir(parents=True, exist_ok=True)
    default_output = str(default_output_dir / 'preprocessed_logs.json')
    
    parser = argparse.ArgumentParser(description='로그 전처리 파이프라인')
    parser.add_argument('--config', type=str, default=default_config,
                       help='설정 파일 경로')
    parser.add_argument('--log-dir', type=str, default=default_log_dir,
                       help='로그 디렉토리 경로')
    parser.add_argument('--output', type=str, default=default_output,
                       help='출력 파일 경로')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    if args.log_dir:
        config['log_directory'] = args.log_dir
    if args.output:
        config['output_path'] = args.output
    
    # 전처리 실행
    preprocessor = LogPreprocessor(config)
    
    # 스트리밍 모드 및 관계 추적 활성화
    stream_mode = config.get('stream_mode', True)
    enable_correlation = config.get('enable_correlation', True)
    batch_by_date = config.get('batch_by_date', True)
    date_filter = config.get('date_filter', None)
    parallel = config.get('parallel', False)
    max_workers = config.get('max_workers', 8)
    
    sessions = preprocessor.process_log_directory(
        config['log_directory'],
        output_file=config['output_path'],
        stream_mode=stream_mode,
        enable_correlation=enable_correlation,
        batch_by_date=batch_by_date,
        date_filter=date_filter,
        parallel=parallel,
        max_workers=max_workers
    )
    
    # 스트리밍 모드가 아닐 때만 저장
    if not stream_mode:
        preprocessor.save_results(sessions, config['output_path'])
    
    logger.info("전처리 완료!")


if __name__ == "__main__":
    main()

