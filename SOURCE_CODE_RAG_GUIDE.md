# 소스 코드 기반 RAG 시스템 가이드

## 📋 개요

로그 분석 결과와 소스 코드를 연결하여 정확한 해결 가이드를 제공하는 RAG 시스템입니다.

---

## 🎯 목표

1. **로그 에러와 소스 코드 연결**: 에러 메시지를 기반으로 관련 소스 코드 위치 찾기
2. **코드 기반 가이드 생성**: 실제 소스 코드를 참조하여 구체적인 해결 방법 제시
3. **자동화된 문제 해결**: LLM이 소스 코드를 분석하여 수정 방법 제안

---

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│              소스 코드 수집 및 파싱                       │
│  Gateway │ Research │ Manager │ Code 등 소스 코드      │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│              코드 파싱 및 청크 분할                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ 파일 파싱│→│ 함수 추출│→│ 클래스 추출│            │
│  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│              코드 벡터화 및 인덱싱                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ CodeBERT│→│ 임베딩 생성│→│ 벡터 DB 저장│            │
│  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│              로그-코드 연결                              │
│  로그 에러 메시지 → 관련 코드 청크 검색                   │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│              소스 코드 기반 가이드 생성                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ 코드 검색│→│ LLM 분석  │→│ 가이드 생성│            │
│  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 구현 단계

### 1단계: 소스 코드 파싱

#### 목적
- 소스 코드를 의미 있는 단위로 분할
- 함수, 클래스, 메서드 단위로 추출
- 메타데이터 추출 (파일 경로, 라인 번호 등)

#### 구현 방법

**1.1 Java 소스 코드 파싱**

```python
import ast
import re
from pathlib import Path
from typing import List, Dict, Any

class JavaCodeParser:
    """Java 소스 코드 파서"""
    
    def parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Java 파일을 파싱하여 코드 청크 추출"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = []
        
        # 클래스 추출
        class_pattern = r'public\s+class\s+(\w+)\s*\{([^}]+)\}'
        for match in re.finditer(class_pattern, content, re.DOTALL):
            class_name = match.group(1)
            class_body = match.group(2)
            line_start = content[:match.start()].count('\n') + 1
            
            # 메서드 추출
            method_pattern = r'public\s+(\w+)\s+(\w+)\s*\([^)]*\)\s*\{([^}]+)\}'
            for method_match in re.finditer(method_pattern, class_body):
                method_return = method_match.group(1)
                method_name = method_match.group(2)
                method_body = method_match.group(3)
                method_line_start = line_start + class_body[:method_match.start()].count('\n')
                method_line_end = method_line_start + method_body.count('\n')
                
                chunks.append({
                    'chunk_id': f"{class_name}_{method_name}",
                    'type': 'method',
                    'class_name': class_name,
                    'method_name': method_name,
                    'return_type': method_return,
                    'code': method_match.group(0),
                    'file_path': str(file_path),
                    'service_name': self._extract_service_name(file_path),
                    'line_start': method_line_start,
                    'line_end': method_line_end,
                    'metadata': {
                        'imports': self._extract_imports(content),
                        'annotations': self._extract_annotations(method_match.group(0))
                    }
                })
        
        return chunks
    
    def _extract_service_name(self, file_path: Path) -> str:
        """파일 경로에서 서비스명 추출"""
        # 예: gateway/src/main/java/... -> gateway
        parts = file_path.parts
        if 'gateway' in parts:
            return 'gateway'
        elif 'research' in parts:
            return 'research'
        # ... 기타 서비스
        return 'unknown'
    
    def _extract_imports(self, content: str) -> List[str]:
        """Import 문 추출"""
        import_pattern = r'import\s+([^;]+);'
        return re.findall(import_pattern, content)
    
    def _extract_annotations(self, code: str) -> List[str]:
        """어노테이션 추출"""
        annotation_pattern = r'@(\w+)'
        return re.findall(annotation_pattern, code)
```

**1.2 Python 소스 코드 파싱**

```python
import ast
from typing import List, Dict, Any

class PythonCodeParser:
    """Python 소스 코드 파서"""
    
    def parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Python 파일을 파싱하여 코드 청크 추출"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        chunks = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                chunk = {
                    'chunk_id': node.name,
                    'type': 'function',
                    'function_name': node.name,
                    'code': ast.get_source_segment(content, node),
                    'file_path': str(file_path),
                    'line_start': node.lineno,
                    'line_end': node.end_lineno,
                    'metadata': {
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [ast.get_source_segment(content, d) for d in node.decorator_list]
                    }
                }
                chunks.append(chunk)
        
        return chunks
```

---

### 2단계: 코드 벡터화

#### 목적
- 코드를 벡터로 변환하여 유사도 검색 가능하게 함
- 코드 전용 임베딩 모델 사용

#### 구현 방법

**2.1 CodeBERT 사용**

```python
from transformers import AutoTokenizer, AutoModel
import torch

class CodeVectorizer:
    """코드 벡터화 클래스"""
    
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def vectorize_code(self, code: str) -> List[float]:
        """코드를 벡터로 변환"""
        # 코드 전처리
        code = self._preprocess_code(code)
        
        # 토큰화
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # 임베딩 생성
        with torch.no_grad():
            outputs = self.model(**inputs)
            # [CLS] 토큰의 임베딩 사용
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        
        return embedding.tolist()
    
    def _preprocess_code(self, code: str) -> str:
        """코드 전처리"""
        # 주석 제거
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # 공백 정규화
        code = ' '.join(code.split())
        return code
```

**2.2 StarCoder 사용 (대안)**

```python
from transformers import AutoTokenizer, AutoModel

class StarCoderVectorizer:
    """StarCoder 기반 벡터화"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")
        self.model = AutoModel.from_pretrained("bigcode/starcoder")
        self.model.eval()
    
    def vectorize_code(self, code: str) -> List[float]:
        """코드를 벡터로 변환"""
        # StarCoder는 코드 생성 모델이지만 임베딩도 추출 가능
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        return embedding.tolist()
```

---

### 3단계: 로그-코드 연결

#### 목적
- 로그 에러 메시지와 관련된 소스 코드 찾기
- 에러 키워드와 코드 내용 매칭

#### 구현 방법

```python
class LogCodeLinker:
    """로그와 코드 연결 클래스"""
    
    def __init__(self, vector_db):
        self.vector_db = vector_db
    
    def find_related_code(
        self,
        error_message: str,
        service_name: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """에러 메시지와 관련된 코드 찾기"""
        
        # 에러 메시지 벡터화
        error_vector = self.vectorize_error(error_message)
        
        # 벡터 DB에서 유사한 코드 검색
        results = self.vector_db.query(
            query_embeddings=[error_vector],
            n_results=top_k,
            where={"service_name": service_name, "type": "code"}
        )
        
        # 결과 정리
        related_code = []
        for result in results['metadatas'][0]:
            related_code.append({
                'file_path': result['file_path'],
                'chunk_id': result['chunk_id'],
                'method_name': result.get('method_name'),
                'line_start': result['line_start'],
                'line_end': result['line_end'],
                'code': result['code'],
                'similarity': result.get('similarity', 0.0)
            })
        
        return related_code
    
    def vectorize_error(self, error_message: str) -> List[float]:
        """에러 메시지 벡터화"""
        # CodeBERT로 에러 메시지도 벡터화
        vectorizer = CodeVectorizer()
        return vectorizer.vectorize_code(error_message)
```

---

### 4단계: 소스 코드 기반 가이드 생성

#### 목적
- 관련 소스 코드를 참조하여 구체적인 해결 방법 제시
- 코드 위치 및 수정 방법 안내

#### 구현 방법

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

class CodeGuideGenerator:
    """소스 코드 기반 가이드 생성 클래스"""
    
    def __init__(self, llm_model: str = "gpt-4"):
        self.llm = OpenAI(model_name=llm_model, temperature=0.3)
        
        self.prompt_template = PromptTemplate(
            input_variables=["error_message", "related_code", "service_name"],
            template="""
다음은 MSA 환경에서 발생한 에러와 관련 소스 코드입니다.

에러 메시지: {error_message}
서비스: {service_name}

관련 소스 코드:
{related_code}

위 정보를 바탕으로 다음을 포함한 해결 가이드를 작성해주세요:
1. 문제 원인 분석
2. 관련 코드 위치 (파일 경로, 라인 번호)
3. 구체적인 해결 단계
4. 코드 수정 제안 (필요한 경우)

가이드를 JSON 형식으로 반환해주세요:
{{
  "title": "문제 제목",
  "severity": "HIGH/MEDIUM/LOW",
  "root_cause": "문제 원인",
  "related_code": [
    {{
      "file_path": "파일 경로",
      "line_start": 시작 라인,
      "line_end": 끝 라인,
      "explanation": "이 코드와의 관련성 설명"
    }}
  ],
  "solution_steps": [
    "1단계 설명",
    "2단계 설명"
  ],
  "code_fix_suggestion": "코드 수정 제안 (있는 경우)"
}}
"""
        )
    
    def generate_guide(
        self,
        error_message: str,
        related_code: List[Dict[str, Any]],
        service_name: str
    ) -> Dict[str, Any]:
        """소스 코드 기반 가이드 생성"""
        
        # 관련 코드 포맷팅
        code_text = self._format_code(related_code)
        
        # 프롬프트 생성
        prompt = self.prompt_template.format(
            error_message=error_message,
            related_code=code_text,
            service_name=service_name
        )
        
        # LLM 호출
        response = self.llm(prompt)
        
        # JSON 파싱
        import json
        guide = json.loads(response)
        
        return guide
    
    def _format_code(self, related_code: List[Dict[str, Any]]) -> str:
        """코드를 텍스트로 포맷팅"""
        formatted = []
        for code in related_code:
            formatted.append(f"""
파일: {code['file_path']}
메서드: {code.get('method_name', 'N/A')}
라인: {code['line_start']}-{code['line_end']}
코드:
{code['code']}
---
""")
        return '\n'.join(formatted)
```

---

## 📊 데이터 구조

### 코드 청크 메타데이터

```json
{
  "chunk_id": "GatewayController_handleRequest",
  "type": "method",
  "class_name": "GatewayController",
  "method_name": "handleRequest",
  "code": "public ResponseEntity<?> handleRequest(...) { ... }",
  "file_path": "gateway/src/main/java/com/example/GatewayController.java",
  "service_name": "gateway",
  "line_start": 45,
  "line_end": 78,
  "embedding": [0.1, 0.2, ...],
  "metadata": {
    "imports": ["org.springframework.web.bind.annotation.*"],
    "annotations": ["@RestController", "@RequestMapping"],
    "parameters": ["HttpServletRequest request"],
    "return_type": "ResponseEntity"
  }
}
```

### 로그-코드 연결 메타데이터

```json
{
  "trace_id": "abc123",
  "log_chunk_id": "log_abc123_1",
  "code_chunk_ids": ["GatewayController_handleRequest", "RequestHandler_process"],
  "error_message": "Connection timeout",
  "service_name": "gateway",
  "similarity_scores": {
    "GatewayController_handleRequest": 0.85,
    "RequestHandler_process": 0.72
  }
}
```

---

## 🔄 전체 파이프라인

```python
class SourceCodeRAGPipeline:
    """소스 코드 기반 RAG 파이프라인"""
    
    def __init__(self):
        self.code_parser = JavaCodeParser()
        self.vectorizer = CodeVectorizer()
        self.vector_db = ChromaDB()
        self.linker = LogCodeLinker(self.vector_db)
        self.guide_generator = CodeGuideGenerator()
    
    def build_index(self, source_code_dir: Path):
        """소스 코드 인덱스 구축"""
        # 1. 소스 코드 파일 수집
        java_files = list(source_code_dir.rglob("*.java"))
        
        # 2. 각 파일 파싱
        all_chunks = []
        for file_path in java_files:
            chunks = self.code_parser.parse_file(file_path)
            all_chunks.extend(chunks)
        
        # 3. 벡터화 및 저장
        for chunk in all_chunks:
            embedding = self.vectorizer.vectorize_code(chunk['code'])
            self.vector_db.add(
                embeddings=[embedding],
                metadatas=[chunk],
                ids=[chunk['chunk_id']]
            )
    
    def generate_guide_from_error(
        self,
        error_message: str,
        service_name: str,
        trace_id: str
    ) -> Dict[str, Any]:
        """에러로부터 가이드 생성"""
        # 1. 관련 코드 찾기
        related_code = self.linker.find_related_code(
            error_message,
            service_name,
            top_k=5
        )
        
        # 2. 가이드 생성
        guide = self.guide_generator.generate_guide(
            error_message,
            related_code,
            service_name
        )
        
        # 3. Trace ID 연결
        guide['trace_id'] = trace_id
        guide['related_code'] = related_code
        
        return guide
```

---

## 📝 사용 예시

### 1. 소스 코드 인덱스 구축

```python
pipeline = SourceCodeRAGPipeline()

# 소스 코드 디렉토리에서 인덱스 구축
source_code_dir = Path("../source_code")
pipeline.build_index(source_code_dir)
```

### 2. 에러 기반 가이드 생성

```python
# 에러 메시지로 가이드 생성
error_message = "Connection timeout in gateway service"
service_name = "gateway"
trace_id = "abc123"

guide = pipeline.generate_guide_from_error(
    error_message,
    service_name,
    trace_id
)

print(guide)
# 출력:
# {
#   "title": "Connection Timeout 해결 방법",
#   "severity": "HIGH",
#   "related_code": [
#     {
#       "file_path": "gateway/src/main/java/com/example/GatewayController.java",
#       "line_start": 45,
#       "line_end": 78,
#       "explanation": "이 메서드에서 타임아웃 설정을 확인하세요"
#     }
#   ],
#   "solution_steps": [
#     "1. GatewayController.java의 handleRequest 메서드 확인",
#     "2. 타임아웃 설정 값 확인 및 조정"
#   ]
# }
```

---

이 가이드를 따라 소스 코드 기반 RAG 시스템을 구축할 수 있습니다! 🚀
