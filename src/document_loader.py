#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Loader Module
문서 로더 모듈
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Generator
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DocumentLoader(ABC):
    """문서 로더 추상 클래스"""
    
    @abstractmethod
    def load(self, file_path: Path) -> Generator[Dict[str, str], None, None]:
        """파일에서 문서 로드"""
        pass


class QAFileLoader(DocumentLoader):
    """Q&A 형식 텍스트 파일 로더"""
    
    def load(self, file_path: Path) -> Generator[Dict[str, str], None, None]:
        """Q&A 패턴 매칭으로 문서 로드"""
        import re
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 다양한 Q&A 패턴
            patterns = [
                # [1.Q] ... [1.A] ...
                r'\[(\d+)\.Q\]\s*([\s\S]*?)\n\s*\[(\d+)\.A\]\s*([\s\S]*?)(?=\n\s*\[\d+\.Q\]|$)',
                # Q1: ... A1: ...
                r'Q(\d+):\s*([\s\S]*?)\n\s*A(\d+):\s*([\s\S]*?)(?=\n\s*Q\d+:|$)',
                # 질문1: ... 답변1: ...
                r'질문(\d+):\s*([\s\S]*?)\n\s*답변(\d+):\s*([\s\S]*?)(?=\n\s*질문\d+:|$)'
            ]
            
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
                if found:
                    matches.extend(found)
                    logger.debug(f"Pattern found {len(found)} matches in {file_path.name}")
            
            for match in matches:
                q_num, question, a_num, answer = match
                if q_num == a_num:
                    question = question.strip()
                    answer = answer.strip()
                    
                    if question and answer:
                        yield {
                            "question": question,
                            "answer": answer
                        }
                        
        except Exception as e:
            logger.warning(f"Error loading QA file {file_path}: {e}")


class JSONFileLoader(DocumentLoader):
    """JSON/JSONL 파일 로더"""
    
    # 지원하는 질문/답변 필드명
    QUESTION_FIELDS = [
        'question', 'q', 'query', '질문', 'input', 
        'instruction', 'Context', 'context', 'prompt'
    ]
    ANSWER_FIELDS = [
        'answer', 'a', 'response', '답변', 'output', 
        'completion', 'Response', 'reply', 'text'
    ]
    
    def load(self, file_path: Path) -> Generator[Dict[str, str], None, None]:
        """JSON/JSONL 파일 로드"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix == '.jsonl':
                    # JSONL 파일 처리
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                doc = self._extract_qa(data)
                                if doc:
                                    yield doc
                            except json.JSONDecodeError as e:
                                logger.debug(f"Line {line_num} JSON parse error: {e}")
                else:
                    # JSON 파일 처리
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            doc = self._extract_qa(item)
                            if doc:
                                yield doc
                    elif isinstance(data, dict):
                        doc = self._extract_qa(data)
                        if doc:
                            yield doc
                            
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
    
    def _extract_qa(self, record: dict) -> Optional[Dict[str, str]]:
        """레코드에서 Q&A 추출"""
        question = None
        answer = None
        
        # 질문 필드 찾기
        for field in self.QUESTION_FIELDS:
            if field in record and record[field]:
                question = str(record[field]).strip()
                break
                
        # 답변 필드 찾기
        for field in self.ANSWER_FIELDS:
            if field in record and record[field]:
                answer = str(record[field]).strip()
                break
        
        if question and answer:
            return {
                "question": question,
                "answer": answer
            }
        return None


class CSVFileLoader(DocumentLoader):
    """CSV/TSV 파일 로더"""
    
    def load(self, file_path: Path) -> Generator[Dict[str, str], None, None]:
        """CSV/TSV 파일 로드"""
        try:
            import pandas as pd
            
            delimiter = '\t' if file_path.suffix == '.tsv' else ','
            df = pd.read_csv(file_path, delimiter=delimiter)
            
            json_loader = JSONFileLoader()
            for _, row in df.iterrows():
                doc = json_loader._extract_qa(row.to_dict())
                if doc:
                    yield doc
                    
        except ImportError:
            logger.warning("pandas not installed, skipping CSV file")
        except Exception as e:
            logger.warning(f"Error loading CSV file {file_path}: {e}")


class DocumentLoaderFactory:
    """문서 로더 팩토리"""
    
    @staticmethod
    def get_loader(file_path: Path) -> Optional[DocumentLoader]:
        """파일 확장자에 따른 적절한 로더 반환"""
        ext = file_path.suffix.lower()
        
        if ext == '.txt':
            return QAFileLoader()
        elif ext in ['.json', '.jsonl']:
            return JSONFileLoader()
        elif ext in ['.csv', '.tsv']:
            return CSVFileLoader()
        else:
            logger.warning(f"Unsupported file extension: {ext}")
            return None


class DatasetLoader:
    """데이터셋 로더"""
    
    def __init__(self, paths: List[str], file_extensions: List[str], max_documents: int = 10000):
        """
        데이터셋 로더 초기화
        
        Args:
            paths: 검색할 경로 리스트
            file_extensions: 지원할 파일 확장자 패턴
            max_documents: 최대 문서 수
        """
        self.paths = paths
        self.file_extensions = file_extensions
        self.max_documents = max_documents
        
    def load_documents(self) -> List[Dict[str, str]]:
        """모든 문서 로드"""
        documents = []
        
        # 데이터셋 경로 찾기
        dataset_path = self._find_dataset_path()
        if not dataset_path:
            logger.warning("No dataset path found")
            return documents
        
        # 파일 수집
        all_files = self._collect_files(dataset_path)
        logger.info(f"Found {len(all_files)} files to process")
        
        # 문서 로드
        for file_path in all_files:
            if len(documents) >= self.max_documents:
                logger.info(f"Reached max documents limit: {self.max_documents}")
                break
                
            loader = DocumentLoaderFactory.get_loader(file_path)
            if loader:
                file_docs = 0
                for doc in loader.load(file_path):
                    documents.append(doc)
                    file_docs += 1
                    
                    if len(documents) >= self.max_documents:
                        break
                        
                if file_docs > 0:
                    logger.info(f"Loaded {file_docs} documents from {file_path.name}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def _find_dataset_path(self) -> Optional[Path]:
        """데이터셋 경로 찾기"""
        import os
        
        logger.info(f"Current working directory: {os.getcwd()}")
        
        for path_str in self.paths:
            path = Path(path_str)
            logger.debug(f"Checking path: {path.absolute()}")
            
            if path.exists() and path.is_dir():
                logger.info(f"Found dataset path: {path.absolute()}")
                return path
                
        return None
    
    def _collect_files(self, dataset_path: Path) -> List[Path]:
        """파일 수집"""
        all_files = []
        
        for pattern in self.file_extensions:
            files = list(dataset_path.rglob(pattern))
            if files:
                all_files.extend(files)
                logger.debug(f"Found {len(files)} files matching {pattern}")
        
        # 숨김 파일 제외
        all_files = [f for f in all_files if not f.name.startswith('.')]
        
        return sorted(all_files)