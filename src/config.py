#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module for the RAG system
시스템 설정 모듈
"""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LLMConfig:
    """LLM 설정"""
    base_url: str = "http://localhost:8000"
    # model_name: str = "test_model"
    # model_name: str = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
    model_name: str = "rtzr/ko-gemma-2-9b-it"
    max_tokens: int = 800
    temperature: float = 0.1
    top_p: float = 0.85
    presence_penalty: float = 0.05
    frequency_penalty: float = 0.05
    timeout: int = 45
    health_check_timeout: int = 2


@dataclass
class RAGConfig:
    """RAG 시스템 설정"""
    embedding_model_name: str = "jinaai/jina-embeddings-v3"
    collection_name: str = "qa_collection_v3"
    chroma_db_path: str = "./chroma_db"
    max_documents: int = 10000
    batch_size: int = 500
    
    # 검색 설정
    search_k: int = 10
    rerank_k: int = 5
    
    # 점수 임계값
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.6
    low_confidence_threshold: float = 0.2
    
    # 점수 조정 가중치
    keyword_match_bonus: float = 0.1
    substring_match_bonus: float = 0.05
    category_bonus: float = 0.2


@dataclass
class DatasetConfig:
    """데이터셋 설정"""
    paths: List[str] = None
    file_extensions: List[str] = None
    
    def __post_init__(self):
        if self.paths is None:
            self.paths = [
                "/dataset",
                "./dataset",
                "./2_documents",
                "/data",
                "./data"
            ]
        if self.file_extensions is None:
            self.file_extensions = ['*.txt', '*.json', '*.jsonl', '*.csv', '*.tsv']


@dataclass
class WebSearchConfig:
    """웹 검색 설정"""
    max_results: int = 3
    context_limit: int = 500
    db_context_char_limit: int = 150
    web_context_char_limit: int = 100


@dataclass
class AppConfig:
    """애플리케이션 설정"""
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False
    
    # UI 설정
    title: str = "AI 챗봇"
    description: str = """
    # AI 챗봇
    
    **특징:**
    - 고성능 임베딩 모델 (jina-embeddings-v3)
    - 향상된 검색 정확도
    - 실시간 웹 검색 통합
    """
    
    # 예제 질문
    example_questions: List[str] = None
    
    def __post_init__(self):
        if self.example_questions is None:
            self.example_questions = [
                "파이썬에서 리스트와 튜플의 차이는?",
                "머신러닝과 딥러닝의 차이점은?",
                "REST API란 무엇인가요?",
                "데이터베이스 정규화란?",
                "객체지향 프로그래밍의 특징은?"
            ]


class Config:
    """전체 설정 관리"""
    def __init__(self):
        self.llm = LLMConfig()
        self.rag = RAGConfig()
        self.dataset = DatasetConfig()
        self.web_search = WebSearchConfig()
        self.app = AppConfig()
        
        # 환경 변수 오버라이드
        self._load_from_env()
    
    def _load_from_env(self):
        """환경 변수에서 설정 로드"""
        # LLM 설정
        if os.getenv("LLM_BASE_URL"):
            self.llm.base_url = os.getenv("LLM_BASE_URL")
        if os.getenv("LLM_MODEL_NAME"):
            self.llm.model_name = os.getenv("LLM_MODEL_NAME")
        
        # RAG 설정
        if os.getenv("EMBEDDING_MODEL"):
            self.rag.embedding_model_name = os.getenv("EMBEDDING_MODEL")
        if os.getenv("CHROMA_DB_PATH"):
            self.rag.chroma_db_path = os.getenv("CHROMA_DB_PATH")
        
        # 데이터셋 경로
        if os.getenv("DATASET_PATH"):
            self.dataset.paths = [os.getenv("DATASET_PATH")]
        
        # 서버 설정
        if os.getenv("SERVER_PORT"):
            self.app.server_port = int(os.getenv("SERVER_PORT"))


# 싱글톤 인스턴스
config = Config()