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
    model_name: str = "test_model"
    # model_name: str = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
    max_tokens: int = 800
    temperature: float = 0.1
    top_p: float = 0.85
    presence_penalty: float = 0.05
    frequency_penalty: float = 0.05
    repetition_penalty: float = 1.2  # 반복 방지를 위한 penalty 추가
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
    
    # 점수 임계값 (현실적인 값으로 조정)
    high_confidence_threshold: float = 0.75  # 높은 신뢰도
    medium_confidence_threshold: float = 0.5   # 중간 신뢰도
    low_confidence_threshold: float = 0.25    # 낮은 신뢰도
    
    # 점수 조정 가중치
    keyword_match_bonus: float = 0.1
    substring_match_bonus: float = 0.05
    category_bonus: float = 0.2
    
    # Intelligent RAG 설정
    use_intelligent_rag: bool = False
    intelligent_rag_mode: str = "adaptive"  # adaptive, always, never
    intelligent_features: dict = None  # 런타임에 초기화
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.intelligent_features is None:
            self.intelligent_features = {
                "intent_detection": True,
                "knowledge_graph": False,
                "adaptive_response": True,
                "complexity_threshold": 0.25  # 테스트 결과에 따라 조정
            }


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
class OpenAIConfig:
    """OpenAI API 설정"""
    api_key: Optional[str] = None
    vision_model: str = "gpt-4o"  # gpt-4-vision-preview, gpt-4o, gpt-4o-mini
    text_model: str = "gpt-4-turbo-preview"  # gpt-4-turbo, gpt-3.5-turbo
    max_tokens: int = 1000
    temperature: float = 0.2
    use_vision_api: bool = False  # Vision API 사용 여부 - API 권한 문제로 비활성화
    use_for_llm: bool = False  # LLM 응답 생성에 사용 여부 - API 권한 문제로 비활성화
    
    def __post_init__(self):
        """초기화 후 처리"""
        # 환경 변수에서 API 키 로드
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class AppConfig:
    """애플리케이션 설정"""
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False
    
    # UI 설정
    title: str = "AI 테스트"
    description: str = """
    # AI 테스트
    """
    
    # 이미지 분석 설정
    use_simple_analyzer: bool = False  # OCR 엔진 우선 사용으로 변경
    force_ocr_engines: bool = True  # OCR 엔진 강제 사용
    
    # 예제 질문
    example_questions: List[str] = None
    
    def __post_init__(self):
        if self.example_questions is None:
            self.example_questions = [
                "다산에듀는 너의 친구입니까?",
                "회로도에 대해서 알려줘.",
                "객체지향 프로그래밍의 특징은?"
            ]


class Config:
    """전체 설정 관리"""
    def __init__(self):
        # API 키 설정 (Storage 우선)
        self._setup_api_keys()
        
        self.llm = LLMConfig()
        self.rag = RAGConfig()
        self.dataset = DatasetConfig()
        self.web_search = WebSearchConfig()
        self.openai = OpenAIConfig()
        self.app = AppConfig()
        
        # 환경 변수 오버라이드
        self._load_from_env()
    
    def _setup_api_keys(self):
        """API 키 초기 설정"""
        try:
            from api_key_loader import setup_api_keys
            setup_api_keys()
        except Exception as e:
            print(f"Warning: Failed to load API keys from storage: {e}")
    
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
        
        # Intelligent RAG 설정
        if os.getenv("USE_INTELLIGENT_RAG"):
            self.rag.use_intelligent_rag = os.getenv("USE_INTELLIGENT_RAG").lower() == "true"
        if os.getenv("INTELLIGENT_RAG_MODE"):
            self.rag.intelligent_rag_mode = os.getenv("INTELLIGENT_RAG_MODE")
        
        # 데이터셋 경로
        if os.getenv("DATASET_PATH"):
            self.dataset.paths = [os.getenv("DATASET_PATH")]
        
        # 서버 설정
        if os.getenv("SERVER_PORT"):
            self.app.server_port = int(os.getenv("SERVER_PORT"))
        
        # OpenAI 설정
        if os.getenv("OPENAI_API_KEY"):
            self.openai.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_VISION_MODEL"):
            self.openai.vision_model = os.getenv("OPENAI_VISION_MODEL")
        if os.getenv("OPENAI_TEXT_MODEL"):
            self.openai.text_model = os.getenv("OPENAI_TEXT_MODEL")
        if os.getenv("USE_OPENAI_VISION"):
            self.openai.use_vision_api = os.getenv("USE_OPENAI_VISION").lower() == "true"
        if os.getenv("USE_OPENAI_LLM"):
            self.openai.use_for_llm = os.getenv("USE_OPENAI_LLM").lower() == "true"


# 싱글톤 인스턴스
config = Config()
