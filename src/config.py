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
    """OpenAI API 설정 - 이미지+텍스트 분석 전용"""
    api_key: Optional[str] = None
    # GPT-5 통합 모델: 이미지+텍스트 분석 전용 (최종 답변 생성 금지)
    unified_model: str = "gpt-5"  # 이미지+텍스트 분석 전용
    max_tokens: int = 500  # 더 상세한 분석을 위해 증가
    temperature: float = 0.3  # 약간 더 창의적인 분석
    # 사용 제한 설정
    use_for_analysis_only: bool = True  # 분석 전용, 답변 생성 금지
    use_for_final_response: bool = False  # 최종 답변 생성 금지
    max_calls_per_query: int = 1  # 질의당 1회만 호출
    
    def __post_init__(self):
        """초기화 후 처리"""
        # 환경 변수에서 API 키 로드
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class NextGenConfig:
    """차세대 시스템 설정"""
    # Multi-Agent 설정
    enable_multi_agent: bool = True
    agent_timeout: int = 30
    max_parallel_agents: int = 4
    
    # Intent Analysis 설정
    enable_intent_analysis: bool = True
    intent_confidence_threshold: float = 0.8
    
    # Memory System 설정
    enable_memory_system: bool = True
    memory_retention_days: int = 30
    max_conversation_history: int = 100
    
    # Adaptive Response 설정
    enable_adaptive_response: bool = True
    response_personalization: bool = True
    
    # Performance 설정
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_response_time: int = 15


@dataclass
class AppConfig:
    """애플리케이션 설정"""
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False
    
    # UI 설정
    title: str = "🚀 Next-Generation AI Chatbot"
    description: str = """
    # 🚀 Next-Generation AI Chatbot
    
    **Advanced Multi-Agent System with Vision, RAG, and Reasoning**
    
    ✨ **Features:**
    - 🎯 Intelligent Intent Analysis
    - 🤖 Multi-Agent Collaboration  
    - 👁️ Advanced Vision Processing
    - 🧠 Adaptive Memory System
    - 🔄 Chain-of-Thought Reasoning
    """
    
    # 이미지 분석 설정
    use_simple_analyzer: bool = False  # OpenAI Vision API 우선 사용
    force_ocr_engines: bool = False  # OCR 엔진 선택적 사용
    
    # 예제 질문
    example_questions: List[str] = None
    
    def __post_init__(self):
        if self.example_questions is None:
            self.example_questions = [
                "이미지의 내용을 분석해주세요",
                "데이터 처리 방법을 단계별로 설명해주세요",
                "머신러닝의 기본 개념을 알려주세요",
                "이 이미지의 특징을 분석해주세요",
                "최신 AI 기술 트렌드를 설명해주세요"
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
        self.next_gen = NextGenConfig()
        
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
        if os.getenv("VLLM_API_URL"):
            self.llm.base_url = os.getenv("VLLM_API_URL")
        elif os.getenv("LLM_BASE_URL"):
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
        if os.getenv("OPENAI_UNIFIED_MODEL"):
            self.openai.unified_model = os.getenv("OPENAI_UNIFIED_MODEL")
        
        # 강제 설정: OpenAI는 분석 전용으로만 사용
        self.openai.use_for_analysis_only = True
        self.openai.use_for_final_response = False


# 싱글톤 인스턴스
config = Config()
