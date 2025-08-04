#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent RAG Adapter
Intelligent RAG 시스템을 기존 시스템과 통합하는 어댑터
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import time

from config import Config
from llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class QueryComplexity:
    """쿼리 복잡도 평가 결과"""
    score: float  # 0.0 ~ 1.0
    factors: Dict[str, float]
    should_use_intelligent: bool


class IntelligentRAGAdapter:
    """Intelligent RAG 시스템 어댑터"""
    
    def __init__(self, config: Config, llm_client: LLMClient):
        """
        어댑터 초기화
        
        Args:
            config: 전체 설정 객체
            llm_client: LLM 클라이언트
        """
        self.config = config
        self.llm_client = llm_client
        self.enabled = config.rag.use_intelligent_rag
        self.mode = config.rag.intelligent_rag_mode
        self.features = config.rag.intelligent_features
        
        self.orchestrator = None
        self.initialized = False
        
        # 복잡도 평가 기준
        self.complexity_keywords = {
            'high': ['비교', '분석', '설명', '차이점', '원리', '이유', '어떻게', '왜'],
            'medium': ['무엇', '언제', '어디', '누가', '정의', '개념'],
            'low': ['예/아니오', '숫자', '날짜', '이름']
        }
    
    async def initialize(self):
        """비동기 초기화"""
        if not self.enabled or self.initialized:
            return
        
        try:
            # Intelligent RAG 시스템을 동적으로 임포트
            from intelligent_rag_system import IntelligentRAGOrchestrator
            
            # 의존성 문제 해결을 위한 모킹
            self._mock_dependencies()
            
            self.orchestrator = IntelligentRAGOrchestrator(
                config=self.config,
                llm_client=self.llm_client
            )
            
            await self.orchestrator.initialize()
            self.initialized = True
            logger.info("Intelligent RAG Adapter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Intelligent RAG: {e}")
            self.enabled = False
    
    def _mock_dependencies(self):
        """의존성 문제 해결을 위한 임시 모킹"""
        import sys
        from types import ModuleType
        
        # universal_ocr_pipeline 모킹
        if 'universal_ocr_pipeline' not in sys.modules:
            mock_module = ModuleType('universal_ocr_pipeline')
            
            class DomainAdaptiveOCR:
                def __init__(self):
                    pass
                
                def process(self, image):
                    return {"text": "", "confidence": 0.0}
            
            mock_module.DomainAdaptiveOCR = DomainAdaptiveOCR
            sys.modules['universal_ocr_pipeline'] = mock_module
    
    def evaluate_complexity(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryComplexity:
        """
        쿼리 복잡도 평가
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트
            
        Returns:
            QueryComplexity: 복잡도 평가 결과
        """
        factors = {
            'length': min(len(query) / 200, 1.0),  # 쿼리 길이
            'keywords': 0.0,  # 복잡도 키워드
            'multimodal': 0.0,  # 멀티모달 여부
            'context': 0.0  # 컨텍스트 복잡도
        }
        
        # 키워드 기반 평가
        query_lower = query.lower()
        for level, keywords in self.complexity_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if level == 'high':
                        factors['keywords'] = max(factors['keywords'], 0.8)
                    elif level == 'medium':
                        factors['keywords'] = max(factors['keywords'], 0.5)
                    else:
                        factors['keywords'] = max(factors['keywords'], 0.2)
        
        # 멀티모달 평가
        if context and context.get('image') is not None:
            factors['multimodal'] = 0.7
        
        # 컨텍스트 평가
        if context and context.get('conversation_history'):
            factors['context'] = min(len(context['conversation_history']) / 5, 1.0)
        
        # 총점 계산
        score = sum(factors.values()) / len(factors)
        
        # 임계값 확인
        threshold = self.features.get('complexity_threshold', 0.7)
        should_use = score >= threshold and self.mode != 'never'
        
        if self.mode == 'always':
            should_use = True
        
        return QueryComplexity(
            score=score,
            factors=factors,
            should_use_intelligent=should_use
        )
    
    def should_use_intelligent(self, query: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Intelligent RAG 사용 여부 결정
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트
            
        Returns:
            bool: 사용 여부
        """
        if not self.enabled:
            return False
        
        if self.mode == 'always':
            return True
        elif self.mode == 'never':
            return False
        else:  # adaptive
            complexity = self.evaluate_complexity(query, context)
            return complexity.should_use_intelligent
    
    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Intelligent RAG 처리
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트
            
        Returns:
            Dict: 처리 결과
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.orchestrator:
            raise RuntimeError("Intelligent RAG Orchestrator not initialized")
        
        start_time = time.time()
        
        try:
            # Intelligent RAG 처리
            result = await self.orchestrator.process_async(
                query=query,
                image=context.get('image') if context else None,
                context=context
            )
            
            # 처리 시간 추가
            result['processing_time'] = time.time() - start_time
            result['used_intelligent'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Intelligent RAG processing failed: {e}")
            raise
    
    def process_sync(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        동기적 처리 (기존 인터페이스 호환)
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트
            
        Returns:
            Dict: 처리 결과
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process(query, context))
        finally:
            loop.close()