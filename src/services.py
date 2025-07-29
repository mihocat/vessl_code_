#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Service classes for the RAG system
RAG 시스템 서비스 클래스
"""

import logging
from typing import List, Dict, Optional

try:
    from duckduckgo_search import DDGS
except ImportError:
    # Fallback to old import name
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None
        logging.warning("DuckDuckGo search not available. Install with: pip install duckduckgo-search")

from config import WebSearchConfig
from rag_system import SearchResult

logger = logging.getLogger(__name__)


class WebSearchService:
    """웹 검색 서비스"""
    
    def __init__(self, config: Optional[WebSearchConfig] = None):
        """
        웹 검색 서비스 초기화
        
        Args:
            config: 웹 검색 설정
        """
        self.config = config or WebSearchConfig()
        
    def search(self, query: str) -> List[Dict[str, str]]:
        """
        웹 검색 수행
        
        Args:
            query: 검색 쿼리
            
        Returns:
            검색 결과 리스트
        """
        if DDGS is None:
            logger.warning("Web search not available - DDGS not installed")
            return []
            
        try:
            results = []
            
            with DDGS() as ddgs:
                search_results = ddgs.text(
                    query, 
                    max_results=self.config.max_results
                )
                
                for result in search_results:
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("body", "")[:200],
                        "url": result.get("href", "")
                    })
            
            logger.debug(f"Web search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []


class ResponseGenerator:
    """응답 생성기"""
    
    def __init__(self, config: WebSearchConfig):
        """
        응답 생성기 초기화
        
        Args:
            config: 웹 검색 설정 (컨텍스트 제한 포함)
        """
        self.config = config
        
    def prepare_context(
        self, 
        rag_results: List[SearchResult],
        web_results: List[Dict[str, str]]
    ) -> str:
        """
        LLM을 위한 컨텍스트 준비
        
        Args:
            rag_results: RAG 검색 결과
            web_results: 웹 검색 결과
            
        Returns:
            준비된 컨텍스트 문자열
        """
        context_parts = []
        
        # RAG 결과 처리
        for i, result in enumerate(rag_results[:3]):
            if result.score >= 0.4:
                answer_preview = result.answer[:self.config.db_context_char_limit]
                if len(result.answer) > self.config.db_context_char_limit:
                    answer_preview += "..."
                context_parts.append(f"참고 {i+1}: {answer_preview}")
        
        # 웹 검색 결과 처리
        for i, web in enumerate(web_results[:2]):
            snippet = web['snippet'][:self.config.web_context_char_limit]
            if len(web['snippet']) > self.config.web_context_char_limit:
                snippet += "..."
            context_parts.append(f"{web['title']}: {snippet}")
        
        # 전체 컨텍스트 길이 제한
        full_context = "\n\n".join(context_parts)
        if len(full_context) > self.config.context_limit:
            full_context = full_context[:self.config.context_limit] + "..."
            
        return full_context
    
    def generate_prompt(
        self,
        question: str,
        context: str,
        confidence_level: str
    ) -> str:
        """
        프롬프트 생성
        
        Args:
            question: 사용자 질문
            context: 참고 컨텍스트
            confidence_level: 신뢰도 수준 (high/medium/low)
            
        Returns:
            생성된 프롬프트
        """
        if confidence_level == "high":
            # 높은 신뢰도 - 직접 답변 사용
            return ""
            
        elif confidence_level == "medium":
            # 중간 신뢰도 - 재구성
            return f"""질문: {question}

참고자료:
{context}

위 질문에 대해 참고자료를 활용하여 답변하세요.
- 2-3문단으로 간결하게
- 전문용어는 정확히
- 한국어로 자연스럽게"""
            
        else:
            # 낮은 신뢰도 - 일반적 답변
            if context:
                return f"""질문: {question}

참고: {context}

위 질문에 대해 간결하게 답변하세요."""
            else:
                return f"""질문: {question}

위 질문에 대해 2-3문단으로 간결하게 답변하세요."""