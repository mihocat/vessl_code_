#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Service classes for the RAG system
RAG 시스템 서비스 클래스
"""

import logging
from typing import List, Dict, Optional
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup

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
        웹 검색 수행 (구글 한국어, 네이버 우선순위)
        
        Args:
            query: 검색 쿼리
            
        Returns:
            검색 결과 리스트
        """
        results = []
        
        # 1. 구글 검색 시도 (한국어 우선)
        google_results = self._search_google(query)
        if google_results:
            results.extend(google_results)
            if len(results) >= self.config.max_results:
                return results[:self.config.max_results]
        
        # 2. 네이버 검색 시도 (구글 결과가 부족한 경우)
        needed = self.config.max_results - len(results)
        if needed > 0:
            naver_results = self._search_naver(query)
            if naver_results:
                results.extend(naver_results[:needed])
        
        # 3. DuckDuckGo 검색 시도 (결과가 여전히 부족한 경우)
        needed = self.config.max_results - len(results)
        if needed > 0 and DDGS is not None:
            ddg_results = self._search_duckduckgo(query)
            if ddg_results:
                results.extend(ddg_results[:needed])
        
        logger.debug(f"Web search for '{query}' returned {len(results)} results")
        return results
    
    def _search_naver(self, query: str) -> List[Dict[str, str]]:
        """네이버 검색"""
        try:
            # 네이버 검색 URL
            url = f"https://search.naver.com/search.naver?query={quote(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # 검색 결과 추출 (네이버 DOM 구조에 맞게)
            search_items = soup.select('.lst_total .total_wrap')[:self.config.max_results]
            
            for item in search_items:
                title_elem = item.select_one('.total_tit')
                desc_elem = item.select_one('.dsc_txt')
                link_elem = item.select_one('a')
                
                if title_elem and desc_elem:
                    results.append({
                        "title": title_elem.get_text(strip=True),
                        "snippet": desc_elem.get_text(strip=True)[:200],
                        "url": link_elem.get('href', '') if link_elem else ""
                    })
            
            logger.info(f"Naver search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Naver search failed: {e}")
            return []
    
    def _search_google(self, query: str) -> List[Dict[str, str]]:
        """구글 검색 (한국어 우선)"""
        try:
            # 구글 검색 URL (한국어 검색 우선)
            url = f"https://www.google.com/search?q={quote(query)}&hl=ko"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # 검색 결과 추출
            search_items = soup.select('div.g')[:self.config.max_results]
            
            for item in search_items:
                title_elem = item.select_one('h3')
                desc_elem = item.select_one('.VwiC3b')
                link_elem = item.select_one('a')
                
                if title_elem and desc_elem:
                    results.append({
                        "title": title_elem.get_text(strip=True),
                        "snippet": desc_elem.get_text(strip=True)[:200],
                        "url": link_elem.get('href', '') if link_elem else ""
                    })
            
            logger.info(f"Google search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []
    
    def _search_duckduckgo(self, query: str) -> List[Dict[str, str]]:
        """DuckDuckGo 검색 (폴백)"""
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
            
            logger.info(f"DuckDuckGo search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
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
        web_results: List[Dict[str, str]],
        image_context: Optional[Dict[str, str]] = None
    ) -> str:
        """
        LLM을 위한 컨텍스트 준비
        
        Args:
            rag_results: RAG 검색 결과
            web_results: 웹 검색 결과
            image_context: 이미지 분석 결과
            
        Returns:
            준비된 컨텍스트 문자열
        """
        context_parts = []
        
        # RAG 결과 처리 (적절히 제한)
        # 신뢰도에 따라 결과 수 조정
        num_results = 2 if rag_results and rag_results[0].score >= 0.8 else 1
        for i, result in enumerate(rag_results[:num_results]):
            if result.score >= 0.5:  # 임계값 상향
                # 답변 길이를 더 효율적으로 제한
                answer_preview = result.answer[:200] if len(result.answer) > 200 else result.answer
                # 중복 제거
                if answer_preview not in ' '.join(context_parts):
                    context_parts.append(f"[참고{i+1}] {answer_preview}")
                    if len(result.answer) > 200:
                        context_parts[-1] += "..."
        
        # 이미지 컨텍스트 처리
        if image_context:
            # 이미지 캡션과 OCR 텍스트 매우 짧게
            caption = image_context.get('caption', '')[:100]
            if caption:
                context_parts.append(f"\n[이미지] {caption}...")
            
            ocr_text = image_context.get('ocr_text', '')[:50]
            if ocr_text and len(ocr_text) > 10:  # 너무 짧은 OCR은 제외
                context_parts.append(f"[OCR] {ocr_text}...")
        
        # 웹 검색 결과 추가 (매우 짧게)
        if web_results and len(web_results) > 0:
            result = web_results[0]
            title = result.get('title', '')[:30]
            snippet = result.get('snippet', '')[:50]
            if title and snippet:
                context_parts.append(f"\n[웹] {title}: {snippet}...")
        
        # 컨텍스트 결합 및 길이 제한
        context = "\n".join(context_parts)
        max_context_length = 500  # 매우 짧게
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            
        return context
    
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

위 참고자료를 바탕으로 정확하게 답변해주세요.
전기공학 용어와 개념을 올바르게 사용해주세요."""
            
        else:
            # 낮은 신뢰도 - 일반적 답변
            if context:
                return f"""질문: {question}

참고자료:
{context}

참고자료를 참조하여 최대한 정확하게 답변해주세요.
만약 확실하지 않다면 그렇게 명시해주세요."""
            else:
                return f"""질문: {question}

전기공학 지식을 바탕으로 최대한 정확하게 답변해주세요.
확실하지 않은 부분은 그렇게 명시해주세요."""