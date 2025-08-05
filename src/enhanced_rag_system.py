#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced RAG System with Modern AI Patterns
향상된 RAG 시스템 - 최신 AI 패턴 적용
"""

import logging
import time
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from sentence_transformers import SentenceTransformer
import chromadb

from config import RAGConfig
from rag_system import RAGSystem, SearchResult
from intent_analyzer import IntentAnalyzer, IntentAnalysisResult, ProcessingMode

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """검색 전략"""
    SEMANTIC = "semantic"              # 의미적 검색
    KEYWORD = "keyword"                # 키워드 검색
    HYBRID = "hybrid"                  # 하이브리드 검색
    MULTI_QUERY = "multi_query"        # 다중 쿼리 검색
    CHAIN_OF_THOUGHT = "chain_of_thought"  # 사고 연쇄 검색


@dataclass 
class EnhancedSearchResult(SearchResult):
    """향상된 검색 결과"""
    retrieval_strategy: RetrievalStrategy
    reasoning_path: List[str]
    confidence_breakdown: Dict[str, float]
    evidence_strength: float
    domain_relevance: float


class QueryExpander:
    """쿼리 확장기 - 검색 성능 향상을 위한 쿼리 변형"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
    def expand_query(self, original_query: str, intent: IntentAnalysisResult) -> List[str]:
        """
        쿼리 확장
        
        Args:
            original_query: 원본 쿼리
            intent: 의도 분석 결과
            
        Returns:
            확장된 쿼리 리스트
        """
        expanded_queries = [original_query]
        
        # 1. 동의어 확장
        expanded_queries.extend(self._synonym_expansion(original_query))
        
        # 2. 도메인 특화 확장
        if intent.metadata.get('domain_hints'):
            expanded_queries.extend(
                self._domain_specific_expansion(original_query, intent.metadata['domain_hints'])
            )
        
        # 3. 구조적 변형
        expanded_queries.extend(self._structural_variations(original_query))
        
        # 4. LLM 기반 확장 (가능한 경우)
        if self.llm_client:
            llm_expansions = self._llm_based_expansion(original_query, intent)
            expanded_queries.extend(llm_expansions)
        
        # 중복 제거 및 정리
        unique_queries = list(dict.fromkeys(expanded_queries))
        
        logger.info(f"Query expanded from 1 to {len(unique_queries)} variants")
        return unique_queries[:10]  # 최대 10개로 제한
    
    def _synonym_expansion(self, query: str) -> List[str]:
        """동의어 확장"""
        synonym_map = {
            '전력': ['파워', '전기력', '전력량'],
            '전압': ['볼트', '전위차', 'V'],
            '전류': ['암페어', '전류량', 'A'],
            '저항': ['옴', '임피던스', 'R'],
            '계산': ['산출', '연산', '구하기'],
            '방법': ['방식', '절차', '과정'],
            '설명': ['해설', '기술', '서술'],
            '분석': ['해석', '검토', '평가'],
        }
        
        expanded = []
        for original, synonyms in synonym_map.items():
            if original in query:
                for synonym in synonyms:
                    expanded.append(query.replace(original, synonym))
        
        return expanded
    
    def _domain_specific_expansion(self, query: str, domains: List[str]) -> List[str]:
        """도메인 특화 확장"""
        expanded = []
        
        for domain in domains:
            if domain == "electrical_engineering":
                # 전기공학 용어 추가
                if '계산' in query:
                    expanded.append(query + " 공식")
                    expanded.append(query + " 수식")
                if '설명' in query:
                    expanded.append(query + " 이론")
                    expanded.append(query + " 원리")
            
            elif domain == "mathematics":
                # 수학 용어 추가
                if '함수' in query:
                    expanded.append(query + " 정의")
                    expanded.append(query + " 성질")
                if '적분' in query or '미분' in query:
                    expanded.append(query + " 공식")
                    expanded.append(query + " 계산법")
        
        return expanded
    
    def _structural_variations(self, query: str) -> List[str]:
        """구조적 변형"""
        variations = []
        
        # 질문 형태 변형
        if query.endswith('?'):
            # 평서문으로 변형
            base = query[:-1]
            variations.append(f"{base}에 대해")
            variations.append(f"{base}는 무엇인가")
        else:
            # 질문 형태로 변형
            variations.append(f"{query}는 무엇인가?")
            variations.append(f"{query}에 대해 설명해주세요")
        
        # 키워드 강조
        important_terms = ['계산', '공식', '방법', '설명', '분석']
        for term in important_terms:
            if term in query:
                variations.append(query.replace(term, f"**{term}**"))
        
        return variations
    
    def _llm_based_expansion(self, query: str, intent: IntentAnalysisResult) -> List[str]:
        """LLM 기반 쿼리 확장"""
        if not self.llm_client:
            return []
        
        try:
            prompt = f"""
다음 질문을 다양한 방식으로 표현해주세요. 의미는 동일하게 유지하면서 다른 단어나 구조를 사용하세요.

원본 질문: {query}
질문 유형: {intent.query_type.value}
복잡도: {intent.complexity_level}

3가지 다른 표현을 제공해주세요:
1.
2.
3.
"""
            
            response = self.llm_client.query(prompt, "")
            
            # 응답에서 변형된 질문 추출
            lines = response.split('\n')
            variations = []
            for line in lines:
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                    variation = line[2:].strip()
                    if variation and variation != query:
                        variations.append(variation)
            
            return variations
            
        except Exception as e:
            logger.warning(f"LLM-based query expansion failed: {e}")
            return []


class MultiRetrievalRAG:
    """다중 검색 전략을 사용하는 향상된 RAG 시스템"""
    
    def __init__(self, base_rag: RAGSystem, config: RAGConfig):
        self.base_rag = base_rag
        self.config = config
        self.intent_analyzer = IntentAnalyzer()
        self.query_expander = QueryExpander(base_rag.llm_client)
        
        # 검색 전략별 가중치
        self.strategy_weights = {
            RetrievalStrategy.SEMANTIC: 0.4,
            RetrievalStrategy.KEYWORD: 0.3,
            RetrievalStrategy.HYBRID: 0.3
        }
    
    async def enhanced_search(
        self, 
        query: str, 
        k: int = 10,
        has_image: bool = False,
        context: Optional[Dict] = None
    ) -> Tuple[List[EnhancedSearchResult], float]:
        """
        향상된 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            has_image: 이미지 포함 여부
            context: 추가 컨텍스트
            
        Returns:
            (향상된 검색 결과, 최고 점수)
        """
        start_time = time.time()
        
        # 1. 의도 분석
        intent = self.intent_analyzer.analyze_intent(query, has_image, context)
        logger.info(f"Intent analysis completed: {intent.query_type.value}")
        
        # 2. 검색 전략 결정
        strategies = self._determine_strategies(intent)
        
        # 3. 쿼리 확장
        expanded_queries = self.query_expander.expand_query(query, intent)
        
        # 4. 병렬 검색 수행
        all_results = await self._parallel_search(expanded_queries, strategies, k)
        
        # 5. 결과 융합 및 재순위
        final_results = self._fuse_and_rerank(all_results, intent, k)
        
        # 6. 최고 점수 계산
        max_score = max([r.score for r in final_results]) if final_results else 0.0
        
        elapsed_time = time.time() - start_time
        logger.info(f"Enhanced search completed in {elapsed_time:.2f}s: "
                   f"{len(final_results)} results, max score: {max_score:.3f}")
        
        return final_results, max_score
    
    def _determine_strategies(self, intent: IntentAnalysisResult) -> List[RetrievalStrategy]:
        """검색 전략 결정"""
        strategies = [RetrievalStrategy.SEMANTIC]  # 기본 전략
        
        if intent.processing_mode == ProcessingMode.MULTI_AGENT:
            strategies.extend([RetrievalStrategy.KEYWORD, RetrievalStrategy.HYBRID])
        
        if intent.complexity_level >= 4:
            strategies.append(RetrievalStrategy.MULTI_QUERY)
        
        if intent.requires_reasoning:
            strategies.append(RetrievalStrategy.CHAIN_OF_THOUGHT)
        
        return list(set(strategies))  # 중복 제거
    
    async def _parallel_search(
        self, 
        queries: List[str], 
        strategies: List[RetrievalStrategy], 
        k: int
    ) -> Dict[RetrievalStrategy, List[SearchResult]]:
        """병렬 검색 수행"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 각 전략별로 검색 작업 제출
            future_to_strategy = {}
            
            for strategy in strategies:
                if strategy == RetrievalStrategy.SEMANTIC:
                    future = executor.submit(self._semantic_search, queries[0], k * 2)
                elif strategy == RetrievalStrategy.KEYWORD:
                    future = executor.submit(self._keyword_search, queries[0], k * 2)
                elif strategy == RetrievalStrategy.HYBRID:
                    future = executor.submit(self._hybrid_search, queries[0], k * 2)
                elif strategy == RetrievalStrategy.MULTI_QUERY:
                    future = executor.submit(self._multi_query_search, queries, k)
                elif strategy == RetrievalStrategy.CHAIN_OF_THOUGHT:
                    future = executor.submit(self._chain_of_thought_search, queries[0], k)
                
                future_to_strategy[future] = strategy
            
            # 결과 수집
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    strategy_results = future.result(timeout=30)
                    results[strategy] = strategy_results
                    logger.info(f"{strategy.value} search completed: {len(strategy_results)} results")
                except Exception as e:
                    logger.error(f"{strategy.value} search failed: {e}")
                    results[strategy] = []
        
        return results
    
    def _semantic_search(self, query: str, k: int) -> List[SearchResult]:
        """의미적 검색 (기존 RAG 시스템 사용)"""
        results, _ = self.base_rag.search(query, k)
        return results
    
    def _keyword_search(self, query: str, k: int) -> List[SearchResult]:
        """키워드 기반 검색"""
        # 키워드 추출
        keywords = query.split()
        
        # 각 키워드로 개별 검색 후 합성
        all_results = []
        for keyword in keywords:
            if len(keyword) > 2:  # 2자 이상의 키워드만
                results, _ = self.base_rag.search(keyword, k // len(keywords) + 1)
                all_results.extend(results)
        
        # 중복 제거 및 점수 조정
        unique_results = {}
        for result in all_results:
            key = result.question
            if key in unique_results:
                # 점수 합성
                unique_results[key].score = max(unique_results[key].score, result.score)
            else:
                unique_results[key] = result
        
        return list(unique_results.values())[:k]
    
    def _hybrid_search(self, query: str, k: int) -> List[SearchResult]:
        """하이브리드 검색 (의미적 + 키워드)"""
        semantic_results = self._semantic_search(query, k // 2)
        keyword_results = self._keyword_search(query, k // 2)
        
        # 결과 병합
        all_results = semantic_results + keyword_results
        
        # 중복 제거 및 점수 조정
        unique_results = {}
        for result in all_results:
            key = result.question
            if key in unique_results:
                # 가중 평균으로 점수 합성
                old_score = unique_results[key].score
                new_score = (old_score + result.score) / 2
                unique_results[key].score = new_score
            else:
                unique_results[key] = result
        
        # 점수 기준 정렬
        sorted_results = sorted(unique_results.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:k]
    
    def _multi_query_search(self, queries: List[str], k: int) -> List[SearchResult]:
        """다중 쿼리 검색"""
        all_results = []
        
        for query in queries:
            results, _ = self.base_rag.search(query, k // len(queries) + 1)
            # 각 쿼리별로 가중치 적용
            weight = 1.0 / len(queries)
            for result in results:
                result.score *= weight
            all_results.extend(results)
        
        # 결과 병합 및 정렬
        unique_results = {}
        for result in all_results:
            key = result.question
            if key in unique_results:
                unique_results[key].score += result.score
            else:
                unique_results[key] = result
        
        sorted_results = sorted(unique_results.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:k]
    
    def _chain_of_thought_search(self, query: str, k: int) -> List[SearchResult]:
        """사고 연쇄 검색"""
        # 쿼리를 하위 질문으로 분해
        sub_queries = self._decompose_query(query)
        
        all_results = []
        for sub_query in sub_queries:
            results, _ = self.base_rag.search(sub_query, k // len(sub_queries) + 1)
            all_results.extend(results)
        
        return all_results[:k]
    
    def _decompose_query(self, query: str) -> List[str]:
        """복잡한 쿼리를 하위 질문으로 분해"""
        # 간단한 분해 로직 (실제로는 LLM 사용 가능)
        sub_queries = [query]
        
        # '그리고', '또한' 등으로 분리
        if '그리고' in query:
            parts = query.split('그리고')
            sub_queries.extend([part.strip() for part in parts if part.strip()])
        
        # '어떻게', '왜' 등의 키워드로 세분화
        if '어떻게' in query and '왜' in query:
            base = query.replace('어떻게', '').replace('왜', '').strip()
            sub_queries.append(f"어떻게 {base}")
            sub_queries.append(f"왜 {base}")
        
        return list(set(sub_queries))
    
    def _fuse_and_rerank(
        self, 
        strategy_results: Dict[RetrievalStrategy, List[SearchResult]], 
        intent: IntentAnalysisResult, 
        k: int
    ) -> List[EnhancedSearchResult]:
        """검색 결과 융합 및 재순위"""
        # 모든 결과 수집
        all_results = {}
        
        for strategy, results in strategy_results.items():
            weight = self.strategy_weights.get(strategy, 0.2)
            
            for result in results:
                key = result.question
                
                if key not in all_results:
                    # 향상된 검색 결과로 변환
                    enhanced_result = EnhancedSearchResult(
                        question=result.question,
                        answer=result.answer,
                        score=result.score * weight,
                        category=result.category,
                        metadata=result.metadata,
                        retrieval_strategy=strategy,
                        reasoning_path=[f"Retrieved via {strategy.value}"],
                        confidence_breakdown={strategy.value: result.score},
                        evidence_strength=result.score,
                        domain_relevance=self._calculate_domain_relevance(result, intent)
                    )
                    all_results[key] = enhanced_result
                else:
                    # 기존 결과와 병합
                    existing = all_results[key]
                    existing.score = max(existing.score, result.score * weight)
                    existing.reasoning_path.append(f"Also found via {strategy.value}")
                    existing.confidence_breakdown[strategy.value] = result.score
                    existing.evidence_strength = max(existing.evidence_strength, result.score)
        
        # 최종 점수 계산 및 정렬
        final_results = list(all_results.values())
        for result in final_results:
            result.score = self._calculate_final_score(result, intent)
        
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results[:k]
    
    def _calculate_domain_relevance(
        self, 
        result: SearchResult, 
        intent: IntentAnalysisResult
    ) -> float:
        """도메인 관련성 계산"""
        relevance = 0.5  # 기본 관련성
        
        if intent.metadata.get('domain_hints'):
            domains = intent.metadata['domain_hints']
            
            # 결과 텍스트에서 도메인 키워드 검사
            text = (result.question + " " + result.answer).lower()
            
            for domain in domains:
                if domain == "electrical_engineering":
                    keywords = ['전력', '전압', '전류', '저항', '모터', '변압기']
                elif domain == "mathematics":
                    keywords = ['함수', '적분', '미분', '행렬', '벡터', '방정식']
                elif domain == "physics":
                    keywords = ['힘', '에너지', '운동', '파동', '열역학']
                else:
                    keywords = []
                
                matches = sum(1 for keyword in keywords if keyword in text)
                relevance += matches * 0.1
        
        return min(relevance, 1.0)
    
    def _calculate_final_score(
        self, 
        result: EnhancedSearchResult, 
        intent: IntentAnalysisResult
    ) -> float:
        """최종 점수 계산"""
        base_score = result.score
        
        # 도메인 관련성 보너스
        domain_bonus = result.domain_relevance * 0.1
        
        # 증거 강도 보너스
        evidence_bonus = result.evidence_strength * 0.05
        
        # 다중 전략 발견 보너스
        multi_strategy_bonus = len(result.confidence_breakdown) * 0.02
        
        final_score = base_score + domain_bonus + evidence_bonus + multi_strategy_bonus
        
        return min(final_score, 1.0)


# 향상된 RAG 시스템 래퍼
class EnhancedRAGSystem:
    """향상된 RAG 시스템 메인 클래스"""
    
    def __init__(self, base_rag: RAGSystem, config: RAGConfig):
        self.base_rag = base_rag
        self.multi_retrieval = MultiRetrievalRAG(base_rag, config)
        
    async def search(
        self, 
        query: str, 
        k: int = 10,
        has_image: bool = False,
        context: Optional[Dict] = None
    ) -> Tuple[List[EnhancedSearchResult], float]:
        """향상된 검색 인터페이스"""
        return await self.multi_retrieval.enhanced_search(query, k, has_image, context)
    
    def search_sync(
        self, 
        query: str, 
        k: int = 10,
        has_image: bool = False,
        context: Optional[Dict] = None
    ) -> Tuple[List[EnhancedSearchResult], float]:
        """동기 검색 인터페이스"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.search(query, k, has_image, context)
            )
        finally:
            loop.close()
    
    def get_stats(self) -> Dict:
        """통계 반환"""
        return self.base_rag.get_stats()


if __name__ == "__main__":
    # 테스트 예시
    from config import Config
    from llm_client_openai import LLMClient
    
    config = Config()
    llm_client = LLMClient(config.llm)
    base_rag = RAGSystem(config.rag, config.dataset, llm_client)
    
    enhanced_rag = EnhancedRAGSystem(base_rag, config.rag)
    
    # 테스트 쿼리
    test_query = "전력 계산 방법을 설명해주세요"
    results, max_score = enhanced_rag.search_sync(test_query, k=5)
    
    print(f"Found {len(results)} results with max score: {max_score:.3f}")
    for i, result in enumerate(results):
        print(f"{i+1}. {result.question} (score: {result.score:.3f})")
        print(f"   Strategy: {result.retrieval_strategy.value}")
        print(f"   Domain relevance: {result.domain_relevance:.3f}")