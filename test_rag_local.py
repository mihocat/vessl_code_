#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 시스템 로컬 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_system import ConcreteKoreanElectricalRAG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_system():
    """RAG 시스템 테스트"""
    logger.info("=== RAG 시스템 테스트 시작 ===")
    
    # RAG 시스템 초기화
    rag = ConcreteKoreanElectricalRAG(embedding_model_name="jinaai/jina-embeddings-v3")
    
    # 샘플 데이터로 테스트
    logger.info("샘플 데이터 로드...")
    rag._load_sample_data()
    
    # 테스트 질문들 (EXAMPLE.md 기반)
    test_queries = [
        "다산에듀는 무엇인가요?",
        "R-C회로 합성 임피던스에서 -j를 붙이는 이유는?",
        "과도현상과 인덕턴스 L의 관계는?",
        "댐 부속설비 중 수로와 여수로의 차이는?",
        "서보모터의 동작 원리는?",
        "옴의 법칙이 무엇인가요?",
        "변압기의 동작 원리를 설명해주세요"
    ]
    
    logger.info("\n=== 테스트 시작 ===")
    for query in test_queries:
        logger.info(f"\n질문: {query}")
        
        # 벡터 검색
        results, found = rag.search_vector_database(query, k=3)
        
        if found and results:
            logger.info(f"검색 결과 {len(results)}개 발견:")
            for i, result in enumerate(results[:2]):
                logger.info(f"\n결과 {i+1}:")
                logger.info(f"  점수: {result['final_score']:.3f}")
                logger.info(f"  카테고리: {result['doc_info']['category']}")
                logger.info(f"  질문: {result['doc_info']['question'][:50]}...")
                logger.info(f"  답변: {result['doc_info']['answer'][:100]}...")
        else:
            logger.info("검색 결과 없음")
            
            # 향상된 검색 파이프라인 테스트
            search_results, search_type = rag.enhanced_search_pipeline(query)
            logger.info(f"향상된 검색 타입: {search_type}")
            
            if search_type == "web_only":
                logger.info(f"웹 검색 결과: {len(search_results)}개")

if __name__ == "__main__":
    test_rag_system()