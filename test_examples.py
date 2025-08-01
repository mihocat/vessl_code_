#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
예제 질문 테스트 스크립트
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
from config import Config
from llm_client import LLMClient
from enhanced_rag_system import EnhancedVectorDatabase, EnhancedRAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 예제 질문들
EXAMPLE_QUESTIONS = [
    "3상 전력 시스템에서 선간전압이 380V이고 부하전류가 10A일 때 전력을 구하시오.",
    "RLC 직렬회로에서 공진주파수를 구하는 공식은?",
    "변압기의 철손과 동손의 차이점을 설명하세요.",
    "인버터의 PWM 제어 방식에 대해 설명하세요.",
    "유도전동기의 슬립이 0.05일 때 회전속도는?",
    "두 점 간의 거리 벡터를 구할 때 어떤 순서로 계산해야 하는지 설명해주세요.",
    "복소미분에서 d/da를 적용하면 왜 s는 1이고 a는 0이 되는지 설명해주세요.",
    "Pr'을 구할 때 Pr에서 Qc를 빼는 이유와 단위가 다른데도 연산이 가능한 이유를 설명해주세요."
]

def test_rag_system():
    """RAG 시스템 테스트"""
    try:
        # 설정 및 클라이언트 초기화
        config = Config()
        llm_client = LLMClient(config.llm)
        
        # 향상된 벡터 DB
        vector_db = EnhancedVectorDatabase(
            persist_directory=config.rag.persist_directory
        )
        
        # 향상된 RAG 시스템
        enhanced_rag = EnhancedRAGSystem(
            vector_db=vector_db,
            llm_client=llm_client
        )
        
        logger.info(f"Testing {len(EXAMPLE_QUESTIONS)} example questions...")
        
        # 각 질문 테스트
        for i, question in enumerate(EXAMPLE_QUESTIONS):
            logger.info(f"\n{'='*50}")
            logger.info(f"Question {i+1}: {question}")
            logger.info("="*50)
            
            try:
                # 쿼리 처리
                result = enhanced_rag.process_query(
                    query=question,
                    response_style='comprehensive'
                )
                
                if result['success']:
                    # 응답 출력
                    logger.info("\nGenerated Response:")
                    print(result['response'][:500] + "..." if len(result['response']) > 500 else result['response'])
                    
                    # 검색 결과 정보
                    if result.get('search_results'):
                        logger.info(f"\nSearch Results: {len(result['search_results'])} documents found")
                        for j, sr in enumerate(result['search_results'][:3]):
                            score = sr.get('hybrid_score', 0)
                            logger.info(f"  {j+1}. Score: {score:.3f}")
                else:
                    logger.error("Failed to generate response")
                    
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
            
            # 간단한 딜레이
            import time
            time.sleep(1)
        
        logger.info("\n\nTest completed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    test_rag_system()