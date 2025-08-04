#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent RAG 통합 테스트 (간단 버전)
"""

import sys
import os

# src 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from intelligent_rag_adapter import IntelligentRAGAdapter
from unittest.mock import Mock


def test_basic_functionality():
    """기본 기능 테스트"""
    print("=== 기본 기능 테스트 ===\n")
    
    # 설정
    config = Config()
    config.rag.use_intelligent_rag = True
    config.rag.intelligent_rag_mode = "adaptive"
    
    # 어댑터 생성
    adapter = IntelligentRAGAdapter(config, Mock())
    
    # 테스트 쿼리
    test_queries = [
        ("안녕하세요", "간단한 인사"),
        ("머신러닝이 뭔가요?", "중간 복잡도"),
        ("딥러닝과 머신러닝의 차이점을 자세히 설명해주세요", "높은 복잡도"),
    ]
    
    for query, desc in test_queries:
        complexity = adapter.evaluate_complexity(query)
        should_use = adapter.should_use_intelligent(query)
        
        print(f"쿼리: {query}")
        print(f"설명: {desc}")
        print(f"복잡도 점수: {complexity.score:.3f}")
        print(f"Intelligent 사용: {should_use}")
        print(f"요인: {complexity.factors}")
        print("-" * 50)


def test_configuration():
    """설정 테스트"""
    print("\n=== 설정 테스트 ===\n")
    
    config = Config()
    
    print(f"Intelligent RAG 활성화: {config.rag.use_intelligent_rag}")
    print(f"Intelligent RAG 모드: {config.rag.intelligent_rag_mode}")
    print(f"기능 설정: {config.rag.intelligent_features}")
    
    # 환경 변수 테스트
    os.environ['USE_INTELLIGENT_RAG'] = 'true'
    os.environ['INTELLIGENT_RAG_MODE'] = 'always'
    
    config2 = Config()
    print(f"\n환경 변수 적용 후:")
    print(f"Intelligent RAG 활성화: {config2.rag.use_intelligent_rag}")
    print(f"Intelligent RAG 모드: {config2.rag.intelligent_rag_mode}")


def test_app_integration():
    """앱 통합 테스트"""
    print("\n=== 앱 통합 테스트 ===\n")
    
    try:
        from app import ChatService
        
        config = Config()
        config.rag.use_intelligent_rag = True
        
        # Mock LLM 클라이언트
        mock_llm = Mock()
        mock_llm.is_healthy.return_value = True
        
        # ChatService 생성
        service = ChatService(config, mock_llm)
        
        print(f"ChatService 생성 성공")
        print(f"Intelligent 어댑터 활성화: {service.intelligent_adapter.enabled}")
        print(f"Intelligent 어댑터 모드: {service.intelligent_adapter.mode}")
        
        # 간단한 쿼리 테스트
        test_query = "테스트 질문입니다"
        should_use = service.intelligent_adapter.should_use_intelligent(test_query)
        print(f"\n'{test_query}' - Intelligent 사용: {should_use}")
        
    except Exception as e:
        print(f"앱 통합 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_basic_functionality()
    test_configuration()
    test_app_integration()
    
    print("\n✅ 테스트 완료!")