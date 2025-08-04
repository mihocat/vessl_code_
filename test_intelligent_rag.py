#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent RAG 통합 테스트
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# src 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from intelligent_rag_adapter import IntelligentRAGAdapter, QueryComplexity


class TestIntelligentRAGAdapter(unittest.TestCase):
    """Intelligent RAG 어댑터 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.config = Config()
        self.config.rag.use_intelligent_rag = True
        self.config.rag.intelligent_rag_mode = "adaptive"
        self.llm_client = Mock()
        
        self.adapter = IntelligentRAGAdapter(self.config, self.llm_client)
    
    def test_complexity_evaluation_simple(self):
        """단순 쿼리 복잡도 평가"""
        query = "오늘 날씨는?"
        complexity = self.adapter.evaluate_complexity(query)
        
        self.assertLess(complexity.score, 0.5)
        self.assertFalse(complexity.should_use_intelligent)
    
    def test_complexity_evaluation_complex(self):
        """복잡한 쿼리 복잡도 평가"""
        query = "양자역학과 상대성이론의 차이점을 비교하고, 각각의 주요 원리와 응용 분야를 설명해주세요."
        complexity = self.adapter.evaluate_complexity(query)
        
        self.assertGreater(complexity.score, 0.5)
        self.assertTrue(complexity.should_use_intelligent)
    
    def test_complexity_with_image(self):
        """이미지 포함 쿼리 복잡도 평가"""
        query = "이 이미지를 분석해주세요"
        context = {'image': Mock()}
        complexity = self.adapter.evaluate_complexity(query, context)
        
        self.assertGreater(complexity.factors['multimodal'], 0.5)
    
    def test_should_use_intelligent_modes(self):
        """모드별 사용 여부 테스트"""
        query = "테스트 쿼리"
        
        # Always 모드
        self.adapter.mode = "always"
        self.assertTrue(self.adapter.should_use_intelligent(query))
        
        # Never 모드
        self.adapter.mode = "never"
        self.assertFalse(self.adapter.should_use_intelligent(query))
        
        # Adaptive 모드
        self.adapter.mode = "adaptive"
        self.adapter.enabled = False
        self.assertFalse(self.adapter.should_use_intelligent(query))
    
    @patch('intelligent_rag_adapter.IntelligentRAGOrchestrator')
    async def test_async_initialization(self, mock_orchestrator):
        """비동기 초기화 테스트"""
        self.adapter.enabled = True
        await self.adapter.initialize()
        
        self.assertTrue(self.adapter.initialized)
        mock_orchestrator.assert_called_once()
    
    def test_sync_processing(self):
        """동기 처리 테스트"""
        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.process_async = asyncio.coroutine(
            lambda *args, **kwargs: {'response': '테스트 응답', 'used_intelligent': True}
        )
        self.adapter.orchestrator = mock_orchestrator
        self.adapter.initialized = True
        
        result = self.adapter.process_sync("테스트 쿼리")
        
        self.assertEqual(result['response'], '테스트 응답')
        self.assertTrue(result['used_intelligent'])


class TestIntegration(unittest.TestCase):
    """통합 테스트"""
    
    @patch('app.Florence2ImageAnalyzer')
    @patch('app.LLMClient')
    def test_app_integration(self, mock_llm, mock_analyzer):
        """앱 통합 테스트"""
        from app import ChatService
        
        config = Config()
        config.rag.use_intelligent_rag = True
        
        # Mock 설정
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        
        # ChatService 생성
        service = ChatService(config, mock_llm_instance)
        
        # Intelligent RAG 어댑터 확인
        self.assertIsNotNone(service.intelligent_adapter)
        self.assertTrue(service.intelligent_adapter.enabled)


def run_performance_test():
    """성능 테스트"""
    import time
    
    config = Config()
    config.rag.use_intelligent_rag = True
    adapter = IntelligentRAGAdapter(config, Mock())
    
    queries = [
        "안녕하세요",
        "오늘 날씨는 어때요?",
        "파이썬에서 리스트와 튜플의 차이점을 설명해주세요",
        "머신러닝과 딥러닝의 차이점과 각각의 장단점을 비교 분석해주세요",
        "양자 컴퓨터의 작동 원리와 현재 기술의 한계점에 대해 설명하고, 향후 발전 가능성을 논의해주세요"
    ]
    
    print("\n=== 복잡도 평가 성능 테스트 ===")
    for query in queries:
        start = time.time()
        complexity = adapter.evaluate_complexity(query)
        elapsed = (time.time() - start) * 1000
        
        print(f"\n쿼리: {query[:50]}...")
        print(f"복잡도 점수: {complexity.score:.3f}")
        print(f"Intelligent 사용: {complexity.should_use_intelligent}")
        print(f"처리 시간: {elapsed:.2f}ms")
        print(f"요인: {complexity.factors}")


if __name__ == '__main__':
    # 단위 테스트 실행
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 성능 테스트 실행
    run_performance_test()