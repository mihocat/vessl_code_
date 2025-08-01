#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Integration Test Script
시스템 통합 테스트 스크립트
"""

import sys
import os
import time
import logging
import json
import traceback
from typing import Dict, List, Any, Optional

# src 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemTester:
    """시스템 테스터"""
    
    def __init__(self):
        """테스터 초기화"""
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("=== System Integration Test Started ===")
        
        # 1. 모듈 임포트 테스트
        self.test_module_imports()
        
        # 2. 기본 기능 테스트
        self.test_basic_features()
        
        # 3. 향상된 기능 테스트
        self.test_enhanced_features()
        
        # 4. 시각화 기능 테스트
        self.test_visualization_features()
        
        # 5. 성능 테스트
        self.test_performance()
        
        # 결과 요약
        self.print_summary()
    
    def test_module_imports(self):
        """모듈 임포트 테스트"""
        logger.info("\n--- Testing Module Imports ---")
        
        modules_to_test = [
            ("Basic RAG", ["config", "llm_client", "rag_system", "services"]),
            ("Enhanced RAG", ["enhanced_rag_system", "enhanced_image_analyzer", "chatgpt_response_generator"]),
            ("Multimodal OCR", ["multimodal_ocr"]),
            ("Visualization", ["visualization_components"]),
            ("UI Modules", ["app", "enhanced_app", "advanced_ui"])
        ]
        
        for category, modules in modules_to_test:
            logger.info(f"\nTesting {category}:")
            for module in modules:
                self.test_import(module)
    
    def test_import(self, module_name: str):
        """단일 모듈 임포트 테스트"""
        self.total_tests += 1
        try:
            exec(f"import {module_name}")
            self.passed_tests += 1
            self.test_results.append({
                'test': f'Import {module_name}',
                'status': 'PASS',
                'message': 'Successfully imported'
            })
            logger.info(f"  ✓ {module_name}")
        except Exception as e:
            self.failed_tests += 1
            self.test_results.append({
                'test': f'Import {module_name}',
                'status': 'FAIL',
                'message': str(e)
            })
            logger.error(f"  ✗ {module_name}: {e}")
    
    def test_basic_features(self):
        """기본 기능 테스트"""
        logger.info("\n--- Testing Basic Features ---")
        
        self.total_tests += 1
        try:
            from config import Config
            config = Config()
            
            # 설정 검증
            assert hasattr(config, 'llm'), "LLM config missing"
            assert hasattr(config, 'rag'), "RAG config missing"
            assert hasattr(config, 'app'), "App config missing"
            
            self.passed_tests += 1
            self.test_results.append({
                'test': 'Config Validation',
                'status': 'PASS',
                'message': 'All required configurations present'
            })
            logger.info("  ✓ Configuration validation")
            
        except Exception as e:
            self.failed_tests += 1
            self.test_results.append({
                'test': 'Config Validation',
                'status': 'FAIL',
                'message': str(e)
            })
            logger.error(f"  ✗ Configuration validation: {e}")
        
        # 벡터 DB 테스트
        self.total_tests += 1
        try:
            from rag_system import VectorStore
            from config import RAGConfig
            from sentence_transformers import SentenceTransformer
            
            # RAG 설정과 임베딩 모델 준비
            rag_config = RAGConfig()
            rag_config.chroma_db_path = "./test_chroma_db"  # 테스트용 경로
            embedding_model = SentenceTransformer(rag_config.embedding_model_name, trust_remote_code=True)
            
            # VectorStore 생성
            vector_db = VectorStore(rag_config, embedding_model)
            
            # 기본 작업 테스트
            test_doc = {
                "question": "테스트 질문",
                "answer": "테스트 답변",
                "category": "test"
            }
            
            self.passed_tests += 1
            self.test_results.append({
                'test': 'Vector Database',
                'status': 'PASS',
                'message': 'Basic operations working'
            })
            logger.info("  ✓ Vector database operations")
            
        except Exception as e:
            self.failed_tests += 1
            self.test_results.append({
                'test': 'Vector Database',
                'status': 'FAIL',
                'message': str(e)
            })
            logger.error(f"  ✗ Vector database operations: {e}")
    
    def test_enhanced_features(self):
        """향상된 기능 테스트"""
        logger.info("\n--- Testing Enhanced Features ---")
        
        # 멀티모달 임베딩 테스트
        self.total_tests += 1
        try:
            from enhanced_rag_system import MultimodalEmbedder
            embedder = MultimodalEmbedder()
            
            # 텍스트 임베딩
            text_emb = embedder.embed_text("테스트 텍스트")
            assert text_emb.shape[0] > 0, "Empty text embedding"
            
            # 수식 임베딩
            formula_emb = embedder.embed_formula("E = mc^2")
            assert formula_emb.shape[0] > 0, "Empty formula embedding"
            
            self.passed_tests += 1
            self.test_results.append({
                'test': 'Multimodal Embedding',
                'status': 'PASS',
                'message': 'Text and formula embedding working'
            })
            logger.info("  ✓ Multimodal embedding")
            
        except Exception as e:
            self.failed_tests += 1
            self.test_results.append({
                'test': 'Multimodal Embedding',
                'status': 'FAIL',
                'message': str(e)
            })
            logger.error(f"  ✗ Multimodal embedding: {e}")
        
        # ChatGPT 스타일 응답 생성 테스트
        self.total_tests += 1
        try:
            from chatgpt_response_generator import ChatGPTResponseGenerator
            generator = ChatGPTResponseGenerator()
            
            test_context = {
                'key_concepts': ['전압', '전류'],
                'formulas': ['V = IR'],
                'visual_elements': {'has_circuit': True}
            }
            
            response = generator.generate_response(
                "옴의 법칙을 설명해주세요",
                test_context,
                'comprehensive'
            )
            
            assert len(response) > 0, "Empty response"
            assert "✅" in response, "Missing response formatting"
            
            self.passed_tests += 1
            self.test_results.append({
                'test': 'ChatGPT Style Response',
                'status': 'PASS',
                'message': 'Response generation working'
            })
            logger.info("  ✓ ChatGPT style response generation")
            
        except Exception as e:
            self.failed_tests += 1
            self.test_results.append({
                'test': 'ChatGPT Style Response',
                'status': 'FAIL',
                'message': str(e)
            })
            logger.error(f"  ✗ ChatGPT style response generation: {e}")
    
    def test_visualization_features(self):
        """시각화 기능 테스트"""
        logger.info("\n--- Testing Visualization Features ---")
        
        self.total_tests += 1
        try:
            from visualization_components import VisualizationManager
            viz_manager = VisualizationManager()
            
            # 수식 렌더링 테스트
            formula_img = viz_manager.create_visualization('formula', {
                'formula': 'P = \\sqrt{3} \\times V_L \\times I_L \\times \\cos\\theta'
            })
            assert formula_img is not None, "Formula rendering failed"
            
            # 회로도 생성 테스트
            circuit_img = viz_manager.create_visualization('circuit', {
                'components': [
                    {'type': 'resistor', 'label': 'R1', 'position': (4, 4)},
                    {'type': 'capacitor', 'label': 'C1', 'position': (6, 4)}
                ],
                'connections': [
                    {'from': (4.5, 4), 'to': (5.5, 4)}
                ]
            })
            assert circuit_img is not None, "Circuit diagram generation failed"
            
            self.passed_tests += 1
            self.test_results.append({
                'test': 'Visualization Components',
                'status': 'PASS',
                'message': 'Formula and circuit visualization working'
            })
            logger.info("  ✓ Visualization components")
            
        except Exception as e:
            self.failed_tests += 1
            self.test_results.append({
                'test': 'Visualization Components',
                'status': 'FAIL',
                'message': str(e)
            })
            logger.error(f"  ✗ Visualization components: {e}")
    
    def test_performance(self):
        """성능 테스트"""
        logger.info("\n--- Testing Performance ---")
        
        self.total_tests += 1
        try:
            from enhanced_rag_system import MultimodalEmbedder
            embedder = MultimodalEmbedder()
            
            # 임베딩 속도 테스트
            start_time = time.time()
            for i in range(10):
                embedder.embed_text(f"테스트 텍스트 {i}")
            embedding_time = (time.time() - start_time) / 10
            
            # 성능 기준: 평균 100ms 이하
            if embedding_time < 0.1:
                status = 'PASS'
                message = f'Average embedding time: {embedding_time*1000:.2f}ms'
            else:
                status = 'WARN'
                message = f'Slow embedding: {embedding_time*1000:.2f}ms (target: <100ms)'
            
            if status == 'PASS':
                self.passed_tests += 1
            else:
                self.failed_tests += 1
            
            self.test_results.append({
                'test': 'Embedding Performance',
                'status': status,
                'message': message
            })
            logger.info(f"  {'✓' if status == 'PASS' else '⚠'} Embedding performance: {message}")
            
        except Exception as e:
            self.failed_tests += 1
            self.test_results.append({
                'test': 'Embedding Performance',
                'status': 'FAIL',
                'message': str(e)
            })
            logger.error(f"  ✗ Embedding performance: {e}")
    
    def print_summary(self):
        """테스트 결과 요약"""
        logger.info("\n" + "="*50)
        logger.info("TEST SUMMARY")
        logger.info("="*50)
        logger.info(f"Total Tests: {self.total_tests}")
        logger.info(f"Passed: {self.passed_tests} ({self.passed_tests/self.total_tests*100:.1f}%)")
        logger.info(f"Failed: {self.failed_tests} ({self.failed_tests/self.total_tests*100:.1f}%)")
        
        if self.failed_tests > 0:
            logger.info("\nFailed Tests:")
            for result in self.test_results:
                if result['status'] == 'FAIL':
                    logger.info(f"  - {result['test']}: {result['message']}")
        
        # 결과 저장
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'summary': {
                    'total': self.total_tests,
                    'passed': self.passed_tests,
                    'failed': self.failed_tests,
                    'success_rate': f"{self.passed_tests/self.total_tests*100:.1f}%"
                },
                'results': self.test_results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info("\nTest results saved to test_results.json")
        
        # 반환 코드
        return 0 if self.failed_tests == 0 else 1


def test_example_questions():
    """예제 질문 테스트"""
    logger.info("\n--- Testing Example Questions ---")
    
    # EXAMPLE.md에서 질문 로드
    example_questions = [
        "3상 전력 시스템에서 선간전압이 380V이고 부하전류가 10A일 때 전력을 구하시오.",
        "RLC 직렬회로에서 공진주파수를 구하는 공식은?",
        "변압기의 철손과 동손의 차이점을 설명하세요.",
        "인버터의 PWM 제어 방식에 대해 설명하세요.",
        "유도전동기의 슬립이 0.05일 때 회전속도는?"
    ]
    
    try:
        from enhanced_rag_system import EnhancedVectorDatabase
        vector_db = EnhancedVectorDatabase()
        
        logger.info("\nTesting search for example questions:")
        for i, question in enumerate(example_questions[:3]):  # 처음 3개만 테스트
            results = vector_db.search_multimodal(question, k=3)
            logger.info(f"\n{i+1}. Q: {question[:50]}...")
            logger.info(f"   Found {len(results)} results")
            if results:
                logger.info(f"   Top score: {results[0].get('hybrid_score', 0):.3f}")
    
    except Exception as e:
        logger.error(f"Example questions test failed: {e}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="System Integration Test")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Test example questions"
    )
    
    args = parser.parse_args()
    
    # 테스터 실행
    tester = SystemTester()
    exit_code = tester.run_all_tests()
    
    # 예제 질문 테스트
    if args.examples:
        test_example_questions()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()