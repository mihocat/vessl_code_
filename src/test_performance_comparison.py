#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Comparison Test Script
성능 비교 테스트 스크립트
"""

import sys
import time
import logging
import json
from typing import Dict, Any, List
from PIL import Image

# 설정
from config import Config
from llm_client import LLMClient

# 시스템 임포트
from enhanced_rag_system import EnhancedVectorDatabase, EnhancedRAGSystem
from advanced_rag_system import AdvancedRAGSystem
from enhanced_app import EnhancedChatService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 테스트 질문들 (EXAMPLE_IMG.md 기반)
TEST_QUESTIONS = [
    {
        "id": "Q1",
        "question": "55쪽 4번 문제와 56쪽 7번 문제를 풀면서 두 점 사이의 거리 단위벡터를 구하는 방식에 혼란이 생겼습니다. 4번에서는 거리벡터를 P - Q로 구하였고, 부호에 따라 답이 달라졌습니다. 반면, 7번 문제에서는 전기장 세기를 구하는 위치에서 전하가 있는 위치를 빼는 방식으로 벡터를 구했습니다. 이처럼 문제마다 벡터의 방향이 달라지는 이유가 무엇인지, 거리벡터를 어떤 기준으로 구해야 하는지 명확히 알고 싶습니다.",
        "expected_keywords": ["종점", "시점", "영향을 받는", "작용하는", "방향"],
        "category": "vector_direction"
    },
    {
        "id": "Q2",
        "question": "미분할 때 d/da를 적용하면 왜 s는 1이고 a는 0이 되는 건가요? 이유가 잘 감이 안 잡혀서요. 쉽게 설명해주시면 감사하겠습니다!",
        "expected_keywords": ["변수", "상수", "미분", "s에 대해"],
        "category": "calculus"
    },
    {
        "id": "Q3",
        "question": "Pr'을 계산할 때 Pr에서 Qc를 차감하는 이유가 명확히 이해되지 않습니다. Pr은 무효전력(kVar)이고 Qc는 피상전력(kVA)로 알고 있는데, 서로 단위가 다른데도 불구하고 연산이 가능한 이유가 궁금합니다. 혹시 전력용 콘덴서가 무효전력 성분에만 영향을 주기 때문에 이런 방식의 계산이 이루어지는 건가요?",
        "expected_keywords": ["무효전력", "콘덴서", "역률", "피상전력", "전력용 콘덴서"],
        "category": "power_factor"
    }
]


class PerformanceComparator:
    """성능 비교 도구"""
    
    def __init__(self, config: Config):
        """초기화"""
        self.config = config
        self.results = {
            "enhanced": [],
            "advanced_fast": [],
            "advanced_balanced": [],
            "advanced_reasoning": []
        }
        
        # LLM 클라이언트
        self.llm_client = LLMClient(config.llm)
        
        # 시스템 초기화
        self._initialize_systems()
    
    def _initialize_systems(self):
        """시스템 초기화"""
        logger.info("Initializing systems for comparison...")
        
        # 벡터 DB
        self.vector_db = EnhancedVectorDatabase(
            persist_directory=self.config.rag.persist_directory
        )
        
        # Enhanced RAG
        self.enhanced_rag = EnhancedRAGSystem(
            vector_db=self.vector_db,
            llm_client=self.llm_client
        )
        
        # Advanced RAG
        self.advanced_rag = AdvancedRAGSystem(
            vector_db=self.vector_db,
            llm_client=self.llm_client
        )
        
        logger.info("All systems initialized")
    
    def run_comparison(self):
        """비교 테스트 실행"""
        logger.info("Starting performance comparison test...")
        
        for test in TEST_QUESTIONS:
            logger.info(f"\nTesting question {test['id']}: {test['question'][:50]}...")
            
            # 1. Enhanced RAG 테스트
            result = self._test_enhanced_rag(test)
            self.results["enhanced"].append(result)
            
            # 2. Advanced RAG - Fast mode
            result = self._test_advanced_rag(test, mode="fast")
            self.results["advanced_fast"].append(result)
            
            # 3. Advanced RAG - Balanced mode
            result = self._test_advanced_rag(test, mode="balanced")
            self.results["advanced_balanced"].append(result)
            
            # 4. Advanced RAG - Reasoning mode
            result = self._test_advanced_rag(test, mode="reasoning")
            self.results["advanced_reasoning"].append(result)
        
        # 결과 분석
        self._analyze_results()
    
    def _test_enhanced_rag(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced RAG 테스트"""
        start_time = time.time()
        
        try:
            result = self.enhanced_rag.process_query(
                query=test["question"],
                response_style="comprehensive"
            )
            
            elapsed_time = time.time() - start_time
            
            # 키워드 매칭 점수
            keyword_score = self._calculate_keyword_score(
                result.get("response", ""),
                test["expected_keywords"]
            )
            
            return {
                "test_id": test["id"],
                "success": result.get("success", False),
                "time": elapsed_time,
                "keyword_score": keyword_score,
                "response_length": len(result.get("response", "")),
                "search_score": self._get_avg_search_score(result.get("search_results", []))
            }
            
        except Exception as e:
            logger.error(f"Enhanced RAG test failed: {e}")
            return {
                "test_id": test["id"],
                "success": False,
                "error": str(e),
                "time": time.time() - start_time
            }
    
    def _test_advanced_rag(self, test: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """Advanced RAG 테스트"""
        start_time = time.time()
        
        try:
            result = self.advanced_rag.process_query_advanced(
                query=test["question"],
                mode=mode,
                response_style="comprehensive"
            )
            
            elapsed_time = time.time() - start_time
            
            # 키워드 매칭 점수
            keyword_score = self._calculate_keyword_score(
                result.get("response", ""),
                test["expected_keywords"]
            )
            
            return {
                "test_id": test["id"],
                "mode": mode,
                "success": result.get("success", False),
                "time": elapsed_time,
                "keyword_score": keyword_score,
                "response_length": len(result.get("response", "")),
                "search_score": self._get_avg_search_score(result.get("search_results", [])),
                "has_reasoning": "reasoning_trace" in result,
                "confidence": result.get("confidence", 0)
            }
            
        except Exception as e:
            logger.error(f"Advanced RAG ({mode}) test failed: {e}")
            return {
                "test_id": test["id"],
                "mode": mode,
                "success": False,
                "error": str(e),
                "time": time.time() - start_time
            }
    
    def _calculate_keyword_score(self, response: str, keywords: List[str]) -> float:
        """키워드 매칭 점수 계산"""
        if not keywords or not response:
            return 0.0
        
        response_lower = response.lower()
        matched = sum(1 for keyword in keywords if keyword.lower() in response_lower)
        
        return matched / len(keywords)
    
    def _get_avg_search_score(self, search_results: List[Dict[str, Any]]) -> float:
        """평균 검색 점수"""
        if not search_results:
            return 0.0
        
        scores = []
        for result in search_results:
            if "hybrid_score" in result:
                scores.append(result["hybrid_score"])
            elif "score" in result:
                scores.append(result["score"])
            elif "final_rerank_score" in result:
                scores.append(result["final_rerank_score"])
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _analyze_results(self):
        """결과 분석"""
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE COMPARISON RESULTS")
        logger.info("="*60)
        
        # 각 시스템별 평균 계산
        for system_name, results in self.results.items():
            if not results:
                continue
            
            avg_time = sum(r.get("time", 0) for r in results) / len(results)
            avg_keyword_score = sum(r.get("keyword_score", 0) for r in results) / len(results)
            avg_search_score = sum(r.get("search_score", 0) for r in results) / len(results)
            success_rate = sum(1 for r in results if r.get("success", False)) / len(results)
            
            logger.info(f"\n{system_name.upper()}:")
            logger.info(f"  Average Response Time: {avg_time:.2f}s")
            logger.info(f"  Average Keyword Score: {avg_keyword_score:.2%}")
            logger.info(f"  Average Search Score: {avg_search_score:.3f}")
            logger.info(f"  Success Rate: {success_rate:.2%}")
            
            # Advanced 시스템의 추가 정보
            if "advanced" in system_name:
                reasoning_count = sum(1 for r in results if r.get("has_reasoning", False))
                avg_confidence = sum(r.get("confidence", 0) for r in results) / len(results)
                logger.info(f"  With Reasoning: {reasoning_count}/{len(results)}")
                logger.info(f"  Average Confidence: {avg_confidence:.2f}")
        
        # 상세 결과 저장
        self._save_detailed_results()
    
    def _save_detailed_results(self):
        """상세 결과 저장"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"/Users/macbookair/ai/vessl_test/logs/{timestamp}_performance_comparison.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "test_questions": TEST_QUESTIONS,
                "results": self.results,
                "summary": self._generate_summary()
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\nDetailed results saved to: {filename}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """요약 생성"""
        summary = {}
        
        for system_name, results in self.results.items():
            if not results:
                continue
            
            summary[system_name] = {
                "avg_response_time": sum(r.get("time", 0) for r in results) / len(results),
                "avg_keyword_score": sum(r.get("keyword_score", 0) for r in results) / len(results),
                "avg_search_score": sum(r.get("search_score", 0) for r in results) / len(results),
                "success_rate": sum(1 for r in results if r.get("success", False)) / len(results),
                "total_tests": len(results)
            }
        
        return summary


def main():
    """메인 함수"""
    logger.info("Starting performance comparison test...")
    
    # 설정 로드
    config = Config()
    
    # 비교 도구 생성
    comparator = PerformanceComparator(config)
    
    # LLM 서버 대기
    logger.info("Waiting for LLM server...")
    if not comparator.llm_client.wait_for_server():
        logger.error("Failed to connect to LLM server")
        sys.exit(1)
    
    # 비교 테스트 실행
    comparator.run_comparison()
    
    logger.info("\nPerformance comparison test completed!")


if __name__ == "__main__":
    main()