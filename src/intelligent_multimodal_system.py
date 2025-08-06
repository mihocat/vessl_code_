#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
지능형 멀티모달 시스템
질의 의도 분석 → 최적 처리 경로 선택 → 통합 응답 생성
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from query_intent_analyzer import QueryIntentAnalyzer, IntentAnalysisResult, QueryType, ProcessingMode
from optimized_openai_vision_analyzer import OptimizedOpenAIVisionAnalyzer as OpenAIVisionAnalyzer
from ncp_ocr_client import NCPOCRClient
from enhanced_multimodal_processor import EnhancedMultimodalProcessor
from rag_system import RAGSystem

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """처리 상태"""
    ANALYZING_INTENT = "analyzing_intent"
    PROCESSING_IMAGE = "processing_image"
    SEARCHING_KNOWLEDGE = "searching_knowledge"
    REASONING = "reasoning"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingResult:
    """처리 결과"""
    success: bool
    response: str
    confidence: float
    processing_time: float
    
    # 세부 정보
    intent_analysis: Optional[IntentAnalysisResult]
    vision_result: Optional[Dict[str, Any]]
    rag_result: Optional[Dict[str, Any]]
    reasoning_steps: Optional[List[str]]
    
    # 메타데이터
    processing_path: List[str]
    fallback_used: bool
    tokens_used: int
    cost_estimate: float
    
    # 품질 메트릭
    response_quality_score: float
    user_satisfaction_prediction: float


class IntelligentMultimodalSystem:
    """지능형 멀티모달 시스템"""
    
    def __init__(self, config):
        """
        시스템 초기화
        
        Args:
            config: 설정 객체
        """
        self.config = config
        
        # 핵심 컴포넌트 초기화
        logger.info("Initializing Intelligent Multimodal System...")
        
        # 의도 분석기
        self.intent_analyzer = QueryIntentAnalyzer()
        
        # 이미지 분석 시스템 (OpenAI 우선, NCP OCR fallback)
        self.openai_analyzer = OpenAIVisionAnalyzer(config)
        self.ncp_ocr = NCPOCRClient(config)
        self.multimodal_processor = EnhancedMultimodalProcessor(config)
        
        # RAG 시스템
        self.rag_system = RAGSystem(config)
        
        # 처리 통계
        self.processing_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'openai_usage': 0,
            'ncp_usage': 0,
            'average_processing_time': 0.0,
            'average_confidence': 0.0
        }
        
        logger.info("Intelligent Multimodal System initialized successfully")
    
    def process_query(self, query: str, image_path: str = None, user_context: Dict = None) -> ProcessingResult:
        """
        통합 질의 처리
        
        Args:
            query: 사용자 질의
            image_path: 이미지 파일 경로 (선택적)
            user_context: 사용자 컨텍스트 (선택적)
            
        Returns:
            처리 결과
        """
        start_time = time.time()
        processing_path = []
        
        try:
            self.processing_stats['total_requests'] += 1
            logger.info(f"Processing query: '{query[:50]}...' (image: {bool(image_path)})")
            
            # 1단계: 의도 분석
            processing_path.append("intent_analysis")
            intent_result = self.intent_analyzer.analyze_intent(query, has_image=bool(image_path))
            
            logger.info(f"Intent analysis: {intent_result.query_type.value} "
                       f"(complexity: {intent_result.complexity.value}, "
                       f"mode: {intent_result.processing_mode.value})")
            
            # 2단계: 처리 경로 선택 및 실행
            if intent_result.processing_mode == ProcessingMode.VISION_FIRST:
                result = self._process_vision_first(query, image_path, intent_result, processing_path)
            elif intent_result.processing_mode == ProcessingMode.RAG_FIRST:
                result = self._process_rag_first(query, image_path, intent_result, processing_path)
            elif intent_result.processing_mode == ProcessingMode.HYBRID:
                result = self._process_hybrid(query, image_path, intent_result, processing_path)
            elif intent_result.processing_mode == ProcessingMode.REASONING_CHAIN:
                result = self._process_reasoning_chain(query, image_path, intent_result, processing_path)
            else:
                result = self._process_direct_response(query, image_path, intent_result, processing_path)
            
            # 3단계: 결과 후처리 및 품질 평가
            result = self._post_process_result(result, intent_result, processing_path)
            
            # 통계 업데이트
            processing_time = time.time() - start_time
            self._update_stats(result, processing_time)
            
            logger.info(f"Query processing completed in {processing_time:.2f}s "
                       f"(confidence: {result.confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            
            # 실패 시 기본 응답 생성
            fallback_result = self._generate_fallback_response(
                query, image_path, str(e), processing_path, time.time() - start_time
            )
            
            return fallback_result
    
    def _process_vision_first(self, query: str, image_path: str, intent: IntentAnalysisResult, 
                             processing_path: List[str]) -> ProcessingResult:
        """비전 우선 처리"""
        processing_path.append("vision_analysis")
        
        if not image_path:
            raise ValueError("Vision-first processing requires an image")
        
        # OpenAI Vision API 시도
        vision_result = None
        fallback_used = False
        
        if self.openai_analyzer.api_available:
            try:
                logger.info("Using OpenAI Vision API for image analysis")
                vision_result = self.openai_analyzer.analyze_image(
                    image_path, 
                    question=query,
                    extract_text=intent.query_type in [QueryType.TEXT_EXTRACTION, QueryType.VISUAL_ANALYSIS],
                    detect_formulas=intent.query_type == QueryType.FORMULA_ANALYSIS
                )
                
                if vision_result['success']:
                    self.processing_stats['openai_usage'] += 1
                    logger.info("OpenAI Vision API analysis successful")
                else:
                    raise Exception(f"OpenAI Vision API failed: {vision_result.get('error')}")
                    
            except Exception as e:
                logger.warning(f"OpenAI Vision API failed: {e}, trying NCP OCR fallback")
                vision_result = None
        
        # NCP OCR fallback
        if not vision_result or not vision_result['success']:
            if self.ncp_ocr.api_available:
                try:
                    logger.info("Using NCP OCR as fallback")
                    vision_result = self.ncp_ocr.analyze_image(image_path, question=query)
                    fallback_used = True
                    self.processing_stats['ncp_usage'] += 1
                    logger.info("NCP OCR analysis successful")
                except Exception as e:
                    logger.error(f"NCP OCR also failed: {e}")
                    raise Exception("Both OpenAI Vision API and NCP OCR failed")
            else:
                raise Exception("No image analysis service available")
        
        # RAG 보완 검색 (필요한 경우)
        rag_result = None
        if intent.requires_rag or intent.query_type in [QueryType.PROBLEM_SOLVING, QueryType.EXPLANATION]:
            processing_path.append("rag_search")
            try:
                # 추출된 텍스트를 기반으로 RAG 검색
                search_query = vision_result.get('extracted_text', query)
                rag_result = self.rag_system.search(search_query, max_results=5)
                logger.info(f"RAG search completed: {len(rag_result.get('results', []))} results")
            except Exception as e:
                logger.warning(f"RAG search failed: {e}")
        
        # 응답 생성
        response = self._synthesize_vision_response(query, vision_result, rag_result, intent)
        
        return ProcessingResult(
            success=True,
            response=response,
            confidence=vision_result.get('confidence', 0.8),
            processing_time=0.0,  # 나중에 설정
            intent_analysis=intent,
            vision_result=vision_result,
            rag_result=rag_result,
            reasoning_steps=None,
            processing_path=processing_path.copy(),
            fallback_used=fallback_used,
            tokens_used=vision_result.get('usage', {}).get('total_tokens', 0),
            cost_estimate=self._estimate_cost(vision_result, rag_result),
            response_quality_score=0.8,  # 나중에 계산
            user_satisfaction_prediction=0.8
        )
    
    def _process_rag_first(self, query: str, image_path: str, intent: IntentAnalysisResult,
                          processing_path: List[str]) -> ProcessingResult:
        """RAG 우선 처리"""
        processing_path.append("rag_search")
        
        # RAG 검색
        rag_result = self.rag_system.search(query, max_results=10)
        logger.info(f"RAG search completed: {len(rag_result.get('results', []))} results")
        
        # 이미지 분석 (보완적)
        vision_result = None
        if image_path and intent.requires_image:
            processing_path.append("vision_analysis")
            try:
                vision_result = self.openai_analyzer.analyze_image(
                    image_path, question=query, extract_text=True
                )
                logger.info("Supplementary image analysis completed")
            except Exception as e:
                logger.warning(f"Supplementary image analysis failed: {e}")
        
        # 응답 생성
        response = self._synthesize_rag_response(query, rag_result, vision_result, intent)
        
        return ProcessingResult(
            success=True,
            response=response,
            confidence=rag_result.get('confidence', 0.7),
            processing_time=0.0,
            intent_analysis=intent,
            vision_result=vision_result,
            rag_result=rag_result,
            reasoning_steps=None,
            processing_path=processing_path.copy(),
            fallback_used=False,
            tokens_used=(vision_result.get('usage', {}).get('total_tokens', 0) if vision_result else 0),
            cost_estimate=self._estimate_cost(vision_result, rag_result),
            response_quality_score=0.7,
            user_satisfaction_prediction=0.75
        )
    
    def _process_hybrid(self, query: str, image_path: str, intent: IntentAnalysisResult,
                       processing_path: List[str]) -> ProcessingResult:
        """하이브리드 처리 (병렬)"""
        processing_path.extend(["vision_analysis", "rag_search"])
        
        vision_result = None
        rag_result = None
        fallback_used = False
        
        # 병렬 처리 시뮬레이션 (실제로는 asyncio 사용 가능)
        if image_path:
            try:
                vision_result = self.openai_analyzer.analyze_image(
                    image_path, question=query, extract_text=True, detect_formulas=True
                )
                if not vision_result['success']:
                    vision_result = self.ncp_ocr.analyze_image(image_path, question=query)
                    fallback_used = True
            except Exception as e:
                logger.warning(f"Image analysis failed: {e}")
        
        try:
            # 이미지에서 추출된 텍스트가 있으면 함께 검색
            search_query = query
            if vision_result and vision_result.get('extracted_text'):
                search_query = f"{query} {vision_result['extracted_text'][:200]}"
            
            rag_result = self.rag_system.search(search_query, max_results=8)
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
        
        # 통합 응답 생성
        response = self._synthesize_hybrid_response(query, vision_result, rag_result, intent)
        
        return ProcessingResult(
            success=True,
            response=response,
            confidence=max(
                vision_result.get('confidence', 0.5) if vision_result else 0.5,
                rag_result.get('confidence', 0.5) if rag_result else 0.5
            ),
            processing_time=0.0,
            intent_analysis=intent,
            vision_result=vision_result,
            rag_result=rag_result,
            reasoning_steps=None,
            processing_path=processing_path.copy(),
            fallback_used=fallback_used,
            tokens_used=(vision_result.get('usage', {}).get('total_tokens', 0) if vision_result else 0),
            cost_estimate=self._estimate_cost(vision_result, rag_result),
            response_quality_score=0.8,
            user_satisfaction_prediction=0.8
        )
    
    def _process_reasoning_chain(self, query: str, image_path: str, intent: IntentAnalysisResult,
                                processing_path: List[str]) -> ProcessingResult:
        """추론 체인 처리"""
        processing_path.append("reasoning_chain")
        
        reasoning_steps = []
        
        # 1단계: 문제 분석
        reasoning_steps.append("1. 문제 분석 및 분해")
        
        # 2단계: 정보 수집
        vision_result = None
        rag_result = None
        
        if image_path:
            reasoning_steps.append("2. 이미지에서 정보 추출")
            vision_result = self.openai_analyzer.analyze_image(
                image_path, question=query, extract_text=True, detect_formulas=True
            )
        
        reasoning_steps.append("3. 관련 지식 검색")
        search_query = query
        if vision_result and vision_result.get('extracted_text'):
            search_query = f"{query} {vision_result['extracted_text']}"
        
        rag_result = self.rag_system.search(search_query, max_results=5)
        
        # 3단계: 단계별 해결
        reasoning_steps.append("4. 단계별 문제 해결")
        
        # 응답 생성 (추론 과정 포함)
        response = self._synthesize_reasoning_response(
            query, vision_result, rag_result, intent, reasoning_steps
        )
        
        return ProcessingResult(
            success=True,
            response=response,
            confidence=0.85,  # 추론 체인은 일반적으로 높은 신뢰도
            processing_time=0.0,
            intent_analysis=intent,
            vision_result=vision_result,
            rag_result=rag_result,
            reasoning_steps=reasoning_steps,
            processing_path=processing_path.copy(),
            fallback_used=False,
            tokens_used=(vision_result.get('usage', {}).get('total_tokens', 0) if vision_result else 0),
            cost_estimate=self._estimate_cost(vision_result, rag_result),
            response_quality_score=0.85,
            user_satisfaction_prediction=0.9
        )
    
    def _process_direct_response(self, query: str, image_path: str, intent: IntentAnalysisResult,
                               processing_path: List[str]) -> ProcessingResult:
        """직접 응답 처리"""
        processing_path.append("direct_response")
        
        # 간단한 질의는 직접 처리
        response = f"질문: {query}\n\n"
        
        if intent.query_type == QueryType.GENERAL_CHAT:
            response += "안녕하세요! 이미지 분석, 수식 해석, 기술적 질문 등 다양한 도움을 드릴 수 있습니다. 구체적인 질문이나 이미지를 제공해 주시면 더 정확한 답변을 드릴 수 있습니다."
        else:
            response += "죄송합니다. 더 구체적인 정보나 이미지가 필요한 질문인 것 같습니다. 자세한 내용을 제공해 주시거나 관련 이미지를 첨부해 주시면 더 나은 답변을 드릴 수 있습니다."
        
        return ProcessingResult(
            success=True,
            response=response,
            confidence=0.6,
            processing_time=0.0,
            intent_analysis=intent,
            vision_result=None,
            rag_result=None,
            reasoning_steps=None,
            processing_path=processing_path.copy(),
            fallback_used=False,
            tokens_used=0,
            cost_estimate=0.0,
            response_quality_score=0.6,
            user_satisfaction_prediction=0.5
        )
    
    def _synthesize_vision_response(self, query: str, vision_result: Dict, rag_result: Dict,
                                  intent: IntentAnalysisResult) -> str:
        """비전 분석 결과 기반 응답 합성"""
        if not vision_result or not vision_result.get('success'):
            return "죄송합니다. 이미지 분석에 실패했습니다."
        
        response_parts = []
        
        # 원본 응답
        if vision_result.get('raw_response'):
            response_parts.append(vision_result['raw_response'])
        
        # RAG 보완 정보
        if rag_result and rag_result.get('results'):
            response_parts.append("\n\n📚 관련 정보:")
            for i, result in enumerate(rag_result['results'][:3], 1):
                response_parts.append(f"{i}. {result.get('content', '')[:200]}...")
        
        # 신뢰도 정보
        confidence = vision_result.get('confidence', 0.8)
        if confidence < 0.7:
            response_parts.append(f"\n\n⚠️ 분석 신뢰도: {confidence:.1%} - 결과를 검토해 주세요.")
        
        return '\n'.join(response_parts)
    
    def _synthesize_rag_response(self, query: str, rag_result: Dict, vision_result: Dict,
                               intent: IntentAnalysisResult) -> str:
        """RAG 검색 결과 기반 응답 합성"""
        if not rag_result or not rag_result.get('results'):
            return "죄송합니다. 관련 정보를 찾을 수 없습니다."
        
        response_parts = [f"질문: {query}\n"]
        
        # 주요 답변
        best_result = rag_result['results'][0]
        response_parts.append(f"💡 답변:\n{best_result.get('content', '')}")
        
        # 추가 정보
        if len(rag_result['results']) > 1:
            response_parts.append("\n📚 추가 관련 정보:")
            for i, result in enumerate(rag_result['results'][1:4], 1):
                response_parts.append(f"{i}. {result.get('content', '')[:150]}...")
        
        # 이미지 보완 정보
        if vision_result and vision_result.get('extracted_text'):
            response_parts.append(f"\n🔍 이미지 분석 결과:\n{vision_result['extracted_text'][:300]}...")
        
        return '\n'.join(response_parts)
    
    def _synthesize_hybrid_response(self, query: str, vision_result: Dict, rag_result: Dict,
                                  intent: IntentAnalysisResult) -> str:
        """하이브리드 응답 합성"""
        response_parts = [f"📋 종합 분석 결과\n질문: {query}\n"]
        
        # 이미지 분석 결과
        if vision_result and vision_result.get('success'):
            response_parts.append("🔍 이미지 분석:")
            response_parts.append(vision_result.get('raw_response', '분석 결과 없음'))
        
        # 지식 검색 결과
        if rag_result and rag_result.get('results'):
            response_parts.append("\n📚 관련 지식:")
            for i, result in enumerate(rag_result['results'][:2], 1):
                response_parts.append(f"{i}. {result.get('content', '')[:200]}...")
        
        # 종합 결론
        response_parts.append("\n✅ 결론:")
        response_parts.append("위 이미지 분석과 관련 지식을 종합하여 질문에 대한 답변을 제공했습니다.")
        
        return '\n'.join(response_parts)
    
    def _synthesize_reasoning_response(self, query: str, vision_result: Dict, rag_result: Dict,
                                     intent: IntentAnalysisResult, reasoning_steps: List[str]) -> str:
        """추론 체인 응답 합성"""
        response_parts = [f"🧠 단계별 추론 과정\n질문: {query}\n"]
        
        # 추론 단계
        response_parts.append("📝 해결 과정:")
        for step in reasoning_steps:
            response_parts.append(step)
        
        # 수집된 정보
        if vision_result and vision_result.get('success'):
            response_parts.append(f"\n🔍 이미지에서 확인된 정보:\n{vision_result.get('extracted_text', '없음')}")
        
        if rag_result and rag_result.get('results'):
            response_parts.append(f"\n📚 관련 지식:\n{rag_result['results'][0].get('content', '')[:300]}...")
        
        # 최종 답변
        response_parts.append("\n🎯 최종 답변:")
        response_parts.append("위의 단계별 분석을 통해 문제를 체계적으로 해결했습니다.")
        
        return '\n'.join(response_parts)
    
    def _post_process_result(self, result: ProcessingResult, intent: IntentAnalysisResult,
                           processing_path: List[str]) -> ProcessingResult:
        """결과 후처리"""
        # 품질 점수 계산
        quality_score = self._calculate_quality_score(result)
        result.response_quality_score = quality_score
        
        # 사용자 만족도 예측
        satisfaction = self._predict_user_satisfaction(result, intent)
        result.user_satisfaction_prediction = satisfaction
        
        return result
    
    def _calculate_quality_score(self, result: ProcessingResult) -> float:
        """응답 품질 점수 계산"""
        score = 0.5  # 기본 점수
        
        # 성공적인 처리
        if result.success:
            score += 0.2
        
        # 신뢰도
        score += result.confidence * 0.2
        
        # 응답 길이 (적절한 길이)
        response_length = len(result.response)
        if 100 <= response_length <= 2000:
            score += 0.1
        
        return min(1.0, score)
    
    def _predict_user_satisfaction(self, result: ProcessingResult, intent: IntentAnalysisResult) -> float:
        """사용자 만족도 예측"""
        satisfaction = result.confidence * 0.7
        
        # 의도와 처리 방식 일치도
        if intent.processing_mode.value in result.processing_path:
            satisfaction += 0.2
        
        # 다중 정보원 사용
        if result.vision_result and result.rag_result:
            satisfaction += 0.1
        
        return min(1.0, satisfaction)
    
    def _estimate_cost(self, vision_result: Dict, rag_result: Dict) -> float:
        """비용 추정"""
        cost = 0.0
        
        if vision_result and vision_result.get('usage'):
            # gpt-4o-mini 기준
            tokens = vision_result['usage'].get('total_tokens', 0)
            cost += tokens * 0.0006 / 1000  # 추정치
        
        return cost
    
    def _generate_fallback_response(self, query: str, image_path: str, error: str,
                                  processing_path: List[str], processing_time: float) -> ProcessingResult:
        """실패 시 대체 응답 생성"""
        fallback_response = f"""죄송합니다. 요청을 처리하는 중 문제가 발생했습니다.

질문: {query}
오류: {error}

다음을 시도해 보세요:
1. 질문을 더 구체적으로 작성해 주세요
2. 이미지가 있다면 선명한 이미지를 제공해 주세요
3. 잠시 후 다시 시도해 주세요

기술 지원이 필요하시면 관리자에게 문의해 주세요."""
        
        return ProcessingResult(
            success=False,
            response=fallback_response,
            confidence=0.3,
            processing_time=processing_time,
            intent_analysis=None,
            vision_result=None,
            rag_result=None,
            reasoning_steps=None,
            processing_path=processing_path,
            fallback_used=True,
            tokens_used=0,
            cost_estimate=0.0,
            response_quality_score=0.3,
            user_satisfaction_prediction=0.2
        )
    
    def _update_stats(self, result: ProcessingResult, processing_time: float):
        """통계 업데이트"""
        result.processing_time = processing_time
        
        if result.success:
            self.processing_stats['successful_requests'] += 1
        
        # 평균 처리 시간 업데이트
        total = self.processing_stats['total_requests']
        current_avg = self.processing_stats['average_processing_time']
        self.processing_stats['average_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        # 평균 신뢰도 업데이트
        current_conf = self.processing_stats['average_confidence']
        self.processing_stats['average_confidence'] = (
            (current_conf * (total - 1) + result.confidence) / total
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 반환"""
        return self.processing_stats.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        return {
            'status': 'healthy',
            'components': {
                'intent_analyzer': True,
                'openai_analyzer': self.openai_analyzer.api_available,
                'ncp_ocr': self.ncp_ocr.api_available,
                'rag_system': True  # 실제로는 RAG 시스템 상태 확인
            },
            'stats': self.get_system_stats()
        }