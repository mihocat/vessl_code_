#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
차세대 멀티모달 처리 파이프라인
Next-Generation Multimodal Processing Pipeline

질의 우선 분석 → 이미지 처리 → RAG + LLM 통합 시스템
Query-First Analysis → Image Processing → RAG + LLM Integration
"""

import os
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# 시스템 컴포넌트 임포트
from query_intent_analyzer import QueryIntentAnalyzer, QueryType, ComplexityLevel, IntentAnalysisResult
from openai_vision_client import OpenAIVisionClient, VisionAnalysisResult, AnalysisType
from ncp_ocr_client import NCPOCRClient, OCRResult
from enhanced_rag_system import EnhancedRAGSystem, SearchStrategy, DomainType
from enhanced_llm_system import EnhancedLLMSystem, ModelType, ModelDomain, ModelResponse, ModelConfig

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """처리 모드"""
    VISION_ONLY = "vision_only"          # ChatGPT Vision만 사용
    OCR_ONLY = "ocr_only"                # NCP OCR만 사용
    VISION_FIRST = "vision_first"        # ChatGPT Vision 우선, 실패시 OCR
    OCR_FIRST = "ocr_first"              # OCR 우선, 실패시 Vision
    PARALLEL = "parallel"                # 병렬 처리 후 결과 비교
    ADAPTIVE = "adaptive"                # 상황에 따라 자동 선택


class ConfidenceLevel(Enum):
    """신뢰도 레벨"""
    VERY_HIGH = "very_high"    # 0.9+
    HIGH = "high"              # 0.7-0.9
    MEDIUM = "medium"          # 0.5-0.7
    LOW = "low"                # 0.3-0.5
    VERY_LOW = "very_low"      # 0.0-0.3


@dataclass
class ProcessingResult:
    """멀티모달 처리 결과"""
    success: bool
    query_analysis: Optional[IntentAnalysisResult]
    image_analysis: Optional[VisionAnalysisResult]
    ocr_fallback: Optional[OCRResult]
    rag_results: List[Dict[str, Any]]
    llm_response: Optional[ModelResponse]
    final_answer: str
    confidence_score: float
    processing_time: float
    processing_mode: ProcessingMode
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # 일반 설정
    processing_mode: ProcessingMode = ProcessingMode.ADAPTIVE
    use_rag: bool = True
    use_llm: bool = True
    enable_caching: bool = True
    max_processing_time: float = 60.0
    
    # 신뢰도 임계값
    vision_confidence_threshold: float = 0.7
    ocr_confidence_threshold: float = 0.6
    rag_confidence_threshold: float = 0.5
    final_confidence_threshold: float = 0.6
    
    # 병렬 처리 설정
    enable_parallel_processing: bool = False
    parallel_timeout: float = 30.0
    
    # 품질 향상 설정
    enable_result_validation: bool = True
    enable_multi_pass_refinement: bool = True
    max_refinement_iterations: int = 2


class MultimodalPipeline:
    """차세대 멀티모달 처리 파이프라인"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        파이프라인 초기화
        
        Args:
            config: 파이프라인 설정
        """
        self.config = config or PipelineConfig()
        
        # 컴포넌트 초기화
        self.query_analyzer = None
        self.vision_client = None
        self.ocr_client = None
        self.rag_system = None
        self.llm_system = None
        
        # 처리 통계
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'vision_usage': 0,
            'ocr_usage': 0,
            'rag_usage': 0,
            'llm_usage': 0,
            'average_processing_time': 0.0,
            'average_confidence': 0.0,
            'mode_usage': {mode.value: 0 for mode in ProcessingMode}
        }
        
        # 초기화
        self._initialize_components()
        
        logger.info("Multimodal Pipeline initialized successfully")
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        try:
            # 질의 분석기
            self.query_analyzer = QueryIntentAnalyzer()
            logger.info("Query Intent Analyzer initialized")
            
            # ChatGPT Vision 클라이언트
            try:
                self.vision_client = OpenAIVisionClient()
                logger.info("OpenAI Vision Client initialized")
            except Exception as e:
                logger.warning(f"OpenAI Vision Client initialization failed: {e}")
            
            # NCP OCR 클라이언트
            try:
                self.ocr_client = NCPOCRClient()
                logger.info("NCP OCR Client initialized")
            except Exception as e:
                logger.warning(f"NCP OCR Client initialization failed: {e}")
            
            # 향상된 RAG 시스템
            try:
                self.rag_system = EnhancedRAGSystem()
                logger.info("Enhanced RAG System initialized")
            except Exception as e:
                logger.warning(f"Enhanced RAG System initialization failed: {e}")
            
            # 향상된 LLM 시스템
            try:
                kollama_config = ModelConfig(
                    name="LLM-Fine-Tuned",
                    model_type=ModelType.FINE_TUNED,
                    base_url="http://localhost:8000",
                    model_path="llm-fine-tuned",
                    domain_specialization=[ModelDomain.ELECTRICAL, ModelDomain.GENERAL_SCIENCE]
                )
                self.llm_system = EnhancedLLMSystem(kollama_config)
                logger.info("Enhanced LLM System initialized")
            except Exception as e:
                logger.warning(f"Enhanced LLM System initialization failed: {e}")
                
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    async def process_multimodal_query(self, 
                                     query: str, 
                                     image_data: Optional[Union[str, bytes]] = None,
                                     custom_config: Optional[Dict] = None) -> ProcessingResult:
        """
        멀티모달 쿼리 처리 (메인 엔트리 포인트)
        
        Args:
            query: 사용자 질문
            image_data: 이미지 데이터 (선택적)
            custom_config: 사용자 정의 설정
            
        Returns:
            ProcessingResult: 처리 결과
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            logger.info(f"Starting multimodal query processing: '{query[:50]}...'")
            
            # 1단계: 질의 의도 분석 (핵심)
            query_analysis = await self._analyze_query_intent(query)
            
            # 2단계: 처리 모드 결정
            processing_mode = self._determine_processing_mode(query_analysis, image_data, custom_config)
            self.stats['mode_usage'][processing_mode.value] += 1
            
            # 3단계: 이미지 처리 (있는 경우)
            image_analysis, ocr_fallback = await self._process_image(
                image_data, query_analysis, processing_mode
            )
            
            # 4단계: RAG 검색
            rag_results = await self._perform_rag_search(query, query_analysis, image_analysis)
            
            # 5단계: LLM 응답 생성
            llm_response = await self._generate_llm_response(
                query, query_analysis, image_analysis, rag_results
            )
            
            # 6단계: 최종 답변 구성
            final_answer, confidence_score = self._compose_final_answer(
                query_analysis, image_analysis, ocr_fallback, rag_results, llm_response
            )
            
            processing_time = time.time() - start_time
            
            # 결과 구성
            result = ProcessingResult(
                success=True,
                query_analysis=query_analysis,
                image_analysis=image_analysis,
                ocr_fallback=ocr_fallback,
                rag_results=rag_results,
                llm_response=llm_response,
                final_answer=final_answer,
                confidence_score=confidence_score,
                processing_time=processing_time,
                processing_mode=processing_mode,
                metadata=self._create_metadata(processing_time, custom_config)
            )
            
            # 통계 업데이트
            self._update_stats(result)
            
            logger.info(f"Multimodal processing completed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Multimodal processing failed: {e}")
            
            error_result = ProcessingResult(
                success=False,
                query_analysis=None,
                image_analysis=None,
                ocr_fallback=None,
                rag_results=[],
                llm_response=None,
                final_answer=f"처리 중 오류가 발생했습니다: {str(e)}",
                confidence_score=0.0,
                processing_time=processing_time,
                processing_mode=ProcessingMode.ADAPTIVE,
                metadata={'error': True, 'processing_time': processing_time},
                error_message=str(e)
            )
            
            self.stats['failed_requests'] += 1
            return error_result
    
    async def _analyze_query_intent(self, query: str) -> IntentAnalysisResult:
        """질의 의도 분석"""
        try:
            # 동기 함수를 비동기로 래핑
            loop = asyncio.get_event_loop()
            analysis = await loop.run_in_executor(
                None, self.query_analyzer.analyze_query, query
            )
            
            logger.info(f"Query analysis: type={analysis.query_type.value}, "
                       f"complexity={analysis.complexity.value}, confidence={analysis.confidence:.2f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query intent analysis failed: {e}")
            # 기본 분석 결과 반환
            return IntentAnalysisResult(
                query_type=QueryType.GENERAL_QUESTION,
                complexity=ComplexityLevel.MEDIUM,
                confidence=0.5,
                domain="general",
                suggested_processing="general",
                metadata={'error': str(e)}
            )
    
    def _determine_processing_mode(self, 
                                 query_analysis: IntentAnalysisResult, 
                                 image_data: Optional[Union[str, bytes]],
                                 custom_config: Optional[Dict]) -> ProcessingMode:
        """처리 모드 결정"""
        # 사용자 정의 설정 우선
        if custom_config and 'processing_mode' in custom_config:
            return ProcessingMode(custom_config['processing_mode'])
        
        # 이미지가 없으면 텍스트 전용
        if not image_data:
            return ProcessingMode.VISION_ONLY  # RAG + LLM만 사용
        
        # 질의 유형에 따른 자동 선택
        if query_analysis.query_type in [QueryType.VISUAL_ANALYSIS, QueryType.DIAGRAM_INTERPRETATION]:
            return ProcessingMode.VISION_FIRST
        elif query_analysis.query_type == QueryType.TEXT_EXTRACTION:
            return ProcessingMode.OCR_FIRST
        elif query_analysis.query_type == QueryType.FORMULA_ANALYSIS:
            return ProcessingMode.VISION_FIRST  # ChatGPT가 수식에 더 강함
        elif query_analysis.complexity == ComplexityLevel.VERY_HIGH:
            return ProcessingMode.PARALLEL  # 복잡한 경우 병렬 처리
        else:
            return ProcessingMode.ADAPTIVE
    
    async def _process_image(self, 
                           image_data: Optional[Union[str, bytes]], 
                           query_analysis: IntentAnalysisResult,
                           processing_mode: ProcessingMode) -> Tuple[Optional[VisionAnalysisResult], Optional[OCRResult]]:
        """이미지 처리"""
        if not image_data:
            return None, None
        
        image_analysis = None
        ocr_fallback = None
        
        try:
            if processing_mode == ProcessingMode.VISION_ONLY:
                image_analysis = await self._analyze_with_vision(image_data, query_analysis)
                
            elif processing_mode == ProcessingMode.OCR_ONLY:
                ocr_fallback = await self._analyze_with_ocr(image_data, query_analysis)
                
            elif processing_mode == ProcessingMode.VISION_FIRST:
                image_analysis = await self._analyze_with_vision(image_data, query_analysis)
                
                # Vision 실패하거나 신뢰도가 낮으면 OCR 시도
                if (not image_analysis or 
                    image_analysis.confidence_score < self.config.vision_confidence_threshold):
                    logger.info("Vision analysis failed or low confidence, trying OCR fallback")
                    ocr_fallback = await self._analyze_with_ocr(image_data, query_analysis)
                    
            elif processing_mode == ProcessingMode.OCR_FIRST:
                ocr_fallback = await self._analyze_with_ocr(image_data, query_analysis)
                
                # OCR 실패하거나 신뢰도가 낮으면 Vision 시도
                if (not ocr_fallback or 
                    ocr_fallback.confidence_score < self.config.ocr_confidence_threshold):
                    logger.info("OCR analysis failed or low confidence, trying Vision")
                    image_analysis = await self._analyze_with_vision(image_data, query_analysis)
                    
            elif processing_mode == ProcessingMode.PARALLEL:
                # 병렬 처리
                tasks = []
                if self.vision_client:
                    tasks.append(self._analyze_with_vision(image_data, query_analysis))
                if self.ocr_client:
                    tasks.append(self._analyze_with_ocr(image_data, query_analysis))
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, VisionAnalysisResult):
                            image_analysis = result
                        elif isinstance(result, OCRResult):
                            ocr_fallback = result
            
            return image_analysis, ocr_fallback
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return None, None
    
    async def _analyze_with_vision(self, 
                                 image_data: Union[str, bytes],
                                 query_analysis: IntentAnalysisResult) -> Optional[VisionAnalysisResult]:
        """ChatGPT Vision으로 이미지 분석"""
        if not self.vision_client:
            return None
        
        try:
            # 질의 유형에 따른 분석 타입 결정
            analysis_type_map = {
                QueryType.VISUAL_ANALYSIS: AnalysisType.VISUAL_ANALYSIS,
                QueryType.TEXT_EXTRACTION: AnalysisType.TEXT_EXTRACTION,
                QueryType.FORMULA_ANALYSIS: AnalysisType.FORMULA_ANALYSIS,
                QueryType.PROBLEM_SOLVING: AnalysisType.PROBLEM_SOLVING,
                QueryType.DIAGRAM_INTERPRETATION: AnalysisType.DIAGRAM_INTERPRETATION
            }
            
            analysis_type = analysis_type_map.get(
                query_analysis.query_type, AnalysisType.GENERAL_ANALYSIS
            )
            
            # 비동기 래핑
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.vision_client.analyze_image,
                image_data,
                analysis_type,
                None,  # custom_prompt
                asdict(query_analysis)  # query_context
            )
            
            if result.success:
                self.stats['vision_usage'] += 1
                logger.info(f"Vision analysis successful: confidence={result.confidence_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return None
    
    async def _analyze_with_ocr(self, 
                              image_data: Union[str, bytes],
                              query_analysis: IntentAnalysisResult) -> Optional[OCRResult]:
        """NCP OCR로 이미지 분석"""
        if not self.ocr_client:
            return None
        
        try:
            # 질의 유형에 따른 분석 타입 결정
            analysis_type_map = {
                QueryType.TEXT_EXTRACTION: "text_extraction",
                QueryType.FORMULA_ANALYSIS: "formula_analysis",
                QueryType.PROBLEM_SOLVING: "problem_solving"
            }
            
            analysis_type = analysis_type_map.get(
                query_analysis.query_type, "text_extraction"
            )
            
            # 비동기 래핑
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.ocr_client.analyze_image,
                image_data,
                analysis_type,
                None,  # custom_prompt
                asdict(query_analysis)  # query_context
            )
            
            if result.success:
                self.stats['ocr_usage'] += 1
                logger.info(f"OCR analysis successful: confidence={result.confidence_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"OCR analysis failed: {e}")
            return None
    
    async def _perform_rag_search(self, 
                                query: str,
                                query_analysis: IntentAnalysisResult,
                                image_analysis: Optional[VisionAnalysisResult]) -> List[Dict[str, Any]]:
        """RAG 검색 수행"""
        if not self.rag_system or not self.config.use_rag:
            return []
        
        try:
            # 검색 쿼리 구성
            search_queries = [query]
            
            # 이미지 분석 결과가 있으면 추가 컨텍스트로 활용
            if image_analysis and image_analysis.success:
                if image_analysis.extracted_text:
                    search_queries.append(image_analysis.extracted_text)
                if image_analysis.analysis_result:
                    search_queries.append(image_analysis.analysis_result[:200])  # 처음 200자만
            
            # 도메인 결정
            domain_map = {
                'electrical': DomainType.ELECTRICAL,
                'mathematics': DomainType.MATHEMATICS,
                'physics': DomainType.PHYSICS,
                'chemistry': DomainType.CHEMISTRY
            }
            domain = domain_map.get(query_analysis.domain, DomainType.GENERAL)
            
            # 검색 전략 결정
            strategy = SearchStrategy.ADAPTIVE
            if query_analysis.complexity == ComplexityLevel.VERY_HIGH:
                strategy = SearchStrategy.CHAIN_OF_THOUGHT
            elif query_analysis.query_type == QueryType.FORMULA_ANALYSIS:
                strategy = SearchStrategy.SEMANTIC
            
            # 비동기 래핑하여 RAG 검색 수행
            loop = asyncio.get_event_loop()
            rag_results = []
            
            for search_query in search_queries:
                results = await loop.run_in_executor(
                    None,
                    self.rag_system.search,
                    search_query,
                    domain,
                    strategy
                )
                
                if results:
                    rag_results.extend(results[:3])  # 각 쿼리당 최대 3개 결과
            
            # 중복 제거 및 점수 정렬
            unique_results = []
            seen_content = set()
            
            for result in rag_results:
                content_key = result.get('content', '')[:100]  # 처음 100자로 중복 체크
                if content_key not in seen_content:
                    unique_results.append(result)
                    seen_content.add(content_key)
            
            unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            final_results = unique_results[:5]  # 최대 5개 결과
            
            if final_results:
                self.stats['rag_usage'] += 1
                logger.info(f"RAG search successful: {len(final_results)} results found")
            
            return final_results
            
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []
    
    async def _generate_llm_response(self, 
                                   query: str,
                                   query_analysis: IntentAnalysisResult,
                                   image_analysis: Optional[VisionAnalysisResult],
                                   rag_results: List[Dict[str, Any]]) -> Optional[ModelResponse]:
        """LLM 응답 생성"""
        if not self.llm_system or not self.config.use_llm:
            return None
        
        try:
            # 컨텍스트 구성
            context_parts = []
            
            # RAG 결과 추가
            if rag_results:
                context_parts.append("=== 참고 자료 ===")
                for i, result in enumerate(rag_results, 1):
                    content = result.get('content', '')
                    score = result.get('score', 0)
                    context_parts.append(f"{i}. {content} (관련도: {score:.2f})")
                context_parts.append("")
            
            # 이미지 분석 결과 추가
            if image_analysis and image_analysis.success:
                context_parts.append("=== 이미지 분석 결과 ===")
                if image_analysis.extracted_text:
                    context_parts.append(f"추출된 텍스트: {image_analysis.extracted_text}")
                if image_analysis.analysis_result:
                    context_parts.append(f"분석 결과: {image_analysis.analysis_result}")
                if image_analysis.formulas:
                    context_parts.append("감지된 수식:")
                    for formula in image_analysis.formulas[:3]:  # 최대 3개
                        context_parts.append(f"- {formula.get('latex', '')}")
                context_parts.append("")
            
            context = "\n".join(context_parts)
            
            # 도메인 결정
            domain_map = {
                'electrical': ModelDomain.ELECTRICAL,
                'mathematics': ModelDomain.MATHEMATICS,
                'physics': ModelDomain.PHYSICS
            }
            domain = domain_map.get(query_analysis.domain, ModelDomain.ELECTRICAL)
            
            # 비동기 래핑하여 LLM 응답 생성
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.llm_system.generate,
                query,
                context,
                domain
            )
            
            if response and not response.error_message:
                self.stats['llm_usage'] += 1
                logger.info(f"LLM response generated: confidence={response.confidence_score:.2f}")
            
            return response
            
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return None
    
    def _compose_final_answer(self, 
                            query_analysis: IntentAnalysisResult,
                            image_analysis: Optional[VisionAnalysisResult],
                            ocr_fallback: Optional[OCRResult],
                            rag_results: List[Dict[str, Any]],
                            llm_response: Optional[ModelResponse]) -> Tuple[str, float]:
        """최종 답변 구성"""
        try:
            answer_parts = []
            confidence_scores = []
            
            # 1. LLM 응답이 있고 품질이 좋으면 우선 사용
            if llm_response and not llm_response.error_message and llm_response.confidence_score >= 0.7:
                answer_parts.append(llm_response.content)
                confidence_scores.append(llm_response.confidence_score)
                
                # 추가 정보 제공
                if image_analysis and image_analysis.success:
                    if image_analysis.extracted_text and len(image_analysis.extracted_text) > 20:
                        answer_parts.append(f"\n\n📷 이미지에서 추출된 텍스트:\n{image_analysis.extracted_text}")
                
            # 2. LLM 응답이 없거나 품질이 낮으면 다른 소스 활용
            else:
                # 이미지 분석 결과 우선
                if image_analysis and image_analysis.success:
                    if image_analysis.analysis_result:
                        answer_parts.append(image_analysis.analysis_result)
                        confidence_scores.append(image_analysis.confidence_score)
                    
                    if image_analysis.extracted_text:
                        answer_parts.append(f"\n\n📝 추출된 텍스트:\n{image_analysis.extracted_text}")
                
                # OCR 폴백 결과
                elif ocr_fallback and ocr_fallback.success:
                    if ocr_fallback.extracted_text:
                        answer_parts.append(f"📄 이미지에서 추출된 텍스트:\n{ocr_fallback.extracted_text}")
                        confidence_scores.append(ocr_fallback.confidence_score)
                    
                    if ocr_fallback.formulas:
                        answer_parts.append("\n🔢 감지된 수식:")
                        for formula in ocr_fallback.formulas[:3]:
                            answer_parts.append(f"- {formula.get('latex', '')}")
                
                # RAG 결과만 있는 경우
                elif rag_results:
                    answer_parts.append("📚 관련 정보:")
                    for result in rag_results[:2]:
                        content = result.get('content', '')
                        answer_parts.append(f"• {content}")
                        confidence_scores.append(result.get('score', 0.5))
                
                # 아무 결과도 없는 경우
                else:
                    answer_parts.append("죄송합니다. 질문에 대한 적절한 답변을 찾을 수 없습니다.")
                    confidence_scores.append(0.1)
            
            # 최종 답변 구성
            final_answer = "\n".join(answer_parts)
            
            # 신뢰도 계산 (가중 평균)
            if confidence_scores:
                final_confidence = sum(confidence_scores) / len(confidence_scores)
            else:
                final_confidence = 0.1
            
            # 답변 품질 개선
            if len(final_answer) < 50:
                final_answer += "\n\n💡 더 자세한 정보가 필요하시면 구체적인 질문을 해주세요."
            
            return final_answer, final_confidence
            
        except Exception as e:
            logger.error(f"Failed to compose final answer: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}", 0.0
    
    def _create_metadata(self, processing_time: float, custom_config: Optional[Dict]) -> Dict[str, Any]:
        """메타데이터 생성"""
        return {
            'pipeline_version': '1.0.0',
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'components_used': {
                'query_analyzer': self.query_analyzer is not None,
                'vision_client': self.vision_client is not None,
                'ocr_client': self.ocr_client is not None,
                'rag_system': self.rag_system is not None,
                'llm_system': self.llm_system is not None
            },
            'custom_config': custom_config or {},
            'config': asdict(self.config)
        }
    
    def _update_stats(self, result: ProcessingResult):
        """통계 업데이트"""
        if result.success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
        
        # 평균 처리 시간 업데이트
        n = self.stats['total_requests']
        prev_avg_time = self.stats['average_processing_time']
        self.stats['average_processing_time'] = (
            (prev_avg_time * (n - 1) + result.processing_time) / n
        )
        
        # 평균 신뢰도 업데이트
        if result.success:
            prev_avg_conf = self.stats['average_confidence']
            success_count = self.stats['successful_requests']
            self.stats['average_confidence'] = (
                (prev_avg_conf * (success_count - 1) + result.confidence_score) / success_count
            )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """파이프라인 통계 반환"""
        success_rate = 0.0
        if self.stats['total_requests'] > 0:
            success_rate = self.stats['successful_requests'] / self.stats['total_requests']
        
        return {
            'pipeline_stats': self.stats,
            'success_rate': success_rate,
            'component_health': {
                'query_analyzer': self.query_analyzer is not None,
                'vision_client': self.vision_client is not None and self.vision_client.api_available,
                'ocr_client': self.ocr_client is not None and self.ocr_client.api_available,
                'rag_system': self.rag_system is not None,
                'llm_system': self.llm_system is not None
            }
        }
    
    # 편의 메서드들
    def process_text_only(self, query: str) -> ProcessingResult:
        """텍스트 전용 처리"""
        return asyncio.run(self.process_multimodal_query(query, None))
    
    def process_with_image(self, query: str, image_path: str) -> ProcessingResult:
        """이미지 포함 처리"""
        return asyncio.run(self.process_multimodal_query(query, image_path))


# 편의 함수들
def create_default_pipeline() -> MultimodalPipeline:
    """기본 파이프라인 생성"""
    config = PipelineConfig(
        processing_mode=ProcessingMode.ADAPTIVE,
        use_rag=True,
        use_llm=True,
        enable_caching=True
    )
    return MultimodalPipeline(config)

async def process_query_async(query: str, 
                            image_data: Optional[Union[str, bytes]] = None,
                            config: Optional[PipelineConfig] = None) -> ProcessingResult:
    """비동기 쿼리 처리 편의 함수"""
    pipeline = MultimodalPipeline(config)
    return await pipeline.process_multimodal_query(query, image_data)

def process_query_sync(query: str, 
                      image_data: Optional[Union[str, bytes]] = None,
                      config: Optional[PipelineConfig] = None) -> ProcessingResult:
    """동기 쿼리 처리 편의 함수"""
    return asyncio.run(process_query_async(query, image_data, config))


# 테스트 함수
async def test_pipeline():
    """파이프라인 테스트"""
    pipeline = create_default_pipeline()
    
    # 텍스트 전용 테스트
    result1 = await pipeline.process_multimodal_query("전압과 전류의 관계를 설명해주세요.")
    print(f"Text-only test: success={result1.success}, confidence={result1.confidence_score:.2f}")
    
    # 통계 출력
    stats = pipeline.get_pipeline_stats()
    print(f"Pipeline stats: {stats}")


if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test_pipeline())