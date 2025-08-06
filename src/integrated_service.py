#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
차세대 AI 챗봇 통합 서비스
Next-Generation AI Chatbot Integrated Service

질의 우선 분석 + 멀티모달 처리 + RAG + 파인튜닝 LLM + 에이전트 시스템
Query-First Analysis + Multimodal Processing + RAG + Fine-tuned LLM + Agent System
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

# 통합 시스템 컴포넌트 임포트
from advanced_ai_system import AdvancedAISystem, ReasoningType
from multimodal_pipeline import MultimodalPipeline, ProcessingMode, PipelineConfig
from query_intent_analyzer import QueryIntentAnalyzer, QueryType, ComplexityLevel
from openai_vision_client import OpenAIVisionClient, AnalysisType
from ncp_ocr_client import NCPOCRClient
from enhanced_rag_system import EnhancedRAGSystem, SearchStrategy, DomainType
from enhanced_llm_system import EnhancedLLMSystem, ModelType, ModelDomain

# 기존 시스템 호환성
from rag_system import RAGSystem
from llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """서비스 설정"""
    # 서비스 모드
    service_mode: str = "advanced"  # "basic", "advanced", "hybrid"
    
    # API 설정
    enable_openai_vision: bool = True
    enable_ncp_ocr: bool = True
    enable_rag: bool = True
    enable_fine_tuned_llm: bool = True
    enable_reasoning: bool = True
    enable_memory: bool = True
    enable_agents: bool = True
    
    # 성능 설정
    max_concurrent_requests: int = 10
    request_timeout: float = 120.0
    cache_results: bool = True
    cache_ttl: int = 3600  # 1시간
    
    # 품질 설정
    min_confidence_threshold: float = 0.6
    enable_result_validation: bool = True
    enable_fallback_chain: bool = True
    
    # 로깅 설정
    log_level: str = "INFO"
    log_detailed_processing: bool = True
    log_user_queries: bool = False  # 개인정보 보호


@dataclass
class ServiceRequest:
    """서비스 요청"""
    request_id: str
    query: str
    image_data: Optional[Union[str, bytes]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    processing_mode: Optional[str] = None
    reasoning_type: Optional[str] = None
    custom_config: Optional[Dict] = None
    metadata: Optional[Dict] = None


@dataclass
class ServiceResponse:
    """서비스 응답"""
    request_id: str
    success: bool
    response: str
    confidence_score: float
    processing_time: float
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None
    debug_info: Optional[Dict] = None


class IntegratedAIService:
    """차세대 AI 챗봇 통합 서비스"""
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        """
        서비스 초기화
        
        Args:
            config: 서비스 설정
        """
        self.config = config or ServiceConfig()
        self.service_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        # 시스템 컴포넌트
        self.advanced_ai_system = None
        self.multimodal_pipeline = None
        self.basic_rag_system = None
        self.basic_llm_client = None
        
        # 서비스 상태
        self.service_status = "initializing"
        self.active_requests: Dict[str, ServiceRequest] = {}
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        
        # 캐시
        self.response_cache: Dict[str, ServiceResponse] = {}
        
        # 초기화
        self._setup_logging()
        self._initialize_systems()
        
        self.service_status = "ready"
        logger.info(f"Integrated AI Service initialized: {self.service_id}")
    
    def _setup_logging(self):
        """로깅 설정"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_systems(self):
        """시스템 초기화"""
        try:
            # 고급 AI 시스템 (최우선)
            if self.config.service_mode in ["advanced", "hybrid"]:
                try:
                    self.advanced_ai_system = AdvancedAISystem()
                    logger.info("Advanced AI System initialized")
                except Exception as e:
                    logger.warning(f"Advanced AI System initialization failed: {e}")
            
            # 멀티모달 파이프라인
            if self.config.service_mode in ["advanced", "hybrid"] or not self.advanced_ai_system:
                try:
                    pipeline_config = PipelineConfig(
                        processing_mode=ProcessingMode.ADAPTIVE,
                        use_rag=self.config.enable_rag,
                        use_llm=self.config.enable_fine_tuned_llm,
                        enable_caching=self.config.cache_results
                    )
                    self.multimodal_pipeline = MultimodalPipeline(pipeline_config)
                    logger.info("Multimodal Pipeline initialized")
                except Exception as e:
                    logger.warning(f"Multimodal Pipeline initialization failed: {e}")
            
            # 기본 시스템 (폴백용)
            if self.config.service_mode in ["basic", "hybrid"] or (
                not self.advanced_ai_system and not self.multimodal_pipeline):
                try:
                    self.basic_rag_system = RAGSystem()
                    self.basic_llm_client = LLMClient()
                    logger.info("Basic systems initialized")
                except Exception as e:
                    logger.warning(f"Basic systems initialization failed: {e}")
            
            # 시스템 가용성 검증
            available_systems = []
            if self.advanced_ai_system:
                available_systems.append("Advanced AI")
            if self.multimodal_pipeline:
                available_systems.append("Multimodal Pipeline")
            if self.basic_rag_system and self.basic_llm_client:
                available_systems.append("Basic Systems")
            
            if not available_systems:
                raise Exception("No AI systems available")
            
            logger.info(f"Available systems: {', '.join(available_systems)}")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.service_status = "error"
            raise
    
    async def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """
        요청 처리 (메인 엔트리 포인트)
        
        Args:
            request: 서비스 요청
            
        Returns:
            ServiceResponse: 처리 결과
        """
        start_time = time.time()
        self.request_stats['total_requests'] += 1
        self.active_requests[request.request_id] = request
        
        try:
            logger.info(f"Processing request {request.request_id}: '{request.query[:50]}...'")
            
            # 캐시 확인
            if self.config.cache_results:
                cached_response = self._check_cache(request)
                if cached_response:
                    logger.info(f"Cache hit for request {request.request_id}")
                    return cached_response
            
            # 처리 모드 결정
            processing_system = self._determine_processing_system(request)
            
            # 시스템별 처리
            if processing_system == "advanced":
                response = await self._process_with_advanced_system(request)
            elif processing_system == "multimodal":
                response = await self._process_with_multimodal_pipeline(request)
            elif processing_system == "basic":
                response = await self._process_with_basic_system(request)
            else:
                raise Exception(f"Unknown processing system: {processing_system}")
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            response.processing_time = processing_time
            
            # 응답 검증
            if self.config.enable_result_validation:
                response = self._validate_response(response, request)
            
            # 통계 업데이트
            if response.success:
                self.request_stats['successful_requests'] += 1
            else:
                self.request_stats['failed_requests'] += 1
            
            self._update_processing_stats(processing_time)
            
            # 캐시 저장
            if self.config.cache_results and response.success:
                self._cache_response(request, response)
            
            # 로깅
            if self.config.log_detailed_processing:
                self._log_processing_details(request, response, processing_system)
            
            logger.info(f"Request {request.request_id} completed: "
                       f"success={response.success}, time={processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.request_stats['failed_requests'] += 1
            self._update_processing_stats(processing_time)
            
            logger.error(f"Request {request.request_id} failed: {e}")
            
            error_response = ServiceResponse(
                request_id=request.request_id,
                success=False,
                response=f"처리 중 오류가 발생했습니다: {str(e)}",
                confidence_score=0.0,
                processing_time=processing_time,
                session_id=request.session_id,
                error_message=str(e),
                metadata={'processing_system': 'error', 'error_type': type(e).__name__}
            )
            
            return error_response
            
        finally:
            # 요청 정리
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
    
    def _determine_processing_system(self, request: ServiceRequest) -> str:
        """처리 시스템 결정"""
        # 사용자 지정 모드 우선
        if request.processing_mode:
            if request.processing_mode == "advanced" and self.advanced_ai_system:
                return "advanced"
            elif request.processing_mode == "multimodal" and self.multimodal_pipeline:
                return "multimodal"
            elif request.processing_mode == "basic" and self.basic_rag_system:
                return "basic"
        
        # 자동 선택
        if self.config.service_mode == "advanced" and self.advanced_ai_system:
            return "advanced"
        elif self.config.service_mode == "basic" and self.basic_rag_system:
            return "basic"
        elif self.config.service_mode == "hybrid":
            # 하이브리드 모드: 복잡도에 따라 선택
            if self.advanced_ai_system and (
                request.image_data or 
                len(request.query) > 100 or 
                '?' in request.query and '계산' in request.query
            ):
                return "advanced"
            elif self.multimodal_pipeline:
                return "multimodal"
            else:
                return "basic"
        
        # 폴백 시스템 선택
        if self.advanced_ai_system:
            return "advanced"
        elif self.multimodal_pipeline:
            return "multimodal"
        elif self.basic_rag_system:
            return "basic"
        else:
            raise Exception("No processing system available")
    
    async def _process_with_advanced_system(self, request: ServiceRequest) -> ServiceResponse:
        """고급 AI 시스템으로 처리"""
        try:
            # 추론 타입 결정
            reasoning_type_map = {
                'chain_of_thought': ReasoningType.CHAIN_OF_THOUGHT,
                'deductive': ReasoningType.DEDUCTIVE,
                'inductive': ReasoningType.INDUCTIVE,
                'abductive': ReasoningType.ABDUCTIVE,
                'causal': ReasoningType.CAUSAL
            }
            
            reasoning_type = reasoning_type_map.get(
                request.reasoning_type or 'chain_of_thought',
                ReasoningType.CHAIN_OF_THOUGHT
            )
            
            # 고급 시스템 처리
            result = await self.advanced_ai_system.process_advanced_query(
                query=request.query,
                image_data=request.image_data,
                session_id=request.session_id,
                reasoning_type=reasoning_type,
                memory_context=self.config.enable_memory
            )
            
            response = ServiceResponse(
                request_id=request.request_id,
                success=result['success'],
                response=result['final_response'],
                confidence_score=result['confidence_score'],
                processing_time=result['processing_time'],
                session_id=result['session_id'],
                metadata={
                    'processing_system': 'advanced',
                    'reasoning_type': reasoning_type.value,
                    'memory_context_used': self.config.enable_memory,
                    'context_memories': result.get('context_memories', 0),
                    'reasoning_steps': len(result.get('reasoning_steps', [])),
                    'system_capabilities': ['agents', 'reasoning', 'memory', 'multimodal']
                }
            )
            
            # 디버그 정보 (상세 모드에서만)
            if self.config.log_detailed_processing:
                response.debug_info = {
                    'pipeline_result': result.get('pipeline_result'),
                    'reasoning_steps': result.get('reasoning_steps', [])[:3],  # 처음 3단계만
                    'memory_id': result.get('memory_id')
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Advanced system processing failed: {e}")
            raise
    
    async def _process_with_multimodal_pipeline(self, request: ServiceRequest) -> ServiceResponse:
        """멀티모달 파이프라인으로 처리"""
        try:
            # 처리 모드 설정
            custom_config = request.custom_config or {}
            if request.processing_mode and hasattr(ProcessingMode, request.processing_mode.upper()):
                custom_config['processing_mode'] = request.processing_mode
            
            # 멀티모달 파이프라인 처리
            result = await self.multimodal_pipeline.process_multimodal_query(
                query=request.query,
                image_data=request.image_data,
                custom_config=custom_config
            )
            
            response = ServiceResponse(
                request_id=request.request_id,
                success=result.success,
                response=result.final_answer,
                confidence_score=result.confidence_score,
                processing_time=result.processing_time,
                session_id=request.session_id,
                metadata={
                    'processing_system': 'multimodal',
                    'processing_mode': result.processing_mode.value,
                    'query_analysis': asdict(result.query_analysis) if result.query_analysis else None,
                    'image_processed': result.image_analysis is not None,
                    'ocr_used': result.ocr_fallback is not None,
                    'rag_results': len(result.rag_results),
                    'system_capabilities': ['query_analysis', 'vision', 'ocr', 'rag', 'llm']
                }
            )
            
            # 디버그 정보
            if self.config.log_detailed_processing:
                response.debug_info = {
                    'query_analysis': asdict(result.query_analysis) if result.query_analysis else None,
                    'image_analysis_success': result.image_analysis.success if result.image_analysis else False,
                    'ocr_success': result.ocr_fallback.success if result.ocr_fallback else False,
                    'rag_results_count': len(result.rag_results)
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Multimodal pipeline processing failed: {e}")
            raise
    
    async def _process_with_basic_system(self, request: ServiceRequest) -> ServiceResponse:
        """기본 시스템으로 처리"""
        try:
            # 이미지가 있으면 에러 (기본 시스템은 텍스트만 처리)
            if request.image_data:
                raise Exception("Basic system does not support image processing")
            
            # RAG 검색
            rag_results = []
            if self.basic_rag_system:
                rag_results, max_score = self.basic_rag_system.search(request.query)
            
            # 컨텍스트 구성
            context = ""
            if rag_results:
                context_parts = []
                for result in rag_results[:3]:  # 상위 3개만
                    context_parts.append(f"- {result.answer}")
                context = "\n".join(context_parts)
            
            # LLM 응답 생성
            llm_response = ""
            if self.basic_llm_client:
                llm_response = self.basic_llm_client.query(request.query, context)
            
            # 최종 응답
            if llm_response:
                final_response = llm_response
                confidence = 0.8 if rag_results else 0.6
            elif rag_results:
                final_response = f"관련 정보를 찾았습니다:\n\n{rag_results[0].answer}"
                confidence = max_score if rag_results else 0.4
            else:
                final_response = "죄송합니다. 관련 정보를 찾을 수 없습니다."
                confidence = 0.1
            
            response = ServiceResponse(
                request_id=request.request_id,
                success=confidence >= self.config.min_confidence_threshold,
                response=final_response,
                confidence_score=confidence,
                processing_time=0.0,  # 별도 계산됨
                session_id=request.session_id,
                metadata={
                    'processing_system': 'basic',
                    'rag_results': len(rag_results),
                    'llm_response_generated': bool(llm_response),
                    'max_rag_score': max_score if rag_results else 0.0,
                    'system_capabilities': ['rag', 'llm']
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Basic system processing failed: {e}")
            raise
    
    def _validate_response(self, response: ServiceResponse, request: ServiceRequest) -> ServiceResponse:
        """응답 검증"""
        # 기본 검증
        if not response.response or len(response.response.strip()) < 10:
            response.success = False
            response.confidence_score = 0.0
            response.response = "적절한 응답을 생성할 수 없습니다."
        
        # 신뢰도 임계값 검증
        if response.confidence_score < self.config.min_confidence_threshold:
            if self.config.enable_fallback_chain:
                # 폴백 체인 시도 (다른 시스템으로 재처리)
                # 실제 구현에서는 다른 시스템으로 재요청
                response.metadata = response.metadata or {}
                response.metadata['fallback_attempted'] = True
            else:
                response.response += "\n\n⚠️ 응답의 신뢰도가 낮습니다. 추가 확인이 필요할 수 있습니다."
        
        return response
    
    def _check_cache(self, request: ServiceRequest) -> Optional[ServiceResponse]:
        """캐시 확인"""
        if not self.config.cache_results:
            return None
        
        # 간단한 캐시 키 생성
        cache_key = f"{request.query}_{request.image_data is not None}"
        
        if cache_key in self.response_cache:
            cached = self.response_cache[cache_key]
            # TTL 확인 (간단한 구현)
            if hasattr(cached, 'cached_at'):
                cache_age = time.time() - cached.cached_at
                if cache_age < self.config.cache_ttl:
                    # 새로운 request_id로 복사
                    cached.request_id = request.request_id
                    cached.session_id = request.session_id
                    return cached
                else:
                    # 만료된 캐시 제거
                    del self.response_cache[cache_key]
        
        return None
    
    def _cache_response(self, request: ServiceRequest, response: ServiceResponse):
        """응답 캐시"""
        if not self.config.cache_results or not response.success:
            return
        
        cache_key = f"{request.query}_{request.image_data is not None}"
        response.cached_at = time.time()
        self.response_cache[cache_key] = response
        
        # 캐시 크기 제한 (간단한 LRU)
        if len(self.response_cache) > 1000:
            # 가장 오래된 항목 제거
            oldest_key = min(self.response_cache.keys(), 
                           key=lambda k: getattr(self.response_cache[k], 'cached_at', 0))
            del self.response_cache[oldest_key]
    
    def _update_processing_stats(self, processing_time: float):
        """처리 통계 업데이트"""
        self.request_stats['total_processing_time'] += processing_time
        
        total_requests = self.request_stats['total_requests']
        if total_requests > 0:
            self.request_stats['average_processing_time'] = (
                self.request_stats['total_processing_time'] / total_requests
            )
    
    def _log_processing_details(self, request: ServiceRequest, 
                              response: ServiceResponse, processing_system: str):
        """처리 상세 로깅"""
        if not self.config.log_user_queries:
            query_log = f"[Query length: {len(request.query)}]"
        else:
            query_log = f"Query: {request.query[:100]}..."
        
        logger.info(f"Processing details - {query_log}, "
                   f"System: {processing_system}, "
                   f"Success: {response.success}, "
                   f"Confidence: {response.confidence_score:.2f}, "
                   f"Time: {response.processing_time:.2f}s")
    
    # 서비스 관리 메서드들
    def get_service_status(self) -> Dict[str, Any]:
        """서비스 상태 조회"""
        uptime = datetime.now() - self.start_time
        success_rate = 0.0
        
        if self.request_stats['total_requests'] > 0:
            success_rate = (self.request_stats['successful_requests'] / 
                          self.request_stats['total_requests'])
        
        return {
            'service_id': self.service_id,
            'status': self.service_status,
            'uptime_seconds': uptime.total_seconds(),
            'uptime_formatted': str(uptime),
            'config': asdict(self.config),
            'request_stats': self.request_stats,
            'success_rate': success_rate,
            'active_requests': len(self.active_requests),
            'cache_size': len(self.response_cache) if self.config.cache_results else 0,
            'available_systems': self._get_available_systems(),
            'capabilities': self._get_service_capabilities()
        }
    
    def _get_available_systems(self) -> List[str]:
        """사용 가능한 시스템 목록"""
        systems = []
        if self.advanced_ai_system:
            systems.append("Advanced AI System")
        if self.multimodal_pipeline:
            systems.append("Multimodal Pipeline")
        if self.basic_rag_system and self.basic_llm_client:
            systems.append("Basic RAG + LLM")
        return systems
    
    def _get_service_capabilities(self) -> List[str]:
        """서비스 기능 목록"""
        capabilities = ["text_processing"]
        
        if self.advanced_ai_system or self.multimodal_pipeline:
            capabilities.extend(["image_processing", "vision_analysis", "ocr_fallback"])
        
        if self.advanced_ai_system:
            capabilities.extend(["reasoning", "memory_management", "agent_coordination"])
        
        if self.config.enable_rag:
            capabilities.append("knowledge_retrieval")
        
        if self.config.enable_fine_tuned_llm:
            capabilities.append("domain_expertise")
        
        return capabilities
    
    def clear_cache(self):
        """캐시 초기화"""
        self.response_cache.clear()
        logger.info("Response cache cleared")
    
    def shutdown(self):
        """서비스 종료"""
        self.service_status = "shutting_down"
        self.clear_cache()
        logger.info("Integrated AI Service shutdown completed")


# 편의 함수들
def create_integrated_service(config: Optional[ServiceConfig] = None) -> IntegratedAIService:
    """통합 서비스 생성"""
    return IntegratedAIService(config)

async def process_query_with_service(query: str, 
                                   image_data: Optional[Union[str, bytes]] = None,
                                   service_config: Optional[ServiceConfig] = None) -> ServiceResponse:
    """서비스로 쿼리 처리"""
    service = create_integrated_service(service_config)
    
    request = ServiceRequest(
        request_id=str(uuid.uuid4()),
        query=query,
        image_data=image_data
    )
    
    return await service.process_request(request)


# 서비스 실행 함수
async def run_service_demo():
    """서비스 데모 실행"""
    print("=== 차세대 AI 챗봇 통합 서비스 데모 ===\n")
    
    # 서비스 설정
    config = ServiceConfig(
        service_mode="hybrid",
        enable_openai_vision=True,
        enable_ncp_ocr=True,
        enable_rag=True,
        enable_fine_tuned_llm=True,
        enable_reasoning=True,
        enable_memory=True,
        log_detailed_processing=True
    )
    
    # 서비스 생성
    service = create_integrated_service(config)
    
    # 서비스 상태 출력
    status = service.get_service_status()
    print(f"서비스 상태: {status['status']}")
    print(f"사용 가능한 시스템: {', '.join(status['available_systems'])}")
    print(f"지원 기능: {', '.join(status['capabilities'])}")
    print()
    
    # 테스트 쿼리들
    test_requests = [
        ServiceRequest(
            request_id="test_1",
            query="전압과 전류의 관계를 설명해주세요.",
            processing_mode="advanced"
        ),
        ServiceRequest(
            request_id="test_2", 
            query="옴의 법칙을 사용해서 저항값을 계산하는 방법은?",
            reasoning_type="chain_of_thought"
        ),
        ServiceRequest(
            request_id="test_3",
            query="전기 회로에서 병렬 연결과 직렬 연결의 차이점은?",
            processing_mode="multimodal"
        )
    ]
    
    # 요청 처리
    for i, request in enumerate(test_requests, 1):
        print(f"=== 테스트 {i}: {request.query} ===")
        
        response = await service.process_request(request)
        
        print(f"성공: {response.success}")
        print(f"신뢰도: {response.confidence_score:.2f}")
        print(f"처리 시간: {response.processing_time:.2f}초")
        print(f"처리 시스템: {response.metadata.get('processing_system', 'unknown')}")
        print(f"응답: {response.response[:200]}...")
        print()
    
    # 최종 서비스 통계
    final_status = service.get_service_status()
    print("=== 서비스 통계 ===")
    print(f"총 요청 수: {final_status['request_stats']['total_requests']}")
    print(f"성공률: {final_status['success_rate']:.2%}")
    print(f"평균 처리 시간: {final_status['request_stats']['average_processing_time']:.2f}초")
    
    # 서비스 종료
    service.shutdown()


if __name__ == "__main__":
    # 데모 실행
    asyncio.run(run_service_demo())