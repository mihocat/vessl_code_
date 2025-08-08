#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 파이프라인
OpenAI 분석 (1회) → RAG 검색 → 파인튜닝 LLM → 최종 답변
"""

import os
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from PIL import Image

from config import Config
from unified_analysis_processor import UnifiedAnalysisProcessor
from rag_system import RAGSystem, SearchResult
from llm_client import LLMClient

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """파이프라인 결과"""
    success: bool
    final_answer: str
    analysis_result: Optional[Dict] = None
    rag_results: Optional[List[SearchResult]] = None
    processing_times: Optional[Dict[str, float]] = None
    total_cost: Optional[float] = None
    error_message: Optional[str] = None
    pipeline_steps: Optional[List[str]] = None


class IntegratedPipeline:
    """통합 처리 파이프라인"""
    
    def __init__(self, config: Config):
        """
        파이프라인 초기화
        
        Args:
            config: 전체 설정 객체
        """
        self.config = config
        
        # 1. OpenAI 통합 분석 프로세서 (1회 호출 제한)
        openai_config = {
            'api_key': config.openai.api_key,
            'unified_model': config.openai.unified_model,
            'max_tokens': config.openai.max_tokens,
            'temperature': config.openai.temperature,
            'max_calls_per_query': config.openai.max_calls_per_query
        }
        self.openai_processor = UnifiedAnalysisProcessor(openai_config)
        
        # 2. RAG 시스템 및 LLM 클라이언트 초기화 (SKIP_VLLM 확인)
        skip_vllm = os.getenv("SKIP_VLLM", "false").lower() == "true"
        logger.info(f"🔧 SKIP_VLLM 환경변수: {skip_vllm}")
        
        if skip_vllm:
            logger.info("⚠️ SKIP_VLLM=true - 파인튜닝 LLM 및 RAG 시스템 비활성화")
            self.llm_client = None
            self.rag_system = None
        else:
            try:
                # vLLM 기반 파인튜닝 모델 사용
                logger.info("🔧 vLLM 기반 파인튜닝 LLM 클라이언트 초기화 중...")
                self.llm_client = LLMClient(config.llm)
                self.rag_system = RAGSystem(
                    rag_config=config.rag,
                    dataset_config=config.dataset,
                    llm_client=self.llm_client
                )
                logger.info("✅ RAG 시스템 및 LLM 클라이언트 초기화 완료")
            except Exception as e:
                logger.error(f"❌ RAG/LLM 시스템 초기화 실패: {e}")
                self.llm_client = None
                self.rag_system = None
        
        # 처리 통계
        self.processing_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'openai_calls': 0,
            'total_cost': 0.0
        }
        
        logger.info("IntegratedPipeline initialized")
    
    def process_query(
        self,
        question: str,
        image: Optional[Union[Image.Image, str, bytes]] = None,
        use_rag: bool = True,
        use_llm: bool = True
    ) -> PipelineResult:
        """
        질의 처리 메인 파이프라인
        
        Args:
            question: 사용자 질문
            image: 선택적 이미지
            use_rag: RAG 검색 사용 여부
            use_llm: 파인튜닝 LLM 사용 여부
            
        Returns:
            PipelineResult: 처리 결과
        """
        start_time = time.time()
        pipeline_steps = []
        processing_times = {}
        total_cost = 0.0
        
        self.processing_stats['total_queries'] += 1
        self.openai_processor.reset_call_count()  # 질의당 호출 횟수 초기화
        
        try:
            analysis_result = None
            rag_results = []
            
            # ========== 이미지 유무에 따른 파이프라인 분기 ==========
            if image is not None:
                # 경로 1: 이미지 포함 → OpenAI → RAG → LLM
                logger.info("🖼️ 이미지 포함 질의 - OpenAI → RAG → LLM 파이프라인 시작")
                
                # 단계 1: OpenAI 통합 분석 (이미지 처리)
                step1_start = time.time()
                pipeline_steps.append("OpenAI_Analysis")
                
                logger.info("🔍 단계 1: OpenAI GPT-5 이미지+텍스트 통합 분석 시작")
                analysis_result = self.openai_processor.analyze_image_and_text(question, image)
                
                if not analysis_result.success:
                    return PipelineResult(
                        success=False,
                        final_answer="이미지 분석에 실패했습니다.",
                        error_message=analysis_result.error_message,
                        pipeline_steps=pipeline_steps
                    )
                
                processing_times['openai_analysis'] = time.time() - step1_start
                total_cost += analysis_result.cost or 0.0
                self.processing_stats['openai_calls'] += 1
                
                logger.info(f"✅ OpenAI 이미지 분석 완료 - 비용: ${analysis_result.cost:.4f}")
                
                # 단계 2: RAG 검색 (OpenAI 분석 결과 활용)
                if use_rag and self.rag_system:
                    step2_start = time.time()
                    pipeline_steps.append("RAG_Search")
                    
                    logger.info("📚 단계 2: RAG 검색 시작 (OpenAI 분석 결과 활용)")
                    
                    # 검색 쿼리 구성 (OpenAI 분석 결과 활용)
                    search_query = question
                    if analysis_result.key_concepts:
                        search_query += " " + " ".join(analysis_result.key_concepts[:3])
                    if analysis_result.extracted_text:
                        search_query += " " + analysis_result.extracted_text[:200]
                    
                    rag_results, max_score = self.rag_system.search(search_query)
                    processing_times['rag_search'] = time.time() - step2_start
                    
                    logger.info(f"📚 RAG 검색 완료 - {len(rag_results)}개 문서 발견, 최고점수: {max_score:.3f}")
                else:
                    logger.warning("⚠️ RAG 시스템 비활성화 또는 초기화 실패")
                    
            else:
                # 경로 2: 텍스트만 → RAG → LLM (OpenAI 건너뛰기)
                logger.info("📝 텍스트 전용 질의 - RAG → LLM 파이프라인 시작")
                
                # 단계 1: RAG 검색 (텍스트 기반)
                if use_rag and self.rag_system:
                    step1_start = time.time()
                    pipeline_steps.append("RAG_Search")
                    
                    logger.info("🔍 단계 1: RAG 검색 시작 (텍스트 전용)")
                    
                    # 검색 쿼리 구성 (질문 텍스트만 사용)
                    search_query = question
                    
                    rag_results, max_score = self.rag_system.search(search_query)
                    processing_times['rag_search'] = time.time() - step1_start
                    
                    logger.info(f"📚 RAG 검색 완료 - {len(rag_results)}개 문서 발견, 최고점수: {max_score:.3f}")
                else:
                    logger.warning("⚠️ RAG 시스템 비활성화 또는 초기화 실패")
            
            # ========== 최종 단계: 파인튜닝 LLM 답변 생성 ==========
            if use_llm and self.llm_client:
                llm_step_start = time.time()
                pipeline_steps.append("LLM_Response")
                
                # 파이프라인 경로에 따른 로깅
                if image is not None:
                    logger.info("🤖 단계 3: 파인튜닝 LLM 답변 생성 시작 (이미지+텍스트)")
                else:
                    logger.info("🤖 단계 2: 파인튜닝 LLM 답변 생성 시작 (텍스트 전용)")
                
                # 컨텍스트 구성
                context = self._build_context(analysis_result, rag_results, question)
                
                # 프롬프트 구성
                prompt = self._build_prompt(context, question, analysis_result)
                
                # LLM 답변 생성 시도
                final_answer = self.llm_client.generate_response(
                    question=question,
                    context=context
                )
                processing_times['llm_generation'] = time.time() - llm_step_start
                
                # LLM 연결 실패 시 에러 메시지 (파이프라인 경로별로 다른 메시지)
                if "응답 생성 중 오류가 발생했습니다" in final_answer:
                    logger.error("❌ LLM 서버 연결 실패 - 파인튜닝 모델 사용 불가")
                    
                    if image is not None:
                        # 이미지 포함 질의 실패
                        final_answer = f"""죄송합니다. 현재 파인튜닝된 모델 서버에 연결할 수 없습니다.

**처리 완료된 단계:**
- OpenAI 이미지 분석: 완료
- RAG 문서 검색: {len(rag_results)}개 결과
- 파인튜닝 LLM: 연결 실패"""
                    else:
                        # 텍스트 전용 질의 실패
                        final_answer = f"""죄송합니다. 현재 파인튜닝된 모델 서버에 연결할 수 없습니다.

**처리 완료된 단계:**
- RAG 문서 검색: {len(rag_results)}개 결과  
- 파인튜닝 LLM: 연결 실패

시스템 관리자에게 문의하시거나 잠시 후 다시 시도해 주세요."""
                
                logger.info("🤖 최종 답변 생성 완료")
                
            else:
                logger.info("⚠️ 파인튜닝 LLM 단계 건너뛰기 - 기본 답변 제공")
                
                # 파이프라인 경로별 fallback 답변 구성
                if image is not None:
                    # 이미지 포함 질의 - OpenAI 분석 결과 활용
                    if analysis_result and hasattr(analysis_result, 'extracted_text') and analysis_result.extracted_text:
                        final_answer = f"""**OpenAI GPT-5 이미지 분석 결과:**

{analysis_result.extracted_text}

**참고사항:**
- 현재 파인튜닝된 전문 모델은 비활성화 상태입니다
- 위 답변은 OpenAI GPT-5 의 이미지 분석 결과입니다
- 더 전문적인 답변이 필요하시면 시스템 관리자에게 문의해 주세요"""
                    else:
                        final_answer = f"""**이미지 분석 결과:**

질문: {question[:100]}{'...' if len(question) > 100 else ''}

현재 파인튜닝된 전문 모델이 비활성화되어 있어 상세한 전문 분석을 제공할 수 없습니다.

**처리 완료된 단계:**
- OpenAI 이미지 분석: 완료
- RAG 검색: {len(rag_results)}개 문서
- 파인튜닝 LLM: 비활성화됨"""
                else:
                    # 텍스트 전용 질의 - RAG 결과 기반
                    final_answer = f"""**텍스트 질의 처리 결과:**

질문: {question[:100]}{'...' if len(question) > 100 else ''}

현재 파인튜닝된 전문 모델이 비활성화되어 있어 상세한 전문 분석을 제공할 수 없습니다.

**처리 완료된 단계:**
- RAG 검색: {len(rag_results)}개 문서
- 파인튜닝 LLM: 비활성화됨

보다 전문적인 답변이 필요하시면 시스템 관리자에게 문의해 주세요."""
            
            # ========== 결과 정리 ==========
            total_time = time.time() - start_time
            processing_times['total'] = total_time
            
            self.processing_stats['successful_queries'] += 1
            self.processing_stats['total_cost'] += total_cost
            
            logger.info(f"✅ 통합 파이프라인 완료 - 총 시간: {total_time:.2f}s, 비용: ${total_cost:.4f}")
            
            # 분석 결과 구성 (이미지 포함 질의인 경우에만)
            analysis_dict = None
            if analysis_result is not None:
                analysis_dict = {
                    'extracted_text': getattr(analysis_result, 'extracted_text', ''),
                    'formulas': getattr(analysis_result, 'formulas', []),
                    'key_concepts': getattr(analysis_result, 'key_concepts', []),
                    'question_intent': getattr(analysis_result, 'question_intent', ''),
                    'token_usage': getattr(analysis_result, 'token_usage', {}),
                    'cost': getattr(analysis_result, 'cost', 0.0)
                }
            
            return PipelineResult(
                success=True,
                final_answer=final_answer,
                analysis_result=analysis_dict,
                rag_results=rag_results,
                processing_times=processing_times,
                total_cost=total_cost,
                pipeline_steps=pipeline_steps
            )
            
        except Exception as e:
            self.processing_stats['failed_queries'] += 1
            total_time = time.time() - start_time
            
            logger.error(f"❌ 파이프라인 처리 실패: {e}")
            
            return PipelineResult(
                success=False,
                final_answer="처리 중 오류가 발생했습니다.",
                error_message=str(e),
                processing_times={'total': total_time},
                pipeline_steps=pipeline_steps
            )
    
    def _build_context(
        self, 
        analysis_result, 
        rag_results: List[SearchResult], 
        question: str
    ) -> str:
        """컨텍스트 구성 - 토큰 제한 고려"""
        context_parts = []
        max_total_length = 1500  # 전체 컨텍스트 최대 길이
        current_length = 0
        
        # OpenAI 분석 결과 (이미지 포함 질의인 경우에만 존재)
        if analysis_result is not None:
            if hasattr(analysis_result, 'extracted_text') and analysis_result.extracted_text:
                text = f"이미지에서 추출된 텍스트:\n{analysis_result.extracted_text[:300]}"
                context_parts.append(text)
                current_length += len(text)
            
            if hasattr(analysis_result, 'formulas') and analysis_result.formulas:
                formulas = analysis_result.formulas[:3]  # 최대 3개 수식만
                text = f"감지된 수식:\n" + "\n".join(formulas)
                if current_length + len(text) < max_total_length:
                    context_parts.append(text)
                    current_length += len(text)
            
            if hasattr(analysis_result, 'key_concepts') and analysis_result.key_concepts:
                concepts = analysis_result.key_concepts[:5]  # 최대 5개 개념만
                text = f"핵심 개념:\n" + ", ".join(concepts)
                if current_length + len(text) < max_total_length:
                    context_parts.append(text)
                    current_length += len(text)
        
        # RAG 검색 결과
        if rag_results:
            rag_context = []
            remaining_length = max_total_length - current_length
            per_result_length = min(300, remaining_length // min(3, len(rag_results)))
            
            for i, result in enumerate(rag_results[:3], 1):  # 상위 3개만
                # SearchResult 클래스의 올바른 속성 사용: answer (content가 아님)
                truncated_answer = result.answer[:per_result_length]
                if len(result.answer) > per_result_length:
                    truncated_answer += "..."
                rag_context.append(f"참고자료 {i}: {truncated_answer}")
                
            if rag_context:
                context_parts.append("관련 전문 자료:\n" + "\n".join(rag_context))
        
        final_context = "\n\n".join(context_parts)
        
        # 최종 길이 확인
        if len(final_context) > max_total_length:
            final_context = final_context[:max_total_length] + "..."
            logger.warning(f"컨텍스트가 {max_total_length}자로 제한됨")
        
        return final_context
    
    def _build_prompt(self, context: str, question: str, analysis_result) -> str:
        """프롬프트 구성"""
        prompt = f"""다음 정보를 바탕으로 사용자의 질문에 전문적이고 정확하게 답변해주세요.

질문: {question}

{context}

답변 지침:
1. 한국어로 자연스럽게 답변하세요
2. 전문 용어를 정확히 사용하세요
3. 수식이 있으면 LaTeX 형식으로 표현하세요
4. 단계별로 체계적으로 설명하세요
5. 실무에 도움이 되는 구체적인 정보를 포함하세요

답변:"""
        
        return prompt
    
    def get_statistics(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        success_rate = 0.0
        if self.processing_stats['total_queries'] > 0:
            success_rate = self.processing_stats['successful_queries'] / self.processing_stats['total_queries']
        
        avg_cost = 0.0
        if self.processing_stats['successful_queries'] > 0:
            avg_cost = self.processing_stats['total_cost'] / self.processing_stats['successful_queries']
        
        return {
            **self.processing_stats,
            'success_rate': success_rate,
            'average_cost_per_query': avg_cost,
            'openai_call_efficiency': f"{self.processing_stats['openai_calls']}/{self.processing_stats['total_queries']} (1:1 목표)"
        }
    
    def health_check(self) -> Dict[str, bool]:
        """시스템 상태 확인"""
        status = {
            'openai_processor': False,
            'rag_system': False,
            'llm_client': False
        }
        
        # OpenAI 프로세서 확인
        try:
            stats = self.openai_processor.get_call_statistics()
            status['openai_processor'] = True
        except Exception as e:
            logger.error(f"OpenAI processor health check failed: {e}")
        
        # RAG 시스템 확인
        if self.rag_system:
            try:
                # 간단한 검색 테스트
                results = self.rag_system.search("테스트", k=1)
                status['rag_system'] = True
            except Exception as e:
                logger.error(f"RAG system health check failed: {e}")
        
        # LLM 클라이언트 확인
        if self.llm_client:
            try:
                # 간단한 생성 테스트
                response = self.llm_client.generate_response(
                    question="안녕하세요", 
                    max_tokens=10
                )
                # 실제 응답이 있고 오류 메시지가 아닌지 확인
                if response and "응답 생성 중 오류가 발생했습니다" not in response:
                    status['llm_client'] = True
                else:
                    status['llm_client'] = False
            except Exception as e:
                logger.error(f"LLM client health check failed: {e}")
                status['llm_client'] = False
        
        return status
    


def create_integrated_pipeline(config: Config) -> IntegratedPipeline:
    """Create integrated pipeline convenience function"""
    return IntegratedPipeline(config)