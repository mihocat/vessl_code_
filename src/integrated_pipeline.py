#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 파이프라인
OpenAI 분석 (1회) → RAG 검색 → 파인튜닝 LLM → 최종 답변
"""

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
        
        # 2. RAG 시스템 초기화 (파인튜닝 LLM 클라이언트 필요)
        try:
            # vLLM 기반 파인튜닝 모델 사용
            self.llm_client = LLMClient(config.llm)
            self.rag_system = RAGSystem(
                rag_config=config.rag,
                dataset_config=config.dataset,
                llm_client=self.llm_client
            )
            logger.info("✅ RAG 시스템 초기화 완료")
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
            # ========== 단계 1: OpenAI 통합 분석 (1회 호출) ==========
            step1_start = time.time()
            pipeline_steps.append("OpenAI_Analysis")
            
            logger.info("🔍 단계 1: OpenAI GPT-4.1 통합 분석 시작")
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
            
            logger.info(f"✅ OpenAI 분석 완료 - 비용: ${analysis_result.cost:.4f}")
            
            # ========== 단계 2: RAG 검색 ==========
            rag_results = []
            if use_rag and self.rag_system:
                step2_start = time.time()
                pipeline_steps.append("RAG_Search")
                
                logger.info("📚 단계 2: RAG 검색 시작")
                
                # 검색 쿼리 구성 (분석 결과 활용)
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
            
            # ========== 단계 3: 파인튜닝 LLM 최종 답변 생성 ==========
            if use_llm and self.llm_client:
                step3_start = time.time()
                pipeline_steps.append("LLM_Response")
                
                logger.info("🤖 단계 3: 파인튜닝 LLM 답변 생성 시작")
                
                # 컨텍스트 구성
                context = self._build_context(analysis_result, rag_results, question)
                
                # 프롬프트 구성
                prompt = self._build_prompt(context, question, analysis_result)
                
                # LLM 답변 생성
                final_answer = self.llm_client.generate_response(prompt)
                processing_times['llm_generation'] = time.time() - step3_start
                
                logger.info("🤖 파인튜닝 LLM 답변 생성 완료")
                
            else:
                logger.error("❌ 파인튜닝 LLM 사용 불가")
                final_answer = "파인튜닝된 LLM을 사용할 수 없습니다. 시스템 설정을 확인해주세요."
            
            # ========== 결과 정리 ==========
            total_time = time.time() - start_time
            processing_times['total'] = total_time
            
            self.processing_stats['successful_queries'] += 1
            self.processing_stats['total_cost'] += total_cost
            
            logger.info(f"✅ 통합 파이프라인 완료 - 총 시간: {total_time:.2f}s, 비용: ${total_cost:.4f}")
            
            return PipelineResult(
                success=True,
                final_answer=final_answer,
                analysis_result={
                    'extracted_text': analysis_result.extracted_text,
                    'formulas': analysis_result.formulas,
                    'key_concepts': analysis_result.key_concepts,
                    'question_intent': analysis_result.question_intent,
                    'token_usage': analysis_result.token_usage,
                    'cost': analysis_result.cost
                },
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
        """컨텍스트 구성"""
        context_parts = []
        
        # OpenAI 분석 결과
        if analysis_result.extracted_text:
            context_parts.append(f"이미지에서 추출된 텍스트:\n{analysis_result.extracted_text}")
        
        if analysis_result.formulas:
            context_parts.append(f"감지된 수식:\n" + "\n".join(analysis_result.formulas))
        
        if analysis_result.key_concepts:
            context_parts.append(f"핵심 개념:\n" + ", ".join(analysis_result.key_concepts))
        
        # RAG 검색 결과
        if rag_results:
            rag_context = []
            for i, result in enumerate(rag_results[:5], 1):  # 상위 5개만
                rag_context.append(f"참고자료 {i}: {result.content[:500]}...")
            if rag_context:
                context_parts.append("관련 전문 자료:\n" + "\n".join(rag_context))
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, context: str, question: str, analysis_result) -> str:
        """프롬프트 구성"""
        prompt = f"""다음 정보를 바탕으로 사용자의 질문에 전문적이고 정확하게 답변해주세요.

질문: {question}

{context}

답변 지침:
1. 한국어로 자연스럽게 답변하세요
2. 전기공학 전문 용어를 정확히 사용하세요
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
                response = self.llm_client.generate_response("안녕하세요", max_tokens=10)
                status['llm_client'] = True
            except Exception as e:
                logger.error(f"LLM client health check failed: {e}")
        
        return status


def create_integrated_pipeline(config: Config) -> IntegratedPipeline:
    """통합 파이프라인 생성 편의 함수"""
    return IntegratedPipeline(config)