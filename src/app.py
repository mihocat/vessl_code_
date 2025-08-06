#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
차세대 AI 챗봇 통합 Gradio UI 애플리케이션
Next-Generation AI Chatbot Integrated Gradio UI Application

통합 서비스 + 멀티모달 처리 + 고급 AI 시스템
Integrated Service + Multimodal Processing + Advanced AI System
"""

import sys
import time
import asyncio
import logging
import uuid
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image
import torch

import gradio as gr

from config import Config
# 차세대 통합 시스템 임포트
from integrated_service import IntegratedAIService, ServiceConfig, ServiceRequest, ServiceResponse
from advanced_ai_system import ReasoningType

# 기존 시스템 호환성 유지
from rag_system import RAGSystem, SearchResult
from services import WebSearchService, ResponseGenerator
from intelligent_rag_adapter import IntelligentRAGAdapter

# 조건부 LLM 클라이언트 임포트 (폴백용)
import os
if os.getenv("USE_OPENAI_LLM", "false").lower() == "true" or os.getenv("SKIP_VLLM", "false").lower() == "true":
    from llm_client_openai import LLMClient
else:
    from llm_client import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # 단순화된 이미지 분석기 우선 시도
    from simple_image_analyzer import Florence2ImageAnalyzer, SimpleMultimodalRAGService
    logger.info("Using Simple Image Analyzer")
    USE_SIMPLE_ANALYZER = True
except ImportError as e:
    logger.warning(f"Simple Image Analyzer import failed: {e}")
    USE_SIMPLE_ANALYZER = False
    try:
        # Vision Transformer 분석기 폴백
        from vision_transformer_analyzer import Florence2ImageAnalyzer
        logger.info("Using Vision Transformer Analyzer as fallback")
    except ImportError as e2:
        logger.warning(f"Vision Transformer import failed: {e2}")
        # 최종 폴백: 원본 Florence-2
        from image_analyzer import Florence2ImageAnalyzer
        logger.info("Using original Florence-2 Analyzer as final fallback")

# MultimodalRAGService import
if USE_SIMPLE_ANALYZER:
    MultimodalRAGService = SimpleMultimodalRAGService
else:
    from image_analyzer import MultimodalRAGService


class NextGenChatService:
    """차세대 통합 챗봇 서비스"""
    
    def __init__(self, config: Config):
        """
        차세대 챗봇 서비스 초기화
        
        Args:
            config: 전체 설정 객체
        """
        self.config = config
        
        # 서비스 설정 구성
        service_config = ServiceConfig(
            service_mode="hybrid",  # 고급/기본 시스템 자동 선택
            enable_openai_vision=True,
            enable_ncp_ocr=True,
            enable_rag=True,
            enable_fine_tuned_llm=True,
            enable_reasoning=True,
            enable_memory=True,
            enable_agents=True,
            max_concurrent_requests=10,
            request_timeout=120.0,
            cache_results=True,
            min_confidence_threshold=0.6,
            enable_result_validation=True,
            enable_fallback_chain=True,
            log_detailed_processing=True
        )
        
        # 통합 AI 서비스 초기화
        try:
            self.ai_service = IntegratedAIService(service_config)
            self.service_available = True
            logger.info("Next-generation AI service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize next-gen AI service: {e}")
            self.ai_service = None
            self.service_available = False
            
            # 폴백: 기존 시스템 초기화
            logger.info("Falling back to legacy system...")
            self._initialize_legacy_system()
        
        # 대화 이력 및 세션 관리
        self.conversation_history = []
        self.current_session_id = None
        self.session_stats = {
            'queries_count': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'total_processing_time': 0.0
        }
    
    def _initialize_legacy_system(self):
        """레거시 시스템 초기화 (폴백용)"""
        try:
            # LLM 클라이언트 초기화
            llm_client = LLMClient(self.config.llm)
            
            # 서버 대기
            logger.info("Waiting for LLM server...")
            if not llm_client.wait_for_server():
                logger.error("Failed to connect to LLM server")
                raise RuntimeError("LLM server connection failed")
            
            # 기존 컴포넌트 초기화
            self.legacy_rag_system = RAGSystem(
                rag_config=self.config.rag,
                dataset_config=self.config.dataset,
                llm_client=llm_client
            )
            self.legacy_web_search = WebSearchService(self.config.web_search)
            self.legacy_response_generator = ResponseGenerator(self.config.web_search)
            self.legacy_intelligent_adapter = IntelligentRAGAdapter(self.config, llm_client)
            
            self.legacy_available = True
            logger.info("Legacy system initialized successfully")
            
        except Exception as e:
            logger.error(f"Legacy system initialization failed: {e}")
            self.legacy_available = False
        
    def process_query(
        self, 
        question: str, 
        history: List[Tuple[str, str]],
        image: Optional[Image.Image] = None,
        processing_mode: Optional[str] = None,
        reasoning_type: Optional[str] = None
    ) -> str:
        """
        사용자 질의 처리 (차세대 통합 시스템)
        
        Args:
            question: 사용자 질문
            history: 대화 이력
            image: 선택적 이미지 입력
            processing_mode: 처리 모드 (advanced, multimodal, basic)
            reasoning_type: 추론 타입 (chain_of_thought, deductive, etc.)
            
        Returns:
            생성된 응답
        """
        start_time = time.time()
        self.session_stats['queries_count'] += 1
        
        # 빈 질문 처리
        if not question or not question.strip():
            return "질문을 입력해주세요."
        
        try:
            # 차세대 시스템 사용
            if self.service_available and self.ai_service:
                return self._process_with_nextgen_system(
                    question, history, image, processing_mode, reasoning_type
                )
            
            # 폴백: 레거시 시스템 사용
            elif hasattr(self, 'legacy_available') and self.legacy_available:
                return self._process_with_legacy_system(question, history, image)
            
            else:
                return "죄송합니다. 사용 가능한 AI 시스템이 없습니다."
                
        except Exception as e:
            self.session_stats['failed_responses'] += 1
            logger.error(f"Query processing failed: {e}")
            return f"처리 중 오류가 발생했습니다: {str(e)}"
    
    def _process_with_nextgen_system(self, 
                                   question: str, 
                                   history: List[Tuple[str, str]], 
                                   image: Optional[Image.Image],
                                   processing_mode: Optional[str],
                                   reasoning_type: Optional[str]) -> str:
        """차세대 시스템으로 처리"""
        try:
            # 이미지 데이터 준비
            image_data = None
            if image:
                import io
                import base64
                
                # PIL 이미지를 바이트로 변환
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                image_data = img_buffer.read()
            
            # 세션 ID 관리
            if not self.current_session_id:
                self.current_session_id = str(uuid.uuid4())
            
            # 서비스 요청 생성
            request = ServiceRequest(
                request_id=str(uuid.uuid4()),
                query=question,
                image_data=image_data,
                session_id=self.current_session_id,
                processing_mode=processing_mode,
                reasoning_type=reasoning_type,
                metadata={
                    'conversation_history': len(history),
                    'has_image': image is not None
                }
            )
            
            # 비동기 처리를 동기로 래핑
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response = loop.run_until_complete(
                    self.ai_service.process_request(request)
                )
            finally:
                loop.close()
            
            # 통계 업데이트
            processing_time = response.processing_time
            self.session_stats['total_processing_time'] += processing_time
            
            if response.success:
                self.session_stats['successful_responses'] += 1
                
                # 대화 이력 업데이트
                self.conversation_history.append((question, response.response))
                
                # 응답 구성
                final_response = response.response
                
                # 메타데이터 정보 추가
                if response.metadata:
                    system_info = []
                    if 'processing_system' in response.metadata:
                        system_info.append(f"시스템: {response.metadata['processing_system']}")
                    if 'system_capabilities' in response.metadata:
                        capabilities = response.metadata['system_capabilities']
                        if len(capabilities) > 3:
                            system_info.append(f"기능: {', '.join(capabilities[:3])}...")
                        else:
                            system_info.append(f"기능: {', '.join(capabilities)}")
                    
                    if system_info:
                        final_response += f"\n\n_[{', '.join(system_info)}]_"
                
                # 처리 시간 및 신뢰도 정보
                final_response += (f"\n\n_응답시간: {processing_time:.2f}초, "
                                 f"신뢰도: {response.confidence_score:.2f}_")
                
                return final_response
                
            else:
                self.session_stats['failed_responses'] += 1
                error_msg = response.response
                if response.error_message:
                    error_msg += f"\n\n오류 세부사항: {response.error_message}"
                return error_msg
                
        except Exception as e:
            logger.error(f"Next-gen system processing failed: {e}")
            raise
    
    def _process_with_legacy_system(self, 
                                  question: str, 
                                  history: List[Tuple[str, str]], 
                                  image: Optional[Image.Image]) -> str:
        """레거시 시스템으로 처리"""
        try:
            # 기존 로직과 유사하지만 단순화
            logger.info("Using legacy system for processing")
            
            # 이미지가 있으면 경고 메시지
            image_warning = ""
            if image:
                image_warning = "\n\n⚠️ 레거시 모드에서는 이미지 분석이 제한적입니다."
            
            # RAG 검색 수행
            results, max_score = self.legacy_rag_system.search(question)
            
            # 신뢰도에 따른 응답 생성
            if max_score >= 0.8:
                response = results[0].answer if results else "관련 정보를 찾을 수 없습니다."
            elif max_score >= 0.6:
                # 중간 신뢰도: 웹 검색 추가
                web_results = self.legacy_web_search.search(question)
                context = self.legacy_response_generator.prepare_context(results, web_results)
                prompt = self.legacy_response_generator.generate_prompt(question, context, "medium")
                response = "RAG 시스템을 통해 검색된 정보를 바탕으로 답변드립니다.\n\n"
                if results:
                    response += results[0].answer
                else:
                    response += "관련 정보를 찾을 수 없습니다."
            else:
                response = "죄송합니다. 관련 정보를 찾을 수 없습니다."
            
            # 최종 응답 구성
            final_response = response + image_warning
            final_response += f"\n\n_[레거시 모드, 점수: {max_score:.3f}]_"
            
            # 대화 이력 업데이트
            self.conversation_history.append((question, final_response))
            
            return final_response
            
        except Exception as e:
            logger.error(f"Legacy system processing failed: {e}")
            return f"레거시 시스템 처리 중 오류가 발생했습니다: {str(e)}"
    
    def get_service_status(self) -> Dict[str, Any]:
        """서비스 상태 조회"""
        status = {
            'next_gen_available': self.service_available,
            'legacy_available': getattr(self, 'legacy_available', False),
            'current_session_id': self.current_session_id,
            'session_stats': self.session_stats.copy()
        }
        
        # 차세대 시스템 상태 추가
        if self.service_available and self.ai_service:
            ai_status = self.ai_service.get_service_status()
            status['ai_service_status'] = ai_status
        
        return status
    
    
    def _generate_response(
        self, 
        question: str, 
        results: List[SearchResult], 
        max_score: float,
        image_context: Optional[dict] = None
    ) -> str:
        """
        검색 결과를 기반으로 응답 생성
        
        Args:
            question: 사용자 질문
            results: RAG 검색 결과
            max_score: 최고 유사도 점수
            image_context: 이미지 분석 결과
            
        Returns:
            생성된 응답
        """
        # response_header = f"### 질문: {question}\n\n"
        response_header = "답변: "
        
        # 사용자 요구사항: 2분기 구조 (0.75 이상 = RAG 직접, 0.75 미만 = 자체 LLM)
        # ChatGPT API는 질의당 1번만 호출
        
        # 신뢰도 0.75 이상: RAG 직접 답변 사용
        if max_score >= self.config.rag.high_confidence_threshold and results:
            best_result = results[0]
            response = response_header + best_result.answer
            
            if best_result.category and best_result.category != "general":
                response += f"\n\n_[카테고리: {best_result.category}]_"
                
            confidence_level = "high"
        
        # 신뢰도 0.75 미만: 자체 LLM 사용 (ChatGPT API 또는 vLLM)
        else:
            confidence_level = "low"
            
            # 이미지가 있는 경우 ChatGPT API 1회 호출 (Vision + Text 통합)
            if image_context:
                # 이미지 분석 실패 메시지 추가
                image_error_prefix = ""
                if "error" in image_context:
                    if "caption" in image_context and "[이미지 분석" in image_context["caption"]:
                        image_error_prefix = "[이미지 분석 실패] "
                
                # 컨텍스트 준비 (이미지 컨텍스트 포함)
                context = self.response_generator.prepare_context(results, [], image_context)
                
                # 프롬프트 생성
                prompt = self.response_generator.generate_prompt(
                    question, context, confidence_level
                )
                
                # ChatGPT API 1회 호출 (이미지+텍스트 통합)
                try:
                    if prompt:
                        llm_response = self.llm_client.query(prompt, "")
                        response = response_header + image_error_prefix + llm_response
                    else:
                        response = response_header + image_error_prefix + "죄송합니다. 관련 정보를 찾을 수 없습니다."
                except Exception as e:
                    logger.error(f"LLM response generation failed: {e}")
                    response = response_header + image_error_prefix + "죄송합니다. 응답 생성 중 오류가 발생했습니다."
            
            # 이미지가 없는 경우 자체 LLM 사용 (vLLM 우선)
            else:
                # 웹 검색 추가 (필요시)
                web_results = self.web_search.search(question)
                
                # 컨텍스트 준비
                context = self.response_generator.prepare_context(results, web_results, None)
                
                # 프롬프트 생성
                prompt = self.response_generator.generate_prompt(
                    question, context, confidence_level
                )
                
                # 자체 LLM 호출 (vLLM 우선, ChatGPT 폴백)
                try:
                    if prompt:
                        llm_response = self.llm_client.query(prompt, "")
                        response = response_header + llm_response
                    else:
                        response = response_header + "죄송합니다. 관련 정보를 찾을 수 없습니다."
                except Exception as e:
                    logger.error(f"LLM response generation failed: {e}")
                    response = response_header + "죄송합니다. 응답 생성 중 오류가 발생했습니다."
        
        # 관련 질문 추천
        related_questions = self._get_related_questions(question, results)
        if related_questions:
            response += "\n\n**💡 관련 질문:**"
            for q in related_questions:
                response += f"\n- {q}"
        
        # 점수와 신뢰도 정보 표시 (2분기 구조 표시)
        system_type = "RAG 직접응답" if confidence_level == "high" else "자체 LLM"
        response += f"\n\n[점수: {max_score:.3f}, 시스템: {system_type}]"
        
        return response
    
    def _get_related_questions(
        self, 
        question: str, 
        results: List[SearchResult]
    ) -> List[str]:
        """
        관련 질문 추출
        
        Args:
            question: 원본 질문
            results: 검색 결과
            
        Returns:
            관련 질문 리스트
        """
        if not results or len(results) < 2:
            return []
        
        related = []
        seen_questions = {question.lower()}
        
        for result in results[1:]:  # 첫 번째 결과는 제외
            if result.score >= 0.6:
                q_lower = result.question.lower()
                if q_lower not in seen_questions:
                    related.append(result.question)
                    seen_questions.add(q_lower)
                    
                    if len(related) >= 3:
                        break
        
        return related


def create_gradio_app(config: Optional[Config] = None) -> gr.Blocks:
    """
    차세대 Gradio 애플리케이션 생성
    
    Args:
        config: 설정 객체 (없으면 기본값 사용)
        
    Returns:
        Gradio Blocks 인스턴스
    """
    # 설정 초기화
    if config is None:
        config = Config()
    
    # 차세대 챗봇 서비스 초기화
    chat_service = NextGenChatService(config)
    
    # Gradio 인터페이스
    with gr.Blocks(title=config.app.title, theme=gr.themes.Soft()) as app:
        gr.Markdown(config.app.description)
        
        # 메인 인터페이스
        with gr.Row():
            with gr.Column(scale=8):
                # 채팅 인터페이스
                chatbot = gr.Chatbot(
                    label="대화",
                    height=500,
                    bubble_full_width=False,
                    show_label=True
                )
                
                # 입력 영역
                with gr.Row():
                    with gr.Column(scale=4):
                        msg = gr.Textbox(
                            label="질문을 입력하세요",
                            placeholder="궁금한 것을 물어보세요...",
                            lines=2
                        )
                        image_input = gr.Image(
                            label="이미지 업로드 (선택사항)",
                            type="pil",
                            height=200
                        )
                    with gr.Column(scale=1):
                        submit = gr.Button("전송", variant="primary")
                        clear = gr.Button("초기화")
            
            # 사이드바
            with gr.Column(scale=2):
                # 예제 질문
                gr.Markdown("### 💡 예제 질문")
                examples = gr.Examples(
                    examples=config.app.example_questions,
                    inputs=msg,
                    label="클릭하여 사용"
                )
        
        # 고급 설정 패널
        with gr.Accordion("🔧 고급 설정", open=False):
            with gr.Row():
                processing_mode = gr.Dropdown(
                    choices=["auto", "advanced", "multimodal", "basic"],
                    value="auto",
                    label="처리 모드",
                    info="auto: 자동 선택, advanced: 고급 AI, multimodal: 멀티모달, basic: 기본"
                )
                reasoning_type = gr.Dropdown(
                    choices=["chain_of_thought", "deductive", "inductive", "abductive", "causal"],
                    value="chain_of_thought",
                    label="추론 타입",
                    info="사고 과정의 유형 선택"
                )
        
        # 이벤트 핸들러
        def respond(message: str, image, chat_history: List[Tuple[str, str]], 
                   proc_mode: str, reason_type: str):
            """메시지 응답 처리 (차세대 시스템)"""
            if not message.strip():
                return "", None, chat_history, proc_mode, reason_type
            
            # 처리 모드 결정
            selected_mode = None if proc_mode == "auto" else proc_mode
            selected_reasoning = None if reason_type == "chain_of_thought" else reason_type
            
            response = chat_service.process_query(
                message, chat_history, image, selected_mode, selected_reasoning
            )
            
            # 이미지가 있는 경우 대화에 표시
            if image:
                chat_history.append((f"{message}\n[이미지 첨부됨]", response))
            else:
                chat_history.append((message, response))
            
            return "", None, chat_history, proc_mode, reason_type
        
        def clear_chat():
            """대화 초기화"""
            chat_service.conversation_history.clear()
            # 세션 ID도 리셋
            chat_service.current_session_id = None
            return None, "", None, "auto", "chain_of_thought"
        
        # 이벤트 바인딩
        submit.click(
            respond, 
            [msg, image_input, chatbot, processing_mode, reasoning_type], 
            [msg, image_input, chatbot, processing_mode, reasoning_type]
        )
        msg.submit(
            respond, 
            [msg, image_input, chatbot, processing_mode, reasoning_type], 
            [msg, image_input, chatbot, processing_mode, reasoning_type]
        )
        clear.click(
            clear_chat, 
            None, 
            [chatbot, msg, image_input, processing_mode, reasoning_type]
        )
        
        # 통계 표시
        with gr.Accordion("📊 서비스 통계", open=False):
            with gr.Row():
                with gr.Column():
                    stats_display = gr.Textbox(
                        label="세션 통계",
                        interactive=False,
                        lines=8
                    )
                with gr.Column():
                    system_status_display = gr.Textbox(
                        label="시스템 상태",
                        interactive=False,
                        lines=8
                    )
                
                def update_stats():
                    """통계 업데이트"""
                    try:
                        service_status = chat_service.get_service_status()
                        session_stats = service_status['session_stats']
                        
                        # 세션 통계
                        total_queries = session_stats['queries_count']
                        success_rate = 0.0
                        avg_time = 0.0
                        
                        if total_queries > 0:
                            success_rate = (session_stats['successful_responses'] / total_queries) * 100
                            avg_time = session_stats['total_processing_time'] / total_queries
                        
                        session_info = (
                            f"세션 ID: {service_status['current_session_id'] or 'None'}\n"
                            f"총 질문 수: {total_queries}\n"
                            f"성공한 응답: {session_stats['successful_responses']}\n"
                            f"실패한 응답: {session_stats['failed_responses']}\n"
                            f"성공률: {success_rate:.1f}%\n"
                            f"평균 처리시간: {avg_time:.2f}초\n"
                            f"총 처리시간: {session_stats['total_processing_time']:.2f}초"
                        )
                        
                        # 시스템 상태
                        system_info = (
                            f"차세대 시스템: {'✅ 사용 가능' if service_status['next_gen_available'] else '❌ 비활성화'}\n"
                            f"레거시 시스템: {'✅ 사용 가능' if service_status['legacy_available'] else '❌ 비활성화'}\n"
                        )
                        
                        # AI 서비스 상태 추가
                        if 'ai_service_status' in service_status:
                            ai_status = service_status['ai_service_status']
                            system_info += (
                                f"AI 서비스 상태: {ai_status['status']}\n"
                                f"사용 가능한 시스템: {len(ai_status['available_systems'])}\n"
                                f"지원 기능: {len(ai_status['capabilities'])}\n"
                                f"활성 요청: {ai_status['active_requests']}\n"
                                f"캐시 크기: {ai_status['cache_size']}"
                            )
                        
                        return session_info, system_info
                        
                    except Exception as e:
                        error_msg = f"통계 조회 실패: {str(e)}"
                        return error_msg, error_msg
                
                refresh_stats = gr.Button("새로고침", size="sm")
                refresh_stats.click(update_stats, None, [stats_display, system_status_display])
                
                # 초기 통계 표시
                app.load(update_stats, None, [stats_display, system_status_display])
    
    return app


def launch_app():
    """앱 실행 함수 (run_app.py에서 호출)"""
    # 설정 로드
    config = Config()
    
    # 앱 생성 및 실행
    try:
        app = create_gradio_app(config)
        app.launch(
            server_name=config.app.server_name,
            server_port=config.app.server_port,
            share=config.app.share,
            show_error=True,
            quiet=False
        )
    except Exception as e:
        logger.error(f"Failed to launch app: {e}")
        sys.exit(1)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System Gradio UI")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to config file (JSON format)"
    )
    parser.add_argument(
        "--server-port", 
        type=int, 
        default=7860,
        help="Server port (default: 7860)"
    )
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Create public Gradio link"
    )
    
    args = parser.parse_args()
    
    # 설정 로드
    config = Config()
    
    # 명령줄 인자로 오버라이드
    if args.server_port:
        config.app.server_port = args.server_port
    if args.share:
        config.app.share = args.share
    
    # 앱 생성 및 실행
    try:
        app = create_gradio_app(config)
        app.launch(
            server_name=config.app.server_name,
            server_port=config.app.server_port,
            share=config.app.share
        )
    except Exception as e:
        logger.error(f"Failed to launch app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()