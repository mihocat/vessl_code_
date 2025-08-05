#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Gradio UI Application with Improved Image and Formula Processing
향상된 이미지 및 수식 처리 기능을 갖춘 Gradio UI 애플리케이션
"""

import sys
import time
import logging
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image
import torch
import gradio as gr

from config import Config
from llm_client import LLMClient
from rag_system import RAGSystem, SearchResult
from services import WebSearchService, ResponseGenerator
from intelligent_rag_adapter import IntelligentRAGAdapter

# 향상된 멀티모달 프로세서
from enhanced_multimodal_processor import EnhancedMultimodalProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedChatService:
    """향상된 챗봇 서비스"""
    
    def __init__(self, config: Config, llm_client: LLMClient):
        """
        챗봇 서비스 초기화
        
        Args:
            config: 전체 설정 객체
            llm_client: LLM 클라이언트
        """
        self.config = config
        self.llm_client = llm_client
        
        # 컴포넌트 초기화
        self.rag_system = RAGSystem(
            rag_config=config.rag,
            dataset_config=config.dataset,
            llm_client=llm_client
        )
        self.web_search = WebSearchService(config.web_search)
        self.response_generator = ResponseGenerator(config.web_search)
        
        # Intelligent RAG 어댑터 초기화
        self.intelligent_adapter = IntelligentRAGAdapter(config, llm_client)
        
        # 향상된 멀티모달 프로세서 초기화
        use_openai_vision = config.openai.use_vision_api if hasattr(config, 'openai') else False
        self.multimodal_processor = EnhancedMultimodalProcessor(
            use_gpu=torch.cuda.is_available(),
            use_openai_vision=use_openai_vision
        )
        logger.info(f"Enhanced multimodal processor initialized (OpenAI Vision: {use_openai_vision})")
        
        # 대화 이력
        self.conversation_history = []
        
        # 처리 상태 추적
        self.last_processing_status = {
            'ocr_engines': [],
            'formulas_detected': 0,
            'processing_time': 0,
            'errors': []
        }
    
    def process_query(
        self, 
        question: str, 
        history: List[Tuple[str, str]],
        image: Optional[Image.Image] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        사용자 질의 처리 (향상된 버전)
        
        Args:
            question: 사용자 질문
            history: 대화 이력
            image: 선택적 이미지 입력
            **kwargs: 추가 옵션
            
        Returns:
            (응답 텍스트, 처리 상태)
        """
        start_time = time.time()
        
        # 상태 초기화
        self.last_processing_status = {
            'ocr_engines': [],
            'formulas_detected': 0,
            'processing_time': 0,
            'errors': []
        }
        
        # 빈 질문 처리
        if not question or not question.strip():
            return "질문을 입력해주세요.", self.last_processing_status
        
        try:
            # Intelligent RAG 사용 여부 결정
            context = {
                'conversation_history': history,
                'image': image
            }
            
            use_intelligent = self.intelligent_adapter.should_use_intelligent(question, context)
            
            if use_intelligent:
                logger.info("Using Intelligent RAG for complex query")
                try:
                    result = self.intelligent_adapter.process_sync(question, context)
                    response = result.get('response', "죄송합니다. 응답을 생성할 수 없습니다.")
                    self.last_processing_status['intelligent_rag'] = True
                except Exception as e:
                    logger.error(f"Intelligent RAG failed, falling back to standard: {e}")
                    self.last_processing_status['errors'].append(f"Intelligent RAG: {str(e)}")
                    use_intelligent = False
            
            if not use_intelligent:
                # 표준 처리 (향상된 버전)
                response = self._process_standard_query(question, history, image)
            
            # 응답 시간 추가
            elapsed_time = time.time() - start_time
            self.last_processing_status['processing_time'] = elapsed_time
            response += f"\n\n_응답시간: {elapsed_time:.2f}초_"
            
            # 대화 이력 업데이트
            self.conversation_history.append((question, response))
            
            return response, self.last_processing_status
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            self.last_processing_status['errors'].append(str(e))
            return "죄송합니다. 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", self.last_processing_status
    
    def _process_standard_query(
        self,
        question: str,
        history: List[Tuple[str, str]],
        image: Optional[Image.Image]
    ) -> str:
        """표준 쿼리 처리 (향상된 버전)"""
        # 이미지 분석 (이미지가 있는 경우)
        image_context = None
        original_question = question
        
        if image:
            try:
                logger.info("Processing image with enhanced multimodal processor...")
                
                # 멀티모달 처리
                multimodal_result = self.multimodal_processor.process_multimodal_query(
                    question, image
                )
                
                # 상태 업데이트
                if 'image_analysis' in multimodal_result:
                    analysis = multimodal_result['image_analysis']
                    self.last_processing_status['ocr_engines'] = analysis.get('engines_used', [])
                    self.last_processing_status['formulas_detected'] = len(analysis.get('formulas', []))
                    
                    # 이미지 컨텍스트 설정
                    image_context = analysis
                    
                    # 질문 확장
                    if multimodal_result.get('combined_query'):
                        question = multimodal_result['combined_query']
                    
                    logger.info(f"Multimodal processing completed: {self.last_processing_status}")
                    
            except Exception as e:
                logger.error(f"Multimodal processing failed: {e}", exc_info=True)
                self.last_processing_status['errors'].append(f"Multimodal: {str(e)}")
                image_context = {
                    "error": str(e),
                    "caption": "[이미지 분석 실패]",
                    "ocr_text": ""
                }
                question = original_question
        
        # RAG 검색 수행
        results, max_score = self.rag_system.search(question)
        
        # 응답 생성
        response = self._generate_enhanced_response(question, results, max_score, image_context)
        
        return response
    
    def _generate_enhanced_response(
        self, 
        question: str, 
        results: List[SearchResult], 
        max_score: float,
        image_context: Optional[dict] = None
    ) -> str:
        """향상된 응답 생성"""
        response_header = "답변: "
        
        # 신뢰도 수준 결정
        if image_context:
            confidence_level = "medium" if max_score >= self.config.rag.medium_confidence_threshold else "low"
        else:
            if max_score >= self.config.rag.high_confidence_threshold:
                confidence_level = "high"
            elif max_score >= self.config.rag.medium_confidence_threshold:
                confidence_level = "medium"
            else:
                confidence_level = "low"
        
        # 높은 신뢰도 - 직접 답변 사용 (이미지가 없는 경우만)
        if confidence_level == "high" and results and not image_context:
            best_result = results[0]
            response = response_header + best_result.answer
            
            if best_result.category and best_result.category != "general":
                response += f"\n\n_[카테고리: {best_result.category}]_"
        
        # 중간/낮은 신뢰도 또는 이미지가 있는 경우 - LLM 활용
        else:
            # 웹 검색 수행 (낮은 신뢰도이고 이미지가 없는 경우)
            web_results = []
            if confidence_level == "low" and not image_context:
                web_results = self.web_search.search(question)
            
            # 향상된 컨텍스트 준비
            context = self._prepare_enhanced_context(results, web_results, image_context)
            
            # 수식 해결 결과 추가
            if image_context and 'formula_solutions' in image_context:
                context += "\n\n수식 해결 결과:\n"
                for formula, solution in image_context['formula_solutions'].items():
                    context += f"- {formula} = {solution}\n"
            
            # 프롬프트 생성
            prompt = self.response_generator.generate_prompt(
                question, context, confidence_level
            )
            
            # LLM 응답 생성
            try:
                if prompt:
                    llm_response = self.llm_client.query(prompt, "")
                    response = response_header + llm_response
                else:
                    response = response_header + "죄송합니다. 관련 정보를 찾을 수 없습니다."
            except Exception as e:
                logger.error(f"LLM response generation failed: {e}")
                response = response_header + "죄송합니다. 응답 생성 중 오류가 발생했습니다."
        
        # 이미지 분석 정보 추가
        if image_context:
            if image_context.get('formulas'):
                response += f"\n\n📐 감지된 수식: {len(image_context['formulas'])}개"
            if image_context.get('ocr_text'):
                ocr_preview = image_context['ocr_text'][:100]
                if len(image_context['ocr_text']) > 100:
                    ocr_preview += "..."
                response += f"\n📝 추출된 텍스트: {ocr_preview}"
        
        # 관련 질문 추천
        related_questions = self._get_related_questions(question, results)
        if related_questions:
            response += "\n\n**💡 관련 질문:**"
            for q in related_questions:
                response += f"\n- {q}"
        
        # 점수와 신뢰도 정보 표시
        response += f"\n\n[점수: {max_score:.3f}, 신뢰도: {confidence_level}]"
        
        return response
    
    def _prepare_enhanced_context(
        self,
        results: List[SearchResult],
        web_results: List[dict],
        image_context: Optional[dict]
    ) -> str:
        """향상된 컨텍스트 준비"""
        context_parts = []
        
        # 이미지 컨텍스트
        if image_context:
            if image_context.get('caption'):
                context_parts.append(f"이미지 설명: {image_context['caption']}")
            
            if image_context.get('ocr_text'):
                context_parts.append(f"추출된 텍스트:\n{image_context['ocr_text']}")
            
            if image_context.get('formulas'):
                formula_text = "감지된 수식:\n"
                for i, formula in enumerate(image_context['formulas'], 1):
                    formula_text += f"{i}. {formula}\n"
                context_parts.append(formula_text)
        
        # RAG 검색 결과
        if results:
            rag_context = self.response_generator.prepare_context(results, web_results, None)
            context_parts.append(rag_context)
        
        return "\n\n".join(context_parts)
    
    def _get_related_questions(
        self, 
        question: str, 
        results: List[SearchResult]
    ) -> List[str]:
        """관련 질문 추출"""
        if not results or len(results) < 2:
            return []
        
        related = []
        seen_questions = {question.lower()}
        
        for result in results[1:]:
            if result.score >= 0.6:
                q_lower = result.question.lower()
                if q_lower not in seen_questions:
                    related.append(result.question)
                    seen_questions.add(q_lower)
                    
                    if len(related) >= 3:
                        break
        
        return related


def create_enhanced_gradio_app(config: Optional[Config] = None) -> gr.Blocks:
    """
    향상된 Gradio 애플리케이션 생성
    
    Args:
        config: 설정 객체 (없으면 기본값 사용)
        
    Returns:
        Gradio Blocks 인스턴스
    """
    # 설정 초기화
    if config is None:
        config = Config()
    
    # LLM 클라이언트 초기화
    llm_client = LLMClient(config.llm)
    
    # 서버 대기
    logger.info("Waiting for LLM server...")
    if not llm_client.wait_for_server():
        logger.error("Failed to connect to LLM server")
        raise RuntimeError("LLM server connection failed")
    
    # 향상된 챗봇 서비스 초기화
    chat_service = EnhancedChatService(config, llm_client)
    
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
                
                # 처리 상태 표시
                with gr.Accordion("처리 상태", open=False):
                    status_display = gr.JSON(label="최근 처리 상태")
            
            # 사이드바
            with gr.Column(scale=2):
                # 예제 질문
                gr.Markdown("### 💡 예제 질문")
                examples = gr.Examples(
                    examples=[
                        ["다산에듀는 너의 친구입니까?"],
                        ["회로도에 대해서 알려줘."],
                        ["객체지향 프로그래밍의 특징은?"],
                        ["이 수식을 풀어줘: ∫x²dx"],
                        ["피타고라스 정리를 증명해줘."]
                    ],
                    inputs=msg,
                    label="클릭하여 사용"
                )
                
                # 기능 안내
                gr.Markdown("""
                ### 🚀 향상된 기능
                - **다중 OCR 엔진**: 한국어/영어 텍스트 정확도 향상
                - **수식 인식**: LaTeX 수식 자동 감지 및 해결
                - **지능형 이미지 분석**: 캡션 생성 및 컨텍스트 이해
                """)
        
        # 이벤트 핸들러
        def respond(message: str, image, chat_history: List[Tuple[str, str]]):
            """메시지 응답 처리"""
            if not message.strip():
                return "", None, chat_history, {}
            
            response, status = chat_service.process_query(message, chat_history, image)
            
            # 이미지가 있는 경우 대화에 표시
            if image:
                display_msg = f"{message}\n[이미지 첨부됨]"
                if status.get('formulas_detected', 0) > 0:
                    display_msg += f" (수식 {status['formulas_detected']}개 감지)"
                chat_history.append((display_msg, response))
            else:
                chat_history.append((message, response))
            
            return "", None, chat_history, status
        
        def clear_chat():
            """대화 초기화"""
            chat_service.conversation_history.clear()
            return None, "", None, {}
        
        # 이벤트 바인딩
        submit.click(
            respond, 
            [msg, image_input, chatbot], 
            [msg, image_input, chatbot, status_display]
        )
        msg.submit(
            respond, 
            [msg, image_input, chatbot], 
            [msg, image_input, chatbot, status_display]
        )
        clear.click(
            clear_chat, 
            None, 
            [chatbot, msg, image_input, status_display]
        )
        
        # 통계 표시
        with gr.Accordion("📊 서비스 통계", open=False):
            with gr.Row():
                stats_display = gr.Textbox(
                    label="통계",
                    interactive=False,
                    lines=6
                )
                
                def update_stats():
                    """통계 업데이트"""
                    stats = chat_service.rag_system.get_stats()
                    total = stats['total_queries']
                    
                    if total > 0:
                        high_pct = (stats['high_confidence_hits'] / total) * 100
                        medium_pct = (stats['medium_confidence_hits'] / total) * 100
                        low_pct = (stats['low_confidence_hits'] / total) * 100
                    else:
                        high_pct = medium_pct = low_pct = 0
                    
                    # 멀티모달 프로세서 상태
                    mm_status = "활성화" if chat_service.multimodal_processor else "비활성화"
                    
                    return (
                        f"총 쿼리 수: {total}\n"
                        f"높은 신뢰도: {stats['high_confidence_hits']} ({high_pct:.1f}%)\n"
                        f"중간 신뢰도: {stats['medium_confidence_hits']} ({medium_pct:.1f}%)\n"
                        f"낮은 신뢰도: {stats['low_confidence_hits']} ({low_pct:.1f}%)\n"
                        f"평균 응답시간: {stats['avg_response_time']:.2f}초\n"
                        f"멀티모달 프로세서: {mm_status}"
                    )
                
                refresh_stats = gr.Button("새로고침", size="sm")
                refresh_stats.click(update_stats, None, stats_display)
                
                # 초기 통계 표시
                app.load(update_stats, None, stats_display)
    
    return app


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced RAG System Gradio UI")
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
        app = create_enhanced_gradio_app(config)
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