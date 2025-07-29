#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio UI Application for RAG System
RAG 시스템 Gradio UI 애플리케이션
"""

import sys
import time
import logging
from typing import List, Tuple, Optional

import gradio as gr

from config import Config
from llm_client import LLMClient
from rag_system import RAGSystem, SearchResult
from services import WebSearchService, ResponseGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatService:
    """통합 챗봇 서비스"""
    
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
        
        # 대화 이력
        self.conversation_history = []
        
    def process_query(self, question: str, history: List[Tuple[str, str]]) -> str:
        """
        사용자 질의 처리
        
        Args:
            question: 사용자 질문
            history: 대화 이력
            
        Returns:
            생성된 응답
        """
        start_time = time.time()
        
        # 빈 질문 처리
        if not question or not question.strip():
            return "질문을 입력해주세요."
        
        try:
            # RAG 검색 수행
            results, max_score = self.rag_system.search(question)
            
            # 응답 생성
            response = self._generate_response(question, results, max_score)
            
            # 응답 시간 추가
            elapsed_time = time.time() - start_time
            response += f"\n\n_응답시간: {elapsed_time:.2f}초_"
            
            # 대화 이력 업데이트
            self.conversation_history.append((question, response))
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return "죄송합니다. 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    
    def _generate_response(self, question: str, results: List[SearchResult], max_score: float) -> str:
        """
        검색 결과를 기반으로 응답 생성
        
        Args:
            question: 사용자 질문
            results: RAG 검색 결과
            max_score: 최고 유사도 점수
            
        Returns:
            생성된 응답
        """
        # response_header = f"### 질문: {question}\n\n"
        response_header = "답변: "
        
        # 신뢰도 수준 결정
        if max_score >= self.config.rag.high_confidence_threshold:
            confidence_level = "high"
        elif max_score >= self.config.rag.medium_confidence_threshold:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # 높은 신뢰도 - 직접 답변 사용
        if confidence_level == "high" and results:
            best_result = results[0]
            response = response_header + best_result.answer
            
            if best_result.category and best_result.category != "general":
                response += f"\n\n_[카테고리: {best_result.category}]_"
        
        # 중간/낮은 신뢰도 - LLM 활용
        else:
            # 웹 검색 수행 (낮은 신뢰도인 경우)
            web_results = []
            if confidence_level == "low":
                web_results = self.web_search.search(question)
            
            # 컨텍스트 준비
            context = self.response_generator.prepare_context(results, web_results)
            
            # 프롬프트 생성
            prompt = self.response_generator.generate_prompt(
                question, context, confidence_level
            )
            
            # LLM 응답 생성
            try:
                if prompt:  # 프롬프트가 있는 경우만 LLM 호출
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
        
        # 디버그 정보 (개발 모드에서만)
        # if logger.isEnabledFor(logging.DEBUG):
            response += f"\n\n_[점수: {max_score:.3f}, 신뢰도: {confidence_level}]_"
        
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
    Gradio 애플리케이션 생성
    
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
    
    # 챗봇 서비스 초기화
    chat_service = ChatService(config, llm_client)
    
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
                    msg = gr.Textbox(
                        label="질문을 입력하세요",
                        placeholder="궁금한 것을 물어보세요...",
                        lines=2,
                        scale=4
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
        
        # 이벤트 핸들러
        def respond(message: str, chat_history: List[Tuple[str, str]]):
            """메시지 응답 처리"""
            if not message.strip():
                return "", chat_history
            
            response = chat_service.process_query(message, chat_history)
            chat_history.append((message, response))
            return "", chat_history
        
        def clear_chat():
            """대화 초기화"""
            chat_service.conversation_history.clear()
            return None, ""
        
        # 이벤트 바인딩
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(clear_chat, None, [chatbot, msg])
        
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
                    
                    return (
                        f"총 쿼리 수: {total}\n"
                        f"높은 신뢰도: {stats['high_confidence_hits']} ({high_pct:.1f}%)\n"
                        f"중간 신뢰도: {stats['medium_confidence_hits']} ({medium_pct:.1f}%)\n"
                        f"낮은 신뢰도: {stats['low_confidence_hits']} ({low_pct:.1f}%)\n"
                        f"평균 응답시간: {stats['avg_response_time']:.2f}초"
                    )
                
                refresh_stats = gr.Button("새로고침", size="sm")
                refresh_stats.click(update_stats, None, stats_display)
                
                # 초기 통계 표시
                app.load(update_stats, None, stats_display)
    
    return app


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