#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Gradio UI Application with Multimodal RAG System
멀티모달 RAG 시스템을 갖춘 향상된 Gradio UI 애플리케이션
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

# 향상된 시스템 임포트
from enhanced_rag_system import (
    EnhancedVectorDatabase, 
    EnhancedRAGSystem,
    RAGSystemAdapter
)
from enhanced_image_analyzer import ChatGPTStyleAnalyzer
from chatgpt_response_generator import ChatGPTResponseGenerator

# 기존 서비스 (호환성)
from services import WebSearchService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedChatService:
    """향상된 통합 챗봇 서비스"""
    
    def __init__(self, config: Config, llm_client: LLMClient):
        """
        향상된 챗봇 서비스 초기화
        
        Args:
            config: 전체 설정 객체
            llm_client: LLM 클라이언트
        """
        self.config = config
        self.llm_client = llm_client
        
        # 향상된 컴포넌트 초기화
        logger.info("Initializing enhanced components...")
        
        # 1. 향상된 벡터 DB
        self.vector_db = EnhancedVectorDatabase(
            persist_directory=config.rag.persist_directory
        )
        
        # 2. 향상된 RAG 시스템
        self.enhanced_rag = EnhancedRAGSystem(
            vector_db=self.vector_db,
            llm_client=llm_client
        )
        
        # 3. 호환성을 위한 어댑터
        self.rag_system = RAGSystemAdapter(self.enhanced_rag)
        
        # 4. 이미지 분석기
        self.image_analyzer = ChatGPTStyleAnalyzer(use_florence=True)
        
        # 5. 멀티모달 OCR (추가)
        try:
            from multimodal_ocr import MultimodalOCRPipeline
            self.ocr_pipeline = MultimodalOCRPipeline()
            logger.info("Multimodal OCR pipeline loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load multimodal OCR pipeline: {e}")
            self.ocr_pipeline = None
        
        # 6. 응답 생성기
        self.response_generator = ChatGPTResponseGenerator()
        
        # 7. 웹 검색 서비스
        self.web_search = WebSearchService(config.web_search)
        
        # 대화 이력
        self.conversation_history = []
        
        logger.info("Enhanced chat service initialized successfully")
        
    def process_query(
        self, 
        question: str, 
        history: List[Tuple[str, str]],
        image: Optional[Image.Image] = None
    ) -> str:
        """
        사용자 질의 처리 (향상된 버전)
        
        Args:
            question: 사용자 질문
            history: 대화 이력
            image: 선택적 이미지 입력
            
        Returns:
            생성된 응답
        """
        start_time = time.time()
        
        # 빈 질문 처리
        if not question or not question.strip():
            return "질문을 입력해주세요."
        
        try:
            # 이미지가 있는 경우 멀티모달 처리
            response_style = self._determine_response_style(question, image)
            
            # 향상된 RAG 시스템으로 처리
            result = self.enhanced_rag.process_query(
                query=question,
                image=image,
                response_style=response_style
            )
            
            if result['success']:
                response = result['response']
                
                # 소스 정보 추가 (선택적)
                if self.config.app.show_sources:
                    response += self._format_sources(result['search_results'])
                
                # 응답 시간 추가
                elapsed_time = time.time() - start_time
                response += f"\n\n_응답시간: {elapsed_time:.2f}초_"
                
                # 대화 이력 업데이트
                self.conversation_history.append((question, response))
                
                return response
            else:
                return "죄송합니다. 응답을 생성할 수 없습니다."
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            return "죄송합니다. 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    def _determine_response_style(self, question: str, image: Optional[Image.Image]) -> str:
        """응답 스타일 결정"""
        question_lower = question.lower()
        
        # 단계별 설명 요청
        if any(keyword in question_lower for keyword in ['단계', '순서', '방법', '어떻게']):
            return 'step_by_step'
        
        # 개념 설명 요청
        if any(keyword in question_lower for keyword in ['무엇', '정의', '개념', '의미']):
            return 'concept'
        
        # 이미지가 있는 경우 종합적 응답
        if image:
            return 'comprehensive'
        
        # 기본값
        return 'comprehensive'
    
    def _format_sources(self, search_results: List[Dict[str, Any]]) -> str:
        """검색 결과 소스 포맷팅"""
        if not search_results:
            return ""
        
        sources_text = "\n\n📚 **참고 자료:**"
        for i, result in enumerate(search_results[:3]):
            score = result.get('hybrid_score', 0)
            metadata = result.get('metadata', {})
            
            sources_text += f"\n{i+1}. "
            if metadata.get('title'):
                sources_text += f"{metadata['title']} "
            sources_text += f"(점수: {score:.3f})"
        
        return sources_text
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 조회"""
        stats = {
            'total_queries': len(self.conversation_history),
            'vector_db_size': self.vector_db.collection.count(),
            'model_status': 'active' if self.llm_client else 'inactive',
            'image_analyzer': 'active' if self.image_analyzer else 'inactive'
        }
        return stats


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
    with gr.Blocks(
        title="AI 전기공학 튜터 (향상된 버전)",
        theme=gr.themes.Soft(),
        css="""
        .message { font-size: 16px; }
        .latex-math { font-family: 'Computer Modern', serif; }
        pre { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
        """
    ) as app:
        gr.Markdown("""
        # 🎓 AI 전기공학 튜터 (향상된 버전)
        
        ChatGPT 스타일의 전문적인 전기공학 학습 도우미입니다.
        - ✅ **구조화된 답변**: 핵심 정리, 단계별 설명, 시각적 요소
        - 📊 **수식 지원**: LaTeX 수식 인식 및 표현
        - 🖼️ **이미지 분석**: 문제 사진을 업로드하면 OCR로 분석
        - 💡 **전문가 수준**: 전기공학 특화 지식 기반
        """)
        
        # 메인 인터페이스
        with gr.Row():
            with gr.Column(scale=7):
                # 채팅 인터페이스
                chatbot = gr.Chatbot(
                    label="대화",
                    height=600,
                    bubble_full_width=False,
                    show_label=True,
                    elem_classes=["message"]
                )
                
                # 입력 영역
                with gr.Row():
                    with gr.Column(scale=4):
                        msg = gr.Textbox(
                            label="질문을 입력하세요",
                            placeholder="전기공학 관련 질문을 입력하세요...",
                            lines=3,
                            max_lines=5
                        )
                    with gr.Column(scale=1):
                        submit = gr.Button("전송", variant="primary", size="lg")
                        clear = gr.Button("대화 초기화", size="sm")
                
                # 이미지 업로드
                with gr.Row():
                    image_input = gr.Image(
                        label="문제 이미지 업로드 (선택사항)",
                        type="pil",
                        height=200
                    )
                    image_preview = gr.Markdown("이미지를 업로드하면 OCR로 텍스트와 수식을 추출합니다.")
            
            # 사이드바
            with gr.Column(scale=3):
                # 응답 스타일 선택
                with gr.Box():
                    gr.Markdown("### ⚙️ 응답 스타일")
                    response_style = gr.Radio(
                        choices=[
                            ("종합적 설명", "comprehensive"),
                            ("단계별 풀이", "step_by_step"),
                            ("개념 설명", "concept")
                        ],
                        value="comprehensive",
                        label="원하는 응답 형식을 선택하세요"
                    )
                
                # 예제 질문
                gr.Markdown("### 💡 예제 질문")
                with gr.Tab("기본 개념"):
                    gr.Examples(
                        examples=[
                            "3상 전력 시스템의 장점은 무엇인가요?",
                            "역률이란 무엇이고 왜 중요한가요?",
                            "변압기의 동작 원리를 설명해주세요."
                        ],
                        inputs=msg,
                        label="클릭하여 사용"
                    )
                
                with gr.Tab("문제 풀이"):
                    gr.Examples(
                        examples=[
                            "3상 전력에서 선간전압이 380V이고 부하전류가 10A일 때 전력을 계산하세요.",
                            "RLC 직렬회로에서 공진주파수를 구하는 방법을 설명해주세요.",
                            "유도전동기의 슬립이 0.05일 때 회전속도를 구하세요."
                        ],
                        inputs=msg
                    )
                
                with gr.Tab("이미지 예시"):
                    gr.Markdown("""
                    📷 **이미지 업로드 팁:**
                    - 문제 사진을 찍어 업로드하세요
                    - 회로도나 그래프도 분석 가능합니다
                    - 손글씨도 인식됩니다 (정자로 쓸수록 정확)
                    """)
        
        # 통계 및 정보
        with gr.Accordion("📊 시스템 정보", open=False):
            with gr.Row():
                stats_display = gr.JSON(
                    label="시스템 통계",
                    visible=True
                )
                refresh_stats = gr.Button("새로고침", size="sm")
        
        # 이벤트 핸들러
        def respond(message: str, image, chat_history: List[Tuple[str, str]], style: str):
            """메시지 응답 처리"""
            if not message.strip():
                return "", None, chat_history
            
            # 스타일 설정 임시 저장
            original_style = chat_service.enhanced_rag.response_generator
            
            response = chat_service.process_query(message, chat_history, image)
            
            # 이미지가 있는 경우 대화에 표시
            if image:
                chat_history.append((f"{message}\n📎 [이미지 첨부됨]", response))
            else:
                chat_history.append((message, response))
            
            return "", None, chat_history
        
        def clear_chat():
            """대화 초기화"""
            chat_service.conversation_history.clear()
            return None, "", None
        
        def update_stats():
            """통계 업데이트"""
            return chat_service.get_system_stats()
        
        def analyze_image(image):
            """이미지 미리보기 분석"""
            if not image:
                return "이미지를 업로드하면 OCR로 텍스트와 수식을 추출합니다."
            
            try:
                # 간단한 분석 수행
                analysis = chat_service.image_analyzer.analyze_image(image)
                if analysis['success']:
                    preview = "🔍 **이미지 분석 결과:**\n"
                    if analysis.get('ocr_text'):
                        preview += f"- 텍스트: {analysis['ocr_text'][:100]}...\n"
                    if analysis.get('formulas'):
                        preview += f"- 수식: {len(analysis['formulas'])}개 감지\n"
                    return preview
                else:
                    return "이미지 분석 중 오류가 발생했습니다."
            except:
                return "이미지 분석 기능을 사용할 수 없습니다."
        
        # 이벤트 바인딩
        submit.click(
            respond, 
            [msg, image_input, chatbot, response_style], 
            [msg, image_input, chatbot]
        )
        msg.submit(
            respond, 
            [msg, image_input, chatbot, response_style], 
            [msg, image_input, chatbot]
        )
        clear.click(clear_chat, None, [chatbot, msg, image_input])
        refresh_stats.click(update_stats, None, stats_display)
        image_input.change(analyze_image, image_input, image_preview)
        
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
    config.app.show_sources = True  # 소스 표시 활성화
    
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