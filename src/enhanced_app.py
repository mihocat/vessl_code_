#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Gradio UI Application with Advanced Image Analysis
향상된 이미지 분석 기능을 갖춘 Gradio UI 애플리케이션
"""

import sys
import time
import logging
import asyncio
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image
import torch
import numpy as np

import gradio as gr

from config import Config
from llm_client import LLMClient
from rag_system import RAGSystem, SearchResult
from services import WebSearchService, ResponseGenerator
from enhanced_image_analyzer import EnhancedImageAnalyzer
from next_gen_orchestrator import NextGenOrchestrator, SystemMode

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
        
        # 기본 컴포넌트 초기화
        self.rag_system = RAGSystem(
            rag_config=config.rag,
            dataset_config=config.dataset,
            llm_client=llm_client
        )
        self.web_search = WebSearchService(config.web_search)
        self.response_generator = ResponseGenerator(config.web_search)
        
        # 향상된 이미지 분석기 초기화
        self.enhanced_image_analyzer = None
        self.next_gen_orchestrator = None
        
        # 비동기 초기화를 위한 이벤트 루프
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            logger.info("Initializing Enhanced Image Analyzer...")
            self.enhanced_image_analyzer = EnhancedImageAnalyzer()
            loop.run_until_complete(self.enhanced_image_analyzer.initialize())
            logger.info("Enhanced Image Analyzer initialized successfully")
            
            # Next Generation Orchestrator 초기화 (선택적)
            try:
                logger.info("Initializing Next Generation Orchestrator...")
                self.next_gen_orchestrator = NextGenOrchestrator()
                loop.run_until_complete(self.next_gen_orchestrator.initialize())
                logger.info("Next Generation Orchestrator initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Next Gen Orchestrator: {e}")
                self.next_gen_orchestrator = None
                
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Image Analyzer: {e}")
            self.enhanced_image_analyzer = None
        finally:
            loop.close()
        
        # 대화 이력
        self.conversation_history = []
        
    def process_query(
        self, 
        question: str, 
        image: Optional[Image.Image] = None,
        use_web_search: bool = True,
        processing_mode: str = "standard"
    ) -> Tuple[str, List[SearchResult], List[str], float, str]:
        """
        향상된 쿼리 처리
        
        Args:
            question: 사용자 질문
            image: 이미지 (선택)
            use_web_search: 웹 검색 사용 여부
            processing_mode: 처리 모드 (standard, enhanced, next_gen)
            
        Returns:
            (답변, RAG 검색 결과, 웹 검색 결과, 소요 시간, 이미지 분석 결과)
        """
        start_time = time.time()
        
        # 이미지 분석 (있는 경우)
        image_analysis = ""
        if image and self.enhanced_image_analyzer:
            try:
                logger.info("Processing image with Enhanced Image Analyzer...")
                
                # 비동기 함수를 동기적으로 실행
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    analysis_result = loop.run_until_complete(
                        self.enhanced_image_analyzer.analyze_image(image)
                    )
                    
                    # 분석 결과 포맷팅
                    image_analysis = self._format_image_analysis(analysis_result)
                    
                    # 수식이 감지된 경우 질문에 포함
                    if analysis_result.get('formulas'):
                        formulas_text = self._format_formulas(analysis_result['formulas'])
                        question = f"{question}\n\n이미지에서 감지된 수식:\n{formulas_text}"
                    
                    # 전기공학 컨텍스트 추가
                    if analysis_result.get('electrical_context'):
                        question = f"{question}\n\n컨텍스트: {analysis_result['electrical_context']}"
                        
                except Exception as e:
                    logger.error(f"Error in image analysis: {e}")
                    image_analysis = f"이미지 분석 중 오류 발생: {str(e)}"
                finally:
                    loop.close()
        
        # Next Gen Orchestrator 사용 (가능한 경우)
        if processing_mode == "next_gen" and self.next_gen_orchestrator:
            try:
                logger.info("Using Next Generation Orchestrator...")
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    orchestrator_input = {
                        'query': question,
                        'image': image,
                        'mode': SystemMode.UNIFIED,
                        'context': {
                            'conversation_history': self.conversation_history,
                            'use_web_search': use_web_search
                        }
                    }
                    
                    result = loop.run_until_complete(
                        self.next_gen_orchestrator.process(orchestrator_input)
                    )
                    
                    answer = result.get('response', '')
                    rag_results = result.get('rag_results', [])
                    web_results = result.get('web_results', [])
                    
                    elapsed_time = time.time() - start_time
                    
                    return answer, rag_results, web_results, elapsed_time, image_analysis
                    
                except Exception as e:
                    logger.error(f"Error in Next Gen Orchestrator: {e}")
                    # Fallback to standard processing
                finally:
                    loop.close()
        
        # 표준 처리 (RAG + 웹 검색)
        rag_results = self.rag_system.search_with_rerank(question, top_k=5)
        
        # 웹 검색
        web_results = []
        if use_web_search:
            web_results = self.web_search.search(question, max_results=3)
        
        # LLM 응답 생성
        response_context = self._prepare_context(rag_results, web_results, image_analysis)
        answer = self.llm_client.generate_response(question, response_context)
        
        # 대화 이력 업데이트
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'has_image': image is not None
        })
        
        elapsed_time = time.time() - start_time
        
        return answer, rag_results, web_results, elapsed_time, image_analysis
    
    def _format_image_analysis(self, analysis: Dict[str, Any]) -> str:
        """이미지 분석 결과 포맷팅"""
        lines = []
        
        if analysis.get('caption'):
            lines.append(f"이미지 설명: {analysis['caption']}")
        
        if analysis.get('regions'):
            regions = analysis['regions']
            lines.append(f"\n검출된 영역:")
            lines.append(f"- 텍스트: {regions.get('text', 0)}개")
            lines.append(f"- 수식: {regions.get('formula', 0)}개")
            lines.append(f"- 회로도: {regions.get('circuit', 0)}개")
            lines.append(f"- 다이어그램: {regions.get('diagram', 0)}개")
        
        if analysis.get('formulas'):
            lines.append(f"\n감지된 수식: {len(analysis['formulas'])}개")
            for i, formula in enumerate(analysis['formulas'][:3]):  # 최대 3개
                lines.append(f"  {i+1}. {formula.get('raw', '')}")
                if formula.get('type'):
                    lines.append(f"     유형: {formula['type']}")
        
        if analysis.get('electrical_context'):
            lines.append(f"\n전기공학 컨텍스트: {analysis['electrical_context']}")
        
        return '\n'.join(lines)
    
    def _format_formulas(self, formulas: List[Dict]) -> str:
        """수식 포맷팅"""
        formatted = []
        for formula in formulas:
            if formula.get('latex'):
                formatted.append(formula['latex'])
            else:
                formatted.append(formula.get('raw', ''))
        return '\n'.join(formatted)
    
    def _prepare_context(
        self, 
        rag_results: List[SearchResult], 
        web_results: List[Dict],
        image_analysis: str
    ) -> str:
        """LLM을 위한 컨텍스트 준비"""
        context_parts = []
        
        # RAG 결과
        if rag_results:
            context_parts.append("참고 자료:")
            for result in rag_results[:3]:
                context_parts.append(f"- Q: {result.question[:100]}...")
                context_parts.append(f"  A: {result.answer[:200]}...")
        
        # 웹 검색 결과
        if web_results:
            context_parts.append("\n웹 검색 결과:")
            for result in web_results:
                context_parts.append(f"- {result.get('title', '')}")
                context_parts.append(f"  {result.get('snippet', '')[:150]}...")
        
        # 이미지 분석 결과
        if image_analysis:
            context_parts.append(f"\n이미지 분석:\n{image_analysis}")
        
        return '\n'.join(context_parts)


def create_gradio_interface():
    """향상된 Gradio 인터페이스 생성"""
    
    # 설정 로드
    config = Config()
    
    # LLM 클라이언트 초기화
    llm_client = LLMClient(config.llm)
    
    # 향상된 챗 서비스 초기화
    chat_service = EnhancedChatService(config, llm_client)
    
    def chat_function(
        message: str, 
        image: Optional[Image.Image],
        use_web_search: bool,
        processing_mode: str,
        history: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], str, str, str]:
        """채팅 처리 함수"""
        
        if not message.strip():
            return history, "", "", ""
        
        # 쿼리 처리
        answer, rag_results, web_results, elapsed_time, image_analysis = chat_service.process_query(
            message, 
            image,
            use_web_search,
            processing_mode
        )
        
        # 이력 업데이트
        history.append((message, answer))
        
        # 추가 정보 포맷팅
        rag_info = _format_rag_results(rag_results)
        web_info = _format_web_results(web_results)
        stats = f"처리 시간: {elapsed_time:.2f}초\n처리 모드: {processing_mode}"
        
        if image_analysis:
            stats = f"{stats}\n\n--- 이미지 분석 결과 ---\n{image_analysis}"
        
        return history, rag_info, web_info, stats
    
    def _format_rag_results(results: List[SearchResult]) -> str:
        """RAG 결과 포맷팅"""
        if not results:
            return "관련 자료를 찾을 수 없습니다."
        
        lines = []
        for i, result in enumerate(results[:3], 1):
            lines.append(f"{i}. {result.question[:100]}...")
            lines.append(f"   신뢰도: {result.confidence:.3f}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def _format_web_results(results: List[Dict]) -> str:
        """웹 검색 결과 포맷팅"""
        if not results:
            return "웹 검색 결과가 없습니다."
        
        lines = []
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.get('title', 'No title')}")
            lines.append(f"   {result.get('url', '')}")
            lines.append("")
        
        return '\n'.join(lines)
    
    # Gradio 인터페이스 구성
    with gr.Blocks(title="Enhanced Electrical Engineering AI Assistant") as demo:
        gr.Markdown(
            """
            # ⚡ Enhanced Electrical Engineering AI Assistant
            
            향상된 이미지 분석 기능을 갖춘 전기공학 AI 어시스턴트입니다.
            - 수식 인식 특화 OCR
            - 다단계 이미지 분석 파이프라인
            - 전기공학 도메인 특화 처리
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="대화",
                    height=500,
                    show_label=True,
                    show_copy_button=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="질문을 입력하세요",
                        placeholder="전기공학 관련 질문을 입력하세요...",
                        lines=2,
                        scale=4
                    )
                    submit = gr.Button("전송", variant="primary", scale=1)
                
                with gr.Row():
                    image_input = gr.Image(
                        label="이미지 업로드 (선택)",
                        type="pil",
                        height=200
                    )
                    
                    with gr.Column():
                        use_web = gr.Checkbox(
                            label="웹 검색 사용",
                            value=True
                        )
                        processing_mode = gr.Radio(
                            choices=["standard", "enhanced", "next_gen"],
                            value="enhanced",
                            label="처리 모드"
                        )
                
                clear = gr.Button("대화 초기화")
                
            with gr.Column(scale=1):
                rag_output = gr.Textbox(
                    label="RAG 검색 결과",
                    lines=10,
                    max_lines=15
                )
                
                web_output = gr.Textbox(
                    label="웹 검색 결과",
                    lines=10,
                    max_lines=15
                )
                
                stats_output = gr.Textbox(
                    label="처리 정보",
                    lines=10,
                    max_lines=20
                )
        
        # 이벤트 핸들러
        def submit_message(message, image, use_web, mode, history):
            return chat_function(message, image, use_web, mode, history)
        
        msg.submit(
            submit_message,
            inputs=[msg, image_input, use_web, processing_mode, chatbot],
            outputs=[chatbot, rag_output, web_output, stats_output]
        ).then(
            lambda: ("", None),
            outputs=[msg, image_input]
        )
        
        submit.click(
            submit_message,
            inputs=[msg, image_input, use_web, processing_mode, chatbot],
            outputs=[chatbot, rag_output, web_output, stats_output]
        ).then(
            lambda: ("", None),
            outputs=[msg, image_input]
        )
        
        clear.click(lambda: [], outputs=[chatbot])
        
        # 예제
        gr.Examples(
            examples=[
                ["옴의 법칙에 대해 설명해주세요", None, True, "standard"],
                ["이 회로의 전체 저항을 계산해주세요", None, True, "enhanced"],
                ["변압기의 동작 원리를 설명해주세요", None, True, "next_gen"]
            ],
            inputs=[msg, image_input, use_web, processing_mode]
        )
    
    return demo


def main():
    """메인 함수"""
    try:
        # Gradio 인터페이스 생성 및 실행
        demo = create_gradio_interface()
        
        # 서버 실행
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()