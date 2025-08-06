#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 파이프라인 Gradio 앱 v3
OpenAI 분석(1회) → RAG → 파인튜닝 LLM 파이프라인
"""

import sys
import time
import logging
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import gradio as gr

from config import Config
from integrated_pipeline import IntegratedPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedChatService:
    """통합 파이프라인 채팅 서비스"""
    
    def __init__(self, config: Config):
        """서비스 초기화"""
        self.config = config
        self.pipeline = IntegratedPipeline(config)
        
        # 대화 이력
        self.conversation_history = []
        
        logger.info("IntegratedChatService initialized")
    
    def process_query(
        self, 
        question: str, 
        history: List[Tuple[str, str]],
        image: Optional[Image.Image] = None
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        질의 처리
        
        Args:
            question: 사용자 질문
            history: 대화 이력
            image: 선택적 이미지
            
        Returns:
            (응답 메시지, 업데이트된 대화 이력)
        """
        if not question.strip():
            return "질문을 입력해주세요.", history
        
        # 파이프라인 처리
        result = self.pipeline.process_query(
            question=question,
            image=image,
            use_rag=True,
            use_llm=True
        )
        
        if result.success:
            # 성공적인 처리
            response = self._format_response(result, question, image)
        else:
            # 실패 처리
            response = f"처리 중 오류가 발생했습니다: {result.error_message}"
        
        # 대화 이력 업데이트
        history.append((question, response))
        self.conversation_history.append({
            'question': question,
            'response': response,
            'has_image': image is not None,
            'timestamp': time.time(),
            'pipeline_result': result
        })
        
        return response, history
    
    def _format_response(self, result, question: str, image: Optional[Image.Image]) -> str:
        """응답 포맷팅"""
        response_parts = []
        
        # 메인 답변
        response_parts.append(result.final_answer)
        
        # 처리 정보 추가
        if result.analysis_result or result.processing_times:
            response_parts.append("\n" + "="*50)
            response_parts.append("📊 **처리 정보**")
            
            # OpenAI 분석 결과
            if result.analysis_result:
                analysis = result.analysis_result
                if analysis.get('formulas'):
                    response_parts.append(f"📐 감지된 수식: {len(analysis['formulas'])}개")
                if analysis.get('key_concepts'):
                    response_parts.append(f"🔑 핵심 개념: {', '.join(analysis['key_concepts'][:3])}")
                if analysis.get('token_usage'):
                    tokens = analysis['token_usage']
                    response_parts.append(f"🪙 토큰 사용: {tokens['total_tokens']}개 (입력: {tokens['prompt_tokens']}, 출력: {tokens['completion_tokens']})")
                if analysis.get('cost'):
                    response_parts.append(f"💰 OpenAI 비용: ${analysis['cost']:.4f}")
            
            # RAG 검색 결과
            if result.rag_results:
                response_parts.append(f"📚 RAG 검색: {len(result.rag_results)}개 문서 활용")
            
            # 처리 시간
            if result.processing_times:
                times = result.processing_times
                response_parts.append(f"⏱️ 처리 시간: {times.get('total', 0):.2f}초")
                if 'openai_analysis' in times:
                    response_parts.append(f"  - OpenAI 분석: {times['openai_analysis']:.2f}초")
                if 'rag_search' in times:
                    response_parts.append(f"  - RAG 검색: {times['rag_search']:.2f}초")
                if 'llm_generation' in times:
                    response_parts.append(f"  - LLM 생성: {times['llm_generation']:.2f}초")
            
            # 파이프라인 단계
            if result.pipeline_steps:
                response_parts.append(f"🔄 파이프라인: {' → '.join(result.pipeline_steps)}")
        
        return "\n".join(response_parts)
    
    def get_system_status(self) -> str:
        """시스템 상태 반환"""
        try:
            # 파이프라인 상태 확인
            health_status = self.pipeline.health_check()
            stats = self.pipeline.get_statistics()
            
            status_parts = [
                "🏥 **시스템 상태**",
                f"OpenAI 프로세서: {'✅' if health_status['openai_processor'] else '❌'}",
                f"RAG 시스템: {'✅' if health_status['rag_system'] else '❌'}",
                f"파인튜닝 LLM: {'✅' if health_status['llm_client'] else '❌'}",
                "",
                "📈 **처리 통계**",
                f"총 질의 수: {stats['total_queries']}",
                f"성공률: {stats['success_rate']:.1%}",
                f"평균 비용: ${stats['average_cost_per_query']:.4f}",
                f"OpenAI 호출 효율: {stats['openai_call_efficiency']}",
                f"총 비용: ${stats['total_cost']:.4f}"
            ]
            
            return "\n".join(status_parts)
            
        except Exception as e:
            return f"상태 확인 중 오류 발생: {e}"


def create_gradio_interface(service: IntegratedChatService) -> gr.Interface:
    """Gradio 인터페이스 생성"""
    
    def chat_function(question: str, history: List[Tuple[str, str]], image: Optional[Image.Image]):
        """채팅 함수"""
        return service.process_query(question, history, image)
    
    def status_function():
        """상태 확인 함수"""
        return service.get_system_status()
    
    # 채팅 인터페이스
    with gr.Blocks(
        title="🚀 통합 AI 파이프라인",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto;
        }
        .chat-container {
            height: 600px;
        }
        """
    ) as iface:
        gr.Markdown("""
        # 🚀 통합 AI 파이프라인 채팅봇
        
        **새로운 아키텍처:**
        1. 🔍 **OpenAI GPT-4.1** - 이미지+텍스트 분석 (1회 호출)
        2. 📚 **RAG 검색** - ChromaDB 전문 문서 활용
        3. 🤖 **파인튜닝 LLM** - KoLlama 한국어 전문 모델
        
        **특징:**
        - OpenAI API는 분석 전용 (최종 답변 금지)
        - 질의당 1회만 OpenAI 호출
        - 최종 답변은 파인튜닝된 한국어 LLM만 담당
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # 채팅 인터페이스
                chatbot = gr.Chatbot(
                    label="대화",
                    height=600,
                    show_label=True,
                    container=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        label="질문",
                        placeholder="질문을 입력하세요...",
                        lines=2,
                        scale=4
                    )
                    image_input = gr.Image(
                        label="이미지 (선택사항)",
                        type="pil",
                        scale=1
                    )
                
                with gr.Row():
                    submit_btn = gr.Button("전송", variant="primary", scale=1)
                    clear_btn = gr.Button("대화 초기화", scale=1)
            
            with gr.Column(scale=1):
                # 시스템 상태 패널
                status_output = gr.Textbox(
                    label="시스템 상태",
                    lines=20,
                    interactive=False
                )
                status_btn = gr.Button("상태 새로고침")
                
                # 예제 질문들
                gr.Markdown("### 💡 예제 질문")
                example_questions = [
                    "Pr'을 구할 때 왜 Pr에서 Qc를 빼는 건가요?",
                    "미분할 때 d/da를 적용하면 왜 s는 1이고 a는 0이 되는 건가요?",
                    "두 점 사이의 거리 단위벡터를 구하는 방식을 알려주세요",
                    "라플라스 변환의 정의와 성질을 설명해주세요"
                ]
                
                for i, example in enumerate(example_questions, 1):
                    gr.Button(f"{i}. {example[:30]}...", size="sm").click(
                        lambda x=example: (x, []),
                        outputs=[question_input, chatbot]
                    )
        
        # 이벤트 핸들러
        submit_btn.click(
            chat_function,
            inputs=[question_input, chatbot, image_input],
            outputs=[question_input, chatbot]
        ).then(
            lambda: ("", None),
            outputs=[question_input, image_input]
        )
        
        question_input.submit(
            chat_function,
            inputs=[question_input, chatbot, image_input],
            outputs=[question_input, chatbot]
        ).then(
            lambda: ("", None),
            outputs=[question_input, image_input]
        )
        
        clear_btn.click(
            lambda: ([], "", None),
            outputs=[chatbot, question_input, image_input]
        )
        
        status_btn.click(
            status_function,
            outputs=status_output
        )
        
        # 초기 상태 로드
        iface.load(
            status_function,
            outputs=status_output
        )
    
    return iface


def main():
    """메인 함수"""
    try:
        # 설정 로드
        config = Config()
        logger.info("Configuration loaded")
        
        # 서비스 초기화
        service = IntegratedChatService(config)
        logger.info("IntegratedChatService initialized")
        
        # Gradio 인터페이스 생성
        iface = create_gradio_interface(service)
        
        # 서버 시작
        logger.info(f"Starting server on {config.app.server_name}:{config.app.server_port}")
        iface.launch(
            server_name=config.app.server_name,
            server_port=config.app.server_port,
            share=config.app.share,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()