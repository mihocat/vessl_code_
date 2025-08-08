#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Next-Generation Multimodal AI Chatbot Application
차세대 멀티모달 AI 챗봇 애플리케이션
"""

import logging
import time
import asyncio
import uuid
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image

import gradio as gr

from config import Config
from advanced_chatbot_service import AdvancedChatbotService, ConversationMode
from intent_analyzer import IntentAnalyzer
from multi_agent_system import MultiAgentSystem

logger = logging.getLogger(__name__)


class NextGenChatInterface:
    """차세대 채팅 인터페이스"""
    
    def __init__(self, config: Config):
        self.config = config
        self.chatbot_service = AdvancedChatbotService(config)
        self.active_sessions = {}
        
        logger.info("Next-Generation Chat Interface initialized")
    
    def create_gradio_interface(self) -> gr.Blocks:
        """Gradio 인터페이스 생성"""
        
        # 테마 설정
        theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate"
        )
        
        with gr.Blocks(
            title="🚀 Next-Gen AI Chatbot", 
            theme=theme,
            css=self._get_custom_css()
        ) as interface:
            
            # 세션 상태
            session_state = gr.State(self._create_new_session)
            
            # 헤더
            gr.HTML("""
                <div style="text-align: center; margin-bottom: 30px;">
                    <h1>🚀 Next-Generation AI Chatbot</h1>
                    <p>Advanced Multi-Agent System with Vision, RAG, and Reasoning</p>
                </div>
            """)
            
            with gr.Row():
                # 메인 채팅 영역
                with gr.Column(scale=3):
                    # 모드 선택
                    with gr.Row():
                        mode_dropdown = gr.Dropdown(
                            choices=[mode.value for mode in ConversationMode],
                            value=ConversationMode.STANDARD.value,
                            label="🎛️ 대화 모드",
                            info="원하는 대화 스타일을 선택하세요"
                        )
                        difficulty_slider = gr.Slider(
                            minimum=1, maximum=5, value=3, step=1,
                            label="📊 난이도 수준",
                            info="1: 초급, 3: 중급, 5: 고급"
                        )
                    
                    # 채팅봇
                    chatbot = gr.Chatbot(
                        label="💬 대화",
                        height=500,
                        bubble_full_width=False,
                        show_label=True,
                        avatar_images=("🧑‍💻", "🤖")
                    )
                    
                    # 입력 영역
                    with gr.Row():
                        with gr.Column(scale=4):
                            msg_input = gr.Textbox(
                                label="📝 질문을 입력하세요",
                                placeholder="무엇을 도와드릴까요? (이미지와 함께 질문도 가능합니다)",
                                lines=2,
                                max_lines=5
                            )
                            
                            # 이미지 업로드
                            image_input = gr.Image(
                                label="🖼️ 이미지 업로드 (선택사항)",
                                type="pil",
                                height=200
                            )
                        
                        with gr.Column(scale=1):
                            send_btn = gr.Button("🚀 전송", variant="primary", size="lg")
                            clear_btn = gr.Button("🗑️ 초기화", variant="secondary")
                    
                    # 빠른 시작 버튼들
                    with gr.Row():
                        quick_vision = gr.Button("👁️ 이미지 분석", size="sm")
                        quick_math = gr.Button("🧮 수학 문제", size="sm")
                        quick_explain = gr.Button("📚 개념 설명", size="sm")
                        quick_research = gr.Button("🔬 연구 모드", size="sm")
                
                # 사이드바
                with gr.Column(scale=1):
                    # 실시간 상태
                    with gr.Accordion("📊 실시간 상태", open=True):
                        status_display = gr.HTML("""
                            <div id="status">
                                <p>🟢 시스템 준비됨</p>
                                <p>🤖 에이전트: 4개 활성화</p>
                                <p>🧠 메모리: 대기 중</p>
                            </div>
                        """)
                    
                    # 대화 통계
                    with gr.Accordion("📈 세션 통계", open=False):
                        stats_display = gr.JSON(
                            label="통계",
                            value={}
                        )
                    
                    # 제안 질문
                    with gr.Accordion("💡 제안 질문", open=True):
                        suggestions_display = gr.HTML("""
                            <div style="font-size: 14px;">
                                <p><b>시작해보세요:</b></p>
                                <ul>
                                    <li>이미지의 수식을 분석해주세요</li>
                                    <li>데이터 처리 방법을 설명해주세요</li>
                                    <li>머신러닝이 무엇인가요?</li>
                                    <li>이 다이어그램을 분석해주세요</li>
                                </ul>
                            </div>
                        """)
                    
                    # 기능 설명
                    with gr.Accordion("🔧 고급 기능", open=False):
                        gr.Markdown("""
                        **🎯 지능형 의도 분석**
                        - 질문 유형 자동 인식
                        - 최적 처리 경로 선택
                        
                        **🤖 멀티-에이전트 시스템**
                        - 비전 분석 전문가
                        - RAG 검색 전문가
                        - 추론 엔진
                        - 응답 합성기
                        
                        **🧠 적응형 메모리**
                        - 대화 컨텍스트 기억
                        - 개인화된 응답
                        - 학습 패턴 인식
                        
                        **📊 품질 보장**
                        - 응답 신뢰도 평가
                        - 다중 검증 시스템
                        - 실시간 품질 모니터링
                        """)
            
            # 이벤트 핸들러들
            def process_message(message, image, history, session, mode, difficulty):
                """메시지 처리"""
                if not message.strip():
                    return "", None, history, session, self._get_status_html("empty_message")
                
                # 상태 업데이트
                status_html = self._get_status_html("processing")
                
                try:
                    # 세션 정보 업데이트
                    session['mode'] = mode
                    session['difficulty'] = difficulty
                    
                    # 비동기 처리를 동기로 변환
                    result = asyncio.run(self._process_message_async(
                        message, image, session, mode, difficulty
                    ))
                    
                    if result['success']:
                        # 채팅 히스토리 업데이트
                        if image:
                            user_message = f"{message}\n[🖼️ 이미지 첨부됨]"
                        else:
                            user_message = message
                        
                        history.append((user_message, result['response']))
                        
                        # 상태 업데이트
                        status_html = self._get_status_html("success", result['metadata'])
                        
                        return "", None, history, session, status_html
                    else:
                        # 오류 처리
                        error_msg = f"❌ 오류: {result.get('error', '알 수 없는 오류')}"
                        history.append((message, error_msg))
                        
                        status_html = self._get_status_html("error")
                        return "", None, history, session, status_html
                        
                except Exception as e:
                    logger.error(f"Message processing failed: {e}")
                    error_msg = f"❌ 시스템 오류: {str(e)}"
                    history.append((message, error_msg))
                    
                    status_html = self._get_status_html("error")
                    return "", None, history, session, status_html
            
            def clear_conversation():
                """대화 초기화"""
                new_session = self._create_new_session()
                status_html = self._get_status_html("cleared")
                return None, "", None, new_session, status_html
            
            def update_stats(session):
                """통계 업데이트"""
                if not session or 'session_id' not in session:
                    return {}
                
                try:
                    summary = self.chatbot_service.get_conversation_summary(session['session_id'])
                    return summary
                except Exception as e:
                    logger.warning(f"Stats update failed: {e}")
                    return {"error": str(e)}
            
            def quick_action(action_type):
                """빠른 액션"""
                quick_messages = {
                    'vision': "이미지를 업로드하고 분석을 요청해보세요",
                    'math': "수학 문제나 공식에 대해 질문해보세요",
                    'explain': "설명이 필요한 개념을 입력해보세요",
                    'research': "연구 모드로 전환됩니다"
                }
                return quick_messages.get(action_type, "")
            
            # 이벤트 바인딩
            send_btn.click(
                process_message,
                inputs=[msg_input, image_input, chatbot, session_state, mode_dropdown, difficulty_slider],
                outputs=[msg_input, image_input, chatbot, session_state, status_display]
            )
            
            msg_input.submit(
                process_message,
                inputs=[msg_input, image_input, chatbot, session_state, mode_dropdown, difficulty_slider],
                outputs=[msg_input, image_input, chatbot, session_state, status_display]
            )
            
            clear_btn.click(
                clear_conversation,
                outputs=[chatbot, msg_input, image_input, session_state, status_display]
            )
            
            # 빠른 액션 버튼들
            quick_vision.click(lambda: quick_action('vision'), outputs=msg_input)
            quick_math.click(lambda: quick_action('math'), outputs=msg_input)
            quick_explain.click(lambda: quick_action('explain'), outputs=msg_input)
            quick_research.click(
                lambda: (ConversationMode.RESEARCH.value, quick_action('research')),
                outputs=[mode_dropdown, msg_input]
            )
            
            # 주기적 통계 업데이트
            interface.load(
                update_stats,
                inputs=[session_state],
                outputs=[stats_display],
                every=10  # 10초마다 업데이트
            )
        
        return interface
    
    async def _process_message_async(
        self, 
        message: str, 
        image: Optional[Image.Image], 
        session: Dict, 
        mode: str, 
        difficulty: int
    ) -> Dict[str, Any]:
        """비동기 메시지 처리"""
        try:
            # 세션 업데이트
            session['difficulty'] = difficulty
            
            # 모드 변환
            conversation_mode = ConversationMode(mode)
            
            # 고급 처리
            result = await self.chatbot_service.process_query_advanced(
                query=message,
                session_id=session['session_id'],
                image=image,
                mode=conversation_mode,
                user_preferences={'difficulty_level': difficulty}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Async message processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': f"처리 중 오류가 발생했습니다: {str(e)}"
            }
    
    def _create_new_session(self) -> Dict[str, Any]:
        """새 세션 생성"""
        session_id = str(uuid.uuid4())
        return {
            'session_id': session_id,
            'created_at': time.time(),
            'mode': ConversationMode.STANDARD.value,
            'difficulty': 3,
            'message_count': 0
        }
    
    def _get_status_html(self, status: str, metadata: Dict = None) -> str:
        """상태 HTML 생성"""
        if status == "processing":
            return """
                <div id="status">
                    <p>🟡 처리 중...</p>
                    <p>🤖 에이전트 작업 중</p>
                    <p>🧠 분석 진행 중</p>
                </div>
            """
        elif status == "success":
            if metadata:
                features = metadata.get('features_used', [])
                confidence = metadata.get('confidence', 0)
                processing_time = metadata.get('processing_time', 0)
                
                return f"""
                    <div id="status">
                        <p>🟢 완료 ({processing_time:.1f}초)</p>
                        <p>📊 신뢰도: {confidence:.1%}</p>
                        <p>🔧 기능: {len(features)}개 사용</p>
                    </div>
                """
            else:
                return """
                    <div id="status">
                        <p>🟢 처리 완료</p>
                        <p>🤖 에이전트 대기</p>
                        <p>🧠 메모리 업데이트됨</p>
                    </div>
                """
        elif status == "error":
            return """
                <div id="status">
                    <p>🔴 오류 발생</p>
                    <p>🤖 에이전트 복구 중</p>
                    <p>🧠 다시 시도하세요</p>
                </div>
            """
        elif status == "cleared":
            return """
                <div id="status">
                    <p>🆕 새 대화 시작</p>
                    <p>🤖 에이전트 준비됨</p>
                    <p>🧠 메모리 초기화됨</p>
                </div>
            """
        elif status == "empty_message":
            return """
                <div id="status">
                    <p>⚠️ 메시지 입력 필요</p>
                    <p>🤖 에이전트 대기 중</p>
                    <p>🧠 메모리 대기 중</p>
                </div>
            """
        else:
            return """
                <div id="status">
                    <p>🟢 시스템 준비됨</p>
                    <p>🤖 에이전트: 4개 활성화</p>
                    <p>🧠 메모리: 대기 중</p>
                </div>
            """
    
    def _get_custom_css(self) -> str:
        """커스텀 CSS"""
        return """
        #status {
            background: linear-gradient(145deg, #f0f9ff, #e0f2fe);
            border-radius: 8px;
            padding: 12px;
            border-left: 4px solid #0ea5e9;
        }
        
        #status p {
            margin: 4px 0;
            font-size: 14px;
            color: #0f172a;
        }
        
        .gradio-container {
            max-width: 1400px !important;
        }
        
        .chat-message {
            border-radius: 12px;
            padding: 12px;
            margin: 8px 0;
        }
        
        .user-message {
            background: linear-gradient(145deg, #dbeafe, #bfdbfe);
            margin-left: 20%;
        }
        
        .bot-message {
            background: linear-gradient(145deg, #f0fdf4, #dcfce7);
            margin-right: 20%;
        }
        """


def create_next_gen_app(config: Optional[Config] = None) -> gr.Blocks:
    """차세대 앱 생성"""
    if config is None:
        config = Config()
    
    interface = NextGenChatInterface(config)
    return interface.create_gradio_interface()


def launch_next_gen_app():
    """차세대 앱 실행"""
    # 설정 로드
    config = Config()
    
    logger.info("Launching Next-Generation AI Chatbot...")
    
    try:
        # 앱 인터페이스 생성
        app = create_next_gen_app(config)
        
        # 실행
        app.launch(
            server_name=config.app.server_name,
            server_port=config.app.server_port,
            share=config.app.share,
            show_error=True,
            quiet=False,
            inbrowser=True,
            favicon_path=None,
            auth=None
        )
        
    except Exception as e:
        logger.error(f"Failed to launch Next-Gen app: {e}")
        raise


if __name__ == "__main__":
    launch_next_gen_app()