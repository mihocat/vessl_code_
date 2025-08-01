#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced UI with Real-time Feedback and Visualizations
실시간 피드백과 시각화를 갖춘 고급 UI
"""

import sys
import time
import logging
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image
import torch
import numpy as np
import json
import re

import gradio as gr

from config import Config
from llm_client import LLMClient

# 향상된 시스템
from enhanced_rag_system import EnhancedVectorDatabase, EnhancedRAGSystem
from enhanced_image_analyzer import ChatGPTStyleAnalyzer
from chatgpt_response_generator import ChatGPTResponseGenerator
from visualization_components import VisualizationManager

# 유틸리티
from services import WebSearchService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeFeedback:
    """실시간 피드백 시스템"""
    
    def __init__(self):
        """실시간 피드백 초기화"""
        self.current_status = "대기 중..."
        self.processing_steps = []
        self.confidence_score = 0.0
    
    def update_status(self, status: str, step: Optional[str] = None):
        """상태 업데이트"""
        self.current_status = status
        if step:
            self.processing_steps.append({
                'time': time.strftime('%H:%M:%S'),
                'step': step,
                'status': status
            })
    
    def get_status_html(self) -> str:
        """상태를 HTML로 반환"""
        html = f"<div style='padding: 10px; background: #f0f0f0; border-radius: 5px;'>"
        html += f"<b>현재 상태:</b> {self.current_status}<br>"
        
        if self.processing_steps:
            html += "<b>처리 단계:</b><br>"
            for step in self.processing_steps[-5:]:  # 최근 5개만
                html += f"• [{step['time']}] {step['step']}<br>"
        
        if self.confidence_score > 0:
            color = 'green' if self.confidence_score > 0.7 else 'orange' if self.confidence_score > 0.4 else 'red'
            html += f"<b>신뢰도:</b> <span style='color: {color}'>{self.confidence_score:.1%}</span>"
        
        html += "</div>"
        return html
    
    def reset(self):
        """피드백 초기화"""
        self.current_status = "대기 중..."
        self.processing_steps = []
        self.confidence_score = 0.0


class InteractiveComponents:
    """인터랙티브 컴포넌트"""
    
    @staticmethod
    def create_formula_input() -> gr.Textbox:
        """수식 입력 컴포넌트"""
        return gr.Textbox(
            label="LaTeX 수식 입력",
            placeholder="예: P = \\sqrt{3} \\times V_L \\times I_L \\times \\cos\\theta",
            lines=1
        )
    
    @staticmethod
    def create_circuit_builder() -> Dict[str, gr.Component]:
        """회로 빌더 컴포넌트"""
        components = {
            'component_type': gr.Dropdown(
                choices=['resistor', 'capacitor', 'inductor', 'voltage_source', 'ground'],
                label="컴포넌트 타입",
                value='resistor'
            ),
            'component_label': gr.Textbox(
                label="컴포넌트 라벨",
                placeholder="예: R1, C1, L1",
                value="R1"
            ),
            'add_component': gr.Button("컴포넌트 추가", size="sm"),
            'circuit_json': gr.Textbox(
                label="회로 구성 (JSON)",
                lines=5,
                value='{"components": [], "connections": []}'
            )
        }
        return components
    
    @staticmethod
    def create_graph_controls() -> Dict[str, gr.Component]:
        """그래프 컨트롤"""
        return {
            'graph_type': gr.Dropdown(
                choices=['sine', 'square', 'triangle', 'custom'],
                label="파형 타입",
                value='sine'
            ),
            'frequency': gr.Slider(
                minimum=0.1, maximum=10, value=1,
                label="주파수 (Hz)"
            ),
            'amplitude': gr.Slider(
                minimum=0.1, maximum=10, value=1,
                label="진폭"
            ),
            'phase': gr.Slider(
                minimum=0, maximum=360, value=0,
                label="위상 (도)"
            )
        }


class AdvancedChatService:
    """고급 챗봇 서비스"""
    
    def __init__(self, config: Config, llm_client: LLMClient):
        """고급 챗봇 서비스 초기화"""
        self.config = config
        self.llm_client = llm_client
        
        # 컴포넌트 초기화
        self.vector_db = EnhancedVectorDatabase(
            persist_directory=config.rag.persist_directory
        )
        self.enhanced_rag = EnhancedRAGSystem(
            vector_db=self.vector_db,
            llm_client=llm_client
        )
        self.image_analyzer = ChatGPTStyleAnalyzer(use_florence=True)
        self.response_generator = ChatGPTResponseGenerator()
        self.viz_manager = VisualizationManager()
        self.feedback = RealTimeFeedback()
        
        # 대화 이력
        self.conversation_history = []
        
        logger.info("Advanced chat service initialized")
    
    def process_with_feedback(
        self,
        question: str,
        image: Optional[Image.Image] = None,
        formula: Optional[str] = None,
        circuit_data: Optional[str] = None,
        response_style: str = 'comprehensive'
    ) -> Tuple[str, str, Optional[Image.Image]]:
        """
        피드백과 함께 쿼리 처리
        
        Returns:
            (응답 텍스트, 상태 HTML, 시각화 이미지)
        """
        self.feedback.reset()
        start_time = time.time()
        visualization = None
        
        try:
            # 1. 입력 검증
            self.feedback.update_status("입력 검증 중...", "입력 검증")
            if not question.strip():
                return "질문을 입력해주세요.", self.feedback.get_status_html(), None
            
            # 2. 이미지 분석
            if image:
                self.feedback.update_status("이미지 분석 중...", "이미지 OCR")
                image_analysis = self.image_analyzer.analyze_for_chatgpt_response(image)
                self.feedback.confidence_score = 0.8 if image_analysis['success'] else 0.3
            else:
                image_analysis = None
            
            # 3. 수식 처리
            if formula:
                self.feedback.update_status("수식 렌더링 중...", "수식 처리")
                formula_img = self.viz_manager.create_visualization('formula', {
                    'formula': formula
                })
                if formula_img:
                    visualization = formula_img
            
            # 4. 회로도 생성
            if circuit_data:
                try:
                    self.feedback.update_status("회로도 생성 중...", "회로 시각화")
                    circuit_json = json.loads(circuit_data)
                    circuit_img = self.viz_manager.create_visualization('circuit', circuit_json)
                    if circuit_img:
                        visualization = circuit_img
                except:
                    logger.error("Invalid circuit data")
            
            # 5. RAG 검색
            self.feedback.update_status("관련 자료 검색 중...", "RAG 검색")
            result = self.enhanced_rag.process_query(
                query=question,
                image=image,
                response_style=response_style
            )
            
            # 6. 응답 생성
            self.feedback.update_status("응답 생성 중...", "LLM 처리")
            
            if result['success']:
                response = result['response']
                
                # 신뢰도 점수 업데이트
                if result.get('search_results'):
                    top_score = result['search_results'][0].get('hybrid_score', 0)
                    self.feedback.confidence_score = top_score
                
                # 시각화 추천
                viz_recommendation = self._recommend_visualization(question, result)
                if viz_recommendation and not visualization:
                    viz_data = self._prepare_visualization_data(viz_recommendation, result)
                    if viz_data:
                        visualization = self.viz_manager.create_visualization(
                            viz_recommendation['type'],
                            viz_data
                        )
                
                # 응답 시간 추가
                elapsed_time = time.time() - start_time
                response += f"\n\n_처리 시간: {elapsed_time:.2f}초_"
                
                self.feedback.update_status("완료", "응답 생성 완료")
                
                return response, self.feedback.get_status_html(), visualization
            else:
                self.feedback.update_status("오류 발생", "처리 실패")
                return "죄송합니다. 응답을 생성할 수 없습니다.", self.feedback.get_status_html(), None
                
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            self.feedback.update_status("오류", str(e))
            return "처리 중 오류가 발생했습니다.", self.feedback.get_status_html(), None
    
    def _recommend_visualization(self, question: str, result: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """시각화 추천"""
        question_lower = question.lower()
        
        # 회로 관련
        if any(term in question_lower for term in ['회로', '저항', 'rlc', 'rc', 'circuit']):
            return {'type': 'circuit', 'reason': '회로 분석'}
        
        # 그래프 관련
        if any(term in question_lower for term in ['그래프', '파형', 'wave', 'plot']):
            return {'type': 'graph', 'reason': '파형 분석'}
        
        # 보드 선도
        if any(term in question_lower for term in ['보드', 'bode', '주파수 응답']):
            return {'type': 'bode', 'reason': '주파수 응답'}
        
        # 페이저
        if any(term in question_lower for term in ['페이저', 'phasor', '위상']):
            return {'type': 'phasor', 'reason': '페이저 분석'}
        
        return None
    
    def _prepare_visualization_data(self, recommendation: Dict, result: Dict) -> Optional[Dict]:
        """시각화 데이터 준비"""
        viz_type = recommendation['type']
        
        if viz_type == 'circuit':
            # 기본 RLC 회로
            return {
                'components': [
                    {'type': 'voltage_source', 'label': 'V', 'position': (2, 4)},
                    {'type': 'resistor', 'label': 'R', 'position': (4, 4)},
                    {'type': 'inductor', 'label': 'L', 'position': (6, 4)},
                    {'type': 'capacitor', 'label': 'C', 'position': (8, 4)}
                ],
                'connections': [
                    {'from': (2.5, 4), 'to': (3.5, 4)},
                    {'from': (4.5, 4), 'to': (5.5, 4)},
                    {'from': (6.5, 4), 'to': (7.5, 4)},
                    {'from': (8.5, 4), 'to': (8.5, 2)},
                    {'from': (8.5, 2), 'to': (1.5, 2)},
                    {'from': (1.5, 2), 'to': (1.5, 4)}
                ]
            }
        
        elif viz_type == 'graph':
            # 사인파
            x = np.linspace(0, 4*np.pi, 1000)
            return {
                'x': x.tolist(),
                'y': np.sin(x).tolist(),
                'title': 'Sine Wave',
                'xlabel': 'Time (s)',
                'ylabel': 'Voltage (V)'
            }
        
        return None


def create_advanced_gradio_app(config: Optional[Config] = None) -> gr.Blocks:
    """고급 Gradio 애플리케이션 생성"""
    if config is None:
        config = Config()
    
    # LLM 클라이언트 초기화
    llm_client = LLMClient(config.llm)
    
    # 서버 대기
    logger.info("Waiting for LLM server...")
    if not llm_client.wait_for_server():
        logger.error("Failed to connect to LLM server")
        raise RuntimeError("LLM server connection failed")
    
    # 고급 챗봇 서비스 초기화
    chat_service = AdvancedChatService(config, llm_client)
    
    # CSS 스타일
    custom_css = """
    .feedback-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .viz-container {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
    }
    .formula-preview {
        background: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        font-family: 'Computer Modern', serif;
    }
    """
    
    # Gradio 인터페이스
    with gr.Blocks(
        title="AI 전기공학 튜터 - 고급 버전",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as app:
        gr.Markdown("""
        # 🎓 AI 전기공학 튜터 - 고급 버전
        
        ### 주요 기능
        - 🔍 **실시간 피드백**: 처리 과정을 실시간으로 확인
        - 📊 **시각화 지원**: 회로도, 그래프, 수식 자동 생성
        - 🖼️ **향상된 이미지 분석**: OCR + 수식 인식
        - ⚡ **인터랙티브 도구**: 수식 입력, 회로 빌더
        """)
        
        with gr.Row():
            # 메인 채팅 영역
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    label="대화",
                    height=500,
                    bubble_full_width=False,
                    show_label=True
                )
                
                # 입력 영역
                with gr.Row():
                    with gr.Column(scale=5):
                        msg = gr.Textbox(
                            label="질문",
                            placeholder="전기공학 관련 질문을 입력하세요...",
                            lines=2
                        )
                    with gr.Column(scale=1):
                        submit = gr.Button("전송", variant="primary", size="lg")
                        clear = gr.Button("초기화", size="sm")
                
                # 파일 업로드
                with gr.Row():
                    image_input = gr.Image(
                        label="이미지 업로드",
                        type="pil",
                        height=150
                    )
                    visualization_output = gr.Image(
                        label="생성된 시각화",
                        height=150
                    )
            
            # 사이드바 - 도구 및 상태
            with gr.Column(scale=4):
                # 실시간 피드백
                with gr.Box():
                    gr.Markdown("### 📡 실시간 상태")
                    feedback_display = gr.HTML(
                        value="<div class='feedback-box'>대기 중...</div>",
                        label="처리 상태"
                    )
                    auto_refresh = gr.Checkbox(
                        label="자동 새로고침",
                        value=True
                    )
                
                # 인터랙티브 도구
                with gr.Tabs():
                    # 수식 도구
                    with gr.Tab("수식 도구"):
                        formula_input = InteractiveComponents.create_formula_input()
                        formula_preview = gr.HTML(
                            label="수식 미리보기"
                        )
                        render_formula = gr.Button("수식 렌더링", size="sm")
                    
                    # 회로 빌더
                    with gr.Tab("회로 빌더"):
                        circuit_components = InteractiveComponents.create_circuit_builder()
                        circuit_preview = gr.Image(
                            label="회로 미리보기",
                            height=200
                        )
                    
                    # 그래프 도구
                    with gr.Tab("그래프 도구"):
                        graph_controls = InteractiveComponents.create_graph_controls()
                        generate_graph = gr.Button("그래프 생성", size="sm")
                
                # 응답 스타일
                with gr.Box():
                    gr.Markdown("### ⚙️ 설정")
                    response_style = gr.Radio(
                        choices=[
                            ("종합적 설명", "comprehensive"),
                            ("단계별 풀이", "step_by_step"),
                            ("개념 설명", "concept")
                        ],
                        value="comprehensive",
                        label="응답 스타일"
                    )
        
        # 예제 섹션
        with gr.Accordion("💡 예제 및 템플릿", open=False):
            gr.Examples(
                examples=[
                    ["RLC 직렬회로의 공진주파수를 구하는 방법을 설명해주세요."],
                    ["3상 Y결선과 Δ결선의 차이점을 비교해주세요."],
                    ["변압기의 등가회로를 그리고 각 파라미터를 설명해주세요."],
                    ["보드 선도를 이용한 주파수 응답 분석 방법을 설명해주세요."]
                ],
                inputs=msg,
                label="예제 질문"
            )
        
        # 이벤트 핸들러
        def respond(
            message: str,
            image,
            chat_history: List[Tuple[str, str]],
            formula: str,
            circuit_json: str,
            style: str
        ):
            """메시지 응답 처리"""
            if not message.strip():
                return "", None, chat_history, feedback_display.value, None
            
            # 실시간 피드백으로 처리
            response, status_html, viz = chat_service.process_with_feedback(
                question=message,
                image=image,
                formula=formula if formula.strip() else None,
                circuit_data=circuit_json if circuit_json.strip() != '{"components": [], "connections": []}' else None,
                response_style=style
            )
            
            # 대화 이력 업데이트
            display_msg = message
            if image:
                display_msg += "\n📎 [이미지 첨부]"
            if formula:
                display_msg += f"\n📐 [수식: {formula[:30]}...]"
            
            chat_history.append((display_msg, response))
            
            return "", None, chat_history, status_html, viz
        
        def render_formula_preview(formula: str):
            """수식 미리보기"""
            if not formula:
                return ""
            
            try:
                img = chat_service.viz_manager.create_visualization('formula', {
                    'formula': formula
                })
                if img:
                    base64_img = chat_service.viz_manager.image_to_base64(img)
                    return f'<div class="formula-preview"><img src="data:image/png;base64,{base64_img}" style="max-width: 100%;"></div>'
            except:
                pass
            
            return f'<div class="formula-preview">{formula}</div>'
        
        def update_circuit_preview(circuit_json: str):
            """회로 미리보기 업데이트"""
            try:
                circuit_data = json.loads(circuit_json)
                img = chat_service.viz_manager.create_visualization('circuit', circuit_data)
                return img
            except:
                return None
        
        def generate_graph_preview(graph_type: str, freq: float, amp: float, phase: float):
            """그래프 미리보기 생성"""
            x = np.linspace(0, 4*np.pi, 1000)
            phase_rad = np.radians(phase)
            
            if graph_type == 'sine':
                y = amp * np.sin(2*np.pi*freq*x + phase_rad)
            elif graph_type == 'square':
                y = amp * np.sign(np.sin(2*np.pi*freq*x + phase_rad))
            elif graph_type == 'triangle':
                y = amp * (2/np.pi) * np.arcsin(np.sin(2*np.pi*freq*x + phase_rad))
            else:
                y = np.zeros_like(x)
            
            img = chat_service.viz_manager.create_visualization('graph', {
                'x': x.tolist(),
                'y': y.tolist(),
                'title': f'{graph_type.capitalize()} Wave',
                'xlabel': 'Time (s)',
                'ylabel': 'Amplitude'
            })
            
            return img
        
        def add_circuit_component(comp_type: str, label: str, current_json: str):
            """회로 컴포넌트 추가"""
            try:
                data = json.loads(current_json)
                # 자동 위치 결정 (간단한 그리드)
                n = len(data['components'])
                x = 2 + (n % 4) * 2
                y = 4 + (n // 4) * 2
                
                data['components'].append({
                    'type': comp_type,
                    'label': label,
                    'position': (x, y)
                })
                
                return json.dumps(data, indent=2)
            except:
                return current_json
        
        # 이벤트 바인딩
        submit.click(
            respond,
            [msg, image_input, chatbot, formula_input, 
             circuit_components['circuit_json'], response_style],
            [msg, image_input, chatbot, feedback_display, visualization_output]
        )
        
        msg.submit(
            respond,
            [msg, image_input, chatbot, formula_input,
             circuit_components['circuit_json'], response_style],
            [msg, image_input, chatbot, feedback_display, visualization_output]
        )
        
        clear.click(
            lambda: (None, "", None, 
                    "<div class='feedback-box'>대기 중...</div>", 
                    None, '{"components": [], "connections": []}'),
            None,
            [chatbot, msg, image_input, feedback_display, 
             visualization_output, circuit_components['circuit_json']]
        )
        
        # 수식 도구 이벤트
        formula_input.change(render_formula_preview, formula_input, formula_preview)
        render_formula.click(
            lambda f: chat_service.viz_manager.create_visualization('formula', {'formula': f}),
            formula_input,
            visualization_output
        )
        
        # 회로 빌더 이벤트
        circuit_components['add_component'].click(
            add_circuit_component,
            [circuit_components['component_type'], 
             circuit_components['component_label'],
             circuit_components['circuit_json']],
            circuit_components['circuit_json']
        )
        
        circuit_components['circuit_json'].change(
            update_circuit_preview,
            circuit_components['circuit_json'],
            circuit_preview
        )
        
        # 그래프 도구 이벤트
        generate_graph.click(
            generate_graph_preview,
            [graph_controls['graph_type'], graph_controls['frequency'],
             graph_controls['amplitude'], graph_controls['phase']],
            visualization_output
        )
        
        # 자동 새로고침 (피드백)
        def auto_update_feedback():
            """피드백 자동 업데이트"""
            if auto_refresh.value and chat_service.feedback.current_status != "대기 중...":
                return chat_service.feedback.get_status_html()
            return feedback_display.value
        
        # 주기적 업데이트 설정
        app.load(
            auto_update_feedback,
            None,
            feedback_display,
            every=1  # 1초마다 업데이트
        )
    
    return app


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced RAG System UI")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
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
    
    # 오버라이드
    if args.server_port:
        config.app.server_port = args.server_port
    if args.share:
        config.app.share = args.share
    
    # 앱 실행
    try:
        app = create_advanced_gradio_app(config)
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