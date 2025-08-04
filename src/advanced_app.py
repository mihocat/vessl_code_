#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고급 통합 애플리케이션
거시적 시스템을 활용한 전기공학 AI 튜터
"""

import gradio as gr
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import json
import pandas as pd
from datetime import datetime
import os

# 로컬 모듈
from integrated_ai_system import IntegratedAISystem, AutoEvaluator
from config import Config
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class ElectricalAITutor:
    """전기공학 AI 튜터"""
    
    def __init__(self):
        self.ai_system = None
        self.evaluator = None
        self.history = []
        self.initialized = False
        
    async def initialize(self):
        """시스템 초기화"""
        try:
            logger.info("Initializing Electrical AI Tutor...")
            
            # 설정 로드
            config = Config()
            llm_client = LLMClient(config.llm)
            
            # AI 시스템 초기화
            self.ai_system = IntegratedAISystem()
            await self.ai_system.initialize({
                'rag': config.rag,
                'dataset': config.dataset,
                'llm_client': llm_client
            })
            
            # 평가기 초기화
            self.evaluator = AutoEvaluator(self.ai_system)
            
            self.initialized = True
            logger.info("Electrical AI Tutor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
    
    async def solve_problem(
        self,
        text: str,
        image: Optional[np.ndarray] = None,
        show_steps: bool = True,
        explain_detail: bool = True
    ) -> Tuple[str, str, Dict[str, Any]]:
        """문제 해결"""
        
        if not self.initialized:
            await self.initialize()
        
        try:
            # PIL Image로 변환
            pil_image = None
            if image is not None:
                pil_image = Image.fromarray(image)
            
            # 문제 해결
            solution = await self.ai_system.solve_problem(
                image=pil_image,
                text=text
            )
            
            # 결과 포맷팅
            main_answer = self._format_main_answer(solution)
            detailed_solution = self._format_detailed_solution(
                solution, show_steps, explain_detail
            )
            
            # 히스토리 저장
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'problem': text,
                'solution': solution,
                'has_image': image is not None
            })
            
            # 분석 데이터
            analysis = {
                'problem_type': solution.problem_id,
                'confidence': solution.confidence,
                'formulas_used': solution.formulas_used,
                'verification': solution.verification
            }
            
            return main_answer, detailed_solution, analysis
            
        except Exception as e:
            logger.error(f"Problem solving failed: {e}")
            error_msg = f"오류가 발생했습니다: {str(e)}"
            return error_msg, error_msg, {}
    
    def _format_main_answer(self, solution) -> str:
        """주요 답변 포맷팅"""
        answer = "## 최종 답\n\n"
        
        if solution.final_answer and 'primary_answer' in solution.final_answer:
            pa = solution.final_answer['primary_answer']
            answer += f"**{pa['variable']} = {pa['value']:.3f} {pa['unit']}**\n\n"
        
        # 모든 계산된 값
        if solution.final_answer and 'values' in solution.final_answer:
            answer += "### 계산된 값들:\n"
            for var, val in solution.final_answer['values'].items():
                unit = self.ai_system.formula_solver._infer_unit(var)
                answer += f"- {var} = {val:.3f} {unit}\n"
        
        # 신뢰도
        answer += f"\n신뢰도: {solution.confidence:.0%}"
        
        return answer
    
    def _format_detailed_solution(
        self, 
        solution,
        show_steps: bool,
        explain_detail: bool
    ) -> str:
        """상세 해결 과정 포맷팅"""
        detailed = "## 상세 풀이 과정\n\n"
        
        # 사용된 공식
        if solution.formulas_used:
            detailed += "### 사용된 공식\n"
            for formula in solution.formulas_used:
                detailed += f"- {formula}\n"
            detailed += "\n"
        
        # 단계별 풀이
        if show_steps and solution.steps:
            detailed += "### 단계별 계산\n"
            for i, step in enumerate(solution.steps, 1):
                detailed += f"\n**Step {i}: {step['variable']} 계산**\n"
                detailed += f"- 공식: {step['formula']}\n"
                detailed += f"- 계산: {step['calculation']}\n"
                
                # 사용된 값들
                if 'used_values' in step:
                    detailed += "- 사용된 값: "
                    for var, val in step['used_values'].items():
                        detailed += f"{var}={val:.3f}, "
                    detailed = detailed.rstrip(", ") + "\n"
                
                detailed += f"- **결과: {step['result']:.3f}**\n"
        
        # 상세 설명
        if explain_detail and solution.explanation:
            detailed += f"\n### 설명\n{solution.explanation}\n"
        
        # 검증 결과
        if solution.verification:
            detailed += "\n### 검증\n"
            for check, passed in solution.verification.items():
                status = "✅" if passed else "❌"
                detailed += f"- {check}: {status}\n"
        
        return detailed
    
    async def batch_evaluate(self, test_file: str) -> Tuple[str, pd.DataFrame]:
        """일괄 평가"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # 테스트 케이스 로드
            self.evaluator.load_test_cases(test_file)
            
            # 평가 수행
            results = await self.evaluator.evaluate()
            
            # 보고서
            report = results['report']
            
            # 상세 결과 DataFrame
            data = []
            for r in results['results']:
                data.append({
                    'Test ID': r['test_id'],
                    'Problem Type': r['problem_type'],
                    'Correct': '✅' if r['correct'] else '❌',
                    'Confidence': f"{r['solution'].confidence:.2f}",
                    'Score': f"{r['score']:.2f}"
                })
            
            df = pd.DataFrame(data)
            
            return report, df
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            return f"평가 실패: {str(e)}", pd.DataFrame()
    
    def get_history(self) -> pd.DataFrame:
        """히스토리 조회"""
        if not self.history:
            return pd.DataFrame()
        
        data = []
        for h in self.history:
            data.append({
                'Time': h['timestamp'],
                'Problem': h['problem'][:50] + '...' if len(h['problem']) > 50 else h['problem'],
                'Type': h['solution'].problem_id,
                'Confidence': f"{h['solution'].confidence:.0%}",
                'Image': '📷' if h['has_image'] else ''
            })
        
        return pd.DataFrame(data)
    
    def export_history(self, format: str = 'json') -> str:
        """히스토리 내보내기"""
        if format == 'json':
            # JSON으로 내보내기
            export_data = []
            for h in self.history:
                export_data.append({
                    'timestamp': h['timestamp'],
                    'problem': h['problem'],
                    'solution': {
                        'answer': h['solution'].final_answer,
                        'steps': h['solution'].steps,
                        'confidence': h['solution'].confidence
                    }
                })
            
            filename = f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return filename
        
        return ""


def create_advanced_interface():
    """고급 Gradio 인터페이스 생성"""
    tutor = ElectricalAITutor()
    
    # 비동기 래퍼
    async def solve_wrapper(text, image, show_steps, explain_detail):
        return await tutor.solve_problem(text, image, show_steps, explain_detail)
    
    def solve_sync(text, image, show_steps, explain_detail):
        return asyncio.run(solve_wrapper(text, image, show_steps, explain_detail))
    
    async def evaluate_wrapper(file):
        if file is None:
            return "파일을 업로드해주세요.", pd.DataFrame()
        return await tutor.batch_evaluate(file.name)
    
    def evaluate_sync(file):
        return asyncio.run(evaluate_wrapper(file))
    
    def get_history_sync():
        return tutor.get_history()
    
    def export_history_sync(format):
        filename = tutor.export_history(format)
        return f"내보내기 완료: {filename}"
    
    # Gradio 인터페이스
    with gr.Blocks(title="전기공학 AI 튜터", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎓 전기공학 AI 튜터
        
        고급 통합 AI 시스템을 활용한 전기공학 문제 해결 도우미
        
        ## 주요 기능
        - 📸 이미지에서 수식과 문제 자동 인식 (OCR)
        - 🧮 전기공학 공식 기반 자동 계산
        - 📚 RAG 시스템을 통한 관련 지식 검색
        - 🤖 LLM을 활용한 상세 설명 생성
        - ✅ 자동 검증 및 평가
        """)
        
        with gr.Tab("문제 해결"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="문제 설명",
                        placeholder="문제를 입력하거나 이미지를 업로드하세요...",
                        lines=5
                    )
                    image_input = gr.Image(
                        label="문제 이미지 (선택)",
                        type="numpy"
                    )
                    
                    with gr.Row():
                        show_steps = gr.Checkbox(
                            label="단계별 풀이 표시",
                            value=True
                        )
                        explain_detail = gr.Checkbox(
                            label="상세 설명 포함",
                            value=True
                        )
                    
                    solve_btn = gr.Button(
                        "🔍 문제 해결",
                        variant="primary",
                        size="lg"
                    )
                    
                    # 예제
                    gr.Examples(
                        examples=[
                            ["유효전력 320kW, 무효전력 140kVar일 때 역률은?", None],
                            ["3상 부하의 피상전력이 500kVA이고 역률이 0.8일 때, 유효전력과 무효전력을 구하시오.", None],
                            ["100Ω의 저항과 50Ω의 리액턴스가 직렬 연결되어 있을 때 임피던스는?", None]
                        ],
                        inputs=[text_input, image_input]
                    )
                
                with gr.Column(scale=1):
                    main_answer = gr.Markdown(
                        label="최종 답",
                        value="여기에 답이 표시됩니다."
                    )
                    
                    detailed_solution = gr.Markdown(
                        label="상세 풀이",
                        value="상세한 풀이 과정이 여기에 표시됩니다."
                    )
                    
                    with gr.Accordion("분석 정보", open=False):
                        analysis_json = gr.JSON(
                            label="상세 분석"
                        )
            
            solve_btn.click(
                fn=solve_sync,
                inputs=[text_input, image_input, show_steps, explain_detail],
                outputs=[main_answer, detailed_solution, analysis_json]
            )
        
        with gr.Tab("일괄 평가"):
            gr.Markdown("""
            ### 테스트 케이스 일괄 평가
            
            JSON 형식의 테스트 파일을 업로드하여 여러 문제를 한번에 평가할 수 있습니다.
            """)
            
            test_file = gr.File(
                label="테스트 파일 (JSON)",
                file_types=[".json"]
            )
            
            evaluate_btn = gr.Button("📊 평가 시작", variant="primary")
            
            eval_report = gr.Markdown(label="평가 보고서")
            eval_results = gr.DataFrame(label="상세 결과")
            
            evaluate_btn.click(
                fn=evaluate_sync,
                inputs=test_file,
                outputs=[eval_report, eval_results]
            )
        
        with gr.Tab("히스토리"):
            gr.Markdown("### 문제 해결 히스토리")
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 새로고침")
                export_format = gr.Dropdown(
                    choices=["json"],
                    value="json",
                    label="내보내기 형식"
                )
                export_btn = gr.Button("💾 내보내기")
            
            history_table = gr.DataFrame(
                label="히스토리",
                interactive=False
            )
            
            export_status = gr.Textbox(
                label="내보내기 상태",
                interactive=False
            )
            
            refresh_btn.click(
                fn=get_history_sync,
                outputs=history_table
            )
            
            export_btn.click(
                fn=export_history_sync,
                inputs=export_format,
                outputs=export_status
            )
            
            # 초기 히스토리 로드
            demo.load(fn=get_history_sync, outputs=history_table)
        
        with gr.Tab("도움말"):
            gr.Markdown("""
            ## 사용 방법
            
            ### 1. 문제 입력
            - 텍스트로 문제를 직접 입력하거나
            - 문제가 포함된 이미지를 업로드하세요
            - 이미지와 텍스트를 함께 사용할 수도 있습니다
            
            ### 2. 지원되는 문제 유형
            - **역률 계산**: 유효전력, 무효전력, 피상전력, 역률
            - **회로 해석**: 옴의 법칙, 전력 계산
            - **임피던스**: 직렬/병렬 임피던스, 위상각
            - **변압기**: 변압비, 전력 변환
            - **전동기**: 출력, 효율, 역률
            
            ### 3. 출력 설명
            - **최종 답**: 구하고자 하는 값의 계산 결과
            - **상세 풀이**: 단계별 계산 과정과 사용된 공식
            - **검증**: 답의 물리적 타당성 검증
            
            ### 4. 일괄 평가
            테스트 파일 형식:
            ```json
            [
                {
                    "id": "test_001",
                    "type": "power_factor",
                    "text": "문제 설명",
                    "expected_answer": {"value": 0.8}
                }
            ]
            ```
            """)
    
    return demo


if __name__ == "__main__":
    demo = create_advanced_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )