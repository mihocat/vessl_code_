#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MathPix 스타일 OCR 통합 애플리케이션
전기공학 문제 해결을 위한 고급 AI 시스템
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import gradio as gr
from PIL import Image
import numpy as np
import json
from datetime import datetime
import torch

# 로컬 모듈
from mathpix_style_implementation import MathPixStyleOCR
from rag_system import RAGSystem
from llm_client import LLMClient

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ElectricalEngineeringAssistant:
    """전기공학 문제 해결 AI 어시스턴트"""
    
    def __init__(self):
        self.ocr_system = None
        self.rag_system = None
        self.llm_client = None
        self.formula_db = self._load_formula_database()
        self.initialized = False
        
    def _load_formula_database(self) -> Dict[str, Any]:
        """전기공학 공식 데이터베이스 로드"""
        return {
            "power_factor": {
                "역률": {
                    "formula": "cos(θ) = P / S",
                    "latex": r"\cos\theta = \frac{P}{S}",
                    "description": "유효전력과 피상전력의 비율"
                },
                "피상전력": {
                    "formula": "S = √(P² + Q²)",
                    "latex": r"S = \sqrt{P^2 + Q^2}",
                    "description": "유효전력과 무효전력의 벡터 합"
                },
                "콘덴서용량": {
                    "formula": "Qc = P(tan(θ₁) - tan(θ₂))",
                    "latex": r"Q_c = P(\tan\theta_1 - \tan\theta_2)",
                    "description": "역률 개선을 위한 콘덴서 용량"
                }
            },
            "laplace": {
                "기본변환": {
                    "pairs": [
                        {"time": "1", "laplace": "1/s"},
                        {"time": "t", "laplace": "1/s²"},
                        {"time": "e^(-at)", "laplace": "1/(s+a)"},
                        {"time": "sin(ωt)", "laplace": "ω/(s²+ω²)"},
                        {"time": "cos(ωt)", "laplace": "s/(s²+ω²)"}
                    ]
                }
            },
            "electric_field": {
                "쿨롱법칙": {
                    "formula": "F = k·Q₁·Q₂/r²",
                    "latex": r"F = k\frac{Q_1 Q_2}{r^2}",
                    "vector": r"\vec{F} = k\frac{Q_1 Q_2}{r^2}\hat{r}",
                    "constant": "k = 9×10⁹ N·m²/C²"
                }
            }
        }
    
    async def initialize(self):
        """시스템 초기화"""
        try:
            logger.info("Initializing Electrical Engineering Assistant...")
            
            # OCR 시스템 초기화
            logger.info("Initializing MathPix Style OCR...")
            self.ocr_system = MathPixStyleOCR()
            await self.ocr_system.initialize()
            
            # RAG 시스템 초기화
            logger.info("Initializing RAG System...")
            self.rag_system = RAGSystem()
            await self.rag_system.initialize()
            
            # LLM 클라이언트 초기화
            logger.info("Initializing LLM Client...")
            self.llm_client = LLMClient()
            
            self.initialized = True
            logger.info("System initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def process_question(
        self, 
        text: str, 
        image: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """질문 처리"""
        try:
            result = {
                'timestamp': datetime.now().isoformat(),
                'input_text': text,
                'has_image': image is not None
            }
            
            # 1. 이미지가 있는 경우 OCR 처리
            if image:
                logger.info("Processing image with MathPix Style OCR...")
                ocr_result = await self.ocr_system.process_image(image)
                
                result['ocr_result'] = {
                    'math_expressions': [
                        {
                            'type': expr['type'],
                            'latex': expr['expression'].latex,
                            'variables': expr['expression'].variables,
                            'confidence': expr['expression'].confidence
                        }
                        for expr in ocr_result['math_expressions']
                    ],
                    'text_content': ocr_result['text_content'],
                    'combined_content': ocr_result['combined_content']
                }
                
                # OCR 결과를 텍스트에 추가
                enhanced_text = self._enhance_query_with_ocr(text, ocr_result)
            else:
                enhanced_text = text
                result['ocr_result'] = None
            
            # 2. 문제 유형 분류
            problem_type = self._classify_problem(enhanced_text)
            result['problem_type'] = problem_type
            
            # 3. RAG 검색
            logger.info("Searching relevant documents...")
            rag_results = await self.rag_system.search(
                enhanced_text,
                top_k=5
            )
            result['rag_results'] = rag_results
            
            # 4. 솔루션 생성
            logger.info("Generating solution...")
            solution = await self._generate_solution(
                enhanced_text,
                problem_type,
                rag_results,
                ocr_result if image else None
            )
            result['solution'] = solution
            
            # 5. 응답 포맷팅
            formatted_response = self._format_response(result)
            result['formatted_response'] = formatted_response
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _enhance_query_with_ocr(
        self, 
        text: str, 
        ocr_result: Dict[str, Any]
    ) -> str:
        """OCR 결과로 쿼리 강화"""
        enhanced = text + "\n\n[OCR 인식 내용]\n"
        
        # 텍스트 내용 추가
        if ocr_result['text_content']:
            enhanced += f"텍스트: {ocr_result['text_content']}\n"
        
        # 수식 추가
        if ocr_result['math_expressions']:
            enhanced += "\n수식:\n"
            for i, expr in enumerate(ocr_result['math_expressions'], 1):
                latex = expr['expression'].latex
                enhanced += f"{i}. {latex}\n"
                
                # 변수 정보 추가
                if expr['expression'].variables:
                    vars_str = ", ".join(expr['expression'].variables)
                    enhanced += f"   변수: {vars_str}\n"
        
        return enhanced
    
    def _classify_problem(self, text: str) -> str:
        """문제 유형 분류"""
        keywords = {
            '역률개선': ['역률', '콘덴서', 'kVar', 'cosθ', '무효전력'],
            '라플라스변환': ['라플라스', '변환', 's+a', 'L['],
            '전기장': ['쿨롱', '점전하', '전기장', '전위'],
            '회로해석': ['KVL', 'KCL', '임피던스', 'RLC'],
            '전력계산': ['유효전력', '무효전력', '피상전력', 'kW']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for problem_type, keywords_list in keywords.items():
            score = sum(1 for keyword in keywords_list if keyword.lower() in text_lower)
            if score > 0:
                scores[problem_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "일반"
    
    async def _generate_solution(
        self,
        query: str,
        problem_type: str,
        rag_results: List[Dict],
        ocr_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """솔루션 생성"""
        # 프롬프트 구성
        prompt = self._build_solution_prompt(
            query, problem_type, rag_results, ocr_result
        )
        
        # LLM 호출
        response = await self.llm_client.generate(prompt)
        
        # 구조화된 솔루션 파싱
        solution = self._parse_solution(response)
        
        # 공식 데이터베이스에서 관련 공식 추가
        if problem_type in ['역률개선', '전력계산']:
            solution['relevant_formulas'] = self.formula_db.get('power_factor', {})
        elif problem_type == '라플라스변환':
            solution['relevant_formulas'] = self.formula_db.get('laplace', {})
        elif problem_type == '전기장':
            solution['relevant_formulas'] = self.formula_db.get('electric_field', {})
        
        return solution
    
    def _build_solution_prompt(
        self,
        query: str,
        problem_type: str,
        rag_results: List[Dict],
        ocr_result: Optional[Dict] = None
    ) -> str:
        """솔루션 생성을 위한 프롬프트 구성"""
        prompt = f"""당신은 전기공학 교수입니다. 다음 문제를 단계별로 풀어주세요.

문제 유형: {problem_type}

질문: {query}

"""
        
        if ocr_result and ocr_result['math_expressions']:
            prompt += "인식된 수식:\n"
            for expr in ocr_result['math_expressions']:
                prompt += f"- {expr['expression'].latex}\n"
            prompt += "\n"
        
        if rag_results:
            prompt += "참고 자료:\n"
            for result in rag_results[:3]:
                prompt += f"- {result['content'][:200]}...\n"
            prompt += "\n"
        
        prompt += """
풀이 형식:
1. 주어진 조건 정리
2. 사용할 공식 제시
3. 단계별 계산 (중간 과정 포함)
4. 최종 답 (단위 포함)

수식은 LaTeX 형식으로 작성하세요.
"""
        
        return prompt
    
    def _parse_solution(self, response: str) -> Dict[str, Any]:
        """솔루션 파싱"""
        solution = {
            'raw_response': response,
            'steps': [],
            'final_answer': None,
            'formulas_used': []
        }
        
        # 단계별로 파싱 (간단한 예시)
        lines = response.split('\n')
        current_step = None
        
        for line in lines:
            if line.startswith('1.') or line.startswith('2.') or \
               line.startswith('3.') or line.startswith('4.'):
                if current_step:
                    solution['steps'].append(current_step)
                current_step = {'title': line, 'content': []}
            elif current_step and line.strip():
                current_step['content'].append(line.strip())
        
        if current_step:
            solution['steps'].append(current_step)
        
        return solution
    
    def _format_response(self, result: Dict[str, Any]) -> str:
        """응답 포맷팅"""
        response = ""
        
        # OCR 결과가 있는 경우
        if result.get('ocr_result'):
            response += "### 인식된 내용\n"
            
            if result['ocr_result']['text_content']:
                response += f"**텍스트:** {result['ocr_result']['text_content']}\n\n"
            
            if result['ocr_result']['math_expressions']:
                response += "**수식:**\n"
                for expr in result['ocr_result']['math_expressions']:
                    response += f"- ${expr['latex']}$\n"
                response += "\n"
        
        # 문제 유형
        response += f"### 문제 유형: {result['problem_type']}\n\n"
        
        # 솔루션
        if 'solution' in result:
            response += "### 풀이 과정\n\n"
            
            for step in result['solution'].get('steps', []):
                response += f"**{step['title']}**\n"
                for content in step['content']:
                    response += f"{content}\n"
                response += "\n"
            
            # 관련 공식
            if 'relevant_formulas' in result['solution']:
                response += "### 관련 공식\n"
                for name, formula in result['solution']['relevant_formulas'].items():
                    if isinstance(formula, dict) and 'latex' in formula:
                        response += f"- **{name}**: ${formula['latex']}$\n"
                        if 'description' in formula:
                            response += f"  {formula['description']}\n"
                response += "\n"
        
        # RAG 참고 자료
        if result.get('rag_results'):
            response += "### 참고 자료\n"
            for i, rag in enumerate(result['rag_results'][:3], 1):
                response += f"{i}. {rag['content'][:100]}...\n"
        
        return response

# Gradio 인터페이스
def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    assistant = ElectricalEngineeringAssistant()
    
    # 비동기 초기화를 위한 래퍼
    async def init_and_process(text, image):
        if not assistant.initialized:
            await assistant.initialize()
        
        if image is not None:
            # PIL Image로 변환
            pil_image = Image.fromarray(image)
        else:
            pil_image = None
        
        result = await assistant.process_question(text, pil_image)
        
        if 'error' in result:
            return f"오류가 발생했습니다: {result['error']}"
        
        return result['formatted_response']
    
    # 동기 래퍼
    def process_wrapper(text, image):
        return asyncio.run(init_and_process(text, image))
    
    # Gradio 인터페이스
    with gr.Blocks(title="전기공학 AI 어시스턴트") as demo:
        gr.Markdown("""
        # 전기공학 AI 어시스턴트
        
        MathPix 스타일 OCR을 활용한 고급 전기공학 문제 해결 시스템입니다.
        
        ## 주요 기능
        - 수식 인식 (LaTeX 변환)
        - 한국어 텍스트 처리
        - 전기공학 문제 자동 분류
        - 단계별 풀이 제공
        - 관련 공식 제시
        """)
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="질문 입력",
                    placeholder="전기공학 관련 질문을 입력하세요...",
                    lines=5
                )
                image_input = gr.Image(
                    label="문제 이미지 (선택사항)",
                    type="numpy"
                )
                submit_btn = gr.Button("질문하기", variant="primary")
                
                # 예시 질문
                gr.Examples(
                    examples=[
                        ["역률 0.8인 부하에 콘덴서를 설치하여 역률을 0.95로 개선하려고 합니다. 필요한 콘덴서 용량은?"],
                        ["라플라스 변환 L[t·e^(-2t)]를 구하시오."],
                        ["두 점전하 Q1=2μC, Q2=-3μC가 3m 떨어져 있을 때, 중간 지점의 전기장은?"]
                    ],
                    inputs=text_input
                )
            
            with gr.Column():
                output = gr.Markdown(
                    label="AI 응답",
                    value="여기에 AI의 답변이 표시됩니다."
                )
        
        submit_btn.click(
            fn=process_wrapper,
            inputs=[text_input, image_input],
            outputs=output
        )
        
        gr.Markdown("""
        ---
        ### 사용 안내
        1. 텍스트로 질문하거나 문제 이미지를 업로드하세요
        2. 수식이 포함된 이미지는 자동으로 인식되어 LaTeX로 변환됩니다
        3. 문제 유형에 따라 적절한 공식과 풀이 과정이 제공됩니다
        
        ### 지원 문제 유형
        - 역률 개선 문제
        - 라플라스 변환
        - 전기장/자기장 계산
        - 회로 해석
        - 전력 계산
        """)
    
    return demo

# 메인 실행
if __name__ == "__main__":
    # Gradio 앱 생성 및 실행
    demo = create_gradio_interface()
    
    # 서버 실행
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )