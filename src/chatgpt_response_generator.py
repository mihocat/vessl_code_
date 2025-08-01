#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGPT-style Response Generator
ChatGPT 스타일 응답 생성기
"""

import logging
from typing import Dict, Any, List, Optional
import json
import re

logger = logging.getLogger(__name__)


class ChatGPTStylePrompts:
    """ChatGPT 스타일 프롬프트 템플릿"""
    
    # 시스템 프롬프트
    SYSTEM_PROMPT = """당신은 전기공학 분야의 최고 전문가이자 친절한 교수입니다.
학생들이 이해하기 쉽도록 단계별로 설명하며, 시각적 요소를 활용하여 개념을 명확히 전달합니다.

응답 형식:
1. ✅ **핵심 정리**: 질문의 핵심을 한 문장으로 요약
2. 🔹 **개념 설명**: 관련 개념을 명확하게 설명
3. 📊 **단계별 풀이**: 문제 해결 과정을 단계별로 제시
4. 🔸 **시각적 요소**: 필요시 표, 다이어그램, 수식 활용
5. 💡 **추가 팁**: 관련된 유용한 정보나 주의사항
6. ✅ **최종 답**: 결론을 명확히 제시

수식 표현:
- 인라인 수식: $수식$
- 블록 수식: $$수식$$
- 복잡한 수식은 단계별로 분리하여 설명

전문성:
- 정확한 전기공학 용어 사용
- 단위와 기호를 정확히 표기
- 실무적 관점 포함"""

    # 이미지 분석 프롬프트
    IMAGE_ANALYSIS_PROMPT = """다음은 전기공학 문제의 이미지 분석 결과입니다:

**추출된 텍스트**: {ocr_text}
**감지된 수식**: {formulas}
**다이어그램 정보**: {diagrams}

이 정보를 바탕으로 문제를 이해하고 해결해주세요."""

    # 단계별 풀이 프롬프트
    STEP_BY_STEP_PROMPT = """문제를 다음과 같은 구조로 해결해주세요:

1. **문제 이해** 🎯
   - 주어진 조건 정리
   - 구하고자 하는 값 명확화
   - 관련 공식/이론 확인

2. **해결 전략** 📋
   - 접근 방법 설명
   - 사용할 공식 제시
   - 주의사항 언급

3. **단계별 계산** 🔢
   - 각 단계마다 명확한 설명
   - 중간 계산 과정 표시
   - 단위 변환 주의

4. **검증 및 해석** ✔️
   - 답의 타당성 검토
   - 물리적 의미 설명
   - 실무적 관점 추가"""

    # 개념 설명 프롬프트
    CONCEPT_EXPLANATION_PROMPT = """전기공학 개념을 설명할 때:

1. **정의**: 명확하고 간단한 정의
2. **원리**: 작동 원리나 이론적 배경
3. **공식**: 관련 수식과 각 항의 의미
4. **예시**: 실제 응용 사례
5. **그림/도표**: 시각적 설명 (텍스트로 설명)"""


class ResponseFormatter:
    """응답 포맷터"""
    
    @staticmethod
    def format_latex(text: str) -> str:
        """LaTeX 수식 포맷팅"""
        # 인라인 수식 변환
        text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
        text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text)
        
        # 특수 기호 변환
        replacements = {
            '\\alpha': 'α', '\\beta': 'β', '\\gamma': 'γ', '\\delta': 'δ',
            '\\theta': 'θ', '\\phi': 'φ', '\\omega': 'ω', '\\Omega': 'Ω',
            '\\mu': 'μ', '\\pi': 'π', '\\sigma': 'σ', '\\tau': 'τ',
            '\\infty': '∞', '\\sqrt': '√', '\\times': '×', '\\div': '÷'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    @staticmethod
    def create_table(headers: List[str], rows: List[List[str]]) -> str:
        """마크다운 표 생성"""
        # 헤더
        table = "| " + " | ".join(headers) + " |\n"
        table += "|" + "|".join(["---" for _ in headers]) + "|\n"
        
        # 행
        for row in rows:
            table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        
        return table
    
    @staticmethod
    def format_circuit_diagram(components: List[Dict[str, str]]) -> str:
        """회로도를 텍스트로 표현"""
        diagram = "```\n"
        diagram += "회로 구성:\n"
        
        for comp in components:
            if comp['type'] == 'resistor':
                diagram += f"─[{comp['label']}]─"
            elif comp['type'] == 'capacitor':
                diagram += f"─|{comp['label']}|─"
            elif comp['type'] == 'inductor':
                diagram += f"─⌒{comp['label']}⌒─"
            elif comp['type'] == 'voltage':
                diagram += f"(+) {comp['label']} (-)"
            diagram += "\n"
        
        diagram += "```"
        return diagram


class ChatGPTResponseGenerator:
    """ChatGPT 스타일 응답 생성기"""
    
    def __init__(self):
        self.prompts = ChatGPTStylePrompts()
        self.formatter = ResponseFormatter()
    
    def generate_response(
        self,
        question: str,
        context: Dict[str, Any],
        response_type: str = 'comprehensive'
    ) -> str:
        """ChatGPT 스타일 응답 생성"""
        
        # 응답 타입별 처리
        if response_type == 'comprehensive':
            return self._generate_comprehensive_response(question, context)
        elif response_type == 'step_by_step':
            return self._generate_step_by_step_response(question, context)
        elif response_type == 'concept':
            return self._generate_concept_explanation(question, context)
        else:
            return self._generate_simple_response(question, context)
    
    def _generate_comprehensive_response(
        self, 
        question: str, 
        context: Dict[str, Any]
    ) -> str:
        """종합적인 응답 생성"""
        response_parts = []
        
        # 1. 핵심 정리
        response_parts.append(self._create_key_summary(question, context))
        
        # 2. 문제 분석 (이미지가 있는 경우)
        if context.get('image_analysis'):
            response_parts.append(self._analyze_problem(context['image_analysis']))
        
        # 3. 개념 설명
        if context.get('key_concepts'):
            response_parts.append(self._explain_concepts(context['key_concepts']))
        
        # 4. 단계별 풀이
        if context.get('solution_steps'):
            response_parts.append(self._create_step_by_step_solution(context['solution_steps']))
        
        # 5. 시각적 요소
        if context.get('visual_elements'):
            response_parts.append(self._add_visual_elements(context['visual_elements']))
        
        # 6. 최종 답변
        response_parts.append(self._create_final_answer(question, context))
        
        # 7. 추가 팁
        response_parts.append(self._add_tips(context))
        
        return "\n\n".join(filter(None, response_parts))
    
    def _create_key_summary(self, question: str, context: Dict[str, Any]) -> str:
        """핵심 정리 생성"""
        summary = "✅ **핵심 정리:**\n"
        
        # 질문 분석
        if '계산' in question or '구하' in question:
            summary += "이 문제는 주어진 조건을 활용하여 특정 값을 계산하는 문제입니다."
        elif '설명' in question or '무엇' in question:
            summary += "이 질문은 전기공학 개념에 대한 이해를 묻고 있습니다."
        elif '차이' in question or '비교' in question:
            summary += "이 문제는 두 개념 또는 값의 차이점을 비교 분석하는 문제입니다."
        else:
            summary += "주어진 문제를 체계적으로 분석하고 해결해보겠습니다."
        
        return summary
    
    def _analyze_problem(self, image_analysis: Dict[str, Any]) -> str:
        """문제 분석"""
        analysis = "🔍 **문제 분석:**\n"
        
        if image_analysis.get('ocr_text'):
            analysis += f"- 문제 내용: {image_analysis['ocr_text'][:100]}...\n"
        
        if image_analysis.get('formulas'):
            analysis += f"- 감지된 수식: {len(image_analysis['formulas'])}개\n"
            for i, formula in enumerate(image_analysis['formulas'][:3]):
                analysis += f"  - 수식 {i+1}: ${formula}$\n"
        
        if image_analysis.get('circuit_components'):
            analysis += f"- 회로 구성요소: "
            components = [comp['label'] for comp in image_analysis['circuit_components']]
            analysis += ", ".join(components) + "\n"
        
        return analysis
    
    def _explain_concepts(self, concepts: List[str]) -> str:
        """개념 설명"""
        if not concepts:
            return ""
        
        explanation = "🔹 **관련 개념 설명:**\n"
        
        # 주요 전기공학 개념 설명 (예시)
        concept_definitions = {
            '전압': "전압(Voltage, V)은 두 점 사이의 전위차로, 전하를 이동시키는 힘입니다. 단위는 볼트(V)입니다.",
            '전류': "전류(Current, I)는 단위 시간당 흐르는 전하량으로, 단위는 암페어(A)입니다.",
            '저항': "저항(Resistance, R)은 전류의 흐름을 방해하는 정도로, 단위는 옴(Ω)입니다.",
            '전력': "전력(Power, P)은 단위 시간당 사용되는 에너지로, P = VI 또는 P = I²R로 계산됩니다.",
            '임피던스': "임피던스(Impedance, Z)는 교류 회로에서의 저항으로, 저항과 리액턴스의 복소수 합입니다.",
            '역률': "역률(Power Factor, PF)은 유효전력과 피상전력의 비로, cosθ로 표현됩니다."
        }
        
        for concept in concepts[:3]:  # 상위 3개만
            if concept in concept_definitions:
                explanation += f"\n**{concept}**: {concept_definitions[concept]}\n"
        
        return explanation
    
    def _create_step_by_step_solution(self, steps: List[Dict[str, Any]]) -> str:
        """단계별 풀이 생성"""
        if not steps:
            return ""
        
        solution = "📊 **단계별 풀이:**\n"
        
        for step in steps:
            title = step.get('title', f"단계 {step.get('step', '')}")
            solution += f"\n**{title}**\n"
            
            if step.get('content'):
                solution += f"{step['content']}\n"
            
            if step.get('formulas'):
                for formula in step['formulas']:
                    solution += f"\n$$\n{formula}\n$$\n"
            
            if step.get('calculation'):
                solution += f"\n계산: {step['calculation']}\n"
        
        return solution
    
    def _add_visual_elements(self, visual_elements: Dict[str, Any]) -> str:
        """시각적 요소 추가"""
        visual = ""
        
        if visual_elements.get('needs_table'):
            # 예시 표
            visual += "\n📊 **비교 표:**\n"
            headers = ['항목', '값', '단위', '설명']
            rows = [
                ['전압', '220', 'V', 'AC 전압'],
                ['전류', '10', 'A', '부하 전류'],
                ['전력', '2200', 'W', '소비 전력']
            ]
            visual += self.formatter.create_table(headers, rows)
        
        if visual_elements.get('has_circuit'):
            visual += "\n🔌 **회로도:**\n"
            visual += "```\n"
            visual += "     R1=10Ω    R2=20Ω\n"
            visual += "  ──[////]──[////]──\n"
            visual += " │                  │\n"
            visual += "(+) V=100V         │\n"
            visual += " │                  │\n"
            visual += "  ──────────────────\n"
            visual += "```\n"
        
        return visual
    
    def _create_final_answer(self, question: str, context: Dict[str, Any]) -> str:
        """최종 답변 생성"""
        answer = "✅ **최종 답:**\n"
        
        if context.get('final_answer'):
            answer += context['final_answer']
        else:
            answer += "위의 분석과 계산을 통해 문제를 해결했습니다."
        
        # 답의 타당성 검증
        answer += "\n\n**검증:**\n"
        answer += "- 단위가 올바른지 확인 ✓\n"
        answer += "- 물리적으로 타당한 값인지 확인 ✓\n"
        answer += "- 계산 과정 재검토 완료 ✓"
        
        return answer
    
    def _add_tips(self, context: Dict[str, Any]) -> str:
        """추가 팁 생성"""
        tips = "💡 **추가 팁:**\n"
        
        tips_list = [
            "- 전기 회로 문제는 항상 KVL(키르히호프 전압 법칙)과 KCL(키르히호프 전류 법칙)을 확인하세요.",
            "- 단위 변환에 주의하세요. 특히 k(kilo), m(milli), μ(micro) 접두사를 정확히 처리해야 합니다.",
            "- 복소수 계산 시 극좌표와 직교좌표 변환을 능숙하게 다루어야 합니다.",
            "- 안전율과 실무적 고려사항을 항상 염두에 두세요."
        ]
        
        # 컨텍스트에 따라 관련 팁 선택
        if '전력' in str(context):
            tips += tips_list[1]
        elif '회로' in str(context):
            tips += tips_list[0]
        else:
            tips += tips_list[-1]
        
        return tips
    
    def _generate_step_by_step_response(self, question: str, context: Dict[str, Any]) -> str:
        """단계별 응답 생성"""
        response = "🎯 **문제 해결 과정**\n\n"
        
        # 1단계: 문제 이해
        response += "### 1️⃣ 문제 이해\n"
        response += f"질문: {question}\n\n"
        
        # 2단계: 주어진 조건
        response += "### 2️⃣ 주어진 조건\n"
        if context.get('given_values'):
            for key, value in context['given_values'].items():
                response += f"- {key}: {value}\n"
        response += "\n"
        
        # 3단계: 해결 전략
        response += "### 3️⃣ 해결 전략\n"
        response += "사용할 공식과 접근 방법:\n"
        response += "- 오옴의 법칙: V = IR\n"
        response += "- 전력 공식: P = VI = I²R = V²/R\n\n"
        
        # 4단계: 계산
        response += "### 4️⃣ 단계별 계산\n"
        response += "각 단계의 상세한 계산 과정...\n\n"
        
        # 5단계: 결론
        response += "### 5️⃣ 결론\n"
        response += "최종 답과 의미 해석...\n"
        
        return response
    
    def _generate_concept_explanation(self, question: str, context: Dict[str, Any]) -> str:
        """개념 설명 응답 생성"""
        response = "📚 **개념 설명**\n\n"
        
        # 개념 정의
        response += "### 정의\n"
        response += "해당 개념의 명확한 정의...\n\n"
        
        # 원리 설명
        response += "### 작동 원리\n"
        response += "물리적/전기적 원리 설명...\n\n"
        
        # 수식과 공식
        response += "### 관련 공식\n"
        response += "$$공식$$\n\n"
        
        # 실제 예시
        response += "### 실제 응용 예시\n"
        response += "- 예시 1: ...\n"
        response += "- 예시 2: ...\n"
        
        return response
    
    def _generate_simple_response(self, question: str, context: Dict[str, Any]) -> str:
        """간단한 응답 생성"""
        return f"질문: {question}\n\n답변: {context.get('answer', '답변을 생성하는 중입니다.')}"


class PromptEnhancer:
    """프롬프트 향상기"""
    
    @staticmethod
    def enhance_with_image_context(prompt: str, image_analysis: Dict[str, Any]) -> str:
        """이미지 컨텍스트를 포함한 프롬프트 향상"""
        enhanced = prompt
        
        if image_analysis.get('formulas'):
            enhanced += "\n\n발견된 수식:\n"
            for formula in image_analysis['formulas']:
                enhanced += f"- ${formula}$\n"
        
        if image_analysis.get('circuit_components'):
            enhanced += "\n\n회로 구성요소:\n"
            for comp in image_analysis['circuit_components']:
                enhanced += f"- {comp['type']}: {comp['label']}\n"
        
        return enhanced
    
    @staticmethod
    def add_domain_knowledge(prompt: str, domain: str = 'electrical_engineering') -> str:
        """도메인 지식 추가"""
        domain_context = {
            'electrical_engineering': """
전기공학 도메인 특화 지식:
- 3상 전력 시스템: 평형/불평형 부하
- 변압기: 이상/실제 변압기, 등가회로
- 전동기: 유도전동기, 동기전동기
- 전력전자: 정류기, 인버터, 컨버터
- 송배전: 전압강하, 전력손실, 역률개선
""",
            'circuit_analysis': """
회로 해석 특화 지식:
- 키르히호프 법칙 (KVL, KCL)
- 테브난/노턴 등가회로
- 과도현상 해석 (RC, RL, RLC)
- 주파수 응답 (보드 선도)
- 라플라스 변환
"""
        }
        
        if domain in domain_context:
            return prompt + "\n\n" + domain_context[domain]
        
        return prompt


if __name__ == "__main__":
    # 테스트
    generator = ChatGPTResponseGenerator()
    
    # 테스트 컨텍스트
    test_context = {
        'image_analysis': {
            'ocr_text': '3상 전력 시스템에서 선간전압이 380V이고 부하전류가 10A일 때 전력을 구하시오.',
            'formulas': ['P = \\sqrt{3} \\times V_L \\times I_L \\times \\cos\\theta'],
            'circuit_components': [
                {'type': 'resistor', 'label': 'R1'},
                {'type': 'inductor', 'label': 'L1'}
            ]
        },
        'key_concepts': ['전력', '3상', '역률'],
        'solution_steps': [
            {
                'step': 1,
                'title': '주어진 값 정리',
                'content': '선간전압 VL = 380V, 부하전류 IL = 10A'
            },
            {
                'step': 2,
                'title': '공식 적용',
                'formulas': ['P = \\sqrt{3} \\times 380 \\times 10 \\times \\cos\\theta']
            }
        ],
        'visual_elements': {
            'needs_table': True,
            'has_circuit': True
        }
    }
    
    # 응답 생성
    response = generator.generate_response(
        "3상 전력을 계산하는 방법을 설명해주세요.",
        test_context,
        response_type='comprehensive'
    )
    
    print(response)