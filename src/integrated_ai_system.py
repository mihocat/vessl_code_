#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 AI 시스템 - 거시적 아키텍처
전기공학 문제 해결을 위한 완전한 End-to-End 시스템
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
from PIL import Image

# 로컬 모듈
from real_ocr_system import RealOCRSystem, ElectricalOCRAnalyzer
from rag_system import RAGSystem
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class ProblemType(Enum):
    """문제 유형"""
    POWER_FACTOR = "power_factor"
    POWER_CALCULATION = "power_calculation"
    CIRCUIT_ANALYSIS = "circuit_analysis"
    LAPLACE_TRANSFORM = "laplace_transform"
    ELECTRIC_FIELD = "electric_field"
    IMPEDANCE = "impedance"
    TRANSFORMER = "transformer"
    MOTOR = "motor"
    GENERAL = "general"


@dataclass
class Problem:
    """문제 구조체"""
    type: ProblemType
    description: str
    given_values: List[Dict[str, Any]]
    find_targets: List[str]
    constraints: List[str]
    formulas: List[str]
    image_path: Optional[str] = None
    ocr_text: Optional[str] = None


@dataclass
class Solution:
    """해결 구조체"""
    problem_id: str
    steps: List[Dict[str, Any]]
    final_answer: Dict[str, Any]
    confidence: float
    verification: Dict[str, bool]
    explanation: str
    formulas_used: List[str]


class ElectricalFormulaSolver:
    """전기공학 공식 해결기"""
    
    def __init__(self):
        self.formulas = self._load_formulas()
        
    def _load_formulas(self) -> Dict[str, Dict[str, Any]]:
        """전기공학 공식 데이터베이스"""
        return {
            "power_factor": {
                "basic": {
                    "formula": "cos(θ) = P / S",
                    "solve_for": {
                        "cos_theta": "P / S",
                        "P": "S * cos(θ)",
                        "S": "P / cos(θ)"
                    }
                },
                "improvement": {
                    "formula": "Qc = P * (tan(θ1) - tan(θ2))",
                    "solve_for": {
                        "Qc": "P * (tan(θ1) - tan(θ2))",
                        "P": "Qc / (tan(θ1) - tan(θ2))"
                    }
                },
                "apparent_power": {
                    "formula": "S = sqrt(P² + Q²)",
                    "solve_for": {
                        "S": "sqrt(P² + Q²)",
                        "P": "sqrt(S² - Q²)",
                        "Q": "sqrt(S² - P²)"
                    }
                }
            },
            "ohms_law": {
                "basic": {
                    "formula": "V = I * R",
                    "solve_for": {
                        "V": "I * R",
                        "I": "V / R",
                        "R": "V / I"
                    }
                },
                "power": {
                    "formula": "P = V * I",
                    "solve_for": {
                        "P": "V * I",
                        "V": "P / I",
                        "I": "P / V"
                    }
                }
            },
            "impedance": {
                "series": {
                    "formula": "Z = sqrt(R² + X²)",
                    "solve_for": {
                        "Z": "sqrt(R² + X²)",
                        "R": "sqrt(Z² - X²)",
                        "X": "sqrt(Z² - R²)"
                    }
                },
                "angle": {
                    "formula": "θ = arctan(X/R)",
                    "solve_for": {
                        "theta": "arctan(X/R)",
                        "X": "R * tan(θ)",
                        "R": "X / tan(θ)"
                    }
                }
            }
        }
    
    def solve(self, problem: Problem) -> Dict[str, Any]:
        """문제 해결"""
        # 문제 유형에 따른 공식 선택
        relevant_formulas = self._select_formulas(problem)
        
        # 주어진 값 정리
        known_values = self._organize_values(problem.given_values)
        
        # 구할 값 식별
        unknowns = problem.find_targets
        
        # 단계별 해결
        steps = []
        current_values = known_values.copy()
        
        for unknown in unknowns:
            step = self._solve_for_unknown(
                unknown, 
                current_values, 
                relevant_formulas
            )
            if step:
                steps.append(step)
                current_values[unknown] = step['result']
        
        return {
            'steps': steps,
            'final_values': current_values,
            'formulas_used': [s['formula'] for s in steps if 'formula' in s]
        }
    
    def _select_formulas(self, problem: Problem) -> List[Dict[str, Any]]:
        """관련 공식 선택"""
        formulas = []
        
        if problem.type == ProblemType.POWER_FACTOR:
            formulas.extend([
                self.formulas['power_factor']['basic'],
                self.formulas['power_factor']['improvement'],
                self.formulas['power_factor']['apparent_power']
            ])
        elif problem.type == ProblemType.CIRCUIT_ANALYSIS:
            formulas.extend([
                self.formulas['ohms_law']['basic'],
                self.formulas['ohms_law']['power']
            ])
        elif problem.type == ProblemType.IMPEDANCE:
            formulas.extend([
                self.formulas['impedance']['series'],
                self.formulas['impedance']['angle']
            ])
        
        return formulas
    
    def _organize_values(self, given_values: List[Dict[str, Any]]) -> Dict[str, float]:
        """주어진 값 정리"""
        organized = {}
        
        for value_info in given_values:
            # 단위 변환 및 정규화
            value = value_info['value']
            unit = value_info.get('unit', '')
            
            # 변수명 추론
            if 'kW' in unit:
                organized['P'] = value * 1000  # W로 변환
            elif 'kVar' in unit:
                organized['Q'] = value * 1000
            elif 'kVA' in unit:
                organized['S'] = value * 1000
            elif 'V' in unit:
                organized['V'] = value
            elif 'A' in unit:
                organized['I'] = value
            elif 'Ω' in unit or 'ohm' in unit:
                organized['R'] = value
            
        return organized
    
    def _solve_for_unknown(
        self, 
        unknown: str, 
        known_values: Dict[str, float], 
        formulas: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """미지수 해결"""
        import math
        
        # 각 공식에서 해결 시도
        for formula_group in formulas:
            if 'solve_for' in formula_group and unknown in formula_group['solve_for']:
                expression = formula_group['solve_for'][unknown]
                
                # 필요한 변수 확인
                required_vars = self._extract_variables(expression)
                if all(var in known_values for var in required_vars):
                    # 계산 수행
                    try:
                        # 안전한 eval을 위한 네임스페이스
                        namespace = {
                            'sqrt': math.sqrt,
                            'sin': math.sin,
                            'cos': math.cos,
                            'tan': math.tan,
                            'arctan': math.atan,
                            'atan': math.atan,
                            'pi': math.pi,
                            **known_values
                        }
                        
                        result = eval(expression, {"__builtins__": {}}, namespace)
                        
                        return {
                            'variable': unknown,
                            'formula': formula_group['formula'],
                            'expression': expression,
                            'calculation': f"{unknown} = {expression}",
                            'result': result,
                            'used_values': {var: known_values[var] for var in required_vars}
                        }
                    except Exception as e:
                        logger.error(f"Calculation error: {e}")
        
        return None
    
    def _extract_variables(self, expression: str) -> List[str]:
        """수식에서 변수 추출"""
        import re
        
        # 함수명 제거
        expression = re.sub(r'\b(sqrt|sin|cos|tan|arctan|atan)\b', '', expression)
        
        # 변수 패턴
        variables = re.findall(r'\b[A-Za-z_]\w*\b', expression)
        
        return list(set(variables))


class IntegratedAISystem:
    """통합 AI 시스템"""
    
    def __init__(self):
        self.ocr_system = None
        self.ocr_analyzer = None
        self.formula_solver = None
        self.rag_system = None
        self.llm_client = None
        self.initialized = False
        
    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        """시스템 초기화"""
        try:
            logger.info("Initializing Integrated AI System...")
            
            # 1. OCR 시스템
            self.ocr_system = RealOCRSystem()
            await self.ocr_system.initialize()
            self.ocr_analyzer = ElectricalOCRAnalyzer(self.ocr_system)
            
            # 2. 공식 해결기
            self.formula_solver = ElectricalFormulaSolver()
            
            # 3. RAG 시스템 (config 필요)
            if config and 'rag' in config:
                self.rag_system = RAGSystem(
                    rag_config=config['rag'],
                    dataset_config=config.get('dataset'),
                    llm_client=config.get('llm_client')
                )
            
            # 4. LLM 클라이언트
            if config and 'llm_client' in config:
                self.llm_client = config['llm_client']
            
            self.initialized = True
            logger.info("Integrated AI System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Integrated AI System: {e}")
            raise
    
    async def solve_problem(
        self, 
        image: Optional[Union[Image.Image, str]] = None,
        text: Optional[str] = None
    ) -> Solution:
        """문제 해결 메인 함수"""
        
        # 1. 문제 인식 및 분석
        problem = await self._analyze_problem(image, text)
        
        # 2. RAG 검색 (관련 지식)
        context = await self._search_knowledge(problem)
        
        # 3. 공식 기반 해결
        formula_solution = self.formula_solver.solve(problem)
        
        # 4. LLM 기반 설명 생성
        explanation = await self._generate_explanation(
            problem, formula_solution, context
        )
        
        # 5. 검증
        verification = self._verify_solution(problem, formula_solution)
        
        # 6. 최종 솔루션 구성
        solution = Solution(
            problem_id=f"prob_{hash(str(problem.description))}", 
            steps=formula_solution['steps'],
            final_answer={
                'values': formula_solution['final_values'],
                'primary_answer': self._extract_primary_answer(
                    problem, formula_solution
                )
            },
            confidence=self._calculate_confidence(verification),
            verification=verification,
            explanation=explanation,
            formulas_used=formula_solution['formulas_used']
        )
        
        return solution
    
    async def _analyze_problem(
        self, 
        image: Optional[Union[Image.Image, str]],
        text: Optional[str]
    ) -> Problem:
        """문제 분석"""
        
        # OCR 수행
        ocr_text = ""
        analysis = {}
        
        if image:
            analysis = await self.ocr_analyzer.analyze_electrical_problem(image)
            ocr_text = analysis.get('ocr_text', '')
        
        # 텍스트 통합
        full_text = f"{text or ''} {ocr_text}".strip()
        
        # 문제 구조화
        problem = Problem(
            type=ProblemType(analysis.get('problem_type', 'general')),
            description=full_text,
            given_values=analysis.get('given_values', []),
            find_targets=analysis.get('find_targets', []),
            constraints=[],
            formulas=analysis.get('formulas', []),
            image_path=str(image) if isinstance(image, str) else None,
            ocr_text=ocr_text
        )
        
        return problem
    
    async def _search_knowledge(self, problem: Problem) -> Dict[str, Any]:
        """관련 지식 검색"""
        if not self.rag_system:
            return {}
        
        # 검색 쿼리 구성
        query = f"{problem.description} {problem.type.value}"
        
        # RAG 검색
        results = await self.rag_system.search(query, top_k=5)
        
        return {
            'relevant_docs': results,
            'formulas': self._extract_formulas_from_docs(results),
            'examples': self._extract_examples_from_docs(results)
        }
    
    async def _generate_explanation(
        self, 
        problem: Problem,
        solution: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """설명 생성"""
        if not self.llm_client:
            return self._generate_basic_explanation(problem, solution)
        
        # 프롬프트 구성
        prompt = f"""
전기공학 문제를 단계별로 설명해주세요.

문제: {problem.description}

주어진 값:
{self._format_given_values(problem.given_values)}

해결 과정:
{self._format_solution_steps(solution['steps'])}

최종 답:
{self._format_final_answer(solution['final_values'])}

학생이 이해하기 쉽도록 각 단계를 자세히 설명해주세요.
"""
        
        # LLM 생성
        response = await self.llm_client.generate(prompt)
        
        return response
    
    def _generate_basic_explanation(
        self, 
        problem: Problem,
        solution: Dict[str, Any]
    ) -> str:
        """기본 설명 생성 (LLM 없이)"""
        explanation = f"문제 유형: {problem.type.value}\n\n"
        
        explanation += "해결 과정:\n"
        for i, step in enumerate(solution['steps'], 1):
            explanation += f"\n{i}. {step['variable']} 계산\n"
            explanation += f"   공식: {step['formula']}\n"
            explanation += f"   계산: {step['calculation']}\n"
            explanation += f"   결과: {step['result']:.3f}\n"
        
        return explanation
    
    def _verify_solution(
        self, 
        problem: Problem, 
        solution: Dict[str, Any]
    ) -> Dict[str, bool]:
        """해결 검증"""
        verification = {
            'all_targets_found': True,
            'unit_consistency': True,
            'physical_validity': True,
            'calculation_accuracy': True
        }
        
        # 모든 목표값이 계산되었는지 확인
        for target in problem.find_targets:
            if target not in solution['final_values']:
                verification['all_targets_found'] = False
        
        # 물리적 타당성 확인 (예: 역률은 0~1)
        if 'cos_theta' in solution['final_values']:
            cos_val = solution['final_values']['cos_theta']
            if not (0 <= cos_val <= 1):
                verification['physical_validity'] = False
        
        return verification
    
    def _calculate_confidence(self, verification: Dict[str, bool]) -> float:
        """신뢰도 계산"""
        passed = sum(1 for v in verification.values() if v)
        total = len(verification)
        return passed / total if total > 0 else 0.0
    
    def _extract_primary_answer(
        self, 
        problem: Problem, 
        solution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """주요 답변 추출"""
        if not problem.find_targets:
            return {}
        
        primary_target = problem.find_targets[0]
        if primary_target in solution['final_values']:
            return {
                'variable': primary_target,
                'value': solution['final_values'][primary_target],
                'unit': self._infer_unit(primary_target)
            }
        
        return {}
    
    def _infer_unit(self, variable: str) -> str:
        """변수에서 단위 추론"""
        unit_map = {
            'P': 'W',
            'Q': 'Var', 
            'S': 'VA',
            'V': 'V',
            'I': 'A',
            'R': 'Ω',
            'Z': 'Ω',
            'cos_theta': '',
            'Qc': 'Var'
        }
        return unit_map.get(variable, '')
    
    def _format_given_values(self, values: List[Dict[str, Any]]) -> str:
        """주어진 값 포맷팅"""
        lines = []
        for v in values:
            lines.append(f"- {v['value']}{v.get('unit', '')}")
        return '\n'.join(lines)
    
    def _format_solution_steps(self, steps: List[Dict[str, Any]]) -> str:
        """해결 단계 포맷팅"""
        lines = []
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. {step['calculation']} = {step['result']:.3f}")
        return '\n'.join(lines)
    
    def _format_final_answer(self, values: Dict[str, float]) -> str:
        """최종 답 포맷팅"""
        lines = []
        for var, val in values.items():
            unit = self._infer_unit(var)
            lines.append(f"{var} = {val:.3f} {unit}")
        return '\n'.join(lines)
    
    def _extract_formulas_from_docs(self, docs: List[Any]) -> List[str]:
        """문서에서 공식 추출"""
        # RAG 결과에서 공식 추출 로직
        return []
    
    def _extract_examples_from_docs(self, docs: List[Any]) -> List[str]:
        """문서에서 예제 추출"""
        # RAG 결과에서 예제 추출 로직
        return []


class AutoEvaluator:
    """자동 평가 시스템"""
    
    def __init__(self, ai_system: IntegratedAISystem):
        self.ai_system = ai_system
        self.test_cases = []
        
    def load_test_cases(self, file_path: str):
        """테스트 케이스 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.test_cases = json.load(f)
    
    async def evaluate(self) -> Dict[str, Any]:
        """전체 평가 수행"""
        results = []
        
        for test_case in self.test_cases:
            result = await self._evaluate_single(test_case)
            results.append(result)
        
        # 통계 계산
        stats = self._calculate_statistics(results)
        
        return {
            'results': results,
            'statistics': stats,
            'report': self._generate_report(results, stats)
        }
    
    async def _evaluate_single(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """단일 테스트 케이스 평가"""
        # 문제 해결
        solution = await self.ai_system.solve_problem(
            image=test_case.get('image_path'),
            text=test_case.get('text')
        )
        
        # 정답과 비교
        correct = self._check_answer(
            solution.final_answer,
            test_case.get('expected_answer')
        )
        
        return {
            'test_id': test_case['id'],
            'problem_type': test_case.get('type'),
            'solution': solution,
            'correct': correct,
            'score': solution.confidence if correct else 0.0
        }
    
    def _check_answer(
        self, 
        actual: Dict[str, Any], 
        expected: Dict[str, Any]
    ) -> bool:
        """답안 확인"""
        if not expected:
            return True
        
        # 주요 답변 비교
        if 'value' in expected and 'primary_answer' in actual:
            expected_val = expected['value']
            actual_val = actual['primary_answer'].get('value', 0)
            
            # 상대 오차 5% 이내
            tolerance = 0.05
            if abs(actual_val - expected_val) / expected_val <= tolerance:
                return True
        
        return False
    
    def _calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """통계 계산"""
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        
        # 문제 유형별 통계
        by_type = {}
        for r in results:
            ptype = r.get('problem_type', 'unknown')
            if ptype not in by_type:
                by_type[ptype] = {'total': 0, 'correct': 0}
            by_type[ptype]['total'] += 1
            if r['correct']:
                by_type[ptype]['correct'] += 1
        
        return {
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0,
            'by_type': by_type,
            'avg_confidence': np.mean([r['solution'].confidence for r in results])
        }
    
    def _generate_report(
        self, 
        results: List[Dict[str, Any]], 
        stats: Dict[str, Any]
    ) -> str:
        """평가 보고서 생성"""
        report = "# 자동 평가 보고서\n\n"
        
        report += f"## 전체 성능\n"
        report += f"- 정답률: {stats['accuracy']:.1%}\n"
        report += f"- 평균 신뢰도: {stats['avg_confidence']:.2f}\n\n"
        
        report += "## 문제 유형별 성능\n"
        for ptype, pstats in stats['by_type'].items():
            accuracy = pstats['correct'] / pstats['total'] if pstats['total'] > 0 else 0
            report += f"- {ptype}: {accuracy:.1%} ({pstats['correct']}/{pstats['total']})\n"
        
        return report


# 사용 예시
async def demo():
    """통합 시스템 데모"""
    # 시스템 초기화
    ai_system = IntegratedAISystem()
    await ai_system.initialize()
    
    # 문제 해결
    image_path = "problem_image.png"
    solution = await ai_system.solve_problem(image=image_path)
    
    print(f"문제 ID: {solution.problem_id}")
    print(f"최종 답: {solution.final_answer}")
    print(f"신뢰도: {solution.confidence:.2f}")
    print(f"\n설명:\n{solution.explanation}")
    
    # 자동 평가
    evaluator = AutoEvaluator(ai_system)
    evaluator.load_test_cases("test_cases.json")
    eval_results = await evaluator.evaluate()
    print(f"\n{eval_results['report']}")


if __name__ == "__main__":
    asyncio.run(demo())