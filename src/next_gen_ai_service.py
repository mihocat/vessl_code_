#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
차세대 AI 서비스
최신 AI 트렌드 통합: 에이전트, 추론 체인, 메모리, 도구 사용
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from collections import deque

from intelligent_multimodal_system import IntelligentMultimodalSystem, ProcessingResult
from query_intent_analyzer import QueryIntentAnalyzer, IntentAnalysisResult, QueryType, ComplexityLevel
from enhanced_rag_system import EnhancedRAGSystem, SearchStrategy

logger = logging.getLogger(__name__)


class ConversationMode(Enum):
    """대화 모드"""
    STANDARD = "standard"        # 일반 모드
    EXPERT = "expert"           # 전문가 모드 (상세한 기술 정보)
    TUTORIAL = "tutorial"       # 튜토리얼 모드 (학습 지원)
    RESEARCH = "research"       # 연구 모드 (심층 분석)
    COLLABORATIVE = "collaborative"  # 협업 모드 (다단계 상호작용)


class ResponseStyle(Enum):
    """응답 스타일"""
    CONCISE = "concise"         # 간결한 답변
    DETAILED = "detailed"       # 상세한 답변
    STEP_BY_STEP = "step_by_step"  # 단계별 설명
    VISUAL = "visual"           # 시각적 요소 포함
    INTERACTIVE = "interactive"  # 상호작용적 답변


@dataclass
class UserContext:
    """사용자 컨텍스트"""
    user_id: str
    session_id: str
    conversation_mode: ConversationMode
    response_style: ResponseStyle
    expertise_level: int  # 1-5 (초보자-전문가)
    
    # 개인화 정보
    preferred_language: str
    domain_interests: List[str]
    learning_goals: List[str]
    
    # 대화 기록
    conversation_history: List[Dict[str, Any]]
    interaction_count: int
    success_rate: float


@dataclass
class ConversationMemory:
    """대화 메모리"""
    short_term: deque  # 최근 5-10개 대화
    semantic_memory: Dict[str, Any]  # 주제별 기억
    episodic_memory: List[Dict[str, Any]]  # 특정 상황 기억
    procedural_memory: Dict[str, List[str]]  # 절차적 지식
    
    # 메타 정보
    memory_strength: Dict[str, float]  # 기억 강도
    access_frequency: Dict[str, int]   # 접근 빈도
    last_updated: Dict[str, float]     # 마지막 업데이트


class ChainOfThoughtReasoner:
    """추론 체인 처리기"""
    
    def __init__(self):
        self.reasoning_templates = self._load_reasoning_templates()
    
    def _load_reasoning_templates(self):
        """추론 템플릿 로드"""
        return {
            'problem_solving': [
                "1. 문제 이해 및 정의",
                "2. 관련 정보 수집",
                "3. 해결 방법 탐색",
                "4. 단계별 해결 과정",
                "5. 결과 검증 및 해석"
            ],
            'analysis': [
                "1. 주요 요소 식별",
                "2. 관계성 분석",
                "3. 패턴 및 특징 파악",
                "4. 원인과 결과 관계",
                "5. 결론 및 시사점"
            ],
            'comparison': [
                "1. 비교 대상 정의",
                "2. 비교 기준 설정",
                "3. 각 요소별 분석",
                "4. 유사점과 차이점",
                "5. 종합적 평가"
            ],
            'explanation': [
                "1. 핵심 개념 정의",
                "2. 기본 원리 설명",
                "3. 구체적 예시 제시",
                "4. 응용 및 확장",
                "5. 요약 및 정리"
            ]
        }
    
    def reason(self, query: str, context: Dict[str, Any], reasoning_type: str = 'problem_solving') -> List[str]:
        """추론 체인 생성"""
        template = self.reasoning_templates.get(reasoning_type, self.reasoning_templates['problem_solving'])
        
        # 쿼리와 컨텍스트에 맞춰 추론 단계 구체화
        reasoning_steps = []
        
        for step in template:
            # 각 단계를 쿼리에 맞게 구체화
            specific_step = self._contextualize_step(step, query, context)
            reasoning_steps.append(specific_step)
        
        return reasoning_steps
    
    def _contextualize_step(self, step: str, query: str, context: Dict[str, Any]) -> str:
        """추론 단계를 쿼리에 맞게 구체화"""
        # 간단한 구현: 실제로는 더 정교한 LLM 기반 처리
        if "문제 이해" in step:
            return f"문제 이해: '{query}' 분석"
        elif "정보 수집" in step:
            return f"관련 정보 수집: {query}와 관련된 데이터 및 지식"
        elif "해결 방법" in step:
            return f"해결 방법 탐색: {query}에 대한 접근 방식"
        else:
            return step


class ToolAgent:
    """도구 사용 에이전트"""
    
    def __init__(self):
        self.available_tools = self._load_available_tools()
    
    def _load_available_tools(self):
        """사용 가능한 도구 로드"""
        return {
            'calculator': {
                'description': '수학 계산 수행',
                'usage': 'mathematical expressions, equations',
                'examples': ['2+2', 'sqrt(16)', 'sin(30)', 'solve x^2 + 2x - 3 = 0']
            },
            'unit_converter': {
                'description': '단위 변환',
                'usage': 'unit conversions',
                'examples': ['convert 100 km to miles', '50 celsius to fahrenheit']
            },
            'formula_solver': {
                'description': '공식 및 방정식 해결',
                'usage': 'formula evaluation, equation solving',
                'examples': ['ohms law V=IR', 'quadratic formula']
            },
            'web_search': {
                'description': '실시간 웹 검색',
                'usage': 'current information, recent data',
                'examples': ['latest research on AI', 'current exchange rates']
            },
            'code_executor': {
                'description': '코드 실행',
                'usage': 'programming calculations, simulations',
                'examples': ['python script', 'data analysis']
            }
        }
    
    def select_tools(self, query: str, intent: IntentAnalysisResult) -> List[str]:
        """쿼리에 적합한 도구 선택"""
        selected_tools = []
        
        # 수학적 계산이 필요한 경우
        if intent.requires_calculation or any(op in query for op in ['+', '-', '*', '/', '=', '계산', 'calculate']):
            selected_tools.append('calculator')
            selected_tools.append('formula_solver')
        
        # 단위 변환이 필요한 경우
        if any(unit in query.lower() for unit in ['km', 'mile', 'celsius', 'fahrenheit', 'volt', 'ampere']):
            selected_tools.append('unit_converter')
        
        # 최신 정보가 필요한 경우
        if any(term in query.lower() for term in ['최신', '현재', '요즘', 'latest', 'current', 'recent']):
            selected_tools.append('web_search')
        
        # 프로그래밍 관련 질의
        if any(term in query.lower() for term in ['코드', 'code', '프로그래밍', 'programming', 'python']):
            selected_tools.append('code_executor')
        
        return selected_tools
    
    async def use_tool(self, tool_name: str, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """도구 사용"""
        if tool_name not in self.available_tools:
            return {'success': False, 'error': f'Tool {tool_name} not available'}
        
        try:
            # 실제 구현에서는 각 도구별 API 호출
            # 현재는 시뮬레이션
            if tool_name == 'calculator':
                return await self._use_calculator(query, parameters)
            elif tool_name == 'unit_converter':
                return await self._use_unit_converter(query, parameters)
            elif tool_name == 'formula_solver':
                return await self._use_formula_solver(query, parameters)
            elif tool_name == 'web_search':
                return await self._use_web_search(query, parameters)
            elif tool_name == 'code_executor':
                return await self._use_code_executor(query, parameters)
            else:
                return {'success': False, 'error': 'Tool implementation not found'}
                
        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _use_calculator(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """계산기 사용"""
        # 간단한 계산 구현
        import re
        
        # 수식 추출
        math_expressions = re.findall(r'[\d+\-*/().\s]+', query)
        results = []
        
        for expr in math_expressions:
            try:
                # 안전한 계산을 위한 eval 대안 사용 (실제로는 ast.literal_eval 등 사용)
                if len(expr.strip()) > 3 and all(c in '0123456789+-*/.() ' for c in expr):
                    result = eval(expr.strip())
                    results.append(f"{expr.strip()} = {result}")
            except:
                continue
        
        return {
            'success': True,
            'results': results,
            'tool': 'calculator'
        }
    
    async def _use_unit_converter(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """단위 변환기 사용"""
        # 간단한 단위 변환 구현
        conversions = {
            'km_to_mile': lambda x: x * 0.621371,
            'mile_to_km': lambda x: x * 1.60934,
            'celsius_to_fahrenheit': lambda x: (x * 9/5) + 32,
            'fahrenheit_to_celsius': lambda x: (x - 32) * 5/9
        }
        
        results = []
        if 'km' in query and 'mile' in query:
            # 숫자 추출 및 변환 (간단한 구현)
            import re
            numbers = re.findall(r'\d+', query)
            if numbers:
                km = float(numbers[0])
                miles = conversions['km_to_mile'](km)
                results.append(f"{km} km = {miles:.2f} miles")
        
        return {
            'success': True,
            'results': results,
            'tool': 'unit_converter'
        }
    
    async def _use_formula_solver(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """공식 해결기 사용"""
        # 기본적인 과학 공식들
        formulas = {
            'ohms_law': {
                'V=IR': lambda I, R: I * R,
                'I=V/R': lambda V, R: V / R,
                'R=V/I': lambda V, I: V / I
            },
            'power': {
                'P=VI': lambda V, I: V * I,
                'P=V²/R': lambda V, R: (V ** 2) / R,
                'P=I²R': lambda I, R: (I ** 2) * R
            }
        }
        
        results = []
        if any(term in query.lower() for term in ['옴의법칙', 'ohm', 'V=IR', 'I=V/R']):
            results.append("옴의 법칙: V = I × R (전압 = 전류 × 저항)")
            results.append("변형: I = V / R, R = V / I")
        
        return {
            'success': True,
            'results': results,
            'tool': 'formula_solver'
        }
    
    async def _use_web_search(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """웹 검색 사용"""
        # 실제로는 웹 검색 API 호출
        return {
            'success': True,
            'results': [f"웹 검색 결과: '{query}'에 대한 최신 정보를 찾았습니다."],
            'tool': 'web_search'
        }
    
    async def _use_code_executor(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """코드 실행기 사용"""
        # 실제로는 안전한 코드 실행 환경
        return {
            'success': True,
            'results': [f"코드 실행 결과: '{query}' 처리 완료"],
            'tool': 'code_executor'
        }


class AdaptiveResponseGenerator:
    """적응형 응답 생성기"""
    
    def __init__(self):
        self.response_templates = self._load_response_templates()
    
    def _load_response_templates(self):
        """응답 템플릿 로드"""
        return {
            ConversationMode.STANDARD: {
                ResponseStyle.CONCISE: "간결하고 명확한 답변을 제공합니다.",
                ResponseStyle.DETAILED: "상세한 설명과 함께 포괄적인 답변을 제공합니다.",
                ResponseStyle.STEP_BY_STEP: "단계별로 차근차근 설명드리겠습니다."
            },
            ConversationMode.EXPERT: {
                ResponseStyle.CONCISE: "전문적인 관점에서 핵심 내용을 제시합니다.",
                ResponseStyle.DETAILED: "기술적 세부사항과 이론적 배경을 포함하여 설명합니다.",
                ResponseStyle.STEP_BY_STEP: "각 단계의 이론적 근거와 함께 상세히 분석합니다."
            },
            ConversationMode.TUTORIAL: {
                ResponseStyle.CONCISE: "학습에 필요한 핵심 포인트를 제시합니다.",
                ResponseStyle.DETAILED: "개념 설명, 예시, 연습 문제를 포함한 학습 자료를 제공합니다.",
                ResponseStyle.STEP_BY_STEP: "단계별 학습 과정과 이해도 확인을 포함합니다."
            }
        }
    
    def generate_response(self, processing_result: ProcessingResult, user_context: UserContext,
                         reasoning_steps: List[str] = None, tool_results: List[Dict] = None) -> str:
        """사용자 컨텍스트에 맞는 응답 생성"""
        
        # 기본 응답
        base_response = processing_result.response
        
        # 사용자 레벨에 맞는 조정
        adjusted_response = self._adjust_for_expertise_level(
            base_response, user_context.expertise_level
        )
        
        # 대화 모드에 맞는 스타일 적용
        styled_response = self._apply_conversation_style(
            adjusted_response, user_context.conversation_mode, user_context.response_style
        )
        
        # 추론 과정 추가 (필요한 경우)
        if reasoning_steps and user_context.conversation_mode in [ConversationMode.EXPERT, ConversationMode.TUTORIAL]:
            styled_response = self._add_reasoning_explanation(styled_response, reasoning_steps)
        
        # 도구 사용 결과 통합
        if tool_results:
            styled_response = self._integrate_tool_results(styled_response, tool_results)
        
        # 개인화 요소 추가
        personalized_response = self._add_personalization(styled_response, user_context)
        
        return personalized_response
    
    def _adjust_for_expertise_level(self, response: str, level: int) -> str:
        """전문성 수준에 맞게 응답 조정"""
        if level <= 2:  # 초보자
            return f"🔰 초보자를 위한 설명:\n{response}\n\n💡 추가 학습 팁: 관련 기초 개념을 더 공부해보세요."
        elif level >= 4:  # 전문가
            return f"🎓 전문가 수준 분석:\n{response}\n\n🔬 심화 고려사항: 최신 연구 동향과 고급 응용 방안을 검토해보세요."
        else:  # 중급자
            return response
    
    def _apply_conversation_style(self, response: str, mode: ConversationMode, style: ResponseStyle) -> str:
        """대화 스타일 적용"""
        style_prefix = ""
        style_suffix = ""
        
        if mode == ConversationMode.TUTORIAL:
            style_prefix = "📚 학습 가이드:\n"
            style_suffix = "\n\n✅ 이해도 확인: 핵심 개념을 자신의 말로 설명해보세요."
        elif mode == ConversationMode.RESEARCH:
            style_prefix = "🔬 연구 분석:\n"
            style_suffix = "\n\n📊 추가 연구 방향: 관련 논문이나 최신 연구를 참고해보세요."
        elif mode == ConversationMode.COLLABORATIVE:
            style_prefix = "🤝 협업 모드:\n"
            style_suffix = "\n\n💬 피드백 요청: 추가 질문이나 다른 관점이 있으신가요?"
        
        if style == ResponseStyle.STEP_BY_STEP:
            # 단계별 형식으로 재구성
            lines = response.split('\n')
            numbered_lines = []
            step = 1
            for line in lines:
                if line.strip():
                    numbered_lines.append(f"{step}. {line}")
                    step += 1
                else:
                    numbered_lines.append(line)
            response = '\n'.join(numbered_lines)
        
        return f"{style_prefix}{response}{style_suffix}"
    
    def _add_reasoning_explanation(self, response: str, reasoning_steps: List[str]) -> str:
        """추론 과정 설명 추가"""
        reasoning_text = "\n\n🧠 사고 과정:\n"
        for i, step in enumerate(reasoning_steps, 1):
            reasoning_text += f"{i}. {step}\n"
        
        return f"{response}{reasoning_text}"
    
    def _integrate_tool_results(self, response: str, tool_results: List[Dict]) -> str:
        """도구 사용 결과 통합"""
        if not tool_results:
            return response
        
        tools_text = "\n\n🛠️ 도구 사용 결과:\n"
        for result in tool_results:
            if result.get('success'):
                tool_name = result.get('tool', 'Unknown')
                results = result.get('results', [])
                tools_text += f"📋 {tool_name}: {', '.join(results)}\n"
        
        return f"{response}{tools_text}"
    
    def _add_personalization(self, response: str, user_context: UserContext) -> str:
        """개인화 요소 추가"""
        # 사용자 관심 도메인 반영
        if user_context.domain_interests:
            domain = user_context.domain_interests[0]
            response += f"\n\n🎯 {domain} 관련 추가 정보가 필요하시면 언제든 문의해주세요."
        
        # 성공률에 따른 격려 메시지
        if user_context.success_rate < 0.7:
            response += f"\n\n💪 질문이 더 있으시면 자세히 설명드리겠습니다!"
        
        return response


class NextGenAIService:
    """차세대 AI 서비스"""
    
    def __init__(self, config):
        """서비스 초기화"""
        self.config = config
        
        # 핵심 시스템들
        self.multimodal_system = IntelligentMultimodalSystem(config)
        self.intent_analyzer = QueryIntentAnalyzer()
        self.enhanced_rag = EnhancedRAGSystem(config)
        
        # AI 트렌드 컴포넌트들
        self.reasoning_engine = ChainOfThoughtReasoner()
        self.tool_agent = ToolAgent()
        self.response_generator = AdaptiveResponseGenerator()
        
        # 메모리 시스템
        self.conversation_memories = {}  # user_id -> ConversationMemory
        
        # 서비스 통계
        self.service_stats = {
            'total_conversations': 0,
            'successful_interactions': 0,
            'average_satisfaction': 0.0,
            'tool_usage_count': 0,
            'memory_efficiency': 0.0
        }
        
        logger.info("Next-Generation AI Service initialized")
    
    async def process_conversation(self, query: str, image_path: str = None, 
                                 user_context: UserContext = None) -> Dict[str, Any]:
        """
        차세대 대화 처리
        
        Args:
            query: 사용자 질의
            image_path: 이미지 경로 (선택적)
            user_context: 사용자 컨텍스트
            
        Returns:
            처리 결과
        """
        start_time = time.time()
        
        try:
            self.service_stats['total_conversations'] += 1
            
            # 기본 사용자 컨텍스트 생성
            if not user_context:
                user_context = self._create_default_user_context()
            
            # 대화 메모리 로드
            memory = self._get_or_create_memory(user_context.user_id)
            
            # 1단계: 의도 분석
            intent_result = self.intent_analyzer.analyze_intent(query, has_image=bool(image_path))
            
            # 2단계: 메모리 기반 컨텍스트 확장
            enhanced_query = self._enhance_query_with_memory(query, memory, intent_result)
            
            # 3단계: 추론 체인 생성 (복잡한 문제의 경우)
            reasoning_steps = None
            if intent_result.complexity.value >= 3:
                reasoning_type = self._determine_reasoning_type(intent_result)
                reasoning_steps = self.reasoning_engine.reason(
                    enhanced_query, 
                    {'intent': intent_result, 'memory': memory}, 
                    reasoning_type
                )
            
            # 4단계: 도구 선택 및 사용
            tool_results = []
            if intent_result.requires_calculation or intent_result.complexity.value >= 4:
                selected_tools = self.tool_agent.select_tools(enhanced_query, intent_result)
                tool_results = await self._use_tools_parallel(selected_tools, enhanced_query)
            
            # 5단계: 멀티모달 처리
            processing_result = self.multimodal_system.process_query(
                enhanced_query, 
                image_path, 
                user_context=asdict(user_context)
            )
            
            # 6단계: 적응형 응답 생성
            final_response = self.response_generator.generate_response(
                processing_result, 
                user_context,
                reasoning_steps,
                tool_results
            )
            
            # 7단계: 메모리 업데이트
            self._update_memory(memory, query, final_response, processing_result, user_context)
            
            # 8단계: 사용자 컨텍스트 업데이트
            updated_context = self._update_user_context(user_context, processing_result, intent_result)
            
            processing_time = time.time() - start_time
            
            # 결과 구성
            result = {
                'response': final_response,
                'processing_time': processing_time,
                'confidence': processing_result.confidence,
                'intent_analysis': asdict(intent_result),
                'reasoning_steps': reasoning_steps,
                'tools_used': [r.get('tool') for r in tool_results if r.get('success')],
                'memory_updated': True,
                'user_context': asdict(updated_context),
                'service_metadata': {
                    'response_quality': processing_result.response_quality_score,
                    'user_satisfaction_prediction': processing_result.user_satisfaction_prediction,
                    'tokens_used': processing_result.tokens_used,
                    'cost_estimate': processing_result.cost_estimate
                }
            }
            
            # 통계 업데이트
            self._update_service_stats(result)
            
            logger.info(f"Next-gen conversation completed in {processing_time:.2f}s "
                       f"(confidence: {processing_result.confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Next-gen conversation processing failed: {e}")
            return self._generate_error_response(str(e), time.time() - start_time)
    
    def _create_default_user_context(self) -> UserContext:
        """기본 사용자 컨텍스트 생성"""
        return UserContext(
            user_id="anonymous",
            session_id=f"session_{int(time.time())}",
            conversation_mode=ConversationMode.STANDARD,
            response_style=ResponseStyle.DETAILED,
            expertise_level=3,
            preferred_language="korean",
            domain_interests=["general"],
            learning_goals=[],
            conversation_history=[],
            interaction_count=0,
            success_rate=0.8
        )
    
    def _get_or_create_memory(self, user_id: str) -> ConversationMemory:
        """대화 메모리 획득 또는 생성"""
        if user_id not in self.conversation_memories:
            self.conversation_memories[user_id] = ConversationMemory(
                short_term=deque(maxlen=10),
                semantic_memory={},
                episodic_memory=[],
                procedural_memory={},
                memory_strength={},
                access_frequency={},
                last_updated={}
            )
        
        return self.conversation_memories[user_id]
    
    def _enhance_query_with_memory(self, query: str, memory: ConversationMemory, 
                                  intent: IntentAnalysisResult) -> str:
        """메모리를 활용한 쿼리 향상"""
        enhanced_parts = [query]
        
        # 최근 대화 컨텍스트 추가
        if memory.short_term:
            recent_topics = []
            for conversation in list(memory.short_term)[-3:]:  # 최근 3개
                if conversation.get('topic'):
                    recent_topics.append(conversation['topic'])
            
            if recent_topics:
                enhanced_parts.append(f"최근 대화 주제: {', '.join(recent_topics)}")
        
        # 의미적 메모리에서 관련 정보 추가
        for topic, info in memory.semantic_memory.items():
            if any(keyword in query.lower() for keyword in topic.split()):
                enhanced_parts.append(f"관련 기억: {info}")
                break
        
        return " | ".join(enhanced_parts)
    
    def _determine_reasoning_type(self, intent: IntentAnalysisResult) -> str:
        """추론 유형 결정"""
        if intent.query_type == QueryType.PROBLEM_SOLVING:
            return 'problem_solving'
        elif intent.query_type == QueryType.COMPARISON:
            return 'comparison'
        elif intent.query_type == QueryType.EXPLANATION:
            return 'explanation'
        else:
            return 'analysis'
    
    async def _use_tools_parallel(self, selected_tools: List[str], query: str) -> List[Dict[str, Any]]:
        """도구들을 병렬로 사용"""
        if not selected_tools:
            return []
        
        # 병렬 도구 실행
        tasks = []
        for tool in selected_tools[:3]:  # 최대 3개 도구
            task = self.tool_agent.use_tool(tool, query, {})
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 성공한 결과만 반환
        successful_results = []
        for result in results:
            if isinstance(result, dict) and result.get('success'):
                successful_results.append(result)
                self.service_stats['tool_usage_count'] += 1
        
        return successful_results
    
    def _update_memory(self, memory: ConversationMemory, query: str, response: str,
                      processing_result: ProcessingResult, user_context: UserContext):
        """메모리 업데이트"""
        current_time = time.time()
        
        # 단기 메모리 업데이트
        conversation_entry = {
            'timestamp': current_time,
            'query': query,
            'response': response,
            'topic': self._extract_topic(query),
            'confidence': processing_result.confidence,
            'user_satisfaction': processing_result.user_satisfaction_prediction
        }
        memory.short_term.append(conversation_entry)
        
        # 의미적 메모리 업데이트
        topic = self._extract_topic(query)
        if topic:
            if topic in memory.semantic_memory:
                # 기존 정보 강화
                memory.memory_strength[topic] = memory.memory_strength.get(topic, 0.5) + 0.1
            else:
                # 새 주제 추가
                memory.semantic_memory[topic] = response[:200]  # 처음 200자만 저장
                memory.memory_strength[topic] = 0.7
            
            memory.access_frequency[topic] = memory.access_frequency.get(topic, 0) + 1
            memory.last_updated[topic] = current_time
        
        # 메모리 정리 (너무 많아지면)
        if len(memory.semantic_memory) > 50:
            self._cleanup_memory(memory)
    
    def _extract_topic(self, query: str) -> Optional[str]:
        """쿼리에서 주제 추출"""
        # 간단한 키워드 기반 주제 추출
        keywords = query.split()[:3]  # 처음 3개 단어
        return ' '.join(keywords) if keywords else None
    
    def _cleanup_memory(self, memory: ConversationMemory):
        """메모리 정리"""
        current_time = time.time()
        
        # 1주일 이상 접근하지 않은 약한 기억 제거
        topics_to_remove = []
        for topic, last_update in memory.last_updated.items():
            if (current_time - last_update > 604800 and  # 1주일
                memory.memory_strength.get(topic, 0) < 0.3):
                topics_to_remove.append(topic)
        
        for topic in topics_to_remove:
            memory.semantic_memory.pop(topic, None)
            memory.memory_strength.pop(topic, None)
            memory.access_frequency.pop(topic, None)
            memory.last_updated.pop(topic, None)
    
    def _update_user_context(self, user_context: UserContext, 
                           processing_result: ProcessingResult,
                           intent_result: IntentAnalysisResult) -> UserContext:
        """사용자 컨텍스트 업데이트"""
        # 상호작용 카운트 증가
        user_context.interaction_count += 1
        
        # 성공률 업데이트
        current_success = 1.0 if processing_result.confidence > 0.7 else 0.5
        user_context.success_rate = (
            (user_context.success_rate * (user_context.interaction_count - 1) + current_success) 
            / user_context.interaction_count
        )
        
        # 전문성 수준 조정 (복잡한 질문을 계속하면 레벨 상승)
        if intent_result.complexity.value >= 4 and user_context.expertise_level < 5:
            user_context.expertise_level = min(5, user_context.expertise_level + 0.1)
        
        # 도메인 관심사 업데이트
        if intent_result.domain and intent_result.domain not in user_context.domain_interests:
            user_context.domain_interests.append(intent_result.domain)
            user_context.domain_interests = user_context.domain_interests[-5:]  # 최대 5개
        
        return user_context
    
    def _update_service_stats(self, result: Dict[str, Any]):
        """서비스 통계 업데이트"""
        if result.get('confidence', 0) > 0.7:
            self.service_stats['successful_interactions'] += 1
        
        # 평균 만족도 업데이트
        satisfaction = result.get('service_metadata', {}).get('user_satisfaction_prediction', 0.5)
        total = self.service_stats['total_conversations']
        current_avg = self.service_stats['average_satisfaction']
        self.service_stats['average_satisfaction'] = (
            (current_avg * (total - 1) + satisfaction) / total
        )
    
    def _generate_error_response(self, error: str, processing_time: float) -> Dict[str, Any]:
        """오류 응답 생성"""
        return {
            'response': f"죄송합니다. 처리 중 오류가 발생했습니다: {error}",
            'processing_time': processing_time,
            'confidence': 0.0,
            'error': True,
            'service_metadata': {
                'response_quality': 0.0,
                'user_satisfaction_prediction': 0.2
            }
        }
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """서비스 통계 반환"""
        return self.service_stats.copy()
    
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """사용자 분석 정보 반환"""
        memory = self.conversation_memories.get(user_id)
        if not memory:
            return {'error': 'User not found'}
        
        return {
            'conversation_count': len(memory.short_term),
            'topics_discussed': len(memory.semantic_memory),
            'most_frequent_topics': sorted(
                memory.access_frequency.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            'memory_strength_average': sum(memory.memory_strength.values()) / len(memory.memory_strength) if memory.memory_strength else 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        components = {
            'multimodal_system': True,
            'intent_analyzer': True,
            'enhanced_rag': True,
            'reasoning_engine': True,
            'tool_agent': True,
            'response_generator': True
        }
        
        # 각 컴포넌트 상태 확인
        try:
            multimodal_health = self.multimodal_system.health_check()
            components['multimodal_system'] = multimodal_health['status'] == 'healthy'
        except:
            components['multimodal_system'] = False
        
        overall_health = all(components.values())
        
        return {
            'status': 'healthy' if overall_health else 'degraded',
            'components': components,
            'statistics': self.get_service_statistics(),
            'memory_usage': len(self.conversation_memories)
        }