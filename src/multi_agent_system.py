#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Agent System for Advanced AI Capabilities
다중 에이전트 시스템 - 최신 AI 기능 통합
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import json

from intent_analyzer import IntentAnalysisResult, QueryType, ProcessingMode
from enhanced_rag_system import EnhancedSearchResult

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """에이전트 역할"""
    ORCHESTRATOR = "orchestrator"        # 총괄 조정자
    VISION_ANALYST = "vision_analyst"    # 시각 분석 전문가
    RAG_SPECIALIST = "rag_specialist"    # RAG 검색 전문가
    REASONING_ENGINE = "reasoning_engine" # 추론 엔진
    MATH_SOLVER = "math_solver"          # 수학 문제 해결사
    DOMAIN_EXPERT = "domain_expert"      # 도메인 전문가
    SYNTHESIZER = "synthesizer"          # 응답 합성기
    VALIDATOR = "validator"              # 검증자


@dataclass
class AgentMessage:
    """에이전트 간 메시지"""
    sender: AgentRole
    recipient: AgentRole
    message_type: str
    content: Any
    metadata: Dict[str, Any]
    timestamp: float


@dataclass
class AgentResult:
    """에이전트 작업 결과"""
    agent_role: AgentRole
    success: bool
    result: Any
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]
    reasoning_trace: List[str]


class BaseAgent(ABC):
    """기본 에이전트 클래스"""
    
    def __init__(self, role: AgentRole, config: Dict = None):
        self.role = role
        self.config = config or {}
        self.message_queue = asyncio.Queue()
        self.active = True
        
    @abstractmethod
    async def process(self, task: Dict[str, Any]) -> AgentResult:
        """작업 처리 (하위 클래스에서 구현)"""
        pass
    
    async def send_message(self, recipient: AgentRole, message_type: str, content: Any):
        """다른 에이전트에게 메시지 전송"""
        message = AgentMessage(
            sender=self.role,
            recipient=recipient,
            message_type=message_type,
            content=content,
            metadata={},
            timestamp=time.time()
        )
        logger.debug(f"{self.role.value} -> {recipient.value}: {message_type}")
        return message
    
    def create_result(
        self, 
        success: bool, 
        result: Any, 
        confidence: float,
        processing_time: float,
        reasoning_trace: List[str] = None
    ) -> AgentResult:
        """결과 객체 생성"""
        return AgentResult(
            agent_role=self.role,
            success=success,
            result=result,
            confidence=confidence,
            processing_time=processing_time,
            metadata=self.config,
            reasoning_trace=reasoning_trace or []
        )


class OrchestratorAgent(BaseAgent):
    """총괄 조정 에이전트"""
    
    def __init__(self, config: Dict = None):
        super().__init__(AgentRole.ORCHESTRATOR, config)
        self.agents = {}
        self.workflow_templates = self._initialize_workflows()
    
    def register_agent(self, agent: BaseAgent):
        """에이전트 등록"""
        self.agents[agent.role] = agent
        logger.info(f"Registered agent: {agent.role.value}")
    
    async def process(self, task: Dict[str, Any]) -> AgentResult:
        """작업 총괄 조정"""
        start_time = time.time()
        reasoning_trace = ["Orchestrator starting task coordination"]
        
        try:
            # 의도 분석 결과 가져오기
            intent = task.get('intent')
            query = task.get('query')
            has_image = task.get('has_image', False)
            
            # 워크플로우 결정
            workflow = self._determine_workflow(intent)
            reasoning_trace.append(f"Selected workflow: {workflow['name']}")
            
            # 에이전트 작업 순서 실행
            results = {}
            for step in workflow['steps']:
                agent_role = step['agent']
                if agent_role not in self.agents:
                    reasoning_trace.append(f"Warning: Agent {agent_role.value} not available")
                    continue
                
                # 작업 데이터 준비
                agent_task = {
                    'query': query,
                    'has_image': has_image,
                    'intent': intent,
                    'previous_results': results,
                    'step_config': step.get('config', {})
                }
                
                # 에이전트 실행
                agent = self.agents[agent_role]
                result = await agent.process(agent_task)
                results[agent_role] = result
                
                reasoning_trace.append(f"Completed: {agent_role.value} (confidence: {result.confidence:.2f})")
                
                # 실패 시 대체 전략
                if not result.success and step.get('required', True):
                    reasoning_trace.append(f"Critical step failed: {agent_role.value}")
                    break
            
            # 최종 결과 합성
            final_result = await self._synthesize_results(results, intent)
            processing_time = time.time() - start_time
            
            return self.create_result(
                success=True,
                result=final_result,
                confidence=self._calculate_overall_confidence(results),
                processing_time=processing_time,
                reasoning_trace=reasoning_trace
            )
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return self.create_result(
                success=False,
                result=str(e),
                confidence=0.0,
                processing_time=time.time() - start_time,
                reasoning_trace=reasoning_trace + [f"Error: {str(e)}"]
            )
    
    def _initialize_workflows(self) -> Dict[str, Dict]:
        """워크플로우 템플릿 초기화"""
        return {
            'vision_analysis': {
                'name': 'Vision Analysis Workflow',
                'steps': [
                    {'agent': AgentRole.VISION_ANALYST, 'required': True},
                    {'agent': AgentRole.RAG_SPECIALIST, 'required': False},
                    {'agent': AgentRole.SYNTHESIZER, 'required': True}
                ]
            },
            'mathematical_reasoning': {
                'name': 'Mathematical Reasoning Workflow',
                'steps': [
                    {'agent': AgentRole.MATH_SOLVER, 'required': True},
                    {'agent': AgentRole.RAG_SPECIALIST, 'required': False},
                    {'agent': AgentRole.VALIDATOR, 'required': True},
                    {'agent': AgentRole.SYNTHESIZER, 'required': True}
                ]
            },
            'complex_analysis': {
                'name': 'Complex Analysis Workflow',
                'steps': [
                    {'agent': AgentRole.RAG_SPECIALIST, 'required': True},
                    {'agent': AgentRole.REASONING_ENGINE, 'required': True},
                    {'agent': AgentRole.DOMAIN_EXPERT, 'required': False},
                    {'agent': AgentRole.VALIDATOR, 'required': True},
                    {'agent': AgentRole.SYNTHESIZER, 'required': True}
                ]
            },
            'standard': {
                'name': 'Standard Workflow',
                'steps': [
                    {'agent': AgentRole.RAG_SPECIALIST, 'required': True},
                    {'agent': AgentRole.SYNTHESIZER, 'required': True}
                ]
            }
        }
    
    def _determine_workflow(self, intent: IntentAnalysisResult) -> Dict:
        """적절한 워크플로우 결정"""
        if intent.requires_image:
            return self.workflow_templates['vision_analysis']
        elif intent.query_type == QueryType.MATHEMATICAL:
            return self.workflow_templates['mathematical_reasoning']
        elif intent.complexity_level >= 4 or intent.requires_reasoning:
            return self.workflow_templates['complex_analysis']
        else:
            return self.workflow_templates['standard']
    
    async def _synthesize_results(self, results: Dict[AgentRole, AgentResult], intent: IntentAnalysisResult) -> Dict:
        """결과 합성"""
        synthesis = {
            'primary_answer': None,
            'supporting_evidence': [],
            'confidence_scores': {},
            'processing_summary': [],
            'metadata': {}
        }
        
        # 각 에이전트 결과 통합
        for role, result in results.items():
            synthesis['confidence_scores'][role.value] = result.confidence
            synthesis['processing_summary'].extend(result.reasoning_trace)
            
            if role == AgentRole.SYNTHESIZER and result.success:
                synthesis['primary_answer'] = result.result
            elif role == AgentRole.RAG_SPECIALIST and result.success:
                synthesis['supporting_evidence'] = result.result
            elif role == AgentRole.VISION_ANALYST and result.success:
                synthesis['metadata']['vision_analysis'] = result.result
        
        return synthesis
    
    def _calculate_overall_confidence(self, results: Dict[AgentRole, AgentResult]) -> float:
        """전체 신뢰도 계산"""
        if not results:
            return 0.0
        
        confidences = [r.confidence for r in results.values() if r.success]
        if not confidences:
            return 0.0
        
        # 가중 평균 (중요한 에이전트에 더 높은 가중치)
        weights = {
            AgentRole.SYNTHESIZER: 0.3,
            AgentRole.RAG_SPECIALIST: 0.25,
            AgentRole.REASONING_ENGINE: 0.2,
            AgentRole.VISION_ANALYST: 0.15,
            AgentRole.MATH_SOLVER: 0.1
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for role, result in results.items():
            if result.success:
                weight = weights.get(role, 0.05)
                weighted_sum += result.confidence * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class VisionAnalystAgent(BaseAgent):
    """시각 분석 전문 에이전트"""
    
    def __init__(self, vision_analyzer=None, config: Dict = None):
        super().__init__(AgentRole.VISION_ANALYST, config)
        self.vision_analyzer = vision_analyzer
    
    async def process(self, task: Dict[str, Any]) -> AgentResult:
        """이미지 분석 처리"""
        start_time = time.time()
        reasoning_trace = ["Vision analysis started"]
        
        if not self.vision_analyzer:
            return self.create_result(
                success=False,
                result="Vision analyzer not available",
                confidence=0.0,
                processing_time=time.time() - start_time,
                reasoning_trace=reasoning_trace + ["Error: No vision analyzer"]
            )
        
        try:
            image = task.get('image')
            query = task.get('query', '')
            
            if not image:
                return self.create_result(
                    success=False,
                    result="No image provided",
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    reasoning_trace=reasoning_trace + ["Error: No image"]
                )
            
            # 이미지 분석 수행
            analysis_result = self.vision_analyzer.analyze_image(
                image, 
                question=query,
                extract_text=True,
                detect_formulas=True
            )
            
            if analysis_result.get('success', False):
                reasoning_trace.append(f"Vision analysis completed successfully")
                reasoning_trace.append(f"Text extracted: {len(analysis_result.get('text_content', ''))} chars")
                
                # 결과 구조화
                structured_result = {
                    'text_content': analysis_result.get('text_content', ''),
                    'formulas': analysis_result.get('formulas', []),
                    'has_formula': analysis_result.get('has_formula', False),
                    'description': analysis_result.get('description', ''),
                    'confidence': 0.8 if analysis_result.get('text_content') else 0.4
                }
                
                return self.create_result(
                    success=True,
                    result=structured_result,
                    confidence=structured_result['confidence'],
                    processing_time=time.time() - start_time,
                    reasoning_trace=reasoning_trace
                )
            else:
                return self.create_result(
                    success=False,
                    result=analysis_result.get('error', 'Unknown error'),
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    reasoning_trace=reasoning_trace + ["Vision analysis failed"]
                )
                
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return self.create_result(
                success=False,
                result=str(e),
                confidence=0.0,
                processing_time=time.time() - start_time,
                reasoning_trace=reasoning_trace + [f"Exception: {str(e)}"]
            )


class RAGSpecialistAgent(BaseAgent):
    """RAG 검색 전문 에이전트"""
    
    def __init__(self, rag_system=None, config: Dict = None):
        super().__init__(AgentRole.RAG_SPECIALIST, config)
        self.rag_system = rag_system
    
    async def process(self, task: Dict[str, Any]) -> AgentResult:
        """RAG 검색 처리"""
        start_time = time.time()
        reasoning_trace = ["RAG search started"]
        
        if not self.rag_system:
            return self.create_result(
                success=False,
                result="RAG system not available",
                confidence=0.0,
                processing_time=time.time() - start_time,
                reasoning_trace=reasoning_trace + ["Error: No RAG system"]
            )
        
        try:
            query = task.get('query', '')
            has_image = task.get('has_image', False)
            k = task.get('step_config', {}).get('k', 10)
            
            # 향상된 검색 수행
            if hasattr(self.rag_system, 'search_sync'):
                results, max_score = self.rag_system.search_sync(
                    query, k=k, has_image=has_image
                )
            else:
                results, max_score = self.rag_system.search(query, k=k)
            
            reasoning_trace.append(f"Found {len(results)} results, max score: {max_score:.3f}")
            
            # 결과 구조화
            structured_results = []
            for result in results:
                structured_results.append({
                    'question': result.question,
                    'answer': result.answer,
                    'score': result.score,
                    'category': result.category,
                    'metadata': getattr(result, 'metadata', {})
                })
            
            return self.create_result(
                success=True,
                result={
                    'results': structured_results,
                    'max_score': max_score,
                    'count': len(results)
                },
                confidence=min(max_score, 1.0),
                processing_time=time.time() - start_time,
                reasoning_trace=reasoning_trace
            )
            
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return self.create_result(
                success=False,
                result=str(e),
                confidence=0.0,
                processing_time=time.time() - start_time,
                reasoning_trace=reasoning_trace + [f"Exception: {str(e)}"]
            )


class ReasoningEngineAgent(BaseAgent):
    """추론 엔진 에이전트"""
    
    def __init__(self, llm_client=None, config: Dict = None):
        super().__init__(AgentRole.REASONING_ENGINE, config)
        self.llm_client = llm_client
    
    async def process(self, task: Dict[str, Any]) -> AgentResult:
        """추론 처리"""
        start_time = time.time()
        reasoning_trace = ["Reasoning engine started"]
        
        if not self.llm_client:
            return self.create_result(
                success=False,
                result="LLM client not available",
                confidence=0.0,
                processing_time=time.time() - start_time,
                reasoning_trace=reasoning_trace + ["Error: No LLM client"]
            )
        
        try:
            query = task.get('query', '')
            previous_results = task.get('previous_results', {})
            
            # 추론 체인 구성
            reasoning_prompt = self._build_reasoning_prompt(query, previous_results)
            
            # LLM을 통한 추론 수행
            reasoning_response = self.llm_client.query(reasoning_prompt, "")
            
            reasoning_trace.append("Chain-of-thought reasoning completed")
            
            # 추론 결과 구조화
            structured_result = {
                'reasoning_steps': self._extract_reasoning_steps(reasoning_response),
                'conclusion': reasoning_response,
                'confidence': self._assess_reasoning_confidence(reasoning_response)
            }
            
            return self.create_result(
                success=True,
                result=structured_result,
                confidence=structured_result['confidence'],
                processing_time=time.time() - start_time,
                reasoning_trace=reasoning_trace
            )
            
        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            return self.create_result(
                success=False,
                result=str(e),
                confidence=0.0,
                processing_time=time.time() - start_time,
                reasoning_trace=reasoning_trace + [f"Exception: {str(e)}"]
            )
    
    def _build_reasoning_prompt(self, query: str, previous_results: Dict) -> str:
        """추론 프롬프트 구성"""
        prompt_parts = [
            "다음 질문에 대해 단계별로 추론해주세요:",
            f"질문: {query}",
            "",
            "이전 분석 결과:"
        ]
        
        # 이전 결과 통합
        for agent_role, result in previous_results.items():
            if result.success:
                prompt_parts.append(f"- {agent_role.value}: {str(result.result)[:200]}...")
        
        prompt_parts.extend([
            "",
            "다음 형식으로 답변해주세요:",
            "1. 문제 분석:",
            "2. 필요한 정보:",
            "3. 추론 과정:",
            "4. 결론:",
            "",
            "각 단계를 명확하게 설명하고, 논리적 근거를 제시해주세요."
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """추론 단계 추출"""
        steps = []
        lines = response.split('\n')
        
        current_step = ""
        for line in lines:
            line = line.strip()
            if line and (line.startswith('1.') or line.startswith('2.') or 
                        line.startswith('3.') or line.startswith('4.')):
                if current_step:
                    steps.append(current_step.strip())
                current_step = line
            elif current_step:
                current_step += " " + line
        
        if current_step:
            steps.append(current_step.strip())
        
        return steps
    
    def _assess_reasoning_confidence(self, response: str) -> float:
        """추론 신뢰도 평가"""
        # 간단한 휴리스틱 기반 평가
        confidence = 0.5
        
        # 구체적인 단계가 있으면 신뢰도 증가
        if "1." in response and "2." in response:
            confidence += 0.2
        
        # 결론이 명확하면 신뢰도 증가
        if "결론" in response or "따라서" in response:
            confidence += 0.1
        
        # 불확실성 표현이 있으면 신뢰도 감소
        uncertainty_words = ["아마", "추정", "가능성", "불확실"]
        for word in uncertainty_words:
            if word in response:
                confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))


class SynthesizerAgent(BaseAgent):
    """응답 합성 에이전트"""
    
    def __init__(self, llm_client=None, config: Dict = None):
        super().__init__(AgentRole.SYNTHESIZER, config)
        self.llm_client = llm_client
    
    async def process(self, task: Dict[str, Any]) -> AgentResult:
        """응답 합성 처리"""
        start_time = time.time()
        reasoning_trace = ["Response synthesis started"]
        
        try:
            query = task.get('query', '')
            previous_results = task.get('previous_results', {})
            
            # 모든 결과 통합
            synthesis_prompt = self._build_synthesis_prompt(query, previous_results)
            
            if self.llm_client:
                # LLM을 통한 고급 합성
                final_response = self.llm_client.query(synthesis_prompt, "")
                reasoning_trace.append("LLM-based synthesis completed")
            else:
                # 규칙 기반 합성
                final_response = self._rule_based_synthesis(query, previous_results)
                reasoning_trace.append("Rule-based synthesis completed")
            
            return self.create_result(
                success=True,
                result=final_response,
                confidence=0.8,
                processing_time=time.time() - start_time,
                reasoning_trace=reasoning_trace
            )
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return self.create_result(
                success=False,
                result=str(e),
                confidence=0.0,
                processing_time=time.time() - start_time,
                reasoning_trace=reasoning_trace + [f"Exception: {str(e)}"]
            )
    
    def _build_synthesis_prompt(self, query: str, previous_results: Dict) -> str:
        """합성 프롬프트 구성"""
        prompt_parts = [
            f"사용자 질문: {query}",
            "",
            "다음은 여러 전문 에이전트가 분석한 결과입니다:",
            ""
        ]
        
        # 각 에이전트 결과 통합
        for agent_role, result in previous_results.items():
            if result.success:
                prompt_parts.append(f"【{agent_role.value.upper()}】")
                prompt_parts.append(f"신뢰도: {result.confidence:.2f}")
                prompt_parts.append(f"결과: {str(result.result)}")
                prompt_parts.append("")
        
        prompt_parts.extend([
            "위 정보를 종합하여 사용자에게 도움이 되는 완전하고 정확한 답변을 작성해주세요.",
            "",
            "답변 작성 지침:",
            "1. 핵심 내용을 명확하게 설명",
            "2. 여러 소스의 정보를 일관성 있게 통합",
            "3. 불확실한 부분은 명시적으로 표현",
            "4. 사용자가 이해하기 쉬운 언어 사용",
            "5. 필요시 예시나 추가 설명 포함"
        ])
        
        return "\n".join(prompt_parts)
    
    def _rule_based_synthesis(self, query: str, previous_results: Dict) -> str:
        """규칙 기반 응답 합성"""
        response_parts = [f"질문: {query}", "", "답변:"]
        
        # RAG 결과 우선 사용
        if AgentRole.RAG_SPECIALIST in previous_results:
            rag_result = previous_results[AgentRole.RAG_SPECIALIST]
            if rag_result.success and rag_result.result.get('results'):
                best_result = rag_result.result['results'][0]
                response_parts.append(best_result['answer'])
        
        # 비전 분석 결과 추가
        if AgentRole.VISION_ANALYST in previous_results:
            vision_result = previous_results[AgentRole.VISION_ANALYST]
            if vision_result.success:
                vision_data = vision_result.result
                if vision_data.get('text_content'):
                    response_parts.append("")
                    response_parts.append("이미지 분석 결과:")
                    response_parts.append(vision_data['text_content'][:500] + "...")
        
        # 추론 결과 추가
        if AgentRole.REASONING_ENGINE in previous_results:
            reasoning_result = previous_results[AgentRole.REASONING_ENGINE]
            if reasoning_result.success:
                reasoning_data = reasoning_result.result
                if reasoning_data.get('conclusion'):
                    response_parts.append("")
                    response_parts.append("추론 분석:")
                    response_parts.append(reasoning_data['conclusion'][:300] + "...")
        
        return "\n".join(response_parts)


class MultiAgentSystem:
    """다중 에이전트 시스템 관리자"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.orchestrator = OrchestratorAgent(config)
        self.agents = {AgentRole.ORCHESTRATOR: self.orchestrator}
        
    def register_agent(self, agent: BaseAgent):
        """에이전트 등록"""
        self.agents[agent.role] = agent
        self.orchestrator.register_agent(agent)
    
    async def process_query(
        self, 
        query: str, 
        intent: IntentAnalysisResult,
        has_image: bool = False,
        image=None,
        context: Dict = None
    ) -> Dict[str, Any]:
        """쿼리 처리"""
        start_time = time.time()
        
        # 작업 구성
        task = {
            'query': query,
            'intent': intent,
            'has_image': has_image,
            'image': image,
            'context': context or {}
        }
        
        # 오케스트레이터를 통한 처리
        result = await self.orchestrator.process(task)
        
        # 결과 구조화
        response = {
            'success': result.success,
            'answer': result.result.get('primary_answer', str(result.result)) if result.success else result.result,
            'confidence': result.confidence,
            'processing_time': time.time() - start_time,
            'agent_results': result.result if result.success else {},
            'reasoning_trace': result.reasoning_trace,
            'metadata': {
                'agents_used': list(self.agents.keys()),
                'workflow': 'multi_agent',
                'total_agents': len(self.agents)
            }
        }
        
        logger.info(f"Multi-agent processing completed in {response['processing_time']:.2f}s")
        
        return response
    
    def process_query_sync(
        self, 
        query: str, 
        intent: IntentAnalysisResult,
        has_image: bool = False,
        image=None,
        context: Dict = None
    ) -> Dict[str, Any]:
        """동기 쿼리 처리"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.process_query(query, intent, has_image, image, context)
            )
        finally:
            loop.close()


if __name__ == "__main__":
    # 테스트 예시
    import asyncio
    from intent_analyzer import IntentAnalyzer
    
    async def test_multi_agent():
        # 다중 에이전트 시스템 초기화
        mas = MultiAgentSystem()
        
        # 기본 에이전트들 등록
        rag_agent = RAGSpecialistAgent()
        synthesis_agent = SynthesizerAgent()
        reasoning_agent = ReasoningEngineAgent()
        
        mas.register_agent(rag_agent)
        mas.register_agent(synthesis_agent)
        mas.register_agent(reasoning_agent)
        
        # 의도 분석
        analyzer = IntentAnalyzer()
        intent = analyzer.analyze_intent("전력 계산 방법을 단계별로 설명해주세요", False)
        
        # 쿼리 처리
        result = await mas.process_query(
            "전력 계산 방법을 단계별로 설명해주세요",
            intent
        )
        
        print(f"Success: {result['success']}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Processing time: {result['processing_time']:.2f}s")
    
    # 테스트 실행
    asyncio.run(test_multi_agent())