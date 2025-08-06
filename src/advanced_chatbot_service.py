#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Chatbot Service with Next-Generation Features
차세대 기능을 탑재한 고급 챗봇 서비스
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from datetime import datetime, timedelta

from PIL import Image

from config import Config
from intent_analyzer import IntentAnalyzer, IntentAnalysisResult
from enhanced_rag_system import EnhancedRAGSystem
from multi_agent_system import MultiAgentSystem, VisionAnalystAgent, RAGSpecialistAgent, ReasoningEngineAgent, SynthesizerAgent
from optimized_openai_vision_analyzer import OptimizedOpenAIVisionAnalyzer as OpenAIVisionAnalyzer

logger = logging.getLogger(__name__)


class ConversationMode(Enum):
    """대화 모드"""
    STANDARD = "standard"           # 표준 모드
    EXPERT = "expert"              # 전문가 모드
    TUTORIAL = "tutorial"          # 튜토리얼 모드
    RESEARCH = "research"          # 연구 모드
    COLLABORATIVE = "collaborative" # 협업 모드


@dataclass
class ConversationContext:
    """대화 컨텍스트"""
    session_id: str
    user_id: Optional[str]
    mode: ConversationMode
    domain_focus: Optional[str]
    difficulty_level: int  # 1-5
    preferences: Dict[str, Any]
    history: List[Dict[str, Any]]
    memory: Dict[str, Any]
    created_at: datetime
    last_active: datetime


@dataclass
class ResponseMetadata:
    """응답 메타데이터"""
    processing_time: float
    confidence: float
    source_count: int
    agent_count: int
    reasoning_depth: int
    token_usage: int
    cost_estimate: float
    quality_score: float


class MemorySystem:
    """대화 메모리 시스템"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.short_term_memory = {}  # 세션별 단기 메모리
        self.long_term_memory = {}   # 사용자별 장기 메모리
        self.concept_graph = {}      # 개념 관계 그래프
        
    def store_conversation(
        self, 
        session_id: str, 
        query: str, 
        response: str, 
        metadata: Dict
    ):
        """대화 저장"""
        if session_id not in self.short_term_memory:
            self.short_term_memory[session_id] = []
        
        conversation_entry = {
            'timestamp': datetime.now(),
            'query': query,
            'response': response,
            'metadata': metadata
        }
        
        self.short_term_memory[session_id].append(conversation_entry)
        
        # 개념 추출 및 그래프 업데이트
        self._update_concept_graph(query, response, metadata)
    
    def get_relevant_context(
        self, 
        session_id: str, 
        current_query: str, 
        max_context: int = 5
    ) -> List[Dict]:
        """관련 컨텍스트 검색"""
        if session_id not in self.short_term_memory:
            return []
        
        conversations = self.short_term_memory[session_id]
        
        # 최근 대화 우선
        recent_conversations = conversations[-max_context:]
        
        # 의미적 유사성 기반 필터링 (간단한 키워드 매칭)
        relevant_conversations = []
        query_words = set(current_query.lower().split())
        
        for conv in recent_conversations:
            conv_words = set((conv['query'] + ' ' + conv['response']).lower().split())
            similarity = len(query_words.intersection(conv_words)) / len(query_words.union(conv_words))
            
            if similarity > 0.1:  # 임계값
                relevant_conversations.append({
                    'conversation': conv,
                    'similarity': similarity
                })
        
        # 유사성 기준 정렬
        relevant_conversations.sort(key=lambda x: x['similarity'], reverse=True)
        
        return [rc['conversation'] for rc in relevant_conversations]
    
    def _update_concept_graph(self, query: str, response: str, metadata: Dict):
        """개념 그래프 업데이트"""
        # 간단한 키워드 기반 개념 추출
        concepts = self._extract_concepts(query + ' ' + response)
        
        for concept in concepts:
            if concept not in self.concept_graph:
                self.concept_graph[concept] = {
                    'frequency': 0,
                    'related_concepts': set(),
                    'last_seen': datetime.now()
                }
            
            self.concept_graph[concept]['frequency'] += 1
            self.concept_graph[concept]['last_seen'] = datetime.now()
            
            # 관련 개념 연결
            for other_concept in concepts:
                if other_concept != concept:
                    self.concept_graph[concept]['related_concepts'].add(other_concept)
    
    def _extract_concepts(self, text: str) -> List[str]:
        """텍스트에서 개념 추출"""
        # 간단한 키워드 추출 (실제로는 NER, 키워드 추출 알고리즘 사용)
        import re
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # 일반적인 불용어 제거
        stopwords = {'그리고', '하지만', '그런데', '따라서', '만약', '때문에', '통해서'}
        concepts = [word for word in words if word not in stopwords and len(word) > 2]
        
        return list(set(concepts))


class AdaptiveResponseGenerator:
    """적응형 응답 생성기"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.response_templates = self._initialize_templates()
        
    def generate_adaptive_response(
        self, 
        base_response: str,
        context: ConversationContext,
        metadata: ResponseMetadata
    ) -> str:
        """컨텍스트 기반 적응형 응답 생성"""
        
        # 모드별 응답 조정
        if context.mode == ConversationMode.EXPERT:
            response = self._enhance_for_expert_mode(base_response, metadata)
        elif context.mode == ConversationMode.TUTORIAL:
            response = self._enhance_for_tutorial_mode(base_response, metadata)
        elif context.mode == ConversationMode.RESEARCH:
            response = self._enhance_for_research_mode(base_response, metadata)
        else:
            response = base_response
        
        # 난이도 조정
        response = self._adjust_for_difficulty(response, context.difficulty_level)
        
        # 개인화
        response = self._personalize_response(response, context)
        
        # 메타정보 추가
        response = self._add_metadata_info(response, metadata)
        
        return response
    
    def _enhance_for_expert_mode(self, response: str, metadata: ResponseMetadata) -> str:
        """전문가 모드용 응답 강화"""
        enhancements = []
        
        # 신뢰도 정보 추가
        enhancements.append(f"\n\n**신뢰도 분석**: {metadata.confidence:.1%}")
        
        # 소스 정보 추가
        if metadata.source_count > 0:
            enhancements.append(f"**참조 소스**: {metadata.source_count}개 문서")
        
        # 처리 과정 정보
        if metadata.agent_count > 1:
            enhancements.append(f"**분석 에이전트**: {metadata.agent_count}개 전문 시스템")
        
        # 품질 점수
        if metadata.quality_score > 0.8:
            enhancements.append("**품질**: 높음 ✓")
        elif metadata.quality_score > 0.6:
            enhancements.append("**품질**: 중간 ○")
        else:
            enhancements.append("**품질**: 낮음 △")
        
        return response + "".join(enhancements)
    
    def _enhance_for_tutorial_mode(self, response: str, metadata: ResponseMetadata) -> str:
        """튜토리얼 모드용 응답 강화"""
        # 단계별 설명 추가
        tutorial_elements = [
            "\n\n**📚 학습 도움말**:",
            "• 이해하기 어려운 부분이 있으면 더 자세한 설명을 요청하세요",
            "• 예시를 더 보고 싶으면 '예시 더 보여줘'라고 말씀하세요",
            "• 연관된 개념을 알고 싶으면 '관련 내용'을 요청하세요"
        ]
        
        if metadata.reasoning_depth > 2:
            tutorial_elements.append("• 복잡한 내용이므로 단계별로 차근차근 읽어보세요")
        
        return response + "\n".join(tutorial_elements)
    
    def _enhance_for_research_mode(self, response: str, metadata: ResponseMetadata) -> str:
        """연구 모드용 응답 강화"""
        research_elements = []
        
        # 추가 연구 방향 제시
        research_elements.append("\n\n**🔬 연구 확장 제안**:")
        research_elements.append("• 이 주제와 관련된 최신 연구 동향을 확인해보세요")
        research_elements.append("• 다른 관점에서의 접근 방법을 탐색해보세요")
        
        # 참고문헌 스타일 정보
        if metadata.source_count > 0:
            research_elements.append(f"• 참조된 {metadata.source_count}개 소스의 원문을 확인하시기 바랍니다")
        
        # 신뢰도 기반 주의사항
        if metadata.confidence < 0.8:
            research_elements.append("⚠️ **주의**: 이 답변의 신뢰도가 높지 않으므로 추가 검증이 필요합니다")
        
        return response + "\n".join(research_elements)
    
    def _adjust_for_difficulty(self, response: str, difficulty_level: int) -> str:
        """난이도별 응답 조정"""
        if difficulty_level <= 2:  # 초급
            # 쉬운 용어로 대체, 더 많은 설명 추가
            response += "\n\n**💡 쉬운 설명**: 복잡한 용어가 있다면 더 쉽게 설명해드릴 수 있습니다."
        elif difficulty_level >= 4:  # 고급
            # 전문 용어 유지, 심화 내용 추가
            response += "\n\n**🎓 심화 학습**: 더 깊이 있는 내용이나 수학적 접근이 필요하면 말씀하세요."
        
        return response
    
    def _personalize_response(self, response: str, context: ConversationContext) -> str:
        """개인화 응답"""
        # 사용자 선호도 반영
        preferences = context.preferences
        
        if preferences.get('include_examples', True):
            response += "\n\n**📋 관련 예시**: 구체적인 예시가 필요하면 요청해주세요."
        
        if preferences.get('show_formulas', True) and '수식' in response:
            response += "\n\n**🧮 수식 설명**: 수식의 의미나 유도 과정이 궁금하면 알려주세요."
        
        return response
    
    def _add_metadata_info(self, response: str, metadata: ResponseMetadata) -> str:
        """메타데이터 정보 추가"""
        footer_elements = []
        
        # 처리 시간 (느린 경우만)
        if metadata.processing_time > 10:
            footer_elements.append(f"⏱️ 처리시간: {metadata.processing_time:.1f}초")
        
        # 비용 정보 (높은 경우만)
        if metadata.cost_estimate > 0.01:
            footer_elements.append(f"💰 예상비용: ${metadata.cost_estimate:.3f}")
        
        if footer_elements:
            response += f"\n\n_{' | '.join(footer_elements)}_"
        
        return response
    
    def _initialize_templates(self) -> Dict[str, str]:
        """응답 템플릿 초기화"""
        return {
            'greeting': "안녕하세요! 무엇을 도와드릴까요?",
            'clarification': "질문을 더 구체적으로 설명해주시겠어요?",
            'no_results': "죄송합니다. 관련 정보를 찾을 수 없습니다.",
            'error': "처리 중 오류가 발생했습니다. 다시 시도해주세요.",
            'thinking': "분석 중입니다... 잠시만 기다려주세요."
        }


class AdvancedChatbotService:
    """차세대 고급 챗봇 서비스"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # 핵심 컴포넌트 초기화
        self.intent_analyzer = IntentAnalyzer()
        self.memory_system = MemorySystem()
        self.response_generator = AdaptiveResponseGenerator()
        
        # RAG 시스템 초기화
        from rag_system import RAGSystem
        from llm_client_openai import LLMClient
        
        self.llm_client = LLMClient(config.llm)
        self.base_rag = RAGSystem(config.rag, config.dataset, self.llm_client)
        self.enhanced_rag = EnhancedRAGSystem(self.base_rag, config.rag)
        
        # 이미지 분석기 초기화
        self.vision_analyzer = OpenAIVisionAnalyzer(config)
        
        # 다중 에이전트 시스템 초기화
        self.multi_agent_system = MultiAgentSystem()
        self._initialize_agents()
        
        # 대화 컨텍스트 저장소
        self.conversation_contexts = {}
        
        # 통계
        self.stats = {
            'total_queries': 0,
            'successful_responses': 0,
            'average_response_time': 0.0,
            'mode_usage': {mode.value: 0 for mode in ConversationMode},
            'feature_usage': {
                'vision_analysis': 0,
                'multi_agent': 0,
                'memory_recall': 0,
                'adaptive_response': 0
            }
        }
        
        logger.info("Advanced Chatbot Service initialized successfully")
    
    def _initialize_agents(self):
        """에이전트 초기화"""
        # 비전 분석 에이전트
        vision_agent = VisionAnalystAgent(self.vision_analyzer)
        self.multi_agent_system.register_agent(vision_agent)
        
        # RAG 전문 에이전트
        rag_agent = RAGSpecialistAgent(self.enhanced_rag)
        self.multi_agent_system.register_agent(rag_agent)
        
        # 추론 엔진
        reasoning_agent = ReasoningEngineAgent(self.llm_client)
        self.multi_agent_system.register_agent(reasoning_agent)
        
        # 응답 합성기
        synthesizer = SynthesizerAgent(self.llm_client)
        self.multi_agent_system.register_agent(synthesizer)
        
        logger.info("Multi-agent system initialized with 4 specialized agents")
    
    def create_conversation_context(
        self, 
        session_id: str,
        user_id: Optional[str] = None,
        mode: ConversationMode = ConversationMode.STANDARD,
        preferences: Dict = None
    ) -> ConversationContext:
        """대화 컨텍스트 생성"""
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            mode=mode,
            domain_focus=None,
            difficulty_level=3,  # 기본 중간 난이도
            preferences=preferences or {},
            history=[],
            memory={},
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        
        self.conversation_contexts[session_id] = context
        return context
    
    async def process_query_advanced(
        self,
        query: str,
        session_id: str,
        image: Optional[Image.Image] = None,
        mode: Optional[ConversationMode] = None,
        user_preferences: Dict = None
    ) -> Dict[str, Any]:
        """고급 쿼리 처리"""
        start_time = time.time()
        
        # 대화 컨텍스트 가져오기 또는 생성
        if session_id not in self.conversation_contexts:
            self.create_conversation_context(session_id, preferences=user_preferences)
        
        context = self.conversation_contexts[session_id]
        context.last_active = datetime.now()
        
        # 모드 업데이트
        if mode:
            context.mode = mode
        
        try:
            # 1. 의도 분석
            intent = self.intent_analyzer.analyze_intent(
                query, 
                has_image=image is not None,
                context={'conversation_context': context}
            )
            
            # 2. 메모리에서 관련 컨텍스트 검색
            relevant_context = self.memory_system.get_relevant_context(
                session_id, query
            )
            
            # 3. 처리 전략 결정
            use_multi_agent = self._should_use_multi_agent(intent, context)
            
            # 4. 쿼리 처리
            if use_multi_agent:
                logger.info("Using multi-agent processing")
                self.stats['feature_usage']['multi_agent'] += 1
                
                result = await self.multi_agent_system.process_query(
                    query, intent, 
                    has_image=image is not None, 
                    image=image,
                    context={'conversation_context': context, 'relevant_history': relevant_context}
                )
                
                base_response = result['answer']
                confidence = result['confidence']
                processing_metadata = result['metadata']
                
            else:
                logger.info("Using standard RAG processing")
                
                # 표준 RAG 처리
                if hasattr(self.enhanced_rag, 'search_sync'):
                    search_results, max_score = self.enhanced_rag.search_sync(
                        query, k=10, has_image=image is not None
                    )
                else:
                    search_results, max_score = self.base_rag.search(query, k=10)
                
                # 기본 응답 생성
                base_response = self._generate_basic_response(
                    query, search_results, max_score, image
                )
                confidence = max_score
                processing_metadata = {'agents_used': ['rag_only']}
            
            # 5. 응답 메타데이터 생성
            response_metadata = ResponseMetadata(
                processing_time=time.time() - start_time,
                confidence=confidence,
                source_count=len(search_results if 'search_results' in locals() else []),
                agent_count=len(processing_metadata.get('agents_used', [])),
                reasoning_depth=intent.complexity_level,
                token_usage=intent.estimated_tokens,
                cost_estimate=self._estimate_cost(intent.estimated_tokens),
                quality_score=self._assess_quality(base_response, confidence)
            )
            
            # 6. 적응형 응답 생성
            final_response = self.response_generator.generate_adaptive_response(
                base_response, context, response_metadata
            )
            
            # 7. 대화 메모리에 저장
            self.memory_system.store_conversation(
                session_id, query, final_response, 
                {'intent': intent, 'metadata': response_metadata}
            )
            self.stats['feature_usage']['memory_recall'] += 1
            
            # 8. 컨텍스트 업데이트
            context.history.append({
                'query': query,
                'response': final_response,
                'timestamp': datetime.now(),
                'intent': intent,
                'metadata': response_metadata
            })
            
            # 9. 통계 업데이트
            self._update_stats(context.mode, True, response_metadata.processing_time)
            
            # 10. 결과 반환
            return {
                'success': True,
                'response': final_response,
                'metadata': {
                    'confidence': confidence,
                    'processing_time': response_metadata.processing_time,
                    'mode': context.mode.value,
                    'intent_type': intent.query_type.value,
                    'complexity': intent.complexity_level,
                    'features_used': self._get_features_used(intent, use_multi_agent),
                    'quality_score': response_metadata.quality_score,
                    'cost_estimate': response_metadata.cost_estimate
                },
                'suggestions': self._generate_follow_up_suggestions(query, intent, context)
            }
            
        except Exception as e:
            logger.error(f"Advanced query processing failed: {e}")
            self._update_stats(context.mode if 'context' in locals() else ConversationMode.STANDARD, False, time.time() - start_time)
            
            return {
                'success': False,
                'response': "죄송합니다. 처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                'error': str(e),
                'metadata': {
                    'processing_time': time.time() - start_time,
                    'mode': context.mode.value if 'context' in locals() else 'standard'
                }
            }
    
    def _should_use_multi_agent(self, intent: IntentAnalysisResult, context: ConversationContext) -> bool:
        """다중 에이전트 사용 여부 결정"""
        # 복잡한 쿼리
        if intent.complexity_level >= 4:
            return True
        
        # 이미지 분석 필요
        if intent.requires_image:
            return True
        
        # 추론이 필요한 경우
        if intent.requires_reasoning:
            return True
        
        # 전문가 모드
        if context.mode in [ConversationMode.EXPERT, ConversationMode.RESEARCH]:
            return True
        
        # 다단계 처리 모드
        if intent.processing_mode.value in ['reasoning_chain', 'multi_agent', 'hybrid']:
            return True
        
        return False
    
    def _generate_basic_response(
        self, 
        query: str, 
        search_results: List, 
        max_score: float,
        image: Optional[Image.Image] = None
    ) -> str:
        """기본 응답 생성"""
        if not search_results:
            return "관련 정보를 찾을 수 없습니다. 질문을 다시 구성해서 물어보시거나 더 구체적인 내용을 제공해주세요."
        
        best_result = search_results[0]
        
        response_parts = [f"답변: {best_result.answer}"]
        
        # 이미지 분석 결과 추가 (비전 분석기 사용)
        if image and self.vision_analyzer.api_available:
            try:
                vision_result = self.vision_analyzer.analyze_image(image, query)
                if vision_result.get('success'):
                    response_parts.append(f"\n\n이미지 분석: {vision_result['raw_response'][:300]}...")
            except Exception as e:
                logger.warning(f"Vision analysis failed: {e}")
        
        # 신뢰도 정보
        if max_score >= 0.8:
            confidence_text = "높음"
        elif max_score >= 0.6:
            confidence_text = "중간"
        else:
            confidence_text = "낮음"
        
        response_parts.append(f"\n\n[신뢰도: {confidence_text} ({max_score:.3f})]")
        
        return "".join(response_parts)
    
    def _estimate_cost(self, tokens: int) -> float:
        """비용 추정"""
        # OpenAI gpt-4o-mini 기준 ($0.15 입력, $0.60 출력 per 1M tokens)
        input_cost = tokens * 0.00000015
        output_cost = tokens * 0.5 * 0.0000006  # 출력은 입력의 절반 가정
        return input_cost + output_cost
    
    def _assess_quality(self, response: str, confidence: float) -> float:
        """응답 품질 평가"""
        quality_score = confidence * 0.6  # 신뢰도 기반
        
        # 길이 기반 보정
        if len(response) > 100:
            quality_score += 0.1
        if len(response) > 500:
            quality_score += 0.1
        
        # 구조 기반 보정
        if '단계' in response or '방법' in response:
            quality_score += 0.1
        
        # 예시 포함 여부
        if '예시' in response or '예를 들어' in response:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _get_features_used(self, intent: IntentAnalysisResult, use_multi_agent: bool) -> List[str]:
        """사용된 기능 목록"""
        features = []
        
        if intent.requires_image:
            features.append('vision_analysis')
        
        if use_multi_agent:
            features.append('multi_agent_system')
        
        if intent.requires_reasoning:
            features.append('chain_of_thought')
        
        if intent.requires_external_search:
            features.append('web_search')
        
        features.append('enhanced_rag')
        features.append('adaptive_response')
        
        return features
    
    def _generate_follow_up_suggestions(
        self, 
        query: str, 
        intent: IntentAnalysisResult, 
        context: ConversationContext
    ) -> List[str]:
        """후속 질문 제안"""
        suggestions = []
        
        if intent.query_type.value == 'mathematical':
            suggestions.extend([
                "이 수식의 유도 과정을 보여주세요",
                "실제 적용 예시를 알려주세요",
                "관련된 다른 공식들을 설명해주세요"
            ])
        
        elif intent.query_type.value == 'analytical':
            suggestions.extend([
                "더 자세한 분석 결과를 보여주세요",
                "다른 관점에서 접근해보세요",
                "실무에서 어떻게 활용되는지 알려주세요"
            ])
        
        elif intent.requires_image:
            suggestions.extend([
                "이미지의 다른 부분을 분석해주세요",
                "유사한 예시를 더 보여주세요",
                "이론적 배경을 설명해주세요"
            ])
        
        # 모드별 제안
        if context.mode == ConversationMode.TUTORIAL:
            suggestions.extend([
                "단계별로 더 자세히 설명해주세요",
                "연습 문제를 제공해주세요",
                "관련 개념을 복습해주세요"
            ])
        
        return suggestions[:3]  # 최대 3개
    
    def _update_stats(self, mode: ConversationMode, success: bool, processing_time: float):
        """통계 업데이트"""
        self.stats['total_queries'] += 1
        self.stats['mode_usage'][mode.value] += 1
        
        if success:
            self.stats['successful_responses'] += 1
        
        # 평균 처리 시간 업데이트
        n = self.stats['total_queries']
        prev_avg = self.stats['average_response_time']
        self.stats['average_response_time'] = (prev_avg * (n - 1) + processing_time) / n
    
    def get_stats(self) -> Dict:
        """통계 반환"""
        return self.stats.copy()
    
    def get_conversation_summary(self, session_id: str) -> Dict:
        """대화 요약 반환"""
        if session_id not in self.conversation_contexts:
            return {"error": "Session not found"}
        
        context = self.conversation_contexts[session_id]
        
        return {
            'session_id': session_id,
            'mode': context.mode.value,
            'conversation_count': len(context.history),
            'duration': (context.last_active - context.created_at).total_seconds(),
            'dominant_topics': self._extract_dominant_topics(context.history),
            'avg_complexity': sum(h.get('intent', type('obj', (object,), {'complexity_level': 3})).complexity_level 
                                for h in context.history) / max(len(context.history), 1),
            'last_active': context.last_active.isoformat()
        }
    
    def _extract_dominant_topics(self, history: List[Dict]) -> List[str]:
        """주요 대화 주제 추출"""
        if not history:
            return []
        
        # 간단한 키워드 빈도 분석
        word_freq = {}
        for entry in history:
            words = entry['query'].lower().split()
            for word in words:
                if len(word) > 2:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # 상위 5개 키워드
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [word for word, freq in top_words]


# 동기 인터페이스
def create_advanced_chatbot_service(config: Config) -> AdvancedChatbotService:
    """고급 챗봇 서비스 생성"""
    return AdvancedChatbotService(config)


if __name__ == "__main__":
    # 테스트 예시
    from config import Config
    
    config = Config()
    service = create_advanced_chatbot_service(config)
    
    # 비동기 테스트
    async def test_advanced_service():
        result = await service.process_query_advanced(
            "전력 계산 방법을 단계별로 설명해주세요",
            session_id="test_session_1",
            mode=ConversationMode.TUTORIAL
        )
        
        print(f"Success: {result['success']}")
        print(f"Response: {result['response'][:200]}...")
        print(f"Features: {result['metadata']['features_used']}")
        print(f"Quality: {result['metadata']['quality_score']:.2f}")
    
    # 테스트 실행
    import asyncio
    asyncio.run(test_advanced_service())