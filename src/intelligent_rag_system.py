#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent RAG System
지능형 RAG 시스템 - 범용 지식 시스템과 통합
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# 내부 모듈
from universal_knowledge_system import (
    UniversalKnowledgeOrchestrator,
    KnowledgeDomain,
    ContentComplexity,
    InformationType
)
from modular_rag_system import (
    ModularRAGPipeline,
    ProcessingContext,
    QueryType
)
from universal_ocr_pipeline import DomainAdaptiveOCR

logger = logging.getLogger(__name__)


class ResponseStrategy(Enum):
    """응답 전략"""
    EDUCATIONAL = "educational"  # 교육적 설명
    TECHNICAL = "technical"  # 기술적 상세
    PRACTICAL = "practical"  # 실용적 가이드
    CONVERSATIONAL = "conversational"  # 대화형
    ANALYTICAL = "analytical"  # 분석적
    CREATIVE = "creative"  # 창의적


@dataclass
class IntelligentContext:
    """지능형 처리 컨텍스트"""
    query: str
    detected_domains: List[KnowledgeDomain]
    complexity_level: ContentComplexity
    user_intent: str
    response_strategy: ResponseStrategy
    multimodal_data: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    user_profile: Optional[Dict[str, Any]] = None


class IntelligentQueryProcessor:
    """지능형 쿼리 처리기"""
    
    def __init__(self):
        """초기화"""
        self.knowledge_orchestrator = UniversalKnowledgeOrchestrator()
        self.intent_patterns = self._init_intent_patterns()
        self.domain_keywords = self._init_domain_keywords()
    
    def _init_intent_patterns(self) -> Dict[str, List[str]]:
        """의도 패턴 초기화"""
        return {
            'learn': ['배우고 싶', '알고 싶', '이해하고 싶', '공부', 'learn', 'study', 'understand'],
            'solve': ['해결', '풀어', '계산', '구하', 'solve', 'calculate', 'find'],
            'explain': ['설명', '알려', '가르쳐', 'explain', 'tell', 'teach'],
            'compare': ['비교', '차이', '다른 점', 'compare', 'difference', 'versus'],
            'implement': ['구현', '만들', '개발', '코딩', 'implement', 'create', 'develop'],
            'analyze': ['분석', '평가', '검토', 'analyze', 'evaluate', 'review'],
            'debug': ['오류', '에러', '문제', '고치', 'error', 'bug', 'fix'],
            'optimize': ['최적화', '개선', '향상', 'optimize', 'improve', 'enhance']
        }
    
    def _init_domain_keywords(self) -> Dict[KnowledgeDomain, List[str]]:
        """도메인 키워드 매핑"""
        return {
            KnowledgeDomain.MATHEMATICS: ['수학', '미적분', '선형대수', '확률', 'math', 'calculus', 'algebra'],
            KnowledgeDomain.PHYSICS: ['물리', '역학', '전자기', '양자', 'physics', 'mechanics', 'quantum'],
            KnowledgeDomain.COMPUTER: ['컴퓨터', '프로그래밍', '알고리즘', 'computer', 'programming', 'algorithm'],
            KnowledgeDomain.AI_ML: ['인공지능', 'AI', '머신러닝', 'ML', '딥러닝', 'deep learning'],
            KnowledgeDomain.ELECTRICAL: ['전기', '전자', '회로', 'electrical', 'circuit', 'electronics'],
            KnowledgeDomain.CHEMISTRY: ['화학', '분자', '반응', 'chemistry', 'molecule', 'reaction'],
            KnowledgeDomain.BIOLOGY: ['생물', '생명', '세포', 'biology', 'life', 'cell'],
            KnowledgeDomain.MEDICINE: ['의학', '의료', '질병', 'medicine', 'medical', 'disease']
        }
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> IntelligentContext:
        """쿼리 처리 및 컨텍스트 생성"""
        # 기본 분석
        intent = self._detect_intent(query)
        domains = self._detect_domains(query)
        complexity = self._estimate_complexity(query, context)
        strategy = self._determine_strategy(intent, domains, complexity)
        
        # 지식 시스템을 통한 심층 분석
        knowledge_result = self.knowledge_orchestrator.process_content(query, context)
        
        # 감지된 도메인 보강
        if knowledge_result['detected_domains']:
            domains.extend(knowledge_result['detected_domains'])
            domains = list(set(domains))[:3]  # 상위 3개 도메인
        
        # 복잡도 조정
        if knowledge_result.get('complexity_level'):
            complexity = knowledge_result['complexity_level']
        
        return IntelligentContext(
            query=query,
            detected_domains=domains,
            complexity_level=complexity,
            user_intent=intent,
            response_strategy=strategy,
            multimodal_data=context.get('multimodal_data'),
            conversation_history=context.get('conversation_history'),
            user_profile=context.get('user_profile')
        )
    
    def _detect_intent(self, query: str) -> str:
        """사용자 의도 감지"""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return 'explain'  # 기본값
    
    def _detect_domains(self, query: str) -> List[KnowledgeDomain]:
        """도메인 감지"""
        query_lower = query.lower()
        detected_domains = []
        
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_domains.append(domain)
        
        return detected_domains[:3] if detected_domains else [KnowledgeDomain.GENERAL]
    
    def _estimate_complexity(self, query: str, context: Optional[Dict[str, Any]]) -> ContentComplexity:
        """복잡도 추정"""
        # 사용자 프로필 기반
        if context and context.get('user_profile'):
            profile = context['user_profile']
            if 'education_level' in profile:
                return ContentComplexity(profile['education_level'])
        
        # 쿼리 기반 추정
        complexity_indicators = {
            ContentComplexity.ELEMENTARY: ['기초', '쉽게', '초보', 'basic', 'simple'],
            ContentComplexity.HIGH_SCHOOL: ['고등학교', '수능', 'high school'],
            ContentComplexity.UNDERGRADUATE: ['대학', '학부', 'university'],
            ContentComplexity.GRADUATE: ['대학원', '논문', 'research', 'paper'],
            ContentComplexity.EXPERT: ['전문가', '고급', 'expert', 'advanced']
        }
        
        query_lower = query.lower()
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return complexity
        
        return ContentComplexity.UNDERGRADUATE  # 기본값
    
    def _determine_strategy(self, intent: str, domains: List[KnowledgeDomain], complexity: ContentComplexity) -> ResponseStrategy:
        """응답 전략 결정"""
        # 의도 기반 전략
        intent_strategy_map = {
            'learn': ResponseStrategy.EDUCATIONAL,
            'solve': ResponseStrategy.PRACTICAL,
            'explain': ResponseStrategy.EDUCATIONAL,
            'compare': ResponseStrategy.ANALYTICAL,
            'implement': ResponseStrategy.TECHNICAL,
            'analyze': ResponseStrategy.ANALYTICAL,
            'debug': ResponseStrategy.TECHNICAL,
            'optimize': ResponseStrategy.TECHNICAL
        }
        
        # 도메인 기반 조정
        if KnowledgeDomain.MATHEMATICS in domains or KnowledgeDomain.PHYSICS in domains:
            if complexity in [ContentComplexity.GRADUATE, ContentComplexity.EXPERT]:
                return ResponseStrategy.TECHNICAL
        
        if KnowledgeDomain.LITERATURE in domains or KnowledgeDomain.ART in domains:
            return ResponseStrategy.CREATIVE
        
        return intent_strategy_map.get(intent, ResponseStrategy.CONVERSATIONAL)


class IntelligentRAGOrchestrator:
    """지능형 RAG 오케스트레이터"""
    
    def __init__(self, config: Dict[str, Any]):
        """초기화"""
        self.config = config
        self.query_processor = IntelligentQueryProcessor()
        self.knowledge_orchestrator = UniversalKnowledgeOrchestrator()
        self.ocr_pipeline = DomainAdaptiveOCR()
        self.rag_pipeline = ModularRAGPipeline(config)
        self.cache = {}  # 간단한 캐시
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_async(
        self,
        query: str,
        image: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """비동기 처리"""
        # 캐시 확인
        cache_key = self._generate_cache_key(query, image is not None)
        if cache_key in self.cache:
            logger.info("Returning cached result")
            return self.cache[cache_key]
        
        start_time = time.time()
        
        # 병렬 처리를 위한 태스크
        tasks = []
        
        # 1. 쿼리 분석
        tasks.append(asyncio.create_task(self._analyze_query_async(query, context)))
        
        # 2. 이미지 처리 (있는 경우)
        if image:
            tasks.append(asyncio.create_task(self._process_image_async(image)))
        
        # 3. 기본 검색 수행
        tasks.append(asyncio.create_task(self._initial_search_async(query)))
        
        # 태스크 완료 대기
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 통합
        intelligent_context = results[0] if not isinstance(results[0], Exception) else None
        image_analysis = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
        initial_search = results[2 if image else 1] if len(results) > (2 if image else 1) else None
        
        # 지능형 처리
        final_result = await self._intelligent_processing(
            intelligent_context,
            image_analysis,
            initial_search,
            query
        )
        
        # 캐시 저장
        self.cache[cache_key] = final_result
        
        final_result['processing_time'] = time.time() - start_time
        
        return final_result
    
    def process_sync(
        self,
        query: str,
        image: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """동기 처리 (비동기 래퍼)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_async(query, image, context))
        finally:
            loop.close()
    
    async def _analyze_query_async(self, query: str, context: Optional[Dict[str, Any]]) -> IntelligentContext:
        """비동기 쿼리 분석"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.query_processor.process_query,
            query,
            context
        )
    
    async def _process_image_async(self, image: Any) -> Dict[str, Any]:
        """비동기 이미지 처리"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.ocr_pipeline.process_adaptive,
            image,
            True  # auto_detect
        )
    
    async def _initial_search_async(self, query: str) -> List[Dict[str, Any]]:
        """비동기 초기 검색"""
        # RAG 파이프라인의 검색 모듈 활용
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._perform_initial_search,
            query
        )
    
    def _perform_initial_search(self, query: str) -> List[Dict[str, Any]]:
        """초기 검색 수행"""
        if 'retrieval' in self.rag_pipeline.modules:
            context = ProcessingContext(
                query=query,
                query_type=QueryType.FACTUAL,
                language="ko"
            )
            return self.rag_pipeline.modules['retrieval'].process(context, {'keywords': []})
        return []
    
    async def _intelligent_processing(
        self,
        context: IntelligentContext,
        image_analysis: Optional[Dict[str, Any]],
        initial_search: List[Dict[str, Any]],
        original_query: str
    ) -> Dict[str, Any]:
        """지능형 처리"""
        result = {
            'success': True,
            'query': original_query,
            'context': context,
            'domains': [d.value for d in context.detected_domains] if context else [],
            'complexity': context.complexity_level.value if context else 'unknown',
            'intent': context.user_intent if context else 'unknown',
            'strategy': context.response_strategy.value if context else 'unknown'
        }
        
        # 멀티모달 정보 통합
        if image_analysis:
            result['image_analysis'] = image_analysis
            
            # 도메인 정보 업데이트
            if 'domain_info' in image_analysis:
                detected_domain = image_analysis['domain_info'].get('detected_domain')
                if detected_domain and detected_domain != 'generic':
                    # 도메인 매핑
                    domain_map = {
                        'engineering': KnowledgeDomain.ELECTRICAL,
                        'medical': KnowledgeDomain.MEDICINE,
                        'academic': KnowledgeDomain.GENERAL
                    }
                    if detected_domain in domain_map:
                        context.detected_domains.insert(0, domain_map[detected_domain])
        
        # 도메인별 전문 처리
        domain_results = await self._process_by_domains(context, image_analysis)
        result['domain_analysis'] = domain_results
        
        # 지식 그래프 활용
        if context:
            # 학습 경로 생성
            if context.user_intent == 'learn':
                learning_path = self._generate_learning_path(context)
                result['learning_path'] = learning_path
            
            # 관련 개념 추천
            recommendations = self._get_recommendations(context)
            result['recommendations'] = recommendations
        
        # 최종 응답 생성
        response = await self._generate_intelligent_response(context, result)
        result['response'] = response
        
        return result
    
    async def _process_by_domains(
        self,
        context: IntelligentContext,
        image_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """도메인별 처리"""
        domain_results = {}
        
        # 각 도메인별 전문 처리 (병렬)
        domain_tasks = []
        
        for domain in context.detected_domains[:2]:  # 상위 2개 도메인
            task = asyncio.create_task(
                self._process_single_domain(domain, context, image_analysis)
            )
            domain_tasks.append((domain, task))
        
        # 완료 대기
        for domain, task in domain_tasks:
            try:
                result = await task
                domain_results[domain.value] = result
            except Exception as e:
                logger.error(f"Domain processing failed for {domain}: {e}")
                domain_results[domain.value] = {'error': str(e)}
        
        return domain_results
    
    async def _process_single_domain(
        self,
        domain: KnowledgeDomain,
        context: IntelligentContext,
        image_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """단일 도메인 처리"""
        # 도메인 전문가 활용
        if domain in self.knowledge_orchestrator.domain_experts:
            expert = self.knowledge_orchestrator.domain_experts[domain]
            
            # 컨텐츠 준비
            content = context.query
            if image_analysis and 'unified_content' in image_analysis:
                content += "\n" + image_analysis['unified_content']
            
            # 전문가 분석
            expert_analysis = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                expert.analyze,
                content
            )
            
            return expert_analysis
        
        return {'message': 'No expert available for this domain'}
    
    def _generate_learning_path(self, context: IntelligentContext) -> List[Dict[str, Any]]:
        """학습 경로 생성"""
        learning_path = []
        
        # 주요 개념 추출
        main_concepts = self._extract_main_concepts(context.query)
        
        if main_concepts:
            # 각 개념에 대한 학습 단계 생성
            for i, concept in enumerate(main_concepts[:3]):
                step = {
                    'order': i + 1,
                    'concept': concept,
                    'complexity': context.complexity_level.value,
                    'estimated_time': self._estimate_learning_time(context.complexity_level),
                    'prerequisites': [],
                    'resources': self._suggest_resources(concept, context.detected_domains)
                }
                learning_path.append(step)
        
        return learning_path
    
    def _extract_main_concepts(self, query: str) -> List[str]:
        """주요 개념 추출"""
        # 간단한 구현 - 실제로는 NLP 사용
        import re
        
        # 명사구 패턴
        noun_patterns = [
            r'[가-힣]+(?:의\s+)?[가-힣]+',
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            r'\b\w+(?:\s+\w+){0,2}\b'
        ]
        
        concepts = []
        for pattern in noun_patterns:
            matches = re.findall(pattern, query)
            concepts.extend(matches)
        
        # 중복 제거 및 필터링
        unique_concepts = list(set(concepts))
        
        # 불용어 제거
        stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '에서', 'the', 'is', 'at', 'which', 'on'}
        filtered_concepts = [c for c in unique_concepts if c.lower() not in stopwords and len(c) > 1]
        
        return filtered_concepts[:5]
    
    def _estimate_learning_time(self, complexity: ContentComplexity) -> int:
        """학습 시간 추정 (분)"""
        time_map = {
            ContentComplexity.ELEMENTARY: 30,
            ContentComplexity.MIDDLE_SCHOOL: 45,
            ContentComplexity.HIGH_SCHOOL: 60,
            ContentComplexity.UNDERGRADUATE: 90,
            ContentComplexity.GRADUATE: 120,
            ContentComplexity.RESEARCH: 180,
            ContentComplexity.EXPERT: 240
        }
        return time_map.get(complexity, 60)
    
    def _suggest_resources(self, concept: str, domains: List[KnowledgeDomain]) -> List[str]:
        """학습 자료 제안"""
        resources = ['textbook', 'online_tutorial']
        
        # 도메인별 추가 자료
        if KnowledgeDomain.MATHEMATICS in domains:
            resources.extend(['Khan Academy', 'MIT OCW Math'])
        elif KnowledgeDomain.COMPUTER in domains:
            resources.extend(['Coursera', 'LeetCode', 'GitHub'])
        elif KnowledgeDomain.AI_ML in domains:
            resources.extend(['fast.ai', 'Papers with Code', 'Google Colab'])
        
        return resources[:5]
    
    def _get_recommendations(self, context: IntelligentContext) -> List[Dict[str, Any]]:
        """추천 생성"""
        recommendations = []
        
        # 1. 다음 학습 주제
        if context.user_intent == 'learn':
            next_topics = self.knowledge_orchestrator.learning_assistant.recommend_next_topic(
                context.query[:50]
            )
            for topic in next_topics[:3]:
                recommendations.append({
                    'type': 'next_topic',
                    'content': topic['concept'],
                    'reason': topic['reason'],
                    'relevance': topic['relevance_score']
                })
        
        # 2. 관련 개념
        main_concepts = self._extract_main_concepts(context.query)
        if main_concepts:
            for concept in main_concepts[:2]:
                related = self.knowledge_orchestrator.knowledge_graph.get_related_concepts(
                    concept,
                    max_depth=1
                )
                for rel_id, depth in related[:2]:
                    node_data = self.knowledge_orchestrator.knowledge_graph.nodes.get(rel_id)
                    if node_data:
                        recommendations.append({
                            'type': 'related_concept',
                            'content': node_data['unit'].content[:50],
                            'reason': f"{concept}와(과) 관련됨",
                            'relevance': 0.8
                        })
        
        # 3. 연습 문제
        if context.user_intent in ['learn', 'solve']:
            recommendations.append({
                'type': 'practice',
                'content': '연습 문제 세트 생성 가능',
                'reason': '실력 향상을 위한 연습',
                'relevance': 0.7
            })
        
        return recommendations
    
    async def _generate_intelligent_response(
        self,
        context: IntelligentContext,
        analysis_result: Dict[str, Any]
    ) -> str:
        """지능형 응답 생성"""
        if not context:
            return "쿼리를 처리할 수 없습니다."
        
        # 응답 템플릿 선택
        template = self._select_response_template(context.response_strategy)
        
        # 컨텍스트 준비
        response_context = {
            'query': context.query,
            'domains': ', '.join([d.value for d in context.detected_domains]),
            'complexity': context.complexity_level.value,
            'intent': context.user_intent,
            'domain_analysis': analysis_result.get('domain_analysis', {}),
            'image_content': analysis_result.get('image_analysis', {}).get('unified_content', ''),
            'recommendations': analysis_result.get('recommendations', [])
        }
        
        # 템플릿 채우기
        response = template.format(**response_context)
        
        # 스타일 적용
        response = self._apply_response_style(response, context.response_strategy)
        
        return response
    
    def _select_response_template(self, strategy: ResponseStrategy) -> str:
        """응답 템플릿 선택"""
        templates = {
            ResponseStrategy.EDUCATIONAL: """
질문: {query}

관련 분야: {domains}
난이도: {complexity}

설명:
{domain_analysis}

학습 포인트:
1. 핵심 개념 이해
2. 실습을 통한 적용
3. 관련 주제 탐색

추천 학습 자료:
{recommendations}
""",
            ResponseStrategy.TECHNICAL: """
Query: {query}
Domains: {domains}
Complexity: {complexity}

Technical Analysis:
{domain_analysis}

Implementation Details:
- Architecture considerations
- Performance optimization
- Best practices

Related Topics:
{recommendations}
""",
            ResponseStrategy.PRACTICAL: """
문제: {query}

실용적 해결 방법:
{domain_analysis}

단계별 가이드:
1. 문제 분석
2. 해결 방법 선택
3. 구현 및 테스트

추가 리소스:
{recommendations}
""",
            ResponseStrategy.ANALYTICAL: """
분석 대상: {query}

도메인별 분석:
{domain_analysis}

주요 발견사항:
- 핵심 패턴
- 상관 관계
- 개선 가능성

관련 분석:
{recommendations}
""",
            ResponseStrategy.CONVERSATIONAL: """
{query}에 대해 말씀드리겠습니다.

{domain_analysis}

이와 관련해서 다음도 참고하시면 좋을 것 같습니다:
{recommendations}
""",
            ResponseStrategy.CREATIVE: """
"{query}"라는 주제로 창의적으로 접근해보겠습니다.

{domain_analysis}

영감을 주는 관련 아이디어:
{recommendations}

새로운 관점에서 바라보면 더 많은 가능성이 열릴 수 있습니다.
"""
        }
        
        return templates.get(strategy, templates[ResponseStrategy.CONVERSATIONAL])
    
    def _apply_response_style(self, response: str, strategy: ResponseStrategy) -> str:
        """응답 스타일 적용"""
        # 전략별 스타일 조정
        if strategy == ResponseStrategy.EDUCATIONAL:
            # 친근한 톤
            response = response.replace("설명:", "함께 알아보겠습니다:")
            response = response.replace("추천", "도움이 될 만한")
        
        elif strategy == ResponseStrategy.TECHNICAL:
            # 전문적 톤 유지
            pass
        
        elif strategy == ResponseStrategy.CONVERSATIONAL:
            # 대화체
            if not response.endswith(('습니다.', '니다.', '세요.', '입니다.')):
                response += "습니다."
        
        return response
    
    def _generate_cache_key(self, query: str, has_image: bool) -> str:
        """캐시 키 생성"""
        import hashlib
        
        key_parts = [query, str(has_image)]
        key_string = '|'.join(key_parts)
        
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def generate_practice_problems(
        self,
        concept: str,
        domain: KnowledgeDomain,
        difficulty: ContentComplexity,
        count: int = 5
    ) -> List[Dict[str, Any]]:
        """연습 문제 생성"""
        return self.knowledge_orchestrator.learning_assistant.generate_practice_problems(
            concept=concept,
            difficulty=difficulty,
            count=count
        )
    
    def create_personalized_curriculum(
        self,
        user_profile: Dict[str, Any],
        target_domain: KnowledgeDomain,
        duration_weeks: int
    ) -> Dict[str, Any]:
        """개인화 커리큘럼 생성"""
        # 사용자 현재 수준 파악
        current_level = ContentComplexity(
            user_profile.get('current_level', ContentComplexity.UNDERGRADUATE.value)
        )
        
        # 목표 수준 설정
        target_level = ContentComplexity(
            user_profile.get('target_level', ContentComplexity.GRADUATE.value)
        )
        
        # 커리큘럼 생성
        curriculum = self.knowledge_orchestrator.create_curriculum(
            target_domain=target_domain,
            start_level=current_level,
            end_level=target_level,
            duration_weeks=duration_weeks
        )
        
        # 개인화 요소 추가
        curriculum['personalization'] = {
            'learning_style': user_profile.get('learning_style', 'balanced'),
            'time_per_week': user_profile.get('available_hours_per_week', 10),
            'preferred_resources': user_profile.get('preferred_resources', []),
            'strengths': user_profile.get('strengths', []),
            'weaknesses': user_profile.get('weaknesses', [])
        }
        
        return curriculum


# 사용 예시
if __name__ == "__main__":
    # 설정
    config = {
        'vector_db': None,  # 실제 벡터 DB
        'llm_client': None,  # 실제 LLM 클라이언트
        'cache_size': 100,
        'max_workers': 4
    }
    
    # 시스템 초기화
    orchestrator = IntelligentRAGOrchestrator(config)
    
    # 동기 처리 예시
    result = orchestrator.process_sync(
        query="미분방정식을 이용한 회로 해석 방법을 설명해주세요",
        context={
            'user_profile': {
                'education_level': 'undergraduate',
                'preferred_domains': ['electrical', 'mathematics']
            }
        }
    )
    
    print(f"Detected domains: {result['domains']}")
    print(f"Complexity: {result['complexity']}")
    print(f"Strategy: {result['strategy']}")
    print(f"Processing time: {result.get('processing_time', 0):.2f}s")
    
    # 비동기 처리 예시
    async def async_example():
        result = await orchestrator.process_async(
            query="인공지능의 최신 트렌드와 미래 전망",
            context={'user_profile': {'interests': ['AI', 'ML']}}
        )
        return result
    
    # asyncio.run(async_example())