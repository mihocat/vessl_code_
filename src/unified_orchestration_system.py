#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Orchestration System
통합 오케스트레이션 시스템 - 모든 AI 시스템의 지휘자
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from datetime import datetime
import json
import hashlib
from collections import defaultdict
import numpy as np

# 기존 시스템들
from cognitive_ai_system import CognitiveAISystem
from intelligent_rag_system import IntelligentRAGOrchestrator
from universal_knowledge_system import UniversalKnowledgeOrchestrator
from advanced_rag_system import AdvancedRAGSystem
from modular_rag_system import ModularRAGPipeline
from universal_ocr_pipeline import DomainAdaptiveOCR

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """처리 모드"""
    COGNITIVE = "cognitive"  # 인지적 처리
    INTELLIGENT = "intelligent"  # 지능형 RAG
    ADVANCED = "advanced"  # 고급 RAG
    MODULAR = "modular"  # 모듈형 RAG
    HYBRID = "hybrid"  # 하이브리드
    ADAPTIVE = "adaptive"  # 적응형


class SystemState(Enum):
    """시스템 상태"""
    IDLE = "idle"
    PROCESSING = "processing"
    LEARNING = "learning"
    ADAPTING = "adapting"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class UserContext:
    """사용자 컨텍스트"""
    user_id: str
    session_id: str
    profile: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    current_task: Optional[str] = None
    learning_progress: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)


@dataclass
class SystemMetrics:
    """시스템 메트릭"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    mode_usage: Dict[str, int] = field(default_factory=dict)
    domain_distribution: Dict[str, int] = field(default_factory=dict)
    user_satisfaction: float = 0.0
    system_load: float = 0.0
    memory_usage: float = 0.0
    cache_hit_rate: float = 0.0


class UnifiedOrchestrationSystem:
    """통합 오케스트레이션 시스템"""
    
    def __init__(self, config: Dict[str, Any]):
        """초기화"""
        self.config = config
        self.state = SystemState.IDLE
        
        # 하위 시스템 초기화
        self._initialize_subsystems()
        
        # 사용자 관리
        self.user_contexts: Dict[str, UserContext] = {}
        
        # 시스템 메트릭
        self.metrics = SystemMetrics()
        
        # 캐시
        self.response_cache: Dict[str, Any] = {}
        self.cache_size_limit = config.get('cache_size_limit', 1000)
        
        # 실행기
        self.executor = asyncio.get_event_loop().run_in_executor
        
        # 학습 데이터
        self.feedback_buffer: List[Dict[str, Any]] = []
        self.adaptation_threshold = config.get('adaptation_threshold', 100)
        
        logger.info("Unified Orchestration System initialized")
    
    def _initialize_subsystems(self):
        """하위 시스템 초기화"""
        try:
            # Cognitive AI System
            self.cognitive_system = CognitiveAISystem()
            logger.info("Cognitive AI System loaded")
        except Exception as e:
            logger.error(f"Failed to load Cognitive AI System: {e}")
            self.cognitive_system = None
        
        try:
            # Intelligent RAG Orchestrator
            self.intelligent_rag = IntelligentRAGOrchestrator(self.config)
            logger.info("Intelligent RAG Orchestrator loaded")
        except Exception as e:
            logger.error(f"Failed to load Intelligent RAG: {e}")
            self.intelligent_rag = None
        
        try:
            # Universal Knowledge Orchestrator
            self.knowledge_system = UniversalKnowledgeOrchestrator()
            logger.info("Universal Knowledge System loaded")
        except Exception as e:
            logger.error(f"Failed to load Knowledge System: {e}")
            self.knowledge_system = None
        
        try:
            # Advanced RAG System
            if 'vector_db' in self.config and 'llm_client' in self.config:
                self.advanced_rag = AdvancedRAGSystem(
                    self.config['vector_db'],
                    self.config['llm_client']
                )
                logger.info("Advanced RAG System loaded")
            else:
                self.advanced_rag = None
        except Exception as e:
            logger.error(f"Failed to load Advanced RAG: {e}")
            self.advanced_rag = None
        
        try:
            # Modular RAG Pipeline
            self.modular_rag = ModularRAGPipeline(self.config)
            logger.info("Modular RAG Pipeline loaded")
        except Exception as e:
            logger.error(f"Failed to load Modular RAG: {e}")
            self.modular_rag = None
        
        try:
            # Domain Adaptive OCR
            self.ocr_system = DomainAdaptiveOCR()
            logger.info("Domain Adaptive OCR loaded")
        except Exception as e:
            logger.error(f"Failed to load OCR System: {e}")
            self.ocr_system = None
    
    async def process(
        self,
        query: str,
        user_id: str,
        image: Optional[Any] = None,
        mode: Optional[ProcessingMode] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """통합 처리"""
        start_time = time.time()
        self.state = SystemState.PROCESSING
        
        try:
            # 1. 사용자 컨텍스트 관리
            user_context = self._get_or_create_user_context(user_id)
            user_context.last_interaction = datetime.now()
            
            # 2. 캐시 확인
            cache_key = self._generate_cache_key(query, image is not None, user_id)
            if cache_key in self.response_cache:
                self.metrics.cache_hit_rate = self._update_cache_hit_rate(True)
                cached_response = self.response_cache[cache_key]
                cached_response['from_cache'] = True
                return cached_response
            
            self.metrics.cache_hit_rate = self._update_cache_hit_rate(False)
            
            # 3. 처리 모드 결정
            if mode is None:
                mode = await self._determine_processing_mode(query, image, user_context)
            
            # 4. 메트릭 업데이트
            self.metrics.total_requests += 1
            self.metrics.mode_usage[mode.value] = self.metrics.mode_usage.get(mode.value, 0) + 1
            
            # 5. 모드별 처리
            result = await self._process_by_mode(
                mode, query, image, user_context, context
            )
            
            # 6. 결과 향상
            result = await self._enhance_result(result, user_context)
            
            # 7. 사용자 히스토리 업데이트
            self._update_user_history(user_context, query, result)
            
            # 8. 캐시 저장
            self._save_to_cache(cache_key, result)
            
            # 9. 메트릭 업데이트
            self.metrics.successful_requests += 1
            processing_time = time.time() - start_time
            self._update_response_time(processing_time)
            
            result['processing_time'] = processing_time
            result['mode'] = mode.value
            
            self.state = SystemState.IDLE
            return result
            
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            self.metrics.failed_requests += 1
            self.state = SystemState.ERROR
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'fallback_response': self._generate_fallback_response(query)
            }
    
    async def _determine_processing_mode(
        self,
        query: str,
        image: Optional[Any],
        user_context: UserContext
    ) -> ProcessingMode:
        """처리 모드 자동 결정"""
        # 1. 사용자 선호도 확인
        if user_context.preferences.get('preferred_mode'):
            return ProcessingMode(user_context.preferences['preferred_mode'])
        
        # 2. 쿼리 복잡도 분석
        complexity_score = await self._analyze_query_complexity(query, image)
        
        # 3. 사용 가능한 시스템 확인
        available_systems = self._get_available_systems()
        
        # 4. 모드 결정 로직
        if complexity_score > 0.8 and self.cognitive_system in available_systems:
            return ProcessingMode.COGNITIVE
        elif complexity_score > 0.6 and self.intelligent_rag in available_systems:
            return ProcessingMode.INTELLIGENT
        elif complexity_score > 0.4 and self.advanced_rag in available_systems:
            return ProcessingMode.ADVANCED
        elif self.modular_rag in available_systems:
            return ProcessingMode.MODULAR
        else:
            return ProcessingMode.ADAPTIVE
    
    async def _analyze_query_complexity(
        self,
        query: str,
        image: Optional[Any]
    ) -> float:
        """쿼리 복잡도 분석"""
        complexity_score = 0.0
        
        # 길이 기반
        complexity_score += min(len(query) / 500, 0.3)
        
        # 멀티모달 여부
        if image:
            complexity_score += 0.2
        
        # 추론 필요 여부
        reasoning_keywords = ['왜', '어떻게', '분석', '비교', '평가', '추론', '설명']
        if any(keyword in query for keyword in reasoning_keywords):
            complexity_score += 0.3
        
        # 도메인 전문성
        technical_keywords = ['수식', '알고리즘', '이론', '증명', '법칙', '정리']
        if any(keyword in query for keyword in technical_keywords):
            complexity_score += 0.2
        
        return min(complexity_score, 1.0)
    
    async def _process_by_mode(
        self,
        mode: ProcessingMode,
        query: str,
        image: Optional[Any],
        user_context: UserContext,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """모드별 처리"""
        # 컨텍스트 준비
        enhanced_context = self._prepare_context(user_context, context)
        
        if mode == ProcessingMode.COGNITIVE and self.cognitive_system:
            # Cognitive AI 처리
            input_data = {
                'text': query,
                'user_profile': user_context.profile,
                'history': user_context.history[-5:]  # 최근 5개 대화
            }
            if image:
                input_data['image'] = image
            
            result = await self.cognitive_system.think(input_data)
            return self._format_cognitive_result(result)
        
        elif mode == ProcessingMode.INTELLIGENT and self.intelligent_rag:
            # Intelligent RAG 처리
            result = await self.intelligent_rag.process_async(
                query, image, enhanced_context
            )
            return result
        
        elif mode == ProcessingMode.ADVANCED and self.advanced_rag:
            # Advanced RAG 처리
            result = self.advanced_rag.process_query_advanced(
                query, image, mode='reasoning'
            )
            return result
        
        elif mode == ProcessingMode.MODULAR and self.modular_rag:
            # Modular RAG 처리
            result = self.modular_rag.process(query, image)
            return result
        
        else:
            # 적응형 처리 (하이브리드)
            return await self._adaptive_processing(
                query, image, user_context, enhanced_context
            )
    
    async def _adaptive_processing(
        self,
        query: str,
        image: Optional[Any],
        user_context: UserContext,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """적응형 처리 - 여러 시스템 조합"""
        results = []
        
        # 병렬 처리
        tasks = []
        
        if self.intelligent_rag:
            tasks.append(('intelligent', self.intelligent_rag.process_async(
                query, image, context
            )))
        
        if self.modular_rag:
            tasks.append(('modular', asyncio.create_task(
                asyncio.to_thread(self.modular_rag.process, query, image)
            )))
        
        # 결과 수집
        for name, task in tasks:
            try:
                result = await task
                results.append((name, result))
            except Exception as e:
                logger.error(f"{name} processing failed: {e}")
        
        # 결과 통합
        if not results:
            return {
                'success': False,
                'error': 'No processing system available'
            }
        
        # 최적 결과 선택 또는 병합
        return self._merge_results(results, query)
    
    def _merge_results(
        self,
        results: List[Tuple[str, Dict[str, Any]]],
        query: str
    ) -> Dict[str, Any]:
        """결과 병합"""
        # 성공한 결과만 필터링
        successful_results = [
            (name, result) for name, result in results
            if result.get('success', False)
        ]
        
        if not successful_results:
            return results[0][1]  # 첫 번째 결과 반환
        
        # 가장 신뢰도 높은 결과 선택
        best_result = None
        best_score = -1
        
        for name, result in successful_results:
            score = result.get('confidence', 0.5)
            if 'evidence_strength' in result:
                score = result['evidence_strength']
            
            if score > best_score:
                best_score = score
                best_result = result
        
        # 다른 결과에서 유용한 정보 추가
        if best_result:
            best_result['merged_from'] = [name for name, _ in results]
            best_result['alternative_responses'] = [
                {
                    'system': name,
                    'response': result.get('response', '')[:200] + '...'
                }
                for name, result in successful_results
                if result != best_result
            ]
        
        return best_result or results[0][1]
    
    async def _enhance_result(
        self,
        result: Dict[str, Any],
        user_context: UserContext
    ) -> Dict[str, Any]:
        """결과 향상"""
        # 1. 개인화
        if user_context.profile.get('preferred_language'):
            result['language'] = user_context.profile['preferred_language']
        
        # 2. 학습 추천
        if user_context.learning_progress:
            result['learning_recommendations'] = self._generate_learning_recommendations(
                user_context
            )
        
        # 3. 관련 리소스
        if 'domains' in result:
            result['related_resources'] = self._get_related_resources(
                result['domains']
            )
        
        # 4. 피드백 요청
        result['feedback_requested'] = self._should_request_feedback(user_context)
        
        return result
    
    def _get_or_create_user_context(self, user_id: str) -> UserContext:
        """사용자 컨텍스트 가져오기 또는 생성"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = UserContext(
                user_id=user_id,
                session_id=self._generate_session_id()
            )
        return self.user_contexts[user_id]
    
    def _prepare_context(
        self,
        user_context: UserContext,
        external_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """컨텍스트 준비"""
        context = {
            'user_profile': user_context.profile,
            'conversation_history': user_context.history[-5:],
            'learning_progress': user_context.learning_progress,
            'timestamp': datetime.now().isoformat()
        }
        
        if external_context:
            context.update(external_context)
        
        return context
    
    def _update_user_history(
        self,
        user_context: UserContext,
        query: str,
        result: Dict[str, Any]
    ):
        """사용자 히스토리 업데이트"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': result.get('response', '')[:500],  # 요약
            'mode': result.get('mode', 'unknown'),
            'success': result.get('success', False),
            'domains': result.get('domains', [])
        }
        
        user_context.history.append(history_entry)
        
        # 히스토리 크기 제한
        if len(user_context.history) > 100:
            user_context.history = user_context.history[-100:]
    
    def _generate_cache_key(
        self,
        query: str,
        has_image: bool,
        user_id: str
    ) -> str:
        """캐시 키 생성"""
        key_parts = [query, str(has_image), user_id[:8]]
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _save_to_cache(self, key: str, result: Dict[str, Any]):
        """캐시에 저장"""
        # 캐시 크기 제한
        if len(self.response_cache) >= self.cache_size_limit:
            # LRU 방식으로 오래된 항목 제거
            oldest_key = min(
                self.response_cache.keys(),
                key=lambda k: self.response_cache[k].get('cached_at', 0)
            )
            del self.response_cache[oldest_key]
        
        result_copy = result.copy()
        result_copy['cached_at'] = time.time()
        self.response_cache[key] = result_copy
    
    def _update_cache_hit_rate(self, hit: bool) -> float:
        """캐시 히트율 업데이트"""
        # 간단한 이동 평균
        current_rate = self.metrics.cache_hit_rate
        weight = 0.95
        new_value = 1.0 if hit else 0.0
        return weight * current_rate + (1 - weight) * new_value
    
    def _update_response_time(self, new_time: float):
        """응답 시간 업데이트"""
        current_avg = self.metrics.average_response_time
        total_requests = self.metrics.successful_requests
        
        if total_requests == 1:
            self.metrics.average_response_time = new_time
        else:
            # 누적 평균
            self.metrics.average_response_time = (
                (current_avg * (total_requests - 1) + new_time) / total_requests
            )
    
    def _get_available_systems(self) -> List[Any]:
        """사용 가능한 시스템 목록"""
        available = []
        
        if self.cognitive_system:
            available.append(self.cognitive_system)
        if self.intelligent_rag:
            available.append(self.intelligent_rag)
        if self.advanced_rag:
            available.append(self.advanced_rag)
        if self.modular_rag:
            available.append(self.modular_rag)
        
        return available
    
    def _format_cognitive_result(self, cognitive_result: Dict[str, Any]) -> Dict[str, Any]:
        """인지 시스템 결과 포맷팅"""
        formatted = {
            'success': True,
            'response': cognitive_result.get('reasoning', {}).get('conclusion', ''),
            'cognitive_analysis': {
                'perception': cognitive_result.get('perception', {}),
                'attention_focus': cognitive_result.get('attention', {}).get('primary_focus'),
                'memory_retrieval': cognitive_result.get('memory', {}).get('retrieved_memories', []),
                'reasoning_type': cognitive_result.get('reasoning', {}).get('primary_method'),
                'creative_insights': cognitive_result.get('creativity', {}).get('insights', []),
                'confidence': cognitive_result.get('metacognition', {}).get('confidence', 0.5)
            },
            'processing_trace': cognitive_result.get('processing_log', [])
        }
        
        return formatted
    
    def _generate_learning_recommendations(
        self,
        user_context: UserContext
    ) -> List[Dict[str, str]]:
        """학습 추천 생성"""
        recommendations = []
        
        # 최근 관심 도메인 분석
        recent_domains = []
        for entry in user_context.history[-10:]:
            recent_domains.extend(entry.get('domains', []))
        
        # 도메인 빈도 계산
        domain_counts = defaultdict(int)
        for domain in recent_domains:
            domain_counts[domain] += 1
        
        # 상위 도메인에 대한 추천
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            recommendations.append({
                'domain': domain,
                'reason': f'최근 {count}번 관련 질문',
                'suggested_topic': f'{domain} 심화 학습'
            })
        
        return recommendations
    
    def _get_related_resources(self, domains: List[str]) -> List[Dict[str, str]]:
        """관련 리소스 가져오기"""
        resources = []
        
        resource_map = {
            'mathematics': [
                {'type': 'course', 'name': 'Khan Academy Mathematics'},
                {'type': 'book', 'name': 'Calculus by Stewart'}
            ],
            'ai_ml': [
                {'type': 'course', 'name': 'fast.ai'},
                {'type': 'platform', 'name': 'Kaggle'}
            ],
            'physics': [
                {'type': 'simulation', 'name': 'PhET Simulations'},
                {'type': 'textbook', 'name': 'Feynman Lectures'}
            ]
        }
        
        for domain in domains[:2]:  # 상위 2개 도메인
            if domain in resource_map:
                resources.extend(resource_map[domain])
        
        return resources[:5]  # 최대 5개
    
    def _should_request_feedback(self, user_context: UserContext) -> bool:
        """피드백 요청 여부 결정"""
        # 마지막 피드백 시간 확인
        last_feedback = user_context.profile.get('last_feedback_time', 0)
        current_time = time.time()
        
        # 24시간마다 한 번씩 요청
        if current_time - last_feedback > 86400:  # 24시간
            return True
        
        # 10번 상호작용마다 한 번씩 요청
        interaction_count = len(user_context.history)
        if interaction_count % 10 == 0:
            return True
        
        return False
    
    def _generate_fallback_response(self, query: str) -> str:
        """폴백 응답 생성"""
        return f"""
죄송합니다. 현재 시스템에서 "{query}"에 대한 처리 중 오류가 발생했습니다.

다음과 같은 방법을 시도해보세요:
1. 질문을 더 구체적으로 다시 작성해보세요
2. 질문을 여러 개의 작은 질문으로 나누어보세요
3. 잠시 후 다시 시도해보세요

문제가 지속되면 시스템 관리자에게 문의해주세요.
"""
    
    def _generate_session_id(self) -> str:
        """세션 ID 생성"""
        return hashlib.md5(
            f"{time.time()}_{np.random.rand()}".encode()
        ).hexdigest()[:16]
    
    async def collect_feedback(
        self,
        user_id: str,
        query: str,
        response: str,
        rating: int,
        feedback: Optional[str] = None
    ):
        """사용자 피드백 수집"""
        feedback_entry = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response[:500],
            'rating': rating,
            'feedback': feedback
        }
        
        self.feedback_buffer.append(feedback_entry)
        
        # 사용자 프로필 업데이트
        user_context = self._get_or_create_user_context(user_id)
        user_context.profile['last_feedback_time'] = time.time()
        
        # 적응 임계값 도달 시 시스템 적응
        if len(self.feedback_buffer) >= self.adaptation_threshold:
            await self._adapt_system()
    
    async def _adapt_system(self):
        """시스템 적응"""
        self.state = SystemState.ADAPTING
        
        try:
            # 피드백 분석
            positive_feedback = [
                f for f in self.feedback_buffer if f['rating'] >= 4
            ]
            negative_feedback = [
                f for f in self.feedback_buffer if f['rating'] <= 2
            ]
            
            # 긍정적 피드백에서 패턴 학습
            if positive_feedback:
                logger.info(f"Learning from {len(positive_feedback)} positive feedbacks")
                # TODO: 실제 학습 로직 구현
            
            # 부정적 피드백에서 개선점 도출
            if negative_feedback:
                logger.info(f"Analyzing {len(negative_feedback)} negative feedbacks")
                # TODO: 개선 로직 구현
            
            # 버퍼 초기화
            self.feedback_buffer = []
            
            # 사용자 만족도 업데이트
            total_ratings = sum(f['rating'] for f in self.feedback_buffer)
            if self.feedback_buffer:
                self.metrics.user_satisfaction = total_ratings / len(self.feedback_buffer)
            
        except Exception as e:
            logger.error(f"System adaptation failed: {e}")
        finally:
            self.state = SystemState.IDLE
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        return {
            'state': self.state.value,
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'success_rate': (
                    self.metrics.successful_requests / self.metrics.total_requests
                    if self.metrics.total_requests > 0 else 0
                ),
                'average_response_time': self.metrics.average_response_time,
                'cache_hit_rate': self.metrics.cache_hit_rate,
                'user_satisfaction': self.metrics.user_satisfaction,
                'mode_usage': dict(self.metrics.mode_usage),
                'active_users': len(self.user_contexts)
            },
            'subsystems': {
                'cognitive': self.cognitive_system is not None,
                'intelligent_rag': self.intelligent_rag is not None,
                'advanced_rag': self.advanced_rag is not None,
                'modular_rag': self.modular_rag is not None,
                'ocr': self.ocr_system is not None
            },
            'cache_size': len(self.response_cache),
            'feedback_buffer_size': len(self.feedback_buffer)
        }
    
    async def maintenance_mode(self, enable: bool = True):
        """유지보수 모드"""
        if enable:
            self.state = SystemState.MAINTENANCE
            logger.info("System entering maintenance mode")
            
            # 캐시 정리
            self.response_cache.clear()
            
            # 메트릭 백업
            # TODO: 메트릭 저장 로직
            
        else:
            self.state = SystemState.IDLE
            logger.info("System exiting maintenance mode")


# 사용 예시
if __name__ == "__main__":
    # 설정
    config = {
        'cache_size_limit': 1000,
        'adaptation_threshold': 100,
        'llm_client': None,  # 실제 LLM 클라이언트
        'vector_db': None    # 실제 벡터 DB
    }
    
    # 시스템 초기화
    orchestrator = UnifiedOrchestrationSystem(config)
    
    # 비동기 처리 예시
    async def test_unified_system():
        # 테스트 쿼리
        queries = [
            "양자역학의 불확정성 원리를 철학적 관점에서 설명해주세요",
            "머신러닝 모델의 과적합을 방지하는 방법들을 비교 분석해주세요",
            "바흐의 음악이 수학적 구조를 가지는 이유는 무엇인가요?"
        ]
        
        for query in queries:
            print(f"\n질문: {query}")
            
            result = await orchestrator.process(
                query=query,
                user_id="test_user_001"
            )
            
            if result['success']:
                print(f"응답: {result['response'][:200]}...")
                print(f"처리 모드: {result['mode']}")
                print(f"처리 시간: {result['processing_time']:.2f}초")
            else:
                print(f"오류: {result['error']}")
        
        # 시스템 상태 확인
        status = orchestrator.get_system_status()
        print(f"\n시스템 상태: {json.dumps(status, indent=2, ensure_ascii=False)}")
    
    # asyncio.run(test_unified_system())