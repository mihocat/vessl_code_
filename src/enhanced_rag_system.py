#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
향상된 도메인 무관 RAG 시스템
Enhanced Domain-Agnostic RAG System
최신 AI 트렌드 반영: 다중 검색 전략, 적응형 검색, 지능형 필터링, 자동 도메인 감지
"""

import logging
import time
import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np
from collections import defaultdict

try:
    from rag_system import RAGSystem
except ImportError:
    # Fallback for testing without dependencies
    class RAGSystem:
        def __init__(self, config):
            pass
        def search(self, query, max_results=10):
            return {'results': []}

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """검색 전략"""
    SEMANTIC = "semantic"                # 의미적 유사도 검색
    KEYWORD = "keyword"                  # 키워드 기반 검색  
    HYBRID = "hybrid"                    # 하이브리드 검색
    MULTI_QUERY = "multi_query"          # 다중 쿼리 검색
    CHAIN_OF_THOUGHT = "chain_of_thought"  # 사고 과정 기반 검색
    ADAPTIVE = "adaptive"                # 적응형 검색


class QueryExpansionMethod(Enum):
    """쿼리 확장 방법"""
    SYNONYM = "synonym"                  # 동의어 확장
    DOMAIN_SPECIFIC = "domain_specific"  # 도메인 특화 확장
    STRUCTURAL = "structural"            # 구조적 확장
    LLM_BASED = "llm_based"             # LLM 기반 확장
    CONTEXT_AWARE = "context_aware"      # 컨텍스트 인식 확장


class ConfidenceLevel(Enum):
    """신뢰도 수준"""
    VERY_HIGH = 0.9
    HIGH = 0.75
    MEDIUM = 0.6
    LOW = 0.45
    VERY_LOW = 0.3


class DomainType(Enum):
    """도메인 타입 (확장 가능)"""
    ELECTRICAL = "electrical"
    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    COMPUTER_SCIENCE = "computer_science"
    GENERAL_SCIENCE = "general_science"
    ENGINEERING = "engineering"
    GENERAL = "general"


@dataclass
class SearchResult:
    """향상된 검색 결과"""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    relevance_type: str  # semantic, keyword, hybrid
    
    # 추가 메타데이터
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    domain: DomainType = DomainType.GENERAL
    content_type: str = "general"  # definition, procedure, example, etc.
    keywords: List[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        self.confidence_level = self._determine_confidence_level()
    
    def _determine_confidence_level(self) -> ConfidenceLevel:
        """점수 기반 신뢰도 레벨 결정"""
        if self.score >= ConfidenceLevel.VERY_HIGH.value:
            return ConfidenceLevel.VERY_HIGH
        elif self.score >= ConfidenceLevel.HIGH.value:
            return ConfidenceLevel.HIGH
        elif self.score >= ConfidenceLevel.MEDIUM.value:
            return ConfidenceLevel.MEDIUM
        elif self.score >= ConfidenceLevel.LOW.value:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


@dataclass
class QueryAnalysis:
    """쿼리 분석 결과"""
    original_query: str
    processed_query: str
    keywords: List[str]
    detected_domains: List[DomainType]
    query_intent: str  # question, request, search, etc.
    complexity_score: float
    language: str = "mixed"
    expanded_queries: List[str] = None
    
    def __post_init__(self):
        if self.expanded_queries is None:
            self.expanded_queries = []


@dataclass 
class EnhancedSearchResult:
    """향상된 검색 결과"""
    results: List[SearchResult]
    total_found: int
    search_time: float
    strategy_used: SearchStrategy
    confidence: float
    
    # 쿼리 분석 정보
    query_analysis: QueryAnalysis = None
    
    # 고급 메트릭
    coverage_score: float = 0.0      # 쿼리 커버리지 점수
    diversity_score: float = 0.0     # 결과 다양성 점수
    quality_score: float = 0.0       # 결과 품질 점수
    relevance_score: float = 0.0     # 관련성 점수
    
    # 검색 상세 정보
    queries_used: List[str] = None
    fusion_weights: Dict[str, float] = None
    fallback_used: bool = False
    
    # 성능 메트릭
    cache_hit: bool = False
    processing_steps: List[str] = None
    
    def __post_init__(self):
        if self.queries_used is None:
            self.queries_used = []
        if self.fusion_weights is None:
            self.fusion_weights = {}
        if self.processing_steps is None:
            self.processing_steps = []


class EnhancedRAGSystem:
    """향상된 RAG 시스템"""
    
    def __init__(self, config):
        """
        시스템 초기화
        
        Args:
            config: 설정 객체
        """
        self.config = config
        
        # 기존 RAG 시스템 활용
        self.base_rag = RAGSystem(config)
        
        # 도메인별 키워드 사전
        self.domain_keywords = self._load_domain_keywords()
        
        # 동의어 사전  
        self.synonym_dict = self._load_synonym_dict()
        
        # 검색 통계
        self.search_stats = {
            'total_searches': 0,
            'strategy_usage': defaultdict(int),
            'average_results': 0.0,
            'average_confidence': 0.0,
            'cache_hits': 0
        }
        
        # 결과 캐시 (간단한 구현)
        self.result_cache = {}
        self.cache_size_limit = 1000
        
        logger.info("Enhanced RAG System initialized")
    
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """도메인별 키워드 로드 (확장된 버전)"""
        return {
            'electrical': [
                # 한국어 키워드
                '전기', '전압', '전류', '저항', '전력', '회로', '임피던스', '역률',
                '콘덴서', '인덕터', '변압기', '모터', '발전기', '반도체', 'LED',
                '다이오드', '트랜지스터', '전기기기', '전력시스템', '제어',
                '센서', '액추에이터', '스위치', '릴레이', '퓨즈',
                # 영어 키워드
                'voltage', 'current', 'resistance', 'power', 'circuit', 'impedance',
                'capacitor', 'inductor', 'transformer', 'motor', 'generator',
                'semiconductor', 'diode', 'transistor', 'relay', 'sensor'
            ],
            'mathematics': [
                # 한국어 키워드
                '수학', '함수', '미분', '적분', '방정식', '행렬', '벡터', '확률',
                '통계', '기하', '대수', '삼각함수', '로그', '지수', '극한',
                '급수', '수열', '집합', '논리', '증명', '정리',
                '미적분', '선형대수', '해석학', '위상수학', '정수론',
                # 영어 키워드
                'mathematics', 'function', 'derivative', 'integral', 'equation',
                'matrix', 'vector', 'probability', 'statistics', 'geometry',
                'algebra', 'trigonometry', 'logarithm', 'calculus', 'linear'
            ],
            'physics': [
                # 한국어 키워드
                '물리', '힘', '에너지', '운동', '파동', '열', '광학', '양자',
                '상대성', '중력', '전자기', '역학', '유체', '열역학',
                '입자', '원자', '핵물리', '고체물리', '플라즈마',
                '진동', '파장', '주파수', '속도', '가속도', '질량',
                # 영어 키워드
                'physics', 'force', 'energy', 'motion', 'wave', 'quantum',
                'relativity', 'gravity', 'electromagnetic', 'mechanics',
                'thermodynamics', 'optics', 'particle', 'nuclear', 'plasma'
            ],
            'chemistry': [
                # 한국어 키워드
                '화학', '원소', '분자', '반응', '화합물', '이온', '산화', '환원',
                '촉매', '평형', '농도', '몰', '결합', '구조', '유기화학',
                '무기화학', '물리화학', '분석화학', '생화학', '고분자',
                '산', '염기', '염', '용해', '침전', '전해질',
                # 영어 키워드
                'chemistry', 'element', 'molecule', 'reaction', 'compound',
                'ion', 'catalyst', 'equilibrium', 'concentration', 'organic',
                'inorganic', 'acid', 'base', 'salt', 'polymer', 'biochemistry'
            ],
            'computer_science': [
                # 한국어 키워드
                '컴퓨터', '프로그래밍', '알고리즘', '데이터구조', '네트워크', 'AI',
                '머신러닝', '데이터베이스', '소프트웨어', '하드웨어', '보안',
                '운영체제', '컴파일러', '인터넷', '웹', '앱', '게임',
                '그래픽', '인공지능', '딥러닝', '빅데이터', '클라우드',
                # 영어 키워드
                'computer', 'programming', 'algorithm', 'data', 'structure',
                'network', 'machine learning', 'database', 'software',
                'hardware', 'security', 'artificial intelligence', 'web'
            ],
            'engineering': [
                # 한국어 키워드
                '공학', '설계', '제조', '재료', '구조', '시스템', '제어',
                '기계공학', '화학공학', '토목공학', '산업공학', '환경공학',
                '생체공학', '항공공학', '선박공학', '건축',
                '품질', '안전', '효율', '최적화', '자동화',
                # 영어 키워드
                'engineering', 'design', 'manufacturing', 'materials',
                'mechanical', 'chemical', 'civil', 'industrial', 'aerospace',
                'biomedical', 'environmental', 'quality', 'safety', 'optimization'
            ],
            'general_science': [
                # 한국어 키워드
                '과학', '연구', '실험', '이론', '가설', '증명', '분석', '측정',
                '관찰', '데이터', '결과', '결론', '방법', '원리', '법칙',
                '모델', '시뮬레이션', '검증', '예측', '발견',
                # 영어 키워드
                'science', 'research', 'experiment', 'theory', 'hypothesis',
                'analysis', 'measurement', 'observation', 'data', 'method',
                'principle', 'law', 'model', 'simulation', 'discovery'
            ]
        }
    
    def _load_synonym_dict(self) -> Dict[str, List[str]]:
        """동의어 사전 로드"""
        return {
            # 기술
            '데이터': ['data', 'information', '정보', '자료'],
            '알고리즘': ['algorithm', 'algo', '수식', '방법'],
            '프로그램': ['program', 'software', '소프트웨어', '코드'],
            '시스템': ['system', '체계', '환경'],
            '네트워크': ['network', '연결망', '통신망'],
            
            # 수학
            '함수': ['function', 'f(x)', '함수식'],
            '미분': ['derivative', 'differentiation', '도함수'],
            '적분': ['integral', 'integration', '적분값'],
            '방정식': ['equation', '식', '등식'],
            '행렬': ['matrix', '매트릭스', '행렬식'],
            
            # 물리
            '힘': ['force', 'F', '작용력'],
            '에너지': ['energy', 'E', '에너지량'],
            '운동': ['motion', '움직임', '운동학'],
            '파동': ['wave', '파형', '진동'],
            
            # 일반
            '계산': ['calculation', '연산', '산출'],
            '분석': ['analysis', '해석', '검토'],
            '방법': ['method', '방식', '기법'],
            '결과': ['result', '결론', '해답']
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """향상된 쿼리 분석"""
        # 언어 감지
        language = self._detect_language(query)
        
        # 텍스트 전처리
        processed_query = self._preprocess_query(query)
        
        # 키워드 추출
        keywords = self._extract_keywords(processed_query)
        
        # 도메인 감지
        detected_domains = self._detect_domains(processed_query, keywords)
        
        # 질의 의도 분석
        query_intent = self._analyze_query_intent(processed_query)
        
        # 복잡도 계산
        complexity_score = self._calculate_complexity(processed_query, keywords)
        
        return QueryAnalysis(
            original_query=query,
            processed_query=processed_query,
            keywords=keywords,
            detected_domains=detected_domains,
            query_intent=query_intent,
            complexity_score=complexity_score,
            language=language
        )
    
    def _detect_language(self, query: str) -> str:
        """언어 감지"""
        korean_chars = len(re.findall(r'[가-힣]', query))
        english_chars = len(re.findall(r'[a-zA-Z]', query))
        
        if korean_chars > english_chars * 2:
            return "korean"
        elif english_chars > korean_chars * 2:
            return "english"
        else:
            return "mixed"
    
    def _preprocess_query(self, query: str) -> str:
        """쿼리 전처리"""
        # 소문자 변환
        processed = query.lower()
        
        # 불필요한 문자 제거 (한글, 영문, 숫자, 기본 수학 기호 유지)
        processed = re.sub(r'[^\w\s가-힣α-ωΑ-Ω²³⁴⁵⁶⁷⁸⁹⁰₀₁₂₃₄₅₆₇₈₉∑∫±×÷≤≥≠π∞]', ' ', processed)
        
        # 연속된 공백 제거
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def _detect_domains(self, query: str, keywords: List[str]) -> List[DomainType]:
        """도메인 감지 (다중 도메인 지원)"""
        domain_scores = {}
        
        # 키워드 기반 도메인 점수 계산
        for domain_name, domain_keywords in self.domain_keywords.items():
            score = 0
            
            # 직접 매칭
            for keyword in keywords:
                for domain_keyword in domain_keywords:
                    if keyword == domain_keyword.lower():
                        score += 2  # 정확한 매칭은 높은 점수
                    elif keyword in domain_keyword.lower() or domain_keyword.lower() in keyword:
                        score += 1  # 부분 매칭
            
            # 쿼리 내 도메인 키워드 직접 검색
            for domain_keyword in domain_keywords:
                if domain_keyword.lower() in query:
                    score += 1
            
            if score > 0:
                domain_scores[domain_name] = score
        
        # 점수 기반 도메인 선택 (임계값 이상만)
        detected = []
        min_score = 2  # 최소 점수 임계값
        
        for domain_name, score in domain_scores.items():
            if score >= min_score:
                try:
                    # 도메인 이름을 DomainType으로 변환
                    domain_type = DomainType(domain_name)
                    detected.append(domain_type)
                except ValueError:
                    # 매핑되지 않은 도메인은 일반으로 처리
                    continue
        
        # 도메인이 감지되지 않으면 일반으로 분류
        if not detected:
            detected.append(DomainType.GENERAL)
        
        return detected[:3]  # 최대 3개 도메인
    
    def _analyze_query_intent(self, query: str) -> str:
        """질의 의도 분석"""
        intent_patterns = {
            'question': [r'무엇', r'뭔가', r'어떤', r'what', r'which', r'who', r'when', r'where', r'why', r'how'],
            'definition': [r'정의', r'의미', r'뜻', r'define', r'meaning', r'what.*is'],
            'procedure': [r'방법', r'어떻게', r'절차', r'과정', r'how.*to', r'step', r'process'],
            'calculation': [r'계산', r'구하', r'풀', r'solve', r'calculate', r'compute'],
            'comparison': [r'차이', r'비교', r'compare', r'difference', r'versus', r'vs'],
            'explanation': [r'설명', r'왜', r'이유', r'explain', r'because', r'reason'],
            'example': [r'예시', r'예제', r'example', r'case', r'instance']
        }
        
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        
        return 'general'
    
    def _calculate_complexity(self, query: str, keywords: List[str]) -> float:
        """쿼리 복잡도 계산"""
        factors = {
            'length': len(query) / 100.0,  # 길이 요소
            'word_count': len(query.split()) / 20.0,  # 단어 수
            'keyword_count': len(keywords) / 15.0,  # 키워드 수
            'technical_terms': self._count_technical_terms(query),  # 전문 용어
            'mathematical_content': self._count_mathematical_content(query),  # 수학적 내용
            'sentence_structure': self._analyze_sentence_structure(query)  # 문장 구조
        }
        
        # 가중 평균
        weights = {
            'length': 0.15,
            'word_count': 0.2,
            'keyword_count': 0.15,
            'technical_terms': 0.25,
            'mathematical_content': 0.15,
            'sentence_structure': 0.1
        }
        
        complexity = sum(factors[factor] * weights[factor] for factor in factors)
        return min(1.0, complexity)  # 0-1 범위로 제한
    
    def _count_technical_terms(self, query: str) -> float:
        """전문 용어 개수"""
        tech_count = 0
        
        for domain_keywords in self.domain_keywords.values():
            for keyword in domain_keywords:
                if keyword.lower() in query.lower():
                    tech_count += 1
        
        return min(1.0, tech_count / 10.0)  # 정규화
    
    def _count_mathematical_content(self, query: str) -> float:
        """수학적 내용 비율"""
        math_patterns = [
            r'[=+\-*/^]',  # 수학 연산자
            r'[0-9]+',     # 숫자
            r'[α-ωΑ-Ω]',  # 그리스 문자
            r'[²³⁴⁵⁶⁷⁸⁹⁰]',  # 위첨자
            r'[₀₁₂₃₄₅₆₇₈₉]',   # 아래첨자
            r'∑|∫|∞|π|±',  # 수학 기호
        ]
        
        math_count = 0
        for pattern in math_patterns:
            math_count += len(re.findall(pattern, query))
        
        return min(1.0, math_count / 15.0)  # 정규화
    
    def _analyze_sentence_structure(self, query: str) -> float:
        """문장 구조 복잡도"""
        # 문장 분리
        sentences = re.split(r'[.!?]', query)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # 평균 문장 길이
        avg_length = sum(len(s) for s in sentences) / len(sentences)
        
        # 복잡한 연결어 감지
        complex_connectors = ['그러므로', '따라서', '그런데', '하지만', '반면에', 'however', 'therefore']
        connector_count = sum(query.lower().count(connector) for connector in complex_connectors)
        
        # 구조 복잡도 계산
        structure_score = (avg_length / 50.0) + (connector_count / 3.0)
        
        return min(1.0, structure_score)

    def search(self, query: str, strategy: SearchStrategy = SearchStrategy.ADAPTIVE,
              max_results: int = 10, expansion_methods: List[QueryExpansionMethod] = None) -> EnhancedSearchResult:
        """
        향상된 검색 수행
        
        Args:
            query: 검색 쿼리
            strategy: 검색 전략
            max_results: 최대 결과 수
            expansion_methods: 쿼리 확장 방법들
            
        Returns:
            향상된 검색 결과
        """
        start_time = time.time()
        self.search_stats['total_searches'] += 1
        self.search_stats['strategy_usage'][strategy.value] += 1
        
        logger.info(f"Enhanced search: '{query[:50]}...' (strategy: {strategy.value})")
        
        # 캐시 확인
        cache_key = f"{query}_{strategy.value}_{max_results}"
        if cache_key in self.result_cache:
            self.search_stats['cache_hits'] += 1
            logger.info("Cache hit - returning cached result")
            return self.result_cache[cache_key]
        
        try:
            # 쿼리 확장
            expanded_queries = self._expand_query(query, expansion_methods or [QueryExpansionMethod.SYNONYM])
            
            # 전략별 검색 실행
            if strategy == SearchStrategy.SEMANTIC:
                results = self._semantic_search(expanded_queries, max_results)
            elif strategy == SearchStrategy.KEYWORD:
                results = self._keyword_search(expanded_queries, max_results)
            elif strategy == SearchStrategy.HYBRID:
                results = self._hybrid_search(expanded_queries, max_results)
            elif strategy == SearchStrategy.MULTI_QUERY:
                results = self._multi_query_search(expanded_queries, max_results)
            elif strategy == SearchStrategy.CHAIN_OF_THOUGHT:
                results = self._chain_of_thought_search(expanded_queries, max_results)
            else:
                results = self._hybrid_search(expanded_queries, max_results)
            
            # 결과 후처리
            enhanced_result = self._post_process_results(
                results, query, expanded_queries, strategy, time.time() - start_time
            )
            
            # 캐시 저장
            self._cache_result(cache_key, enhanced_result)
            
            # 통계 업데이트
            self._update_search_stats(enhanced_result)
            
            logger.info(f"Search completed: {len(enhanced_result.results)} results "
                       f"(confidence: {enhanced_result.confidence:.2f})")
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            
            # 기본 RAG 시스템으로 fallback
            try:
                fallback_result = self.base_rag.search(query, max_results=max_results)
                return self._convert_to_enhanced_result(fallback_result, query, strategy, time.time() - start_time)
            except Exception as e2:
                logger.error(f"Fallback search also failed: {e2}")
                return self._empty_result(query, strategy, time.time() - start_time)
    
    def _expand_query(self, query: str, methods: List[QueryExpansionMethod]) -> List[str]:
        """쿼리 확장"""
        expanded_queries = [query]  # 원본 쿼리 포함
        
        for method in methods:
            if method == QueryExpansionMethod.SYNONYM:
                expanded_queries.extend(self._expand_with_synonyms(query))
            elif method == QueryExpansionMethod.DOMAIN_SPECIFIC:
                expanded_queries.extend(self._expand_domain_specific(query))
            elif method == QueryExpansionMethod.STRUCTURAL:
                expanded_queries.extend(self._expand_structural(query))
            elif method == QueryExpansionMethod.LLM_BASED:
                expanded_queries.extend(self._expand_llm_based(query))
        
        # 중복 제거 및 제한
        unique_queries = list(dict.fromkeys(expanded_queries))[:10]
        logger.info(f"Query expanded: {len(unique_queries)} variants")
        
        return unique_queries
    
    def _expand_with_synonyms(self, query: str) -> List[str]:
        """동의어 기반 확장"""
        expanded = []
        words = query.split()
        
        for word in words:
            if word in self.synonym_dict:
                for synonym in self.synonym_dict[word][:2]:  # 최대 2개 동의어
                    new_query = query.replace(word, synonym)
                    if new_query != query:
                        expanded.append(new_query)
        
        return expanded[:3]  # 최대 3개
    
    def _expand_domain_specific(self, query: str) -> List[str]:
        """도메인 특화 확장"""
        expanded = []
        detected_domains = []
        
        # 도메인 감지
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query.lower() for keyword in keywords):
                detected_domains.append(domain)
        
        # 감지된 도메인의 관련 키워드 추가
        for domain in detected_domains[:2]:  # 최대 2개 도메인
            related_keywords = self.domain_keywords[domain][:3]  # 최대 3개 키워드
            for keyword in related_keywords:
                if keyword not in query.lower():
                    expanded.append(f"{query} {keyword}")
        
        return expanded[:2]  # 최대 2개
    
    def _expand_structural(self, query: str) -> List[str]:
        """구조적 확장"""
        expanded = []
        
        # 질문 형태 추가
        if not any(q in query for q in ['?', '무엇', '어떻게', 'what', 'how']):
            expanded.extend([
                f"{query}는 무엇인가요?",
                f"{query} 방법",
                f"how to {query}",
                f"what is {query}"
            ])
        
        # 설명 요청 형태
        if '설명' not in query:
            expanded.append(f"{query} 설명")
        
        return expanded[:3]
    
    def _expand_llm_based(self, query: str) -> List[str]:
        """LLM 기반 확장 (향후 구현)"""
        # 실제로는 LLM을 사용하여 쿼리 확장
        # 현재는 간단한 패턴 기반 확장
        expanded = []
        
        # 수식이 포함된 경우
        if re.search(r'[=+\-*/∫∑]', query):
            expanded.extend([
                f"{query} 공식",
                f"{query} 계산 방법",
                f"{query} 해결"
            ])
        
        return expanded[:2]
    
    def _semantic_search(self, queries: List[str], max_results: int) -> List[SearchResult]:
        """의미적 검색"""
        all_results = []
        
        for query in queries[:3]:  # 최대 3개 쿼리만 사용
            try:
                base_result = self.base_rag.search(query, max_results=max_results//len(queries) + 5)
                
                for result in base_result.get('results', []):
                    all_results.append(SearchResult(
                        content=result.get('content', ''),
                        score=result.get('score', 0.0),
                        source=result.get('source', 'unknown'),
                        metadata=result.get('metadata', {}),
                        relevance_type='semantic'
                    ))
            except Exception as e:
                logger.warning(f"Semantic search failed for query '{query}': {e}")
        
        return self._deduplicate_and_rank(all_results, max_results)
    
    def _keyword_search(self, queries: List[str], max_results: int) -> List[SearchResult]:
        """키워드 기반 검색"""
        all_results = []
        
        for query in queries:
            # 키워드 추출
            keywords = self._extract_keywords(query)
            
            # 각 키워드로 검색 (실제로는 더 정교한 키워드 검색 구현 필요)
            for keyword in keywords[:3]:
                try:
                    base_result = self.base_rag.search(keyword, max_results=max_results//len(keywords) + 3)
                    
                    for result in base_result.get('results', []):
                        # 키워드 매칭 점수 계산
                        keyword_score = self._calculate_keyword_score(result.get('content', ''), keywords)
                        
                        all_results.append(SearchResult(
                            content=result.get('content', ''),
                            score=keyword_score,
                            source=result.get('source', 'unknown'),
                            metadata=result.get('metadata', {}),
                            relevance_type='keyword'
                        ))
                except Exception as e:
                    logger.warning(f"Keyword search failed for '{keyword}': {e}")
        
        return self._deduplicate_and_rank(all_results, max_results)
    
    def _hybrid_search(self, queries: List[str], max_results: int) -> List[SearchResult]:
        """하이브리드 검색"""
        # 의미적 검색 결과
        semantic_results = self._semantic_search(queries, max_results)
        
        # 키워드 검색 결과  
        keyword_results = self._keyword_search(queries, max_results)
        
        # 결과 융합
        fused_results = self._fuse_results(semantic_results, keyword_results, max_results)
        
        return fused_results
    
    def _multi_query_search(self, queries: List[str], max_results: int) -> List[SearchResult]:
        """다중 쿼리 검색"""
        all_results = []
        query_weights = self._calculate_query_weights(queries)
        
        for i, query in enumerate(queries[:5]):  # 최대 5개 쿼리
            try:
                base_result = self.base_rag.search(query, max_results=max_results//len(queries) + 3)
                weight = query_weights[i]
                
                for result in base_result.get('results', []):
                    all_results.append(SearchResult(
                        content=result.get('content', ''),
                        score=result.get('score', 0.0) * weight,
                        source=result.get('source', 'unknown'),
                        metadata=result.get('metadata', {}),
                        relevance_type='multi_query'
                    ))
            except Exception as e:
                logger.warning(f"Multi-query search failed for '{query}': {e}")
        
        return self._deduplicate_and_rank(all_results, max_results)
    
    def _chain_of_thought_search(self, queries: List[str], max_results: int) -> List[SearchResult]:
        """사고 과정 기반 검색"""
        # 쿼리를 단계별로 분해
        thought_steps = self._decompose_query_to_steps(queries[0])
        
        all_results = []
        for step in thought_steps:
            try:
                base_result = self.base_rag.search(step, max_results=max_results//len(thought_steps) + 2)
                
                for result in base_result.get('results', []):
                    all_results.append(SearchResult(
                        content=result.get('content', ''),
                        score=result.get('score', 0.0),
                        source=result.get('source', 'unknown'),
                        metadata=result.get('metadata', {}),
                        relevance_type='chain_of_thought'
                    ))
            except Exception as e:
                logger.warning(f"Chain-of-thought search failed for step '{step}': {e}")
        
        return self._deduplicate_and_rank(all_results, max_results)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """키워드 추출"""
        # 불용어 제거
        stop_words = {
            '은', '는', '이', '가', '을', '를', '에', '에서', '로', '으로', '의', '와', '과',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has'
        }
        
        # 단어 추출 및 정규화
        words = re.findall(r'[가-힣a-zA-Z0-9]+', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        return keywords[:10]
    
    def _calculate_keyword_score(self, content: str, keywords: List[str]) -> float:
        """키워드 매칭 점수 계산"""
        if not content or not keywords:
            return 0.0
        
        content_lower = content.lower()
        matches = 0
        total_weight = 0
        
        for keyword in keywords:
            # 정확한 매칭
            if keyword in content_lower:
                matches += 1
                total_weight += 1.0
            # 부분 매칭
            elif any(keyword in word for word in content_lower.split()):
                matches += 0.5
                total_weight += 0.5
        
        return (matches / len(keywords)) * min(1.0, total_weight / 3.0)
    
    def _fuse_results(self, semantic_results: List[SearchResult], keyword_results: List[SearchResult],
                     max_results: int) -> List[SearchResult]:
        """결과 융합"""
        # 내용 기반으로 결과 그룹화
        content_map = {}
        
        # Semantic 결과 추가 (가중치 0.7)
        for result in semantic_results:
            content_key = result.content[:100]  # 처음 100자로 식별
            if content_key not in content_map:
                content_map[content_key] = result
                content_map[content_key].score *= 0.7
            else:
                # 점수 융합
                content_map[content_key].score = max(content_map[content_key].score, result.score * 0.7)
        
        # Keyword 결과 추가 (가중치 0.3)
        for result in keyword_results:
            content_key = result.content[:100]
            if content_key not in content_map:
                content_map[content_key] = result
                content_map[content_key].score *= 0.3
                content_map[content_key].relevance_type = 'keyword'
            else:
                # 하이브리드 점수 계산
                content_map[content_key].score += result.score * 0.3
                content_map[content_key].relevance_type = 'hybrid'
        
        # 점수 순으로 정렬
        fused_results = sorted(content_map.values(), key=lambda x: x.score, reverse=True)
        
        return fused_results[:max_results]
    
    def _calculate_query_weights(self, queries: List[str]) -> List[float]:
        """쿼리 가중치 계산"""
        weights = []
        
        for i, query in enumerate(queries):
            if i == 0:  # 원본 쿼리에 최고 가중치
                weights.append(1.0)
            else:
                # 길이와 복잡도 기반 가중치
                length_weight = min(1.0, len(query) / 50)
                complexity_weight = min(1.0, len(query.split()) / 10)
                weights.append(0.5 + (length_weight + complexity_weight) / 4)
        
        return weights
    
    def _decompose_query_to_steps(self, query: str) -> List[str]:
        """쿼리를 사고 단계로 분해"""
        steps = [query]  # 원본 쿼리
        
        # 문제 해결 패턴 감지
        if any(word in query.lower() for word in ['계산', '구하', '풀이', 'solve', 'calculate']):
            steps.extend([
                f"{query} 공식",
                f"{query} 방법",
                f"{query} 예제"
            ])
        
        # 설명 요청 패턴
        if any(word in query.lower() for word in ['설명', '의미', 'explain', 'what']):
            steps.extend([
                f"{query} 정의",
                f"{query} 원리",
                f"{query} 특징"
            ])
        
        return steps[:5]  # 최대 5단계
    
    def _deduplicate_and_rank(self, results: List[SearchResult], max_results: int) -> List[SearchResult]:
        """중복 제거 및 순위 매기기"""
        # 내용 기반 중복 제거
        seen_content = set()
        unique_results = []
        
        for result in results:
            content_hash = hash(result.content[:200])  # 처음 200자로 중복 판단
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        # 점수순 정렬
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results[:max_results]
    
    def _post_process_results(self, results: List[SearchResult], original_query: str,
                            queries_used: List[str], strategy: SearchStrategy, search_time: float) -> EnhancedSearchResult:
        """결과 후처리"""
        if not results:
            return self._empty_result(original_query, strategy, search_time)
        
        # 품질 메트릭 계산
        coverage_score = self._calculate_coverage_score(results, original_query)
        diversity_score = self._calculate_diversity_score(results)
        quality_score = self._calculate_quality_score(results)
        confidence = self._calculate_confidence(results, coverage_score, quality_score)
        
        return EnhancedSearchResult(
            results=results,
            total_found=len(results),
            search_time=search_time,
            strategy_used=strategy,
            confidence=confidence,
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            quality_score=quality_score,
            queries_used=queries_used,
            fusion_weights={'semantic': 0.7, 'keyword': 0.3},
            fallback_used=False
        )
    
    def _calculate_coverage_score(self, results: List[SearchResult], query: str) -> float:
        """쿼리 커버리지 점수"""
        if not results:
            return 0.0
        
        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            return 0.5
        
        covered_keywords = set()
        for result in results:
            result_keywords = self._extract_keywords(result.content)
            for kw in query_keywords:
                if any(kw in rw for rw in result_keywords):
                    covered_keywords.add(kw)
        
        return len(covered_keywords) / len(query_keywords)
    
    def _calculate_diversity_score(self, results: List[SearchResult]) -> float:
        """결과 다양성 점수"""
        if len(results) <= 1:
            return 0.0
        
        # 간단한 다양성 측정: 고유한 소스 수
        unique_sources = len(set(result.source for result in results))
        return min(1.0, unique_sources / len(results))
    
    def _calculate_quality_score(self, results: List[SearchResult]) -> float:
        """결과 품질 점수"""
        if not results:
            return 0.0
        
        # 평균 점수와 최고점의 조합
        avg_score = sum(result.score for result in results) / len(results)
        max_score = max(result.score for result in results)
        
        return (avg_score * 0.7 + max_score * 0.3)
    
    def _calculate_confidence(self, results: List[SearchResult], coverage: float, quality: float) -> float:
        """신뢰도 계산"""
        if not results:
            return 0.0
        
        # 결과 수, 커버리지, 품질을 종합
        result_count_factor = min(1.0, len(results) / 5)  # 5개 결과면 만점
        
        return (result_count_factor * 0.3 + coverage * 0.4 + quality * 0.3)
    
    def _cache_result(self, cache_key: str, result: EnhancedSearchResult):
        """결과 캐싱"""
        if len(self.result_cache) >= self.cache_size_limit:
            # 가장 오래된 항목 제거 (간단한 구현)
            oldest_key = next(iter(self.result_cache))
            del self.result_cache[oldest_key]
        
        self.result_cache[cache_key] = result
    
    def _update_search_stats(self, result: EnhancedSearchResult):
        """검색 통계 업데이트"""
        total = self.search_stats['total_searches']
        
        # 평균 결과 수 업데이트
        current_avg_results = self.search_stats['average_results']
        self.search_stats['average_results'] = (
            (current_avg_results * (total - 1) + result.total_found) / total
        )
        
        # 평균 신뢰도 업데이트
        current_avg_conf = self.search_stats['average_confidence']
        self.search_stats['average_confidence'] = (
            (current_avg_conf * (total - 1) + result.confidence) / total
        )
    
    def _convert_to_enhanced_result(self, base_result: Dict, query: str, strategy: SearchStrategy,
                                  search_time: float) -> EnhancedSearchResult:
        """기본 RAG 결과를 향상된 결과로 변환"""
        results = []
        
        for result in base_result.get('results', []):
            results.append(SearchResult(
                content=result.get('content', ''),
                score=result.get('score', 0.0),
                source=result.get('source', 'fallback'),
                metadata=result.get('metadata', {}),
                relevance_type='semantic'
            ))
        
        return EnhancedSearchResult(
            results=results,
            total_found=len(results),
            search_time=search_time,
            strategy_used=strategy,
            confidence=base_result.get('confidence', 0.5),
            coverage_score=0.5,
            diversity_score=0.5,
            quality_score=0.5,
            queries_used=[query],
            fusion_weights={},
            fallback_used=True
        )
    
    def _empty_result(self, query: str, strategy: SearchStrategy, search_time: float) -> EnhancedSearchResult:
        """빈 결과 생성"""
        return EnhancedSearchResult(
            results=[],
            total_found=0,
            search_time=search_time,
            strategy_used=strategy,
            confidence=0.0,
            coverage_score=0.0,
            diversity_score=0.0,
            quality_score=0.0,
            queries_used=[query],
            fusion_weights={},
            fallback_used=True
        )
    
    def get_search_stats(self) -> Dict[str, Any]:
        """검색 통계 반환"""
        return self.search_stats.copy()
    
    def clear_cache(self):
        """캐시 지우기"""
        self.result_cache.clear()
        logger.info("Search cache cleared")
    
    def suggest_query_improvements(self, query: str) -> List[str]:
        """쿼리 개선 제안"""
        suggestions = []
        
        # 너무 짧은 쿼리
        if len(query) < 10:
            suggestions.append("더 구체적인 정보를 포함해 주세요")
        
        # 키워드가 없는 경우
        keywords = self._extract_keywords(query)
        if len(keywords) < 2:
            suggestions.append("관련 키워드를 추가해 주세요")
        
        # 도메인 감지 제안
        detected_domains = []
        for domain, domain_keywords in self.domain_keywords.items():
            if any(kw in query.lower() for kw in domain_keywords):
                detected_domains.append(domain)
        
        if detected_domains:
            suggestions.append(f"'{detected_domains[0]}' 관련 용어를 더 추가하면 정확한 결과를 얻을 수 있습니다")
        
        return suggestions