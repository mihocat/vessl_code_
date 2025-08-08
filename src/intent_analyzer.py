#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Intent Analysis Engine
질문 의도 분석 엔진 - 사용자 질의를 분석하여 최적의 처리 경로 결정
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """질의 유형 분류"""
    VISUAL_ANALYSIS = "visual_analysis"       # 이미지 분석이 필요한 질의
    TEXT_EXTRACTION = "text_extraction"      # 텍스트 추출이 주 목적
    MATHEMATICAL = "mathematical"            # 수학/공식 관련
    FACTUAL = "factual"                     # 사실적 정보 요청
    ANALYTICAL = "analytical"               # 분석적 사고 필요
    CONVERSATIONAL = "conversational"       # 일반 대화
    PROCEDURAL = "procedural"               # 절차/방법 설명
    COMPARATIVE = "comparative"             # 비교/대조
    CREATIVE = "creative"                   # 창의적 응답 필요
    MULTI_STEP = "multi_step"              # 다단계 추론 필요


class ProcessingMode(Enum):
    """처리 모드"""
    DIRECT_RAG = "direct_rag"              # 직접 RAG 검색
    VISION_FIRST = "vision_first"          # 이미지 분석 우선
    REASONING_CHAIN = "reasoning_chain"     # 추론 체인 사용
    MULTI_AGENT = "multi_agent"            # 다중 에이전트 협업
    HYBRID = "hybrid"                      # 하이브리드 처리


@dataclass
class IntentAnalysisResult:
    """의도 분석 결과"""
    query_type: QueryType
    processing_mode: ProcessingMode
    confidence: float
    complexity_level: int  # 1-5 (1: simple, 5: very complex)
    requires_image: bool
    requires_reasoning: bool
    requires_external_search: bool
    estimated_tokens: int
    priority: int  # 1-3 (1: high, 2: medium, 3: low)
    metadata: Dict[str, Any]


class IntentAnalyzer:
    """고급 의도 분석기"""
    
    def __init__(self, config=None):
        """초기화"""
        self.config = config
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """패턴 초기화"""
        # 수학/공식 패턴
        self.math_patterns = [
            r'수식|공식|계산|적분|미분|방정식|함수',
            r'변환|알고리즘|처리',
            r'벡터|행렬|선형대수',
            r'확률|통계',
            r'[0-9]+\s*[+\-*/=]\s*[0-9]+',
            r'[a-zA-Z]\s*=\s*[0-9]+',
            r'\\[a-zA-Z]+\{.*?\}',  # LaTeX 패턴
        ]
        
        # 시각 분석 패턴
        self.visual_patterns = [
            r'이미지|그림|도표|차트|그래프',
            r'사진|스크린샷|캡처',
            r'그려진|표시된|나타난',
            r'보이는|나와있는|있는',
            r'읽어|추출|인식',
        ]
        
        # 분석적 사고 패턴
        self.analytical_patterns = [
            r'분석|해석|평가|비교|대조',
            r'왜|어떻게|무엇이|언제|어디서',
            r'장단점|차이점|공통점',
            r'원인|결과|영향|효과',
            r'문제|해결|방법|전략',
        ]
        
        # 절차적 패턴
        self.procedural_patterns = [
            r'방법|절차|단계|과정',
            r'어떻게 하면|어떻게 해야',
            r'설명해|알려줘|가르쳐',
            r'순서|먼저|다음에|마지막',
        ]
        
        # 복합 질의 패턴
        self.complex_patterns = [
            r'그리고|또한|그런데|하지만|그러나',
            r'첫째|둘째|셋째',
            r'먼저.*다음.*마지막',
            r'비교.*분석',
            r'설명.*예시',
        ]
    
    def analyze_intent(
        self, 
        query: str, 
        has_image: bool = False,
        context: Optional[Dict] = None
    ) -> IntentAnalysisResult:
        """
        질의 의도 분석
        
        Args:
            query: 사용자 질의
            has_image: 이미지 첨부 여부
            context: 추가 컨텍스트
            
        Returns:
            의도 분석 결과
        """
        logger.info(f"Analyzing intent for query: {query[:50]}...")
        
        # 기본 분석
        query_lower = query.lower()
        query_type = self._classify_query_type(query_lower, has_image)
        complexity = self._assess_complexity(query_lower)
        
        # 처리 모드 결정
        processing_mode = self._determine_processing_mode(
            query_type, complexity, has_image
        )
        
        # 요구사항 분석
        requires_image = has_image or self._requires_image_analysis(query_lower)
        requires_reasoning = self._requires_reasoning(query_lower, complexity)
        requires_external = self._requires_external_search(query_lower, query_type)
        
        # 토큰 추정
        estimated_tokens = self._estimate_tokens(query, complexity, has_image)
        
        # 우선도 결정
        priority = self._determine_priority(query_type, complexity, has_image)
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(query_lower, query_type)
        
        # 메타데이터 생성
        metadata = self._generate_metadata(query, query_type, complexity)
        
        result = IntentAnalysisResult(
            query_type=query_type,
            processing_mode=processing_mode,
            confidence=confidence,
            complexity_level=complexity,
            requires_image=requires_image,
            requires_reasoning=requires_reasoning,
            requires_external_search=requires_external,
            estimated_tokens=estimated_tokens,
            priority=priority,
            metadata=metadata
        )
        
        logger.info(f"Intent analysis result: {query_type.value}, "
                   f"mode: {processing_mode.value}, "
                   f"complexity: {complexity}, confidence: {confidence:.2f}")
        
        return result
    
    def _classify_query_type(self, query: str, has_image: bool) -> QueryType:
        """질의 유형 분류"""
        # 이미지가 있는 경우 우선적으로 시각 분석으로 분류
        if has_image:
            if any(re.search(pattern, query) for pattern in self.visual_patterns):
                return QueryType.VISUAL_ANALYSIS
            if any(re.search(pattern, query) for pattern in self.math_patterns):
                return QueryType.MATHEMATICAL
            return QueryType.TEXT_EXTRACTION
        
        # 수학/공식 질의
        if any(re.search(pattern, query) for pattern in self.math_patterns):
            return QueryType.MATHEMATICAL
        
        # 분석적 질의
        if any(re.search(pattern, query) for pattern in self.analytical_patterns):
            return QueryType.ANALYTICAL
        
        # 절차적 질의
        if any(re.search(pattern, query) for pattern in self.procedural_patterns):
            return QueryType.PROCEDURAL
        
        # 복합 질의
        if any(re.search(pattern, query) for pattern in self.complex_patterns):
            return QueryType.MULTI_STEP
        
        # 비교 질의
        if '비교' in query or 'vs' in query or '차이' in query:
            return QueryType.COMPARATIVE
        
        # 창의적 질의
        if any(word in query for word in ['창작', '만들', '설계', '제안']):
            return QueryType.CREATIVE
        
        # 기본: 사실적 질의
        return QueryType.FACTUAL
    
    def _assess_complexity(self, query: str) -> int:
        """복잡도 평가 (1-5)"""
        complexity_score = 1
        
        # 길이 기반
        if len(query) > 100:
            complexity_score += 1
        if len(query) > 200:
            complexity_score += 1
        
        # 복합 구문
        if any(re.search(pattern, query) for pattern in self.complex_patterns):
            complexity_score += 1
        
        # 다중 질문
        question_count = query.count('?') + query.count('인가') + query.count('무엇')
        if question_count > 1:
            complexity_score += 1
        
        # 전문 용어
        if any(re.search(pattern, query) for pattern in self.math_patterns):
            complexity_score += 1
        
        return min(complexity_score, 5)
    
    def _determine_processing_mode(
        self, 
        query_type: QueryType, 
        complexity: int, 
        has_image: bool
    ) -> ProcessingMode:
        """처리 모드 결정"""
        # 이미지 우선 처리
        if has_image and query_type in [QueryType.VISUAL_ANALYSIS, QueryType.TEXT_EXTRACTION]:
            return ProcessingMode.VISION_FIRST
        
        # 복잡한 추론 필요
        if complexity >= 4 or query_type == QueryType.MULTI_STEP:
            return ProcessingMode.REASONING_CHAIN
        
        # 다중 에이전트 협업
        if query_type in [QueryType.ANALYTICAL, QueryType.COMPARATIVE] and complexity >= 3:
            return ProcessingMode.MULTI_AGENT
        
        # 하이브리드 처리
        if has_image and complexity >= 3:
            return ProcessingMode.HYBRID
        
        # 기본: 직접 RAG
        return ProcessingMode.DIRECT_RAG
    
    def _requires_image_analysis(self, query: str) -> bool:
        """이미지 분석 필요성 판단"""
        return any(re.search(pattern, query) for pattern in self.visual_patterns)
    
    def _requires_reasoning(self, query: str, complexity: int) -> bool:
        """추론 필요성 판단"""
        if complexity >= 3:
            return True
        return any(re.search(pattern, query) for pattern in self.analytical_patterns)
    
    def _requires_external_search(self, query: str, query_type: QueryType) -> bool:
        """외부 검색 필요성 판단"""
        # 최신 정보 관련 키워드
        recent_keywords = ['최신', '현재', '오늘', '2024', '2025', '새로운']
        if any(keyword in query for keyword in recent_keywords):
            return True
        
        # 특정 질의 유형
        if query_type in [QueryType.FACTUAL, QueryType.COMPARATIVE]:
            return True
        
        return False
    
    def _estimate_tokens(self, query: str, complexity: int, has_image: bool) -> int:
        """토큰 사용량 추정"""
        base_tokens = len(query.split()) * 1.3  # 한국어 토큰 비율
        
        if has_image:
            base_tokens += 25000  # 이미지 처리 토큰
        
        complexity_multiplier = 1 + (complexity - 1) * 0.5
        
        return int(base_tokens * complexity_multiplier)
    
    def _determine_priority(
        self, 
        query_type: QueryType, 
        complexity: int, 
        has_image: bool
    ) -> int:
        """우선도 결정 (1: high, 2: medium, 3: low)"""
        if has_image or query_type == QueryType.MATHEMATICAL:
            return 1
        
        if complexity >= 4 or query_type in [QueryType.ANALYTICAL, QueryType.MULTI_STEP]:
            return 1
        
        if complexity >= 2:
            return 2
        
        return 3
    
    def _calculate_confidence(self, query: str, query_type: QueryType) -> float:
        """신뢰도 계산"""
        # 패턴 매칭 기반 신뢰도
        pattern_matches = 0
        total_patterns = 0
        
        if query_type == QueryType.MATHEMATICAL:
            total_patterns = len(self.math_patterns)
            pattern_matches = sum(1 for pattern in self.math_patterns 
                                if re.search(pattern, query))
        elif query_type == QueryType.VISUAL_ANALYSIS:
            total_patterns = len(self.visual_patterns)
            pattern_matches = sum(1 for pattern in self.visual_patterns 
                                if re.search(pattern, query))
        elif query_type == QueryType.ANALYTICAL:
            total_patterns = len(self.analytical_patterns)
            pattern_matches = sum(1 for pattern in self.analytical_patterns 
                                if re.search(pattern, query))
        
        if total_patterns > 0:
            base_confidence = pattern_matches / total_patterns
        else:
            base_confidence = 0.5
        
        # 길이 기반 보정
        length_factor = min(len(query) / 50, 1.0)
        
        return min(base_confidence + length_factor * 0.3, 1.0)
    
    def _generate_metadata(
        self, 
        query: str, 
        query_type: QueryType, 
        complexity: int
    ) -> Dict[str, Any]:
        """메타데이터 생성"""
        return {
            "query_length": len(query),
            "word_count": len(query.split()),
            "has_question_mark": '?' in query,
            "language": "korean",
            "suggested_timeout": complexity * 30,  # seconds
            "keywords": self._extract_keywords(query),
            "domain_hints": self._detect_domain(query)
        }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """키워드 추출"""
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 사용 가능)
        import re
        words = re.findall(r'\b\w{2,}\b', query)
        return [word for word in words if len(word) > 2][:10]
    
    def _detect_domain(self, query: str) -> List[str]:
        """도메인 감지"""
        domains = []
        
        # 기술 도메인
        tech_terms = ['데이터', '알고리즘', '프로그래밍', '시스템', '네트워크', '보안']
        if any(term in query for term in tech_terms):
            domains.append("technology")
        
        # 수학 도메인
        math_terms = ['적분', '미분', '행렬', '벡터', '함수', '방정식']
        if any(term in query for term in math_terms):
            domains.append("mathematics")
        
        # 물리 도메인
        physics_terms = ['힘', '에너지', '운동', '파동', '열역학']
        if any(term in query for term in physics_terms):
            domains.append("physics")
        
        return domains if domains else ["general"]


# 사용 예시 함수
def analyze_query_intent(query: str, has_image: bool = False) -> IntentAnalysisResult:
    """질의 의도 분석 헬퍼 함수"""
    analyzer = IntentAnalyzer()
    return analyzer.analyze_intent(query, has_image)


if __name__ == "__main__":
    # 테스트
    analyzer = IntentAnalyzer()
    
    test_queries = [
        ("이 이미지의 수식을 설명해주세요", True),
        ("머신러닝의 기본 개념을 설명해주세요", False),
        ("데이터 처리 방법을 단계별로 알려주세요", False),
        ("A와 B의 차이점을 비교 분석해주세요", False),
    ]
    
    for query, has_image in test_queries:
        result = analyzer.analyze_intent(query, has_image)
        print(f"\nQuery: {query}")
        print(f"Type: {result.query_type.value}")
        print(f"Mode: {result.processing_mode.value}")
        print(f"Complexity: {result.complexity_level}")
        print(f"Confidence: {result.confidence:.2f}")