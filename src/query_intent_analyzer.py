#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
질의 의도 분석 시스템
이미지 분석 전 질의 의도를 먼저 파악하여 적절한 처리 방식을 결정
최신 AI 트렌드 반영: 질의 분석 -> 처리 경로 결정 -> 최적화된 응답
"""

import re
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """질의 유형"""
    VISUAL_ANALYSIS = "visual_analysis"          # 이미지 시각적 분석
    TEXT_EXTRACTION = "text_extraction"          # 텍스트 추출
    FORMULA_ANALYSIS = "formula_analysis"        # 수식 분석
    PROBLEM_SOLVING = "problem_solving"          # 문제 해결
    EXPLANATION = "explanation"                  # 설명 요청
    COMPARISON = "comparison"                    # 비교 분석
    KNOWLEDGE_QUERY = "knowledge_query"          # 지식 질의
    PROCEDURAL = "procedural"                    # 절차/방법 질의
    GENERAL_CHAT = "general_chat"                # 일반 대화


class ComplexityLevel(Enum):
    """복잡도 수준"""
    SIMPLE = 1      # 단순 (단어 추출, 기본 질문)
    BASIC = 2       # 기본 (텍스트 추출, 간단한 설명)
    MODERATE = 3    # 중간 (수식 해석, 문제 분석)
    COMPLEX = 4     # 복잡 (다단계 추론, 비교 분석)
    EXPERT = 5      # 전문가 (고급 수학, 복합 분석)


class ProcessingMode(Enum):
    """처리 모드"""
    VISION_FIRST = "vision_first"           # 이미지 분석 우선
    RAG_FIRST = "rag_first"                # 지식 검색 우선
    HYBRID = "hybrid"                       # 하이브리드 처리
    REASONING_CHAIN = "reasoning_chain"     # 추론 체인
    DIRECT_RESPONSE = "direct_response"     # 직접 응답


@dataclass
class IntentAnalysisResult:
    """의도 분석 결과"""
    query_type: QueryType
    complexity: ComplexityLevel
    processing_mode: ProcessingMode
    confidence: float
    
    # 세부 분석
    requires_image: bool
    requires_rag: bool
    requires_reasoning: bool
    requires_calculation: bool
    
    # 키워드 및 특징
    keywords: List[str]
    domain: Optional[str]
    language: str
    
    # 처리 힌트
    suggested_prompt: Optional[str]
    fallback_options: List[str]
    expected_response_type: str


class QueryIntentAnalyzer:
    """질의 의도 분석기"""
    
    def __init__(self):
        """분석기 초기화"""
        self.domain_keywords = self._load_domain_keywords()
        self.processing_patterns = self._load_processing_patterns()
        
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """도메인별 키워드 로드"""
        return {
            'electrical': [
                '전기', '회로', '전압', '전류', '저항', '전력', '역률', '임피던스',
                '콘덴서', '인덕터', '변압기', '모터', '발전기', '옴의법칙',
                'voltage', 'current', 'resistance', 'power', 'circuit'
            ],
            'mathematics': [
                '수학', '수식', '공식', '계산', '적분', '미분', '방정식', '함수',
                '행렬', '벡터', '확률', '통계', '기하', '대수', '삼각함수',
                'integral', 'derivative', 'equation', 'function', 'matrix'
            ],
            'physics': [
                '물리', '힘', '에너지', '운동', '파동', '열', '광학', '양자',
                '역학', '전자기학', '상대성', '중력', '마찰',
                'force', 'energy', 'motion', 'wave', 'quantum', 'mechanics'
            ],
            'chemistry': [
                '화학', '원소', '분자', '반응', '화합물', '이온', '산화', '환원',
                '촉매', '평형', '농도', '몰', 'molecule', 'reaction', 'compound'
            ],
            'computer_science': [
                '컴퓨터', '프로그래밍', '알고리즘', '데이터', '구조', '네트워크',
                '데이터베이스', '인공지능', '머신러닝', '소프트웨어',
                'programming', 'algorithm', 'data', 'network', 'ai', 'ml'
            ]
        }
    
    def _load_processing_patterns(self) -> Dict[str, Dict]:
        """처리 패턴 로드"""
        return {
            'text_extraction': {
                'patterns': [
                    r'(?:텍스트|글자|문자|내용).*(?:추출|읽|뽑|찾)',
                    r'(?:extract|read|get).*(?:text|content)',
                    r'이미지.*(?:내용|글|텍스트)',
                    r'무엇.*(?:쓰여|적혀|나와)',
                ],
                'processing_mode': ProcessingMode.VISION_FIRST,
                'complexity': ComplexityLevel.BASIC
            },
            'formula_analysis': {
                'patterns': [
                    r'(?:수식|공식|식|계산|방정식).*(?:분석|해석|풀|계산)',
                    r'(?:formula|equation|calculate|solve)',
                    r'LaTeX.*(?:변환|형식)',
                    r'(?:적분|미분|함수).*(?:계산|풀)',
                ],
                'processing_mode': ProcessingMode.VISION_FIRST,
                'complexity': ComplexityLevel.MODERATE
            },
            'problem_solving': {
                'patterns': [
                    r'(?:문제|풀이|해결|답).*(?:구하|찾|풀|계산)',
                    r'(?:solve|find|calculate|determine)',
                    r'(?:답|해답|결과).*(?:무엇|얼마)',
                    r'어떻게.*(?:구하|계산|풀)',
                ],
                'processing_mode': ProcessingMode.REASONING_CHAIN,
                'complexity': ComplexityLevel.COMPLEX
            },
            'explanation': {
                'patterns': [
                    r'(?:설명|의미|뜻).*(?:해줘|알려|설명)',
                    r'(?:explain|describe|what.*mean)',
                    r'(?:왜|이유|원리).*(?:그런지|인지)',
                    r'(?:어떤|무슨).*(?:의미|뜻)',
                ],
                'processing_mode': ProcessingMode.RAG_FIRST,
                'complexity': ComplexityLevel.MODERATE
            },
            'comparison': {
                'patterns': [
                    r'(?:차이|비교|다른점|같은점)',
                    r'(?:compare|difference|similar|different)',
                    r'(?:대비|vs|versus)',
                    r'(?:어떤.*좋|더.*나은)',
                ],
                'processing_mode': ProcessingMode.HYBRID,
                'complexity': ComplexityLevel.COMPLEX
            }
        }
    
    def analyze_intent(self, query: str, has_image: bool = False, context: Optional[Dict] = None) -> IntentAnalysisResult:
        """
        질의 의도 분석 - 향상된 버전
        
        Args:
            query: 사용자 질의
            has_image: 이미지 첨부 여부
            context: 이전 대화 컨텍스트
            
        Returns:
            의도 분석 결과
        """
        logger.info(f"Enhanced intent analysis: '{query[:50]}...' (image: {has_image})")
        
        # 전처리 및 정규화
        normalized_query = self._normalize_query(query)
        
        # 기본 정보 추출
        language = self._detect_language(query)
        keywords = self._extract_keywords(query)
        domain = self._detect_domain(query, keywords)
        
        # 질의 유형 분석 (개선된 알고리즘)
        query_type = self._classify_query_type_enhanced(normalized_query, has_image, context)
        
        # 복잡도 평가 (다차원 분석)
        complexity = self._assess_complexity_multidimensional(query, query_type, keywords, has_image)
        
        # 처리 모드 결정 (AI 트렌드 반영)
        processing_mode = self._determine_processing_mode_advanced(query_type, complexity, has_image, keywords)
        
        # 신뢰도 계산 (다중 지표)
        confidence = self._calculate_confidence_multilevel(query, query_type, complexity, keywords)
        
        # 요구사항 분석 (세분화)
        requirements = self._analyze_requirements_detailed(query, query_type, has_image, keywords)
        
        # 프롬프트 및 힌트 생성 (컨텍스트 기반)
        suggested_prompt = self._generate_context_aware_prompt(query, query_type, complexity, domain)
        fallback_options = self._generate_intelligent_fallbacks(query_type, has_image, domain)
        response_type = self._predict_response_type_detailed(query_type, complexity, keywords)
        
        result = IntentAnalysisResult(
            query_type=query_type,
            complexity=complexity,
            processing_mode=processing_mode,
            confidence=confidence,
            
            requires_image=requirements['image'],
            requires_rag=requirements['rag'],
            requires_reasoning=requirements['reasoning'],
            requires_calculation=requirements['calculation'],
            
            keywords=keywords,
            domain=domain,
            language=language,
            
            suggested_prompt=suggested_prompt,
            fallback_options=fallback_options,
            expected_response_type=response_type
        )
        
        # 결과 로깅 및 메트릭 수집
        self._log_analysis_metrics(query, result)
        
        logger.info(f"Enhanced analysis completed: {query_type.value} | Complexity: {complexity.value} | Confidence: {confidence:.3f}")
        return result
    
    def _detect_language(self, query: str) -> str:
        """언어 감지"""
        korean_pattern = re.compile(r'[가-힣]')
        english_pattern = re.compile(r'[a-zA-Z]')
        
        korean_count = len(korean_pattern.findall(query))
        english_count = len(english_pattern.findall(query))
        
        if korean_count > english_count:
            return 'korean'
        elif english_count > korean_count:
            return 'english'
        else:
            return 'mixed'
    
    def _extract_keywords(self, query: str) -> List[str]:
        """키워드 추출"""
        # 불용어 제거
        stop_words = {
            '은', '는', '이', '가', '을', '를', '에', '에서', '로', '으로',
            '의', '와', '과', '도', '만', '까지', '부터', '에게', '께',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would'
        }
        
        # 단어 분리 및 정규화
        words = re.findall(r'[가-힣a-zA-Z0-9]+', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        return keywords[:10]  # 상위 10개만 반환
    
    def _detect_domain(self, query: str, keywords: List[str]) -> Optional[str]:
        """도메인 감지"""
        domain_scores = {}
        
        for domain, domain_keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                for domain_keyword in domain_keywords:
                    if keyword in domain_keyword.lower() or domain_keyword.lower() in keyword:
                        score += 1
            
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return None
    
    def _classify_query_type(self, query: str, has_image: bool) -> QueryType:
        """질의 유형 분류"""
        query_lower = query.lower()
        
        # 패턴 매칭
        for pattern_type, pattern_info in self.processing_patterns.items():
            for pattern in pattern_info['patterns']:
                if re.search(pattern, query_lower):
                    if pattern_type == 'text_extraction':
                        return QueryType.TEXT_EXTRACTION
                    elif pattern_type == 'formula_analysis':
                        return QueryType.FORMULA_ANALYSIS
                    elif pattern_type == 'problem_solving':
                        return QueryType.PROBLEM_SOLVING
                    elif pattern_type == 'explanation':
                        return QueryType.EXPLANATION
                    elif pattern_type == 'comparison':
                        return QueryType.COMPARISON
        
        # 이미지가 있는 경우 시각적 분석 우선
        if has_image:
            return QueryType.VISUAL_ANALYSIS
        
        # 지식 질의 패턴
        knowledge_patterns = [
            r'(?:무엇|뭔가|어떤|what)',
            r'(?:알려줘|설명해|explain)',
            r'(?:방법|how.*to)',
        ]
        
        for pattern in knowledge_patterns:
            if re.search(pattern, query_lower):
                return QueryType.KNOWLEDGE_QUERY
        
        return QueryType.GENERAL_CHAT
    
    def _assess_complexity(self, query: str, query_type: QueryType, keywords: List[str]) -> ComplexityLevel:
        """복잡도 평가"""
        complexity_score = 0
        
        # 길이 기반 점수
        if len(query) > 100:
            complexity_score += 2
        elif len(query) > 50:
            complexity_score += 1
        
        # 키워드 기반 점수
        technical_keywords = [
            '적분', '미분', '방정식', '행렬', '벡터', '함수',
            'integral', 'derivative', 'equation', 'matrix', 'vector',
            '회로', '임피던스', '역률', '변압기', 'circuit', 'impedance'
        ]
        
        for keyword in keywords:
            if any(tech in keyword.lower() for tech in technical_keywords):
                complexity_score += 1
        
        # 질의 유형 기반 점수
        type_complexity = {
            QueryType.TEXT_EXTRACTION: 1,
            QueryType.VISUAL_ANALYSIS: 2,
            QueryType.EXPLANATION: 2,
            QueryType.FORMULA_ANALYSIS: 3,
            QueryType.PROBLEM_SOLVING: 4,
            QueryType.COMPARISON: 3,
            QueryType.KNOWLEDGE_QUERY: 2,
            QueryType.PROCEDURAL: 3,
            QueryType.GENERAL_CHAT: 1
        }
        
        complexity_score += type_complexity.get(query_type, 2)
        
        # 복잡도 수준 매핑
        if complexity_score <= 2:
            return ComplexityLevel.SIMPLE
        elif complexity_score <= 4:
            return ComplexityLevel.BASIC
        elif complexity_score <= 6:
            return ComplexityLevel.MODERATE
        elif complexity_score <= 8:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.EXPERT
    
    def _determine_processing_mode(self, query_type: QueryType, complexity: ComplexityLevel, has_image: bool) -> ProcessingMode:
        """처리 모드 결정"""
        # 이미지가 있고 시각적 분석이 필요한 경우
        if has_image and query_type in [QueryType.VISUAL_ANALYSIS, QueryType.TEXT_EXTRACTION, QueryType.FORMULA_ANALYSIS]:
            return ProcessingMode.VISION_FIRST
        
        # 복잡한 문제 해결의 경우
        if query_type == QueryType.PROBLEM_SOLVING and complexity.value >= 3:
            return ProcessingMode.REASONING_CHAIN
        
        # 비교나 복합 분석의 경우
        if query_type in [QueryType.COMPARISON] or complexity == ComplexityLevel.EXPERT:
            return ProcessingMode.HYBRID
        
        # 지식 질의의 경우
        if query_type in [QueryType.KNOWLEDGE_QUERY, QueryType.EXPLANATION, QueryType.PROCEDURAL]:
            return ProcessingMode.RAG_FIRST
        
        # 기본적으로 직접 응답
        return ProcessingMode.DIRECT_RESPONSE
    
    def _calculate_confidence(self, query: str, query_type: QueryType, complexity: ComplexityLevel) -> float:
        """신뢰도 계산"""
        base_confidence = 0.7
        
        # 명확한 패턴이 매칭된 경우 높은 신뢰도
        query_lower = query.lower()
        clear_indicators = 0
        
        for pattern_info in self.processing_patterns.values():
            for pattern in pattern_info['patterns']:
                if re.search(pattern, query_lower):
                    clear_indicators += 1
        
        if clear_indicators >= 2:
            base_confidence += 0.2
        elif clear_indicators >= 1:
            base_confidence += 0.1
        
        # 길이가 너무 짧으면 신뢰도 감소
        if len(query) < 10:
            base_confidence -= 0.2
        
        return min(0.95, max(0.3, base_confidence))
    
    def _analyze_requirements(self, query: str, query_type: QueryType, has_image: bool) -> Dict[str, bool]:
        """처리 요구사항 분석"""
        return {
            'image': has_image and query_type in [
                QueryType.VISUAL_ANALYSIS, QueryType.TEXT_EXTRACTION, 
                QueryType.FORMULA_ANALYSIS, QueryType.PROBLEM_SOLVING
            ],
            'rag': query_type in [
                QueryType.KNOWLEDGE_QUERY, QueryType.EXPLANATION, 
                QueryType.PROCEDURAL, QueryType.COMPARISON
            ],
            'reasoning': query_type in [
                QueryType.PROBLEM_SOLVING, QueryType.COMPARISON
            ],
            'calculation': any(keyword in query.lower() for keyword in [
                '계산', '풀이', '구하', 'calculate', 'solve', 'find'
            ])
        }
    
    def _generate_prompt_hint(self, query: str, query_type: QueryType, complexity: ComplexityLevel) -> str:
        """프롬프트 힌트 생성"""
        if query_type == QueryType.TEXT_EXTRACTION:
            return "이미지의 모든 텍스트를 정확히 추출해주세요."
        elif query_type == QueryType.FORMULA_ANALYSIS:
            return "이미지의 수식을 LaTeX 형식으로 변환하고 의미를 설명해주세요."
        elif query_type == QueryType.PROBLEM_SOLVING:
            return "문제를 단계별로 분석하고 해결 과정을 자세히 설명해주세요."
        elif query_type == QueryType.EXPLANATION:
            return "개념을 이해하기 쉽게 설명하고 예시를 들어주세요."
        else:
            return "질문에 정확하고 도움이 되는 답변을 제공해주세요."
    
    def _generate_fallback_options(self, query_type: QueryType, has_image: bool) -> List[str]:
        """대안 옵션 생성"""
        fallbacks = []
        
        if has_image:
            fallbacks.append("NCP_OCR")
            fallbacks.append("multi_engine_ocr")
        
        if query_type in [QueryType.KNOWLEDGE_QUERY, QueryType.EXPLANATION]:
            fallbacks.append("rag_search")
            fallbacks.append("web_search")
        
        fallbacks.append("direct_llm")
        
        return fallbacks
    
    def _predict_response_type(self, query_type: QueryType, complexity: ComplexityLevel) -> str:
        """응답 타입 예측"""
        if query_type == QueryType.TEXT_EXTRACTION:
            return "extracted_text"
        elif query_type == QueryType.FORMULA_ANALYSIS:
            return "formula_analysis"
        elif query_type == QueryType.PROBLEM_SOLVING:
            return "step_by_step_solution"
        elif query_type == QueryType.EXPLANATION:
            return "detailed_explanation"
        elif query_type == QueryType.COMPARISON:
            return "comparative_analysis"
        else:
            return "general_response"
    
    # Enhanced analysis methods
    def _normalize_query(self, query: str) -> str:
        """질의 정규화"""
        # 소문자 변환
        normalized = query.lower()
        # 연속된 공백 제거
        normalized = re.sub(r'\s+', ' ', normalized)
        # 특수문자 정리
        normalized = re.sub(r'[^\w\s가-힣]', ' ', normalized)
        return normalized.strip()
    
    def _classify_query_type_enhanced(self, query: str, has_image: bool, context: Optional[Dict]) -> QueryType:
        """향상된 질의 유형 분류"""
        # 기존 분류 결과
        base_type = self._classify_query_type(query, has_image)
        
        # 컨텍스트 기반 조정
        if context and 'previous_type' in context:
            if context['previous_type'] == 'formula_analysis' and '추가' in query:
                return QueryType.FORMULA_ANALYSIS
            elif context['previous_type'] == 'problem_solving' and '다음' in query:
                return QueryType.PROBLEM_SOLVING
        
        # 다중 패턴 매칭으로 신뢰도 향상
        type_scores = {}
        for pattern_type, pattern_info in self.processing_patterns.items():
            score = 0
            for pattern in pattern_info['patterns']:
                matches = len(re.findall(pattern, query))
                score += matches
            if score > 0:
                type_scores[pattern_type] = score
        
        # 최고 점수가 기존 결과와 다르고 충분히 높으면 업데이트
        if type_scores:
            best_pattern = max(type_scores.items(), key=lambda x: x[1])
            if best_pattern[1] >= 2:  # 신뢰도 임계값
                if best_pattern[0] == 'text_extraction':
                    return QueryType.TEXT_EXTRACTION
                elif best_pattern[0] == 'formula_analysis':
                    return QueryType.FORMULA_ANALYSIS
                elif best_pattern[0] == 'problem_solving':
                    return QueryType.PROBLEM_SOLVING
        
        return base_type
    
    def _assess_complexity_multidimensional(self, query: str, query_type: QueryType, keywords: List[str], has_image: bool) -> ComplexityLevel:
        """다차원 복잡도 평가"""
        dimensions = {
            'length': self._assess_length_complexity(query),
            'technical': self._assess_technical_complexity(keywords),
            'linguistic': self._assess_linguistic_complexity(query),
            'contextual': self._assess_contextual_complexity(query_type, has_image),
            'cognitive': self._assess_cognitive_complexity(query, query_type)
        }
        
        # 가중 평균 계산
        weights = {'length': 0.1, 'technical': 0.3, 'linguistic': 0.2, 'contextual': 0.2, 'cognitive': 0.2}
        weighted_score = sum(dimensions[dim] * weights[dim] for dim in dimensions)
        
        # 복잡도 레벨 매핑
        if weighted_score <= 1.5:
            return ComplexityLevel.SIMPLE
        elif weighted_score <= 2.5:
            return ComplexityLevel.BASIC
        elif weighted_score <= 3.5:
            return ComplexityLevel.MODERATE
        elif weighted_score <= 4.5:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.EXPERT
    
    def _determine_processing_mode_advanced(self, query_type: QueryType, complexity: ComplexityLevel, has_image: bool, keywords: List[str]) -> ProcessingMode:
        """고급 처리 모드 결정"""
        # AI 트렌드 반영: Chain-of-Thought, Multi-Agent 고려
        
        # 고복잡도 수식 분석: 추론 체인 필요
        if query_type == QueryType.FORMULA_ANALYSIS and complexity.value >= 4:
            return ProcessingMode.REASONING_CHAIN
            
        # 멀티모달 요구사항이 높은 경우
        multimodal_keywords = ['그래프', '차트', '도표', '이미지', '그림']
        if has_image and any(kw in ' '.join(keywords) for kw in multimodal_keywords):
            return ProcessingMode.VISION_FIRST
            
        # 비교 분석이나 복합 질의
        if query_type == QueryType.COMPARISON or '비교' in ' '.join(keywords):
            return ProcessingMode.HYBRID
            
        # 전문 지식이 필요한 경우
        expert_domains = ['전기', '전자', '회로', '수학', '물리']
        if any(domain in ' '.join(keywords) for domain in expert_domains):
            return ProcessingMode.RAG_FIRST
        
        return self._determine_processing_mode(query_type, complexity, has_image)
    
    def _calculate_confidence_multilevel(self, query: str, query_type: QueryType, complexity: ComplexityLevel, keywords: List[str]) -> float:
        """다중 레벨 신뢰도 계산"""
        confidence_factors = {
            'pattern_matching': self._pattern_matching_confidence(query, query_type),
            'keyword_relevance': self._keyword_relevance_confidence(keywords, query_type),
            'length_adequacy': self._length_adequacy_confidence(query),
            'linguistic_clarity': self._linguistic_clarity_confidence(query),
            'domain_specificity': self._domain_specificity_confidence(keywords)
        }
        
        # 가중 평균
        weights = {'pattern_matching': 0.3, 'keyword_relevance': 0.25, 'length_adequacy': 0.15, 
                  'linguistic_clarity': 0.15, 'domain_specificity': 0.15}
        
        final_confidence = sum(confidence_factors[factor] * weights[factor] for factor in confidence_factors)
        return min(0.95, max(0.2, final_confidence))
    
    def _analyze_requirements_detailed(self, query: str, query_type: QueryType, has_image: bool, keywords: List[str]) -> Dict[str, bool]:
        """세부 요구사항 분석"""
        requirements = self._analyze_requirements(query, query_type, has_image)
        
        # 추가 세밀한 분석
        advanced_requirements = {
            'math_processing': any(kw in ' '.join(keywords) for kw in ['수식', '계산', '방정식', 'equation']),
            'step_by_step': '단계' in query or 'step' in query.lower(),
            'detailed_explanation': '자세히' in query or '설명' in query,
            'comparison_analysis': '비교' in query or 'compare' in query.lower(),
            'contextual_understanding': len(query) > 100 or '관련' in query,
            'multilingual_support': self._detect_language(query) == 'mixed'
        }
        
        requirements.update(advanced_requirements)
        return requirements
    
    def _generate_context_aware_prompt(self, query: str, query_type: QueryType, complexity: ComplexityLevel, domain: Optional[str]) -> str:
        """컨텍스트 인식 프롬프트 생성"""
        base_prompt = self._generate_prompt_hint(query, query_type, complexity)
        
        # 도메인별 특화 프롬프트
        domain_prompts = {
            'electrical': "전문 지식을 바탕으로 정확한 분석을 제공해주세요.",
            'mathematics': "수학적 정확성을 보장하며 LaTeX 형식으로 표현해주세요.",
            'physics': "물리학 원리를 명확히 설명하고 실제 예시를 들어주세요.",
            'chemistry': "화학 반응과 원리를 체계적으로 분석해주세요."
        }
        
        if domain and domain in domain_prompts:
            base_prompt += " " + domain_prompts[domain]
        
        # 복잡도별 추가 지침
        if complexity == ComplexityLevel.EXPERT:
            base_prompt += " 전문가 수준의 깊이 있는 분석을 제공해주세요."
        elif complexity == ComplexityLevel.SIMPLE:
            base_prompt += " 이해하기 쉽게 간단명료하게 설명해주세요."
        
        return base_prompt
    
    def _generate_intelligent_fallbacks(self, query_type: QueryType, has_image: bool, domain: Optional[str]) -> List[str]:
        """지능형 폴백 옵션 생성"""
        fallbacks = self._generate_fallback_options(query_type, has_image)
        
        # 도메인별 특화 폴백
        if domain == 'electrical':
            fallbacks.insert(0, "electrical_engineering_rag")
        elif domain == 'mathematics':
            fallbacks.insert(0, "mathematical_solver")
        
        # 질의 유형별 추가 폴백
        if query_type == QueryType.FORMULA_ANALYSIS:
            fallbacks.append("latex_parser")
            fallbacks.append("mathematical_ocr")
        
        return fallbacks
    
    def _predict_response_type_detailed(self, query_type: QueryType, complexity: ComplexityLevel, keywords: List[str]) -> str:
        """상세 응답 타입 예측"""
        base_type = self._predict_response_type(query_type, complexity)
        
        # 키워드 기반 세분화
        if 'latex' in ' '.join(keywords).lower():
            return base_type + "_with_latex"
        elif '단계' in ' '.join(keywords):
            return base_type + "_step_by_step"
        elif '그래프' in ' '.join(keywords):
            return base_type + "_with_visualization"
        
        return base_type
    
    def _log_analysis_metrics(self, query: str, result: IntentAnalysisResult) -> None:
        """분석 메트릭 로깅"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'query_length': len(query),
            'query_type': result.query_type.value,
            'complexity': result.complexity.value,
            'confidence': result.confidence,
            'processing_mode': result.processing_mode.value,
            'domain': result.domain,
            'requires_multimodal': result.requires_image,
            'keyword_count': len(result.keywords)
        }
        
        logger.debug(f"Analysis metrics: {json.dumps(metrics, ensure_ascii=False)}")
    
    # Helper methods for multidimensional complexity assessment
    def _assess_length_complexity(self, query: str) -> float:
        """길이 기반 복잡도"""
        length = len(query)
        if length < 20: return 1.0
        elif length < 50: return 2.0
        elif length < 100: return 3.0
        elif length < 200: return 4.0
        else: return 5.0
    
    def _assess_technical_complexity(self, keywords: List[str]) -> float:
        """기술적 복잡도"""
        technical_terms = ['적분', '미분', '방정식', '행렬', '임피던스', '변압기', 'circuit', 'integral']
        score = sum(1 for keyword in keywords if any(term in keyword.lower() for term in technical_terms))
        return min(5.0, 1.0 + score * 0.5)
    
    def _assess_linguistic_complexity(self, query: str) -> float:
        """언어적 복잡도"""
        complex_patterns = [r'그러므로', r'따라서', r'반면에', r'however', r'therefore', r'nevertheless']
        score = sum(len(re.findall(pattern, query.lower())) for pattern in complex_patterns)
        return min(5.0, 1.0 + score * 0.5)
    
    def _assess_contextual_complexity(self, query_type: QueryType, has_image: bool) -> float:
        """컨텍스트 복잡도"""
        base = 2.0
        if has_image: base += 1.0
        if query_type in [QueryType.PROBLEM_SOLVING, QueryType.COMPARISON]: base += 1.0
        return min(5.0, base)
    
    def _assess_cognitive_complexity(self, query: str, query_type: QueryType) -> float:
        """인지적 복잡도"""
        cognitive_indicators = ['분석', '비교', '평가', '종합', 'analyze', 'compare', 'evaluate']
        score = sum(query.lower().count(indicator) for indicator in cognitive_indicators)
        type_bonus = 1.0 if query_type in [QueryType.PROBLEM_SOLVING, QueryType.COMPARISON] else 0.0
        return min(5.0, 2.0 + score * 0.3 + type_bonus)
    
    # Confidence calculation helpers
    def _pattern_matching_confidence(self, query: str, query_type: QueryType) -> float:
        """패턴 매칭 신뢰도"""
        return self._calculate_confidence(query, query_type, ComplexityLevel.BASIC)
    
    def _keyword_relevance_confidence(self, keywords: List[str], query_type: QueryType) -> float:
        """키워드 관련성 신뢰도"""
        relevant_keywords = {
            QueryType.TEXT_EXTRACTION: ['텍스트', '글자', '내용'],
            QueryType.FORMULA_ANALYSIS: ['수식', '공식', '계산'],
            QueryType.PROBLEM_SOLVING: ['문제', '풀이', '해결'],
        }
        
        if query_type in relevant_keywords:
            matches = sum(1 for kw in keywords if any(rel in kw for rel in relevant_keywords[query_type]))
            return min(1.0, 0.3 + matches * 0.2)
        return 0.7
    
    def _length_adequacy_confidence(self, query: str) -> float:
        """길이 적절성 신뢰도"""
        length = len(query)
        if 10 <= length <= 200: return 0.9
        elif 5 <= length < 10 or 200 < length <= 500: return 0.7
        else: return 0.5
    
    def _linguistic_clarity_confidence(self, query: str) -> float:
        """언어적 명확성 신뢰도"""
        clarity_score = 0.7
        if re.search(r'[?!.]', query): clarity_score += 0.1
        if len(query.split()) >= 3: clarity_score += 0.1
        return min(1.0, clarity_score)
    
    def _domain_specificity_confidence(self, keywords: List[str]) -> float:
        """도메인 특수성 신뢰도"""
        domain_terms = ['전기', '수학', '물리', '화학', 'electrical', 'mathematics']
        matches = sum(1 for kw in keywords if any(term in kw.lower() for term in domain_terms))
        return min(1.0, 0.5 + matches * 0.1)


# 편의 함수들
def analyze_query_intent(query: str, has_image: bool = False, context: Optional[Dict] = None) -> IntentAnalysisResult:
    """질의 의도 분석 편의 함수"""
    analyzer = QueryIntentAnalyzer()
    return analyzer.analyze_intent(query, has_image, context)

def get_processing_recommendation(query: str, has_image: bool = False) -> Dict[str, Any]:
    """처리 방법 추천"""
    result = analyze_query_intent(query, has_image)
    return {
        'recommended_mode': result.processing_mode.value,
        'priority_order': result.fallback_options,
        'estimated_complexity': result.complexity.value,
        'confidence': result.confidence,
        'special_requirements': {
            'vision': result.requires_image,
            'rag': result.requires_rag,
            'reasoning': result.requires_reasoning,
            'calculation': result.requires_calculation
        }
    }