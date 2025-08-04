#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Knowledge System
범용 지식 시스템 - 모든 학문 분야를 포괄하는 통합 시스템
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class KnowledgeDomain(Enum):
    """지식 도메인 분류 - 더 넓은 범위"""
    # 자연과학
    MATHEMATICS = "mathematics"  # 수학
    PHYSICS = "physics"  # 물리학
    CHEMISTRY = "chemistry"  # 화학
    BIOLOGY = "biology"  # 생물학
    EARTH_SCIENCE = "earth_science"  # 지구과학
    ASTRONOMY = "astronomy"  # 천문학
    
    # 공학
    ELECTRICAL = "electrical"  # 전기전자공학
    MECHANICAL = "mechanical"  # 기계공학
    CIVIL = "civil"  # 토목공학
    COMPUTER = "computer"  # 컴퓨터공학
    CHEMICAL = "chemical"  # 화학공학
    AEROSPACE = "aerospace"  # 항공우주공학
    BIOMEDICAL = "biomedical"  # 의공학
    MATERIALS = "materials"  # 재료공학
    
    # 정보기술
    SOFTWARE = "software"  # 소프트웨어
    AI_ML = "ai_ml"  # 인공지능/머신러닝
    DATA_SCIENCE = "data_science"  # 데이터 과학
    CYBERSECURITY = "cybersecurity"  # 사이버보안
    NETWORKS = "networks"  # 네트워크
    DATABASE = "database"  # 데이터베이스
    
    # 인문사회
    PHILOSOPHY = "philosophy"  # 철학
    HISTORY = "history"  # 역사
    LITERATURE = "literature"  # 문학
    LINGUISTICS = "linguistics"  # 언어학
    PSYCHOLOGY = "psychology"  # 심리학
    SOCIOLOGY = "sociology"  # 사회학
    ECONOMICS = "economics"  # 경제학
    POLITICS = "politics"  # 정치학
    LAW = "law"  # 법학
    
    # 예술
    MUSIC = "music"  # 음악
    ART = "art"  # 미술
    DESIGN = "design"  # 디자인
    ARCHITECTURE = "architecture"  # 건축
    
    # 의학/건강
    MEDICINE = "medicine"  # 의학
    NURSING = "nursing"  # 간호학
    PHARMACY = "pharmacy"  # 약학
    PUBLIC_HEALTH = "public_health"  # 공중보건
    
    # 비즈니스
    BUSINESS = "business"  # 경영학
    FINANCE = "finance"  # 금융
    MARKETING = "marketing"  # 마케팅
    ACCOUNTING = "accounting"  # 회계
    
    # 교육
    EDUCATION = "education"  # 교육학
    PEDAGOGY = "pedagogy"  # 교수법
    
    # 일반/혼합
    GENERAL = "general"  # 일반
    INTERDISCIPLINARY = "interdisciplinary"  # 학제간


class ContentComplexity(Enum):
    """콘텐츠 복잡도 수준"""
    ELEMENTARY = "elementary"  # 초등 수준
    MIDDLE_SCHOOL = "middle_school"  # 중등 수준
    HIGH_SCHOOL = "high_school"  # 고등 수준
    UNDERGRADUATE = "undergraduate"  # 학부 수준
    GRADUATE = "graduate"  # 대학원 수준
    RESEARCH = "research"  # 연구 수준
    EXPERT = "expert"  # 전문가 수준


class InformationType(Enum):
    """정보 유형 분류"""
    CONCEPT = "concept"  # 개념
    DEFINITION = "definition"  # 정의
    THEOREM = "theorem"  # 정리/법칙
    FORMULA = "formula"  # 공식
    ALGORITHM = "algorithm"  # 알고리즘
    PROCEDURE = "procedure"  # 절차/과정
    EXAMPLE = "example"  # 예시
    PROOF = "proof"  # 증명
    EXPERIMENT = "experiment"  # 실험
    CASE_STUDY = "case_study"  # 사례 연구
    HISTORICAL = "historical"  # 역사적 사실
    STATISTICAL = "statistical"  # 통계 데이터
    VISUAL = "visual"  # 시각 자료
    REFERENCE = "reference"  # 참고 자료


@dataclass
class KnowledgeUnit:
    """지식 단위"""
    content: str
    domain: KnowledgeDomain
    subdomain: Optional[str] = None
    info_type: InformationType = InformationType.CONCEPT
    complexity: ContentComplexity = ContentComplexity.UNDERGRADUATE
    language: str = "ko"
    prerequisites: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class UniversalKnowledgeExtractor:
    """범용 지식 추출기"""
    
    def __init__(self):
        """초기화"""
        self.domain_patterns = self._init_domain_patterns()
        self.complexity_indicators = self._init_complexity_indicators()
        self.info_type_patterns = self._init_info_type_patterns()
        
    def _init_domain_patterns(self) -> Dict[KnowledgeDomain, List[str]]:
        """도메인별 키워드 패턴"""
        return {
            # 수학
            KnowledgeDomain.MATHEMATICS: [
                '적분', '미분', '행렬', '벡터', '확률', '통계', '기하', '대수',
                'integral', 'derivative', 'matrix', 'vector', 'probability',
                '정리', '증명', '함수', '방정식', '그래프', '집합', '논리'
            ],
            
            # 물리학
            KnowledgeDomain.PHYSICS: [
                '힘', '에너지', '운동', '파동', '전자기', '양자', '상대성',
                'force', 'energy', 'motion', 'wave', 'electromagnetic', 'quantum',
                '역학', '열역학', '광학', '입자', '장', '물질'
            ],
            
            # 화학
            KnowledgeDomain.CHEMISTRY: [
                '원소', '화합물', '반응', '분자', '원자', '이온', '결합',
                'element', 'compound', 'reaction', 'molecule', 'atom',
                '유기', '무기', '화학식', '주기율표', '산', '염기'
            ],
            
            # 전기전자공학
            KnowledgeDomain.ELECTRICAL: [
                '전압', '전류', '저항', '회로', '반도체', '트랜지스터',
                'voltage', 'current', 'resistance', 'circuit', 'semiconductor',
                '임피던스', '필터', '증폭기', '신호', '전력', '제어'
            ],
            
            # 컴퓨터공학
            KnowledgeDomain.COMPUTER: [
                '알고리즘', '자료구조', '프로그래밍', '운영체제', '네트워크',
                'algorithm', 'data structure', 'programming', 'operating system',
                '컴파일러', '데이터베이스', '소프트웨어', '하드웨어'
            ],
            
            # AI/ML
            KnowledgeDomain.AI_ML: [
                '인공지능', '머신러닝', '딥러닝', '신경망', '학습',
                'artificial intelligence', 'machine learning', 'deep learning',
                'neural network', '모델', '훈련', '추론', '데이터셋'
            ],
            
            # 의학
            KnowledgeDomain.MEDICINE: [
                '진단', '치료', '증상', '질병', '약물', '수술', '환자',
                'diagnosis', 'treatment', 'symptom', 'disease', 'drug',
                '해부학', '생리학', '병리학', '임상', '의료'
            ],
            
            # 경제학
            KnowledgeDomain.ECONOMICS: [
                '수요', '공급', '시장', '가격', '경제', '인플레이션',
                'demand', 'supply', 'market', 'price', 'economy',
                'GDP', '금리', '투자', '소비', '생산'
            ],
            
            # 철학
            KnowledgeDomain.PHILOSOPHY: [
                '존재', '인식', '윤리', '논리', '형이상학', '미학',
                'existence', 'knowledge', 'ethics', 'logic', 'metaphysics',
                '철학자', '사상', '주의', '이론', '비판'
            ]
        }
    
    def _init_complexity_indicators(self) -> Dict[ContentComplexity, List[str]]:
        """복잡도 지표"""
        return {
            ContentComplexity.ELEMENTARY: [
                '기초', '입문', '쉬운', '간단한', 'basic', 'elementary'
            ],
            ContentComplexity.HIGH_SCHOOL: [
                '고등', '수능', '교과서', 'high school', 'textbook'
            ],
            ContentComplexity.UNDERGRADUATE: [
                '대학', '학부', '전공', 'university', 'undergraduate'
            ],
            ContentComplexity.GRADUATE: [
                '대학원', '석사', '박사', 'graduate', 'master', 'PhD'
            ],
            ContentComplexity.RESEARCH: [
                '연구', '논문', '저널', 'research', 'paper', 'journal'
            ],
            ContentComplexity.EXPERT: [
                '전문가', '고급', '심화', 'expert', 'advanced', 'specialized'
            ]
        }
    
    def _init_info_type_patterns(self) -> Dict[InformationType, List[str]]:
        """정보 유형 패턴"""
        return {
            InformationType.DEFINITION: [
                '정의', '뜻', '의미', 'definition', 'meaning', '이란', '란'
            ],
            InformationType.THEOREM: [
                '정리', '법칙', '원리', 'theorem', 'law', 'principle'
            ],
            InformationType.FORMULA: [
                '공식', '수식', '방정식', 'formula', 'equation', '='
            ],
            InformationType.ALGORITHM: [
                '알고리즘', '절차', '단계', 'algorithm', 'procedure', 'step'
            ],
            InformationType.PROOF: [
                '증명', '증거', '귀납', '연역', 'proof', 'prove', 'QED'
            ],
            InformationType.EXAMPLE: [
                '예시', '예제', '예를', 'example', 'instance', 'e.g.'
            ],
            InformationType.EXPERIMENT: [
                '실험', '관찰', '측정', 'experiment', 'observation', 'measure'
            ]
        }
    
    def extract_knowledge(self, content: str, hints: Optional[Dict[str, Any]] = None) -> List[KnowledgeUnit]:
        """콘텐츠에서 지식 단위 추출"""
        knowledge_units = []
        
        # 도메인 감지
        detected_domains = self._detect_domains(content)
        
        # 복잡도 추정
        complexity = self._estimate_complexity(content)
        
        # 정보 유형 분류
        info_types = self._classify_info_types(content)
        
        # 콘텐츠 분할 및 구조화
        segments = self._segment_content(content)
        
        for segment in segments:
            # 각 세그먼트에서 지식 단위 생성
            unit = KnowledgeUnit(
                content=segment['text'],
                domain=segment.get('domain', detected_domains[0] if detected_domains else KnowledgeDomain.GENERAL),
                subdomain=segment.get('subdomain'),
                info_type=segment.get('info_type', InformationType.CONCEPT),
                complexity=complexity,
                language=self._detect_language(segment['text']),
                prerequisites=self._extract_prerequisites(segment['text']),
                related_concepts=self._extract_related_concepts(segment['text']),
                metadata=segment.get('metadata', {}),
                confidence=segment.get('confidence', 0.8)
            )
            knowledge_units.append(unit)
        
        return knowledge_units
    
    def _detect_domains(self, content: str) -> List[KnowledgeDomain]:
        """도메인 감지"""
        content_lower = content.lower()
        domain_scores = defaultdict(int)
        
        for domain, keywords in self.domain_patterns.items():
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    domain_scores[domain] += 1
        
        # 점수 기준 정렬
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 도메인 반환
        detected = [domain for domain, score in sorted_domains if score > 0]
        
        return detected[:3] if detected else [KnowledgeDomain.GENERAL]
    
    def _estimate_complexity(self, content: str) -> ContentComplexity:
        """복잡도 추정"""
        content_lower = content.lower()
        
        # 복잡도 지표 확인
        for complexity, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                if indicator.lower() in content_lower:
                    return complexity
        
        # 기본값: 대학 수준
        return ContentComplexity.UNDERGRADUATE
    
    def _classify_info_types(self, content: str) -> List[InformationType]:
        """정보 유형 분류"""
        content_lower = content.lower()
        detected_types = []
        
        for info_type, patterns in self.info_type_patterns.items():
            for pattern in patterns:
                if pattern.lower() in content_lower:
                    detected_types.append(info_type)
                    break
        
        return detected_types if detected_types else [InformationType.CONCEPT]
    
    def _segment_content(self, content: str) -> List[Dict[str, Any]]:
        """콘텐츠 분할"""
        # 간단한 구현 - 실제로는 더 정교한 분할 필요
        segments = []
        
        # 문단 단위 분할
        paragraphs = content.split('\n\n')
        
        for para in paragraphs:
            if para.strip():
                segments.append({
                    'text': para.strip(),
                    'metadata': {
                        'length': len(para),
                        'has_formula': '=' in para or '∫' in para,
                        'has_code': '```' in para or 'def ' in para
                    }
                })
        
        return segments
    
    def _detect_language(self, text: str) -> str:
        """언어 감지"""
        import re
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if korean_chars > english_chars:
            return "ko"
        elif english_chars > korean_chars:
            return "en"
        else:
            return "mixed"
    
    def _extract_prerequisites(self, text: str) -> List[str]:
        """선수 지식 추출"""
        prerequisites = []
        
        # 패턴 기반 추출
        patterns = [
            r'사전\s*지식[:：]\s*(.+)',
            r'선수\s*과목[:：]\s*(.+)',
            r'Prerequisites?[:：]\s*(.+)',
            r'필요한?\s*지식[:：]\s*(.+)'
        ]
        
        import re
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # 쉼표로 분리
                items = [item.strip() for item in match.split(',')]
                prerequisites.extend(items)
        
        return list(set(prerequisites))
    
    def _extract_related_concepts(self, text: str) -> List[str]:
        """관련 개념 추출"""
        # 간단한 구현 - 실제로는 NER이나 지식 그래프 활용
        related = []
        
        # 괄호 안의 내용 추출
        import re
        parentheses = re.findall(r'\(([^)]+)\)', text)
        related.extend(parentheses)
        
        # 참조 표현 추출
        references = re.findall(r'참고[:：]\s*(.+)', text)
        related.extend(references)
        
        return list(set(related))


class KnowledgeGraphBuilder:
    """지식 그래프 구축기"""
    
    def __init__(self):
        """초기화"""
        self.nodes = {}  # 노드 (개념)
        self.edges = defaultdict(list)  # 엣지 (관계)
        self.domain_hierarchy = self._init_domain_hierarchy()
    
    def _init_domain_hierarchy(self) -> Dict[str, List[str]]:
        """도메인 계층 구조"""
        return {
            'STEM': [
                KnowledgeDomain.MATHEMATICS,
                KnowledgeDomain.PHYSICS,
                KnowledgeDomain.CHEMISTRY,
                KnowledgeDomain.BIOLOGY,
                KnowledgeDomain.COMPUTER,
                KnowledgeDomain.ELECTRICAL
            ],
            'Engineering': [
                KnowledgeDomain.ELECTRICAL,
                KnowledgeDomain.MECHANICAL,
                KnowledgeDomain.CIVIL,
                KnowledgeDomain.COMPUTER,
                KnowledgeDomain.CHEMICAL,
                KnowledgeDomain.AEROSPACE,
                KnowledgeDomain.BIOMEDICAL,
                KnowledgeDomain.MATERIALS
            ],
            'Liberal Arts': [
                KnowledgeDomain.PHILOSOPHY,
                KnowledgeDomain.HISTORY,
                KnowledgeDomain.LITERATURE,
                KnowledgeDomain.LINGUISTICS,
                KnowledgeDomain.ART,
                KnowledgeDomain.MUSIC
            ],
            'Social Sciences': [
                KnowledgeDomain.PSYCHOLOGY,
                KnowledgeDomain.SOCIOLOGY,
                KnowledgeDomain.ECONOMICS,
                KnowledgeDomain.POLITICS,
                KnowledgeDomain.LAW
            ],
            'Life Sciences': [
                KnowledgeDomain.BIOLOGY,
                KnowledgeDomain.MEDICINE,
                KnowledgeDomain.NURSING,
                KnowledgeDomain.PHARMACY,
                KnowledgeDomain.PUBLIC_HEALTH
            ]
        }
    
    def add_knowledge_unit(self, unit: KnowledgeUnit):
        """지식 단위를 그래프에 추가"""
        # 노드 추가
        node_id = self._generate_node_id(unit)
        self.nodes[node_id] = {
            'unit': unit,
            'domain': unit.domain,
            'complexity': unit.complexity,
            'info_type': unit.info_type
        }
        
        # 선수 지식과의 관계 추가
        for prereq in unit.prerequisites:
            prereq_id = self._find_or_create_node(prereq)
            self.edges[prereq_id].append(('prerequisite_of', node_id))
            self.edges[node_id].append(('requires', prereq_id))
        
        # 관련 개념과의 관계 추가
        for related in unit.related_concepts:
            related_id = self._find_or_create_node(related)
            self.edges[node_id].append(('related_to', related_id))
            self.edges[related_id].append(('related_to', node_id))
    
    def _generate_node_id(self, unit: KnowledgeUnit) -> str:
        """노드 ID 생성"""
        # 도메인과 콘텐츠 해시 기반 ID
        import hashlib
        content_hash = hashlib.md5(unit.content.encode()).hexdigest()[:8]
        return f"{unit.domain.value}_{content_hash}"
    
    def _find_or_create_node(self, concept: str) -> str:
        """개념 노드 찾기 또는 생성"""
        # 기존 노드에서 검색
        for node_id, node_data in self.nodes.items():
            if concept.lower() in node_data['unit'].content.lower():
                return node_id
        
        # 새 노드 생성
        new_unit = KnowledgeUnit(
            content=concept,
            domain=KnowledgeDomain.GENERAL,
            info_type=InformationType.CONCEPT
        )
        new_id = self._generate_node_id(new_unit)
        self.nodes[new_id] = {
            'unit': new_unit,
            'domain': new_unit.domain,
            'complexity': new_unit.complexity,
            'info_type': new_unit.info_type
        }
        
        return new_id
    
    def find_learning_path(self, start_concept: str, target_concept: str) -> List[str]:
        """학습 경로 찾기"""
        # BFS를 사용한 최단 경로 탐색
        from collections import deque
        
        start_id = self._find_or_create_node(start_concept)
        target_id = self._find_or_create_node(target_concept)
        
        if start_id == target_id:
            return [start_id]
        
        visited = set()
        queue = deque([(start_id, [start_id])])
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # 인접 노드 탐색
            for relation, neighbor_id in self.edges.get(current_id, []):
                if neighbor_id == target_id:
                    return path + [neighbor_id]
                
                if neighbor_id not in visited:
                    queue.append((neighbor_id, path + [neighbor_id]))
        
        return []  # 경로 없음
    
    def get_related_concepts(self, concept: str, max_depth: int = 2) -> List[Tuple[str, int]]:
        """관련 개념 가져오기"""
        concept_id = self._find_or_create_node(concept)
        
        related = []
        visited = set()
        
        def dfs(node_id: str, depth: int):
            if depth > max_depth or node_id in visited:
                return
            
            visited.add(node_id)
            
            for relation, neighbor_id in self.edges.get(node_id, []):
                if neighbor_id not in visited:
                    related.append((neighbor_id, depth))
                    dfs(neighbor_id, depth + 1)
        
        dfs(concept_id, 1)
        
        return related


class AdaptiveLearningAssistant:
    """적응형 학습 도우미"""
    
    def __init__(self, knowledge_graph: KnowledgeGraphBuilder):
        """초기화"""
        self.knowledge_graph = knowledge_graph
        self.user_profile = self._init_user_profile()
        self.learning_history = []
    
    def _init_user_profile(self) -> Dict[str, Any]:
        """사용자 프로필 초기화"""
        return {
            'expertise_levels': defaultdict(lambda: ContentComplexity.UNDERGRADUATE),
            'preferred_domains': [],
            'learning_style': 'balanced',  # visual, textual, practical, balanced
            'language_preference': 'ko',
            'completed_concepts': set(),
            'current_goals': []
        }
    
    def recommend_next_topic(self, current_topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """다음 학습 주제 추천"""
        recommendations = []
        
        if current_topic:
            # 현재 주제와 관련된 개념 찾기
            related = self.knowledge_graph.get_related_concepts(current_topic, max_depth=2)
            
            for concept_id, distance in related[:5]:
                if concept_id not in self.user_profile['completed_concepts']:
                    node_data = self.knowledge_graph.nodes.get(concept_id)
                    if node_data:
                        recommendations.append({
                            'concept': node_data['unit'].content,
                            'domain': node_data['domain'],
                            'complexity': node_data['complexity'],
                            'relevance_score': 1.0 / (distance + 1),
                            'reason': f"{current_topic}와(과) 관련된 개념"
                        })
        
        # 사용자 선호 도메인 기반 추천
        for domain in self.user_profile['preferred_domains']:
            domain_nodes = [
                node_id for node_id, data in self.knowledge_graph.nodes.items()
                if data['domain'] == domain and node_id not in self.user_profile['completed_concepts']
            ]
            
            for node_id in domain_nodes[:3]:
                node_data = self.knowledge_graph.nodes[node_id]
                recommendations.append({
                    'concept': node_data['unit'].content,
                    'domain': node_data['domain'],
                    'complexity': node_data['complexity'],
                    'relevance_score': 0.8,
                    'reason': f"선호 도메인 ({domain.value}) 기반 추천"
                })
        
        # 중복 제거 및 정렬
        unique_recommendations = {r['concept']: r for r in recommendations}.values()
        return sorted(unique_recommendations, key=lambda x: x['relevance_score'], reverse=True)
    
    def adapt_explanation(self, concept: str, user_level: ContentComplexity) -> str:
        """사용자 수준에 맞춘 설명 조정"""
        concept_node = self.knowledge_graph._find_or_create_node(concept)
        node_data = self.knowledge_graph.nodes.get(concept_node)
        
        if not node_data:
            return "해당 개념을 찾을 수 없습니다."
        
        unit = node_data['unit']
        
        # 복잡도에 따른 설명 조정
        if user_level == ContentComplexity.ELEMENTARY:
            return self._simplify_explanation(unit.content)
        elif user_level == ContentComplexity.EXPERT:
            return self._enhance_explanation(unit.content)
        else:
            return unit.content
    
    def _simplify_explanation(self, content: str) -> str:
        """설명 단순화"""
        # 실제로는 NLP를 사용한 단순화
        simplified = content
        
        # 전문 용어를 쉬운 말로 대체
        replacements = {
            '적분': '더하기를 계속하는 것',
            '미분': '변화율을 구하는 것',
            '벡터': '크기와 방향이 있는 양',
            '행렬': '숫자를 표로 정리한 것'
        }
        
        for term, simple in replacements.items():
            simplified = simplified.replace(term, f"{term}({simple})")
        
        return simplified
    
    def _enhance_explanation(self, content: str) -> str:
        """설명 강화"""
        # 더 많은 기술적 세부사항 추가
        enhanced = content + "\n\n[고급 설명]\n"
        enhanced += "- 수학적 증명 및 유도 과정\n"
        enhanced += "- 관련 연구 논문 참조\n"
        enhanced += "- 실제 응용 사례 및 최신 동향"
        
        return enhanced
    
    def generate_practice_problems(self, concept: str, difficulty: ContentComplexity, count: int = 5) -> List[Dict[str, Any]]:
        """연습 문제 생성"""
        problems = []
        
        # 개념 노드 찾기
        concept_node = self.knowledge_graph._find_or_create_node(concept)
        node_data = self.knowledge_graph.nodes.get(concept_node)
        
        if not node_data:
            return problems
        
        unit = node_data['unit']
        
        # 문제 유형 정의
        problem_types = {
            InformationType.CONCEPT: ['정의 문제', '개념 이해', '참/거짓 판별'],
            InformationType.FORMULA: ['계산 문제', '공식 적용', '변형 문제'],
            InformationType.THEOREM: ['증명 문제', '정리 적용', '반례 찾기'],
            InformationType.ALGORITHM: ['구현 문제', '복잡도 분석', '최적화']
        }
        
        # 난이도별 문제 생성
        for i in range(count):
            problem_type = problem_types.get(unit.info_type, ['일반 문제'])[i % len(problem_types.get(unit.info_type, ['일반 문제']))]
            
            problems.append({
                'id': f"{concept_node}_prob_{i+1}",
                'concept': concept,
                'type': problem_type,
                'difficulty': difficulty,
                'domain': unit.domain,
                'estimated_time': self._estimate_solving_time(difficulty),
                'hints_available': True,
                'solution_steps': self._generate_solution_steps(problem_type, difficulty)
            })
        
        return problems
    
    def _estimate_solving_time(self, difficulty: ContentComplexity) -> int:
        """문제 해결 예상 시간 (분)"""
        time_map = {
            ContentComplexity.ELEMENTARY: 5,
            ContentComplexity.MIDDLE_SCHOOL: 10,
            ContentComplexity.HIGH_SCHOOL: 15,
            ContentComplexity.UNDERGRADUATE: 20,
            ContentComplexity.GRADUATE: 30,
            ContentComplexity.RESEARCH: 45,
            ContentComplexity.EXPERT: 60
        }
        return time_map.get(difficulty, 20)
    
    def _generate_solution_steps(self, problem_type: str, difficulty: ContentComplexity) -> int:
        """해결 단계 수 생성"""
        base_steps = {
            '정의 문제': 2,
            '계산 문제': 4,
            '증명 문제': 6,
            '구현 문제': 8
        }
        
        # 난이도에 따라 단계 수 조정
        difficulty_multiplier = {
            ContentComplexity.ELEMENTARY: 0.5,
            ContentComplexity.HIGH_SCHOOL: 1.0,
            ContentComplexity.UNDERGRADUATE: 1.5,
            ContentComplexity.GRADUATE: 2.0,
            ContentComplexity.EXPERT: 3.0
        }
        
        base = base_steps.get(problem_type, 3)
        multiplier = difficulty_multiplier.get(difficulty, 1.0)
        
        return max(1, int(base * multiplier))


class UniversalKnowledgeOrchestrator:
    """범용 지식 시스템 오케스트레이터"""
    
    def __init__(self):
        """초기화"""
        self.extractor = UniversalKnowledgeExtractor()
        self.knowledge_graph = KnowledgeGraphBuilder()
        self.learning_assistant = AdaptiveLearningAssistant(self.knowledge_graph)
        self.domain_experts = self._init_domain_experts()
    
    def _init_domain_experts(self) -> Dict[KnowledgeDomain, Any]:
        """도메인별 전문가 모듈 초기화"""
        # 실제로는 각 도메인별 특화 모델/규칙
        return {
            KnowledgeDomain.MATHEMATICS: MathematicsExpert(),
            KnowledgeDomain.PHYSICS: PhysicsExpert(),
            KnowledgeDomain.COMPUTER: ComputerScienceExpert(),
            KnowledgeDomain.AI_ML: AIMLExpert(),
            # ... 더 많은 전문가
        }
    
    def process_content(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """콘텐츠 처리"""
        results = {
            'success': True,
            'knowledge_units': [],
            'detected_domains': [],
            'complexity_level': None,
            'recommendations': [],
            'learning_path': []
        }
        
        try:
            # 1. 지식 추출
            knowledge_units = self.extractor.extract_knowledge(content, context)
            results['knowledge_units'] = knowledge_units
            
            # 2. 지식 그래프 구축
            for unit in knowledge_units:
                self.knowledge_graph.add_knowledge_unit(unit)
            
            # 3. 도메인 분석
            domains = list(set(unit.domain for unit in knowledge_units))
            results['detected_domains'] = domains
            
            # 4. 복잡도 분석
            complexities = [unit.complexity for unit in knowledge_units]
            if complexities:
                # 최빈값
                results['complexity_level'] = max(set(complexities), key=complexities.count)
            
            # 5. 학습 추천
            if knowledge_units:
                main_concept = knowledge_units[0].content[:50]
                results['recommendations'] = self.learning_assistant.recommend_next_topic(main_concept)
            
            # 6. 도메인별 전문 처리
            for domain in domains[:2]:  # 상위 2개 도메인
                if domain in self.domain_experts:
                    expert = self.domain_experts[domain]
                    expert_analysis = expert.analyze(content)
                    results[f'{domain.value}_analysis'] = expert_analysis
            
        except Exception as e:
            logger.error(f"Content processing failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def generate_explanation(
        self,
        concept: str,
        target_audience: ContentComplexity,
        style: str = 'balanced'
    ) -> str:
        """개념 설명 생성"""
        explanation = self.learning_assistant.adapt_explanation(concept, target_audience)
        
        # 스타일에 따른 조정
        if style == 'visual':
            explanation += "\n\n[시각적 설명]\n다이어그램과 그래프를 통한 이해..."
        elif style == 'practical':
            explanation += "\n\n[실습 예제]\n실제 적용 사례와 연습..."
        elif style == 'theoretical':
            explanation += "\n\n[이론적 배경]\n수학적 증명과 원리..."
        
        return explanation
    
    def create_curriculum(
        self,
        target_domain: KnowledgeDomain,
        start_level: ContentComplexity,
        end_level: ContentComplexity,
        duration_weeks: int
    ) -> Dict[str, Any]:
        """커리큘럼 생성"""
        curriculum = {
            'domain': target_domain,
            'duration': duration_weeks,
            'start_level': start_level,
            'end_level': end_level,
            'weekly_plan': [],
            'milestones': []
        }
        
        # 도메인의 핵심 개념 수집
        domain_concepts = [
            node_id for node_id, data in self.knowledge_graph.nodes.items()
            if data['domain'] == target_domain
        ]
        
        # 복잡도별로 정렬
        sorted_concepts = sorted(
            domain_concepts,
            key=lambda x: list(ContentComplexity).index(self.knowledge_graph.nodes[x]['complexity'])
        )
        
        # 주차별 계획 생성
        concepts_per_week = max(1, len(sorted_concepts) // duration_weeks)
        
        for week in range(duration_weeks):
            week_concepts = sorted_concepts[
                week * concepts_per_week : (week + 1) * concepts_per_week
            ]
            
            curriculum['weekly_plan'].append({
                'week': week + 1,
                'concepts': [self.knowledge_graph.nodes[c]['unit'].content[:50] for c in week_concepts],
                'estimated_hours': len(week_concepts) * 3,
                'assignments': self._generate_weekly_assignments(week_concepts)
            })
        
        # 마일스톤 설정
        milestone_weeks = [duration_weeks // 4, duration_weeks // 2, 3 * duration_weeks // 4, duration_weeks]
        for i, week in enumerate(milestone_weeks):
            curriculum['milestones'].append({
                'week': week,
                'name': f"Phase {i+1} Assessment",
                'objectives': f"Complete {25 * (i+1)}% of curriculum",
                'assessment_type': 'project' if i == 3 else 'exam'
            })
        
        return curriculum
    
    def _generate_weekly_assignments(self, concept_ids: List[str]) -> List[Dict[str, Any]]:
        """주차별 과제 생성"""
        assignments = []
        
        for concept_id in concept_ids[:3]:  # 주당 최대 3개 과제
            node_data = self.knowledge_graph.nodes.get(concept_id)
            if node_data:
                unit = node_data['unit']
                assignments.append({
                    'type': 'practice' if unit.info_type == InformationType.ALGORITHM else 'study',
                    'concept': unit.content[:30],
                    'estimated_time': 90,  # 분
                    'resources': ['textbook', 'online_tutorial', 'practice_problems']
                })
        
        return assignments


# 도메인별 전문가 클래스들 (예시)
class MathematicsExpert:
    """수학 전문가"""
    
    def analyze(self, content: str) -> Dict[str, Any]:
        """수학적 내용 분석"""
        return {
            'has_formulas': '=' in content or '∫' in content,
            'theorem_count': content.count('정리'),
            'proof_required': '증명' in content,
            'topics': self._extract_math_topics(content)
        }
    
    def _extract_math_topics(self, content: str) -> List[str]:
        """수학 주제 추출"""
        topics = []
        
        topic_keywords = {
            '해석학': ['미분', '적분', '극한', '연속'],
            '선형대수': ['행렬', '벡터', '고유값', '선형변환'],
            '확률론': ['확률', '분포', '기댓값', '분산'],
            '정수론': ['소수', '약수', '합동', '오일러'],
            '위상수학': ['위상', '연결성', '컴팩트', '하우스도르프']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content for keyword in keywords):
                topics.append(topic)
        
        return topics


class PhysicsExpert:
    """물리학 전문가"""
    
    def analyze(self, content: str) -> Dict[str, Any]:
        """물리학적 내용 분석"""
        return {
            'has_units': any(unit in content for unit in ['m/s', 'kg', 'N', 'J']),
            'physics_laws': self._extract_physics_laws(content),
            'experiment_suggested': '실험' in content,
            'topics': self._extract_physics_topics(content)
        }
    
    def _extract_physics_laws(self, content: str) -> List[str]:
        """물리 법칙 추출"""
        laws = []
        
        law_patterns = {
            '뉴턴의 법칙': ['F=ma', '작용 반작용', '관성'],
            '열역학 법칙': ['엔트로피', '에너지 보존', '절대영도'],
            '전자기 법칙': ['쿨롱', '패러데이', '암페어'],
            '상대성 이론': ['E=mc²', '시공간', '광속']
        }
        
        for law, patterns in law_patterns.items():
            if any(pattern in content for pattern in patterns):
                laws.append(law)
        
        return laws
    
    def _extract_physics_topics(self, content: str) -> List[str]:
        """물리학 주제 추출"""
        topics = []
        
        topic_keywords = {
            '역학': ['힘', '운동', '에너지', '운동량'],
            '전자기학': ['전기', '자기', '전자기파', '회로'],
            '열역학': ['열', '온도', '엔트로피', '상전이'],
            '양자물리': ['양자', '파동함수', '불확정성', '터널링'],
            '상대론': ['상대성', '시공간', '중력', '블랙홀']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content for keyword in keywords):
                topics.append(topic)
        
        return topics


class ComputerScienceExpert:
    """컴퓨터공학 전문가"""
    
    def analyze(self, content: str) -> Dict[str, Any]:
        """컴퓨터공학 내용 분석"""
        return {
            'has_code': any(indicator in content for indicator in ['def ', 'class ', 'function', '{']),
            'algorithms': self._extract_algorithms(content),
            'complexity_mentioned': 'O(' in content,
            'topics': self._extract_cs_topics(content)
        }
    
    def _extract_algorithms(self, content: str) -> List[str]:
        """알고리즘 추출"""
        algorithms = []
        
        algo_patterns = [
            '정렬', '탐색', 'BFS', 'DFS', '다이나믹 프로그래밍',
            'DP', '그리디', '분할정복', '백트래킹'
        ]
        
        for pattern in algo_patterns:
            if pattern in content:
                algorithms.append(pattern)
        
        return algorithms
    
    def _extract_cs_topics(self, content: str) -> List[str]:
        """컴퓨터공학 주제 추출"""
        topics = []
        
        topic_keywords = {
            '자료구조': ['배열', '리스트', '트리', '그래프', '해시'],
            '알고리즘': ['정렬', '탐색', '복잡도', '최적화'],
            '운영체제': ['프로세스', '스레드', '메모리', '스케줄링'],
            '네트워크': ['TCP', 'IP', 'HTTP', '라우팅', '프로토콜'],
            '데이터베이스': ['SQL', '트랜잭션', '인덱스', '정규화']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content for keyword in keywords):
                topics.append(topic)
        
        return topics


class AIMLExpert:
    """AI/ML 전문가"""
    
    def analyze(self, content: str) -> Dict[str, Any]:
        """AI/ML 내용 분석"""
        return {
            'ml_algorithms': self._extract_ml_algorithms(content),
            'has_math': any(symbol in content for symbol in ['∇', 'Σ', '∂']),
            'frameworks_mentioned': self._extract_frameworks(content),
            'topics': self._extract_ml_topics(content)
        }
    
    def _extract_ml_algorithms(self, content: str) -> List[str]:
        """ML 알고리즘 추출"""
        algorithms = []
        
        ml_algos = [
            'neural network', '신경망', 'CNN', 'RNN', 'LSTM',
            'transformer', 'SVM', 'decision tree', '결정트리',
            'random forest', 'k-means', 'PCA'
        ]
        
        for algo in ml_algos:
            if algo.lower() in content.lower():
                algorithms.append(algo)
        
        return algorithms
    
    def _extract_frameworks(self, content: str) -> List[str]:
        """ML 프레임워크 추출"""
        frameworks = []
        
        framework_list = [
            'TensorFlow', 'PyTorch', 'Keras', 'scikit-learn',
            'JAX', 'MXNet', 'Caffe', 'Theano'
        ]
        
        for fw in framework_list:
            if fw.lower() in content.lower():
                frameworks.append(fw)
        
        return frameworks
    
    def _extract_ml_topics(self, content: str) -> List[str]:
        """ML 주제 추출"""
        topics = []
        
        topic_keywords = {
            '지도학습': ['분류', '회귀', '레이블', '타겟'],
            '비지도학습': ['클러스터링', '차원축소', 'k-means', 'PCA'],
            '강화학습': ['에이전트', '보상', '정책', 'Q-learning'],
            '딥러닝': ['신경망', 'CNN', 'RNN', '역전파'],
            '자연어처리': ['NLP', '토큰', '임베딩', 'BERT']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content for keyword in keywords):
                topics.append(topic)
        
        return topics


# 사용 예시
if __name__ == "__main__":
    # 시스템 초기화
    orchestrator = UniversalKnowledgeOrchestrator()
    
    # 콘텐츠 처리
    sample_content = """
    미분방정식은 함수와 그 도함수들 사이의 관계를 나타내는 방정식입니다.
    물리학에서 뉴턴의 운동 법칙 F=ma는 2계 미분방정식으로 표현됩니다.
    컴퓨터 과학에서는 미분방정식의 수치해법을 위해 오일러 방법이나 룽게-쿠타 방법을 사용합니다.
    """
    
    result = orchestrator.process_content(sample_content)
    
    print("Detected domains:", [d.value for d in result['detected_domains']])
    print("Complexity level:", result['complexity_level'].value if result['complexity_level'] else None)
    print("Recommendations:", len(result['recommendations']))
    
    # 커리큘럼 생성
    curriculum = orchestrator.create_curriculum(
        target_domain=KnowledgeDomain.MATHEMATICS,
        start_level=ContentComplexity.HIGH_SCHOOL,
        end_level=ContentComplexity.UNDERGRADUATE,
        duration_weeks=16
    )
    
    print(f"Created {len(curriculum['weekly_plan'])}-week curriculum")