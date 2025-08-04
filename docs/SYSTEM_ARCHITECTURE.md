# 범용 지능형 RAG 시스템 아키텍처

## 개요
이 시스템은 모든 학문 분야를 포괄하는 범용 지능형 RAG (Retrieval-Augmented Generation) 시스템으로, 지식 그래프 기반의 적응형 학습과 도메인 전문가 시스템을 통합한 차세대 AI 시스템입니다.

## 핵심 설계 원칙

### 1. 도메인 독립성
- 전기공학은 하나의 예시일 뿐, 시스템은 어떤 도메인에도 적용 가능
- 자동 도메인 감지 및 적응 기능
- 도메인별 최적화는 설정으로 조정 가능

### 2. 모듈형 아키텍처
- 각 기능이 독립적인 모듈로 구성
- 모듈의 추가/제거/교체가 용이
- 파이프라인 구성의 유연성

### 3. 포괄적 정보 추출
- 텍스트, 수식, 표, 다이어그램, 차트, 손글씨, 코드 등 모든 유형의 정보 추출
- 다국어 지원 (한국어, 영어, 일본어, 중국어 등)
- 멀티모달 처리 (텍스트 + 이미지)

## 시스템 구성

### 1. Universal Knowledge System (`universal_knowledge_system.py`)

#### 핵심 구성요소
- **KnowledgeDomain**: 40개 이상의 학문 분야 분류
  - 자연과학: 수학, 물리학, 화학, 생물학, 지구과학, 천문학
  - 공학: 전기전자, 기계, 토목, 컴퓨터, 화학, 항공우주, 의공학, 재료
  - 정보기술: 소프트웨어, AI/ML, 데이터과학, 사이버보안, 네트워크, DB
  - 인문사회: 철학, 역사, 문학, 언어학, 심리학, 사회학, 경제학, 정치학, 법학
  - 예술: 음악, 미술, 디자인, 건축
  - 의학/건강: 의학, 간호학, 약학, 공중보건
  - 비즈니스: 경영학, 금융, 마케팅, 회계
  - 교육: 교육학, 교수법

- **ContentComplexity**: 7단계 복잡도 수준
  - Elementary (초등)
  - Middle School (중등)
  - High School (고등)
  - Undergraduate (학부)
  - Graduate (대학원)
  - Research (연구)
  - Expert (전문가)

- **InformationType**: 13가지 정보 유형
  - 개념, 정의, 정리/법칙, 공식, 알고리즘, 절차, 예시
  - 증명, 실험, 사례연구, 역사적 사실, 통계, 시각자료, 참고자료

#### 주요 기능
1. **UniversalKnowledgeExtractor**
   - 자동 도메인 감지
   - 복잡도 추정
   - 정보 유형 분류
   - 선수 지식 추출
   - 관련 개념 매핑

2. **KnowledgeGraphBuilder**
   - 지식 노드/엣지 구축
   - 도메인 계층 구조 관리
   - 학습 경로 탐색
   - 관련 개념 네트워크

3. **AdaptiveLearningAssistant**
   - 사용자 프로필 관리
   - 맞춤형 학습 추천
   - 난이도 적응
   - 연습 문제 생성
   - 커리큘럼 설계

4. **도메인 전문가 시스템**
   - MathematicsExpert: 수학 전문 분석
   - PhysicsExpert: 물리학 법칙/실험 분석
   - ComputerScienceExpert: 알고리즘/자료구조 분석
   - AIMLExpert: AI/ML 모델/프레임워크 분석

### 2. Universal OCR Pipeline (`universal_ocr_pipeline.py`)

#### 주요 기능
- **다중 모델 앙상블**: 여러 OCR 모델의 결과를 종합하여 최적의 결과 도출
- **콘텐츠 유형별 특화 처리**:
  - 텍스트: EasyOCR, PaddleOCR, TrOCR, Tesseract
  - 수식: LaTeX OCR, Nougat
  - 표: Table Transformer, PaddleOCR Table
  - 다이어그램: DETR, Object Detection
  - 차트: ChartQA, Pix2Struct
  - 손글씨: TrOCR Handwritten
  - 코드: 패턴 기반 감지 + CodeBERT

#### Domain Adaptive OCR
```python
class DomainAdaptiveOCR(UniversalOCRPipeline):
    """도메인 적응형 OCR"""
    
    def auto_detect_domain(self, image):
        # 이미지에서 도메인 자동 감지
        
    def process_adaptive(self, image):
        # 도메인에 최적화된 처리
```

### 2. Modular RAG System (`modular_rag_system.py`)

#### 모듈 구성
1. **Query Analyzer Module**
   - 쿼리 유형 분류 (사실적, 분석적, 비교, 절차적, 개념적 등)
   - 언어 감지
   - 복잡도 추정

2. **OCR Module**
   - Universal OCR Pipeline 래퍼
   - 도메인 적응형 처리

3. **Retrieval Module**
   - 다양한 검색 전략 (벡터, 키워드, 하이브리드, 의미 기반, 문맥 기반)
   - 쿼리 유형별 최적 전략 선택

4. **Reranking Module**
   - Cross-encoder 기반 재순위
   - 쿼리 유형별 차별화된 재순위 전략

5. **Response Generator Module**
   - 쿼리 유형별 맞춤형 템플릿
   - 다국어 지원
   - 도메인별 포맷팅

### 4. Intelligent RAG System (`intelligent_rag_system.py`)

#### 핵심 구성요소
1. **IntelligentQueryProcessor**
   - 의도 감지: learn, solve, explain, compare, implement, analyze, debug, optimize
   - 도메인 자동 감지
   - 복잡도 추정
   - 응답 전략 결정

2. **ResponseStrategy**
   - Educational: 교육적 설명
   - Technical: 기술적 상세
   - Practical: 실용적 가이드
   - Conversational: 대화형
   - Analytical: 분석적
   - Creative: 창의적

3. **IntelligentRAGOrchestrator**
   - 비동기 병렬 처리
   - 도메인별 전문가 활용
   - 지식 그래프 기반 추천
   - 학습 경로 생성
   - 개인화 커리큘럼

#### 주요 기능
- **병렬 처리**: 쿼리 분석, 이미지 처리, 검색을 동시 수행
- **캐싱**: 자주 사용되는 결과 캐싱
- **적응형 응답**: 사용자 수준과 의도에 맞춘 응답
- **학습 지원**: 연습 문제 생성, 학습 경로 추천

### 5. 시스템 통합

#### Enhanced App (`enhanced_app.py`)
```python
# OCR 파이프라인 우선순위
1. Universal Domain-Adaptive OCR (최우선)
2. Korean Electrical OCR (폴백 1)
3. Multimodal OCR (폴백 2)
```

#### Advanced RAG System (`advanced_rag_system.py`)
```python
# 처리 모드
- fast: 빠른 응답 (기본 검색 + 간단한 재순위)
- balanced: 균형잡힌 처리 (하이브리드 검색 + Cross-encoder)
- reasoning: 심층 추론 (다단계 추론 체인)
```

## 주요 개선사항

### 1. 도메인 확장
- **전기공학 특화 → 40개 이상 학문 분야**
- 자연과학, 공학, 인문사회, 예술, 의학 등 전 분야 포괄
- 각 도메인별 전문가 시스템 구축

### 2. 지능형 처리
- **단순 RAG → 지식 그래프 기반 지능형 시스템**
- 사용자 의도 파악 및 맞춤형 응답
- 학습 경로 및 커리큘럼 자동 생성

### 3. 적응형 학습
- **고정 난이도 → 7단계 적응형 난이도**
- 사용자 프로필 기반 개인화
- 연습 문제 자동 생성

### 4. 멀티모달 강화
- **기본 OCR → 다중 모델 앙상블 OCR**
- 텍스트, 수식, 표, 다이어그램, 차트, 손글씨, 코드 인식
- 도메인 적응형 OCR

## 사용 예시

### 1. 범용 지능형 처리
```python
# 지능형 RAG 시스템 초기화
orchestrator = IntelligentRAGOrchestrator(config)

# 수학 문제 해결
result = orchestrator.process_sync(
    query="미분방정식을 이용한 물리 현상 모델링 방법을 설명해주세요",
    context={
        'user_profile': {
            'education_level': 'undergraduate',
            'preferred_domains': ['mathematics', 'physics']
        }
    }
)

# 프로그래밍 학습
result = orchestrator.process_sync(
    query="Python으로 딥러닝 모델을 구현하는 방법을 초보자 수준에서 설명해주세요",
    context={
        'user_profile': {
            'education_level': 'elementary',
            'learning_style': 'practical'
        }
    }
)
```

### 2. 지식 그래프 활용
```python
# 지식 추출 및 그래프 구축
orchestrator = UniversalKnowledgeOrchestrator()
content = "뉴턴의 운동법칙 F=ma는 힘, 질량, 가속도의 관계를 나타냅니다."
result = orchestrator.process_content(content)

# 학습 경로 생성
learning_path = orchestrator.knowledge_graph.find_learning_path(
    start_concept="뉴턴의 법칙",
    target_concept="라그랑주 역학"
)
```

### 3. 개인화 커리큘럼
```python
# 16주 AI/ML 커리큘럼 생성
curriculum = orchestrator.create_personalized_curriculum(
    user_profile={
        'current_level': 'undergraduate',
        'target_level': 'graduate',
        'available_hours_per_week': 20,
        'learning_style': 'balanced'
    },
    target_domain=KnowledgeDomain.AI_ML,
    duration_weeks=16
)

# 연습 문제 생성
problems = orchestrator.generate_practice_problems(
    concept="신경망 역전파",
    domain=KnowledgeDomain.AI_ML,
    difficulty=ContentComplexity.UNDERGRADUATE,
    count=10
)
```

### 4. 멀티모달 처리
```python
# 이미지 + 텍스트 통합 처리
result = await orchestrator.process_async(
    query="이 회로도에서 전류의 흐름을 분석하고 최적화 방안을 제시해주세요",
    image=circuit_diagram_image,
    context={
        'intent': 'analyze',
        'expected_output': 'technical_report'
    }
)
```

## 확장성

### 1. 새로운 OCR 모델 추가
```python
def _init_new_ocr_model(self):
    try:
        self.models['new_model'] = load_new_model()
    except:
        logger.warning("New model not available")
```

### 2. 새로운 도메인 추가
```python
self.domain_configs['new_domain'] = {
    'priority_models': ['text', 'formula'],
    'terminology': ['domain_specific_terms'],
    'layout_hints': ['document_types']
}
```

### 3. 새로운 처리 모듈 추가
```python
class NewProcessingModule(BaseModule):
    def process(self, context, data):
        # 커스텀 처리 로직
        return processed_data
```

## 성능 최적화

### 1. 모델 로드 최적화
- Lazy loading: 필요한 모델만 로드
- 모델 캐싱: 자주 사용되는 모델 메모리 유지

### 2. 병렬 처리
- 멀티 모델 동시 처리
- 비동기 파이프라인 실행

### 3. 결과 캐싱
- OCR 결과 캐싱
- 검색 결과 캐싱

## 모니터링 및 평가

### 1. 품질 메트릭
- Coverage: 얼마나 많은 영역을 커버했는가
- Confidence: 평균 신뢰도
- Completeness: 추출 완전성
- Diversity: 콘텐츠 유형의 다양성

### 2. 성능 메트릭
- 응답 시간
- 메모리 사용량
- GPU 활용률

## 향후 계획

1. **실시간 처리**: 스트리밍 방식의 점진적 응답
2. **멀티 페이지 문서**: PDF 전체 문서 처리
3. **비디오 지원**: 동영상에서 정보 추출
4. **음성 지원**: 음성 질의 및 응답
5. **협업 기능**: 다중 사용자 동시 처리