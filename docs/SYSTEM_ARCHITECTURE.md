# 범용 RAG 시스템 아키텍처

## 개요
이 시스템은 도메인 독립적인 범용 RAG (Retrieval-Augmented Generation) 시스템으로, 모든 분야에서 사용 가능한 포괄적인 정보 추출 및 응답 생성 시스템입니다.

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

### 1. Universal OCR Pipeline (`universal_ocr_pipeline.py`)

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

### 3. 시스템 통합

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

## 사용 예시

### 1. 범용 사용
```python
# 도메인 자동 감지
ocr = DomainAdaptiveOCR()
result = ocr.process_adaptive(image, auto_detect=True)
```

### 2. 특정 도메인 지정
```python
# 의료 도메인
medical_ocr = DomainAdaptiveOCR(domain="medical")

# 법률 도메인
legal_ocr = DomainAdaptiveOCR(domain="legal")

# 학술 도메인
academic_ocr = DomainAdaptiveOCR(domain="academic")
```

### 3. 모듈형 RAG 파이프라인
```python
# 파이프라인 구성
pipeline = ModularRAGPipeline(config)

# 커스텀 모듈 추가
custom_module = MyCustomModule()
pipeline.add_module("custom", custom_module)

# 처리
result = pipeline.process(
    query="질문",
    image=image,
    domain="engineering"  # 선택적
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