# 이미지 및 수식 파싱 개선 계획

**작성일**: 2025-08-04
**목표**: 한국어, 수식, 이미지를 완벽하게 파싱할 수 있는 챗봇 구축

## 현재 문제점

### 1. 이미지 파싱 불안정
- Florence-2 초기화 실패 빈번
- GPU 메모리 부족 문제
- OCR 정확도 낮음 (특히 한국어)

### 2. 수식 인식 부족
- LaTeX 수식 파싱 미지원
- 손글씨 수식 인식 불가
- 수학 기호 해석 오류

### 3. 한국어 처리 문제
- Tesseract OCR 한국어 정확도 낮음
- 한글 폰트 렌더링 문제
- 복합 문서 (한글+영어+수식) 처리 미흡

## 개선 전략

### Phase 1: 안정적인 이미지 처리 파이프라인 구축

#### 1.1 GPU 메모리 최적화
```python
# 현재 문제: Florence-2가 너무 많은 GPU 메모리 사용
# 해결책: 
# - 모델 로딩 최적화
# - 배치 처리 대신 순차 처리
# - 모델 양자화 적용
```

#### 1.2 다중 OCR 엔진 통합
```python
class MultiEngineOCR:
    def __init__(self):
        self.engines = {
            'tesseract': TesseractOCR(),
            'easyocr': EasyOCR(['ko', 'en']),
            'paddleocr': PaddleOCR(use_angle_cls=True, lang='korean')
        }
    
    def extract_text(self, image):
        # 여러 엔진 결과를 앙상블
        results = {}
        for name, engine in self.engines.items():
            try:
                results[name] = engine.extract(image)
            except:
                continue
        return self.ensemble_results(results)
```

### Phase 2: 수식 인식 시스템 구현

#### 2.1 수식 감지 및 추출
```python
class MathFormulaDetector:
    def __init__(self):
        self.formula_detector = YOLO('formula_detection.pt')
        self.latex_converter = LatexOCR()
    
    def detect_formulas(self, image):
        # 1. 이미지에서 수식 영역 감지
        formula_regions = self.formula_detector(image)
        
        # 2. 각 영역을 LaTeX로 변환
        formulas = []
        for region in formula_regions:
            latex = self.latex_converter.convert(region)
            formulas.append({
                'bbox': region.bbox,
                'latex': latex,
                'confidence': region.confidence
            })
        return formulas
```

#### 2.2 수식 해석 및 계산
```python
class MathSolver:
    def __init__(self):
        self.sympy_engine = SympyEngine()
        self.wolfram_api = WolframAPI()  # 선택적
    
    def solve(self, latex_formula):
        # LaTeX를 SymPy 표현식으로 변환
        expr = self.parse_latex(latex_formula)
        
        # 수식 유형 판별 (방정식, 미분, 적분 등)
        formula_type = self.classify_formula(expr)
        
        # 적절한 솔버 적용
        solution = self.apply_solver(expr, formula_type)
        
        return {
            'original': latex_formula,
            'parsed': str(expr),
            'type': formula_type,
            'solution': solution
        }
```

### Phase 3: 통합 파이프라인 구축

#### 3.1 향상된 멀티모달 처리
```python
class EnhancedMultimodalProcessor:
    def __init__(self):
        self.text_extractor = MultiEngineOCR()
        self.formula_detector = MathFormulaDetector()
        self.image_analyzer = LightweightImageAnalyzer()  # Florence-2 대체
        self.layout_analyzer = LayoutAnalyzer()
    
    def process_image(self, image, question):
        # 1. 레이아웃 분석
        layout = self.layout_analyzer.analyze(image)
        
        # 2. 각 영역별 처리
        results = {
            'text_regions': [],
            'formula_regions': [],
            'table_regions': [],
            'figure_regions': []
        }
        
        for region in layout.regions:
            if region.type == 'text':
                text = self.text_extractor.extract_text(region.crop)
                results['text_regions'].append(text)
            
            elif region.type == 'formula':
                formula = self.formula_detector.detect_formulas(region.crop)
                results['formula_regions'].extend(formula)
            
            elif region.type == 'table':
                table = self.extract_table(region.crop)
                results['table_regions'].append(table)
        
        # 3. 컨텍스트 통합
        combined_context = self.combine_results(results, question)
        
        return combined_context
```

#### 3.2 경량화된 이미지 분석기
```python
class LightweightImageAnalyzer:
    """Florence-2 대체용 경량 분석기"""
    
    def __init__(self):
        # CLIP 기반 경량 모델 사용
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 한국어 지원을 위한 번역기
        self.translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    
    def analyze(self, image, question=None):
        # 이미지 특징 추출
        image_features = self.extract_features(image)
        
        # 질문이 있으면 이미지-텍스트 매칭
        if question:
            relevance = self.compute_relevance(image_features, question)
        
        # 간단한 캡션 생성
        caption = self.generate_caption(image_features)
        
        return {
            'caption': caption,
            'relevance': relevance if question else None,
            'features': image_features
        }
```

### Phase 4: Gradio UI 개선

#### 4.1 개선된 UI 컴포넌트
```python
def create_enhanced_gradio_app():
    with gr.Blocks() as app:
        # 이미지 처리 상태 표시
        with gr.Row():
            image_status = gr.Textbox(
                label="이미지 처리 상태",
                interactive=False
            )
        
        # 수식 미리보기
        with gr.Row():
            formula_preview = gr.Markdown(
                label="감지된 수식",
                visible=False
            )
        
        # OCR 결과 편집 가능
        with gr.Row():
            ocr_result = gr.Textbox(
                label="추출된 텍스트 (수정 가능)",
                interactive=True,
                visible=False
            )
```

## 구현 우선순위

1. **즉시 구현 (1일)**
   - MultiEngineOCR 클래스 구현
   - GPU 메모리 최적화
   - 기본 수식 감지 기능

2. **단기 구현 (2-3일)**
   - 전체 파이프라인 통합
   - Gradio UI 개선
   - 테스트 및 최적화

3. **장기 개선 (1주일+)**
   - 고급 수식 솔버 통합
   - 복잡한 레이아웃 분석
   - 성능 최적화

## 필요 라이브러리

```python
# requirements.txt 추가
easyocr>=1.7.0
paddlepaddle==2.6.0
paddleocr>=2.7.0
latex2sympy2>=1.9.0
sympy>=1.12
pdf2image>=1.16.3
layoutparser>=0.3.4
transformers>=4.36.0  # CLIP용
```

## 성공 지표

1. **OCR 정확도**
   - 한국어: 95% 이상
   - 영어: 98% 이상
   - 수식: 90% 이상

2. **처리 속도**
   - 이미지당 3초 이내
   - GPU 메모리 사용량 < 4GB

3. **안정성**
   - 초기화 실패율 < 1%
   - 에러 복구 가능