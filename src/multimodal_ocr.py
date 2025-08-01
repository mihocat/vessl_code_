#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Multimodal OCR System
멀티모달 OCR 시스템 - 한국어, 수식, 다이어그램 통합 처리
"""

import logging
from typing import Dict, List, Any, Union, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
import re
from dataclasses import dataclass

# OCR 모델들
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logging.warning("PaddleOCR not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available")

try:
    from pix2tex import LatexOCR
    PIX2TEX_AVAILABLE = True
except ImportError:
    PIX2TEX_AVAILABLE = False
    logging.warning("Pix2Tex not available")

try:
    import layoutparser as lp
    LAYOUT_AVAILABLE = True
except ImportError:
    LAYOUT_AVAILABLE = False
    logging.warning("LayoutParser not available")

logger = logging.getLogger(__name__)


@dataclass
class Region:
    """문서 영역 정보"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    type: str  # 'text', 'formula', 'diagram', 'table'
    confidence: float
    content: Optional[str] = None
    image: Optional[np.ndarray] = None


class DocumentLayoutAnalyzer:
    """문서 레이아웃 분석기"""
    
    def __init__(self):
        """레이아웃 분석기 초기화"""
        self.model = None
        if LAYOUT_AVAILABLE:
            try:
                # Detectron2 기반 레이아웃 모델
                self.model = lp.Detectron2LayoutModel(
                    'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                    label_map={
                        0: "text", 
                        1: "title", 
                        2: "list", 
                        3: "table", 
                        4: "figure"
                    }
                )
                logger.info("Layout analyzer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize layout analyzer: {e}")
                self.model = None
        
    def analyze_layout(self, image: Union[Image.Image, np.ndarray]) -> List[Region]:
        """문서 레이아웃 분석"""
        if self.model is None:
            # 폴백: 간단한 영역 검출
            return self._simple_region_detection(image)
        
        try:
            # PIL Image를 numpy array로 변환
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # 레이아웃 검출
            layout = self.model.detect(image_array)
            
            regions = []
            for block in layout:
                region = Region(
                    bbox=(
                        int(block.block.x_1),
                        int(block.block.y_1),
                        int(block.block.x_2),
                        int(block.block.y_2)
                    ),
                    type=block.type,
                    confidence=block.score,
                    image=image_array[
                        int(block.block.y_1):int(block.block.y_2),
                        int(block.block.x_1):int(block.block.x_2)
                    ]
                )
                regions.append(region)
                
            return regions
            
        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
            return self._simple_region_detection(image)
    
    def _simple_region_detection(self, image: Union[Image.Image, np.ndarray]) -> List[Region]:
        """간단한 영역 검출 (폴백)"""
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
            
        # 전체 이미지를 하나의 영역으로
        h, w = image_array.shape[:2]
        return [Region(
            bbox=(0, 0, w, h),
            type='mixed',
            confidence=1.0,
            image=image_array
        )]


class KoreanOCR:
    """한국어 OCR 처리기"""
    
    def __init__(self):
        """한국어 OCR 초기화"""
        self.paddle_ocr = None
        self.easy_ocr = None
        
        if PADDLE_AVAILABLE:
            try:
                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='korean',
                    use_gpu=True
                )
                logger.info("PaddleOCR initialized for Korean")
            except Exception as e:
                logger.error(f"PaddleOCR initialization failed: {e}")
        
        if EASYOCR_AVAILABLE and not self.paddle_ocr:
            try:
                self.easy_ocr = easyocr.Reader(['ko', 'en'])
                logger.info("EasyOCR initialized for Korean")
            except Exception as e:
                logger.error(f"EasyOCR initialization failed: {e}")
    
    def extract_text(self, image: Union[Image.Image, np.ndarray]) -> str:
        """한국어 텍스트 추출"""
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        text_results = []
        
        # PaddleOCR 시도
        if self.paddle_ocr:
            try:
                results = self.paddle_ocr.ocr(image_array, cls=True)
                if results and results[0]:
                    for line in results[0]:
                        text = line[1][0]
                        confidence = line[1][1]
                        if confidence > 0.5:
                            text_results.append(text)
                            
            except Exception as e:
                logger.error(f"PaddleOCR failed: {e}")
        
        # EasyOCR 시도 (PaddleOCR 실패 시)
        elif self.easy_ocr:
            try:
                results = self.easy_ocr.readtext(image_array)
                for (bbox, text, confidence) in results:
                    if confidence > 0.5:
                        text_results.append(text)
                        
            except Exception as e:
                logger.error(f"EasyOCR failed: {e}")
        
        return ' '.join(text_results)


class MathOCR:
    """수식 OCR 처리기"""
    
    def __init__(self):
        """수식 OCR 초기화"""
        self.latex_ocr = None
        
        if PIX2TEX_AVAILABLE:
            try:
                self.latex_ocr = LatexOCR()
                logger.info("LaTeX OCR initialized")
            except Exception as e:
                logger.error(f"LaTeX OCR initialization failed: {e}")
    
    def extract_formula(self, image: Union[Image.Image, np.ndarray]) -> str:
        """수식을 LaTeX로 변환"""
        if not self.latex_ocr:
            return ""
        
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # LaTeX 변환
            latex_formula = self.latex_ocr(image)
            return latex_formula
            
        except Exception as e:
            logger.error(f"Math OCR failed: {e}")
            return ""


class DiagramAnalyzer:
    """다이어그램/회로도 분석기"""
    
    def __init__(self):
        """다이어그램 분석기 초기화"""
        self.component_patterns = {
            'resistor': r'[R]\d+',
            'capacitor': r'[C]\d+',
            'inductor': r'[L]\d+',
            'voltage': r'[V]\d+',
            'current': r'[I]\d+',
            'ground': r'GND',
        }
    
    def analyze_diagram(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """다이어그램 분석"""
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        analysis = {
            'type': 'circuit',  # or 'block_diagram', 'graph', etc.
            'components': [],
            'connections': [],
            'labels': []
        }
        
        # 간단한 텍스트 추출로 컴포넌트 식별
        korean_ocr = KoreanOCR()
        text = korean_ocr.extract_text(image_array)
        
        # 전기 컴포넌트 패턴 매칭
        for comp_type, pattern in self.component_patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                analysis['components'].append({
                    'type': comp_type,
                    'label': match
                })
        
        return analysis


class MultimodalOCRPipeline:
    """통합 멀티모달 OCR 파이프라인"""
    
    def __init__(self):
        """파이프라인 초기화"""
        logger.info("Initializing Multimodal OCR Pipeline...")
        
        self.layout_analyzer = DocumentLayoutAnalyzer()
        self.korean_ocr = KoreanOCR()
        self.math_ocr = MathOCR()
        self.diagram_analyzer = DiagramAnalyzer()
        
        logger.info("Multimodal OCR Pipeline initialized successfully")
    
    def process_image(self, image: Union[Image.Image, str, bytes]) -> Dict[str, Any]:
        """이미지 통합 처리"""
        # 이미지 로드
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            import io
            image = Image.open(io.BytesIO(image)).convert('RGB')
        
        # 1. 레이아웃 분석
        regions = self.layout_analyzer.analyze_layout(image)
        logger.info(f"Detected {len(regions)} regions")
        
        # 2. 각 영역별 처리
        results = {
            'regions': [],
            'full_text': [],
            'formulas': [],
            'diagrams': [],
            'structured_content': {}
        }
        
        for i, region in enumerate(regions):
            logger.info(f"Processing region {i+1}/{len(regions)}: {region.type}")
            
            region_result = {
                'type': region.type,
                'bbox': region.bbox,
                'confidence': region.confidence,
                'content': None
            }
            
            # 영역 타입에 따른 처리
            if region.type in ['text', 'title', 'list']:
                # 한국어 텍스트 추출
                text = self.korean_ocr.extract_text(region.image)
                region_result['content'] = text
                results['full_text'].append(text)
                
            elif region.type == 'figure':
                # 수식인지 다이어그램인지 판단
                # 간단한 휴리스틱: 수식은 보통 작고, 특정 패턴 포함
                h, w = region.image.shape[:2]
                aspect_ratio = w / h
                
                if aspect_ratio < 3 and h < 200:  # 수식일 가능성
                    formula = self.math_ocr.extract_formula(region.image)
                    if formula:
                        region_result['content'] = formula
                        region_result['type'] = 'formula'
                        results['formulas'].append(formula)
                else:  # 다이어그램
                    diagram_info = self.diagram_analyzer.analyze_diagram(region.image)
                    region_result['content'] = diagram_info
                    region_result['type'] = 'diagram'
                    results['diagrams'].append(diagram_info)
            
            elif region.type == 'table':
                # 표 처리 (현재는 텍스트로 처리)
                text = self.korean_ocr.extract_text(region.image)
                region_result['content'] = text
                results['full_text'].append(text)
            
            results['regions'].append(region_result)
        
        # 3. 구조화된 컨텐츠 생성
        results['structured_content'] = self._structure_content(results)
        
        return results
    
    def _structure_content(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """추출된 내용을 구조화"""
        structured = {
            'document_type': 'electrical_engineering',
            'sections': []
        }
        
        # 섹션별로 구성
        current_section = {
            'title': '',
            'content': [],
            'formulas': [],
            'diagrams': []
        }
        
        for region in results['regions']:
            if region['type'] == 'title':
                # 새 섹션 시작
                if current_section['content'] or current_section['formulas']:
                    structured['sections'].append(current_section)
                current_section = {
                    'title': region['content'],
                    'content': [],
                    'formulas': [],
                    'diagrams': []
                }
            elif region['type'] == 'text':
                current_section['content'].append(region['content'])
            elif region['type'] == 'formula':
                current_section['formulas'].append(region['content'])
            elif region['type'] == 'diagram':
                current_section['diagrams'].append(region['content'])
        
        # 마지막 섹션 추가
        if current_section['content'] or current_section['formulas']:
            structured['sections'].append(current_section)
        
        return structured
    
    def extract_question_and_solution(self, image: Union[Image.Image, str]) -> Dict[str, Any]:
        """문제와 풀이 분리 추출"""
        results = self.process_image(image)
        
        # 문제와 풀이 분리 로직
        question_keywords = ['문제', '질문', '구하시오', '계산하시오', '?']
        solution_keywords = ['풀이', '해답', '답', '해설', '따라서']
        
        question_parts = []
        solution_parts = []
        current_mode = 'question'  # 기본은 문제로 시작
        
        for text in results['full_text']:
            # 키워드 기반 모드 전환
            if any(keyword in text for keyword in solution_keywords):
                current_mode = 'solution'
            
            if current_mode == 'question':
                question_parts.append(text)
            else:
                solution_parts.append(text)
        
        return {
            'question': {
                'text': ' '.join(question_parts),
                'formulas': results['formulas'][:len(question_parts)],
                'diagrams': results['diagrams'][:1] if results['diagrams'] else []
            },
            'solution': {
                'text': ' '.join(solution_parts),
                'formulas': results['formulas'][len(question_parts):],
                'diagrams': results['diagrams'][1:] if len(results['diagrams']) > 1 else []
            },
            'raw_results': results
        }


# 테스트 및 유틸리티 함수
def test_ocr_pipeline():
    """OCR 파이프라인 테스트"""
    pipeline = MultimodalOCRPipeline()
    
    # 테스트 이미지 (예시)
    test_image_path = "test_image.png"
    
    try:
        results = pipeline.process_image(test_image_path)
        
        print("=== OCR Results ===")
        print(f"Regions detected: {len(results['regions'])}")
        print(f"Text extracted: {len(results['full_text'])} segments")
        print(f"Formulas found: {len(results['formulas'])}")
        print(f"Diagrams found: {len(results['diagrams'])}")
        
        print("\n=== Structured Content ===")
        for i, section in enumerate(results['structured_content']['sections']):
            print(f"\nSection {i+1}: {section['title']}")
            print(f"Content: {' '.join(section['content'][:100])}...")
            if section['formulas']:
                print(f"Formulas: {section['formulas']}")
                
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 실행
    test_ocr_pipeline()