#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal OCR Pipeline
범용 OCR 파이프라인 - 모든 도메인에 적용 가능
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image
import numpy as np
import torch
import io
import base64
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """콘텐츠 유형"""
    TEXT = "text"
    FORMULA = "formula"
    TABLE = "table"
    DIAGRAM = "diagram"
    CHART = "chart"
    HANDWRITING = "handwriting"
    CODE = "code"
    MIXED = "mixed"


@dataclass
class OCRResult:
    """OCR 결과 구조체"""
    content_type: ContentType
    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    metadata: Optional[Dict[str, Any]] = None
    

class UniversalOCRPipeline:
    """범용 OCR 파이프라인"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        범용 OCR 파이프라인 초기화
        
        Args:
            config: 설정 옵션
        """
        self.config = config or {}
        self.models = {}
        self.processors = {}
        
        # 1. 텍스트 OCR 모델들
        self._init_text_ocr_models()
        
        # 2. 수식 OCR 모델들
        self._init_formula_ocr_models()
        
        # 3. 표 추출 모델들
        self._init_table_extraction_models()
        
        # 4. 다이어그램/차트 분석 모델들
        self._init_diagram_analysis_models()
        
        # 5. 손글씨 인식 모델들
        self._init_handwriting_models()
        
        # 6. 코드 인식 모델들
        self._init_code_recognition_models()
        
        # 7. 레이아웃 분석 모델
        self._init_layout_analysis_models()
        
        logger.info(f"Universal OCR Pipeline initialized with {len(self.models)} models")
    
    def _init_text_ocr_models(self):
        """텍스트 OCR 모델 초기화"""
        # EasyOCR - 다국어 지원
        try:
            import easyocr
            # 주요 언어들 지원
            self.models['easyocr'] = easyocr.Reader(
                ['ko', 'en', 'ja', 'zh', 'es', 'fr', 'de', 'ru', 'ar'],
                gpu=torch.cuda.is_available()
            )
            logger.info("EasyOCR loaded with multi-language support")
        except Exception as e:
            logger.warning(f"Failed to load EasyOCR: {e}")
        
        # PaddleOCR - 다국어 지원
        try:
            from paddleocr import PaddleOCR
            self.models['paddleocr'] = PaddleOCR(
                use_angle_cls=True,
                lang='ch',  # Supports multiple languages
                use_gpu=torch.cuda.is_available(),
                show_log=False
            )
            logger.info("PaddleOCR loaded")
        except Exception as e:
            logger.warning(f"Failed to load PaddleOCR: {e}")
        
        # TrOCR - Transformer 기반
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.processors['trocr'] = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            self.models['trocr'] = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
            if torch.cuda.is_available():
                self.models['trocr'] = self.models['trocr'].cuda()
            logger.info("TrOCR loaded")
        except Exception as e:
            logger.warning(f"Failed to load TrOCR: {e}")
        
        # Tesseract OCR
        try:
            import pytesseract
            self.models['tesseract'] = pytesseract
            logger.info("Tesseract OCR loaded")
        except Exception as e:
            logger.warning(f"Failed to load Tesseract: {e}")
    
    def _init_formula_ocr_models(self):
        """수식 OCR 모델 초기화"""
        # LaTeX OCR
        try:
            from pix2tex.cli import LatexOCR
            self.models['latex_ocr'] = LatexOCR()
            logger.info("LaTeX OCR loaded")
        except Exception as e:
            logger.warning(f"Failed to load LaTeX OCR: {e}")
        
        # MathPix 스타일 수식 인식 (오픈소스 대안)
        try:
            from nougat import NougatModel
            self.models['nougat'] = NougatModel.from_pretrained("facebook/nougat-base")
            logger.info("Nougat (academic document OCR) loaded")
        except Exception as e:
            logger.warning(f"Failed to load Nougat: {e}")
    
    def _init_table_extraction_models(self):
        """표 추출 모델 초기화"""
        # Table Transformer
        try:
            from transformers import AutoModelForObjectDetection
            self.models['table_transformer'] = AutoModelForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            logger.info("Table Transformer loaded")
        except Exception as e:
            logger.warning(f"Failed to load Table Transformer: {e}")
        
        # PaddleOCR Table
        try:
            from paddleocr import PPStructure
            self.models['paddle_table'] = PPStructure(
                table=True,
                ocr=True,
                show_log=False
            )
            logger.info("PaddleOCR Table loaded")
        except Exception as e:
            logger.warning(f"Failed to load PaddleOCR Table: {e}")
    
    def _init_diagram_analysis_models(self):
        """다이어그램/차트 분석 모델 초기화"""
        # DETR for object detection
        try:
            from transformers import DetrImageProcessor, DetrForObjectDetection
            self.processors['detr'] = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.models['detr'] = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            logger.info("DETR object detection loaded")
        except Exception as e:
            logger.warning(f"Failed to load DETR: {e}")
        
        # ChartQA 모델 (차트 이해)
        try:
            from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
            self.processors['chartqa'] = Pix2StructProcessor.from_pretrained("google/pix2struct-chartqa-base")
            self.models['chartqa'] = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-chartqa-base")
            logger.info("ChartQA model loaded")
        except Exception as e:
            logger.warning(f"Failed to load ChartQA: {e}")
    
    def _init_handwriting_models(self):
        """손글씨 인식 모델 초기화"""
        # TrOCR Handwritten
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.processors['trocr_handwritten'] = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.models['trocr_handwritten'] = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            if torch.cuda.is_available():
                self.models['trocr_handwritten'] = self.models['trocr_handwritten'].cuda()
            logger.info("TrOCR Handwritten loaded")
        except Exception as e:
            logger.warning(f"Failed to load TrOCR Handwritten: {e}")
    
    def _init_code_recognition_models(self):
        """코드 인식 모델 초기화"""
        # CodeBERT 기반 모델
        try:
            from transformers import AutoTokenizer, AutoModel
            self.processors['codebert'] = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.models['codebert'] = AutoModel.from_pretrained("microsoft/codebert-base")
            logger.info("CodeBERT loaded")
        except Exception as e:
            logger.warning(f"Failed to load CodeBERT: {e}")
    
    def _init_layout_analysis_models(self):
        """레이아웃 분석 모델 초기화"""
        # LayoutLM
        try:
            from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
            self.processors['layoutlm'] = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
            self.models['layoutlm'] = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")
            logger.info("LayoutLMv3 loaded")
        except Exception as e:
            logger.warning(f"Failed to load LayoutLM: {e}")
        
        # Detectron2 기반 레이아웃 파서
        try:
            import layoutparser as lp
            self.models['layout_parser'] = lp.Detectron2LayoutModel(
                'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            )
            logger.info("Layout Parser loaded")
        except Exception as e:
            logger.warning(f"Failed to load Layout Parser: {e}")
    
    def process_image(
        self,
        image: Union[Image.Image, str, bytes],
        content_hints: Optional[List[ContentType]] = None,
        languages: Optional[List[str]] = None,
        detail_level: str = "high"
    ) -> Dict[str, Any]:
        """
        이미지에서 모든 정보 추출
        
        Args:
            image: 입력 이미지
            content_hints: 예상되는 콘텐츠 유형 힌트
            languages: 처리할 언어 목록
            detail_level: 상세도 ("low", "medium", "high")
            
        Returns:
            추출된 모든 정보
        """
        # 이미지 전처리
        pil_image = self._preprocess_image(image)
        
        # 결과 초기화
        results = {
            'success': True,
            'layout_analysis': {},
            'text_regions': [],
            'formulas': [],
            'tables': [],
            'diagrams': [],
            'charts': [],
            'handwriting': [],
            'code_blocks': [],
            'metadata': {
                'image_size': pil_image.size,
                'detail_level': detail_level,
                'processed_models': []
            }
        }
        
        # 1. 레이아웃 분석
        if detail_level in ["medium", "high"]:
            layout_results = self._analyze_layout(pil_image)
            results['layout_analysis'] = layout_results
            results['metadata']['processed_models'].append('layout_analysis')
        
        # 2. 콘텐츠 유형별 처리
        if not content_hints:
            # 자동으로 콘텐츠 유형 감지
            content_hints = self._detect_content_types(pil_image, layout_results)
        
        # 3. 각 콘텐츠 유형에 따른 처리
        for content_type in content_hints:
            if content_type == ContentType.TEXT:
                text_results = self._extract_text(pil_image, languages)
                results['text_regions'].extend(text_results)
                
            elif content_type == ContentType.FORMULA:
                formula_results = self._extract_formulas(pil_image)
                results['formulas'].extend(formula_results)
                
            elif content_type == ContentType.TABLE:
                table_results = self._extract_tables(pil_image)
                results['tables'].extend(table_results)
                
            elif content_type == ContentType.DIAGRAM:
                diagram_results = self._analyze_diagrams(pil_image)
                results['diagrams'].extend(diagram_results)
                
            elif content_type == ContentType.CHART:
                chart_results = self._analyze_charts(pil_image)
                results['charts'].extend(chart_results)
                
            elif content_type == ContentType.HANDWRITING:
                handwriting_results = self._extract_handwriting(pil_image)
                results['handwriting'].extend(handwriting_results)
                
            elif content_type == ContentType.CODE:
                code_results = self._extract_code(pil_image)
                results['code_blocks'].extend(code_results)
        
        # 4. 통합 콘텐츠 생성
        results['unified_content'] = self._create_unified_content(results)
        
        # 5. 품질 평가
        results['quality_metrics'] = self._evaluate_extraction_quality(results)
        
        return results
    
    def _preprocess_image(self, image: Union[Image.Image, str, bytes]) -> Image.Image:
        """이미지 전처리"""
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        
        if isinstance(image, str):
            if image.startswith('data:image'):
                # Base64 디코딩
                base64_str = image.split(',')[1]
                image_bytes = base64.b64decode(base64_str)
                return Image.open(io.BytesIO(image_bytes)).convert('RGB')
            else:
                # 파일 경로
                return Image.open(image).convert('RGB')
        
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert('RGB')
        
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _analyze_layout(self, image: Image.Image) -> Dict[str, Any]:
        """레이아웃 분석"""
        layout_info = {
            'regions': [],
            'structure': {},
            'reading_order': []
        }
        
        if 'layout_parser' in self.models:
            try:
                layout = self.models['layout_parser'].detect(image)
                for block in layout:
                    layout_info['regions'].append({
                        'type': block.type,
                        'bbox': block.coordinates,
                        'confidence': block.score
                    })
            except Exception as e:
                logger.error(f"Layout analysis failed: {e}")
        
        return layout_info
    
    def _detect_content_types(self, image: Image.Image, layout_info: Dict[str, Any]) -> List[ContentType]:
        """콘텐츠 유형 자동 감지"""
        detected_types = []
        
        # 레이아웃 정보 기반 감지
        if layout_info.get('regions'):
            for region in layout_info['regions']:
                region_type = region.get('type', '').lower()
                if 'table' in region_type:
                    detected_types.append(ContentType.TABLE)
                elif 'figure' in region_type or 'image' in region_type:
                    detected_types.append(ContentType.DIAGRAM)
                elif 'formula' in region_type or 'equation' in region_type:
                    detected_types.append(ContentType.FORMULA)
        
        # 기본적으로 텍스트는 항상 포함
        if ContentType.TEXT not in detected_types:
            detected_types.append(ContentType.TEXT)
        
        return detected_types
    
    def _extract_text(self, image: Image.Image, languages: Optional[List[str]] = None) -> List[OCRResult]:
        """텍스트 추출"""
        text_results = []
        
        # EasyOCR 사용
        if 'easyocr' in self.models:
            try:
                results = self.models['easyocr'].readtext(np.array(image))
                for bbox, text, confidence in results:
                    text_results.append(OCRResult(
                        content_type=ContentType.TEXT,
                        text=text,
                        confidence=confidence,
                        bbox=self._convert_bbox(bbox),
                        metadata={'model': 'easyocr'}
                    ))
            except Exception as e:
                logger.error(f"EasyOCR failed: {e}")
        
        # PaddleOCR 사용
        if 'paddleocr' in self.models and len(text_results) == 0:
            try:
                results = self.models['paddleocr'].ocr(np.array(image), cls=True)
                if results and results[0]:
                    for line in results[0]:
                        if line[1]:
                            text_results.append(OCRResult(
                                content_type=ContentType.TEXT,
                                text=line[1][0],
                                confidence=line[1][1],
                                bbox=self._convert_bbox(line[0]),
                                metadata={'model': 'paddleocr'}
                            ))
            except Exception as e:
                logger.error(f"PaddleOCR failed: {e}")
        
        return text_results
    
    def _extract_formulas(self, image: Image.Image) -> List[OCRResult]:
        """수식 추출"""
        formula_results = []
        
        # LaTeX OCR
        if 'latex_ocr' in self.models:
            try:
                latex_text = self.models['latex_ocr'](image)
                if latex_text:
                    formula_results.append(OCRResult(
                        content_type=ContentType.FORMULA,
                        text=latex_text,
                        confidence=0.8,
                        metadata={'model': 'latex_ocr', 'format': 'latex'}
                    ))
            except Exception as e:
                logger.error(f"LaTeX OCR failed: {e}")
        
        # Nougat for academic formulas
        if 'nougat' in self.models and len(formula_results) == 0:
            try:
                # Nougat 처리 로직
                pass
            except Exception as e:
                logger.error(f"Nougat failed: {e}")
        
        return formula_results
    
    def _extract_tables(self, image: Image.Image) -> List[Dict[str, Any]]:
        """표 추출"""
        table_results = []
        
        # PaddleOCR Table
        if 'paddle_table' in self.models:
            try:
                results = self.models['paddle_table'](np.array(image))
                for result in results:
                    if result['type'] == 'table':
                        table_results.append({
                            'type': 'table',
                            'html': result.get('res', {}).get('html', ''),
                            'cells': result.get('res', {}).get('cells', []),
                            'confidence': result.get('score', 0.0),
                            'metadata': {'model': 'paddle_table'}
                        })
            except Exception as e:
                logger.error(f"Table extraction failed: {e}")
        
        return table_results
    
    def _analyze_diagrams(self, image: Image.Image) -> List[Dict[str, Any]]:
        """다이어그램 분석"""
        diagram_results = []
        
        # DETR object detection
        if 'detr' in self.models and 'detr' in self.processors:
            try:
                inputs = self.processors['detr'](images=image, return_tensors="pt")
                outputs = self.models['detr'](**inputs)
                
                # 객체 감지 결과 처리
                target_sizes = torch.tensor([image.size[::-1]])
                results = self.processors['detr'].post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=0.7
                )[0]
                
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    diagram_results.append({
                        'type': 'object',
                        'label': label.item(),
                        'bbox': box.tolist(),
                        'confidence': score.item(),
                        'metadata': {'model': 'detr'}
                    })
            except Exception as e:
                logger.error(f"Diagram analysis failed: {e}")
        
        return diagram_results
    
    def _analyze_charts(self, image: Image.Image) -> List[Dict[str, Any]]:
        """차트 분석"""
        chart_results = []
        
        # ChartQA
        if 'chartqa' in self.models and 'chartqa' in self.processors:
            try:
                # 차트 분석 질문들
                questions = [
                    "What is the title of this chart?",
                    "What are the axis labels?",
                    "What is the maximum value?",
                    "What is the trend?"
                ]
                
                for question in questions:
                    inputs = self.processors['chartqa'](
                        images=image,
                        text=question,
                        return_tensors="pt"
                    )
                    predictions = self.models['chartqa'].generate(**inputs)
                    answer = self.processors['chartqa'].decode(predictions[0], skip_special_tokens=True)
                    
                    chart_results.append({
                        'question': question,
                        'answer': answer,
                        'metadata': {'model': 'chartqa'}
                    })
            except Exception as e:
                logger.error(f"Chart analysis failed: {e}")
        
        return chart_results
    
    def _extract_handwriting(self, image: Image.Image) -> List[OCRResult]:
        """손글씨 추출"""
        handwriting_results = []
        
        # TrOCR Handwritten
        if 'trocr_handwritten' in self.models and 'trocr_handwritten' in self.processors:
            try:
                pixel_values = self.processors['trocr_handwritten'](
                    images=image,
                    return_tensors="pt"
                ).pixel_values
                
                if torch.cuda.is_available():
                    pixel_values = pixel_values.cuda()
                
                generated_ids = self.models['trocr_handwritten'].generate(pixel_values)
                generated_text = self.processors['trocr_handwritten'].batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
                
                handwriting_results.append(OCRResult(
                    content_type=ContentType.HANDWRITING,
                    text=generated_text,
                    confidence=0.75,
                    metadata={'model': 'trocr_handwritten'}
                ))
            except Exception as e:
                logger.error(f"Handwriting extraction failed: {e}")
        
        return handwriting_results
    
    def _extract_code(self, image: Image.Image) -> List[Dict[str, Any]]:
        """코드 추출"""
        code_results = []
        
        # 먼저 텍스트 추출 후 코드 패턴 검사
        text_results = self._extract_text(image)
        
        # 코드 패턴 감지
        code_patterns = [
            r'^\s*(import|from|def|class|function|var|let|const)\s+',
            r'^\s*(if|for|while|switch)\s*\(',
            r'[{};]$',
            r'^\s*#.*$',  # Comments
            r'^\s*//.*$',  # Comments
        ]
        
        for result in text_results:
            text = result.text
            is_code = any(re.search(pattern, text, re.MULTILINE) for pattern in code_patterns)
            
            if is_code:
                code_results.append({
                    'type': 'code',
                    'text': text,
                    'language': self._detect_programming_language(text),
                    'confidence': result.confidence,
                    'metadata': {'source': result.metadata}
                })
        
        return code_results
    
    def _detect_programming_language(self, code: str) -> str:
        """프로그래밍 언어 감지"""
        # 간단한 휴리스틱 기반 언어 감지
        if 'import' in code or 'def' in code:
            return 'python'
        elif 'function' in code or 'var' in code or 'let' in code:
            return 'javascript'
        elif '#include' in code:
            return 'c++'
        elif 'public class' in code:
            return 'java'
        else:
            return 'unknown'
    
    def _convert_bbox(self, bbox: Any) -> Tuple[int, int, int, int]:
        """바운딩 박스 형식 변환"""
        if isinstance(bbox, list) and len(bbox) >= 4:
            if len(bbox) == 4:
                return tuple(bbox)
            else:
                # 폴리곤 형태인 경우
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        return (0, 0, 0, 0)
    
    def _create_unified_content(self, results: Dict[str, Any]) -> str:
        """통합 콘텐츠 생성"""
        content_parts = []
        
        # 텍스트 영역
        if results['text_regions']:
            text_content = ' '.join([r.text for r in results['text_regions']])
            content_parts.append(f"텍스트:\n{text_content}\n")
        
        # 수식
        if results['formulas']:
            content_parts.append("수식:")
            for i, formula in enumerate(results['formulas']):
                content_parts.append(f"  [{i+1}] {formula.text}")
            content_parts.append("")
        
        # 표
        if results['tables']:
            content_parts.append(f"표 {len(results['tables'])}개 발견됨\n")
        
        # 다이어그램
        if results['diagrams']:
            content_parts.append(f"다이어그램 {len(results['diagrams'])}개 발견됨\n")
        
        # 차트
        if results['charts']:
            content_parts.append("차트 분석:")
            for chart in results['charts']:
                content_parts.append(f"  Q: {chart['question']}")
                content_parts.append(f"  A: {chart['answer']}")
            content_parts.append("")
        
        # 손글씨
        if results['handwriting']:
            handwriting_content = ' '.join([r.text for r in results['handwriting']])
            content_parts.append(f"손글씨:\n{handwriting_content}\n")
        
        # 코드
        if results['code_blocks']:
            content_parts.append("코드:")
            for code in results['code_blocks']:
                content_parts.append(f"  언어: {code['language']}")
                content_parts.append(f"  내용: {code['text'][:100]}...")
            content_parts.append("")
        
        return '\n'.join(content_parts)
    
    def _evaluate_extraction_quality(self, results: Dict[str, Any]) -> Dict[str, float]:
        """추출 품질 평가"""
        metrics = {
            'coverage': 0.0,  # 얼마나 많은 영역을 커버했는가
            'confidence': 0.0,  # 평균 신뢰도
            'completeness': 0.0,  # 완전성
            'diversity': 0.0  # 다양성 (여러 유형의 콘텐츠)
        }
        
        # 신뢰도 계산
        all_confidences = []
        
        for region in results['text_regions']:
            all_confidences.append(region.confidence)
        
        for formula in results['formulas']:
            all_confidences.append(formula.confidence)
        
        for handwriting in results['handwriting']:
            all_confidences.append(handwriting.confidence)
        
        if all_confidences:
            metrics['confidence'] = sum(all_confidences) / len(all_confidences)
        
        # 다양성 계산
        content_types_found = 0
        if results['text_regions']:
            content_types_found += 1
        if results['formulas']:
            content_types_found += 1
        if results['tables']:
            content_types_found += 1
        if results['diagrams']:
            content_types_found += 1
        if results['charts']:
            content_types_found += 1
        if results['handwriting']:
            content_types_found += 1
        if results['code_blocks']:
            content_types_found += 1
        
        metrics['diversity'] = content_types_found / 7.0
        
        # 완전성 (간단한 휴리스틱)
        if results['unified_content']:
            content_length = len(results['unified_content'])
            metrics['completeness'] = min(1.0, content_length / 1000)
        
        # 커버리지 (레이아웃 분석 기반)
        if results['layout_analysis'].get('regions'):
            metrics['coverage'] = min(1.0, len(results['layout_analysis']['regions']) / 10)
        
        return metrics


class DomainAdaptiveOCR(UniversalOCRPipeline):
    """도메인 적응형 OCR - 특정 도메인에 맞게 자동 조정"""
    
    def __init__(self, domain: Optional[str] = None):
        """
        도메인 적응형 OCR 초기화
        
        Args:
            domain: 특정 도메인 (예: "medical", "legal", "engineering", "academic")
        """
        super().__init__()
        self.domain = domain
        self.domain_configs = self._load_domain_configs()
        
        if domain:
            self._adapt_to_domain(domain)
    
    def _load_domain_configs(self) -> Dict[str, Dict[str, Any]]:
        """도메인별 설정 로드"""
        return {
            'medical': {
                'priority_models': ['handwriting', 'formula'],
                'terminology': ['diagnosis', 'prescription', 'symptom'],
                'layout_hints': ['patient_record', 'lab_report']
            },
            'legal': {
                'priority_models': ['text', 'table'],
                'terminology': ['contract', 'clause', 'provision'],
                'layout_hints': ['contract', 'legal_document']
            },
            'engineering': {
                'priority_models': ['formula', 'diagram', 'table'],
                'terminology': ['voltage', 'current', 'resistance'],
                'layout_hints': ['schematic', 'blueprint']
            },
            'academic': {
                'priority_models': ['text', 'formula', 'chart'],
                'terminology': ['theorem', 'proof', 'hypothesis'],
                'layout_hints': ['paper', 'thesis']
            }
        }
    
    def _adapt_to_domain(self, domain: str):
        """특정 도메인에 맞게 파이프라인 조정"""
        if domain not in self.domain_configs:
            logger.warning(f"Unknown domain: {domain}, using generic configuration")
            return
        
        config = self.domain_configs[domain]
        
        # 도메인별 우선순위 설정
        self.priority_models = config.get('priority_models', [])
        self.domain_terminology = config.get('terminology', [])
        self.layout_hints = config.get('layout_hints', [])
        
        logger.info(f"Adapted OCR pipeline for {domain} domain")
    
    def auto_detect_domain(self, image: Image.Image) -> str:
        """이미지에서 도메인 자동 감지"""
        # 간단한 텍스트 샘플링으로 도메인 추측
        sample_text = self._extract_text(image, languages=['en', 'ko'])[:5]
        
        if not sample_text:
            return 'generic'
        
        text_content = ' '.join([r.text for r in sample_text]).lower()
        
        # 도메인별 키워드 매칭
        domain_scores = {}
        
        for domain, config in self.domain_configs.items():
            score = 0
            for term in config['terminology']:
                if term.lower() in text_content:
                    score += 1
            domain_scores[domain] = score
        
        # 가장 높은 점수의 도메인 선택
        if domain_scores:
            detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            if domain_scores[detected_domain] > 0:
                return detected_domain
        
        return 'generic'
    
    def process_adaptive(
        self,
        image: Union[Image.Image, str, bytes],
        auto_detect: bool = True
    ) -> Dict[str, Any]:
        """도메인 적응형 처리"""
        pil_image = self._preprocess_image(image)
        
        # 도메인 자동 감지
        if auto_detect and not self.domain:
            detected_domain = self.auto_detect_domain(pil_image)
            logger.info(f"Auto-detected domain: {detected_domain}")
            self._adapt_to_domain(detected_domain)
        
        # 도메인에 맞는 콘텐츠 힌트 생성
        content_hints = self._get_domain_content_hints()
        
        # 처리
        results = self.process_image(
            pil_image,
            content_hints=content_hints,
            detail_level="high" if self.domain else "medium"
        )
        
        # 도메인 정보 추가
        results['domain_info'] = {
            'detected_domain': self.domain or 'generic',
            'domain_confidence': self._calculate_domain_confidence(results)
        }
        
        return results
    
    def _get_domain_content_hints(self) -> List[ContentType]:
        """도메인별 콘텐츠 힌트 생성"""
        if not self.domain or self.domain not in self.domain_configs:
            return None
        
        priority_models = self.domain_configs[self.domain].get('priority_models', [])
        
        content_hints = []
        model_to_content = {
            'text': ContentType.TEXT,
            'formula': ContentType.FORMULA,
            'table': ContentType.TABLE,
            'diagram': ContentType.DIAGRAM,
            'chart': ContentType.CHART,
            'handwriting': ContentType.HANDWRITING,
            'code': ContentType.CODE
        }
        
        for model in priority_models:
            if model in model_to_content:
                content_hints.append(model_to_content[model])
        
        return content_hints
    
    def _calculate_domain_confidence(self, results: Dict[str, Any]) -> float:
        """도메인 신뢰도 계산"""
        if not self.domain or self.domain not in self.domain_configs:
            return 0.0
        
        config = self.domain_configs[self.domain]
        terminology = config.get('terminology', [])
        
        # 통합 콘텐츠에서 도메인 용어 검색
        unified_content = results.get('unified_content', '').lower()
        
        term_count = 0
        for term in terminology:
            term_count += unified_content.count(term.lower())
        
        # 간단한 신뢰도 계산
        confidence = min(1.0, term_count / 10.0)
        
        return confidence


# 사용 예시
if __name__ == "__main__":
    # 범용 OCR 파이프라인
    universal_ocr = UniversalOCRPipeline()
    
    # 도메인 적응형 OCR
    adaptive_ocr = DomainAdaptiveOCR()
    
    # 특정 도메인 OCR
    engineering_ocr = DomainAdaptiveOCR(domain="engineering")
    
    print("Universal OCR Pipeline ready for all domains")