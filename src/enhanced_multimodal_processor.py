#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Multimodal Processor
향상된 멀티모달 처리 시스템 - 이미지, 텍스트, 수식 통합 처리
"""

import logging
from typing import Dict, List, Optional, Union, Any
from PIL import Image
import numpy as np
import torch
from dataclasses import dataclass

# 컴포넌트 임포트
from multi_engine_ocr import MultiEngineOCR
from math_formula_detector import MathFormulaProcessor, RecognizedFormula

# 기존 이미지 분석기 (폴백용)
try:
    from simple_image_analyzer import Florence2ImageAnalyzer
    FLORENCE_AVAILABLE = True
except ImportError:
    FLORENCE_AVAILABLE = False
    logging.warning("Florence2 not available")

logger = logging.getLogger(__name__)


@dataclass
class ProcessedRegion:
    """처리된 영역 정보"""
    type: str  # 'text', 'formula', 'table', 'figure'
    content: Any
    bbox: Optional[tuple] = None
    confidence: float = 0.0


@dataclass
class MultimodalResult:
    """멀티모달 처리 결과"""
    text_content: str
    formula_content: List[RecognizedFormula]
    image_caption: Optional[str]
    combined_context: str
    metadata: Dict[str, Any]
    processing_time: float


class EnhancedMultimodalProcessor:
    """향상된 멀티모달 처리기"""
    
    def __init__(self, use_gpu: bool = True):
        """
        초기화
        
        Args:
            use_gpu: GPU 사용 여부
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # OCR 엔진 초기화
        try:
            self.ocr_engine = MultiEngineOCR()
            logger.info("MultiEngineOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR engine: {e}")
            self.ocr_engine = None
        
        # 수식 처리기 초기화
        try:
            self.formula_processor = MathFormulaProcessor()
            logger.info("MathFormulaProcessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize formula processor: {e}")
            self.formula_processor = None
        
        # 이미지 분석기 초기화 (선택적)
        self.image_analyzer = None
        if FLORENCE_AVAILABLE:
            try:
                self.image_analyzer = Florence2ImageAnalyzer()
                logger.info("Florence2 image analyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Florence2: {e}")
        
        # GPU 메모리 최적화
        if self.use_gpu:
            self._optimize_gpu_memory()
    
    def _optimize_gpu_memory(self):
        """GPU 메모리 최적화"""
        if torch.cuda.is_available():
            # 캐시 정리
            torch.cuda.empty_cache()
            
            # 메모리 할당 설정
            torch.cuda.set_per_process_memory_fraction(0.7)  # 70% 제한
            
            logger.info(f"GPU memory optimized. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def process_image(
        self,
        image: Union[Image.Image, np.ndarray, str],
        question: Optional[str] = None,
        extract_text: bool = True,
        detect_formulas: bool = True,
        generate_caption: bool = True
    ) -> MultimodalResult:
        """
        이미지 종합 처리
        
        Args:
            image: 입력 이미지
            question: 관련 질문 (선택적)
            extract_text: 텍스트 추출 여부
            detect_formulas: 수식 감지 여부
            generate_caption: 캡션 생성 여부
            
        Returns:
            멀티모달 처리 결과
        """
        import time
        start_time = time.time()
        
        # 이미지 로드
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 이미지 크기 최적화
        image = self._resize_image(image)
        
        # 결과 초기화
        text_content = ""
        formula_content = []
        image_caption = None
        metadata = {
            'image_size': image.size,
            'processing_steps': []
        }
        
        # 1. 텍스트 추출
        if extract_text and self.ocr_engine:
            try:
                logger.info("Extracting text with MultiEngineOCR...")
                ocr_result = self.ocr_engine.extract_text(image, ensemble_method='voting')
                text_content = ocr_result.get('text', '')
                metadata['ocr_confidence'] = ocr_result.get('confidence', 0)
                metadata['ocr_engines_used'] = ocr_result.get('engines_used', [])
                metadata['processing_steps'].append('text_extraction')
                logger.info(f"Text extracted: {len(text_content)} characters")
            except Exception as e:
                logger.error(f"Text extraction failed: {e}")
                metadata['ocr_error'] = str(e)
        
        # 2. 수식 감지 및 인식
        if detect_formulas and self.formula_processor:
            try:
                logger.info("Detecting and recognizing formulas...")
                formula_content = self.formula_processor.process_image(image)
                metadata['formula_count'] = len(formula_content)
                metadata['processing_steps'].append('formula_detection')
                logger.info(f"Formulas detected: {len(formula_content)}")
            except Exception as e:
                logger.error(f"Formula detection failed: {e}")
                metadata['formula_error'] = str(e)
        
        # 3. 이미지 캡션 생성
        if generate_caption and self.image_analyzer:
            try:
                logger.info("Generating image caption...")
                caption_result = self.image_analyzer.analyze_image(image)
                image_caption = caption_result.get('result', '')
                metadata['caption_confidence'] = caption_result.get('success', False)
                metadata['processing_steps'].append('caption_generation')
            except Exception as e:
                logger.error(f"Caption generation failed: {e}")
                metadata['caption_error'] = str(e)
        
        # 4. 컨텍스트 통합
        combined_context = self._combine_context(
            text_content,
            formula_content,
            image_caption,
            question
        )
        
        # 처리 시간 기록
        processing_time = time.time() - start_time
        metadata['processing_time'] = processing_time
        
        return MultimodalResult(
            text_content=text_content,
            formula_content=formula_content,
            image_caption=image_caption,
            combined_context=combined_context,
            metadata=metadata,
            processing_time=processing_time
        )
    
    def _resize_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """이미지 크기 최적화"""
        width, height = image.size
        
        if width > max_size or height > max_size:
            ratio = min(max_size/width, max_size/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
            return image.resize((new_width, new_height), Image.LANCZOS)
        
        return image
    
    def _combine_context(
        self,
        text_content: str,
        formula_content: List[RecognizedFormula],
        image_caption: Optional[str],
        question: Optional[str]
    ) -> str:
        """모든 정보를 통합하여 컨텍스트 생성"""
        context_parts = []
        
        # 질문이 있으면 먼저 추가
        if question:
            context_parts.append(f"질문: {question}\n")
        
        # 이미지 캡션
        if image_caption:
            context_parts.append(f"이미지 설명: {image_caption}\n")
        
        # OCR 텍스트
        if text_content:
            context_parts.append(f"추출된 텍스트:\n{text_content}\n")
        
        # 수식 정보
        if formula_content:
            formula_text = "\n감지된 수식:\n"
            for i, formula in enumerate(formula_content, 1):
                formula_text += f"{i}. {formula.latex}"
                if formula.solution and formula.solution.get('result'):
                    formula_text += f" = {formula.solution['result']}"
                formula_text += "\n"
            context_parts.append(formula_text)
        
        # 통합
        combined = "\n".join(context_parts)
        
        # 컨텍스트가 너무 길면 요약
        if len(combined) > 2000:
            combined = self._summarize_context(combined)
        
        return combined
    
    def _summarize_context(self, context: str, max_length: int = 1500) -> str:
        """긴 컨텍스트 요약"""
        if len(context) <= max_length:
            return context
        
        # 간단한 요약: 처음과 끝 부분 유지
        half_length = max_length // 2
        summary = context[:half_length] + "\n...[중략]...\n" + context[-half_length:]
        
        return summary
    
    def process_multimodal_query(
        self,
        question: str,
        image: Optional[Union[Image.Image, np.ndarray, str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        멀티모달 쿼리 처리 (ChatService와의 인터페이스)
        
        Args:
            question: 사용자 질문
            image: 이미지 (선택적)
            context: 추가 컨텍스트
            
        Returns:
            처리 결과
        """
        result = {
            'combined_query': question,
            'image_analysis': {},
            'requires_llm': True
        }
        
        if image is None:
            return result
        
        try:
            # 이미지 처리
            multimodal_result = self.process_image(
                image,
                question=question,
                extract_text=True,
                detect_formulas=True,
                generate_caption=True
            )
            
            # 결과 매핑
            result['image_analysis'] = {
                'caption': multimodal_result.image_caption or "[캡션 생성 실패]",
                'ocr_text': multimodal_result.text_content,
                'formulas': [f.latex for f in multimodal_result.formula_content],
                'confidence': multimodal_result.metadata.get('ocr_confidence', 0)
            }
            
            # 질문 확장
            if multimodal_result.combined_context:
                result['combined_query'] = f"{question}\n\n[이미지 정보]\n{multimodal_result.combined_context}"
            
            # 수식이 감지되었고 해결된 경우
            if multimodal_result.formula_content:
                solved_formulas = [f for f in multimodal_result.formula_content if f.solution and f.solution.get('result')]
                if solved_formulas:
                    result['formula_solutions'] = {
                        f.latex: f.solution['result'] for f in solved_formulas
                    }
            
        except Exception as e:
            logger.error(f"Multimodal processing failed: {e}")
            result['image_analysis']['error'] = str(e)
            result['image_analysis']['caption'] = "[이미지 분석 실패]"
        
        return result


# 간단한 사용을 위한 래퍼 함수
def create_multimodal_processor(**kwargs) -> EnhancedMultimodalProcessor:
    """멀티모달 프로세서 생성"""
    return EnhancedMultimodalProcessor(**kwargs)