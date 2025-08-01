#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Korean-Specialized OCR Pipeline
한국어 특화 OCR 파이프라인
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image
import numpy as np
import torch
import io
import base64
import re

logger = logging.getLogger(__name__)


class KoreanOCRPipeline:
    """한국어 특화 OCR 파이프라인"""
    
    def __init__(self):
        """한국어 OCR 파이프라인 초기화"""
        self.models_loaded = {}
        
        # 1. EasyOCR (Primary Korean OCR)
        try:
            import easyocr
            self.easy_ocr = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
            self.models_loaded['easyocr'] = True
            logger.info("EasyOCR loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load EasyOCR: {e}")
            self.easy_ocr = None
            self.models_loaded['easyocr'] = False
        
        # 2. TrOCR Korean (Transformer-based)
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            # Using multilingual TrOCR that supports Korean
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
            if torch.cuda.is_available():
                self.trocr_model = self.trocr_model.cuda()
            self.models_loaded['trocr'] = True
            logger.info("TrOCR loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load TrOCR: {e}")
            self.trocr_processor = None
            self.trocr_model = None
            self.models_loaded['trocr'] = False
        
        # 3. PaddleOCR (Already in multimodal_ocr.py, but add Korean support)
        try:
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='korean',  # Korean language support
                use_gpu=torch.cuda.is_available(),
                show_log=False
            )
            self.models_loaded['paddleocr'] = True
            logger.info("PaddleOCR Korean loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load PaddleOCR: {e}")
            self.paddle_ocr = None
            self.models_loaded['paddleocr'] = False
        
        # 4. Pororo OCR (Korean NLP Library)
        try:
            from pororo import Pororo
            self.pororo_ocr = Pororo(task="ocr", lang="ko")
            self.models_loaded['pororo'] = True
            logger.info("Pororo OCR loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Pororo OCR: {e}")
            self.pororo_ocr = None
            self.models_loaded['pororo'] = False
        
        # 5. LaTeX OCR for mathematical formulas (from multimodal_ocr)
        try:
            from pix2tex.cli import LatexOCR
            self.latex_ocr = LatexOCR()
            self.models_loaded['latex_ocr'] = True
            logger.info("LaTeX OCR loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load LaTeX OCR: {e}")
            self.latex_ocr = None
            self.models_loaded['latex_ocr'] = False
        
        logger.info(f"Korean OCR Pipeline initialized. Models loaded: {self.models_loaded}")
    
    def process_image(self, image: Union[Image.Image, str, bytes]) -> Dict[str, Any]:
        """
        이미지에서 한국어 텍스트 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            추출된 텍스트와 메타데이터
        """
        # 이미지 전처리
        pil_image = self._preprocess_image(image)
        
        results = {
            'success': True,
            'text_content': '',
            'formulas': [],
            'structured_content': '',
            'ocr_results': {},
            'confidence_scores': {},
            'layout_regions': []
        }
        
        # 1. EasyOCR로 한국어 텍스트 추출
        if self.easy_ocr:
            try:
                easy_results = self._extract_with_easyocr(pil_image)
                results['ocr_results']['easyocr'] = easy_results
                results['text_content'] = easy_results.get('text', '')
                results['confidence_scores']['easyocr'] = easy_results.get('confidence', 0.0)
            except Exception as e:
                logger.error(f"EasyOCR failed: {e}")
        
        # 2. PaddleOCR로 보완
        if self.paddle_ocr:
            try:
                paddle_results = self._extract_with_paddleocr(pil_image)
                results['ocr_results']['paddleocr'] = paddle_results
                
                # 더 나은 결과 선택
                if paddle_results.get('confidence', 0) > results['confidence_scores'].get('easyocr', 0):
                    results['text_content'] = paddle_results.get('text', '')
                    results['confidence_scores']['paddleocr'] = paddle_results.get('confidence', 0.0)
            except Exception as e:
                logger.error(f"PaddleOCR failed: {e}")
        
        # 3. 수식 추출 (LaTeX OCR)
        if self.latex_ocr:
            try:
                formula_results = self._extract_formulas(pil_image)
                results['formulas'] = formula_results
            except Exception as e:
                logger.error(f"LaTeX OCR failed: {e}")
        
        # 4. 구조화된 콘텐츠 생성
        results['structured_content'] = self._create_structured_content(results)
        
        # 5. 전기공학 특화 후처리
        results = self._postprocess_electrical_engineering(results)
        
        return results
    
    def _preprocess_image(self, image: Union[Image.Image, str, bytes]) -> Image.Image:
        """이미지 전처리"""
        if isinstance(image, Image.Image):
            return image
        
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
    
    def _extract_with_easyocr(self, image: Image.Image) -> Dict[str, Any]:
        """EasyOCR로 텍스트 추출"""
        # PIL Image를 numpy array로 변환
        img_array = np.array(image)
        
        # OCR 수행
        results = self.easy_ocr.readtext(img_array)
        
        # 결과 정리
        all_text = []
        total_confidence = 0
        
        for (bbox, text, confidence) in results:
            all_text.append(text)
            total_confidence += confidence
        
        avg_confidence = total_confidence / len(results) if results else 0
        
        return {
            'text': ' '.join(all_text),
            'confidence': avg_confidence,
            'raw_results': results
        }
    
    def _extract_with_paddleocr(self, image: Image.Image) -> Dict[str, Any]:
        """PaddleOCR로 텍스트 추출"""
        # PIL Image를 numpy array로 변환
        img_array = np.array(image)
        
        # OCR 수행
        results = self.paddle_ocr.ocr(img_array, cls=True)
        
        # 결과 정리
        all_text = []
        total_confidence = 0
        count = 0
        
        if results and results[0]:
            for line in results[0]:
                if line[1]:
                    text = line[1][0]
                    confidence = line[1][1]
                    all_text.append(text)
                    total_confidence += confidence
                    count += 1
        
        avg_confidence = total_confidence / count if count > 0 else 0
        
        return {
            'text': ' '.join(all_text),
            'confidence': avg_confidence,
            'raw_results': results
        }
    
    def _extract_formulas(self, image: Image.Image) -> List[Dict[str, Any]]:
        """수식 추출"""
        formulas = []
        
        try:
            # 전체 이미지에서 수식 추출 시도
            latex_result = self.latex_ocr(image)
            if latex_result:
                formulas.append({
                    'latex': latex_result,
                    'type': 'full_image',
                    'confidence': 0.8
                })
        except Exception as e:
            logger.warning(f"Formula extraction failed: {e}")
        
        return formulas
    
    def _create_structured_content(self, results: Dict[str, Any]) -> str:
        """구조화된 콘텐츠 생성"""
        content = []
        
        # 텍스트 콘텐츠
        if results['text_content']:
            content.append(f"텍스트:\n{results['text_content']}\n")
        
        # 수식
        if results['formulas']:
            content.append("수식:")
            for i, formula in enumerate(results['formulas']):
                content.append(f"  [{i+1}] {formula['latex']}")
            content.append("")
        
        return '\n'.join(content)
    
    def _postprocess_electrical_engineering(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """전기공학 특화 후처리"""
        text = results['text_content']
        
        # 전기공학 단위 정규화
        unit_mappings = {
            r'\bkw\b': 'kW',
            r'\bmw\b': 'MW',
            r'\bkv\b': 'kV',
            r'\bkva\b': 'kVA',
            r'\bkvar\b': 'kVar',
            r'\bma\b': 'mA',
            r'\bω\b': 'Ω',
            r'\bmω\b': 'MΩ',
            r'\bkω\b': 'kΩ',
        }
        
        for pattern, replacement in unit_mappings.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 전기공학 기호 인식 개선
        symbol_mappings = {
            '파이': 'π',
            '세타': 'θ',
            '알파': 'α',
            '베타': 'β',
            '감마': 'γ',
            '델타': 'Δ',
            '오메가': 'Ω',
            '루트': '√',
        }
        
        for korean, symbol in symbol_mappings.items():
            text = text.replace(korean, symbol)
        
        # 수식 패턴 인식
        formula_patterns = [
            r'V\s*=\s*I\s*[×*]\s*R',  # 옴의 법칙
            r'P\s*=\s*V\s*[×*]\s*I\s*[×*]\s*cos\s*θ',  # 교류 전력
            r'P\s*=\s*√3\s*[×*]\s*V[Ll]\s*[×*]\s*I[Ll]\s*[×*]\s*cos\s*θ',  # 3상 전력
            r'Z\s*=\s*√\(R²\s*\+\s*X²\)',  # 임피던스
        ]
        
        detected_formulas = []
        for pattern in formula_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected_formulas.append(pattern)
        
        # 결과 업데이트
        results['text_content'] = text
        results['detected_formulas'] = detected_formulas
        
        # 전기공학 키워드 추출
        ee_keywords = []
        keyword_list = [
            '전압', '전류', '저항', '전력', '임피던스', '역률',
            '변압기', '모터', '전동기', '회로', '콘덴서', '인덕터',
            '3상', '단상', 'RLC', 'PWM', '인버터', '정류기',
            '무효전력', '유효전력', '피상전력', '선간전압', '상전압'
        ]
        
        for keyword in keyword_list:
            if keyword in text:
                ee_keywords.append(keyword)
        
        results['electrical_keywords'] = ee_keywords
        
        return results
    
    def compare_ocr_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """여러 OCR 결과 비교 및 최적 결과 선택"""
        comparison = {
            'best_method': None,
            'best_confidence': 0,
            'method_scores': {}
        }
        
        # 각 방법의 신뢰도 비교
        for method, confidence in results['confidence_scores'].items():
            comparison['method_scores'][method] = confidence
            if confidence > comparison['best_confidence']:
                comparison['best_confidence'] = confidence
                comparison['best_method'] = method
        
        return comparison


class KoreanElectricalOCR(KoreanOCRPipeline):
    """전기공학 특화 한국어 OCR"""
    
    def __init__(self):
        """전기공학 OCR 초기화"""
        super().__init__()
        
        # 전기공학 전문 용어 사전
        self.electrical_terms = {
            'voltage': ['전압', 'V', 'kV', 'voltage'],
            'current': ['전류', 'I', 'A', 'mA', 'current'],
            'resistance': ['저항', 'R', 'Ω', 'kΩ', 'MΩ', 'resistance'],
            'power': ['전력', 'P', 'W', 'kW', 'MW', 'power'],
            'impedance': ['임피던스', 'Z', 'impedance'],
            'power_factor': ['역률', 'cosθ', 'PF', 'power factor'],
            'reactive_power': ['무효전력', 'Q', 'kVar', 'reactive power'],
            'apparent_power': ['피상전력', 'S', 'kVA', 'apparent power'],
        }
    
    def extract_electrical_data(self, image: Union[Image.Image, str, bytes]) -> Dict[str, Any]:
        """전기공학 데이터 추출"""
        # 기본 OCR 수행
        results = self.process_image(image)
        
        # 전기공학 특화 분석
        electrical_data = {
            'values': self._extract_electrical_values(results['text_content']),
            'formulas': self._identify_electrical_formulas(results),
            'circuit_elements': self._identify_circuit_elements(results['text_content']),
            'problem_type': self._classify_problem_type(results)
        }
        
        results['electrical_analysis'] = electrical_data
        
        return results
    
    def _extract_electrical_values(self, text: str) -> List[Dict[str, Any]]:
        """전기공학 수치 추출"""
        values = []
        
        # 전기공학 단위 패턴
        patterns = [
            (r'(\d+(?:\.\d+)?)\s*(kW|MW|W)', 'power'),
            (r'(\d+(?:\.\d+)?)\s*(kV|V|mV)', 'voltage'),
            (r'(\d+(?:\.\d+)?)\s*(kA|A|mA)', 'current'),
            (r'(\d+(?:\.\d+)?)\s*(kΩ|Ω|MΩ)', 'resistance'),
            (r'(\d+(?:\.\d+)?)\s*(kVA|MVA)', 'apparent_power'),
            (r'(\d+(?:\.\d+)?)\s*(kVar|MVar)', 'reactive_power'),
            (r'(\d+(?:\.\d+)?)\s*(%)', 'percentage'),
            (r'cos\s*θ\s*=\s*(\d+(?:\.\d+)?)', 'power_factor'),
        ]
        
        for pattern, value_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                values.append({
                    'value': float(match.group(1)),
                    'unit': match.group(2) if len(match.groups()) > 1 else '',
                    'type': value_type,
                    'text': match.group(0)
                })
        
        return values
    
    def _identify_electrical_formulas(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """전기공학 공식 식별"""
        formulas = []
        
        # 기존 수식 결과 활용
        for formula in results.get('formulas', []):
            formula_type = self._classify_formula(formula['latex'])
            formulas.append({
                'latex': formula['latex'],
                'type': formula_type,
                'description': self._get_formula_description(formula_type)
            })
        
        # 텍스트에서 공식 패턴 찾기
        text = results['text_content']
        formula_patterns = {
            'ohms_law': r'V\s*=\s*I\s*[×*]\s*R',
            'ac_power': r'P\s*=\s*V\s*[×*]\s*I\s*[×*]\s*cos\s*θ',
            'three_phase_power': r'P\s*=\s*√3\s*[×*]\s*V[Ll]\s*[×*]\s*I[Ll]\s*[×*]\s*cos\s*θ',
            'impedance': r'Z\s*=\s*√\(R²\s*\+\s*X²\)',
            'power_triangle': r'S²\s*=\s*P²\s*\+\s*Q²',
        }
        
        for formula_type, pattern in formula_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                formulas.append({
                    'type': formula_type,
                    'pattern': pattern,
                    'description': self._get_formula_description(formula_type)
                })
        
        return formulas
    
    def _identify_circuit_elements(self, text: str) -> List[str]:
        """회로 요소 식별"""
        elements = []
        
        element_keywords = [
            '저항', '콘덴서', '커패시터', '인덕터', '코일',
            '변압기', '트랜스포머', '다이오드', '트랜지스터',
            'MOSFET', 'IGBT', '릴레이', '스위치', '퓨즈',
            '전원', '부하', '모터', '발전기'
        ]
        
        for element in element_keywords:
            if element in text:
                elements.append(element)
        
        return elements
    
    def _classify_problem_type(self, results: Dict[str, Any]) -> str:
        """문제 유형 분류"""
        text = results['text_content'].lower()
        
        if any(word in text for word in ['계산', '구하', '값은']):
            return 'calculation'
        elif any(word in text for word in ['회로', '해석', '분석']):
            return 'circuit_analysis'
        elif any(word in text for word in ['역률', '개선', '보상']):
            return 'power_factor_correction'
        elif any(word in text for word in ['3상', '삼상']):
            return 'three_phase'
        elif any(word in text for word in ['변압기', '트랜스']):
            return 'transformer'
        else:
            return 'general'
    
    def _classify_formula(self, latex: str) -> str:
        """수식 분류"""
        if 'V' in latex and 'I' in latex and 'R' in latex:
            return 'ohms_law'
        elif 'P' in latex and 'cos' in latex:
            return 'power'
        elif 'Z' in latex:
            return 'impedance'
        else:
            return 'general'
    
    def _get_formula_description(self, formula_type: str) -> str:
        """공식 설명"""
        descriptions = {
            'ohms_law': '옴의 법칙',
            'ac_power': '교류 전력',
            'three_phase_power': '3상 전력',
            'impedance': '임피던스',
            'power_triangle': '전력 삼각형',
            'power': '전력 공식',
            'general': '일반 수식'
        }
        return descriptions.get(formula_type, '알 수 없는 공식')


# 테스트용
if __name__ == "__main__":
    # 파이프라인 생성
    ocr = KoreanElectricalOCR()
    
    # 모델 로드 상태 확인
    print("Loaded models:", ocr.models_loaded)
    
    # 테스트 이미지가 있다면
    # results = ocr.extract_electrical_data("test_image.jpg")
    # print(results)