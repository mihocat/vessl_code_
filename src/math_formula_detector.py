#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Formula Detection and Recognition System
수식 감지 및 인식 시스템
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from PIL import Image
import numpy as np
import cv2
import re
from dataclasses import dataclass

# SymPy for mathematical operations
try:
    import sympy as sp
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logging.warning("SymPy not available for formula parsing")

# LaTeX OCR (pix2tex alternative)
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    logging.warning("TrOCR not available for formula recognition")

logger = logging.getLogger(__name__)


@dataclass
class FormulaRegion:
    """수식 영역 정보"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    image: np.ndarray
    confidence: float
    formula_type: str  # 'inline', 'display', 'handwritten'


@dataclass
class RecognizedFormula:
    """인식된 수식 정보"""
    latex: str
    sympy_expr: Optional[Any]  # SymPy expression
    region: FormulaRegion
    solution: Optional[Dict[str, Any]]
    confidence: float


class FormulaDetector:
    """수식 영역 감지기"""
    
    def __init__(self):
        """초기화"""
        # 수식 감지를 위한 패턴
        self.math_patterns = [
            r'[=+\-*/^()]+',  # 수학 연산자
            r'\\[a-zA-Z]+',   # LaTeX 명령어
            r'\d+[a-zA-Z]',   # 숫자+변수 (예: 2x)
            r'[∫∑∏√∞αβγδεζηθ]',  # 수학 기호
        ]
    
    def detect_formulas(self, image: Union[Image.Image, np.ndarray]) -> List[FormulaRegion]:
        """
        이미지에서 수식 영역 감지
        
        Args:
            image: 입력 이미지
            
        Returns:
            감지된 수식 영역 리스트
        """
        # PIL Image를 numpy array로 변환
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # 그레이스케일 변환
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # 엣지 감지
        edges = cv2.Canny(gray, 50, 150)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        formula_regions = []
        
        for contour in contours:
            # 바운딩 박스 계산
            x, y, w, h = cv2.boundingRect(contour)
            
            # 너무 작거나 큰 영역 제외
            if w < 20 or h < 10 or w > img_array.shape[1] * 0.9:
                continue
            
            # 종횡비 확인 (수식은 보통 가로로 긴 형태)
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 10:
                continue
            
            # 영역 추출
            region_img = img_array[y:y+h, x:x+w]
            
            # 수식 가능성 평가
            confidence = self._evaluate_formula_likelihood(region_img)
            
            if confidence > 0.5:
                formula_regions.append(FormulaRegion(
                    bbox=(x, y, x+w, y+h),
                    image=region_img,
                    confidence=confidence,
                    formula_type=self._classify_formula_type(region_img)
                ))
        
        # 겹치는 영역 병합
        merged_regions = self._merge_overlapping_regions(formula_regions)
        
        return merged_regions
    
    def _evaluate_formula_likelihood(self, region: np.ndarray) -> float:
        """수식일 가능성 평가"""
        # 간단한 휴리스틱 기반 평가
        confidence = 0.0
        
        # 대비 확인 (수식은 보통 명확한 대비)
        if region.std() > 50:
            confidence += 0.3
        
        # 수평선 비율 (분수 표현 등)
        edges = cv2.Canny(region, 50, 150)
        horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=5)
        if horizontal_lines is not None and len(horizontal_lines) > 0:
            confidence += 0.4
        
        # 텍스트 밀도 (수식은 일반 텍스트보다 밀도가 낮음)
        non_zero_ratio = np.count_nonzero(region) / region.size
        if 0.1 < non_zero_ratio < 0.7:
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def _classify_formula_type(self, region: np.ndarray) -> str:
        """수식 유형 분류"""
        h, w = region.shape[:2]
        
        # 종횡비로 판단
        aspect_ratio = w / h
        
        if aspect_ratio > 5:
            return 'inline'  # 인라인 수식
        elif h > 50:
            return 'display'  # 디스플레이 수식
        else:
            # 손글씨 여부 판단 (간단한 휴리스틱)
            edges = cv2.Canny(region, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            
            if edge_density > 0.1:
                return 'handwritten'
            else:
                return 'inline'
    
    def _merge_overlapping_regions(self, regions: List[FormulaRegion]) -> List[FormulaRegion]:
        """겹치는 영역 병합"""
        if not regions:
            return []
        
        # 신뢰도 순으로 정렬
        sorted_regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
        merged = []
        
        for region in sorted_regions:
            # 기존 영역과 겹치는지 확인
            overlap = False
            for i, existing in enumerate(merged):
                if self._regions_overlap(region.bbox, existing.bbox):
                    # 더 큰 영역으로 병합
                    merged[i] = self._merge_two_regions(existing, region)
                    overlap = True
                    break
            
            if not overlap:
                merged.append(region)
        
        return merged
    
    def _regions_overlap(self, bbox1: Tuple, bbox2: Tuple) -> bool:
        """두 영역이 겹치는지 확인"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
    
    def _merge_two_regions(self, region1: FormulaRegion, region2: FormulaRegion) -> FormulaRegion:
        """두 영역 병합"""
        x1 = min(region1.bbox[0], region2.bbox[0])
        y1 = min(region1.bbox[1], region2.bbox[1])
        x2 = max(region1.bbox[2], region2.bbox[2])
        y2 = max(region1.bbox[3], region2.bbox[3])
        
        # 더 높은 신뢰도 유지
        merged_region = FormulaRegion(
            bbox=(x1, y1, x2, y2),
            image=region1.image,  # 더 큰 이미지로 업데이트 필요
            confidence=max(region1.confidence, region2.confidence),
            formula_type=region1.formula_type
        )
        
        return merged_region


class FormulaRecognizer:
    """수식 인식기"""
    
    def __init__(self):
        """초기화"""
        self.latex_patterns = {
            'fraction': r'\\frac{([^}]+)}{([^}]+)}',
            'sqrt': r'\\sqrt{([^}]+)}',
            'power': r'\^{([^}]+)}',
            'subscript': r'_{([^}]+)}',
            'integral': r'\\int',
            'sum': r'\\sum',
            'limit': r'\\lim',
        }
        
        # TrOCR 모델 초기화 (가능한 경우)
        self.trocr_model = None
        self.trocr_processor = None
        
        if TROCR_AVAILABLE:
            try:
                # 수식 전용 모델이 있다면 사용, 없으면 일반 모델
                model_name = "microsoft/trocr-base-handwritten"
                self.trocr_processor = TrOCRProcessor.from_pretrained(model_name)
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
                logger.info("TrOCR model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load TrOCR model: {e}")
    
    def recognize_formula(self, formula_region: FormulaRegion) -> RecognizedFormula:
        """
        수식 영역에서 LaTeX 수식 인식
        
        Args:
            formula_region: 수식 영역 정보
            
        Returns:
            인식된 수식 정보
        """
        latex_formula = ""
        confidence = 0.0
        
        # 방법 1: TrOCR 사용 (가능한 경우)
        if self.trocr_model:
            try:
                latex_formula, confidence = self._recognize_with_trocr(formula_region.image)
            except Exception as e:
                logger.error(f"TrOCR recognition failed: {e}")
        
        # 방법 2: 규칙 기반 인식 (폴백)
        if not latex_formula:
            latex_formula, confidence = self._recognize_with_rules(formula_region.image)
        
        # SymPy 표현식으로 변환
        sympy_expr = None
        if SYMPY_AVAILABLE and latex_formula:
            try:
                sympy_expr = parse_latex(latex_formula)
            except Exception as e:
                logger.warning(f"Failed to parse LaTeX to SymPy: {e}")
        
        # 수식 해결 시도
        solution = None
        if sympy_expr:
            solution = self._try_solve_formula(sympy_expr, latex_formula)
        
        return RecognizedFormula(
            latex=latex_formula,
            sympy_expr=sympy_expr,
            region=formula_region,
            solution=solution,
            confidence=confidence
        )
    
    def _recognize_with_trocr(self, image: np.ndarray) -> Tuple[str, float]:
        """TrOCR을 사용한 수식 인식"""
        # 이미지 전처리
        pil_image = Image.fromarray(image)
        
        # 모델 입력 준비
        pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values
        
        # 추론
        generated_ids = self.trocr_model.generate(pixel_values)
        generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 후처리 (일반 텍스트를 LaTeX로 변환)
        latex_formula = self._text_to_latex(generated_text)
        
        # 신뢰도는 모델 출력에서 추출 (간단히 0.8로 설정)
        confidence = 0.8
        
        return latex_formula, confidence
    
    def _recognize_with_rules(self, image: np.ndarray) -> Tuple[str, float]:
        """규칙 기반 수식 인식 (간단한 구현)"""
        # 여기서는 매우 간단한 패턴만 인식
        # 실제로는 더 복잡한 컴퓨터 비전 기법 필요
        
        # 이진화
        _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
        
        # 컨투어 분석으로 수식 구조 파악
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 간단한 예: 분수 감지
        if len(contours) >= 3:
            # 수평선이 있고 위아래에 요소가 있으면 분수로 추정
            return "\\frac{a}{b}", 0.5
        
        # 기본값
        return "a + b = c", 0.3
    
    def _text_to_latex(self, text: str) -> str:
        """일반 텍스트를 LaTeX 수식으로 변환"""
        # 텍스트가 수식이 아닌 일반 텍스트로 보이면 빈 문자열 반환
        # 날짜 형식이나 일반 영어 단어가 포함된 경우 제외
        if any(word in text.lower() for word in ['year', 'sep', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec']):
            logger.info(f"Skipping non-formula text: {text}")
            return ""
        
        # 영어 단어가 많이 포함된 경우 수식이 아닐 가능성이 높음
        if len(re.findall(r'[a-zA-Z]{5,}', text)) > 1:
            logger.info(f"Skipping text with long words: {text}")
            return ""
        
        # 간단한 변환 규칙
        latex = text.strip()
        
        # 기본 연산자 변환
        latex = latex.replace('/', '\\div')
        latex = latex.replace('*', '\\times')
        latex = latex.replace('×', '\\times')
        latex = latex.replace('÷', '\\div')
        latex = latex.replace('±', '\\pm')
        latex = latex.replace('≥', '\\geq')
        latex = latex.replace('≤', '\\leq')
        latex = latex.replace('≠', '\\neq')
        latex = latex.replace('∞', '\\infty')
        latex = latex.replace('π', '\\pi')
        latex = latex.replace('Σ', '\\sum')
        latex = latex.replace('∫', '\\int')
        latex = latex.replace('√', '\\sqrt')
        
        # 함수 이름
        latex = latex.replace('sqrt', '\\sqrt')
        latex = latex.replace('sum', '\\sum')
        latex = latex.replace('int', '\\int')
        latex = latex.replace('lim', '\\lim')
        latex = latex.replace('sin', '\\sin')
        latex = latex.replace('cos', '\\cos')
        latex = latex.replace('tan', '\\tan')
        latex = latex.replace('log', '\\log')
        latex = latex.replace('ln', '\\ln')
        
        # 지수 표현
        latex = re.sub(r'(\w)\^(\w+)', r'\1^{\2}', latex)
        
        # 아래첨자
        latex = re.sub(r'(\w)_(\w+)', r'\1_{\2}', latex)
        
        # 분수 표기
        latex = re.sub(r'(\d+)/(\d+)', r'\\frac{\1}{\2}', latex)
        
        return latex
    
    def _try_solve_formula(self, sympy_expr: Any, latex: str) -> Optional[Dict[str, Any]]:
        """수식 해결 시도"""
        if not SYMPY_AVAILABLE:
            return None
        
        solution = {
            'type': 'unknown',
            'result': None,
            'steps': []
        }
        
        try:
            # 방정식인지 확인
            if isinstance(sympy_expr, sp.Eq):
                solution['type'] = 'equation'
                # 변수 찾기
                variables = list(sympy_expr.free_symbols)
                if len(variables) == 1:
                    # 단일 변수 방정식 해결
                    result = sp.solve(sympy_expr, variables[0])
                    solution['result'] = str(result)
                    solution['steps'].append(f"Solving for {variables[0]}")
            
            # 적분인지 확인
            elif 'int' in latex.lower():
                solution['type'] = 'integral'
                # 간단한 적분 계산
                if hasattr(sympy_expr, 'integrate'):
                    result = sympy_expr.integrate()
                    solution['result'] = str(result)
            
            # 미분인지 확인
            elif 'd/dx' in latex or 'prime' in latex:
                solution['type'] = 'derivative'
                # 간단한 미분 계산
                if hasattr(sympy_expr, 'diff'):
                    result = sympy_expr.diff()
                    solution['result'] = str(result)
            
            # 일반 수식 평가
            else:
                solution['type'] = 'expression'
                # 수식 간소화
                simplified = sp.simplify(sympy_expr)
                solution['result'] = str(simplified)
                
                # 수치 평가 (가능한 경우)
                try:
                    numerical = float(sympy_expr.evalf())
                    solution['numerical_value'] = numerical
                except:
                    pass
        
        except Exception as e:
            logger.warning(f"Failed to solve formula: {e}")
            solution['error'] = str(e)
        
        return solution


class MathFormulaProcessor:
    """수식 처리 통합 시스템"""
    
    def __init__(self):
        """초기화"""
        self.detector = FormulaDetector()
        self.recognizer = FormulaRecognizer()
    
    def process_image(self, image: Union[Image.Image, np.ndarray]) -> List[RecognizedFormula]:
        """
        이미지에서 수식 감지 및 인식
        
        Args:
            image: 입력 이미지
            
        Returns:
            인식된 수식 리스트
        """
        # 수식 영역 감지
        formula_regions = self.detector.detect_formulas(image)
        logger.info(f"Detected {len(formula_regions)} formula regions")
        
        # 각 영역에서 수식 인식
        recognized_formulas = []
        
        for region in formula_regions:
            try:
                formula = self.recognizer.recognize_formula(region)
                recognized_formulas.append(formula)
                logger.info(f"Recognized formula: {formula.latex} (confidence: {formula.confidence:.2f})")
            except Exception as e:
                logger.error(f"Failed to recognize formula in region: {e}")
        
        return recognized_formulas
    
    def format_results(self, formulas: List[RecognizedFormula]) -> str:
        """인식 결과를 포맷팅"""
        if not formulas:
            return "수식이 감지되지 않았습니다."
        
        results = []
        
        for i, formula in enumerate(formulas, 1):
            result_text = f"수식 {i}:\n"
            result_text += f"  LaTeX: {formula.latex}\n"
            result_text += f"  신뢰도: {formula.confidence:.2%}\n"
            
            if formula.solution:
                result_text += f"  유형: {formula.solution['type']}\n"
                if formula.solution.get('result'):
                    result_text += f"  결과: {formula.solution['result']}\n"
                if formula.solution.get('numerical_value'):
                    result_text += f"  수치값: {formula.solution['numerical_value']:.6f}\n"
            
            results.append(result_text)
        
        return "\n".join(results)