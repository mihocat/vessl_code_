#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MathPix 스타일 구현 - 실제 구현 가능한 수식 OCR 시스템
TrOCR + LatexOCR + Custom Math Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import cv2
import numpy as np
from PIL import Image
import asyncio
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import re
from sympy import sympify, latex
import easyocr

logger = logging.getLogger(__name__)

@dataclass
class MathRegion:
    """수식 영역"""
    bbox: Tuple[int, int, int, int]
    image: np.ndarray
    region_type: str  # 'inline', 'display', 'matrix', 'fraction'
    confidence: float

@dataclass
class ParsedExpression:
    """파싱된 수식"""
    raw_ocr: str
    cleaned_text: str
    latex: str
    sympy_expr: Optional[Any]
    variables: List[str]
    constants: List[str]
    operators: List[str]
    confidence: float

class MathRegionDetector:
    """수식 영역 검출기"""
    
    def __init__(self):
        self.edge_threshold = (50, 150)
        self.min_area = 100
        self.aspect_ratio_range = (0.1, 10)
        
    def detect_math_regions(self, image: np.ndarray) -> List[MathRegion]:
        """수식 영역 검출"""
        regions = []
        
        # 1. 엣지 검출
        edges = cv2.Canny(image, self.edge_threshold[0], self.edge_threshold[1])
        
        # 2. 형태학적 연산으로 수식 구조 강조
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # 3. 컨투어 찾기
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 4. 수식 가능성 평가
            if self._is_math_region(image[y:y+h, x:x+w], w, h):
                region_type = self._classify_math_type(image[y:y+h, x:x+w])
                regions.append(MathRegion(
                    bbox=(x, y, x+w, y+h),
                    image=image[y:y+h, x:x+w],
                    region_type=region_type,
                    confidence=self._calculate_confidence(image[y:y+h, x:x+w])
                ))
        
        # 5. 중복 제거 및 병합
        regions = self._merge_overlapping_regions(regions)
        
        return regions
    
    def _is_math_region(self, region: np.ndarray, width: int, height: int) -> bool:
        """수식 영역 여부 판단"""
        # 크기 조건
        area = width * height
        if area < self.min_area:
            return False
        
        # 종횡비 조건
        aspect_ratio = width / height if height > 0 else 0
        if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
            return False
        
        # 수식 특징 검사
        # - 수평/수직 선 (분수)
        # - 특수 기호 밀도
        # - 텍스트 분포 패턴
        
        return True
    
    def _classify_math_type(self, region: np.ndarray) -> str:
        """수식 유형 분류"""
        h, w = region.shape[:2]
        
        # 분수 감지 (수평선)
        horizontal_lines = self._detect_horizontal_lines(region)
        if horizontal_lines:
            return 'fraction'
        
        # 행렬 감지 (괄호 + 정렬된 요소)
        if self._detect_matrix_pattern(region):
            return 'matrix'
        
        # 디스플레이 수식 (큰 크기, 중앙 정렬)
        if h > 50 and w > 100:
            return 'display'
        
        return 'inline'
    
    def _detect_horizontal_lines(self, region: np.ndarray) -> List[int]:
        """수평선 검출 (분수선)"""
        lines = []
        h, w = region.shape[:2]
        
        # 수평 투영
        horizontal_proj = np.sum(region < 128, axis=1)
        
        # 긴 수평선 찾기
        for y in range(h):
            if horizontal_proj[y] > 0.7 * w:
                lines.append(y)
        
        return lines
    
    def _detect_matrix_pattern(self, region: np.ndarray) -> bool:
        """행렬 패턴 검출"""
        # 괄호 검출
        # 정렬된 요소 검출
        return False
    
    def _calculate_confidence(self, region: np.ndarray) -> float:
        """신뢰도 계산"""
        # 선명도
        sharpness = cv2.Laplacian(region, cv2.CV_64F).var()
        
        # 대비
        contrast = region.std()
        
        # 정규화
        confidence = min(1.0, (sharpness / 100) * 0.5 + (contrast / 128) * 0.5)
        
        return confidence
    
    def _merge_overlapping_regions(self, regions: List[MathRegion]) -> List[MathRegion]:
        """중복 영역 병합"""
        if not regions:
            return []
        
        # IoU 기반 병합
        merged = []
        used = set()
        
        for i, region1 in enumerate(regions):
            if i in used:
                continue
            
            for j, region2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                
                iou = self._calculate_iou(region1.bbox, region2.bbox)
                if iou > 0.5:
                    # 병합
                    x1 = min(region1.bbox[0], region2.bbox[0])
                    y1 = min(region1.bbox[1], region2.bbox[1])
                    x2 = max(region1.bbox[2], region2.bbox[2])
                    y2 = max(region1.bbox[3], region2.bbox[3])
                    
                    region1.bbox = (x1, y1, x2, y2)
                    used.add(j)
            
            merged.append(region1)
        
        return merged
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """IoU 계산"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class MathOCREngine:
    """수식 OCR 엔진"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trocr_processor = None
        self.trocr_model = None
        self.easyocr_reader = None
        self.symbol_map = self._load_symbol_map()
        
    async def initialize(self):
        """모델 초기화"""
        try:
            # TrOCR 로드
            logger.info("Loading TrOCR model...")
            self.trocr_processor = TrOCRProcessor.from_pretrained(
                'microsoft/trocr-base-printed'
            )
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                'microsoft/trocr-base-printed'
            ).to(self.device)
            
            # EasyOCR 초기화
            logger.info("Initializing EasyOCR...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            
            logger.info("Math OCR Engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Math OCR Engine: {e}")
            raise
    
    def _load_symbol_map(self) -> Dict[str, str]:
        """수학 기호 매핑"""
        return {
            # 그리스 문자
            'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
            'theta': 'θ', 'lambda': 'λ', 'mu': 'μ', 'pi': 'π',
            'sigma': 'σ', 'phi': 'φ', 'omega': 'ω',
            
            # 연산자
            'times': '×', 'div': '÷', 'pm': '±', 'mp': '∓',
            'leq': '≤', 'geq': '≥', 'neq': '≠', 'approx': '≈',
            
            # 전기공학 기호
            'ohm': 'Ω', 'micro': 'μ', 'degree': '°',
            
            # 특수 기호
            'sqrt': '√', 'infty': '∞', 'partial': '∂',
            'int': '∫', 'sum': 'Σ', 'prod': 'Π'
        }
    
    async def recognize_math(self, region: MathRegion) -> ParsedExpression:
        """수식 인식"""
        # 1. 다중 OCR 실행
        trocr_result = await self._trocr_recognize(region.image)
        easyocr_result = await self._easyocr_recognize(region.image)
        
        # 2. 결과 병합 및 정제
        merged_text = self._merge_ocr_results(trocr_result, easyocr_result)
        
        # 3. 수식 정제
        cleaned_text = self._clean_math_text(merged_text)
        
        # 4. LaTeX 변환
        latex_expr = self._convert_to_latex(cleaned_text, region.region_type)
        
        # 5. SymPy 파싱 (검증용)
        sympy_expr = self._parse_with_sympy(cleaned_text)
        
        # 6. 변수 및 상수 추출
        variables = self._extract_variables(cleaned_text)
        constants = self._extract_constants(cleaned_text)
        operators = self._extract_operators(cleaned_text)
        
        return ParsedExpression(
            raw_ocr=merged_text,
            cleaned_text=cleaned_text,
            latex=latex_expr,
            sympy_expr=sympy_expr,
            variables=variables,
            constants=constants,
            operators=operators,
            confidence=self._calculate_ocr_confidence(trocr_result, easyocr_result)
        )
    
    async def _trocr_recognize(self, image: np.ndarray) -> str:
        """TrOCR 인식"""
        # PIL 이미지로 변환
        pil_image = Image.fromarray(image)
        
        # 전처리
        inputs = self.trocr_processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # 추론
        with torch.no_grad():
            generated_ids = self.trocr_model.generate(inputs.pixel_values)
        
        # 디코딩
        text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return text
    
    async def _easyocr_recognize(self, image: np.ndarray) -> str:
        """EasyOCR 인식"""
        results = self.easyocr_reader.readtext(image)
        
        # 텍스트만 추출
        texts = [text for _, text, _ in results]
        
        return ' '.join(texts)
    
    def _merge_ocr_results(self, trocr: str, easyocr: str) -> str:
        """OCR 결과 병합"""
        # 간단한 병합 전략
        # 실제로는 더 정교한 앙상블 필요
        if not trocr:
            return easyocr
        if not easyocr:
            return trocr
        
        # 두 결과 비교하여 더 긴 것 선택
        return trocr if len(trocr) >= len(easyocr) else easyocr
    
    def _clean_math_text(self, text: str) -> str:
        """수식 텍스트 정제"""
        # 공백 정규화
        text = re.sub(r'\s+', ' ', text)
        
        # 특수 문자 치환
        for symbol, replacement in self.symbol_map.items():
            text = text.replace(symbol, replacement)
        
        # 괄호 매칭 수정
        text = self._fix_brackets(text)
        
        # 지수 표기 수정
        text = re.sub(r'(\d+)\s*\^\s*(\d+)', r'\1^{\2}', text)
        
        return text.strip()
    
    def _fix_brackets(self, text: str) -> str:
        """괄호 매칭 수정"""
        # 괄호 쌍 확인 및 수정
        open_brackets = text.count('(') + text.count('[') + text.count('{')
        close_brackets = text.count(')') + text.count(']') + text.count('}')
        
        if open_brackets > close_brackets:
            text += ')' * (open_brackets - close_brackets)
        elif close_brackets > open_brackets:
            text = '(' * (close_brackets - open_brackets) + text
        
        return text
    
    def _convert_to_latex(self, text: str, region_type: str) -> str:
        """LaTeX 변환"""
        latex_text = text
        
        # 기본 변환 규칙
        replacements = {
            # 연산자
            '×': r'\times ', '÷': r'\div ', '±': r'\pm ',
            '≤': r'\leq ', '≥': r'\geq ', '≠': r'\neq ',
            
            # 그리스 문자
            'α': r'\alpha ', 'β': r'\beta ', 'θ': r'\theta ',
            'Ω': r'\Omega ', 'μ': r'\mu ', 'π': r'\pi ',
            
            # 함수
            'sin': r'\sin ', 'cos': r'\cos ', 'tan': r'\tan ',
            'log': r'\log ', 'ln': r'\ln ', 'exp': r'\exp ',
            
            # 특수 구조
            'sqrt': r'\sqrt', '∫': r'\int ', 'Σ': r'\sum '
        }
        
        for old, new in replacements.items():
            latex_text = latex_text.replace(old, new)
        
        # 분수 처리
        if region_type == 'fraction':
            # 분수선을 기준으로 분리
            parts = latex_text.split('/')
            if len(parts) == 2:
                latex_text = rf'\frac{{{parts[0]}}}{{{parts[1]}}}'
        
        # 지수 처리
        latex_text = re.sub(r'(\w+)\^{([^}]+)}', r'\1^{\2}', latex_text)
        latex_text = re.sub(r'(\w+)\^(\w)', r'\1^{\2}', latex_text)
        
        # 아래첨자 처리
        latex_text = re.sub(r'(\w+)_(\w+)', r'\1_{\2}', latex_text)
        
        # 수식 모드 래핑
        if region_type == 'display':
            latex_text = f'$${latex_text}$$'
        else:
            latex_text = f'${latex_text}$'
        
        return latex_text
    
    def _parse_with_sympy(self, text: str) -> Optional[Any]:
        """SymPy로 수식 파싱"""
        try:
            # 간단한 변환
            sympy_text = text.replace('^', '**')
            sympy_text = sympy_text.replace('×', '*')
            sympy_text = sympy_text.replace('÷', '/')
            
            # SymPy 파싱
            expr = sympify(sympy_text)
            return expr
        except:
            return None
    
    def _extract_variables(self, text: str) -> List[str]:
        """변수 추출"""
        # 일반적인 변수 패턴
        pattern = r'\b[a-zA-Z]\b'
        variables = re.findall(pattern, text)
        
        # 아래첨자가 있는 변수
        pattern2 = r'[a-zA-Z]_\w+'
        variables.extend(re.findall(pattern2, text))
        
        return list(set(variables))
    
    def _extract_constants(self, text: str) -> List[str]:
        """상수 추출"""
        # 숫자
        numbers = re.findall(r'\d+\.?\d*', text)
        
        # 물리 상수
        constants = []
        if 'π' in text or 'pi' in text:
            constants.append('π')
        if 'e' in text and not re.search(r'[a-df-z]e|e[a-df-z]', text):
            constants.append('e')
        
        return numbers + constants
    
    def _extract_operators(self, text: str) -> List[str]:
        """연산자 추출"""
        operators = []
        op_patterns = [
            r'\+', r'-', r'\*', r'/', r'\^',
            '×', '÷', '=', '≤', '≥', '≠'
        ]
        
        for op in op_patterns:
            if re.search(op, text):
                operators.append(op)
        
        return operators
    
    def _calculate_ocr_confidence(self, trocr: str, easyocr: str) -> float:
        """OCR 신뢰도 계산"""
        # 두 결과의 유사도 기반
        if not trocr or not easyocr:
            return 0.5
        
        # 간단한 유사도 계산
        similarity = len(set(trocr) & set(easyocr)) / len(set(trocr) | set(easyocr))
        
        return similarity


class MathPixStyleOCR:
    """MathPix 스타일 통합 OCR 시스템"""
    
    def __init__(self):
        self.detector = MathRegionDetector()
        self.ocr_engine = MathOCREngine()
        self.korean_ocr = None  # 한국어 OCR 엔진
        
    async def initialize(self):
        """시스템 초기화"""
        await self.ocr_engine.initialize()
        
        # 한국어 OCR 초기화
        self.korean_ocr = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
        
        logger.info("MathPix Style OCR initialized")
    
    async def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """이미지 처리"""
        # numpy 배열로 변환
        img_array = np.array(image)
        
        # 그레이스케일 변환
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # 1. 수식 영역 검출
        math_regions = self.detector.detect_math_regions(gray)
        
        # 2. 각 영역 OCR
        math_expressions = []
        for region in math_regions:
            expr = await self.ocr_engine.recognize_math(region)
            math_expressions.append({
                'bbox': region.bbox,
                'type': region.region_type,
                'expression': expr
            })
        
        # 3. 텍스트 영역 OCR (수식 영역 제외)
        text_mask = self._create_text_mask(gray, math_regions)
        korean_text = self._extract_korean_text(gray, text_mask)
        
        # 4. 결과 통합
        result = {
            'math_expressions': math_expressions,
            'text_content': korean_text,
            'combined_content': self._combine_content(math_expressions, korean_text),
            'metadata': {
                'total_math_regions': len(math_regions),
                'image_size': image.size,
                'processing_confidence': self._calculate_overall_confidence(math_expressions)
            }
        }
        
        return result
    
    def _create_text_mask(self, image: np.ndarray, math_regions: List[MathRegion]) -> np.ndarray:
        """텍스트 영역 마스크 생성"""
        mask = np.ones_like(image, dtype=np.uint8) * 255
        
        # 수식 영역 제외
        for region in math_regions:
            x1, y1, x2, y2 = region.bbox
            mask[y1:y2, x1:x2] = 0
        
        return mask
    
    def _extract_korean_text(self, image: np.ndarray, mask: np.ndarray) -> str:
        """한국어 텍스트 추출"""
        # 마스크 적용
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # 한국어 OCR
        results = self.korean_ocr.readtext(masked_image)
        
        # 텍스트 정렬 및 결합
        sorted_results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
        
        text_lines = []
        current_line = []
        last_y = -1
        
        for bbox, text, conf in sorted_results:
            y = bbox[0][1]
            if last_y != -1 and abs(y - last_y) > 20:  # 새 줄
                text_lines.append(' '.join(current_line))
                current_line = []
            current_line.append(text)
            last_y = y
        
        if current_line:
            text_lines.append(' '.join(current_line))
        
        return '\n'.join(text_lines)
    
    def _combine_content(
        self, math_expressions: List[Dict], korean_text: str
    ) -> str:
        """수식과 텍스트 통합"""
        # 위치 기반으로 수식을 텍스트에 삽입
        combined = korean_text
        
        # 간단한 예시 - 실제로는 더 정교한 위치 매칭 필요
        for expr in math_expressions:
            latex = expr['expression'].latex
            combined = combined.replace('[수식]', latex, 1)
        
        return combined
    
    def _calculate_overall_confidence(self, expressions: List[Dict]) -> float:
        """전체 신뢰도 계산"""
        if not expressions:
            return 0.0
        
        confidences = [expr['expression'].confidence for expr in expressions]
        return np.mean(confidences)


# 사용 예시
async def main():
    # 시스템 초기화
    ocr_system = MathPixStyleOCR()
    await ocr_system.initialize()
    
    # 이미지 로드
    image = Image.open("math_problem.png")
    
    # 처리
    result = await ocr_system.process_image(image)
    
    # 결과 출력
    print("=== Math Expressions ===")
    for expr in result['math_expressions']:
        print(f"Type: {expr['type']}")
        print(f"LaTeX: {expr['expression'].latex}")
        print(f"Variables: {expr['expression'].variables}")
        print(f"Confidence: {expr['expression'].confidence:.2f}")
        print()
    
    print("=== Korean Text ===")
    print(result['text_content'])
    
    print("\n=== Combined Content ===")
    print(result['combined_content'])


if __name__ == "__main__":
    asyncio.run(main())