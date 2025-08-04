import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from PIL import Image
import cv2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MathSymbol(Enum):
    """전기공학 수학 기호 정의"""
    VOLTAGE = ['V', 'v', 'U', 'u', 'E', 'e']
    CURRENT = ['I', 'i', 'A']
    RESISTANCE = ['R', 'r', 'Ω']
    POWER = ['P', 'p', 'W']
    CAPACITANCE = ['C', 'c', 'F']
    INDUCTANCE = ['L', 'l', 'H']
    FREQUENCY = ['f', 'Hz']
    IMPEDANCE = ['Z', 'z']

@dataclass
class MathExpression:
    """수식 표현 데이터 클래스"""
    raw_text: str
    latex: Optional[str] = None
    variables: Dict[str, Any] = None
    formula_type: Optional[str] = None
    confidence: float = 0.0

class MathOCRSystem:
    """수식 인식 특화 OCR 시스템"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processors = {}
        self.models = {}
        self.easyocr_reader = None
        self.formula_patterns = self._load_formula_patterns()
        
    async def initialize(self):
        """시스템 초기화"""
        try:
            # TrOCR 모델 로드
            logger.info("Loading TrOCR models...")
            self.processors['trocr'] = TrOCRProcessor.from_pretrained(
                'microsoft/trocr-base-handwritten'
            )
            self.models['trocr'] = VisionEncoderDecoderModel.from_pretrained(
                'microsoft/trocr-base-handwritten'
            ).to(self.device)
            
            # EasyOCR 초기화
            logger.info("Initializing EasyOCR...")
            self.easyocr_reader = easyocr.Reader(
                ['en', 'ko'], 
                gpu=torch.cuda.is_available()
            )
            
            logger.info("Math OCR System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Math OCR System: {e}")
            raise
    
    def _load_formula_patterns(self) -> Dict[str, re.Pattern]:
        """전기공학 공식 패턴 로드"""
        return {
            'ohms_law': re.compile(r'[VvUuEe]\s*=\s*[IiAa]\s*[×*·]\s*[RrΩ]'),
            'power_vi': re.compile(r'[Pp]\s*=\s*[VvUu]\s*[×*·]\s*[IiAa]'),
            'power_i2r': re.compile(r'[Pp]\s*=\s*[IiAa]\s*\^?\s*2\s*[×*·]\s*[RrΩ]'),
            'power_v2r': re.compile(r'[Pp]\s*=\s*[VvUu]\s*\^?\s*2\s*/\s*[RrΩ]'),
            'capacitance': re.compile(r'[Qq]\s*=\s*[Cc]\s*[×*·]\s*[VvUu]'),
            'inductance': re.compile(r'[VvUuEe]\s*=\s*[Ll]\s*d[IiAa]/dt'),
            'impedance': re.compile(r'[Zz]\s*=\s*[RrΩ]\s*\+\s*j\s*[Xx]'),
        }
    
    async def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """이미지 처리 메인 함수"""
        try:
            # 1. 이미지 전처리
            preprocessed = await self._preprocess_image(image)
            
            # 2. 다중 OCR 실행
            ocr_results = await self._multi_ocr_process(preprocessed)
            
            # 3. 수식 추출 및 분석
            math_expressions = await self._extract_math_expressions(ocr_results)
            
            # 4. 전기공학 공식 매칭
            matched_formulas = await self._match_electrical_formulas(math_expressions)
            
            # 5. 결과 통합
            result = await self._integrate_results(
                ocr_results, math_expressions, matched_formulas
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in process_image: {e}")
            return {'error': str(e)}
    
    async def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """이미지 전처리"""
        # PIL to OpenCV
        img_array = np.array(image)
        
        # 그레이스케일 변환
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 이진화
        _, binary = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return binary
    
    async def _multi_ocr_process(self, image: np.ndarray) -> Dict[str, List[str]]:
        """다중 OCR 처리"""
        results = {}
        
        # EasyOCR 처리
        if self.easyocr_reader:
            easy_results = self.easyocr_reader.readtext(image)
            results['easyocr'] = [
                text for _, text, conf in easy_results if conf > 0.5
            ]
        
        # TrOCR 처리 (이미지를 패치로 분할)
        patches = self._split_image_patches(image)
        trocr_results = []
        
        for patch in patches:
            pil_patch = Image.fromarray(patch)
            inputs = self.processors['trocr'](
                images=pil_patch, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.models['trocr'].generate(inputs.pixel_values)
                text = self.processors['trocr'].batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                trocr_results.append(text)
        
        results['trocr'] = trocr_results
        
        return results
    
    def _split_image_patches(self, image: np.ndarray, patch_size: int = 256) -> List[np.ndarray]:
        """이미지를 패치로 분할"""
        h, w = image.shape[:2]
        patches = []
        
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = image[y:y+patch_size, x:x+patch_size]
                if patch.size > 0:
                    patches.append(patch)
        
        return patches
    
    async def _extract_math_expressions(self, ocr_results: Dict[str, List[str]]) -> List[MathExpression]:
        """수식 추출"""
        expressions = []
        
        # 모든 OCR 결과 통합
        all_texts = []
        for source, texts in ocr_results.items():
            all_texts.extend(texts)
        
        # 수식 패턴 검색
        math_pattern = re.compile(
            r'[A-Za-z]\s*[=<>≤≥]\s*[\d\w\s\+\-\*\/\^\(\)\.]+|'
            r'\d+\s*[+\-*/]\s*\d+|'
            r'[A-Za-z]+\s*\(\s*[^)]+\s*\)'
        )
        
        for text in all_texts:
            matches = math_pattern.findall(text)
            for match in matches:
                expr = MathExpression(
                    raw_text=match,
                    confidence=0.8  # 기본 신뢰도
                )
                expressions.append(expr)
        
        return expressions
    
    async def _match_electrical_formulas(self, expressions: List[MathExpression]) -> List[MathExpression]:
        """전기공학 공식 매칭"""
        matched = []
        
        for expr in expressions:
            for formula_name, pattern in self.formula_patterns.items():
                if pattern.search(expr.raw_text):
                    expr.formula_type = formula_name
                    expr.confidence = min(expr.confidence + 0.2, 1.0)
                    
                    # LaTeX 변환
                    expr.latex = self._convert_to_latex(expr.raw_text, formula_name)
                    
                    # 변수 추출
                    expr.variables = self._extract_variables(expr.raw_text)
                    
                    matched.append(expr)
                    break
        
        return matched
    
    def _convert_to_latex(self, text: str, formula_type: str) -> str:
        """텍스트를 LaTeX로 변환"""
        latex = text
        
        # 기본 변환 규칙
        replacements = {
            '^2': '^{2}',
            '^3': '^{3}',
            '*': '\\cdot ',
            '×': '\\times ',
            '÷': '\\div ',
            '≤': '\\leq ',
            '≥': '\\geq ',
            'Ω': '\\Omega ',
            'μ': '\\mu ',
            'π': '\\pi ',
        }
        
        for old, new in replacements.items():
            latex = latex.replace(old, new)
        
        # 분수 형태 변환
        latex = re.sub(r'(\w+)/(\w+)', r'\\frac{\1}{\2}', latex)
        
        return f"${latex}$"
    
    def _extract_variables(self, text: str) -> Dict[str, str]:
        """수식에서 변수 추출"""
        variables = {}
        
        # 숫자와 단위 추출
        number_pattern = re.compile(r'(\d+\.?\d*)\s*([A-Za-zΩμ]+)?')
        matches = number_pattern.findall(text)
        
        for value, unit in matches:
            if unit:
                # 단위에 따른 변수 유형 결정
                if unit in ['V', 'v']:
                    variables['voltage'] = f"{value} {unit}"
                elif unit in ['A', 'mA', 'μA']:
                    variables['current'] = f"{value} {unit}"
                elif unit in ['Ω', 'kΩ', 'MΩ']:
                    variables['resistance'] = f"{value} {unit}"
                elif unit in ['W', 'kW', 'MW']:
                    variables['power'] = f"{value} {unit}"
        
        return variables
    
    async def _integrate_results(
        self, 
        ocr_results: Dict[str, List[str]], 
        math_expressions: List[MathExpression],
        matched_formulas: List[MathExpression]
    ) -> Dict[str, Any]:
        """결과 통합"""
        # 모든 텍스트 결합
        all_text = ' '.join([
            text for texts in ocr_results.values() for text in texts
        ])
        
        # 수식 정보 정리
        formulas_info = []
        for formula in matched_formulas:
            formulas_info.append({
                'raw': formula.raw_text,
                'latex': formula.latex,
                'type': formula.formula_type,
                'variables': formula.variables,
                'confidence': formula.confidence
            })
        
        # 전기공학 컨텍스트 추가
        context = self._generate_electrical_context(matched_formulas)
        
        return {
            'ocr_text': all_text,
            'formulas': formulas_info,
            'electrical_context': context,
            'confidence': np.mean([f.confidence for f in matched_formulas]) if matched_formulas else 0.5
        }
    
    def _generate_electrical_context(self, formulas: List[MathExpression]) -> str:
        """전기공학 컨텍스트 생성"""
        contexts = []
        
        for formula in formulas:
            if formula.formula_type == 'ohms_law':
                contexts.append("옴의 법칙 (V=IR): 전압은 전류와 저항의 곱")
            elif formula.formula_type == 'power_vi':
                contexts.append("전력 공식 (P=VI): 전력은 전압과 전류의 곱")
            elif formula.formula_type == 'power_i2r':
                contexts.append("전력 공식 (P=I²R): 전력은 전류의 제곱과 저항의 곱")
            elif formula.formula_type == 'capacitance':
                contexts.append("축전기 공식 (Q=CV): 전하량은 정전용량과 전압의 곱")
            elif formula.formula_type == 'inductance':
                contexts.append("인덕턴스 공식 (V=L·di/dt): 유도 전압")
        
        return ' | '.join(contexts) if contexts else "전기공학 관련 수식"