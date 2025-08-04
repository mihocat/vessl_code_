#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실제 작동하는 OCR 시스템
전기공학 문제 이미지를 정확히 인식하는 시스템
"""

import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from PIL import Image
import numpy as np
import cv2
import torch
import asyncio
from dataclasses import dataclass
import re

# OCR 엔진들
import easyocr
import pytesseract
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """OCR 결과"""
    text: str
    confidence: float
    bbox: Optional[List[float]] = None
    engine: str = "unknown"

class RealOCRSystem:
    """실제 작동하는 멀티 엔진 OCR 시스템"""
    
    def __init__(self):
        self.engines = {}
        self.initialized = False
        
    async def initialize(self):
        """OCR 엔진들 초기화"""
        try:
            logger.info("Initializing Real OCR System...")
            
            # 1. EasyOCR (한국어, 영어, 수학)
            try:
                self.engines['easyocr'] = easyocr.Reader(
                    ['ko', 'en'], 
                    gpu=torch.cuda.is_available()
                )
                logger.info("EasyOCR initialized")
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
            
            # 2. PaddleOCR (한국어, 영어)
            try:
                # PaddleOCR 파라미터 수정 (use_gpu 대신 gpu_id 사용)
                if torch.cuda.is_available():
                    self.engines['paddleocr'] = PaddleOCR(
                        use_angle_cls=True,
                        lang='korean',
                        gpu_id=0  # GPU ID 지정
                    )
                else:
                    self.engines['paddleocr'] = PaddleOCR(
                        use_angle_cls=True,
                        lang='korean',
                        gpu_id=-1  # CPU 사용
                    )
                logger.info("PaddleOCR initialized")
            except Exception as e:
                logger.warning(f"PaddleOCR initialization failed: {e}")
            
            # 3. Tesseract (백업용)
            try:
                # Tesseract 설치 확인
                pytesseract.get_tesseract_version()
                self.engines['tesseract'] = True
                logger.info("Tesseract initialized")
            except Exception as e:
                logger.warning(f"Tesseract initialization failed: {e}")
            
            self.initialized = True
            logger.info("Real OCR System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR system: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 이진화
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def detect_math_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """수식 영역 검출"""
        regions = []
        
        # 엣지 검출
        edges = cv2.Canny(image, 50, 150)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 수식 가능성 있는 영역 필터링
            if w > 20 and h > 10 and w/h < 10:
                regions.append((x, y, x+w, y+h))
        
        return regions
    
    async def ocr_with_easyocr(self, image: np.ndarray) -> List[OCRResult]:
        """EasyOCR로 텍스트 추출"""
        results = []
        
        if 'easyocr' not in self.engines:
            return results
        
        try:
            # EasyOCR 실행
            ocr_results = self.engines['easyocr'].readtext(image)
            
            for bbox, text, conf in ocr_results:
                results.append(OCRResult(
                    text=text,
                    confidence=conf,
                    bbox=bbox,
                    engine='easyocr'
                ))
                
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
        
        return results
    
    async def ocr_with_paddleocr(self, image: np.ndarray) -> List[OCRResult]:
        """PaddleOCR로 텍스트 추출"""
        results = []
        
        if 'paddleocr' not in self.engines:
            return results
        
        try:
            # PaddleOCR 실행
            ocr_results = self.engines['paddleocr'].ocr(image, cls=True)
            
            if ocr_results and ocr_results[0]:
                for line in ocr_results[0]:
                    if line:
                        bbox = line[0]
                        text, conf = line[1]
                        results.append(OCRResult(
                            text=text,
                            confidence=conf,
                            bbox=bbox,
                            engine='paddleocr'
                        ))
                        
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
        
        return results
    
    async def ocr_with_tesseract(self, image: np.ndarray) -> List[OCRResult]:
        """Tesseract로 텍스트 추출"""
        results = []
        
        if 'tesseract' not in self.engines:
            return results
        
        try:
            # Tesseract 실행 (한국어 + 영어)
            text = pytesseract.image_to_string(image, lang='kor+eng')
            
            if text.strip():
                results.append(OCRResult(
                    text=text.strip(),
                    confidence=0.7,  # Tesseract는 전체 신뢰도 제공 안함
                    engine='tesseract'
                ))
                
        except Exception as e:
            logger.error(f"Tesseract failed: {e}")
        
        return results
    
    def merge_ocr_results(self, all_results: List[List[OCRResult]]) -> str:
        """여러 OCR 결과 병합"""
        # 엔진별 결과 그룹화
        by_engine = {}
        for results in all_results:
            for result in results:
                if result.engine not in by_engine:
                    by_engine[result.engine] = []
                by_engine[result.engine].append(result)
        
        # 가장 신뢰도 높은 결과 선택
        best_text = ""
        best_confidence = 0
        
        for engine, results in by_engine.items():
            if results:
                # 엔진별 평균 신뢰도 계산
                avg_conf = sum(r.confidence for r in results) / len(results)
                
                # 텍스트 조합
                text = ' '.join(r.text for r in results)
                
                if avg_conf > best_confidence:
                    best_confidence = avg_conf
                    best_text = text
        
        return best_text
    
    def clean_ocr_text(self, text: str) -> str:
        """OCR 텍스트 정제"""
        # 원본 보존
        original = text
        
        # 전기공학 관련 패턴 보호
        protected_patterns = [
            # 단위
            r'\b\d+(?:\.\d+)?\s*(?:k|m|M|G)?(?:W|VA|V|A|Hz|Ω|F|H)\b',
            # 변수
            r'\b[VPIRQSZXY]\d*\b',
            # 수식
            r'[=+\-*/()^]',
            # 각도
            r'∠\s*\d+°?',
        ]
        
        # 보호할 부분 추출
        protected_parts = []
        for pattern in protected_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            protected_parts.extend(matches)
        
        # 노이즈 제거
        cleaned = text
        
        # 불필요한 공백 정리
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 특수문자 정리
        cleaned = re.sub(r'[^\w\s\d+\-*/=().,°∠]', ' ', cleaned)
        
        # 다시 공백 정리
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    async def process_image(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """이미지 처리 메인 함수"""
        try:
            # PIL Image를 numpy 배열로 변환
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # 전처리
            preprocessed = self.preprocess_image(img_array)
            
            # 모든 OCR 엔진 병렬 실행
            ocr_tasks = [
                self.ocr_with_easyocr(preprocessed),
                self.ocr_with_paddleocr(img_array),  # PaddleOCR는 원본 이미지 선호
                self.ocr_with_tesseract(preprocessed)
            ]
            
            all_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)
            
            # 예외 필터링
            valid_results = []
            for result in all_results:
                if isinstance(result, list):
                    valid_results.append(result)
                else:
                    logger.warning(f"OCR task failed: {result}")
            
            # 결과 병합
            merged_text = self.merge_ocr_results(valid_results)
            
            # 텍스트 정제
            cleaned_text = self.clean_ocr_text(merged_text)
            
            # 수식 영역 검출
            math_regions = self.detect_math_regions(preprocessed)
            
            # 각 엔진별 상세 결과
            engine_details = {}
            for results in valid_results:
                if results:
                    engine_name = results[0].engine
                    engine_details[engine_name] = {
                        'texts': [r.text for r in results],
                        'avg_confidence': sum(r.confidence for r in results) / len(results)
                    }
            
            return {
                'success': True,
                'text': cleaned_text,
                'raw_text': merged_text,
                'math_regions': len(math_regions),
                'engine_results': engine_details,
                'best_engine': max(engine_details.items(), 
                                 key=lambda x: x[1]['avg_confidence'])[0] if engine_details else None
            }
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'raw_text': ''
            }
    
    async def extract_formulas(self, image: Union[Image.Image, np.ndarray]) -> List[str]:
        """수식 추출 특화 함수"""
        result = await self.process_image(image)
        
        if not result['success']:
            return []
        
        text = result['text']
        
        # 수식 패턴 추출
        formula_patterns = [
            # 전력 공식
            r'[PSQ]\s*=\s*[^,\n]+',
            # 전압/전류 관계
            r'[VI]\s*=\s*[^,\n]+',
            # 임피던스
            r'Z\s*=\s*[^,\n]+',
            # 일반 수식
            r'\b\w+\s*=\s*[^,\n]+',
        ]
        
        formulas = []
        for pattern in formula_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            formulas.extend(matches)
        
        # 중복 제거
        return list(set(formulas))


class ElectricalOCRAnalyzer:
    """전기공학 특화 OCR 분석기"""
    
    def __init__(self, ocr_system: RealOCRSystem):
        self.ocr_system = ocr_system
        self.electrical_terms = self._load_electrical_terms()
        
    def _load_electrical_terms(self) -> Dict[str, List[str]]:
        """전기공학 용어 사전"""
        return {
            'power': ['전력', '유효전력', '무효전력', '피상전력', 'P', 'Q', 'S', 'kW', 'kVar', 'kVA'],
            'voltage': ['전압', 'V', '볼트', 'kV'],
            'current': ['전류', 'I', '암페어', 'A', 'mA'],
            'resistance': ['저항', 'R', '옴', 'Ω', 'ohm'],
            'impedance': ['임피던스', 'Z', '리액턴스', 'X'],
            'capacitance': ['커패시턴스', 'C', '콘덴서', 'F', 'μF'],
            'inductance': ['인덕턴스', 'L', '인덕터', 'H', 'mH'],
            'frequency': ['주파수', 'f', 'Hz', 'kHz'],
            'phase': ['위상', '각도', '∠', '°', 'θ', 'φ'],
            'power_factor': ['역률', 'cosθ', 'pf']
        }
    
    async def analyze_electrical_problem(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """전기공학 문제 분석"""
        # OCR 수행
        ocr_result = await self.ocr_system.process_image(image)
        
        if not ocr_result['success']:
            return {
                'success': False,
                'error': ocr_result.get('error', 'Unknown error'),
                'problem_type': 'unknown'
            }
        
        text = ocr_result['text']
        
        # 문제 유형 분류
        problem_type = self._classify_problem(text)
        
        # 주어진 값 추출
        given_values = self._extract_values(text)
        
        # 구해야 할 값 추출
        find_targets = self._extract_targets(text)
        
        # 수식 추출
        formulas = await self.ocr_system.extract_formulas(image)
        
        return {
            'success': True,
            'ocr_text': text,
            'problem_type': problem_type,
            'given_values': given_values,
            'find_targets': find_targets,
            'formulas': formulas,
            'engine_used': ocr_result.get('best_engine', 'unknown')
        }
    
    def _classify_problem(self, text: str) -> str:
        """문제 유형 분류"""
        text_lower = text.lower()
        
        # 키워드 기반 분류
        type_keywords = {
            'power_factor': ['역률', 'cosθ', '콘덴서', '개선'],
            'power_calculation': ['전력', 'kw', 'kvar', 'kva'],
            'circuit_analysis': ['회로', '전압', '전류', '저항'],
            'impedance': ['임피던스', '리액턴스', 'z'],
            'transformer': ['변압기', '변압비', '1차', '2차'],
            'motor': ['모터', '전동기', '회전', 'rpm']
        }
        
        scores = {}
        for ptype, keywords in type_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[ptype] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return 'general'
    
    def _extract_values(self, text: str) -> List[Dict[str, Any]]:
        """주어진 값 추출"""
        values = []
        
        # 숫자와 단위 패턴
        value_pattern = r'(\d+(?:\.\d+)?)\s*(?:(k|m|M|G)?([WVAHFΩ]|VA|Var|Hz|ohm))'
        
        matches = re.finditer(value_pattern, text, re.IGNORECASE)
        
        for match in matches:
            number = float(match.group(1))
            prefix = match.group(2) or ''
            unit = match.group(3)
            
            # 단위 변환
            if prefix.lower() == 'k':
                number *= 1000
            elif prefix.lower() == 'm':
                number *= 1000000
            elif prefix.lower() == 'g':
                number *= 1000000000
            
            values.append({
                'value': number,
                'unit': unit,
                'original': match.group(0)
            })
        
        return values
    
    def _extract_targets(self, text: str) -> List[str]:
        """구해야 할 값 추출"""
        targets = []
        
        # 물음표 패턴
        question_patterns = [
            r'(\w+)\s*[=?]+',
            r'구하(?:시오|라|세요)',
            r'계산하(?:시오|라|세요)',
            r'얼마인가',
            r'무엇인가'
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            targets.extend(matches)
        
        return list(set(targets))


# 사용 예시
async def test_ocr():
    """OCR 시스템 테스트"""
    # 시스템 초기화
    ocr_system = RealOCRSystem()
    await ocr_system.initialize()
    
    analyzer = ElectricalOCRAnalyzer(ocr_system)
    
    # 이미지 로드
    image = Image.open("test_image.png")
    
    # 분석
    result = await analyzer.analyze_electrical_problem(image)
    
    print("OCR 결과:", result['ocr_text'])
    print("문제 유형:", result['problem_type'])
    print("주어진 값:", result['given_values'])
    print("구할 값:", result['find_targets'])
    print("추출된 수식:", result['formulas'])

if __name__ == "__main__":
    asyncio.run(test_ocr())