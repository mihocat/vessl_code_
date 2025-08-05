#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Engine OCR System
다중 OCR 엔진을 활용한 향상된 텍스트 추출 시스템
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import numpy as np
import cv2
from collections import Counter
import re

# OCR 엔진들
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR not available")

logger = logging.getLogger(__name__)


class OCREngine:
    """OCR 엔진 기본 클래스"""
    
    def extract_text(self, image: Image.Image) -> Dict[str, Any]:
        """텍스트 추출"""
        raise NotImplementedError
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """이미지 전처리"""
        # PIL Image를 OpenCV 형식으로 변환
        img_array = np.array(image)
        
        # 그레이스케일 변환
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 이진화
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary


class TesseractEngine(OCREngine):
    """Tesseract OCR 엔진"""
    
    def __init__(self):
        if not TESSERACT_AVAILABLE:
            raise ImportError("Tesseract is not available")
        
        # 한국어 + 영어 설정
        self.lang = 'kor+eng'
        
        # Tesseract 설정
        self.custom_config = r'--oem 3 --psm 3'
    
    def extract_text(self, image: Image.Image) -> Dict[str, Any]:
        """Tesseract를 사용한 텍스트 추출"""
        try:
            # 이미지 전처리
            processed = self.preprocess_image(image)
            
            # OCR 수행
            text = pytesseract.image_to_string(
                processed,
                lang=self.lang,
                config=self.custom_config
            )
            
            # 상세 정보 추출
            data = pytesseract.image_to_data(
                processed,
                lang=self.lang,
                config=self.custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # 신뢰도 계산
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': text.strip(),
                'confidence': avg_confidence / 100,
                'details': data,
                'engine': 'tesseract'
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {
                'text': '',
                'confidence': 0,
                'error': str(e),
                'engine': 'tesseract'
            }


class EasyOCREngine(OCREngine):
    """EasyOCR 엔진"""
    
    def __init__(self):
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR is not available")
        
        # GPU 사용 여부 자동 감지
        import torch
        gpu = torch.cuda.is_available()
        
        # 한국어 + 영어 리더 초기화
        self.reader = easyocr.Reader(['ko', 'en'], gpu=gpu)
    
    def extract_text(self, image: Image.Image) -> Dict[str, Any]:
        """EasyOCR을 사용한 텍스트 추출"""
        try:
            # PIL Image를 numpy array로 변환
            img_array = np.array(image)
            
            # OCR 수행
            results = self.reader.readtext(img_array, detail=1)
            
            # 텍스트와 신뢰도 추출
            texts = []
            confidences = []
            
            for bbox, text, confidence in results:
                texts.append(text)
                confidences.append(confidence)
            
            # 전체 텍스트 조합
            full_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': full_text.strip(),
                'confidence': avg_confidence,
                'details': results,
                'engine': 'easyocr'
            }
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return {
                'text': '',
                'confidence': 0,
                'error': str(e),
                'engine': 'easyocr'
            }


class PaddleOCREngine(OCREngine):
    """PaddleOCR 엔진"""
    
    def __init__(self):
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR is not available")
        
        # PaddleOCR 초기화 (한국어 지원)
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='korean',
            show_log=False
        )
    
    def extract_text(self, image: Image.Image) -> Dict[str, Any]:
        """PaddleOCR을 사용한 텍스트 추출"""
        try:
            # PIL Image를 numpy array로 변환
            img_array = np.array(image)
            
            # OCR 수행
            results = self.ocr.ocr(img_array, cls=True)
            
            # 결과가 없는 경우
            if not results or not results[0]:
                return {
                    'text': '',
                    'confidence': 0,
                    'details': [],
                    'engine': 'paddleocr'
                }
            
            # 텍스트와 신뢰도 추출
            texts = []
            confidences = []
            
            for line in results[0]:
                if len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]
                    texts.append(text)
                    confidences.append(confidence)
            
            # 전체 텍스트 조합
            full_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': full_text.strip(),
                'confidence': avg_confidence,
                'details': results[0],
                'engine': 'paddleocr'
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return {
                'text': '',
                'confidence': 0,
                'error': str(e),
                'engine': 'paddleocr'
            }


class MultiEngineOCR:
    """다중 OCR 엔진 통합 시스템"""
    
    def __init__(self, engines: Optional[List[str]] = None):
        """
        초기화
        
        Args:
            engines: 사용할 엔진 리스트 (None이면 모든 사용 가능한 엔진)
        """
        self.engines = {}
        
        # 사용 가능한 엔진 확인 및 초기화
        available_engines = {
            'tesseract': (TESSERACT_AVAILABLE, TesseractEngine),
            'easyocr': (EASYOCR_AVAILABLE, EasyOCREngine),
            'paddleocr': (PADDLEOCR_AVAILABLE, PaddleOCREngine)
        }
        
        # 지정된 엔진만 사용하거나 모든 사용 가능한 엔진 사용
        for name, (available, engine_class) in available_engines.items():
            if engines and name not in engines:
                continue
                
            if available:
                try:
                    self.engines[name] = engine_class()
                    logger.info(f"Initialized {name} engine")
                except Exception as e:
                    logger.warning(f"Failed to initialize {name}: {e}")
        
        if not self.engines:
            raise RuntimeError("No OCR engines available")
        
        logger.info(f"MultiEngineOCR initialized with engines: {list(self.engines.keys())}")
    
    def extract_text(self, image: Image.Image, ensemble_method: str = 'voting') -> Dict[str, Any]:
        """
        여러 OCR 엔진을 사용한 텍스트 추출
        
        Args:
            image: 입력 이미지
            ensemble_method: 앙상블 방법 ('voting', 'confidence', 'all')
            
        Returns:
            통합된 OCR 결과
        """
        results = {}
        
        # 각 엔진으로 OCR 수행
        for name, engine in self.engines.items():
            logger.info(f"Running {name} OCR...")
            result = engine.extract_text(image)
            results[name] = result
        
        # 앙상블 처리
        if ensemble_method == 'voting':
            return self._ensemble_voting(results)
        elif ensemble_method == 'confidence':
            return self._ensemble_confidence(results)
        else:  # 'all'
            return self._ensemble_all(results)
    
    def _ensemble_voting(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """투표 기반 앙상블"""
        # 각 엔진의 텍스트를 단어 단위로 분리
        word_votes = Counter()
        
        for engine_name, result in results.items():
            if result.get('text'):
                words = self._tokenize(result['text'])
                for word in words:
                    word_votes[word] += result.get('confidence', 0.5)
        
        # 가장 많은 투표를 받은 단어들로 텍스트 재구성
        if word_votes:
            # 신뢰도 가중치가 높은 순으로 정렬
            sorted_words = sorted(word_votes.items(), key=lambda x: x[1], reverse=True)
            
            # 원본 순서를 유지하면서 재구성
            final_text = self._reconstruct_text(sorted_words, results)
            avg_confidence = sum(r.get('confidence', 0) for r in results.values()) / len(results)
            
            return {
                'text': final_text,
                'confidence': avg_confidence,
                'method': 'voting',
                'engines_used': list(results.keys()),
                'individual_results': results
            }
        
        return {
            'text': '',
            'confidence': 0,
            'method': 'voting',
            'engines_used': list(results.keys()),
            'individual_results': results
        }
    
    def _ensemble_confidence(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """신뢰도 기반 앙상블"""
        # 가장 높은 신뢰도를 가진 결과 선택
        best_result = None
        best_confidence = -1
        best_engine = None
        
        for engine_name, result in results.items():
            confidence = result.get('confidence', 0)
            if confidence > best_confidence and result.get('text'):
                best_confidence = confidence
                best_result = result
                best_engine = engine_name
        
        if best_result:
            return {
                'text': best_result['text'],
                'confidence': best_confidence,
                'method': 'confidence',
                'selected_engine': best_engine,
                'engines_used': list(results.keys()),
                'individual_results': results
            }
        
        return {
            'text': '',
            'confidence': 0,
            'method': 'confidence',
            'engines_used': list(results.keys()),
            'individual_results': results
        }
    
    def _ensemble_all(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """모든 결과 반환"""
        combined_texts = []
        
        for engine_name, result in results.items():
            if result.get('text'):
                combined_texts.append(f"[{engine_name}] {result['text']}")
        
        avg_confidence = sum(r.get('confidence', 0) for r in results.values()) / len(results)
        
        return {
            'text': '\n'.join(combined_texts),
            'confidence': avg_confidence,
            'method': 'all',
            'engines_used': list(results.keys()),
            'individual_results': results
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰(단어)으로 분리"""
        # 한국어와 영어를 모두 처리
        # 공백, 구두점 등으로 분리
        tokens = re.findall(r'[\w가-힣]+|[^\w\s가-힣]+', text)
        return [t.strip() for t in tokens if t.strip()]
    
    def _reconstruct_text(self, word_votes: List[Tuple[str, float]], results: Dict[str, Dict]) -> str:
        """투표 결과를 바탕으로 텍스트 재구성"""
        # 가장 신뢰도가 높은 원본 텍스트를 기준으로 사용
        base_result = self._ensemble_confidence(results)
        
        if base_result.get('text'):
            # 기준 텍스트에서 낮은 신뢰도 단어를 높은 신뢰도 단어로 교체
            # 간단한 구현을 위해 기준 텍스트 그대로 반환
            return base_result['text']
        
        # 기준 텍스트가 없으면 투표 결과 단어들을 조합
        words = [word for word, _ in word_votes]
        return ' '.join(words[:len(words)//2])  # 상위 50% 단어만 사용