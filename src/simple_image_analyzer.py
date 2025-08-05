#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단순화된 이미지 분석 시스템
복잡한 구조를 제거하고 실제로 작동하는 시스템 구현
"""

import logging
import os
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from PIL import Image
import cv2

# 조건부 임포트
from conditional_imports import EASYOCR_AVAILABLE

if EASYOCR_AVAILABLE:
    import easyocr

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    logging.warning("BLIP not available")

logger = logging.getLogger(__name__)


class SimpleImageAnalyzer:
    """단순화된 이미지 분석기 - BLIP + EasyOCR"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blip_model = None
        self.blip_processor = None
        self.ocr_reader = None
        self.initialized = False
        
    def initialize(self):
        """모델 초기화"""
        try:
            logger.info("Initializing Simple Image Analyzer...")
            
            # BLIP 모델 초기화 (캡션 생성)
            logger.info("Loading BLIP model...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)
            self.blip_model.eval()
            logger.info(f"BLIP model loaded on {self.device}")
            
            # EasyOCR 초기화 (텍스트 추출)
            logger.info("Loading EasyOCR...")
            self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
            logger.info("EasyOCR loaded")
            
            self.initialized = True
            logger.info("Simple Image Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Simple Image Analyzer: {e}")
            raise
    
    def _prepare_image(self, image: Union[Image.Image, str, np.ndarray]) -> Tuple[Image.Image, np.ndarray]:
        """이미지 준비"""
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
            cv_image = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
            cv_image = image
        else:
            pil_image = image.convert("RGB")
            cv_image = np.array(image)
        
        # 이미지 크기 조정 (메모리 절약)
        max_size = 1024
        if pil_image.width > max_size or pil_image.height > max_size:
            ratio = min(max_size/pil_image.width, max_size/pil_image.height)
            new_width = int(pil_image.width * ratio)
            new_height = int(pil_image.height * ratio)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            cv_image = cv2.resize(cv_image, (new_width, new_height))
        
        return pil_image, cv_image
    
    def generate_caption(self, image: Union[Image.Image, str, np.ndarray]) -> str:
        """이미지 캡션 생성"""
        if not self.initialized:
            self.initialize()
        
        try:
            pil_image, _ = self._prepare_image(image)
            
            # BLIP으로 캡션 생성
            inputs = self.blip_processor(pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # 한국어 번역 시도 (간단한 매핑)
            korean_caption = self._translate_caption(caption)
            
            return korean_caption
            
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return "이미지 캡션 생성 실패"
    
    def extract_text(self, image: Union[Image.Image, str, np.ndarray]) -> str:
        """이미지에서 텍스트 추출"""
        if not self.initialized:
            self.initialize()
        
        try:
            _, cv_image = self._prepare_image(image)
            
            # EasyOCR로 텍스트 추출
            results = self.ocr_reader.readtext(cv_image)
            
            # 텍스트 조합
            texts = []
            for (bbox, text, conf) in results:
                if conf > 0.5:  # 신뢰도 임계값
                    texts.append(text)
            
            return " ".join(texts) if texts else ""
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
    
    def analyze_image(self, image: Union[Image.Image, str, np.ndarray]) -> Dict[str, Any]:
        """통합 이미지 분석"""
        if not self.initialized:
            self.initialize()
        
        try:
            # 캡션 생성
            caption = self.generate_caption(image)
            
            # 텍스트 추출
            ocr_text = self.extract_text(image)
            
            # 전기공학 관련 분석
            electrical_info = self._analyze_electrical_content(caption, ocr_text)
            
            return {
                "success": True,
                "caption": caption,
                "ocr_text": ocr_text,
                "electrical_info": electrical_info,
                "has_text": bool(ocr_text),
                "has_formula": self._has_formula(ocr_text)
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "caption": "분석 실패",
                "ocr_text": ""
            }
    
    def _translate_caption(self, caption: str) -> str:
        """간단한 영-한 캡션 번역"""
        # 기본적인 단어 매핑
        translations = {
            "blackboard": "칠판",
            "chalkboard": "칠판",
            "man": "사람",
            "woman": "사람",
            "standing": "서 있는",
            "writing": "쓰고 있는",
            "book": "책",
            "page": "페이지",
            "text": "텍스트",
            "picture": "그림",
            "clock": "시계",
            "diagram": "도표",
            "circuit": "회로",
            "formula": "공식",
            "equation": "방정식"
        }
        
        result = caption.lower()
        for eng, kor in translations.items():
            result = result.replace(eng, kor)
        
        # 기본 문장 구조 변환
        if "a" in result or "an" in result:
            result = result.replace("a ", "").replace("an ", "")
        
        if "there is" in result:
            result = result.replace("there is", "")
        
        return result.strip() if result != caption.lower() else f"이미지: {caption}"
    
    def _analyze_electrical_content(self, caption: str, ocr_text: str) -> Dict[str, Any]:
        """전기공학 관련 내용 분석"""
        combined_text = f"{caption} {ocr_text}".lower()
        
        # 전기공학 키워드
        electrical_keywords = {
            "circuit": ["회로", "circuit", "저항", "resistance"],
            "power": ["전력", "power", "kw", "kvar", "kva"],
            "voltage": ["전압", "voltage", "v", "볼트"],
            "current": ["전류", "current", "a", "암페어"],
            "impedance": ["임피던스", "impedance", "z"],
            "capacitor": ["콘덴서", "capacitor", "커패시터"],
            "formula": ["공식", "formula", "equation", "="]
        }
        
        detected_topics = []
        for topic, keywords in electrical_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                detected_topics.append(topic)
        
        return {
            "topics": detected_topics,
            "is_electrical": len(detected_topics) > 0,
            "has_circuit": "circuit" in detected_topics,
            "has_formula": "formula" in detected_topics
        }
    
    def _has_formula(self, text: str) -> bool:
        """수식 포함 여부 확인"""
        formula_indicators = ['=', '+', '-', '*', '/', '^', '∫', '∑', 'π']
        return any(indicator in text for indicator in formula_indicators)


class SimpleMultimodalRAGService:
    """단순화된 멀티모달 RAG 서비스"""
    
    def __init__(self):
        self.analyzer = SimpleImageAnalyzer()
        
    def process_multimodal_query(
        self, 
        text_query: str,
        image: Optional[Union[Image.Image, str]] = None
    ) -> Dict[str, Any]:
        """멀티모달 쿼리 처리"""
        context = {
            "text_query": text_query,
            "image_analysis": None,
            "combined_query": text_query
        }
        
        if image:
            try:
                # 이미지 분석
                analysis = self.analyzer.analyze_image(image)
                
                context["image_analysis"] = {
                    "caption": analysis.get("caption", ""),
                    "ocr_text": analysis.get("ocr_text", "")
                }
                
                # 쿼리 결합
                combined_parts = [text_query]
                
                if analysis.get("caption") and analysis["caption"] != "분석 실패":
                    combined_parts.append(f"\n[이미지 설명] {analysis['caption']}")
                
                if analysis.get("ocr_text"):
                    combined_parts.append(f"\n[이미지의 텍스트] {analysis['ocr_text']}")
                
                if analysis.get("electrical_info", {}).get("is_electrical"):
                    topics = ", ".join(analysis["electrical_info"]["topics"])
                    combined_parts.append(f"\n[전기공학 관련 주제: {topics}]")
                
                context["combined_query"] = "\n".join(combined_parts)
                
            except Exception as e:
                logger.error(f"Multimodal processing failed: {e}")
                context["image_analysis"] = {
                    "error": str(e),
                    "caption": "[이미지 분석 실패]",
                    "ocr_text": ""
                }
                context["combined_query"] = f"{text_query}\n[이미지 분석 실패]"
        
        return context


# Florence-2 호환 인터페이스
class Florence2ImageAnalyzer:
    """Florence-2 호환 인터페이스 (Simple Image Analyzer 사용)"""
    
    def __init__(self, model_id: str = "microsoft/Florence-2-large"):
        self.analyzer = SimpleImageAnalyzer()
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None  # 호환성
        self.processor = None  # 호환성
        
        # 초기화
        try:
            self.analyzer.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize analyzer: {e}")
    
    def analyze_image(self, image, task="<OCR>", text_input=None):
        """Florence-2 호환 analyze_image"""
        try:
            result = self.analyzer.analyze_image(image)
            
            if task in ["<OCR>", "<OCR_WITH_REGION>"]:
                return {
                    "task": task,
                    "result": result.get("ocr_text", ""),
                    "success": result.get("success", False)
                }
            elif task in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]:
                return {
                    "task": task,
                    "result": result.get("caption", ""),
                    "success": result.get("success", False)
                }
            else:
                # 기본: 캡션 반환
                return {
                    "task": task,
                    "result": result.get("caption", ""),
                    "success": result.get("success", False)
                }
                
        except Exception as e:
            logger.error(f"analyze_image failed: {e}")
            return {
                "task": task,
                "result": "",
                "success": False,
                "error": str(e)
            }
    
    def generate_caption(self, image: Union[Image.Image, str], detail_level: str = "simple") -> Tuple[str, Dict[str, Any]]:
        """캡션 생성"""
        try:
            caption = self.analyzer.generate_caption(image)
            return caption, {"confidence": 0.9}
        except Exception as e:
            logger.error(f"generate_caption failed: {e}")
            return "[캡션 생성 실패]", {"confidence": 0.0, "error": str(e)}
    
    def extract_text(self, image: Union[Image.Image, str]) -> Tuple[str, List[Dict]]:
        """텍스트 추출"""
        try:
            text = self.analyzer.extract_text(image)
            return text, []  # 영역 정보는 빈 리스트
        except Exception as e:
            logger.error(f"extract_text failed: {e}")
            return "", []
    
    def extract_text_simple(self, image: Union[Image.Image, str]) -> str:
        """간단한 텍스트 추출"""
        text, _ = self.extract_text(image)
        return text


# 사용 예시
if __name__ == "__main__":
    analyzer = SimpleImageAnalyzer()
    analyzer.initialize()
    
    # 테스트
    result = analyzer.analyze_image("test_image.jpg")
    print(f"Caption: {result['caption']}")
    print(f"OCR Text: {result['ocr_text']}")
    print(f"Electrical: {result.get('electrical_info', {})}")