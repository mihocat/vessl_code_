#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
새로운 이미지 분석 시스템
Florence-2를 대체하는 실제 작동하는 시스템
"""

import logging
from typing import Dict, Any, Optional, List, Union
from PIL import Image
import numpy as np
import asyncio
from real_ocr_system import RealOCRSystem, ElectricalOCRAnalyzer

logger = logging.getLogger(__name__)


class NewImageAnalyzer:
    """새로운 이미지 분석기 - Florence-2 대체"""
    
    def __init__(self):
        self.ocr_system = None
        self.analyzer = None
        self.initialized = False
        
    async def initialize(self):
        """시스템 초기화"""
        try:
            logger.info("Initializing New Image Analyzer...")
            
            # OCR 시스템 초기화
            self.ocr_system = RealOCRSystem()
            await self.ocr_system.initialize()
            
            # 전기공학 분석기 초기화
            self.analyzer = ElectricalOCRAnalyzer(self.ocr_system)
            
            self.initialized = True
            logger.info("New Image Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize New Image Analyzer: {e}")
            raise
    
    def analyze_image(
        self, 
        image: Union[Image.Image, str, bytes],
        task: str = "<OCR>",
        text_input: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Florence-2 인터페이스와 호환되는 이미지 분석
        
        Args:
            image: 분석할 이미지
            task: 작업 유형 (호환성을 위해 유지)
            text_input: 추가 텍스트 입력
            
        Returns:
            분석 결과
        """
        # 동기 래퍼
        return asyncio.run(self._analyze_image_async(image, task, text_input))
    
    async def _analyze_image_async(
        self,
        image: Union[Image.Image, str, bytes],
        task: str,
        text_input: Optional[str]
    ) -> Dict[str, Any]:
        """비동기 이미지 분석"""
        
        if not self.initialized:
            await self.initialize()
        
        try:
            # 이미지 로드
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, bytes):
                import io
                image = Image.open(io.BytesIO(image)).convert("RGB")
            
            # 작업별 처리
            if task in ["<OCR>", "<OCR_WITH_REGION>"]:
                # OCR 수행
                result = await self.ocr_system.process_image(image)
                
                if result['success']:
                    return {
                        "task": task,
                        "result": result['text'],
                        "success": True,
                        "details": result
                    }
                else:
                    return {
                        "task": task,
                        "result": "",
                        "success": False,
                        "error": result.get('error', 'OCR failed')
                    }
            
            elif task in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]:
                # 캡션 생성 - OCR 기반 설명
                analysis = await self.analyzer.analyze_electrical_problem(image)
                
                if analysis['success']:
                    caption = self._generate_caption_from_analysis(analysis, task)
                    return {
                        "task": task,
                        "result": caption,
                        "success": True
                    }
                else:
                    return {
                        "task": task,
                        "result": "이미지 분석 실패",
                        "success": False
                    }
            
            else:
                # 지원하지 않는 작업
                return {
                    "task": task,
                    "result": f"Task {task} not supported",
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                "task": task,
                "result": "",
                "success": False,
                "error": str(e)
            }
    
    def extract_text(self, image: Union[Image.Image, str, bytes]) -> str:
        """
        이미지에서 텍스트 추출 (Florence-2 호환 인터페이스)
        
        Args:
            image: 분석할 이미지
            
        Returns:
            추출된 텍스트
        """
        result = self.analyze_image(image, task="<OCR>")
        if result["success"]:
            return result["result"]
        return ""
    
    def generate_caption(
        self, 
        image: Union[Image.Image, str, bytes],
        detail_level: str = "simple"
    ) -> str:
        """
        이미지 캡션 생성 (Florence-2 호환 인터페이스)
        
        Args:
            image: 분석할 이미지
            detail_level: 상세도 수준
            
        Returns:
            생성된 캡션
        """
        task_map = {
            "simple": "<CAPTION>",
            "detailed": "<DETAILED_CAPTION>",
            "very_detailed": "<MORE_DETAILED_CAPTION>"
        }
        
        task = task_map.get(detail_level, "<CAPTION>")
        result = self.analyze_image(image, task=task)
        
        if result["success"]:
            return result["result"]
        return "이미지 분석에 실패했습니다."
    
    def _generate_caption_from_analysis(
        self, 
        analysis: Dict[str, Any], 
        task: str
    ) -> str:
        """분석 결과로부터 캡션 생성"""
        
        # 기본 정보
        problem_type = analysis.get('problem_type', 'unknown')
        ocr_text = analysis.get('ocr_text', '')
        given_values = analysis.get('given_values', [])
        formulas = analysis.get('formulas', [])
        
        # 상세도에 따른 캡션 생성
        if task == "<CAPTION>":
            # 간단한 캡션
            if problem_type == 'power_factor':
                caption = "역률 관련 전기공학 문제"
            elif problem_type == 'power_calculation':
                caption = "전력 계산 문제"
            elif problem_type == 'circuit_analysis':
                caption = "회로 해석 문제"
            else:
                caption = "전기공학 문제"
            
            if given_values:
                caption += f" ({len(given_values)}개의 주어진 값 포함)"
                
        elif task == "<DETAILED_CAPTION>":
            # 상세한 캡션
            caption = f"{self._get_problem_type_korean(problem_type)} 문제입니다. "
            
            if given_values:
                values_str = ", ".join([f"{v['value']}{v['unit']}" for v in given_values[:3]])
                caption += f"주어진 값: {values_str}"
                if len(given_values) > 3:
                    caption += f" 외 {len(given_values)-3}개. "
                else:
                    caption += ". "
            
            if formulas:
                caption += f"{len(formulas)}개의 수식이 포함되어 있습니다."
                
        else:
            # 매우 상세한 캡션
            caption = f"이 이미지는 {self._get_problem_type_korean(problem_type)} 문제를 담고 있습니다. "
            
            if ocr_text:
                caption += f"인식된 텍스트: '{ocr_text[:100]}{'...' if len(ocr_text) > 100 else ''}'. "
            
            if given_values:
                caption += "주어진 값들: "
                for v in given_values:
                    caption += f"{v['value']}{v['unit']}, "
                caption = caption.rstrip(", ") + ". "
            
            if formulas:
                caption += "포함된 수식: "
                for f in formulas[:3]:
                    caption += f"{f}, "
                if len(formulas) > 3:
                    caption += f"외 {len(formulas)-3}개. "
                else:
                    caption = caption.rstrip(", ") + ". "
        
        return caption
    
    def _get_problem_type_korean(self, problem_type: str) -> str:
        """문제 유형 한글 변환"""
        type_map = {
            'power_factor': '역률 개선',
            'power_calculation': '전력 계산',
            'circuit_analysis': '회로 해석',
            'impedance': '임피던스 계산',
            'transformer': '변압기',
            'motor': '전동기',
            'general': '일반 전기공학'
        }
        return type_map.get(problem_type, '전기공학')
    
    def detect_objects(
        self, 
        image: Union[Image.Image, str, bytes],
        object_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """객체 검출 (더미 구현)"""
        # Florence-2 호환성을 위한 더미 구현
        return []
    
    def analyze_formula(self, image: Union[Image.Image, str, bytes]) -> Dict[str, Any]:
        """수식 분석"""
        return asyncio.run(self._analyze_formula_async(image))
    
    async def _analyze_formula_async(self, image: Union[Image.Image, str, bytes]) -> Dict[str, Any]:
        """비동기 수식 분석"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # 이미지 로드
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, bytes):
                import io
                image = Image.open(io.BytesIO(image)).convert("RGB")
            
            # 전기공학 문제 분석
            analysis = await self.analyzer.analyze_electrical_problem(image)
            
            if analysis['success']:
                return {
                    "formula_text": ' '.join(analysis.get('formulas', [])),
                    "description": self._generate_caption_from_analysis(analysis, "<DETAILED_CAPTION>"),
                    "regions": [],  # 호환성을 위해 유지
                    "success": True,
                    "details": analysis
                }
            else:
                return {
                    "formula_text": "",
                    "description": "수식 분석 실패",
                    "regions": [],
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Formula analysis failed: {e}")
            return {
                "formula_text": "",
                "description": f"오류: {str(e)}",
                "regions": [],
                "success": False
            }
    
    def batch_analyze(
        self, 
        images: List[Union[Image.Image, str, bytes]],
        task: str = "<OCR>"
    ) -> List[Dict[str, Any]]:
        """여러 이미지 일괄 분석"""
        results = []
        for image in images:
            result = self.analyze_image(image, task=task)
            results.append(result)
        return results


# Florence2ImageAnalyzer를 대체
Florence2ImageAnalyzer = NewImageAnalyzer