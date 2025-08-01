#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Image Analyzer with Multimodal OCR
멀티모달 OCR을 통합한 향상된 이미지 분석기
"""

import logging
from typing import Dict, Any, Optional, Union
from PIL import Image
import torch

# 기존 Florence-2 분석기
from image_analyzer import Florence2ImageAnalyzer

# 새로운 멀티모달 OCR
from multimodal_ocr import MultimodalOCRPipeline

logger = logging.getLogger(__name__)


class EnhancedImageAnalyzer:
    """향상된 이미지 분석기 - Florence-2 + Multimodal OCR"""
    
    def __init__(self, use_florence: bool = True):
        """
        향상된 이미지 분석기 초기화
        
        Args:
            use_florence: Florence-2 사용 여부
        """
        self.use_florence = use_florence
        self.florence_analyzer = None
        self.ocr_pipeline = None
        
        # Florence-2 초기화
        if use_florence:
            try:
                self.florence_analyzer = Florence2ImageAnalyzer()
                logger.info("Florence-2 analyzer initialized")
            except Exception as e:
                logger.warning(f"Florence-2 initialization failed: {e}")
                self.florence_analyzer = None
        
        # 멀티모달 OCR 초기화
        try:
            self.ocr_pipeline = MultimodalOCRPipeline()
            logger.info("Multimodal OCR pipeline initialized")
        except Exception as e:
            logger.error(f"Multimodal OCR initialization failed: {e}")
            self.ocr_pipeline = None
    
    def analyze_image(
        self, 
        image: Union[Image.Image, str, bytes],
        extract_mode: str = 'all'
    ) -> Dict[str, Any]:
        """
        통합 이미지 분석
        
        Args:
            image: 분석할 이미지
            extract_mode: 추출 모드 ('all', 'text', 'formula', 'diagram')
            
        Returns:
            분석 결과
        """
        results = {
            'success': False,
            'caption': '',
            'ocr_text': '',
            'formulas': [],
            'diagrams': [],
            'structured_content': {},
            'error': None
        }
        
        try:
            # 1. Florence-2로 전체 캡션 생성 (선택적)
            if self.use_florence and self.florence_analyzer:
                try:
                    florence_result = self.florence_analyzer.generate_caption(
                        image, 
                        detail_level='detailed'
                    )
                    results['caption'] = florence_result
                except Exception as e:
                    logger.warning(f"Florence-2 analysis failed: {e}")
            
            # 2. 멀티모달 OCR로 상세 분석
            if self.ocr_pipeline:
                ocr_results = self.ocr_pipeline.process_image(image)
                
                # OCR 텍스트 통합
                results['ocr_text'] = ' '.join(ocr_results['full_text'])
                results['formulas'] = ocr_results['formulas']
                results['diagrams'] = ocr_results['diagrams']
                results['structured_content'] = ocr_results['structured_content']
                
                # 문제/풀이 분리 (전기공학 문제인 경우)
                if extract_mode == 'question_solution':
                    qs_results = self.ocr_pipeline.extract_question_and_solution(image)
                    results['question'] = qs_results['question']
                    results['solution'] = qs_results['solution']
            
            results['success'] = True
            
        except Exception as e:
            logger.error(f"Enhanced image analysis failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def extract_text(self, image: Union[Image.Image, str, bytes]) -> str:
        """이미지에서 텍스트만 추출"""
        if self.ocr_pipeline:
            try:
                results = self.ocr_pipeline.process_image(image)
                return ' '.join(results['full_text'])
            except Exception as e:
                logger.error(f"Text extraction failed: {e}")
        
        # 폴백: Florence-2 사용
        if self.florence_analyzer:
            return self.florence_analyzer.extract_text(image)
        
        return ""
    
    def extract_formulas(self, image: Union[Image.Image, str, bytes]) -> list:
        """이미지에서 수식만 추출"""
        if self.ocr_pipeline:
            try:
                results = self.ocr_pipeline.process_image(image)
                return results['formulas']
            except Exception as e:
                logger.error(f"Formula extraction failed: {e}")
        
        return []
    
    def analyze_electrical_problem(self, image: Union[Image.Image, str, bytes]) -> Dict[str, Any]:
        """전기공학 문제 특화 분석"""
        results = self.analyze_image(image, extract_mode='question_solution')
        
        # 전기공학 특화 후처리
        if results['success']:
            # 전기 단위 정규화
            results['normalized_units'] = self._normalize_electrical_units(results['ocr_text'])
            
            # 회로 컴포넌트 추출
            if results['diagrams']:
                results['circuit_components'] = self._extract_circuit_components(results['diagrams'])
        
        return results
    
    def _normalize_electrical_units(self, text: str) -> str:
        """전기 단위 정규화"""
        import re
        
        # 단위 변환 규칙
        unit_mappings = {
            r'(\d+)\s*k\s*W': r'\1 kW',
            r'(\d+)\s*k\s*VA': r'\1 kVA',
            r'(\d+)\s*m\s*A': r'\1 mA',
            r'(\d+)\s*μ\s*F': r'\1 μF',
            r'(\d+)\s*Ω': r'\1 Ω',
            r'(\d+)\s*ohm': r'\1 Ω',
        }
        
        normalized = text
        for pattern, replacement in unit_mappings.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized
    
    def _extract_circuit_components(self, diagrams: list) -> list:
        """회로 컴포넌트 추출"""
        components = []
        
        for diagram in diagrams:
            if isinstance(diagram, dict) and 'components' in diagram:
                components.extend(diagram['components'])
        
        return components


class ChatGPTStyleAnalyzer(EnhancedImageAnalyzer):
    """ChatGPT 스타일 응답을 위한 특화 분석기"""
    
    def analyze_for_chatgpt_response(self, image: Union[Image.Image, str, bytes]) -> Dict[str, Any]:
        """ChatGPT 스타일 응답을 위한 분석"""
        # 기본 분석
        results = self.analyze_electrical_problem(image)
        
        # ChatGPT 스타일을 위한 추가 정보
        if results['success']:
            # 핵심 개념 추출
            results['key_concepts'] = self._extract_key_concepts(results)
            
            # 단계별 접근법 준비
            results['solution_steps'] = self._prepare_solution_steps(results)
            
            # 시각적 요소 준비
            results['visual_elements'] = self._prepare_visual_elements(results)
        
        return results
    
    def _extract_key_concepts(self, results: Dict[str, Any]) -> list:
        """핵심 개념 추출"""
        concepts = []
        
        # 텍스트에서 전기공학 키워드 추출
        keywords = [
            '전압', '전류', '저항', '임피던스', '전력', '역률',
            '회로', '변압기', '모터', '발전기', '송전', '배전'
        ]
        
        text = results.get('ocr_text', '')
        for keyword in keywords:
            if keyword in text:
                concepts.append(keyword)
        
        return concepts
    
    def _prepare_solution_steps(self, results: Dict[str, Any]) -> list:
        """해결 단계 준비"""
        steps = []
        
        if 'question' in results and 'solution' in results:
            # 문제 이해 단계
            steps.append({
                'step': 1,
                'title': '문제 이해',
                'content': results['question']['text'],
                'formulas': results['question']['formulas']
            })
            
            # 풀이 단계 (수식별로 분리)
            solution_formulas = results['solution']['formulas']
            for i, formula in enumerate(solution_formulas):
                steps.append({
                    'step': i + 2,
                    'title': f'계산 단계 {i + 1}',
                    'content': '',
                    'formulas': [formula]
                })
        
        return steps
    
    def _prepare_visual_elements(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """시각적 요소 준비"""
        visual = {
            'has_circuit': len(results.get('diagrams', [])) > 0,
            'has_formulas': len(results.get('formulas', [])) > 0,
            'needs_table': False,
            'needs_graph': False
        }
        
        # 표가 필요한지 판단
        if '비교' in results.get('ocr_text', '') or '차이' in results.get('ocr_text', ''):
            visual['needs_table'] = True
        
        # 그래프가 필요한지 판단
        if '그래프' in results.get('ocr_text', '') or '변화' in results.get('ocr_text', ''):
            visual['needs_graph'] = True
        
        return visual


# 기존 시스템과의 호환성을 위한 어댑터
class ImageAnalyzerAdapter:
    """기존 Florence2ImageAnalyzer 인터페이스와 호환되는 어댑터"""
    
    def __init__(self):
        self.analyzer = ChatGPTStyleAnalyzer(use_florence=True)
    
    def analyze_image(self, image, task="<CAPTION>", text_input=None):
        """기존 인터페이스와 호환"""
        results = self.analyzer.analyze_image(image)
        
        # 기존 형식으로 변환
        if task == "<CAPTION>":
            return {
                "success": results['success'],
                "task": task,
                "result": results['caption']
            }
        elif task == "<OCR>":
            return {
                "success": results['success'],
                "task": task,
                "result": results['ocr_text']
            }
        else:
            return {
                "success": results['success'],
                "task": task,
                "result": results
            }
    
    def extract_text(self, image):
        """기존 인터페이스와 호환"""
        return self.analyzer.extract_text(image)
    
    def generate_caption(self, image, detail_level="simple"):
        """기존 인터페이스와 호환"""
        results = self.analyzer.analyze_image(image)
        return results['caption']


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    analyzer = ChatGPTStyleAnalyzer()
    
    # 테스트 이미지 경로
    test_image = "test_electrical_problem.png"
    
    try:
        results = analyzer.analyze_for_chatgpt_response(test_image)
        
        print("=== Analysis Results ===")
        print(f"Success: {results['success']}")
        print(f"Caption: {results['caption'][:100]}...")
        print(f"OCR Text: {results['ocr_text'][:100]}...")
        print(f"Formulas: {len(results['formulas'])}")
        print(f"Key Concepts: {results['key_concepts']}")
        print(f"Solution Steps: {len(results['solution_steps'])}")
        
    except Exception as e:
        print(f"Test failed: {e}")