import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import io
import base64
from dataclasses import dataclass
from enum import Enum

from math_ocr_system import MathOCRSystem, MathExpression

logger = logging.getLogger(__name__)

class ImageRegionType(Enum):
    """이미지 영역 타입"""
    TEXT = "text"
    FORMULA = "formula"
    DIAGRAM = "diagram"
    CIRCUIT = "circuit"
    GRAPH = "graph"
    TABLE = "table"

@dataclass
class ImageRegion:
    """이미지 영역 정보"""
    type: ImageRegionType
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    content: Optional[Any] = None

class EnhancedImageAnalyzer:
    """향상된 이미지 분석 시스템"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.florence_processor = None
        self.florence_model = None
        self.math_ocr = MathOCRSystem()
        self.region_detector = None
        
    async def initialize(self):
        """시스템 초기화"""
        try:
            # Florence-2 모델 로드 (기본 이미지 이해용)
            logger.info("Loading Florence-2 model...")
            self.florence_processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-large", trust_remote_code=True
            )
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large", 
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).to(self.device)
            
            # Math OCR 시스템 초기화
            await self.math_ocr.initialize()
            
            # 영역 검출기 초기화
            self._initialize_region_detector()
            
            logger.info("Enhanced Image Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Image Analyzer: {e}")
            raise
    
    def _initialize_region_detector(self):
        """영역 검출기 초기화"""
        # OpenCV 기반 영역 검출
        self.region_detector = {
            'text_cascade': cv2.CascadeClassifier(),
            'mser': cv2.MSER_create(),
            'contour_params': {
                'min_area': 100,
                'max_area': 0.5,  # 이미지 크기의 50%
                'aspect_ratio_range': (0.1, 10)
            }
        }
    
    async def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """이미지 분석 메인 함수"""
        try:
            # 1. 이미지 영역 검출
            regions = await self._detect_regions(image)
            
            # 2. 각 영역별 처리
            region_results = await self._process_regions(image, regions)
            
            # 3. 전체 이미지 캡션 생성
            caption = await self._generate_caption(image)
            
            # 4. 결과 통합
            integrated_result = await self._integrate_analysis(
                caption, regions, region_results
            )
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"Error in analyze_image: {e}")
            return {'error': str(e)}
    
    async def _detect_regions(self, image: Image.Image) -> List[ImageRegion]:
        """이미지에서 영역 검출"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        regions = []
        
        # 1. MSER을 사용한 텍스트 영역 검출
        mser_regions = self.region_detector['mser'].detectRegions(gray)
        for region in mser_regions[0]:
            x, y, w, h = cv2.boundingRect(region)
            if self._is_valid_region(w, h, gray.shape):
                regions.append(ImageRegion(
                    type=ImageRegionType.TEXT,
                    bbox=(x, y, x+w, y+h),
                    confidence=0.7
                ))
        
        # 2. 엣지 검출을 통한 다이어그램/회로도 검출
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if self._is_valid_region(w, h, gray.shape):
                # 회로도 패턴 검사
                if self._is_circuit_pattern(gray[y:y+h, x:x+w]):
                    region_type = ImageRegionType.CIRCUIT
                else:
                    region_type = ImageRegionType.DIAGRAM
                
                regions.append(ImageRegion(
                    type=region_type,
                    bbox=(x, y, x+w, y+h),
                    confidence=0.6
                ))
        
        # 3. 영역 병합 및 중복 제거
        regions = self._merge_overlapping_regions(regions)
        
        return regions
    
    def _is_valid_region(self, width: int, height: int, image_shape: Tuple) -> bool:
        """유효한 영역인지 검사"""
        params = self.region_detector['contour_params']
        area = width * height
        image_area = image_shape[0] * image_shape[1]
        aspect_ratio = width / height if height > 0 else 0
        
        return (
            area >= params['min_area'] and
            area <= params['max_area'] * image_area and
            params['aspect_ratio_range'][0] <= aspect_ratio <= params['aspect_ratio_range'][1]
        )
    
    def _is_circuit_pattern(self, region: np.ndarray) -> bool:
        """회로도 패턴인지 검사"""
        # 직선과 연결점이 많은지 검사
        edges = cv2.Canny(region, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10
        )
        
        return lines is not None and len(lines) > 5
    
    def _merge_overlapping_regions(self, regions: List[ImageRegion]) -> List[ImageRegion]:
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
                    region1.confidence = max(region1.confidence, region2.confidence)
                    used.add(j)
            
            merged.append(region1)
        
        return merged
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """IoU (Intersection over Union) 계산"""
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
    
    async def _process_regions(
        self, image: Image.Image, regions: List[ImageRegion]
    ) -> Dict[str, Any]:
        """각 영역 처리"""
        results = {
            'text_regions': [],
            'formula_regions': [],
            'diagram_regions': [],
            'circuit_regions': []
        }
        
        for region in regions:
            # 영역 추출
            x1, y1, x2, y2 = region.bbox
            region_image = image.crop((x1, y1, x2, y2))
            
            if region.type in [ImageRegionType.TEXT, ImageRegionType.FORMULA]:
                # 수식 OCR 처리
                math_result = await self.math_ocr.process_image(region_image)
                
                if math_result.get('formulas'):
                    results['formula_regions'].append({
                        'bbox': region.bbox,
                        'formulas': math_result['formulas'],
                        'context': math_result.get('electrical_context', '')
                    })
                else:
                    results['text_regions'].append({
                        'bbox': region.bbox,
                        'text': math_result.get('ocr_text', '')
                    })
            
            elif region.type == ImageRegionType.CIRCUIT:
                # 회로도 분석
                circuit_analysis = await self._analyze_circuit(region_image)
                results['circuit_regions'].append({
                    'bbox': region.bbox,
                    'analysis': circuit_analysis
                })
            
            elif region.type == ImageRegionType.DIAGRAM:
                # 다이어그램 분석
                diagram_analysis = await self._analyze_diagram(region_image)
                results['diagram_regions'].append({
                    'bbox': region.bbox,
                    'analysis': diagram_analysis
                })
        
        return results
    
    async def _analyze_circuit(self, image: Image.Image) -> Dict[str, Any]:
        """회로도 분석"""
        # 간단한 회로 요소 검출
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 회로 요소 템플릿 매칭 (실제로는 더 복잡한 모델 필요)
        components = {
            'resistors': 0,
            'capacitors': 0,
            'inductors': 0,
            'voltage_sources': 0,
            'current_sources': 0
        }
        
        # Florence-2로 간단한 설명 생성
        description = await self._get_florence_description(image, "circuit diagram")
        
        return {
            'components': components,
            'description': description
        }
    
    async def _analyze_diagram(self, image: Image.Image) -> Dict[str, Any]:
        """다이어그램 분석"""
        description = await self._get_florence_description(image, "technical diagram")
        
        return {
            'type': 'general_diagram',
            'description': description
        }
    
    async def _get_florence_description(self, image: Image.Image, prompt: str) -> str:
        """Florence-2를 사용한 설명 생성"""
        inputs = self.florence_processor(
            text=f"<CAPTION> {prompt}",
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.florence_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.7
            )
        
        generated_text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        
        parsed = self.florence_processor.post_process_generation(
            generated_text, 
            task="<CAPTION>",
            image_size=(image.width, image.height)
        )
        
        return parsed.get('<CAPTION>', '')
    
    async def _generate_caption(self, image: Image.Image) -> str:
        """전체 이미지 캡션 생성"""
        return await self._get_florence_description(image, "electrical engineering content")
    
    async def _integrate_analysis(
        self, caption: str, regions: List[ImageRegion], region_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """분석 결과 통합"""
        # 모든 수식 수집
        all_formulas = []
        for formula_region in region_results.get('formula_regions', []):
            all_formulas.extend(formula_region['formulas'])
        
        # 모든 텍스트 수집
        all_text = ' '.join([
            region['text'] for region in region_results.get('text_regions', [])
        ])
        
        # 전기공학 컨텍스트 생성
        electrical_context = self._generate_comprehensive_context(
            all_formulas, region_results
        )
        
        return {
            'caption': caption,
            'ocr_text': all_text,
            'formulas': all_formulas,
            'electrical_context': electrical_context,
            'regions': {
                'total': len(regions),
                'text': len(region_results.get('text_regions', [])),
                'formula': len(region_results.get('formula_regions', [])),
                'circuit': len(region_results.get('circuit_regions', [])),
                'diagram': len(region_results.get('diagram_regions', []))
            },
            'detailed_results': region_results
        }
    
    def _generate_comprehensive_context(
        self, formulas: List[Dict], region_results: Dict[str, Any]
    ) -> str:
        """종합적인 전기공학 컨텍스트 생성"""
        contexts = []
        
        # 수식 기반 컨텍스트
        formula_types = set(f['type'] for f in formulas if f.get('type'))
        if 'ohms_law' in formula_types:
            contexts.append("옴의 법칙 관련 문제")
        if 'power_vi' in formula_types or 'power_i2r' in formula_types:
            contexts.append("전력 계산 문제")
        
        # 회로도 기반 컨텍스트
        if region_results.get('circuit_regions'):
            contexts.append("회로 분석 문제")
        
        # 변수 기반 컨텍스트
        all_variables = {}
        for formula in formulas:
            if formula.get('variables'):
                all_variables.update(formula['variables'])
        
        if 'voltage' in all_variables and 'current' in all_variables:
            contexts.append("전압-전류 관계")
        if 'resistance' in all_variables:
            contexts.append("저항 관련")
        
        return ' | '.join(contexts) if contexts else "전기공학 문제"