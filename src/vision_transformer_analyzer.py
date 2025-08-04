#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Transformer 기반 고급 이미지 분석기
Florence-2를 대체하는 더 강력한 모델
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from PIL import Image
import asyncio
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    BlipProcessor, 
    BlipForConditionalGeneration,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer
)

logger = logging.getLogger(__name__)


class VisionTransformerAnalyzer:
    """비전 트랜스포머 기반 이미지 분석기"""
    
    def __init__(self, model_type: str = "blip"):
        self.model_type = model_type
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        
    async def initialize(self):
        """모델 초기화"""
        try:
            logger.info(f"Initializing Vision Transformer ({self.model_type})...")
            
            if self.model_type == "blip":
                # BLIP 모델 (이미지 캡셔닝 + VQA)
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
                self.model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-large",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            elif self.model_type == "git":
                # GIT (Generative Image-to-Text) 모델
                self.processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    "microsoft/git-large-coco",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            else:
                # 기본: TrOCR (텍스트 인식 특화)
                self.processor = ViTImageProcessor.from_pretrained("microsoft/trocr-large-printed")
                self.model = VisionEncoderDecoderModel.from_pretrained(
                    "microsoft/trocr-large-printed",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-large-printed")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.initialized = True
            logger.info(f"Vision Transformer initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vision Transformer: {e}")
            raise
    
    async def analyze_image(
        self, 
        image: Union[Image.Image, str, np.ndarray],
        task: str = "caption",
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """이미지 분석"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # 이미지 준비
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
            
            # 작업별 처리
            if task == "caption":
                return await self._generate_caption(image)
            elif task == "ocr":
                return await self._extract_text(image)
            elif task == "vqa" and prompt:
                return await self._answer_question(image, prompt)
            elif task == "electrical":
                return await self._analyze_electrical_problem(image)
            else:
                # 기본: 종합 분석
                return await self._comprehensive_analysis(image)
                
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                "error": str(e),
                "caption": "[분석 실패]",
                "text": "",
                "analysis": {}
            }
    
    async def _generate_caption(self, image: Image.Image) -> Dict[str, Any]:
        """이미지 캡션 생성"""
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return {
            "caption": caption,
            "confidence": 0.9
        }
    
    async def _extract_text(self, image: Image.Image) -> Dict[str, Any]:
        """텍스트 추출 (OCR)"""
        if self.model_type == "trocr":
            # TrOCR 사용
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, max_length=200)
                text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            # BLIP으로 텍스트 추출 시도
            inputs = self.processor(image, "What text is written in this image?", return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=100)
                text = self.processor.decode(out[0], skip_special_tokens=True)
        
        return {
            "text": text,
            "confidence": 0.85
        }
    
    async def _answer_question(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """비주얼 질문 답변 (VQA)"""
        if self.model_type == "blip":
            inputs = self.processor(image, question, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50)
                answer = self.processor.decode(out[0], skip_special_tokens=True)
        else:
            # 다른 모델은 캡션 기반으로 답변
            caption = await self._generate_caption(image)
            answer = f"Based on the image ({caption['caption']}), the answer might be related to the visual content."
        
        return {
            "question": question,
            "answer": answer,
            "confidence": 0.8
        }
    
    async def _analyze_electrical_problem(self, image: Image.Image) -> Dict[str, Any]:
        """전기공학 문제 분석"""
        results = {}
        
        # 1. 이미지 캡션
        caption_result = await self._generate_caption(image)
        results["caption"] = caption_result["caption"]
        
        # 2. 텍스트 추출
        text_result = await self._extract_text(image)
        results["text"] = text_result["text"]
        
        # 3. 전기공학 관련 질문
        electrical_questions = [
            "What electrical components are shown?",
            "What values or numbers are visible?",
            "Is this a circuit diagram or problem?",
            "What formulas are shown?"
        ]
        
        results["analysis"] = {}
        for q in electrical_questions:
            try:
                answer = await self._answer_question(image, q)
                results["analysis"][q] = answer["answer"]
            except:
                continue
        
        # 4. 종합 분석
        results["problem_type"] = self._infer_problem_type(results)
        results["extracted_values"] = self._extract_values(results["text"])
        
        return results
    
    async def _comprehensive_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """종합 분석"""
        results = {}
        
        # 병렬 처리
        tasks = [
            self._generate_caption(image),
            self._extract_text(image)
        ]
        
        caption_result, text_result = await asyncio.gather(*tasks)
        
        results.update({
            "caption": caption_result["caption"],
            "text": text_result["text"],
            "confidence": (caption_result["confidence"] + text_result["confidence"]) / 2
        })
        
        return results
    
    def _infer_problem_type(self, analysis: Dict[str, Any]) -> str:
        """문제 유형 추론"""
        text = (analysis.get("text", "") + " " + analysis.get("caption", "")).lower()
        
        if any(word in text for word in ["power factor", "역률", "cosφ", "cos"]):
            return "power_factor"
        elif any(word in text for word in ["impedance", "임피던스", "z ="]):
            return "impedance"
        elif any(word in text for word in ["circuit", "회로", "저항", "resistance"]):
            return "circuit_analysis"
        elif any(word in text for word in ["field", "전기장", "전위"]):
            return "electric_field"
        else:
            return "general"
    
    def _extract_values(self, text: str) -> List[Dict[str, Any]]:
        """텍스트에서 수치 추출"""
        import re
        
        values = []
        
        # 숫자와 단위 패턴
        patterns = [
            (r'(\d+(?:\.\d+)?)\s*(?:×|x)\s*10\^(-?\d+)\s*(\w+)', 'scientific'),
            (r'(\d+(?:\.\d+)?)\s*(kW|kVar|kVA|MW|MVA|W|VA|Var)', 'power'),
            (r'(\d+(?:\.\d+)?)\s*(V|kV|mV)', 'voltage'),
            (r'(\d+(?:\.\d+)?)\s*(A|mA|kA)', 'current'),
            (r'(\d+(?:\.\d+)?)\s*(Ω|ohm|kΩ|MΩ)', 'resistance'),
            (r'(\d+(?:\.\d+)?)\s*(F|μF|mF|pF)', 'capacitance'),
            (r'(\d+(?:\.\d+)?)\s*(H|mH|μH)', 'inductance'),
        ]
        
        for pattern, value_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if value_type == 'scientific':
                    base = float(match.group(1))
                    exp = int(match.group(2))
                    unit = match.group(3)
                    value = base * (10 ** exp)
                else:
                    value = float(match.group(1))
                    unit = match.group(2)
                
                values.append({
                    'value': value,
                    'unit': unit,
                    'type': value_type,
                    'original': match.group(0)
                })
        
        return values


class HybridImageAnalyzer:
    """하이브리드 이미지 분석기 (VisionTransformer + OCR)"""
    
    def __init__(self):
        self.vision_analyzer = VisionTransformerAnalyzer(model_type="blip")
        self.ocr_system = None  # RealOCRSystem will be injected
        self.initialized = False
        self._init_lock = asyncio.Lock()  # 초기화 동시성 제어
    
    async def initialize(self, ocr_system=None):
        """초기화"""
        try:
            # Vision Transformer 초기화
            await self.vision_analyzer.initialize()
            
            # OCR 시스템 주입
            if ocr_system:
                self.ocr_system = ocr_system
            else:
                # 기본 OCR 시스템 생성
                from real_ocr_system import RealOCRSystem
                self.ocr_system = RealOCRSystem()
                await self.ocr_system.initialize()
            
            self.initialized = True
            logger.info("Hybrid Image Analyzer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hybrid Analyzer: {e}")
            raise
    
    async def analyze_image(
        self, 
        image: Union[Image.Image, str, np.ndarray],
        task: str = "electrical",
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """하이브리드 이미지 분석"""
        if not self.initialized:
            async with self._init_lock:
                if not self.initialized:  # 이중 체크
                    await self.initialize()
        
        try:
            # 병렬 분석
            tasks = []
            
            # 1. Vision Transformer 분석
            tasks.append(self.vision_analyzer.analyze_image(image, task, prompt))
            
            # 2. OCR 분석
            if self.ocr_system:
                tasks.append(self._ocr_analysis(image))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 통합
            combined = {}
            
            # Vision Transformer 결과
            if not isinstance(results[0], Exception):
                combined.update(results[0])
            
            # OCR 결과
            if len(results) > 1 and not isinstance(results[1], Exception):
                ocr_result = results[1]
                # OCR 텍스트가 더 정확하면 대체
                if ocr_result.get("confidence", 0) > 0.8:
                    combined["text"] = ocr_result.get("text", "")
                    combined["ocr_text"] = ocr_result.get("text", "")
                    combined["extracted_values"] = ocr_result.get("extracted_values", [])
            
            # Florence-2 호환 형식으로 변환
            return self._format_for_compatibility(combined)
            
        except Exception as e:
            logger.error(f"Hybrid analysis failed: {e}")
            return {
                "error": str(e),
                "caption": "[분석 실패]",
                "ocr_text": ""
            }
    
    async def _ocr_analysis(self, image: Union[Image.Image, str, np.ndarray]) -> Dict[str, Any]:
        """OCR 분석"""
        if isinstance(image, str):
            result = await self.ocr_system.extract_text(image)
        else:
            # PIL Image를 임시 파일로 저장
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                if isinstance(image, np.ndarray):
                    Image.fromarray(image).save(tmp.name)
                else:
                    image.save(tmp.name)
                result = await self.ocr_system.extract_text(tmp.name)
        
        return result
    
    def _format_for_compatibility(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Florence-2 호환 형식으로 변환"""
        formatted = {
            "caption": result.get("caption", "[No caption]"),
            "ocr_text": result.get("text", result.get("ocr_text", "")),
            "labels": [],
            "analysis": result.get("analysis", {}),
            "problem_type": result.get("problem_type", "general"),
            "extracted_values": result.get("extracted_values", []),
            "confidence": result.get("confidence", 0.5)
        }
        
        # 레이블 생성
        if formatted["problem_type"] != "general":
            formatted["labels"].append(formatted["problem_type"])
        
        if formatted["extracted_values"]:
            formatted["labels"].append("has_values")
        
        return formatted


# Florence-2 호환 인터페이스
class Florence2ImageAnalyzer:
    """Florence-2 호환 인터페이스 (Vision Transformer 사용)"""
    
    def __init__(self, model_id: str = "microsoft/Florence-2-large"):
        self.analyzer = HybridImageAnalyzer()
        self.model_id = model_id  # 호환성을 위해 유지
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None  # 호환성을 위해 추가
        self.processor = None  # 호환성을 위해 추가
        
    def analyze_image(self, image, task="<OCR>", text_input=None):
        """동기 인터페이스 (Florence-2 호환)"""
        # 비동기를 동기로 변환
        import asyncio
        
        # 작업 매핑
        task_map = {
            "<OCR>": "ocr",
            "<CAPTION>": "caption",
            "<DETAILED_CAPTION>": "electrical",
            "<MORE_DETAILED_CAPTION>": "electrical",
            "<OD>": "electrical",
            "<DENSE_REGION_CAPTION>": "comprehensive",
            "<OCR_WITH_REGION>": "ocr"
        }
        
        mapped_task = task_map.get(task, "electrical")
        
        # 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.analyzer.analyze_image(image, mapped_task, text_input)
            )
            
            # Florence-2 호환 형식으로 변환
            if "caption" in result or "ocr_text" in result or "text" in result:
                return {
                    "task": task,
                    "result": result.get("caption") or result.get("ocr_text") or result.get("text", ""),
                    "success": True
                }
            else:
                return {
                    "task": task,
                    "result": "",
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
        except Exception as e:
            logger.error(f"analyze_image failed: {e}")
            return {
                "task": task,
                "result": "",
                "success": False,
                "error": str(e)
            }
        finally:
            loop.close()
    
    def generate_caption(self, image: Union[Image.Image, str, np.ndarray]) -> Tuple[str, Dict[str, Any]]:
        """
        이미지 캡션 생성 (Florence-2 호환)
        
        Returns:
            Tuple[str, Dict]: (캡션 텍스트, 추가 정보)
        """
        result = self.analyze_image(image, task="<CAPTION>")
        caption = result.get("result", "")
        return caption, {"confidence": 0.9 if result.get("success") else 0.1}
    
    def extract_text(self, image: Union[Image.Image, str, np.ndarray]) -> Tuple[str, List[Dict]]:
        """
        이미지에서 텍스트 추출 (OCR)
        
        Returns:
            Tuple[str, List[Dict]]: (추출된 텍스트, 영역 정보)
        """
        result = self.analyze_image(image, task="<OCR>")
        text = result.get("result", "")
        regions = []  # Vision Transformer는 영역 정보를 제공하지 않음
        return text, regions
    
    def extract_text_simple(self, image: Union[Image.Image, str, bytes]) -> str:
        """
        이미지에서 텍스트 추출 (OCR)
        
        Args:
            image: 분석할 이미지
            
        Returns:
            추출된 텍스트
        """
        result = self.analyze_image(image, task="<OCR>")
        if result.get("success") and result.get("result"):
            text = result["result"]
            # OCR 결과 정리
            if isinstance(text, str):
                import re
                
                # 전기공학 관련 유효한 패턴 추출
                valid_patterns = [
                    r'\b\d+(?:\.\d+)?\s*(?:k|m|M|G)?(?:W|VA|V|A|Hz|Ω|ohm|F|H|s|m|Wb|T|°C)\b',
                    r'\b[VPIRQSZCLFXY][a-z0-9_]*\b',
                    r'[=+\-*/()\[\]{}|∠φθ°]',
                    r'\b\d+/\d+\b',
                    r'\b\d+:\d+\b',
                ]
                
                # 유효한 부분 추출
                valid_parts = []
                for pattern in valid_patterns:
                    matches = re.findall(pattern, text)
                    valid_parts.extend(matches)
                
                if valid_parts:
                    # 중복 제거하고 순서 유지
                    seen = set()
                    unique_parts = []
                    for part in valid_parts:
                        if part not in seen:
                            seen.add(part)
                            unique_parts.append(part)
                    
                    text = ' '.join(unique_parts)
                
                # 길이 제한
                if len(text) > 200:
                    text = text[:200] + "..."
                
            return text
        return ""
    
    def generate_caption(
        self, 
        image: Union[Image.Image, str, bytes],
        detail_level: str = "simple"
    ) -> str:
        """
        이미지 캡션 생성
        
        Args:
            image: 분석할 이미지
            detail_level: 상세도 수준 ("simple", "detailed", "very_detailed")
            
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
        
        if result.get("success") and result.get("result"):
            return result["result"]
        return "이미지 분석에 실패했습니다."
    
    def detect_objects(
        self, 
        image: Union[Image.Image, str, bytes],
        object_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        객체 검출
        
        Args:
            image: 분석할 이미지
            object_name: 검출할 특정 객체명 (없으면 모든 객체)
            
        Returns:
            검출된 객체 리스트
        """
        # Vision Transformer는 객체 검출을 직접 지원하지 않으므로 빈 리스트 반환
        return []
    
    def analyze_formula(self, image: Union[Image.Image, str, bytes]) -> Dict[str, Any]:
        """
        수식 분석 (OCR + 상세 설명)
        
        Args:
            image: 수식이 포함된 이미지
            
        Returns:
            수식 분석 결과
        """
        # OCR로 수식 텍스트 추출
        ocr_result = self.analyze_image(image, task="<OCR_WITH_REGION>")
        
        # 상세 캡션으로 수식 설명
        caption_result = self.analyze_image(image, task="<DETAILED_CAPTION>")
        
        analysis = {
            "formula_text": "",
            "description": "",
            "regions": [],
            "success": False
        }
        
        if ocr_result.get("success"):
            analysis["formula_text"] = ocr_result.get("result", "")
        
        if caption_result.get("success"):
            analysis["description"] = caption_result.get("result", "")
        
        analysis["success"] = ocr_result.get("success", False) or caption_result.get("success", False)
        
        return analysis
    
    def batch_analyze(
        self, 
        images: List[Union[Image.Image, str, bytes]],
        task: str = "<DETAILED_CAPTION>"
    ) -> List[Dict[str, Any]]:
        """
        여러 이미지 일괄 분석
        
        Args:
            images: 이미지 리스트
            task: 수행할 작업
            
        Returns:
            분석 결과 리스트
        """
        results = []
        for image in images:
            result = self.analyze_image(image, task=task)
            results.append(result)
        return results


# 사용 예시
async def demo():
    """데모"""
    analyzer = HybridImageAnalyzer()
    await analyzer.initialize()
    
    # 이미지 분석
    result = await analyzer.analyze_image(
        "electrical_problem.png",
        task="electrical"
    )
    
    print(f"Caption: {result['caption']}")
    print(f"OCR Text: {result['ocr_text']}")
    print(f"Problem Type: {result['problem_type']}")
    print(f"Extracted Values: {result['extracted_values']}")


if __name__ == "__main__":
    asyncio.run(demo())