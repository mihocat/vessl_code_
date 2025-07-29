#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Analysis Service using Florence-2
Florence-2를 사용한 이미지 분석 서비스
"""

import logging
from typing import Dict, Any, Optional, List, Union
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import io
import base64

logger = logging.getLogger(__name__)


class Florence2ImageAnalyzer:
    """Florence-2 기반 이미지 분석기"""
    
    def __init__(self, model_id: str = "microsoft/Florence-2-large"):
        """
        Florence-2 이미지 분석기 초기화
        
        Args:
            model_id: Florence-2 모델 ID
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._load_model()
        
    def _load_model(self):
        """모델 로드"""
        try:
            logger.info(f"Loading Florence-2 model: {self.model_id}")
            # Florence-2 모델 로드 시 dtype 일치 문제 해결
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # 모델이 이미 device에 있으면 to() 호출 생략
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            logger.info("Florence-2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Florence-2 model: {e}")
            raise
    
    def analyze_image(
        self, 
        image: Union[Image.Image, str, bytes],
        task: str = "<DETAILED_CAPTION>",
        text_input: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        이미지 분석 수행
        
        Args:
            image: PIL Image 객체, 파일 경로, 또는 바이트 데이터
            task: Florence-2 작업 유형
                - <CAPTION>: 간단한 캡션
                - <DETAILED_CAPTION>: 상세한 캡션
                - <MORE_DETAILED_CAPTION>: 매우 상세한 캡션
                - <OCR>: 텍스트 추출
                - <OCR_WITH_REGION>: 위치 정보 포함 텍스트 추출
                - <DENSE_REGION_CAPTION>: 영역별 상세 설명
                - <REGION_PROPOSAL>: 객체 영역 제안
                - <CAPTION_TO_PHRASE_GROUNDING>: 캡션 기반 객체 위치
                - <REFERRING_EXPRESSION_SEGMENTATION>: 참조 표현 분할
                - <OPEN_VOCABULARY_DETECTION>: 개방형 어휘 검출
            text_input: 추가 텍스트 입력 (일부 작업에 필요)
            
        Returns:
            분석 결과 딕셔너리
        """
        try:
            # 이미지 로드
            if isinstance(image, str):
                # 파일 경로인 경우
                image = Image.open(image).convert("RGB")
            elif isinstance(image, bytes):
                # 바이트 데이터인 경우
                image = Image.open(io.BytesIO(image)).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise ValueError("Invalid image format")
            
            # 프롬프트 구성
            if text_input:
                prompt = f"{task} {text_input}"
            else:
                prompt = task
            
            # 이미지 처리 - dtype 일치를 위해 수정
            inputs = self.processor(
                text=prompt, 
                images=image, 
                return_tensors="pt"
            )
            
            # 입력 텐서의 dtype을 모델과 일치시킴
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # 추론
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False
                )
            
            # 결과 디코딩
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]
            
            # 결과 파싱
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task, 
                image_size=(image.width, image.height)
            )
            
            return {
                "task": task,
                "result": parsed_answer[task] if task in parsed_answer else parsed_answer,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                "task": task,
                "result": None,
                "error": str(e),
                "success": False
            }
    
    def extract_text(self, image: Union[Image.Image, str, bytes]) -> str:
        """
        이미지에서 텍스트 추출 (OCR)
        
        Args:
            image: 분석할 이미지
            
        Returns:
            추출된 텍스트
        """
        result = self.analyze_image(image, task="<OCR>")
        if result["success"] and result["result"]:
            return result["result"]
        return ""
    
    def generate_caption(
        self, 
        image: Union[Image.Image, str, bytes],
        detail_level: str = "detailed"
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
        
        task = task_map.get(detail_level, "<DETAILED_CAPTION>")
        result = self.analyze_image(image, task=task)
        
        if result["success"] and result["result"]:
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
        if object_name:
            # 특정 객체 검출
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
            result = self.analyze_image(image, task=task, text_input=object_name)
        else:
            # 모든 객체 검출
            task = "<REGION_PROPOSAL>"
            result = self.analyze_image(image, task=task)
        
        if result["success"] and result["result"]:
            return result["result"]
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
        
        if ocr_result["success"]:
            analysis["formula_text"] = ocr_result.get("result", "")
            if isinstance(ocr_result["result"], dict):
                analysis["regions"] = ocr_result["result"].get("quad_boxes", [])
        
        if caption_result["success"]:
            analysis["description"] = caption_result.get("result", "")
        
        analysis["success"] = ocr_result["success"] or caption_result["success"]
        
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


class MultimodalRAGService:
    """멀티모달 RAG 서비스"""
    
    def __init__(
        self, 
        image_analyzer: Florence2ImageAnalyzer,
        text_embedding_model: Any
    ):
        """
        멀티모달 RAG 서비스 초기화
        
        Args:
            image_analyzer: 이미지 분석기
            text_embedding_model: 텍스트 임베딩 모델
        """
        self.image_analyzer = image_analyzer
        self.text_embedding_model = text_embedding_model
        
    def process_multimodal_query(
        self, 
        text_query: str,
        image: Optional[Union[Image.Image, str, bytes]] = None
    ) -> Dict[str, Any]:
        """
        멀티모달 쿼리 처리 (Image-to-text → RAG → LLM)
        
        Args:
            text_query: 텍스트 질문
            image: 선택적 이미지 입력
            
        Returns:
            처리 결과
        """
        context = {
            "text_query": text_query,
            "image_analysis": None,
            "combined_query": text_query
        }
        
        if image:
            # 이미지 분석
            try:
                # 상세 캡션 생성
                caption = self.image_analyzer.generate_caption(image, detail_level="very_detailed")
                
                # OCR 텍스트 추출
                ocr_text = self.image_analyzer.extract_text(image)
                
                # 수식이 포함된 경우 특별 처리
                if any(keyword in text_query.lower() for keyword in ["수식", "공식", "equation", "formula"]):
                    formula_analysis = self.image_analyzer.analyze_formula(image)
                    if formula_analysis["success"]:
                        ocr_text = formula_analysis.get("formula_text", ocr_text)
                        caption = formula_analysis.get("description", caption)
                
                context["image_analysis"] = {
                    "caption": caption,
                    "ocr_text": ocr_text
                }
                
                # 텍스트 쿼리와 이미지 분석 결과 결합
                combined_parts = [text_query]
                
                if caption and caption != "이미지 분석에 실패했습니다.":
                    combined_parts.append(f"\n[이미지 내용]\n{caption}")
                
                if ocr_text:
                    combined_parts.append(f"\n[이미지에서 추출된 텍스트]\n{ocr_text}")
                
                context["combined_query"] = "\n".join(combined_parts)
                
            except Exception as e:
                logger.error(f"Multimodal query processing failed: {e}")
                context["image_analysis"] = {
                    "error": str(e),
                    "caption": "이미지 분석 실패",
                    "ocr_text": ""
                }
                context["combined_query"] = f"{text_query}\n[이미지 분석 실패]"
        
        return context