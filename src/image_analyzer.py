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
    
    def __init__(self, model_id: str = "microsoft/Florence-2-base"):
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
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # CPU에서 먼저 로드 후 GPU로 이동하는 방식으로 변경
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            
            # 모델 로드 - dtype 자동 처리
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                torch_dtype="auto",  # 자동으로 적절한 dtype 선택
                low_cpu_mem_usage=True
            )
            
            # GPU 사용 가능 시 이동
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    self.model = self.model.to(self.device)
                except torch.cuda.OutOfMemoryError:
                    logger.warning("GPU out of memory, falling back to CPU")
                    self.device = "cpu"
                    self.model = self.model.to(self.device)
                
            # 모델을 평가 모드로 설정
            self.model.eval()
            
            logger.info(f"Florence-2 model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Florence-2 model: {e}")
            self.model = None
            self.processor = None
            raise
    
    def analyze_image(
        self, 
        image: Union[Image.Image, str, bytes],
        task: str = "<CAPTION>",  # 기본값을 간단한 캡션으로 변경
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
        # 기본 에러 응답
        error_response = {
            "caption": "[이미지 분석 실패]",
            "ocr_text": "",
            "error": None
        }
        
        try:
            # 모델이 로드되지 않은 경우
            if self.model is None or self.processor is None:
                error_response["error"] = "Model not loaded"
                logger.error("Florence-2 model is not loaded")
                return error_response
            
            # 이미지 로드
            if isinstance(image, str):
                # 파일 경로인 경우
                image = Image.open(image).convert("RGB")
            elif isinstance(image, bytes):
                # 바이트 데이터인 경우
                image = Image.open(io.BytesIO(image)).convert("RGB")
            elif not isinstance(image, Image.Image):
                error_response["error"] = "Invalid image format"
                return error_response
            
            # 프롬프트 구성
            if text_input:
                prompt = f"{task} {text_input}"
            else:
                prompt = task
            
            # 이미지 처리
            inputs = self.processor(
                text=prompt, 
                images=image, 
                return_tensors="pt"
            )
            
            # 추론을 위한 입력 준비
            generated_ids = None
            
            # 추론 - dtype 문제 해결을 위한 안전한 방법
            with torch.no_grad():
                try:
                    # 모든 입력을 적절한 device로 이동
                    input_ids = inputs["input_ids"].to(self.device)
                    pixel_values = inputs["pixel_values"].to(self.device)
                    
                    # 모델의 dtype과 일치시키기
                    if hasattr(self.model, 'dtype') and self.model.dtype == torch.float16:
                        pixel_values = pixel_values.half()
                    elif hasattr(self.model, 'dtype') and self.model.dtype == torch.bfloat16:
                        pixel_values = pixel_values.bfloat16()
                    
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        max_new_tokens=256,  # 토큰 수 대폭 축소
                        num_beams=2,  # 빔 수 축소
                        do_sample=False
                    )
                except RuntimeError as e:
                    if "dtype" in str(e):
                        logger.warning(f"Dtype mismatch detected, retrying with float32: {e}")
                        # float32로 재시도
                        pixel_values = inputs["pixel_values"].to(self.device).float()
                        generated_ids = self.model.generate(
                            input_ids=input_ids,
                            pixel_values=pixel_values,
                            max_new_tokens=256,  # 토큰 수 대폭 축소
                            num_beams=2,  # 빔 수 축소
                            do_sample=False
                        )
                    else:
                        raise
            
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
            
            # 결과 후처리 - 길이 제한
            result = parsed_answer[task] if task in parsed_answer else parsed_answer
            if isinstance(result, str) and len(result) > 500:
                result = result[:500] + "..."
            
            return {
                "task": task,
                "result": result,
                "success": True
            }
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory during image analysis")
            error_response["error"] = "GPU memory error"
            return error_response
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            error_response["error"] = str(e)
            return error_response
    
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
            text = result["result"]
            # OCR 결과 정리 - 의미없는 반복 문자 제거
            if isinstance(text, str):
                # 연속된 한글 자모음만으로 이루어진 긴 문자열 제거
                import re
                # 의미없는 패턴 감지 (같은 문자 반복, 무작위 한글 자모음 등)
                if len(text) > 100 and (
                    len(set(text)) < len(text) * 0.1 or  # 문자 다양성이 너무 낮음
                    re.search(r'[\u3130-\u318F]{20,}', text) or  # 한글 자모음만 20자 이상
                    re.search(r'(.)\1{10,}', text)  # 같은 문자 10번 이상 반복
                ):
                    logger.warning("OCR result seems to be noise, returning empty")
                    return ""
                # 길이 제한
                if len(text) > 200:
                    text = text[:200] + "..."
            return text
        return ""
    
    def generate_caption(
        self, 
        image: Union[Image.Image, str, bytes],
        detail_level: str = "simple"  # 기본값을 simple로 변경
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
        
        task = task_map.get(detail_level, "<CAPTION>")  # 기본값도 simple로
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
                # 간단한 캡션 생성 (토큰 절약)
                caption = self.image_analyzer.generate_caption(image, detail_level="simple")
                
                # OCR 텍스트 추출
                ocr_text = self.image_analyzer.extract_text(image)
                
                # 수식이 포함된 경우에도 일반 OCR 사용 (더 안정적)
                # analyze_formula는 더 복잡하고 오류가 발생하기 쉬움
                
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