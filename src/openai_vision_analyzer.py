#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI Vision API를 사용한 이미지 분석기
"""

import logging
import base64
import time
from typing import Dict, Any, Optional, Union
from PIL import Image
import io
import os
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIVisionAnalyzer:
    """OpenAI Vision API 기반 이미지 분석기"""
    
    def __init__(self, config=None):
        """
        초기화
        
        Args:
            config: Config 객체 (없으면 기본 config 사용)
        """
        if config is None:
            from config import config as default_config
            config = default_config
        
        self.config = config.openai
        
        if not self.config.api_key or self.config.api_key.startswith("sk-fallback"):
            logger.warning("OpenAI API key not properly configured - Vision API will be disabled")
            self.api_available = False
        else:
            self.api_available = True
        
        if self.api_available:
            self.client = OpenAI(api_key=self.config.api_key)
            self.model = self.config.vision_model
            self.max_tokens = self.config.max_tokens
            self.temperature = self.config.temperature
            logger.info(f"OpenAI Vision Analyzer initialized with model: {self.model}")
        else:
            self.client = None
            self.model = None
            self.max_tokens = 1000
            self.temperature = 0.2
            logger.info("OpenAI Vision Analyzer initialized in fallback mode")
    
    def _encode_image(self, image: Union[Image.Image, str]) -> str:
        """이미지를 base64로 인코딩"""
        if isinstance(image, str):
            # 파일 경로인 경우
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            # PIL Image인 경우
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def analyze_image(
        self, 
        image: Union[Image.Image, str],
        question: Optional[str] = None,
        extract_text: bool = True,
        detect_formulas: bool = True,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        OpenAI Vision API로 이미지 분석
        
        Args:
            image: 분석할 이미지
            question: 사용자 질문
            extract_text: 텍스트 추출 여부
            detect_formulas: 수식 감지 여부
            max_tokens: 최대 토큰 수
            
        Returns:
            분석 결과
        """
        # API 사용 불가능한 경우 즉시 실패 반환
        if not self.api_available:
            logger.warning("OpenAI Vision API not available - API key not configured")
            return {
                "success": False,
                "error": "OpenAI API key not configured - using OCR fallback",
                "raw_response": ""
            }
        
        # 재시도 로직 적용
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # 이미지 인코딩
                base64_image = self._encode_image(image)
                
                # 프롬프트 구성
                system_prompt = "You are an expert in analyzing images, especially technical documents with Korean text and mathematical formulas."
                
                user_prompt_parts = []
                if question:
                    user_prompt_parts.append(f"User question: {question}")
                
                user_prompt_parts.append("Please analyze this image and provide:")
                
                if extract_text:
                    user_prompt_parts.append("1. All text content (especially Korean text)")
                
                if detect_formulas:
                    user_prompt_parts.append("2. Any mathematical formulas (provide in LaTeX format)")
                
                user_prompt_parts.append("3. A brief description of the image")
                user_prompt_parts.append("\nRespond in Korean.")
                
                user_prompt = "\n".join(user_prompt_parts)
                
                if attempt > 0:
                    logger.info(f"OpenAI Vision API retry attempt {attempt + 1}/{max_retries}")
                
                # API 호출
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                # 응답 파싱
                content = response.choices[0].message.content
                
                # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
                result = {
                    "success": True,
                    "raw_response": content,
                    "text_content": content,
                    "description": content,
                    "formulas": [],
                    "model": self.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
                
                # 텍스트와 수식 분리 시도
                if "LaTeX" in content or "수식" in content or "$$" in content or "\\(" in content:
                    result["has_formula"] = True
                    # 간단한 LaTeX 수식 추출 (실제로는 더 정교한 파싱 필요)
                    import re
                    latex_patterns = [
                        r'\$\$(.+?)\$\$',  # $$...$$
                        r'\\\((.+?)\\\)',  # \(...\)
                        r'\\\[(.+?)\\\]'   # \[...\]
                    ]
                    for pattern in latex_patterns:
                        matches = re.findall(pattern, content, re.DOTALL)
                        for match in matches:
                            result["formulas"].append({
                                "latex": match.strip(),
                                "confidence": 0.8
                            })
                else:
                    result["has_formula"] = False
                
                logger.info(f"OpenAI Vision analysis completed. Tokens used: {response.usage.total_tokens}")
                
                return result
                
            except Exception as e:
                error_str = str(e)
                logger.error(f"OpenAI Vision API attempt {attempt + 1} failed: {error_str}")
                
                # 권한 오류나 할당량 오류는 재시도하지 않음
                if "401" in error_str or "insufficient" in error_str.lower() or "quota" in error_str.lower():
                    logger.error("Authentication or quota error - not retrying")
                    return {
                        "success": False,
                        "error": error_str,
                        "raw_response": ""
                    }
                
                # 마지막 시도가 아니라면 대기 후 재시도
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # 지수 백오프
                    logger.info(f"Waiting {delay}s before retry...")
                    time.sleep(delay)
                else:
                    # 모든 재시도 실패
                    logger.error(f"All {max_retries} attempts failed")
                    return {
                        "success": False,
                        "error": error_str,
                        "raw_response": ""
                    }
    
    def process_multimodal_query(
        self,
        question: str,
        image: Optional[Union[Image.Image, str]] = None
    ) -> Dict[str, Any]:
        """
        멀티모달 쿼리 처리 (기존 시스템과의 호환성)
        
        Args:
            question: 사용자 질문
            image: 이미지 (선택적)
            
        Returns:
            처리 결과
        """
        if image is None:
            return {
                "combined_query": question,
                "image_analysis": None
            }
        
        # 이미지 분석
        analysis = self.analyze_image(image, question)
        
        if analysis["success"]:
            return {
                "combined_query": f"{question}\n\n[이미지 분석 결과]\n{analysis['raw_response']}",
                "image_analysis": {
                    "content": analysis["raw_response"],
                    "has_formula": analysis.get("has_formula", False),
                    "tokens_used": analysis["usage"]["total_tokens"]
                }
            }
        else:
            return {
                "combined_query": question,
                "image_analysis": {
                    "error": analysis["error"]
                }
            }


# 사용 예시
if __name__ == "__main__":
    # config에서 API 키 로드
    from config import config
    
    # API 키가 설정되어 있는지 확인
    if not config.openai.api_key:
        print("OPENAI_API_KEY 환경 변수를 설정해주세요.")
        exit(1)
    
    analyzer = OpenAIVisionAnalyzer(config)
    
    # 이미지 분석
    result = analyzer.analyze_image(
        "test_image.jpg",
        question="이 이미지의 수식을 설명해주세요"
    )
    
    print(f"모델: {result.get('model')}")
    print(f"응답: {result.get('raw_response')}")
    print(f"토큰 사용량: {result.get('usage', {}).get('total_tokens')}")