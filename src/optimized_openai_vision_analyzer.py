#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최적화된 OpenAI Vision API 분석기
Optimized OpenAI Vision API Analyzer

토큰 사용량 최적화 및 응답 품질 개선
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


class OptimizedOpenAIVisionAnalyzer:
    """토큰 사용량을 최적화한 OpenAI Vision API 분석기"""
    
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
        
        if not self.config.api_key:
            logger.error("OpenAI API key not found - Vision API disabled")
            self.api_available = False
        elif self.config.api_key.startswith("sk-fallback"):
            logger.warning("OpenAI API key in fallback mode - Vision API disabled")
            self.api_available = False
        else:
            self.api_available = True
            logger.info(f"Optimized OpenAI Vision API enabled with key: {self.config.api_key[:7]}...")
        
        if self.api_available:
            self.client = OpenAI(api_key=self.config.api_key)
            self.model = self.config.vision_model
            # 최적화된 설정
            self.max_tokens = 1500  # 기존 1000 → 1500으로 증가 (더 상세한 응답)
            self.temperature = 0.1   # 기존 0.2 → 0.1로 감소 (더 일관된 응답)
            
            # 이미지 최적화 설정
            self.max_image_size = (800, 600)  # 기존 1024x576 → 800x600으로 감소
            self.image_quality = 85  # JPEG 품질 85% (압축률 증가)
            
            logger.info(f"Optimized OpenAI Vision Analyzer initialized - Model: {self.model}, Max tokens: {self.max_tokens}")
        else:
            self.client = None
            self.model = None
            self.max_tokens = 1500
            self.temperature = 0.1
            logger.info("Optimized OpenAI Vision Analyzer initialized in fallback mode")
    
    def _optimize_image(self, image: Union[Image.Image, str]) -> Image.Image:
        """이미지 최적화 (크기 축소 및 압축)"""
        if isinstance(image, str):
            # 파일 경로인 경우
            pil_image = Image.open(image)
        else:
            pil_image = image
        
        # RGB 모드로 변환 (RGBA → RGB)
        if pil_image.mode in ['RGBA', 'LA']:
            # 투명 배경을 흰색으로 변환
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            if pil_image.mode == 'RGBA':
                background.paste(pil_image, mask=pil_image.split()[-1])
            else:
                background.paste(pil_image)
            pil_image = background
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 크기 최적화
        original_size = pil_image.size
        pil_image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
        optimized_size = pil_image.size
        
        logger.info(f"Image optimized: {original_size} → {optimized_size}")
        return pil_image
    
    def _encode_image_optimized(self, image: Union[Image.Image, str]) -> str:
        """최적화된 이미지 base64 인코딩"""
        # 이미지 최적화
        optimized_image = self._optimize_image(image)
        
        # JPEG로 압축하여 인코딩 (PNG 대신 JPEG 사용으로 토큰 절약)
        buffered = io.BytesIO()
        optimized_image.save(buffered, format="JPEG", quality=self.image_quality, optimize=True)
        
        encoded_size = len(buffered.getvalue())
        logger.info(f"Image encoded size: {encoded_size / 1024:.1f} KB")
        
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _create_optimized_prompt(
        self, 
        question: Optional[str] = None,
        extract_text: bool = True,
        detect_formulas: bool = True
    ) -> tuple[str, str]:
        """질의 중심 최적화된 프롬프트 생성"""
        
        # 간결한 시스템 프롬프트
        system_prompt = "질의 중심 이미지 분석. 한국어+수식 혼재 처리 전문."
        
        # 질의 중심 사용자 프롬프트
        if question:
            user_prompt = f"""질문: "{question}"

위 질문 답변에 필요한 부분만 분석:
1. 질문 관련 한국어 텍스트만 추출
2. 관련 수식만 LaTeX 형식 ($$수식$$)
3. 불필요한 내용 생략

간결하게 한국어로 응답하세요."""
        else:
            # 질문이 없는 경우 기본 분석
            tasks = []
            if extract_text:
                tasks.append("중요한 한국어 텍스트만")
            if detect_formulas:
                tasks.append("핵심 수식만 LaTeX로")
            
            task_str = ", ".join(tasks) if tasks else "주요 내용만"
            user_prompt = f"{task_str} 추출하여 간결하게 한국어로 응답하세요."
        
        return system_prompt, user_prompt
    
    def analyze_image_optimized(
        self, 
        image: Union[Image.Image, str],
        question: Optional[str] = None,
        extract_text: bool = True,
        detect_formulas: bool = True
    ) -> Dict[str, Any]:
        """
        최적화된 OpenAI Vision API 이미지 분석
        
        Args:
            image: 분석할 이미지
            question: 사용자 질문
            extract_text: 텍스트 추출 여부
            detect_formulas: 수식 감지 여부
            
        Returns:
            분석 결과
        """
        # API 사용 불가능한 경우 즉시 실패 반환
        if not self.api_available:
            logger.warning("OpenAI Vision API not available - API key not configured")
            return {
                "success": False,
                "error": "OpenAI API key not configured - using OCR fallback",
                "raw_response": "",
                "token_usage": {"total_tokens": 0}
            }
        
        start_time = time.time()
        
        try:
            # 최적화된 이미지 인코딩
            base64_image = self._encode_image_optimized(image)
            
            # 최적화된 프롬프트 생성
            system_prompt, user_prompt = self._create_optimized_prompt(
                question, extract_text, detect_formulas
            )
            
            # 요청 정보 로깅
            request_info = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system_prompt_length": len(system_prompt),
                "user_prompt_length": len(user_prompt),
                "image_size_kb": len(base64_image) * 3 / 4 / 1024,
                "detail_level": "low",
                "question": question[:100] + "..." if question and len(question) > 100 else question
            }
            logger.info(f"🚀 최적화된 OpenAI Vision API 요청: {request_info}")
            
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
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"  # 'low' 사용으로 토큰 절약 (기존 auto/high)
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            processing_time = time.time() - start_time
            
            # 응답 파싱
            content = response.choices[0].message.content
            
            # 토큰 사용량 분석
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "estimated_cost_usd": self._calculate_cost(response.usage)
            }
            
            # 응답 정보 로깅
            response_info = {
                "success": True,
                "processing_time": processing_time,
                "prompt_tokens": token_usage["prompt_tokens"],
                "completion_tokens": token_usage["completion_tokens"],
                "total_tokens": token_usage["total_tokens"],
                "estimated_cost_usd": token_usage["estimated_cost_usd"],
                "response_length": len(content),
                "optimization_applied": True
            }
            logger.info(f"✅ 최적화된 OpenAI Vision API 응답 완료: {response_info}")
            
            # 응답 내용 미리보기 로깅
            content_preview = content[:200] + "..." if len(content) > 200 else content
            logger.info(f"📝 최적화된 응답 내용 미리보기: {content_preview}")
            
            # 수식 감지 개선
            formulas = []
            if detect_formulas:
                formulas = self._extract_formulas(content)
            
            result = {
                "success": True,
                "raw_response": content,
                "text_content": content,
                "description": content,
                "formulas": formulas,
                "has_formula": len(formulas) > 0,
                "model": self.model,
                "token_usage": token_usage,
                "processing_time": processing_time,
                "optimization_applied": True
            }
            
            logger.info(f"Optimized Vision analysis completed - Tokens: {token_usage['total_tokens']} "
                       f"(Input: {token_usage['prompt_tokens']}, Output: {token_usage['completion_tokens']}), "
                       f"Cost: ${token_usage['estimated_cost_usd']:.4f}, Time: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_str = str(e)
            logger.error(f"Optimized OpenAI Vision API failed: {error_str}")
            
            return {
                "success": False,
                "error": error_str,
                "raw_response": "",
                "token_usage": {"total_tokens": 0},
                "processing_time": processing_time,
                "optimization_applied": True
            }
    
    def _calculate_cost(self, usage) -> float:
        """토큰 사용량 기반 비용 계산 (gpt-4.1 기준)"""
        # gpt-4.1 가격: Input $2.0/1M tokens, Output $8.0/1M tokens
        input_cost = (usage.prompt_tokens / 1_000_000) * 2.0
        output_cost = (usage.completion_tokens / 1_000_000) * 8.0
        return input_cost + output_cost
    
    def _extract_formulas(self, content: str) -> list:
        """수식 추출 개선"""
        import re
        formulas = []
        
        # 다양한 수식 패턴 감지
        patterns = [
            r'\$\$(.+?)\$\$',  # $$...$$
            r'\\\((.+?)\\\)',  # \(...\)
            r'\\\[(.+?)\\\]',  # \[...\]
            r'([A-Z]\s*=\s*[^가-힣\n]{2,20})',  # 공식 패턴 (예: F = ma)
            r'(\w+\s*[+\-*/=]\s*\w+)',  # 기본 수식
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                
                formula_text = match.strip()
                if len(formula_text) > 1:  # 의미있는 수식만
                    formulas.append({
                        "latex": formula_text,
                        "confidence": 0.8,
                        "type": "mathematical_expression"
                    })
        
        return formulas
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """최적화 통계 조회"""
        return {
            "max_image_size": self.max_image_size,
            "image_quality": self.image_quality,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "model": self.model,
            "optimizations": [
                "이미지 크기 축소 (800x600)",
                "JPEG 압축 (85% 품질)",
                "간결한 프롬프트",
                "detail='low' 설정",
                "RGB 모드 변환"
            ]
        }


# 기존 클래스와의 호환성을 위한 래퍼
class OpenAIVisionAnalyzer(OptimizedOpenAIVisionAnalyzer):
    """기존 코드와의 호환성을 위한 래퍼 클래스"""
    
    def analyze_image(self, *args, **kwargs):
        """기존 메서드명 호환성"""
        return self.analyze_image_optimized(*args, **kwargs)


# 사용 예시 및 테스트
if __name__ == "__main__":
    from config import config
    
    if not config.openai.api_key:
        print("OPENAI_API_KEY 환경 변수를 설정해주세요.")
        exit(1)
    
    analyzer = OptimizedOpenAIVisionAnalyzer(config)
    
    print("=== 최적화된 OpenAI Vision Analyzer ===")
    print(f"최적화 설정: {analyzer.get_optimization_stats()}")
    
    # 테스트 이미지 분석
    if os.path.exists("test_image.jpg"):
        result = analyzer.analyze_image_optimized(
            "test_image.jpg",
            question="이 이미지의 수식을 설명해주세요"
        )
        
        print(f"\n=== 분석 결과 ===")
        print(f"성공: {result.get('success')}")
        print(f"모델: {result.get('model')}")
        print(f"토큰 사용량: {result.get('token_usage', {}).get('total_tokens')}")
        print(f"예상 비용: ${result.get('token_usage', {}).get('estimated_cost_usd', 0):.4f}")
        print(f"처리 시간: {result.get('processing_time', 0):.2f}초")
        print(f"수식 발견: {len(result.get('formulas', []))}개")
        print(f"\n응답 내용:\n{result.get('raw_response', '')[:200]}...")
    else:
        print("test_image.jpg 파일이 없어 테스트를 건너뜁니다.")