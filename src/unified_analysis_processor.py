#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 분석 프로세서
OpenAI GPT-4.1 단일 호출로 이미지+텍스트 분석만 수행하고,
최종 답변은 RAG + 파인튜닝 LLM만 담당
"""

import os
import base64
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from PIL import Image
import io

try:
    from openai import OpenAI
except ImportError as e:
    logging.error("OpenAI library not found. Install with: pip install openai")
    raise e

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """분석 결과"""
    success: bool
    extracted_text: Optional[str] = None
    formulas: Optional[List[str]] = None
    key_concepts: Optional[List[str]] = None
    question_intent: Optional[str] = None
    processing_time: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None
    cost: Optional[float] = None
    error_message: Optional[str] = None


class UnifiedAnalysisProcessor:
    """통합 분석 프로세서 - OpenAI 1회 호출 제한"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.api_key = self._load_api_key()
        
        if not self.api_key:
            raise ValueError("OpenAI API 키를 찾을 수 없습니다.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = self.config.get('unified_model', 'gpt-4.1')
        self.max_tokens = self.config.get('max_tokens', 300)  # 분석만 수행하므로 제한
        self.temperature = self.config.get('temperature', 0.1)
        
        # 1회 호출 제한 추적
        self._call_count = 0
        self._max_calls_per_query = 1  # 무조건 1회만 호출
        self._session_calls = 0  # 세션 전체 호출 추적
        
        logger.info(f"Unified Analysis Processor initialized - Model: {self.model}, Max tokens: {self.max_tokens}")
    
    def _load_api_key(self) -> Optional[str]:
        """API 키 로드"""
        # 1. 설정에서 직접 로드
        if 'api_key' in self.config and self.config['api_key']:
            return self.config['api_key']
        
        # 2. 환경 변수에서 로드
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return api_key
        
        # 3. VESSL 스토리지에서 로드
        vessl_key_paths = [
            '/apikey/openai_api_key.txt',
            '/apikey/OPENAI_API_KEY',
            './apikey/openai_api_key.txt'
        ]
        
        for path in vessl_key_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        key = f.read().strip()
                        if key:
                            logger.info(f"API key loaded from: {path}")
                            return key
            except Exception as e:
                logger.warning(f"Failed to read API key from {path}: {e}")
        
        return None
    
    def reset_call_count(self):
        """질의당 호출 횟수 초기화"""
        self._call_count = 0
    
    def analyze_image_and_text(
        self, 
        question: str,
        image: Optional[Union[Image.Image, str, bytes]] = None
    ) -> AnalysisResult:
        """
        이미지+텍스트 통합 분석 (1회 호출 제한)
        
        Args:
            question: 사용자 질문
            image: 이미지 (PIL Image, base64 문자열, 또는 바이트)
            
        Returns:
            AnalysisResult: 분석 결과
        """
        # 엄격한 1회 호출 제한
        if self._call_count >= self._max_calls_per_query:
            logger.warning(f"🚫 OpenAI API 호출 제한 초과 (허용: {self._max_calls_per_query}회, 시도: {self._call_count + 1}회)")
            return AnalysisResult(
                success=False,
                error_message=f"OpenAI API는 질의당 최대 {self._max_calls_per_query}회만 호출 가능합니다."
            )
        
        start_time = time.time()
        self._call_count += 1
        self._session_calls += 1
        
        logger.info(f"🚀 OpenAI Unified Analysis 요청 시작 (Model: {self.model})")
        logger.info(f"📊 호출 추적: 질의내 {self._call_count}/{self._max_calls_per_query}회, 세션내 {self._session_calls}회")
        
        try:
            # 메시지 구성
            messages = [
                {
                    "role": "system",
                    "content": """당신은 이미지와 텍스트를 분석하는 전문가입니다. 
다음 형식으로만 분석 결과를 제공하세요:

**추출된 텍스트:**
[이미지에서 읽은 모든 텍스트]

**감지된 수식:**
[LaTeX 형식의 수식들]

**핵심 개념:**
[질문과 관련된 핵심 개념 3-5개]

**질문 의도:**
[사용자가 무엇을 묻고자 하는지 한 문장으로]"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"질문: {question}\n\n위 질문과 관련하여 이미지를 분석해주세요."
                        }
                    ]
                }
            ]
            
            # 이미지가 있으면 추가
            if image is not None:
                image_base64 = self._process_image(image)
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": "low"  # 토큰 절약
                    }
                })
            
            logger.info(f"🚀 OpenAI Unified Analysis 요청 시작 (Model: {self.model})")
            logger.info(f"📝 요청 메시지: {len(messages)} 개, 질문: {question[:100]}...")
            if image is not None:
                logger.info(f"🖼️ 이미지 포함: 처리됨")
            
            # API 호출 - GPT-5는 max_completion_tokens 사용
            api_params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature
            }
            
            # GPT-5 모델인 경우 max_completion_tokens 사용
            if "gpt-5" in self.model.lower():
                api_params["max_completion_tokens"] = self.max_tokens
            else:
                api_params["max_tokens"] = self.max_tokens
            
            response = self.client.chat.completions.create(**api_params)
            
            logger.info(f"📥 OpenAI 응답 수신 완료")
            
            processing_time = time.time() - start_time
            
            if response and response.choices:
                content = response.choices[0].message.content
                
                # 토큰 사용량 및 비용 계산
                token_usage = None
                cost = 0.0
                if hasattr(response, 'usage'):
                    token_usage = {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    }
                    cost = self._calculate_cost(token_usage)
                
                # OpenAI 응답 내용 로깅
                logger.info(f"📋 OpenAI 응답 내용 (처음 200자): {content[:200]}...")
                
                # 분석 결과 파싱
                result = self._parse_analysis_result(content, processing_time, token_usage, cost)
                
                logger.info(f"✅ OpenAI Unified Analysis 완료 - "
                          f"Tokens: {token_usage['total_tokens'] if token_usage else 0}, "
                          f"Cost: ${cost:.4f}, Time: {processing_time:.2f}s")
                
                # 파싱된 결과 요약 로깅
                logger.info(f"🔍 분석 결과 요약: "
                          f"텍스트={len(result.extracted_text or '')}, "
                          f"수식={len(result.formulas or [])}, "
                          f"개념={len(result.key_concepts or [])}, "
                          f"의도={'있음' if result.question_intent else '없음'}")
                
                return result
            else:
                return AnalysisResult(
                    success=False,
                    error_message="API 응답을 받지 못했습니다.",
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ OpenAI Unified Analysis 실패: {e}")
            logger.error(f"❌ 오류 상세: {type(e).__name__}: {str(e)}")
            logger.error(f"❌ 호출 횟수: {self._call_count}/{self._max_calls_per_query}")
            return AnalysisResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _process_image(self, image: Union[Image.Image, str, bytes]) -> str:
        """이미지를 base64로 변환"""
        if isinstance(image, str):
            # 이미 base64 문자열인 경우
            return image
        elif isinstance(image, bytes):
            # bytes인 경우
            return base64.b64encode(image).decode('utf-8')
        elif isinstance(image, Image.Image):
            # PIL Image인 경우
            buffer = io.BytesIO()
            # 이미지 크기 최적화 (토큰 절약)
            if image.size[0] > 1024 or image.size[1] > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            image.save(buffer, format="JPEG", quality=85)
            image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _parse_analysis_result(
        self, 
        content: str, 
        processing_time: float,
        token_usage: Optional[Dict],
        cost: float
    ) -> AnalysisResult:
        """분석 결과 파싱"""
        try:
            # 기본값
            extracted_text = ""
            formulas = []
            key_concepts = []
            question_intent = ""
            
            # 섹션별 파싱
            sections = content.split("**")
            current_section = None
            
            for section in sections:
                section = section.strip()
                if "추출된 텍스트" in section:
                    current_section = "text"
                elif "감지된 수식" in section:
                    current_section = "formula"
                elif "핵심 개념" in section:
                    current_section = "concept"
                elif "질문 의도" in section:
                    current_section = "intent"
                elif section and current_section:
                    if current_section == "text":
                        extracted_text = section
                    elif current_section == "formula":
                        formulas = [f.strip() for f in section.split('\n') if f.strip()]
                    elif current_section == "concept":
                        key_concepts = [c.strip() for c in section.split('\n') if c.strip()]
                    elif current_section == "intent":
                        question_intent = section
            
            return AnalysisResult(
                success=True,
                extracted_text=extracted_text if extracted_text else None,
                formulas=formulas if formulas else None,
                key_concepts=key_concepts if key_concepts else None,
                question_intent=question_intent if question_intent else None,
                processing_time=processing_time,
                token_usage=token_usage,
                cost=cost
            )
            
        except Exception as e:
            logger.warning(f"분석 결과 파싱 오류: {e}")
            return AnalysisResult(
                success=True,
                extracted_text=content,  # 원본 그대로 반환
                processing_time=processing_time,
                token_usage=token_usage,
                cost=cost
            )
    
    def _calculate_cost(self, token_usage: Dict[str, int]) -> float:
        """비용 계산 (GPT-4.1 기준)"""
        # gpt-4.1 가격: $2.0/1M input tokens, $8.0/1M output tokens
        input_cost = token_usage.get('prompt_tokens', 0) * 2.0 / 1_000_000
        output_cost = token_usage.get('completion_tokens', 0) * 8.0 / 1_000_000
        return input_cost + output_cost
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """호출 통계 반환"""
        return {
            "current_call_count": self._call_count,
            "max_calls_per_query": self._max_calls_per_query,
            "calls_remaining": max(0, self._max_calls_per_query - self._call_count)
        }


def create_unified_processor(config: Optional[Dict] = None) -> UnifiedAnalysisProcessor:
    """통합 분석 프로세서 생성 편의 함수"""
    return UnifiedAnalysisProcessor(config)