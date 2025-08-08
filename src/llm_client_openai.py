#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI API LLM Client
OpenAI API를 사용하는 LLM 클라이언트
"""

import os
import logging
from typing import Optional, Dict, Any
from openai import OpenAI
from config import OpenAIConfig

logger = logging.getLogger(__name__)


class OpenAILLMClient:
    """OpenAI API를 사용하는 LLM 클라이언트"""
    
    def __init__(self, config: Optional[OpenAIConfig] = None):
        """
        OpenAI LLM 클라이언트 초기화
        
        Args:
            config: OpenAI 설정 객체
        """
        self.config = config or OpenAIConfig()
        
        # API 키 확인
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found")
            
        self.client = OpenAI(api_key=api_key)
        self.model = self.config.text_model
        logger.info(f"OpenAI LLM Client initialized with model: {self.model}")
        
    def check_health(self) -> bool:
        """API 연결 상태 확인"""
        try:
            # 간단한 API 호출로 상태 확인
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI API health check failed: {e}")
            return False
    
    def wait_for_server(self, max_attempts: int = 1) -> bool:
        """서버 대기 (OpenAI는 항상 준비됨)"""
        return self.check_health()
    
    def query(
        self, 
        prompt: str, 
        context: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        LLM 쿼리 실행
        
        Args:
            prompt: 사용자 질문
            context: 추가 컨텍스트
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            
        Returns:
            생성된 응답
        """
        try:
            # 전체 프롬프트 구성
            full_prompt = self._build_prompt(prompt, context)
            
            # API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            # 응답 추출
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content.strip()
            else:
                return "죄송합니다. 응답을 생성할 수 없습니다."
                
        except Exception as e:
            logger.error(f"OpenAI API query failed: {e}")
            return f"오류가 발생했습니다: {str(e)}"
    
    def _get_system_prompt(self) -> str:
        """시스템 프롬프트"""
        return """당신은 AI 전문가입니다. 
다음 원칙을 따라 답변하세요:
1. 참고자료가 있다면 그것을 기반으로 답변하세요
2. 수식이나 계산이 필요한 경우 단계별로 설명하세요
3. 전문 용어를 정확히 사용하세요
4. 간결하고 명확하게 답변하세요
5. 한국어로 답변하세요"""
    
    def _build_prompt(self, prompt: str, context: str) -> str:
        """프롬프트 구성"""
        if context and context.strip():
            return f"""=== 참고자료 ===
{context}
=================

위 참고자료를 바탕으로 다음 질문에 답변해주세요.

질문: {prompt}"""
        else:
            return f"질문: {prompt}"


# vLLM 클라이언트와 호환되는 인터페이스 제공
class LLMClient:
    """통합 LLM 클라이언트 (vLLM/OpenAI 자동 선택)"""
    
    def __init__(self, config=None):
        """초기화"""
        use_openai = os.getenv("USE_OPENAI_LLM", "false").lower() == "true"
        skip_vllm = os.getenv("SKIP_VLLM", "false").lower() == "true"
        
        if use_openai or skip_vllm:
            logger.info("Using OpenAI API for LLM")
            from config import Config
            full_config = Config()
            self.client = OpenAILLMClient(full_config.openai)
            self.is_openai = True
        else:
            logger.info("Using vLLM for LLM")
            from llm_client import LLMClient as VLLMClient
            self.client = VLLMClient(config)
            self.is_openai = False
    
    def check_health(self) -> bool:
        """헬스체크"""
        return self.client.check_health()
    
    def wait_for_server(self, max_attempts: int = 30) -> bool:
        """서버 대기"""
        if self.is_openai:
            return self.client.wait_for_server(max_attempts)
        else:
            # vLLM의 경우 원래 로직 사용
            return self.client.wait_for_server(max_attempts)
    
    def query(self, prompt: str, context: str = "", **kwargs) -> str:
        """쿼리 실행"""
        return self.client.query(prompt, context, **kwargs)