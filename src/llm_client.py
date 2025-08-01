#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Client Module for RAG System
LLM 클라이언트 모듈 - 범용 AI 시스템
"""

import requests
import logging
import time
from typing import Optional, Dict, Any
from config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM 서버 클라이언트"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        LLM 클라이언트 초기화
        
        Args:
            config: LLM 설정 객체
        """
        self.config = config or LLMConfig()
        logger.info(f"LLMClient initialized with model: {self.config.model_name}")
        self.session = requests.Session()
        self._setup_urls()
        
    def _setup_urls(self):
        """API 엔드포인트 설정"""
        self.health_check_url = f"{self.config.base_url}/health"
        self.completions_url = f"{self.config.base_url}/v1/completions"
        
    def check_health(self) -> bool:
        """서버 상태 확인"""
        try:
            response = self.session.get(
                self.health_check_url, 
                timeout=self.config.health_check_timeout
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
    
    def query(
        self, 
        prompt: str, 
        context: str = "", 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        LLM 질의 - 범용 AI 시스템
        
        Args:
            prompt: 사용자 질문
            context: 참고 컨텍스트
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            
        Returns:
            LLM 응답 텍스트
        """
        try:
            # 파라미터 설정
            max_tokens = max_tokens or self.config.max_tokens
            temperature = temperature or self.config.temperature
            
            # 프롬프트 구성
            full_prompt = self._build_prompt(prompt, context)
            
            # API 요청 페이로드
            payload = self._build_payload(full_prompt, max_tokens, temperature)
            
            # API 호출
            response = self._make_request(payload)
            
            if response.status_code == 200:
                result = self._extract_response(response)
                return self._post_process_response(result)
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return self._get_error_message()
                
        except requests.exceptions.Timeout:
            logger.error("LLM request timeout")
            return "요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."
        except Exception as e:
            logger.error(f"LLM query failed: {str(e)}")
            return self._get_error_message()
    
    def _build_prompt(self, prompt: str, context: str) -> str:
        """프롬프트 구성"""
        # 전기공학 전문 AI로 명확히 설정
        system_role = """당신은 전기공학 전문 AI 어시스턴트입니다.
다음 원칙을 따라 답변하세요:
1. 참고자료가 있다면 그것을 기반으로 답변하세요
2. 수식이나 계산이 필요한 경우 단계별로 설명하세요
3. 전기공학 용어를 정확히 사용하세요
4. 간결하고 명확하게 답변하세요"""

        # 컨텍스트가 있는 경우와 없는 경우 구분
        if context and context.strip():
            # 컨텍스트가 있을 때는 참고자료 강조
            full_prompt = f"""{system_role}

=== 참고자료 ===
{context}
=================

위 참고자료를 바탕으로 다음 질문에 답변해주세요.

질문: {prompt}

답변:"""
        else:
            # 컨텍스트가 없을 때는 일반적인 답변
            full_prompt = f"""{system_role}

질문: {prompt}

답변:"""
            
        return full_prompt
    
    def _build_payload(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict[str, Any]:
        """API 요청 페이로드 구성"""
        # 토큰 제한을 적절하게 조정 (2048 토큰 모델 대응)
        # 프롬프트 길이에 따라 동적으로 조정
        prompt_tokens = len(prompt.split())  # 대략적인 토큰 수 추정
        if prompt_tokens > 1200:
            max_tokens = min(max_tokens, 300)  # 긴 프롬프트일 때 더 제한
        elif prompt_tokens > 800:
            max_tokens = min(max_tokens, 500)  # 중간 길이
        else:
            max_tokens = min(max_tokens, 700)  # 짧은 프롬프트일 때 더 많이 허용
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": self.config.top_p,
            "presence_penalty": self.config.presence_penalty,
            "frequency_penalty": self.config.frequency_penalty,
            "stop": ["<|eot_id|>"]
        }
        
        # repetition_penalty 추가 (vLLM이 지원하는 경우)
        if hasattr(self.config, 'repetition_penalty'):
            payload["repetition_penalty"] = self.config.repetition_penalty
        logger.debug(f"Building payload with model: {self.config.model_name}")
        return payload
    
    def _make_request(self, payload: Dict[str, Any]) -> requests.Response:
        """API 요청 실행"""
        return self.session.post(
            self.completions_url,
            json=payload,
            timeout=self.config.timeout
        )
    
    def _extract_response(self, response: requests.Response) -> str:
        """응답에서 텍스트 추출"""
        result = response.json()
        
        # 다양한 응답 형식 처리
        if "choices" in result:
            if result["choices"]:
                return result["choices"][0].get("text", "").strip()
        elif "text" in result:
            return result["text"].strip()
        elif "generated_text" in result:
            return result["generated_text"].strip()
            
        return ""
    
    def _post_process_response(self, response: str) -> str:
        """응답 후처리"""
        if not response:
            return self._get_error_message()
            
        # 불필요한 공백 제거
        response = response.strip()
        
        # 응답이 너무 짧은 경우
        if len(response) < 10:
            return "죄송합니다. 충분한 답변을 생성하지 못했습니다. 다시 시도해주세요."
            
        return response
    
    def _get_error_message(self) -> str:
        """에러 메시지 반환"""
        return "죄송합니다. 응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    def wait_for_server(
        self, 
        max_retries: int = 60, 
        retry_interval: int = 3
    ) -> bool:
        """
        서버가 준비될 때까지 대기
        
        Args:
            max_retries: 최대 재시도 횟수
            retry_interval: 재시도 간격 (초)
            
        Returns:
            서버 준비 여부
        """
        logger.info("Waiting for LLM server...")
        
        for attempt in range(max_retries):
            if self.check_health():
                logger.info("LLM server is ready")
                return True
                
            if attempt < max_retries - 1:
                time.sleep(retry_interval)
                
        logger.error("LLM server failed to start")
        return False
    
    def close(self):
        """세션 종료"""
        self.session.close()
    
    def __enter__(self):
        """컨텍스트 관리자 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 관리자 종료"""
        self.close()
