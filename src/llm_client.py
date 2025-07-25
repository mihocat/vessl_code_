#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Client Module for RAG System
LLM 클라이언트 모듈
"""

import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM 서버 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8000", model_name: str = "test_model"):
        """
        Args:
            base_url: vLLM 서버 주소
            model_name: 모델 이름
        """
        self.base_url = base_url
        self.model_name = model_name
        self.health_check_url = f"{base_url}/health"
        self.completions_url = f"{base_url}/v1/completions"
        
    def check_health(self) -> bool:
        """서버 상태 확인"""
        try:
            response = requests.get(self.health_check_url, timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def query(self, prompt: str, context: str = "", max_tokens: int = 800, temperature: float = 0.3) -> str:
        """LLM 질의"""
        try:
            # 시스템 역할 정의
            system_role = "당신은 AI 상담사입니다. 정확하고 이해하기 쉬운 설명을 제공합니다."
            
            # 프롬프트 구성
            if context:
                full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_role}"
                full_prompt += f"<|eot_id|><|start_header_id|>user<|end_header_id|>"
                full_prompt += f"참고자료:\n{context}\n\n질문: {prompt}"
                full_prompt += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            else:
                full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_role}"
                full_prompt += f"<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}"
                full_prompt += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.1
            }
            
            response = requests.post(self.completions_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["text"].strip()
            else:
                return "일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
                
        except Exception as e:
            logger.error(f"LLM 질의 실패: {str(e)}")
            return "연결에 문제가 발생했습니다."
    
    def wait_for_server(self, max_attempts: int = 60, delay: int = 3) -> bool:
        """서버 준비 대기"""
        import time
        
        logger.info("vLLM 서버 연결 대기 중...")
        for i in range(max_attempts):
            if self.check_health():
                logger.info("vLLM 서버 정상 연결됨")
                return True
            
            time.sleep(delay)
            if i % 5 == 4:
                logger.info(f"vLLM 서버 대기 중... ({i+1}/{max_attempts})")
        
        logger.error("vLLM 서버 연결 실패")
        return False