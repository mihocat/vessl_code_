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
    
    def query(self, prompt: str, context: str = "", max_tokens: int = 1200, temperature: float = 0.1) -> str:
        """LLM 질의 - 전기공학 전문 최적화"""
        try:
            # 전문 시스템 역할 정의
            system_role = """당신은 전기공학 전문가입니다. 다음 원칙을 따라 답변하세요:
1. 정확한 기술적 설명과 수식 제공
2. 실무적 관점에서 구체적 해답 제시  
3. 관련 법규나 기준이 있다면 명시
4. 불확실한 경우 "추가 확인이 필요합니다"라고 명시
5. 한국어로 전문 용어를 정확히 사용"""
            
            # 고급 프롬프트 구성
            if context:
                # 컨텍스트 전처리 및 구조화
                structured_context = self._structure_context(context)
                
                full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_role}<|eot_id|>"
                full_prompt += f"<|start_header_id|>user<|end_header_id|>"
                full_prompt += f"참고 자료:\n{structured_context}\n\n"
                full_prompt += f"질문: {prompt}\n\n"
                full_prompt += f"위 참고자료를 바탕으로 정확하고 전문적인 답변을 제공해주세요."
                full_prompt += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            else:
                full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_role}<|eot_id|>"
                full_prompt += f"<|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|>"
                full_prompt += f"<|start_header_id|>assistant<|end_header_id|>"
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.85,
                "presence_penalty": 0.05,
                "frequency_penalty": 0.05,
                "stop": ["<|eot_id|>"]
            }
            
            response = requests.post(self.completions_url, json=payload, timeout=45)
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["text"].strip()
                # 답변 후처리 및 검증
                return self._post_process_response(result)
            else:
                logger.error(f"LLM API 오류: {response.status_code} - {response.text}")
                return "서버 연결에 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
                
        except Exception as e:
            logger.error(f"LLM 질의 실패: {str(e)}")
            return "시스템 오류가 발생했습니다. 관리자에게 문의해 주세요."
    
    def _structure_context(self, context: str) -> str:
        """컨텍스트 구조화 및 최적화"""
        try:
            # 중복 제거 및 우선순위 정렬
            context_parts = context.split(" | ")
            unique_parts = []
            seen = set()
            
            for part in context_parts:
                part_clean = part.strip()
                if part_clean and part_clean not in seen and len(part_clean) > 20:
                    unique_parts.append(part_clean)
                    seen.add(part_clean)
                    if len(unique_parts) >= 3:  # 최대 3개 컨텍스트로 제한
                        break
            
            return "\n---\n".join(unique_parts)
        except:
            return context
    
    def _post_process_response(self, response: str) -> str:
        """답변 후처리 및 품질 개선"""
        try:
            # 불필요한 접두사/접미사 제거
            response = response.replace("답변:", "").replace("Answer:", "")
            response = response.strip()
            
            # 너무 짧은 답변 감지
            if len(response) < 10:
                return "충분한 정보를 찾을 수 없습니다. 더 구체적인 질문을 해주세요."
            
            # 문장 완성도 확인
            if not response.endswith(('.', '다', '요', '음', '니다', '습니다')):
                response += "."
            
            return response
        except:
            return response
    
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