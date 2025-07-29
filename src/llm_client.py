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
        system_role = """당신은 도움이 되고 정확한 정보를 제공하는 AI 어시스턴트입니다.
        
답변 원칙:
1. 정확하고 신뢰할 수 있는 정보 제공
2. 모르는 내용은 솔직하게 인정
3. 간결하고 명확한 설명
4. 사용자의 질문 의도 파악 및 맞춤 답변"""
        
        if context:
            # 컨텍스트가 있는 경우
            structured_context = self._structure_context(context)
            full_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"{system_role}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>"
                f"[참고 자료]\n{structured_context}\n\n"
                f"[질문] {prompt}\n\n"
                f"위 참고자료를 활용하여 답변해주세요."
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
        else:
            # 컨텍스트가 없는 경우
            full_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"{system_role}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>"
            )
            
        return full_prompt
    
    def _build_payload(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict[str, Any]:
        """API 요청 페이로드 구성"""
        return {
            "model": self.config.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": self.config.top_p,
            "presence_penalty": self.config.presence_penalty,
            "frequency_penalty": self.config.frequency_penalty,
            "stop": ["<|eot_id|>"]
        }
    
    def _make_request(self, payload: Dict[str, Any]) -> requests.Response:
        """API 요청 실행"""
        return self.session.post(
            self.completions_url,
            json=payload,
            timeout=self.config.timeout
        )
    
    def _extract_response(self, response: requests.Response) -> str:
        """응답에서 텍스트 추출"""
        return response.json()["choices"][0]["text"].strip()
    
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
                    if len(unique_parts) >= 3:  # 최대 3개 컨텍스트
                        break
            
            return "\n---\n".join(unique_parts)
        except Exception:
            return context
    
    def _post_process_response(self, response: str) -> str:
        """답변 후처리 및 품질 개선"""
        try:
            # 불필요한 접두사 제거
            prefixes_to_remove = ["답변:", "Answer:", "응답:"]
            for prefix in prefixes_to_remove:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
            
            # 너무 짧은 답변 처리
            if len(response) < 10:
                return "충분한 정보를 찾을 수 없습니다. 더 구체적인 질문을 해주세요."
            
            # 문장 완성도 확인
            endings = ('.', '다', '요', '음', '니다', '습니다', '?', '!')
            if not response.endswith(endings):
                response += "."
            
            return response
        except Exception:
            return response
    
    def _get_error_message(self) -> str:
        """에러 메시지 반환"""
        return "죄송합니다. 응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    def wait_for_server(self, max_attempts: int = 100, delay: int = 3) -> bool:
        """
        서버 준비 대기
        
        Args:
            max_attempts: 최대 시도 횟수
            delay: 시도 간격 (초)
            
        Returns:
            서버 준비 여부
        """
        logger.info("LLM 서버 연결 대기 중...")
        
        for attempt in range(max_attempts):
            if self.check_health():
                logger.info("LLM 서버 정상 연결됨")
                return True
            
            time.sleep(delay)
            
            # 진행 상황 로그
            if (attempt + 1) % 5 == 0:
                logger.info(f"LLM 서버 대기 중... ({attempt + 1}/{max_attempts})")
        
        logger.error("LLM 서버 연결 실패")
        return False
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.session.close()