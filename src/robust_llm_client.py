#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust LLM Client with 3-Tier Fallback Architecture
향상된 3계층 폴백 아키텍처 LLM 클라이언트
"""

import requests
import logging
import time
import os
from typing import Optional, Dict, Any
from config import LLMConfig
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class RobustLLMClient:
    """
    3-Tier Fallback Architecture LLM Client
    Tier 1: vLLM (Primary)
    Tier 2: OpenAI API (Fallback)
    Tier 3: Hybrid Mode (Emergency)
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        로버스트 LLM 클라이언트 초기화
        
        Args:
            config: LLM 설정 객체
        """
        self.config = config or LLMConfig()
        self.primary_client = LLMClient(config)
        self.fallback_client = None
        self.current_mode = "initializing"
        self.mode_history = []
        
        # Fallback configuration
        self.enable_fallback = os.getenv("ENABLE_OPENAI_FALLBACK", "true").lower() == "true"
        self.fallback_delay = int(os.getenv("FALLBACK_DELAY", "180"))
        
        logger.info(f"RobustLLMClient initialized - Fallback: {self.enable_fallback}")
        
    def initialize_and_connect(self) -> str:
        """
        시스템 초기화 및 연결 설정
        
        Returns:
            연결된 모드: "primary", "fallback", "hybrid", "failed"
        """
        logger.info("Starting robust LLM client initialization...")
        
        # Tier 1: Try Primary vLLM Server
        if self._initialize_primary():
            self.current_mode = "primary"
            self.mode_history.append(("primary", time.time()))
            logger.info("✅ Tier 1: Primary vLLM server ready")
            return "primary"
        
        # Tier 2: Fallback to OpenAI API
        if self.enable_fallback and self._initialize_openai_fallback():
            self.current_mode = "fallback"
            self.mode_history.append(("fallback", time.time()))
            logger.info("✅ Tier 2: OpenAI API fallback ready")
            return "fallback"
        
        # Tier 3: Hybrid mode (minimal functionality)
        if self._initialize_hybrid_mode():
            self.current_mode = "hybrid"
            self.mode_history.append(("hybrid", time.time()))
            logger.info("✅ Tier 3: Hybrid mode ready")
            return "hybrid"
        
        # All tiers failed
        self.current_mode = "failed"
        self.mode_history.append(("failed", time.time()))
        logger.error("❌ All tiers failed - no LLM service available")
        return "failed"
    
    def _initialize_primary(self) -> bool:
        """
        Tier 1: Primary vLLM 서버 초기화
        Enhanced multi-stage verification
        """
        logger.info("Tier 1: Initializing primary vLLM server...")
        
        max_attempts = 60
        for attempt in range(max_attempts):
            # Stage 1: Basic health check
            if not self._verify_basic_health():
                if attempt < max_attempts - 1:
                    time.sleep(3)
                continue
            
            # Stage 2: Model endpoint verification
            if not self._verify_model_endpoint():
                if attempt < max_attempts - 1:
                    time.sleep(3)
                continue
            
            # Stage 3: Test completion functionality
            if not self._verify_completion_functionality():
                if attempt < max_attempts - 1:
                    time.sleep(3)
                continue
            
            # Stage 4: Application-level connectivity test
            if not self._verify_application_connectivity():
                if attempt < max_attempts - 1:
                    time.sleep(3)
                continue
            
            # All verification stages passed
            logger.info(f"✅ Primary server verified after {attempt + 1} attempts")
            return True
        
        logger.error("❌ Primary server verification failed after all attempts")
        return False
    
    def _verify_basic_health(self) -> bool:
        """Stage 1: Basic health endpoint verification"""
        try:
            response = requests.get(
                f"{self.config.base_url}/health",
                timeout=self.config.health_check_timeout
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Basic health check failed: {e}")
            return False
    
    def _verify_model_endpoint(self) -> bool:
        """Stage 2: Model endpoint accessibility verification"""
        try:
            response = requests.get(
                f"{self.config.base_url}/v1/models",
                timeout=self.config.health_check_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    # Check if our target model is available
                    model_ids = [model.get("id", "") for model in data["data"]]
                    if self.config.model_name in model_ids:
                        logger.debug(f"Target model {self.config.model_name} found")
                        return True
                    else:
                        logger.warning(f"Target model {self.config.model_name} not found in {model_ids}")
                        return False
            
            return False
        except Exception as e:
            logger.debug(f"Model endpoint verification failed: {e}")
            return False
    
    def _verify_completion_functionality(self) -> bool:
        """Stage 3: Test actual completion functionality"""
        try:
            test_payload = {
                "model": self.config.model_name,
                "prompt": "안녕하세요. 간단한 테스트입니다.",
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.config.base_url}/v1/completions",
                json=test_payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    response_text = result["choices"][0].get("text", "").strip()
                    if response_text:
                        logger.debug(f"Completion test successful: '{response_text[:50]}...'")
                        return True
            
            logger.debug(f"Completion test failed: {response.status_code} - {response.text[:100]}")
            return False
            
        except Exception as e:
            logger.debug(f"Completion functionality test failed: {e}")
            return False
    
    def _verify_application_connectivity(self) -> bool:
        """Stage 4: Application-level connectivity verification"""
        try:
            # Test with primary client
            test_response = self.primary_client.query(
                "테스트",
                max_tokens=5,
                temperature=0.1
            )
            
            # Check if we got a valid response (not an error message)
            if test_response and not test_response.startswith("죄송합니다"):
                logger.debug(f"Application connectivity test successful")
                return True
            else:
                logger.debug(f"Application connectivity test failed: '{test_response}'")
                return False
                
        except Exception as e:
            logger.debug(f"Application connectivity test failed: {e}")
            return False
    
    def _initialize_openai_fallback(self) -> bool:
        """Tier 2: OpenAI API 폴백 초기화"""
        logger.info("Tier 2: Initializing OpenAI API fallback...")
        
        try:
            # Check if OpenAI API key is available
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OpenAI API key not found in environment")
                return False
            
            # Try to initialize OpenAI client
            from llm_client_openai import OpenAILLMClient
            self.fallback_client = OpenAILLMClient()
            
            # Test OpenAI connectivity
            test_response = self.fallback_client.query(
                "Hello",
                max_tokens=5,
                temperature=0.1
            )
            
            if test_response and not test_response.startswith("죄송합니다"):
                logger.info("✅ OpenAI API fallback initialized successfully")
                return True
            else:
                logger.error("❌ OpenAI API test failed")
                return False
                
        except ImportError:
            logger.error("OpenAI client module not available")
            return False
        except Exception as e:
            logger.error(f"OpenAI fallback initialization failed: {e}")
            return False
    
    def _initialize_hybrid_mode(self) -> bool:
        """Tier 3: 하이브리드 모드 초기화 (최소 기능)"""
        logger.info("Tier 3: Initializing hybrid mode...")
        
        try:
            # Hybrid mode always succeeds as it provides minimal functionality
            # This could include local knowledge base, cached responses, etc.
            self.current_mode = "hybrid"
            logger.info("✅ Hybrid mode initialized")
            return True
        except Exception as e:
            logger.error(f"Hybrid mode initialization failed: {e}")
            return False
    
    def query(
        self, 
        prompt: str, 
        context: str = "", 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        루트 쿼리 처리 - 현재 모드에 따라 적절한 서비스로 라우팅
        
        Args:
            prompt: 사용자 질문
            context: 참고 컨텍스트
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            
        Returns:
            LLM 응답 텍스트
        """
        if self.current_mode == "primary":
            return self._query_primary(prompt, context, max_tokens, temperature)
        elif self.current_mode == "fallback":
            return self._query_fallback(prompt, context, max_tokens, temperature)
        elif self.current_mode == "hybrid":
            return self._query_hybrid(prompt, context)
        else:
            return self._get_service_unavailable_message()
    
    def _query_primary(
        self, 
        prompt: str, 
        context: str = "", 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Primary vLLM 서버를 통한 쿼리 처리"""
        try:
            response = self.primary_client.query(prompt, context, max_tokens, temperature)
            
            # Check if response indicates server failure
            if self._is_server_error_response(response):
                logger.warning("Primary server error detected, attempting fallback...")
                return self._handle_primary_failure(prompt, context, max_tokens, temperature)
            
            return response
            
        except Exception as e:
            logger.error(f"Primary query failed: {e}")
            return self._handle_primary_failure(prompt, context, max_tokens, temperature)
    
    def _query_fallback(
        self, 
        prompt: str, 
        context: str = "", 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """OpenAI API 폴백을 통한 쿼리 처리"""
        try:
            if self.fallback_client:
                return self.fallback_client.query(prompt, context, max_tokens, temperature)
            else:
                return self._query_hybrid(prompt, context)
        except Exception as e:
            logger.error(f"Fallback query failed: {e}")
            return self._query_hybrid(prompt, context)
    
    def _query_hybrid(self, prompt: str, context: str = "") -> str:
        """하이브리드 모드 쿼리 처리 (최소 기능)"""
        try:
            if context and context.strip():
                # If we have context, provide a basic response based on it
                return f"""참고자료를 바탕으로 답변드리겠습니다:

{context[:300]}...

현재 시스템이 복구 중이어서 제한된 기능만 제공됩니다. 
더 정확하고 상세한 답변을 원하시면 잠시 후 다시 시도해주세요."""
            else:
                # No context available, provide general response
                return f"""질문을 확인했습니다: "{prompt[:100]}..."

죄송합니다. 현재 시스템 복구 중이어서 정상적인 답변을 제공할 수 없습니다.
잠시 후 다시 시도해주시거나, 더 구체적인 질문으로 다시 문의해주세요."""
                
        except Exception as e:
            logger.error(f"Hybrid query failed: {e}")
            return self._get_service_unavailable_message()
    
    def _handle_primary_failure(
        self, 
        prompt: str, 
        context: str = "", 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Primary 서버 실패 시 처리"""
        if self.enable_fallback and self.fallback_client:
            logger.info("Switching to fallback mode due to primary failure")
            self.current_mode = "fallback"
            self.mode_history.append(("fallback_switch", time.time()))
            return self._query_fallback(prompt, context, max_tokens, temperature)
        else:
            logger.info("Switching to hybrid mode due to primary failure")
            self.current_mode = "hybrid"
            self.mode_history.append(("hybrid_switch", time.time()))
            return self._query_hybrid(prompt, context)
    
    def _is_server_error_response(self, response: str) -> bool:
        """서버 오류 응답인지 확인"""
        error_indicators = [
            "죄송합니다. 응답 생성 중 오류가 발생했습니다",
            "요청 시간이 초과되었습니다",
            "서버에 연결할 수 없습니다",
            "LLM 서버 연결에 실패했습니다"
        ]
        
        return any(indicator in response for indicator in error_indicators)
    
    def _get_service_unavailable_message(self) -> str:
        """서비스 이용 불가 메시지 반환"""
        return """서비스가 일시적으로 이용할 수 없습니다.

시스템 복구 중이니 잠시 후 다시 시도해주세요.
불편을 드려 죄송합니다."""
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 정보 반환"""
        return {
            "current_mode": self.current_mode,
            "mode_history": self.mode_history,
            "fallback_enabled": self.enable_fallback,
            "primary_config": {
                "base_url": self.config.base_url,
                "model_name": self.config.model_name,
                "timeout": self.config.timeout
            },
            "health_status": {
                "primary": self._verify_basic_health() if self.current_mode == "primary" else False,
                "fallback": self.fallback_client is not None,
                "hybrid": True
            }
        }
    
    def switch_mode(self, target_mode: str) -> bool:
        """수동으로 모드 전환"""
        if target_mode == "primary" and self._initialize_primary():
            self.current_mode = "primary"
            self.mode_history.append(("manual_switch_primary", time.time()))
            logger.info("Manually switched to primary mode")
            return True
        elif target_mode == "fallback" and self._initialize_openai_fallback():
            self.current_mode = "fallback"
            self.mode_history.append(("manual_switch_fallback", time.time()))
            logger.info("Manually switched to fallback mode")
            return True
        elif target_mode == "hybrid":
            self.current_mode = "hybrid"
            self.mode_history.append(("manual_switch_hybrid", time.time()))
            logger.info("Manually switched to hybrid mode")
            return True
        else:
            logger.error(f"Failed to switch to {target_mode} mode")
            return False
    
    def close(self):
        """리소스 정리"""
        if self.primary_client:
            self.primary_client.close()
        if self.fallback_client and hasattr(self.fallback_client, 'close'):
            self.fallback_client.close()
    
    def __enter__(self):
        """컨텍스트 관리자 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 관리자 종료"""
        self.close()