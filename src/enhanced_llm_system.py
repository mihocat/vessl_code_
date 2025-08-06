#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
향상된 파인튜닝 LLM 통합 시스템
Enhanced Fine-tuned LLM Integration System
KoLlama 3.2 한국어 전기공학 전문 모델 + 범용 LLM 지원
"""

import os
import time
import json
import logging
import requests
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """모델 타입"""
    FINE_TUNED = "fine_tuned"      # 파인튜닝 모델 (KoLlama)
    GENERAL = "general"            # 범용 모델 (GPT, Claude 등)
    SPECIALIZED = "specialized"    # 특화 모델
    ENSEMBLE = "ensemble"          # 앙상블 모델

class ModelDomain(Enum):
    """모델 도메인"""
    ELECTRICAL = "electrical"
    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    GENERAL_SCIENCE = "general_science"
    GENERAL = "general"

class ResponseQuality(Enum):
    """응답 품질"""
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    POOR = 2
    VERY_POOR = 1

@dataclass
class ModelResponse:
    """모델 응답"""
    content: str
    model_name: str
    model_type: ModelType
    confidence_score: float
    processing_time: float
    token_usage: Dict[str, int] = None
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.token_usage is None:
            self.token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        if self.metadata is None:
            self.metadata = {}
        self.metadata['timestamp'] = datetime.now().isoformat()

@dataclass
class ModelConfig:
    """모델 설정"""
    name: str
    model_type: ModelType
    base_url: str
    model_path: str = ""
    api_key: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.9
    timeout: int = 30
    domain_specialization: List[ModelDomain] = None
    
    def __post_init__(self):
        if self.domain_specialization is None:
            self.domain_specialization = [ModelDomain.GENERAL]

class BaseLLMClient(ABC):
    """기본 LLM 클라이언트 인터페이스"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'average_confidence': 0.0
        }
    
    @abstractmethod
    def generate(self, prompt: str, context: str = "", **kwargs) -> ModelResponse:
        """텍스트 생성"""
        pass
    
    @abstractmethod
    def check_health(self) -> bool:
        """상태 확인"""
        pass
    
    def update_stats(self, response: ModelResponse):
        """통계 업데이트"""
        self.stats['total_requests'] += 1
        
        if response.error_message is None:
            self.stats['successful_requests'] += 1
            
            # 평균 응답 시간
            n = self.stats['successful_requests']
            prev_avg_time = self.stats['average_response_time']
            self.stats['average_response_time'] = (
                (prev_avg_time * (n - 1) + response.processing_time) / n
            )
            
            # 평균 신뢰도
            prev_avg_conf = self.stats['average_confidence']
            self.stats['average_confidence'] = (
                (prev_avg_conf * (n - 1) + response.confidence_score) / n
            )
        else:
            self.stats['failed_requests'] += 1

class KoLlamaClient(BaseLLMClient):
    """KoLlama 3.2 파인튜닝 모델 클라이언트"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.session = requests.Session()
        
        # KoLlama 특화 설정
        self.electrical_engineering_context = """
당신은 한국어 전기공학 전문 AI입니다. 다음 특성을 가지고 있습니다:
1. 전기공학 전문 지식을 바탕으로 정확한 답변 제공
2. 한국어로 자연스럽고 명확한 설명
3. 수식과 계산 과정의 단계별 설명
4. 실무적이고 실용적인 접근
5. 안전 규정과 표준을 고려한 답변
"""
        
        # 도메인별 프롬프트 템플릿
        self.domain_templates = {
            ModelDomain.ELECTRICAL: self._get_electrical_template(),
            ModelDomain.MATHEMATICS: self._get_mathematics_template(),
            ModelDomain.PHYSICS: self._get_physics_template(),
            ModelDomain.GENERAL: self._get_general_template()
        }
        
        logger.info(f"KoLlama client initialized: {config.name}")
    
    def generate(self, prompt: str, context: str = "", domain: ModelDomain = ModelDomain.ELECTRICAL, **kwargs) -> ModelResponse:
        """텍스트 생성"""
        start_time = time.time()
        
        try:
            # 도메인별 프롬프트 구성
            enhanced_prompt = self._build_domain_prompt(prompt, context, domain)
            
            # API 요청
            response = self._make_request(enhanced_prompt, **kwargs)
            
            if response.status_code == 200:
                result = self._process_response(response, start_time)
                self.update_stats(result)
                return result
            else:
                error_response = self._create_error_response(
                    f"API Error: {response.status_code}", start_time
                )
                self.update_stats(error_response)
                return error_response
                
        except Exception as e:
            error_response = self._create_error_response(str(e), start_time)
            self.update_stats(error_response)
            return error_response
    
    def check_health(self) -> bool:
        """상태 확인"""
        try:
            health_url = f"{self.config.base_url}/health"
            response = self.session.get(health_url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def _build_domain_prompt(self, prompt: str, context: str, domain: ModelDomain) -> str:
        """도메인별 프롬프트 구성"""
        template = self.domain_templates.get(domain, self.domain_templates[ModelDomain.GENERAL])
        
        if context and context.strip():
            full_prompt = template['with_context'].format(
                context=context.strip(),
                question=prompt.strip()
            )
        else:
            full_prompt = template['without_context'].format(
                question=prompt.strip()
            )
        
        return full_prompt
    
    def _get_electrical_template(self) -> Dict[str, str]:
        """전기공학 템플릿"""
        return {
            'with_context': """당신은 한국어 전기공학 전문 AI입니다.

=== 참고자료 ===
{context}
================

위 참고자료를 바탕으로 다음 질문에 정확하고 자세하게 답변해주세요.
전기공학 용어를 정확히 사용하고, 필요한 경우 계산 과정을 단계별로 설명해주세요.

질문: {question}

답변:""",
            
            'without_context': """당신은 한국어 전기공학 전문 AI입니다.
전기공학 지식을 바탕으로 정확하고 실용적인 답변을 제공해주세요.

질문: {question}

답변:"""
        }
    
    def _get_mathematics_template(self) -> Dict[str, str]:
        """수학 템플릿"""
        return {
            'with_context': """당신은 수학 전문 AI입니다.

=== 참고자료 ===
{context}
================

위 참고자료를 참고하여 다음 수학 문제나 질문에 답변해주세요.
수식을 사용할 때는 LaTeX 형식으로 표현하고, 계산 과정을 단계별로 설명해주세요.

질문: {question}

답변:""",
            
            'without_context': """당신은 수학 전문 AI입니다.
수학적 정확성을 보장하며 단계별로 설명해주세요.

질문: {question}

답변:"""
        }
    
    def _get_physics_template(self) -> Dict[str, str]:
        """물리학 템플릿"""
        return {
            'with_context': """당신은 물리학 전문 AI입니다.

=== 참고자료 ===
{context}
================

위 참고자료를 바탕으로 물리학 질문에 답변해주세요.
물리 법칙과 원리를 명확히 설명하고, 필요한 경우 수식과 계산을 포함해주세요.

질문: {question}

답변:""",
            
            'without_context': """당신은 물리학 전문 AI입니다.
물리학 원리를 바탕으로 정확하고 이해하기 쉽게 설명해주세요.

질문: {question}

답변:"""
        }
    
    def _get_general_template(self) -> Dict[str, str]:
        """일반 템플릿"""
        return {
            'with_context': """당신은 도움이 되는 AI 어시스턴트입니다.

=== 참고자료 ===
{context}
================

위 참고자료를 바탕으로 질문에 정확하고 도움이 되는 답변을 제공해주세요.

질문: {question}

답변:""",
            
            'without_context': """당신은 도움이 되는 AI 어시스턴트입니다.
질문에 정확하고 유용한 답변을 제공해주세요.

질문: {question}

답변:"""
        }
    
    def _make_request(self, prompt: str, **kwargs) -> requests.Response:
        """API 요청"""
        payload = {
            "model": self.config.model_path,
            "prompt": prompt,
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
            "temperature": kwargs.get('temperature', self.config.temperature),
            "top_p": kwargs.get('top_p', self.config.top_p),
            "stop": ["<|eot_id|>", "질문:", "Question:"]
        }
        
        # vLLM 특화 파라미터
        if hasattr(self.config, 'repetition_penalty'):
            payload["repetition_penalty"] = 1.1
        
        return self.session.post(
            f"{self.config.base_url}/v1/completions",
            json=payload,
            timeout=self.config.timeout
        )
    
    def _process_response(self, response: requests.Response, start_time: float) -> ModelResponse:
        """응답 처리"""
        processing_time = time.time() - start_time
        result = response.json()
        
        # 텍스트 추출
        content = ""
        if "choices" in result and result["choices"]:
            content = result["choices"][0].get("text", "").strip()
        
        # 토큰 사용량
        token_usage = {}
        if "usage" in result:
            token_usage = result["usage"]
        else:
            # 추정 토큰 사용량
            token_usage = {
                'prompt_tokens': len(content.split()) * 1.3,  # 추정
                'completion_tokens': len(content.split()),
                'total_tokens': len(content.split()) * 2.3
            }
        
        # 신뢰도 점수 계산
        confidence_score = self._calculate_confidence(content)
        
        return ModelResponse(
            content=self._post_process_content(content),
            model_name=self.config.name,
            model_type=self.config.model_type,
            confidence_score=confidence_score,
            processing_time=processing_time,
            token_usage=token_usage,
            metadata={
                'domain_specialization': [d.value for d in self.config.domain_specialization]
            }
        )
    
    def _calculate_confidence(self, content: str) -> float:
        """신뢰도 점수 계산"""
        if not content:
            return 0.0
        
        factors = {
            'length': min(1.0, len(content) / 100),  # 적절한 길이
            'korean_ratio': self._get_korean_ratio(content),  # 한국어 비율
            'completeness': self._check_completeness(content),  # 완전성
            'technical_accuracy': self._assess_technical_accuracy(content)  # 기술적 정확성
        }
        
        weights = {'length': 0.2, 'korean_ratio': 0.2, 'completeness': 0.3, 'technical_accuracy': 0.3}
        confidence = sum(factors[f] * weights[f] for f in factors)
        
        return min(1.0, confidence)
    
    def _get_korean_ratio(self, content: str) -> float:
        """한국어 비율"""
        if not content:
            return 0.0
        
        korean_chars = len([c for c in content if '가' <= c <= '힣'])
        total_chars = len(content.replace(' ', ''))
        
        return korean_chars / total_chars if total_chars > 0 else 0.0
    
    def _check_completeness(self, content: str) -> float:
        """답변 완전성 확인"""
        if not content:
            return 0.0
        
        # 완전한 문장으로 끝나는지 확인
        ends_properly = content.rstrip().endswith(('.', '!', '?', '다', '요', '습니다'))
        
        # 적절한 길이인지 확인
        appropriate_length = 20 <= len(content) <= 2000
        
        # 질문이나 불완전한 응답 패턴 확인
        incomplete_patterns = ['죄송합니다', '모르겠습니다', '확실하지 않습니다']
        has_incomplete = any(pattern in content for pattern in incomplete_patterns)
        
        score = 0.0
        if ends_properly:
            score += 0.4
        if appropriate_length:
            score += 0.4
        if not has_incomplete:
            score += 0.2
        
        return score
    
    def _assess_technical_accuracy(self, content: str) -> float:
        """기술적 정확성 평가"""
        # 간단한 휴리스틱 기반 평가
        technical_indicators = [
            '전압', '전류', '저항', '전력', '회로', '임피던스',
            'V', 'A', 'Ω', 'W', 'Hz', 'kW'
        ]
        
        has_technical_terms = any(term in content for term in technical_indicators)
        
        # 수식이나 계산 과정이 포함되어 있는지
        has_calculations = any(char in content for char in '=+-*/()[]')
        
        # 단위나 수치가 적절히 사용되었는지
        import re
        has_units = bool(re.search(r'\d+\s*[a-zA-ZΩ가-힣]+', content))
        
        score = 0.5  # 기본 점수
        if has_technical_terms:
            score += 0.2
        if has_calculations:
            score += 0.2
        if has_units:
            score += 0.1
        
        return min(1.0, score)
    
    def _post_process_content(self, content: str) -> str:
        """내용 후처리"""
        if not content:
            return "죄송합니다. 적절한 답변을 생성하지 못했습니다."
        
        # 불필요한 공백 정리
        content = content.strip()
        
        # 반복적인 내용 제거
        content = self._remove_repetitions(content)
        
        # 최소 길이 확인
        if len(content) < 10:
            return "답변이 너무 짧습니다. 더 자세한 정보가 필요합니다."
        
        return content
    
    def _remove_repetitions(self, content: str) -> str:
        """반복 내용 제거"""
        lines = content.split('\n')
        unique_lines = []
        seen = set()
        
        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean not in seen:
                unique_lines.append(line)
                seen.add(line_clean)
        
        return '\n'.join(unique_lines)
    
    def _create_error_response(self, error_message: str, start_time: float) -> ModelResponse:
        """에러 응답 생성"""
        return ModelResponse(
            content="",
            model_name=self.config.name,
            model_type=self.config.model_type,
            confidence_score=0.0,
            processing_time=time.time() - start_time,
            error_message=error_message
        )

class GeneralLLMClient(BaseLLMClient):
    """범용 LLM 클라이언트 (GPT, Claude 등)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.session = requests.Session()
        
        # API 키 설정
        if config.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {config.api_key}',
                'Content-Type': 'application/json'
            })
        
        logger.info(f"General LLM client initialized: {config.name}")
    
    def generate(self, prompt: str, context: str = "", **kwargs) -> ModelResponse:
        """텍스트 생성"""
        start_time = time.time()
        
        try:
            # 범용 프롬프트 구성
            full_prompt = self._build_general_prompt(prompt, context)
            
            # API 요청
            response = self._make_request(full_prompt, **kwargs)
            
            if response.status_code == 200:
                result = self._process_response(response, start_time)
                self.update_stats(result)
                return result
            else:
                error_response = self._create_error_response(
                    f"API Error: {response.status_code}", start_time
                )
                self.update_stats(error_response)
                return error_response
                
        except Exception as e:
            error_response = self._create_error_response(str(e), start_time)
            self.update_stats(error_response)
            return error_response
    
    def check_health(self) -> bool:
        """상태 확인"""
        try:
            # 간단한 테스트 요청
            test_response = self.generate("Hello", max_tokens=5)
            return test_response.error_message is None
        except:
            return False
    
    def _build_general_prompt(self, prompt: str, context: str) -> str:
        """범용 프롬프트 구성"""
        if context and context.strip():
            return f"""다음 정보를 참고하여 질문에 답변해주세요.

참고정보:
{context.strip()}

질문: {prompt.strip()}

답변:"""
        else:
            return f"""질문: {prompt.strip()}

답변:"""
    
    def _make_request(self, prompt: str, **kwargs) -> requests.Response:
        """API 요청"""
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
            "temperature": kwargs.get('temperature', self.config.temperature),
        }
        
        return self.session.post(
            f"{self.config.base_url}/completions",
            json=payload,
            timeout=self.config.timeout
        )
    
    def _process_response(self, response: requests.Response, start_time: float) -> ModelResponse:
        """응답 처리"""
        processing_time = time.time() - start_time
        result = response.json()
        
        content = result.get('text', '').strip()
        confidence_score = 0.7  # 기본 신뢰도
        
        return ModelResponse(
            content=content,
            model_name=self.config.name,
            model_type=self.config.model_type,
            confidence_score=confidence_score,
            processing_time=processing_time
        )
    
    def _create_error_response(self, error_message: str, start_time: float) -> ModelResponse:
        """에러 응답 생성"""
        return ModelResponse(
            content="",
            model_name=self.config.name,
            model_type=self.config.model_type,
            confidence_score=0.0,
            processing_time=time.time() - start_time,
            error_message=error_message
        )

class EnhancedLLMSystem:
    """향상된 LLM 통합 시스템"""
    
    def __init__(self, primary_config: ModelConfig, fallback_configs: List[ModelConfig] = None):
        """
        시스템 초기화
        
        Args:
            primary_config: 기본 모델 설정 (KoLlama)
            fallback_configs: 대안 모델 설정들
        """
        self.primary_client = self._create_client(primary_config)
        self.fallback_clients = []
        
        if fallback_configs:
            for config in fallback_configs:
                client = self._create_client(config)
                if client:
                    self.fallback_clients.append(client)
        
        # 시스템 통계
        self.system_stats = {
            'total_requests': 0,
            'primary_success_rate': 0.0,
            'fallback_usage_rate': 0.0,
            'average_response_time': 0.0,
            'domain_usage': {domain.value: 0 for domain in ModelDomain}
        }
        
        logger.info(f"Enhanced LLM System initialized with {len(self.fallback_clients)} fallback models")
    
    def _create_client(self, config: ModelConfig) -> Optional[BaseLLMClient]:
        """클라이언트 생성"""
        try:
            if config.model_type == ModelType.FINE_TUNED:
                return KoLlamaClient(config)
            else:
                return GeneralLLMClient(config)
        except Exception as e:
            logger.error(f"Failed to create client for {config.name}: {e}")
            return None
    
    def generate(self, prompt: str, context: str = "", domain: ModelDomain = ModelDomain.ELECTRICAL, 
                use_fallback: bool = True, **kwargs) -> ModelResponse:
        """
        텍스트 생성 (자동 폴백 지원)
        
        Args:
            prompt: 질문
            context: 컨텍스트
            domain: 도메인
            use_fallback: 폴백 사용 여부
            **kwargs: 추가 파라미터
            
        Returns:
            ModelResponse: 생성 결과
        """
        self.system_stats['total_requests'] += 1
        self.system_stats['domain_usage'][domain.value] += 1
        
        # 기본 모델 시도
        try:
            if isinstance(self.primary_client, KoLlamaClient):
                response = self.primary_client.generate(prompt, context, domain, **kwargs)
            else:
                response = self.primary_client.generate(prompt, context, **kwargs)
            
            if response.error_message is None and response.confidence_score >= 0.5:
                self._update_system_stats(response, primary_used=True)
                return response
            
        except Exception as e:
            logger.warning(f"Primary model failed: {e}")
        
        # 폴백 모델 시도
        if use_fallback and self.fallback_clients:
            for fallback_client in self.fallback_clients:
                try:
                    response = fallback_client.generate(prompt, context, **kwargs)
                    if response.error_message is None:
                        response.metadata['fallback_used'] = True
                        self._update_system_stats(response, primary_used=False)
                        return response
                except Exception as e:
                    logger.warning(f"Fallback model {fallback_client.config.name} failed: {e}")
                    continue
        
        # 모든 모델 실패
        return self._create_system_error_response()
    
    def compare_models(self, prompt: str, context: str = "", domain: ModelDomain = ModelDomain.ELECTRICAL) -> Dict[str, ModelResponse]:
        """모델 비교"""
        results = {}
        
        # 기본 모델
        try:
            if isinstance(self.primary_client, KoLlamaClient):
                response = self.primary_client.generate(prompt, context, domain)
            else:
                response = self.primary_client.generate(prompt, context)
            results[self.primary_client.config.name] = response
        except Exception as e:
            logger.error(f"Primary model comparison failed: {e}")
        
        # 폴백 모델들
        for fallback_client in self.fallback_clients:
            try:
                response = fallback_client.generate(prompt, context)
                results[fallback_client.config.name] = response
            except Exception as e:
                logger.error(f"Fallback model {fallback_client.config.name} comparison failed: {e}")
        
        return results
    
    def get_best_model_for_domain(self, domain: ModelDomain) -> Optional[BaseLLMClient]:
        """도메인별 최적 모델 선택"""
        # KoLlama가 전기공학에 특화되어 있으므로
        if domain == ModelDomain.ELECTRICAL:
            return self.primary_client
        
        # 다른 도메인의 경우 가장 적합한 모델 선택
        for client in [self.primary_client] + self.fallback_clients:
            if domain in client.config.domain_specialization:
                return client
        
        # 기본값
        return self.primary_client
    
    def check_system_health(self) -> Dict[str, bool]:
        """시스템 전체 상태 확인"""
        health_status = {}
        
        # 기본 모델
        health_status['primary'] = self.primary_client.check_health()
        
        # 폴백 모델들
        for i, client in enumerate(self.fallback_clients):
            health_status[f'fallback_{i}'] = client.check_health()
        
        return health_status
    
    def _update_system_stats(self, response: ModelResponse, primary_used: bool):
        """시스템 통계 업데이트"""
        n = self.system_stats['total_requests']
        
        # 평균 응답 시간
        prev_avg_time = self.system_stats['average_response_time']
        self.system_stats['average_response_time'] = (
            (prev_avg_time * (n - 1) + response.processing_time) / n
        )
        
        # 기본 모델 성공률
        if primary_used:
            prev_success_rate = self.system_stats['primary_success_rate']
            success = 1.0 if response.error_message is None else 0.0
            self.system_stats['primary_success_rate'] = (
                (prev_success_rate * (n - 1) + success) / n
            )
        else:
            # 폴백 사용률
            prev_fallback_rate = self.system_stats['fallback_usage_rate']
            self.system_stats['fallback_usage_rate'] = (
                (prev_fallback_rate * (n - 1) + 1.0) / n
            )
    
    def _create_system_error_response(self) -> ModelResponse:
        """시스템 에러 응답"""
        return ModelResponse(
            content="죄송합니다. 모든 모델에서 응답을 생성할 수 없습니다. 잠시 후 다시 시도해주세요.",
            model_name="system",
            model_type=ModelType.GENERAL,
            confidence_score=0.0,
            processing_time=0.0,
            error_message="All models failed"
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 반환"""
        return {
            'system_stats': self.system_stats,
            'primary_model_stats': self.primary_client.stats,
            'fallback_model_stats': [client.stats for client in self.fallback_clients],
            'model_count': 1 + len(self.fallback_clients),
            'health_status': self.check_system_health()
        }

# 편의 함수들
def create_kollama_config(base_url: str = "http://localhost:8000", model_path: str = "kollama-3.2-electrical") -> ModelConfig:
    """KoLlama 설정 생성"""
    return ModelConfig(
        name="KoLlama-3.2-Electrical",
        model_type=ModelType.FINE_TUNED,
        base_url=base_url,
        model_path=model_path,
        max_tokens=1024,
        temperature=0.1,
        domain_specialization=[ModelDomain.ELECTRICAL, ModelDomain.GENERAL_SCIENCE]
    )

def create_enhanced_llm_system(kollama_url: str = "http://localhost:8000") -> EnhancedLLMSystem:
    """향상된 LLM 시스템 생성"""
    primary_config = create_kollama_config(kollama_url)
    return EnhancedLLMSystem(primary_config)

def test_llm_system(system: EnhancedLLMSystem, test_query: str = "전압과 전류의 관계를 설명해주세요.") -> ModelResponse:
    """LLM 시스템 테스트"""
    return system.generate(test_query, domain=ModelDomain.ELECTRICAL)