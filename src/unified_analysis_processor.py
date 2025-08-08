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
# 디버그 모드 활성화
logger.setLevel(logging.DEBUG)

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
        self.model = self.config.get('unified_model', 'gpt-5')
        self.max_tokens = self.config.get('max_tokens', 400)  # output 토큰 제한
        self.temperature = self.config.get('temperature', 0.2)  # 일관성 있는 분석
        self.max_input_tokens = self.config.get('max_input_tokens', 1000)  # input 충분히 허용
        self.target_output_tokens = self.config.get('target_output_tokens', 350)
        
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
다음 형식으로 분석 결과를 제공하되, 각 섹션의 내용은 제한된 토큰 내에서 최대한 압축하여 핵심만 전달하세요:

**추출된 텍스트:** [이미지에서 읽은 모든 텍스트를 200자 이내로 요약]
**감지된 수식:** [LaTeX 형식의 주요 수식 3-5개]
**핵심 개념:** [질문과 관련된 핵심 개념 5-8개]
**질문 의도:** [사용자가 묻고자 하는 핵심을 한 문장으로]

중요: 전체 응답을 500토큰 이내로 압축하여 작성하세요."""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""질문: {question}

주의사항:
- 전체 응답은 반드시 {self.target_output_tokens}토큰 이내로 작성
- 핵심 정보만 압축하여 전달
- 불필요한 설명은 제외하고 요청된 형식만 준수"""
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
            
            # API 호출 - GPT-5 파라미터 호환성 처리
            api_params = {
                "model": self.model,
                "messages": messages
            }
            
            # GPT-5 모델인 경우 특별 처리
            if "gpt-5" in self.model.lower():
                api_params["max_completion_tokens"] = self.max_tokens
                # GPT-5는 temperature 기본값(1)만 지원
                # api_params["temperature"] = 1  # 기본값이므로 생략
            else:
                api_params["max_tokens"] = self.max_tokens
                api_params["temperature"] = self.temperature
            
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
                if content:
                    logger.info(f"📋 OpenAI 응답 내용 (처음 500자): {content[:500] if len(content) > 500 else content}")
                    if len(content) > 500:
                        logger.info(f"📋 OpenAI 응답 내용 (전체 {len(content)}자)")
                else:
                    logger.warning("⚠️ OpenAI 응답이 비어있습니다!")
                logger.debug(f"📄 OpenAI 전체 응답: {content}")
                
                # 분석 결과 파싱
                result = self._parse_analysis_result(content, processing_time, token_usage, cost)
                
                logger.info(f"✅ OpenAI Unified Analysis 완료 - "
                          f"Tokens: {token_usage['total_tokens'] if token_usage else 0}, "
                          f"Cost: ${cost:.4f}, Time: {processing_time:.2f}s")
                
                # 파싱된 결과 요약 로깅
                logger.info(f"🔍 분석 결과 요약: "
                          f"텍스트={len(result.extracted_text or '')}자, "
                          f"수식={len(result.formulas or [])}개, "
                          f"개념={len(result.key_concepts or [])}개, "
                          f"의도={'있음' if result.question_intent else '없음'}")
                
                # 토큰 효율성 체크
                if token_usage and token_usage.get('completion_tokens', 0) > self.target_output_tokens:
                    logger.warning(f"⚠️ Output 토큰 초과: {token_usage['completion_tokens']} > {self.target_output_tokens}")
                
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
        """분석 결과 파싱 - 개선된 로직"""
        try:
            # 파싱 전 로깅
            logger.debug(f"파싱 시작 - 컨텐츠 길이: {len(content)}자")
            logger.debug(f"파싱 대상 컨텐츠 처음 200자: {content[:200]}...")
            
            # 기본값
            extracted_text = ""
            formulas = []
            key_concepts = []
            question_intent = ""
            
            # 더 유연한 파싱 (**: 또는 ##를 지원)
            lines = content.split('\n')
            current_section = None
            current_content = []
            
            logger.debug(f"총 {len(lines)}개 라인 파싱 시작")
            
            for i, line in enumerate(lines):
                original_line = line
                line = line.strip()
                
                # 섹션 헤더 감지 (더 유연하게)
                if any(marker in line for marker in ["추출된 텍스트:", "추출된 텍스트：", "**추출된 텍스트**", "추출된 텍스트"]):
                    logger.debug(f"라인 {i}: '추출된 텍스트' 섹션 발견")
                    # 이전 섹션 저장
                    if current_section and current_content:
                        self._save_section(current_section, '\n'.join(current_content), 
                                         locals())
                    current_section = "text"
                    current_content = []
                    # 헤더와 같은 줄에 내용이 있는 경우 처리
                    remaining = line.replace("**추출된 텍스트**", "").replace("추출된 텍스트:", "").replace("추출된 텍스트：", "").strip()
                    if remaining:
                        current_content.append(remaining)
                elif any(marker in line for marker in ["감지된 수식:", "감지된 수식：", "**감지된 수식**", "감지된 수식"]):
                    logger.debug(f"라인 {i}: '감지된 수식' 섹션 발견")
                    if current_section and current_content:
                        self._save_section(current_section, '\n'.join(current_content), 
                                         locals())
                    current_section = "formula"
                    current_content = []
                    remaining = line.replace("**감지된 수식**", "").replace("감지된 수식:", "").replace("감지된 수식：", "").strip()
                    if remaining:
                        current_content.append(remaining)
                elif any(marker in line for marker in ["핵심 개념:", "핵심 개념：", "**핵심 개념**", "핵심 개념"]):
                    logger.debug(f"라인 {i}: '핵심 개념' 섹션 발견")
                    if current_section and current_content:
                        self._save_section(current_section, '\n'.join(current_content), 
                                         locals())
                    current_section = "concept"
                    current_content = []
                    remaining = line.replace("**핵심 개념**", "").replace("핵심 개념:", "").replace("핵심 개념：", "").strip()
                    if remaining:
                        current_content.append(remaining)
                elif any(marker in line for marker in ["질문 의도:", "질문 의도：", "**질문 의도**", "질문 의도"]):
                    logger.debug(f"라인 {i}: '질문 의도' 섹션 발견")
                    if current_section and current_content:
                        self._save_section(current_section, '\n'.join(current_content), 
                                         locals())
                    current_section = "intent"
                    current_content = []
                    remaining = line.replace("**질문 의도**", "").replace("질문 의도:", "").replace("질문 의도：", "").strip()
                    if remaining:
                        current_content.append(remaining)
                elif line and current_section:
                    # 현재 섹션에 내용 추가
                    current_content.append(line)
                elif line and not current_section:
                    logger.debug(f"라인 {i}: 섹션 없이 내용 발견: {line[:50]}...")
            
            # 마지막 섹션 저장
            if current_section and current_content:
                logger.debug(f"마지막 섹션 '{current_section}' 저장, 내용 {len(current_content)}줄")
                self._save_section(current_section, '\n'.join(current_content), locals())
            
            # 파싱 결과 로깅
            logger.debug(f"파싱 완료 - 텍스트: {len(extracted_text)}자, "
                        f"수식: {len(formulas)}개, 개념: {len(key_concepts)}개")
            
            # 파싱 실패 시 원본 그대로 사용
            if not any([extracted_text, formulas, key_concepts, question_intent]):
                logger.warning("파싱 실패 - 원본 내용을 그대로 사용")
                extracted_text = content
            
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
    
    def _save_section(self, section_type: str, content: str, context: dict):
        """섹션 내용 저장 헬퍼 함수"""
        content = content.strip()
        logger.debug(f"섹션 저장 - 타입: {section_type}, 내용 길이: {len(content)}자")
        
        if not content:
            logger.debug(f"섹션 {section_type}의 내용이 비어있음")
            return
            
        if section_type == "text":
            context['extracted_text'] = content
            logger.debug(f"추출된 텍스트 저장: {content[:100]}...")
        elif section_type == "formula":
            context['formulas'] = [f.strip() for f in content.split('\n') if f.strip()]
            logger.debug(f"수식 {len(context['formulas'])}개 저장")
        elif section_type == "concept":
            # 쉼표 또는 줄바꿈으로 구분
            if ',' in content:
                context['key_concepts'] = [c.strip() for c in content.split(',') if c.strip()]
            else:
                context['key_concepts'] = [c.strip() for c in content.split('\n') if c.strip()]
            logger.debug(f"핵심 개념 {len(context['key_concepts'])}개 저장")
        elif section_type == "intent":
            context['question_intent'] = content
            logger.debug(f"질문 의도 저장: {content[:100]}...")
    
    def _calculate_cost(self, token_usage: Dict[str, int]) -> float:
        """비용 계산 (GPT-5 기준)"""
        # GPT-5 예상 가격 (GPT-4보다 약간 높게 책정)
        # input: $3.0/1M tokens, output: $12.0/1M tokens
        input_cost = token_usage.get('prompt_tokens', 0) * 3.0 / 1_000_000
        output_cost = token_usage.get('completion_tokens', 0) * 12.0 / 1_000_000
        total_cost = input_cost + output_cost
        
        # 토큰 사용량 로깅
        logger.info(f"💰 토큰 사용: Input={token_usage.get('prompt_tokens', 0)}, "
                   f"Output={token_usage.get('completion_tokens', 0)}, "
                   f"Cost=${total_cost:.4f}")
        
        return total_cost
    
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