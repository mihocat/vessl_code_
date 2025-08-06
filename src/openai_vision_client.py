#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI Vision API 클라이언트
ChatGPT API를 사용한 이미지 분석 시스템
비용 최적화: gpt-4o-mini 사용으로 94% 비용 절감
"""

import os
import base64
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

try:
    from openai import OpenAI
except ImportError as e:
    logging.error("OpenAI library not found. Install with: pip install openai")
    raise e

logger = logging.getLogger(__name__)

@dataclass
class VisionAnalysisResult:
    """비전 분석 결과"""
    success: bool
    content: str
    extracted_text: Optional[str] = None
    formulas: Optional[List[str]] = None
    analysis_type: Optional[str] = None
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None
    error_message: Optional[str] = None
    fallback_used: bool = False
    metadata: Optional[Dict[str, Any]] = None

class OpenAIVisionClient:
    """OpenAI Vision API 클라이언트 - gpt-4o-mini 최적화"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        클라이언트 초기화
        
        Args:
            config: 설정 딕셔너리 (api_key, model, max_tokens 등)
        """
        self.config = config or {}
        
        # API 키 로드
        self.api_key = self._load_api_key()
        if not self.api_key:
            raise ValueError("OpenAI API 키를 찾을 수 없습니다.")
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=self.api_key)
        
        # 모델 설정 (비용 최적화)
        self.model = self.config.get('vision_model', 'gpt-4o-mini')  # 94% 비용 절감
        self.max_tokens = self.config.get('max_tokens', 4000)
        self.temperature = self.config.get('temperature', 0.1)
        
        # 분석 모드별 프롬프트
        self.analysis_prompts = self._init_analysis_prompts()
        
        # 요청 통계
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0
        }
        
        logger.info(f"OpenAI Vision Client initialized with model: {self.model}")
    
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
        
        # 4. 로컬 설정 파일에서 로드
        local_paths = [
            './config/openai_key.txt',
            '~/.openai/api_key',
            './openai_api_key.txt'
        ]
        
        for path in local_paths:
            expanded_path = os.path.expanduser(path)
            try:
                if os.path.exists(expanded_path):
                    with open(expanded_path, 'r', encoding='utf-8') as f:
                        key = f.read().strip()
                        if key:
                            logger.info(f"API key loaded from: {expanded_path}")
                            return key
            except Exception as e:
                logger.warning(f"Failed to read API key from {expanded_path}: {e}")
        
        logger.warning("No OpenAI API key found")
        return None
    
    def _init_analysis_prompts(self) -> Dict[str, str]:
        """질의 중심 최적화된 분석 프롬프트 초기화"""
        return {
            'text_extraction': """
질문과 관련된 중요한 텍스트만 정확히 추출:
1. 질문 답변에 필요한 한국어 텍스트와 수식 추출
2. 관련 없는 부분은 간략히 처리
3. 한국어+수식 혼재 시 둘 다 정확히 표현
4. LaTeX 형식: $$수식$$ 또는 $수식$
한국어로 응답하세요.
""",
            
            'formula_analysis': """
질문 관련 수식과 텍스트 중점 분석:
1. **핵심 수식**: LaTeX 형식 ($$수식$$)
2. **관련 한국어**: 수식 설명 부분만
3. **변수 설명**: 질문 관련 변수만
4. **계산 과정**: 질문에 필요한 부분만
불필요한 배경 설명은 제외하고 한국어로 응답하세요.
""",
            
            'problem_solving': """
질문 해결에 집중한 분석:
1. **핵심 문제**: 질문이 묻는 것만
2. **필요한 정보**: 문제 해결에 필요한 한국어+수식
3. **해결 단계**: 간결한 풀이
4. **답**: 명확한 결과
관련 없는 내용은 생략하고 한국어로 응답하세요.
""",
            
            'general_analysis': """
질문 중심 이미지 분석:
1. **질문 관련 내용**: 물어본 것과 연관된 부분만
2. **한국어 텍스트**: 질문 답변에 필요한 것만
3. **수식**: 관련 있는 수식만 LaTeX로 ($$수식$$)
4. **핵심 정보**: 질문 해결에 필수적인 것만
불필요한 세부사항은 제외하고 간결하게 한국어로 응답하세요.
"""
        }
    
    def analyze_image(self, image_data: Union[str, bytes], 
                     analysis_type: str = "general_analysis",
                     custom_prompt: Optional[str] = None,
                     query_context: Optional[Dict] = None) -> VisionAnalysisResult:
        """
        이미지 분석 수행
        
        Args:
            image_data: 이미지 데이터 (base64 문자열 또는 바이트)
            analysis_type: 분석 유형 (text_extraction, formula_analysis, problem_solving, general_analysis)
            custom_prompt: 사용자 정의 프롬프트
            query_context: 질의 컨텍스트 정보
            
        Returns:
            VisionAnalysisResult: 분석 결과
        """
        start_time = time.time()
        self.request_stats['total_requests'] += 1
        
        try:
            # 이미지 데이터 준비
            if isinstance(image_data, bytes):
                image_base64 = base64.b64encode(image_data).decode('utf-8')
            else:
                image_base64 = image_data
            
            # 프롬프트 선택
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = self.analysis_prompts.get(analysis_type, self.analysis_prompts['general_analysis'])
            
            # 컨텍스트 기반 프롬프트 향상
            if query_context:
                prompt = self._enhance_prompt_with_context(prompt, query_context)
            
            # API 요청
            response = self._make_vision_request(image_base64, prompt)
            
            if response:
                # 성공적인 응답 처리
                result = self._process_successful_response(
                    response, analysis_type, start_time
                )
                self.request_stats['successful_requests'] += 1
                
                # 후처리
                result = self._post_process_result(result, analysis_type)
                
                logger.info(f"Vision analysis completed: {analysis_type} ({result.processing_time:.2f}s)")
                return result
            else:
                # 실패 처리
                error_result = self._create_error_result(
                    "API 응답을 받지 못했습니다.", start_time
                )
                self.request_stats['failed_requests'] += 1
                return error_result
                
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            error_result = self._create_error_result(str(e), start_time)
            self.request_stats['failed_requests'] += 1
            return error_result
    
    def _make_vision_request(self, image_base64: str, prompt: str) -> Optional[Any]:
        """Vision API 요청 수행 (상세 로깅 포함)"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "low"  # 토큰 최적화: 질의 중심 분석에는 low면 충분
                            }
                        }
                    ]
                }
            ]
            
            # 요청 로깅
            request_info = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "prompt_length": len(prompt),
                "image_size_kb": len(image_base64) * 3 / 4 / 1024,  # base64 크기 추정
                "detail_level": "low"
            }
            logger.info(f"🚀 OpenAI Vision API 요청 시작: {request_info}")
            
            # API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # 응답 로깅
            if response and hasattr(response, 'usage'):
                response_info = {
                    "success": True,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "estimated_cost_usd": self._calculate_cost({
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }),
                    "response_length": len(response.choices[0].message.content) if response.choices else 0
                }
                logger.info(f"✅ OpenAI Vision API 응답 완료: {response_info}")
                
                # 상세 응답 내용 로깅 (처음 200자만)
                if response.choices and response.choices[0].message.content:
                    content_preview = response.choices[0].message.content[:200] + "..." if len(response.choices[0].message.content) > 200 else response.choices[0].message.content
                    logger.info(f"📝 응답 내용 미리보기: {content_preview}")
            
            return response
            
        except Exception as e:
            error_info = {
                "success": False,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "prompt_length": len(prompt),
                "image_size_kb": len(image_base64) * 3 / 4 / 1024
            }
            logger.error(f"❌ OpenAI Vision API 요청 실패: {error_info}")
            return None
    
    def _enhance_prompt_with_context(self, prompt: str, context: Dict) -> str:
        """질의 중심 프롬프트 향상"""
        # 질문 정보 우선 반영
        if 'query' in context and context['query']:
            user_question = context['query']
            enhanced_prompt = f"""사용자 질문: "{user_question}"

위 질문에 답하기 위해 이미지에서 필요한 부분만 분석하세요.

{prompt}

특별 지시사항:
- 질문과 직접 관련된 한국어 텍스트와 수식만 추출
- 한국어+수식 혼재 시 정확한 LaTeX 표현 사용
- 질문 해결에 불필요한 내용은 생략"""
        else:
            enhanced_prompt = prompt
        
        # 도메인별 간결한 힌트
        if 'domain' in context and context['domain']:
            domain_focus = {
                'electrical': "회로도, 전압/전류 값, 전기 공식 중심으로",
                'mathematics': "수식, 그래프, 계산 과정 중심으로", 
                'physics': "물리 공식, 수치, 단위 중심으로",
                'chemistry': "화학식, 분자구조, 반응식 중심으로"
            }
            if context['domain'] in domain_focus:
                enhanced_prompt += f"\n\n{domain_focus[context['domain']]} 분석하세요."
        
        # 핵심 키워드가 있으면 우선 처리
        if 'keywords' in context and context['keywords']:
            key_terms = ', '.join(context['keywords'][:2])  # 최대 2개만
            enhanced_prompt += f"\n\n핵심 키워드: {key_terms} - 이와 관련된 부분을 우선 분석하세요."
        
        return enhanced_prompt
    
    def _process_successful_response(self, response: Any, analysis_type: str, start_time: float) -> VisionAnalysisResult:
        """성공적인 응답 처리"""
        processing_time = time.time() - start_time
        
        # 응답 내용 추출 (안전한 접근)
        try:
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                elif hasattr(choice, 'content'):
                    content = choice.content
                else:
                    content = str(choice)
            else:
                content = str(response)
        except Exception as e:
            logger.warning(f"Response parsing issue: {e}")
            content = "응답을 파싱할 수 없습니다."
        
        # 토큰 사용량 추출
        token_usage = None
        try:
            if hasattr(response, 'usage'):
                token_usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
                self.request_stats['total_tokens'] += token_usage['total_tokens']
                
                # 비용 계산 (gpt-4o-mini 기준)
                cost = self._calculate_cost(token_usage)
                self.request_stats['total_cost'] += cost
        except Exception as e:
            logger.warning(f"Token usage parsing failed: {e}")
        
        # 신뢰도 점수 계산
        confidence_score = self._calculate_confidence_score(content, analysis_type)
        
        return VisionAnalysisResult(
            success=True,
            content=content,
            analysis_type=analysis_type,
            confidence_score=confidence_score,
            processing_time=processing_time,
            token_usage=token_usage,
            metadata={
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _post_process_result(self, result: VisionAnalysisResult, analysis_type: str) -> VisionAnalysisResult:
        """결과 후처리"""
        try:
            if analysis_type == 'text_extraction':
                # 텍스트 추출 결과 정제
                result.extracted_text = self._extract_text_from_content(result.content)
            
            elif analysis_type == 'formula_analysis':
                # 수식 추출
                result.formulas = self._extract_formulas_from_content(result.content)
            
            # 내용 정제
            result.content = self._clean_content(result.content)
            
        except Exception as e:
            logger.warning(f"Post-processing error: {e}")
        
        return result
    
    def _extract_text_from_content(self, content: str) -> Optional[str]:
        """내용에서 텍스트 추출"""
        # 간단한 텍스트 추출 로직
        lines = content.split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('**') and not line.startswith('#'):
                # 마크다운 헤더나 굵은 글씨가 아닌 일반 텍스트
                text_lines.append(line)
        
        return '\n'.join(text_lines) if text_lines else None
    
    def _extract_formulas_from_content(self, content: str) -> List[str]:
        """내용에서 수식 추출"""
        import re
        formulas = []
        
        # LaTeX 패턴 찾기
        latex_patterns = [
            r'\$\$(.+?)\$\$',  # 블록 수식
            r'\$(.+?)\$',      # 인라인 수식
            r'\\begin\{equation\}(.+?)\\end\{equation\}',  # equation 환경
        ]
        
        for pattern in latex_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            formulas.extend(matches)
        
        return list(set(formulas))  # 중복 제거
    
    def _clean_content(self, content: str) -> str:
        """내용 정제"""
        if not content:
            return content
        
        # 불필요한 공백 제거
        content = '\n'.join(line.strip() for line in content.split('\n'))
        
        # 연속된 빈 줄 제거
        import re
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        return content.strip()
    
    def _calculate_confidence_score(self, content: str, analysis_type: str) -> float:
        """신뢰도 점수 계산"""
        if not content:
            return 0.0
        
        score = 0.7  # 기본 점수
        
        # 내용 길이 기반
        if len(content) > 100:
            score += 0.1
        if len(content) > 500:
            score += 0.1
        
        # 분석 유형별 특화 점수
        if analysis_type == 'formula_analysis':
            if 'LaTeX' in content or '$' in content:
                score += 0.1
        elif analysis_type == 'text_extraction':
            if '텍스트' in content and '없음' not in content:
                score += 0.1
        
        # 구조화 점수
        if '**' in content or '#' in content:  # 마크다운 구조
            score += 0.05
        
        return min(1.0, score)
    
    def _calculate_cost(self, token_usage: Dict[str, int]) -> float:
        """비용 계산 (gpt-4o-mini 기준)"""
        # gpt-4o-mini 가격: $0.00015/1K input tokens, $0.0006/1K output tokens
        input_cost = token_usage.get('prompt_tokens', 0) * 0.00015 / 1000
        output_cost = token_usage.get('completion_tokens', 0) * 0.0006 / 1000
        return input_cost + output_cost
    
    def _create_error_result(self, error_message: str, start_time: float) -> VisionAnalysisResult:
        """에러 결과 생성"""
        processing_time = time.time() - start_time
        
        return VisionAnalysisResult(
            success=False,
            content="",
            error_message=error_message,
            processing_time=processing_time,
            metadata={
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def batch_analyze_images(self, images: List[Tuple[Union[str, bytes], str]], 
                           analysis_type: str = "general_analysis") -> List[VisionAnalysisResult]:
        """배치 이미지 분석"""
        results = []
        
        for i, (image_data, image_name) in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}: {image_name}")
            
            result = self.analyze_image(image_data, analysis_type)
            result.metadata = result.metadata or {}
            result.metadata['image_name'] = image_name
            result.metadata['batch_index'] = i
            
            results.append(result)
            
            # API 요청 제한 고려 (필요시)
            if i < len(images) - 1:
                time.sleep(0.1)  # 짧은 지연
        
        logger.info(f"Batch analysis completed: {len(results)} images")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        success_rate = 0.0
        if self.request_stats['total_requests'] > 0:
            success_rate = self.request_stats['successful_requests'] / self.request_stats['total_requests']
        
        avg_cost_per_request = 0.0
        if self.request_stats['successful_requests'] > 0:
            avg_cost_per_request = self.request_stats['total_cost'] / self.request_stats['successful_requests']
        
        return {
            'total_requests': self.request_stats['total_requests'],
            'successful_requests': self.request_stats['successful_requests'],
            'failed_requests': self.request_stats['failed_requests'],
            'success_rate': success_rate,
            'total_tokens_used': self.request_stats['total_tokens'],
            'total_cost_usd': self.request_stats['total_cost'],
            'average_cost_per_request': avg_cost_per_request,
            'model': self.model,
            'cost_optimization': "94% savings with gpt-4o-mini vs gpt-4o"
        }
    
    def test_connection(self) -> Tuple[bool, str]:
        """연결 테스트"""
        try:
            # 간단한 텍스트 완성으로 연결 테스트
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # 더 저렴한 모델로 테스트
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            if response and response.choices:
                return True, "OpenAI API 연결 성공"
            else:
                return False, "OpenAI API 응답 없음"
                
        except Exception as e:
            return False, f"OpenAI API 연결 실패: {e}"

# 편의 함수
def analyze_image_with_openai(image_data: Union[str, bytes], 
                             analysis_type: str = "general_analysis",
                             custom_prompt: Optional[str] = None) -> VisionAnalysisResult:
    """이미지 분석 편의 함수"""
    client = OpenAIVisionClient()
    return client.analyze_image(image_data, analysis_type, custom_prompt)

def test_openai_vision() -> bool:
    """OpenAI Vision API 테스트"""
    try:
        client = OpenAIVisionClient()
        success, message = client.test_connection()
        logger.info(f"OpenAI Vision test: {message}")
        return success
    except Exception as e:
        logger.error(f"OpenAI Vision test failed: {e}")
        return False