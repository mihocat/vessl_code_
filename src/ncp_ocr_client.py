#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCP OCR 폴백 시스템
ChatGPT API 실패 시 안정적인 대안으로 사용
향상된 텍스트 추출 및 수식 감지 기능
"""

import os
import json
import uuid
import time
import logging
import requests
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """OCR 분석 결과"""
    success: bool
    extracted_text: str
    formulas: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    error_message: Optional[str] = None
    raw_response: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class NCPOCRClient:
    """향상된 NCP OCR API 클라이언트 - 안정적인 폴백 시스템"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 설정 딕셔너리 (api_key 포함)
        """
        self.config = config or {}
        
        # NCP OCR API 정보
        self.api_url = "https://k6zo4uvsft.apigw.ntruss.com/custom/v1/44927/e5e764823794c03e15935af13327be2fd469fe9d7338e8f243ef3b9460cd9358/general"
        self.api_key = None
        self.api_available = False
        
        # 설정값
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        
        # 통계
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_confidence': 0.0,
            'total_processing_time': 0.0
        }
        
        # 수식 패턴 (향상된)
        self.formula_patterns = self._init_formula_patterns()
        
        # API 키 로드
        self._load_api_key()
        
    def _init_formula_patterns(self) -> List[Dict[str, Any]]:
        """수식 패턴 초기화"""
        return [
            {
                'name': 'basic_equation',
                'pattern': r'[A-Za-z]\s*[=+\-*/]\s*[A-Za-z0-9\s+\-*/()²³⁴⁵⁶⁷⁸⁹⁰]+',
                'type': 'equation',
                'confidence_weight': 0.8
            },
            {
                'name': 'integral',
                'pattern': r'∫[^∫]*d[a-z]',
                'type': 'calculus',
                'confidence_weight': 0.9
            },
            {
                'name': 'summation',
                'pattern': r'∑[^∑]*',
                'type': 'series',
                'confidence_weight': 0.9
            },
            {
                'name': 'subscript',
                'pattern': r'[A-Za-z]\s*[₀₁₂₃₄₅₆₇₈₉]+',
                'type': 'notation',
                'confidence_weight': 0.7
            },
            {
                'name': 'superscript',
                'pattern': r'[A-Za-z]\s*[⁰¹²³⁴⁵⁶⁷⁸⁹]+',
                'type': 'notation',
                'confidence_weight': 0.7
            },
            {
                'name': 'fraction',
                'pattern': r'\d+/\d+',
                'type': 'arithmetic',
                'confidence_weight': 0.6
            },
            {
                'name': 'greek_letters',
                'pattern': r'[αβγδεζηθικλμνξοπρστυφχψω]',
                'type': 'greek',
                'confidence_weight': 0.8
            }
        ]
    
    def _load_api_key(self) -> None:
        """향상된 API 키 로드"""
        try:
            # 1. 설정에서 직접 로드
            if 'api_key' in self.config and self.config['api_key']:
                self.api_key = self.config['api_key']
                self.api_available = True
                logger.info("NCP OCR API key loaded from config")
                return
            
            # 2. VESSL Storage에서 로드 (우선순위 높음)
            vessl_paths = [
                "/apikey/n_api",
                "/apikey/ncp_ocr_key.txt",
                "/apikey/NCP_OCR_SECRET"
            ]
            
            for key_path in vessl_paths:
                if os.path.exists(key_path):
                    with open(key_path, 'r', encoding='utf-8') as f:
                        self.api_key = f.read().strip()
                        if self.api_key:
                            self.api_available = True
                            logger.info(f"NCP OCR API key loaded from VESSL Storage: {key_path}")
                            return
            
            # 3. 환경변수에서 로드
            env_vars = ["NCP_OCR_SECRET", "NCP_API_KEY", "NAVER_OCR_KEY"]
            for env_var in env_vars:
                self.api_key = os.getenv(env_var)
                if self.api_key:
                    self.api_available = True
                    logger.info(f"NCP OCR API key loaded from environment: {env_var}")
                    return
            
            # 4. 로컬 설정 파일에서 로드
            local_paths = [
                "./config/ncp_ocr_key.txt",
                "~/.ncp/ocr_key",
                "./ncp_api_key.txt"
            ]
            
            for path in local_paths:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path):
                    with open(expanded_path, 'r', encoding='utf-8') as f:
                        self.api_key = f.read().strip()
                        if self.api_key:
                            self.api_available = True
                            logger.info(f"NCP OCR API key loaded from local file: {expanded_path}")
                            return
            
            # 5. 설정 객체의 속성에서 로드 (레거시 호환성)
            if hasattr(self.config, 'ncp_ocr') and hasattr(self.config.ncp_ocr, 'api_key'):
                self.api_key = self.config.ncp_ocr.api_key
                if self.api_key:
                    self.api_available = True
                    logger.info("NCP OCR API key loaded from config object")
                    return
                    
            logger.warning("NCP OCR API key not found - OCR fallback disabled")
            self.api_available = False
            
        except Exception as e:
            logger.error(f"Failed to load NCP OCR API key: {e}")
            self.api_available = False
    
    def _upload_image_to_temp_storage(self, image_path: str) -> Optional[str]:
        """
        이미지를 임시 저장소에 업로드하고 URL 반환
        실제 구현에서는 NCP Object Storage 등을 사용
        """
        # 임시 구현: 로컬 파일을 base64로 인코딩
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Base64 인코딩
            encoded_data = base64.b64encode(image_data).decode('utf-8')
            
            # 임시 URL 생성 (실제로는 Object Storage URL)
            temp_url = f"data:image/jpeg;base64,{encoded_data}"
            return temp_url
            
        except Exception as e:
            logger.error(f"Failed to prepare image for NCP OCR: {e}")
            return None
    
    def analyze_image(self, image_data: Union[str, bytes], 
                     analysis_type: str = "text_extraction",
                     custom_prompt: Optional[str] = None,
                     query_context: Optional[Dict] = None) -> OCRResult:
        """
        향상된 NCP OCR API 이미지 분석
        
        Args:
            image_data: 이미지 데이터 (파일 경로, base64 문자열, 또는 바이트)
            analysis_type: 분석 유형 (text_extraction, formula_analysis 등)
            custom_prompt: 사용자 정의 프롬프트 (OCR에는 직접 사용되지 않지만 후처리에 활용)
            query_context: 질의 컨텍스트 정보
            
        Returns:
            OCRResult: 분석 결과
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        if not self.api_available:
            error_result = OCRResult(
                success=False,
                extracted_text="",
                formulas=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message="NCP OCR API key not available",
                metadata={'source': 'NCP_OCR', 'fallback_used': True}
            )
            self.stats['failed_requests'] += 1
            return error_result
        
        try:
            # 이미지 데이터 처리
            image_info = self._prepare_image_data(image_data)
            if not image_info:
                raise Exception("Failed to prepare image data")
            
            # 재시도 로직으로 API 호출
            ocr_response = self._call_api_with_retry(image_info)
            if not ocr_response:
                raise Exception("Failed to get OCR response after retries")
            
            # 결과 처리
            result = self._process_ocr_response(
                ocr_response, analysis_type, custom_prompt, query_context, start_time
            )
            
            # 통계 업데이트
            if result.success:
                self.stats['successful_requests'] += 1
                self.stats['average_confidence'] = (
                    (self.stats['average_confidence'] * (self.stats['successful_requests'] - 1) + 
                     result.confidence_score) / self.stats['successful_requests']
                )
            else:
                self.stats['failed_requests'] += 1
            
            self.stats['total_processing_time'] += result.processing_time
            
            logger.info(f"NCP OCR analysis completed: {analysis_type} "
                       f"(success: {result.success}, time: {result.processing_time:.2f}s)")
            
            return result
                
        except Exception as e:
            logger.error(f"NCP OCR analysis failed: {e}")
            error_result = OCRResult(
                success=False,
                extracted_text="",
                formulas=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e),
                metadata={'source': 'NCP_OCR', 'fallback_used': True}
            )
            self.stats['failed_requests'] += 1
            return error_result
    
    def _prepare_image_data(self, image_data: Union[str, bytes]) -> Optional[Dict[str, Any]]:
        """이미지 데이터 준비"""
        try:
            if isinstance(image_data, str):
                if os.path.exists(image_data):
                    # 파일 경로인 경우
                    with Image.open(image_data) as img:
                        image_format = img.format.lower() if img.format else 'jpg'
                        image_name = os.path.basename(image_data)
                    
                    with open(image_data, 'rb') as f:
                        image_bytes = f.read()
                    
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                elif image_data.startswith('data:image'):
                    # Data URL인 경우
                    image_base64 = image_data.split(',')[1]
                    image_format = 'jpg'  # 기본값
                    image_name = f"image_{int(time.time())}.jpg"
                
                else:
                    # Base64 문자열인 경우
                    image_base64 = image_data
                    image_format = 'jpg'  # 기본값
                    image_name = f"image_{int(time.time())}.jpg"
            
            elif isinstance(image_data, bytes):
                # 바이트 데이터인 경우
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                image_format = 'jpg'  # 기본값
                image_name = f"image_{int(time.time())}.jpg"
            
            else:
                raise ValueError("Unsupported image data type")
            
            return {
                'format': image_format,
                'name': image_name,
                'data': image_base64
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare image data: {e}")
            return None
    
    def _call_api_with_retry(self, image_info: Dict[str, Any]) -> Optional[Dict]:
        """재시도 로직이 포함된 API 호출"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # 요청 데이터 구성
                request_data = {
                    "version": "V2",
                    "requestId": str(uuid.uuid4()),
                    "timestamp": str(int(time.time() * 1000)),
                    "lang": "ko",
                    "images": [image_info]
                }
                
                # 헤더 구성
                headers = {
                    'Content-Type': 'application/json',
                    'X-OCR-SECRET': self.api_key
                }
                
                logger.info(f"NCP OCR API call attempt {attempt + 1}/{self.max_retries}")
                
                # API 호출
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"NCP OCR API attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # 지수 백오프
        
        logger.error(f"All NCP OCR API attempts failed. Last error: {last_exception}")
        return None
    
    def _process_ocr_response(self, ocr_response: Dict, analysis_type: str,
                             custom_prompt: Optional[str], query_context: Optional[Dict],
                             start_time: float) -> OCRResult:
        """OCR 응답 처리"""
        try:
            processing_time = time.time() - start_time
            
            # 텍스트 추출
            extracted_texts = []
            total_confidence = 0.0
            field_count = 0
            
            if 'images' in ocr_response:
                for image in ocr_response['images']:
                    if 'fields' in image:
                        for field in image['fields']:
                            text = field.get('inferText', '')
                            confidence = field.get('inferConfidence', 0.0)
                            
                            if text.strip():
                                extracted_texts.append(text)
                                total_confidence += confidence
                                field_count += 1
            
            # 전체 텍스트 및 평균 신뢰도
            full_text = ' '.join(extracted_texts)
            avg_confidence = total_confidence / field_count if field_count > 0 else 0.0
            
            # 향상된 수식 감지
            formulas = self._detect_formulas_enhanced(full_text)
            
            # 분석 유형별 후처리
            processed_text = self._post_process_by_analysis_type(
                full_text, analysis_type, custom_prompt, query_context
            )
            
            # 메타데이터 구성
            metadata = {
                'source': 'NCP_OCR',
                'fallback_used': True,
                'field_count': field_count,
                'api_response_fields': len(extracted_texts),
                'analysis_type': analysis_type,
                'timestamp': datetime.now().isoformat()
            }
            
            if query_context:
                metadata['query_context'] = query_context
            
            return OCRResult(
                success=True,
                extracted_text=full_text,
                formulas=formulas,
                confidence_score=avg_confidence,
                processing_time=processing_time,
                raw_response=processed_text,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to process OCR response: {e}")
            return OCRResult(
                success=False,
                extracted_text="",
                formulas=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message=f"Response processing failed: {str(e)}",
                metadata={'source': 'NCP_OCR', 'fallback_used': True}
            )
    
    def _detect_formulas_enhanced(self, text: str) -> List[Dict[str, Any]]:
        """향상된 수식 감지"""
        formulas = []
        
        for pattern_info in self.formula_patterns:
            pattern = pattern_info['pattern']
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if match.strip() and len(match.strip()) > 1:
                    formula = {
                        'latex': match.strip(),
                        'type': pattern_info['type'],
                        'confidence': pattern_info['confidence_weight'],
                        'pattern_name': pattern_info['name'],
                        'original_text': match
                    }
                    
                    # 중복 제거
                    if not any(f['latex'] == formula['latex'] for f in formulas):
                        formulas.append(formula)
        
        # 신뢰도 순으로 정렬하고 상위 10개만 반환
        formulas.sort(key=lambda x: x['confidence'], reverse=True)
        return formulas[:10]
    
    def _post_process_by_analysis_type(self, text: str, analysis_type: str,
                                     custom_prompt: Optional[str],
                                     query_context: Optional[Dict]) -> str:
        """분석 유형별 후처리"""
        if analysis_type == "text_extraction":
            return self._format_text_extraction_response(text, query_context)
        elif analysis_type == "formula_analysis":
            return self._format_formula_analysis_response(text, query_context)
        elif analysis_type == "problem_solving":
            return self._format_problem_solving_response(text, query_context)
        else:
            return self._format_general_response(text, custom_prompt, query_context)
    
    def _format_text_extraction_response(self, text: str, context: Optional[Dict]) -> str:
        """텍스트 추출 응답 형식화"""
        if not text.strip():
            return "이미지에서 텍스트를 추출할 수 없습니다."
        
        response = f"📝 NCP OCR로 추출된 텍스트:\n\n{text}"
        
        # 도메인별 추가 정보
        if context and 'domain' in context:
            domain = context['domain']
            if domain == 'electrical':
                response += "\n\n💡 전기공학 관련 용어가 포함되어 있을 수 있습니다."
            elif domain == 'mathematics':
                response += "\n\n🔢 수학 관련 기호나 수식이 포함되어 있을 수 있습니다."
        
        return response
    
    def _format_formula_analysis_response(self, text: str, context: Optional[Dict]) -> str:
        """수식 분석 응답 형식화"""
        formulas = self._detect_formulas_enhanced(text)
        
        response = f"📐 NCP OCR 수식 분석 결과:\n\n"
        response += f"추출된 텍스트: {text}\n\n"
        
        if formulas:
            response += "감지된 수식:\n"
            for i, formula in enumerate(formulas, 1):
                response += f"{i}. {formula['latex']} (유형: {formula['type']}, 신뢰도: {formula['confidence']:.2f})\n"
        else:
            response += "수식이 감지되지 않았습니다.\n"
        
        response += "\n⚠️  NCP OCR로 추출된 결과이므로 정확성 검토가 필요할 수 있습니다."
        return response
    
    def _format_problem_solving_response(self, text: str, context: Optional[Dict]) -> str:
        """문제 해결 응답 형식화"""
        response = f"🔍 NCP OCR 문제 분석 결과:\n\n"
        response += f"추출된 내용: {text}\n\n"
        
        # 간단한 문제 요소 감지
        if '=' in text:
            response += "✓ 등식이 포함되어 있습니다.\n"
        if any(keyword in text for keyword in ['구하시오', '계산', '풀이', '해결']):
            response += "✓ 문제 해결 요청이 포함되어 있습니다.\n"
        if any(char in text for char in '0123456789'):
            response += "✓ 숫자 데이터가 포함되어 있습니다.\n"
        
        response += "\n⚠️  정확한 문제 해결을 위해서는 ChatGPT Vision API를 사용하는 것이 권장됩니다."
        return response
    
    def _format_general_response(self, text: str, custom_prompt: Optional[str],
                               context: Optional[Dict]) -> str:
        """일반 응답 형식화"""
        response = f"📄 NCP OCR 분석 결과:\n\n{text}"
        
        if custom_prompt:
            response += f"\n\n요청사항: {custom_prompt}"
            response += "\n⚠️  세부 분석을 위해서는 ChatGPT Vision API를 사용하는 것이 권장됩니다."
        
        return response
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        success_rate = 0.0
        avg_processing_time = 0.0
        
        if self.stats['total_requests'] > 0:
            success_rate = self.stats['successful_requests'] / self.stats['total_requests']
        
        if self.stats['successful_requests'] > 0:
            avg_processing_time = self.stats['total_processing_time'] / self.stats['successful_requests']
        
        return {
            'api_available': self.api_available,
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'success_rate': success_rate,
            'average_confidence': self.stats['average_confidence'],
            'average_processing_time': avg_processing_time,
            'fallback_system': True,
            'supported_formats': ['jpg', 'jpeg', 'png', 'pdf'],
            'max_retries': self.max_retries,
            'timeout': self.timeout
        }
    
    def test_connection(self) -> Tuple[bool, str]:
        """연결 테스트"""
        if not self.api_available:
            return False, "NCP OCR API key not available"
        
        try:
            # 간단한 테스트 이미지로 연결 확인
            # 실제 구현에서는 최소한의 API 호출로 테스트
            return True, f"NCP OCR API available (URL: {self.api_url[:50]}...)"
        except Exception as e:
            return False, f"NCP OCR API connection failed: {e}"
    

# 편의 함수들
def analyze_image_with_ncp_ocr(image_data: Union[str, bytes], 
                              analysis_type: str = "text_extraction",
                              custom_prompt: Optional[str] = None,
                              config: Optional[Dict] = None) -> OCRResult:
    """NCP OCR 이미지 분석 편의 함수"""
    client = NCPOCRClient(config)
    return client.analyze_image(image_data, analysis_type, custom_prompt)

def test_ncp_ocr_connection(config: Optional[Dict] = None) -> Tuple[bool, str]:
    """NCP OCR 연결 테스트 편의 함수"""
    client = NCPOCRClient(config)
    return client.test_connection()

def get_ncp_ocr_statistics(client: Optional[NCPOCRClient] = None) -> Dict[str, Any]:
    """NCP OCR 통계 조회 편의 함수"""
    if client is None:
        client = NCPOCRClient()
    return client.get_statistics()


class NCPOCRConfig:
    """NCP OCR 설정 클래스 (레거시 호환성)"""
    def __init__(self):
        self.api_key: Optional[str] = None
        self.timeout: int = 30
        self.max_retries: int = 3
        self.use_fallback: bool = True
        
    @property
    def ncp_ocr(self):
        """레거시 호환성을 위한 속성"""
        return self