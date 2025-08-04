#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask API Server
Flask API 서버 - 통합 오케스트레이션 시스템
"""

import os
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import time
import asyncio
from typing import Dict, Any, Optional

# 설정
from config import Config

# LLM 클라이언트
from llm_client import LLMClient

# 통합 오케스트레이션 시스템
try:
    from unified_orchestration_system import UnifiedOrchestrationSystem, ProcessingMode
    unified_system_available = True
except ImportError:
    unified_system_available = False
    logging.warning("Unified Orchestration System not available")

# 기존 시스템들 (폴백용)
from advanced_rag_system import AdvancedRAGSystem, create_advanced_rag_system
from enhanced_rag_system import EnhancedVectorDatabase

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask 앱 초기화
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 설정 로드
config = Config()

# LLM 클라이언트 초기화
llm_client = LLMClient(config.llm)

# 벡터 데이터베이스 초기화
vector_db = EnhancedVectorDatabase(
    persist_directory=config.rag.persist_directory
)

# 시스템 초기화
if unified_system_available:
    # 통합 오케스트레이션 시스템
    orchestration_config = {
        'llm_client': llm_client,
        'vector_db': vector_db,
        'cache_size_limit': 1000,
        'adaptation_threshold': 100
    }
    unified_system = UnifiedOrchestrationSystem(orchestration_config)
    logger.info("Unified Orchestration System initialized")
else:
    # 폴백: Advanced RAG System
    unified_system = None
    advanced_rag = create_advanced_rag_system(config, llm_client)
    logger.info("Using Advanced RAG System as fallback")


def allowed_file(filename):
    """파일 확장자 확인"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_base64_image(base64_string):
    """Base64 이미지 디코딩"""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None


@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({
        'status': 'healthy',
        'unified_system': unified_system_available,
        'timestamp': time.time()
    })


@app.route('/chat', methods=['POST'])
def chat():
    """채팅 엔드포인트"""
    try:
        # 요청 데이터 파싱
        data = request.get_json()
        query = data.get('query', '')
        user_id = data.get('user_id', 'anonymous')
        mode = data.get('mode', None)  # 처리 모드 (선택적)
        image_base64 = data.get('image', None)
        context = data.get('context', {})
        
        if not query:
            return jsonify({
                'error': 'Query is required',
                'success': False
            }), 400
        
        # 이미지 처리
        image = None
        if image_base64:
            image = decode_base64_image(image_base64)
            if image is None:
                return jsonify({
                    'error': 'Invalid image format',
                    'success': False
                }), 400
        
        # 처리
        if unified_system:
            # 통합 오케스트레이션 시스템 사용
            processing_mode = None
            if mode:
                try:
                    processing_mode = ProcessingMode(mode)
                except ValueError:
                    logger.warning(f"Invalid mode: {mode}, using auto mode")
            
            # 비동기 처리를 동기 환경에서 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    unified_system.process(
                        query=query,
                        user_id=user_id,
                        image=image,
                        mode=processing_mode,
                        context=context
                    )
                )
            finally:
                loop.close()
        else:
            # 폴백: Advanced RAG System
            result = advanced_rag.process_query_advanced(
                query=query,
                image=image,
                mode=mode or 'reasoning',
                response_style=context.get('response_style', 'comprehensive')
            )
        
        # 응답 포맷팅
        response = {
            'success': result.get('success', True),
            'query': query,
            'response': result.get('response', ''),
            'processing_time': result.get('processing_time', 0),
            'mode': result.get('mode', 'unknown'),
            'metadata': {
                'confidence': result.get('confidence', 0),
                'domains': result.get('domains', []),
                'from_cache': result.get('from_cache', False)
            }
        }
        
        # 추가 정보 (요청 시)
        if context.get('include_details', False):
            response['details'] = {
                'reasoning_trace': result.get('reasoning_trace', []),
                'evidence': result.get('evidence', []),
                'recommendations': result.get('learning_recommendations', []),
                'alternative_responses': result.get('alternative_responses', [])
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    """피드백 엔드포인트"""
    if not unified_system:
        return jsonify({
            'error': 'Feedback system not available',
            'success': False
        }), 503
    
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'anonymous')
        query = data.get('query', '')
        response = data.get('response', '')
        rating = data.get('rating', 3)
        feedback_text = data.get('feedback', None)
        
        # 피드백 수집
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                unified_system.collect_feedback(
                    user_id=user_id,
                    query=query,
                    response=response,
                    rating=rating,
                    feedback=feedback_text
                )
            )
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'message': 'Feedback collected successfully'
        })
        
    except Exception as e:
        logger.error(f"Feedback endpoint error: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/status', methods=['GET'])
def system_status():
    """시스템 상태 조회"""
    if unified_system:
        status = unified_system.get_system_status()
        return jsonify({
            'success': True,
            'status': status
        })
    else:
        return jsonify({
            'success': True,
            'status': {
                'state': 'fallback',
                'message': 'Using Advanced RAG System',
                'subsystems': {
                    'advanced_rag': True,
                    'unified_orchestration': False
                }
            }
        })


@app.route('/modes', methods=['GET'])
def available_modes():
    """사용 가능한 처리 모드 조회"""
    if unified_system:
        modes = [
            {
                'value': 'cognitive',
                'name': 'Cognitive AI',
                'description': '인간 인지 프로세스를 모방한 고급 처리'
            },
            {
                'value': 'intelligent',
                'name': 'Intelligent RAG',
                'description': '지능형 의도 감지 및 적응형 응답'
            },
            {
                'value': 'advanced',
                'name': 'Advanced RAG',
                'description': '리랭킹과 추론 체인을 갖춘 고급 RAG'
            },
            {
                'value': 'modular',
                'name': 'Modular RAG',
                'description': '모듈형 파이프라인 처리'
            },
            {
                'value': 'adaptive',
                'name': 'Adaptive',
                'description': '자동 모드 선택 및 하이브리드 처리'
            }
        ]
    else:
        modes = [
            {
                'value': 'fast',
                'name': 'Fast Mode',
                'description': '빠른 기본 검색'
            },
            {
                'value': 'balanced',
                'name': 'Balanced Mode',
                'description': '리랭킹을 활용한 균형잡힌 처리'
            },
            {
                'value': 'reasoning',
                'name': 'Reasoning Mode',
                'description': '추론 체인을 활용한 심층 처리'
            }
        ]
    
    return jsonify({
        'success': True,
        'modes': modes
    })


@app.route('/test', methods=['GET'])
def test_endpoint():
    """테스트 엔드포인트"""
    test_queries = [
        {
            'query': '양자컴퓨터의 큐비트는 어떻게 작동하나요?',
            'domain': 'physics/computer_science'
        },
        {
            'query': '딥러닝에서 배치 정규화의 역할은 무엇인가요?',
            'domain': 'ai/machine_learning'
        },
        {
            'query': '미적분학에서 연쇄 법칙을 설명해주세요.',
            'domain': 'mathematics'
        },
        {
            'query': 'DNA 복제 과정을 단계별로 설명해주세요.',
            'domain': 'biology'
        },
        {
            'query': '시장경제에서 수요와 공급의 균형은 어떻게 이루어지나요?',
            'domain': 'economics'
        }
    ]
    
    results = []
    for test_case in test_queries:
        try:
            # 각 쿼리 처리
            if unified_system:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        unified_system.process(
                            query=test_case['query'],
                            user_id='test_user'
                        )
                    )
                finally:
                    loop.close()
            else:
                result = advanced_rag.process_query_advanced(
                    query=test_case['query'],
                    mode='balanced'
                )
            
            results.append({
                'query': test_case['query'],
                'expected_domain': test_case['domain'],
                'success': result.get('success', False),
                'detected_domains': result.get('domains', []),
                'processing_time': result.get('processing_time', 0),
                'response_preview': result.get('response', '')[:200] + '...'
            })
            
        except Exception as e:
            results.append({
                'query': test_case['query'],
                'expected_domain': test_case['domain'],
                'success': False,
                'error': str(e)
            })
    
    return jsonify({
        'success': True,
        'test_results': results,
        'system_type': 'unified' if unified_system else 'advanced_rag'
    })


@app.errorhandler(404)
def not_found(error):
    """404 에러 핸들러"""
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500 에러 핸들러"""
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500


if __name__ == '__main__':
    # 개발 서버 실행
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True
    )