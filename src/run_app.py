#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application Runner
환경 변수에 따라 적절한 앱 실행
"""

import os
import sys
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """메인 실행 함수"""
    # 환경 변수 확인
    use_enhanced_app = os.getenv("USE_ENHANCED_APP", "false").lower() == "true"
    use_openai_vision = os.getenv("USE_OPENAI_VISION", "false").lower() == "true"
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    
    logger.info(f"USE_ENHANCED_APP: {use_enhanced_app}")
    logger.info(f"USE_OPENAI_VISION: {use_openai_vision}")
    logger.info(f"GRADIO_SERVER_NAME: {server_name}")
    logger.info(f"GRADIO_SERVER_PORT: {server_port}")
    
    # API 키 확인
    if use_openai_vision:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            logger.info("OpenAI API key is configured")
        else:
            logger.warning("OpenAI API key not found - Vision features will be disabled")
            use_openai_vision = False
    
    if use_enhanced_app:
        logger.info("Starting Enhanced Multimodal App...")
        try:
            # Enhanced multimodal processor를 사용하는 앱 실행
            from enhanced_multimodal_processor import EnhancedMultimodalProcessor
            
            # 프로세서 초기화 시 OpenAI Vision API 설정 전달
            processor = EnhancedMultimodalProcessor(
                use_gpu=True,
                use_openai_vision=use_openai_vision
            )
            
            # app_v2 실행 (enhanced 기능 포함)
            from app_v2 import launch_app
            logger.info(f"Launching Enhanced Gradio app on {server_name}:{server_port}")
            launch_app()
            
        except ImportError as e:
            logger.error(f"Failed to import enhanced app components: {e}")
            logger.info("Falling back to standard app...")
            from app import launch_app
            logger.info(f"Launching Standard Gradio app on {server_name}:{server_port}")
            launch_app()
    else:
        logger.info("Starting Standard App...")
        from app import launch_app
        logger.info(f"Launching Standard Gradio app on {server_name}:{server_port}")
        launch_app()

if __name__ == "__main__":
    main()