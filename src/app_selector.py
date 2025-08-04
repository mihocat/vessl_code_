#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
앱 선택기 - 기본 또는 고급 시스템 선택
"""

import os
import sys
import logging
import argparse

logger = logging.getLogger(__name__)

def main():
    """메인 함수 - 환경변수나 인자에 따라 적절한 앱 실행"""
    
    parser = argparse.ArgumentParser(description="AI System App Selector")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["basic", "advanced"],
        default=os.environ.get("APP_MODE", "basic"),
        help="App mode: basic (기존 RAG) or advanced (고급 통합 시스템)"
    )
    parser.add_argument(
        "--server-port", 
        type=int, 
        default=7860,
        help="Server port (default: 7860)"
    )
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Create public Gradio link"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting app in {args.mode} mode...")
    
    if args.mode == "advanced":
        # 고급 통합 시스템 실행
        try:
            from advanced_app import create_advanced_interface
            logger.info("Loading advanced integrated AI system...")
            
            demo = create_advanced_interface()
            demo.launch(
                server_name="0.0.0.0",
                server_port=args.server_port,
                share=args.share
            )
        except ImportError as e:
            logger.error(f"Failed to import advanced system: {e}")
            logger.info("Falling back to basic system...")
            args.mode = "basic"
    
    if args.mode == "basic":
        # 기존 RAG 시스템 실행
        from app import create_gradio_app, Config
        
        config = Config()
        config.app.server_port = args.server_port
        config.app.share = args.share
        
        app = create_gradio_app(config)
        app.launch(
            server_name=config.app.server_name,
            server_port=config.app.server_port,
            share=config.app.share
        )


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()