#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application Runner - Enhanced version with improved multimodal processing
애플리케이션 실행기 - 향상된 멀티모달 처리 버전
"""

import sys
import os
import argparse
import logging

# src 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="AI 챗봇 서버 실행기")
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="서버 포트 (기본값: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="공개 Gradio 링크 생성"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        help="실행 모드 (auto, standard, enhanced)"
    )
    parser.add_argument(
        "--use-enhanced",
        action="store_true",
        help="향상된 멀티모달 처리 사용"
    )
    
    args = parser.parse_args()
    
    # 환경 변수로 향상된 모드 설정 가능
    use_enhanced = args.use_enhanced or os.environ.get("USE_ENHANCED_APP", "true").lower() == "true"
    
    # 앱 실행
    try:
        if use_enhanced:
            logger.info("Starting enhanced application with improved multimodal processing...")
            from app_v2 import main as app_main
        else:
            logger.info("Starting standard application...")
            from app import main as app_main
        
        # sys.argv 백업
        original_argv = sys.argv
        
        # 새로운 argv 설정
        sys.argv = ["app.py"]
        if args.server_port != 7860:
            sys.argv.extend(["--server-port", str(args.server_port)])
        if args.share:
            sys.argv.append("--share")
        
        # 앱 실행
        app_main()
        
        # sys.argv 복원
        sys.argv = original_argv
            
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        logger.exception("Detailed error:")
        sys.exit(1)


if __name__ == "__main__":
    main()