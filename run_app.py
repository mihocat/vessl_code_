#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application Runner - Choose between standard and enhanced version
애플리케이션 실행기 - 표준 버전과 향상된 버전 중 선택
"""

import sys
import os
import argparse
import logging

# src 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_enhanced_requirements():
    """향상된 버전의 요구사항 확인"""
    try:
        # 필수 모듈 확인
        import enhanced_rag_system
        import enhanced_image_analyzer
        import chatgpt_response_generator
        import multimodal_ocr
        return True
    except ImportError as e:
        logger.warning(f"Enhanced modules not fully available: {e}")
        return False


def check_advanced_requirements():
    """고급 버전의 요구사항 확인"""
    try:
        # 고급 UI 모듈 확인
        import advanced_ui
        import visualization_components
        # matplotlib 확인
        import matplotlib.pyplot as plt
        return True
    except ImportError as e:
        logger.warning(f"Advanced modules not fully available: {e}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="AI 전기공학 튜터 실행기")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "enhanced", "advanced", "auto"],
        default="auto",
        help="실행 모드 선택 (standard: 기본, enhanced: 향상된, advanced: 고급, auto: 자동 선택)"
    )
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
    
    args = parser.parse_args()
    
    # 실행 모드 결정
    if args.mode == "auto":
        # 고급 버전 요구사항 확인
        if check_advanced_requirements():
            run_mode = "advanced"
            logger.info("Auto mode: Advanced version available, using advanced app")
        elif check_enhanced_requirements():
            run_mode = "enhanced"
            logger.info("Auto mode: Enhanced version available, using enhanced app")
        else:
            run_mode = "standard"
            logger.info("Auto mode: Using standard app")
    elif args.mode == "advanced":
        run_mode = "advanced"
        if not check_advanced_requirements():
            logger.warning("Advanced mode requested but some modules may be missing")
    elif args.mode == "enhanced":
        run_mode = "enhanced"
        if not check_enhanced_requirements():
            logger.warning("Enhanced mode requested but some modules may be missing")
    else:
        run_mode = "standard"
    
    # 선택된 앱 실행
    try:
        # sys.argv 백업
        original_argv = sys.argv
        
        if run_mode == "advanced":
            logger.info("Starting advanced application...")
            from advanced_ui import main as advanced_main
            
            sys.argv = ["advanced_ui.py"]
            if args.server_port != 7860:
                sys.argv.extend(["--server-port", str(args.server_port)])
            if args.share:
                sys.argv.append("--share")
            
            advanced_main()
            
        elif run_mode == "enhanced":
            logger.info("Starting enhanced application...")
            from enhanced_app import main as enhanced_main
            
            sys.argv = ["enhanced_app.py"]
            if args.server_port != 7860:
                sys.argv.extend(["--server-port", str(args.server_port)])
            if args.share:
                sys.argv.append("--share")
            
            enhanced_main()
            
        else:
            logger.info("Starting standard application...")
            from app import main as standard_main
            
            sys.argv = ["app.py"]
            if args.server_port != 7860:
                sys.argv.extend(["--server-port", str(args.server_port)])
            if args.share:
                sys.argv.append("--share")
            
            standard_main()
        
        # sys.argv 복원
        sys.argv = original_argv
            
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()