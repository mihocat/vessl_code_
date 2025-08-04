#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application Runner - Simplified single mode
애플리케이션 실행기 - 단순화된 단일 모드
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
        help="호환성을 위해 유지 (무시됨)"
    )
    
    args = parser.parse_args()
    
    # 표준 앱 실행
    try:
        logger.info("Starting application...")
        from app import main as app_main
        
        # sys.argv 백업
        original_argv = sys.argv
        
        sys.argv = ["app.py"]
        if args.server_port != 7860:
            sys.argv.extend(["--server-port", str(args.server_port)])
        if args.share:
            sys.argv.append("--share")
        
        app_main()
        
        # sys.argv 복원
        sys.argv = original_argv
            
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()