#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 키 로더
VESSL Storage에서 API 키를 안전하게 로드
"""

import os
import logging

logger = logging.getLogger(__name__)


def load_openai_key_from_storage():
    """
    VESSL Storage에서 OpenAI API 키 로드
    경로: /apikey/o_api 또는 /dataset/o_api
    """
    storage_paths = [
        "/apikey/o_api",  # VESSL Storage APIkey 볼륨
        "/dataset/o_api",  # VESSL Storage 경로
        "./o_api",  # 로컬 테스트용
        "../o_api"   # 대체 경로
    ]
    
    for path in storage_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    api_key = f.read().strip()
                    
                if api_key and api_key.startswith('sk-'):
                    logger.info(f"OpenAI API key loaded from: {path}")
                    return api_key
                else:
                    logger.warning(f"Invalid API key format in: {path}")
                    
            except Exception as e:
                logger.error(f"Failed to read API key from {path}: {e}")
    
    logger.warning("OpenAI API key not found in storage")
    return None


def setup_api_keys():
    """
    API 키 설정
    우선순위:
    1. 환경 변수 (이미 설정된 경우)
    2. VESSL Storage
    3. .env 파일 (로컬 개발)
    """
    # 이미 환경 변수에 설정되어 있으면 스킵
    if os.getenv("OPENAI_API_KEY"):
        logger.info("OpenAI API key already set in environment")
        return True
    
    # VESSL Storage에서 로드
    api_key = load_openai_key_from_storage()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        logger.info("OpenAI API key loaded from storage and set to environment")
        return True
    
    # .env 파일에서 로드 (로컬 개발용)
    try:
        from dotenv import load_dotenv
        if load_dotenv():
            if os.getenv("OPENAI_API_KEY"):
                logger.info("OpenAI API key loaded from .env file")
                return True
    except ImportError:
        pass
    
    logger.warning("No OpenAI API key found in any location")
    return False


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    if setup_api_keys():
        print("✅ API key setup successful")
        print(f"Key starts with: {os.getenv('OPENAI_API_KEY')[:7]}...")
    else:
        print("❌ API key setup failed")