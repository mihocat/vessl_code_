#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional Imports Module
환경 변수에 따른 조건부 임포트 관리
"""

import os
import logging

logger = logging.getLogger(__name__)

# OpenAI Vision API 사용 여부 확인
USE_OPENAI_VISION = os.getenv("USE_OPENAI_VISION", "false").lower() == "true"

# OCR 라이브러리 조건부 임포트
EASYOCR_AVAILABLE = False
PADDLEOCR_AVAILABLE = False
TESSERACT_AVAILABLE = False

if not USE_OPENAI_VISION:
    # OpenAI Vision API를 사용하지 않을 때만 OCR 라이브러리 로드
    try:
        import easyocr
        EASYOCR_AVAILABLE = True
        logger.info("EasyOCR loaded successfully")
    except ImportError:
        logger.warning("EasyOCR not available")
    
    try:
        from paddleocr import PaddleOCR
        PADDLEOCR_AVAILABLE = True
        logger.info("PaddleOCR loaded successfully")
    except ImportError:
        logger.warning("PaddleOCR not available")
    
    try:
        import pytesseract
        from PIL import Image
        TESSERACT_AVAILABLE = True
        logger.info("Tesseract loaded successfully")
    except ImportError:
        logger.warning("Tesseract not available")
else:
    logger.info("Using OpenAI Vision API - skipping local OCR library imports")

# Florence2 (이미지 분석) 조건부 임포트
FLORENCE_AVAILABLE = False
if not USE_OPENAI_VISION:
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        FLORENCE_AVAILABLE = True
        logger.info("Florence2 available for image analysis")
    except ImportError:
        logger.warning("Florence2 not available")

# TrOCR (수식 인식) 조건부 임포트
TROCR_AVAILABLE = False
if not USE_OPENAI_VISION:
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        TROCR_AVAILABLE = True
        logger.info("TrOCR available for formula recognition")
    except ImportError:
        logger.warning("TrOCR not available")