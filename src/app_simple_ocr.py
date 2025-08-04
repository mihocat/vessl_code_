#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 OCR 테스트 애플리케이션
MathPix 스타일 OCR의 최소 구현
"""

import gradio as gr
from PIL import Image
import numpy as np
import logging
import asyncio
from typing import Optional, Dict, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleOCRSystem:
    """간단한 OCR 시스템"""
    
    def __init__(self):
        self.initialized = False
        self.models = {}
        
    async def initialize(self):
        """시스템 초기화"""
        try:
            logger.info("Initializing Simple OCR System...")
            
            # EasyOCR 초기화
            try:
                import easyocr
                self.models['easyocr'] = easyocr.Reader(['ko', 'en'], gpu=False)
                logger.info("EasyOCR initialized")
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
            
            # TrOCR 초기화 시도
            try:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                self.models['trocr_processor'] = TrOCRProcessor.from_pretrained(
                    'microsoft/trocr-base-printed'
                )
                self.models['trocr_model'] = VisionEncoderDecoderModel.from_pretrained(
                    'microsoft/trocr-base-printed'
                )
                logger.info("TrOCR initialized")
            except Exception as e:
                logger.warning(f"TrOCR initialization failed: {e}")
            
            self.initialized = True
            logger.info("OCR System initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR System: {e}")
            raise
    
    async def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """이미지 처리"""
        result = {
            'status': 'processing',
            'ocr_results': [],
            'error': None
        }
        
        try:
            # numpy 배열로 변환
            img_array = np.array(image)
            
            # EasyOCR 처리
            if 'easyocr' in self.models:
                try:
                    easyocr_results = self.models['easyocr'].readtext(img_array)
                    texts = []
                    for bbox, text, conf in easyocr_results:
                        texts.append({
                            'text': text,
                            'confidence': conf,
                            'bbox': bbox
                        })
                    result['ocr_results'].append({
                        'engine': 'EasyOCR',
                        'texts': texts
                    })
                except Exception as e:
                    logger.error(f"EasyOCR processing failed: {e}")
            
            # TrOCR 처리
            if 'trocr_processor' in self.models and 'trocr_model' in self.models:
                try:
                    # 간단한 TrOCR 처리
                    import torch
                    
                    processor = self.models['trocr_processor']
                    model = self.models['trocr_model']
                    
                    # 전처리
                    inputs = processor(images=image, return_tensors="pt")
                    
                    # 추론
                    with torch.no_grad():
                        generated_ids = model.generate(inputs.pixel_values)
                    
                    # 디코딩
                    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    result['ocr_results'].append({
                        'engine': 'TrOCR',
                        'texts': [{'text': text, 'confidence': 0.0}]
                    })
                except Exception as e:
                    logger.error(f"TrOCR processing failed: {e}")
            
            result['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    ocr_system = SimpleOCRSystem()
    
    async def process_wrapper(image):
        if not ocr_system.initialized:
            await ocr_system.initialize()
        
        if image is None:
            return "이미지를 업로드해주세요."
        
        # PIL Image로 변환
        pil_image = Image.fromarray(image)
        
        # OCR 처리
        result = await ocr_system.process_image(pil_image)
        
        # 결과 포맷팅
        output = "## OCR 결과\n\n"
        
        if result['status'] == 'failed':
            output += f"오류 발생: {result['error']}\n"
        else:
            for ocr_result in result['ocr_results']:
                output += f"### {ocr_result['engine']}\n"
                for text_item in ocr_result['texts']:
                    output += f"- {text_item['text']}"
                    if text_item['confidence'] > 0:
                        output += f" (신뢰도: {text_item['confidence']:.2f})"
                    output += "\n"
                output += "\n"
        
        return output
    
    # 동기 래퍼
    def sync_process(image):
        return asyncio.run(process_wrapper(image))
    
    # Gradio 인터페이스
    with gr.Blocks(title="간단한 OCR 테스트") as demo:
        gr.Markdown("""
        # 간단한 OCR 테스트 시스템
        
        이미지에서 텍스트를 추출하는 기본 OCR 시스템입니다.
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="이미지 업로드",
                    type="numpy"
                )
                submit_btn = gr.Button("OCR 실행", variant="primary")
            
            with gr.Column():
                output = gr.Markdown(
                    label="OCR 결과",
                    value="여기에 OCR 결과가 표시됩니다."
                )
        
        submit_btn.click(
            fn=sync_process,
            inputs=image_input,
            outputs=output
        )
        
        # 예시 이미지 경로가 있다면 추가
        gr.Examples(
            examples=[
                # ["examples/math_formula.png"],
                # ["examples/korean_text.png"]
            ],
            inputs=image_input
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )