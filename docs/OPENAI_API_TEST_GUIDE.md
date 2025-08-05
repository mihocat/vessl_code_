# OpenAI API 테스트 가이드

## 1. 환경 설정

### API 키 설정
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 2. 모델 옵션

### Vision 모델
- **gpt-4-vision-preview**: 이미지 분석 가능, 텍스트 생성
- **gpt-4o**: 최신 모델, 더 나은 성능
- **gpt-4o-mini**: 비용 효율적, 빠른 응답

### 텍스트 전용 모델
- **gpt-4-turbo**: 128K 컨텍스트 (현재 2048 토큰 문제 해결)
- **gpt-3.5-turbo**: 16K 컨텍스트, 빠르고 저렴

## 3. 통합 방법

### 옵션 1: 이미지 분석 전용
```python
# enhanced_multimodal_processor.py 수정
from openai_vision_analyzer import OpenAIVisionAnalyzer

class EnhancedMultimodalProcessor:
    def __init__(self, use_openai=False):
        if use_openai:
            self.vision_analyzer = OpenAIVisionAnalyzer()
```

### 옵션 2: LLM 응답 생성 개선
```python
# llm_client.py 수정
class OpenAILLMClient:
    def __init__(self, model="gpt-4-turbo"):
        self.client = OpenAI()
        self.model = model  # 더 큰 컨텍스트
```

## 4. 테스트 스크립트

```python
#!/usr/bin/env python3
"""OpenAI API 테스트"""

import os
from openai_vision_analyzer import OpenAIVisionAnalyzer
from PIL import Image

# 1. Vision API 테스트
def test_vision_api():
    analyzer = OpenAIVisionAnalyzer()
    
    # EXAMPLE.md의 이미지로 테스트
    test_images = [
        "path/to/image1.jpg",  # 수식 이미지
        "path/to/image2.jpg",  # 한국어 텍스트 이미지
    ]
    
    for img_path in test_images:
        result = analyzer.analyze_image(
            img_path,
            question="이미지의 모든 텍스트와 수식을 추출해주세요"
        )
        print(f"\n=== {img_path} ===")
        print(result["raw_response"])
        print(f"토큰 사용량: {result['usage']['total_tokens']}")

# 2. 텍스트 모델 테스트
def test_text_model():
    from openai import OpenAI
    client = OpenAI()
    
    # 긴 컨텍스트 테스트
    long_context = "매우 긴 텍스트..." * 100  # 2048 토큰 이상
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": long_context}
        ]
    )
    print(f"응답: {response.choices[0].message.content[:100]}...")

if __name__ == "__main__":
    test_vision_api()
    test_text_model()
```

## 5. 장단점 비교

### 현재 시스템
- **장점**: 
  - 로컬 모델 사용으로 비용 없음
  - 데이터 프라이버시
  - 커스터마이징 가능
- **단점**:
  - 수식 인식 정확도 낮음
  - 컨텍스트 길이 제한 (2048)
  - 복잡한 파이프라인

### OpenAI API
- **장점**:
  - 뛰어난 이미지 이해력
  - 한국어/수식 동시 처리
  - 간단한 통합
  - 큰 컨텍스트 (128K)
- **단점**:
  - API 비용 발생
  - 인터넷 연결 필요
  - 데이터 외부 전송

## 6. 비용 추정

### Vision API
- Input: ~$0.01 per image
- Output: ~$0.03 per 1K tokens

### Text API (GPT-4 Turbo)
- Input: $0.01 per 1K tokens
- Output: $0.03 per 1K tokens

예상 사용량:
- 이미지당 평균 500 토큰 입력, 500 토큰 출력
- 비용: 약 $0.02 per request

## 7. 권장 하이브리드 접근법

1. **1차 처리**: 로컬 OCR (Tesseract, EasyOCR)
2. **2차 처리**: 수식이나 복잡한 이미지만 OpenAI Vision API
3. **응답 생성**: GPT-4 Turbo (큰 컨텍스트)

이렇게 하면:
- 비용 최적화
- 높은 정확도
- 빠른 응답 속도