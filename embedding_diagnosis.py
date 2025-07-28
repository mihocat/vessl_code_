#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Model Diagnosis Tool
임베딩 모델 진단 도구
"""

import logging
import traceback
from sentence_transformers import SentenceTransformer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_embedding_model(model_name: str):
    """임베딩 모델 상세 진단"""
    print(f"\n{'='*60}")
    print(f"임베딩 모델 진단: {model_name}")
    print(f"{'='*60}")
    
    # 1. trust_remote_code 없이 시도
    print("\n1. 기본 로딩 시도 (trust_remote_code=False)...")
    try:
        model = SentenceTransformer(model_name)
        print("✅ 성공: 기본 로딩으로 모델 로드 완료")
        test_model(model, model_name)
        return model
    except Exception as e:
        print(f"❌ 실패: {e}")
        print(f"에러 타입: {type(e).__name__}")
    
    # 2. trust_remote_code=True로 시도
    print("\n2. 원격 코드 신뢰 모드 시도 (trust_remote_code=True)...")
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
        print("✅ 성공: trust_remote_code=True로 모델 로드 완료")
        test_model(model, model_name)
        return model
    except Exception as e:
        print(f"❌ 실패: {e}")
        print(f"에러 타입: {type(e).__name__}")
        print(f"상세 에러:\n{traceback.format_exc()}")
    
    # 3. 폴백 모델 테스트
    print("\n3. 폴백 모델 테스트...")
    fallback_models = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/distiluse-base-multilingual-cased"
    ]
    
    for fallback in fallback_models:
        try:
            print(f"폴백 시도: {fallback}")
            model = SentenceTransformer(fallback)
            print(f"✅ 성공: 폴백 모델 {fallback} 로드 완료")
            test_model(model, fallback)
            return model
        except Exception as e:
            print(f"❌ 폴백 실패 {fallback}: {e}")
    
    print("\n❌ 모든 모델 로딩 실패")
    return None

def test_model(model, model_name):
    """모델 기능 테스트"""
    print(f"\n모델 테스트: {model_name}")
    print("-" * 40)
    
    try:
        # 기본 정보
        print(f"임베딩 차원: {model.get_sentence_embedding_dimension()}")
        print(f"최대 시퀀스 길이: {getattr(model, 'max_seq_length', 'Unknown')}")
        print(f"디바이스: {getattr(model, 'device', 'Unknown')}")
        print(f"모델 클래스: {model.__class__.__name__}")
        
        # 모듈 정보
        if hasattr(model, '_modules'):
            modules = list(model._modules.keys())
            print(f"포함된 모듈: {modules}")
        
        # 토크나이저 정보
        if hasattr(model, 'tokenizer'):
            tokenizer_type = type(model.tokenizer).__name__
            print(f"토크나이저 타입: {tokenizer_type}")
        
        # 간단한 인코딩 테스트
        test_texts = ["안녕하세요", "Hello", "전기공학 테스트"]
        print(f"\n테스트 텍스트: {test_texts}")
        
        embeddings = model.encode(test_texts)
        print(f"임베딩 형태: {embeddings.shape}")
        print(f"임베딩 타입: {type(embeddings).__name__}")
        print(f"첫 번째 임베딩 샘플 (처음 5개 값): {embeddings[0][:5]}")
        
        # 유사도 테스트
        similarity = model.similarity(embeddings[0:1], embeddings[1:])
        print(f"유사도 테스트 결과: {similarity}")
        
        print("✅ 모든 테스트 통과")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        print(f"상세 에러:\n{traceback.format_exc()}")

def main():
    """메인 진단 함수"""
    models_to_test = [
        "jinaai/jina-embeddings-v3",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ]
    
    results = {}
    
    for model_name in models_to_test:
        try:
            result = diagnose_embedding_model(model_name)
            results[model_name] = "성공" if result else "실패"
        except Exception as e:
            results[model_name] = f"오류: {e}"
    
    # 최종 결과 요약
    print(f"\n{'='*60}")
    print("진단 결과 요약")
    print(f"{'='*60}")
    
    for model_name, status in results.items():
        print(f"{model_name}: {status}")
    
    # 권장사항
    print(f"\n{'='*60}")
    print("권장사항")
    print(f"{'='*60}")
    
    if results.get("jinaai/jina-embeddings-v3") == "성공":
        print("✅ jinaai/jina-embeddings-v3 모델을 사용하는 것을 권장합니다.")
    else:
        print("❌ jinaai/jina-embeddings-v3 모델 사용 불가")
        print("✅ 폴백 모델 사용을 권장합니다:")
        print("   - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        print("   - sentence-transformers/all-MiniLM-L6-v2")

if __name__ == "__main__":
    main()