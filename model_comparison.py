#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
임베딩 모델 비교 도구
"""

from sentence_transformers import SentenceTransformer
import numpy as np

def compare_models():
    """임베딩 모델 성능 비교"""
    
    # 테스트할 한국어 전기공학 텍스트
    korean_electrical_texts = [
        "옴의 법칙은 전압과 전류의 관계를 나타낸다",
        "변압기는 전자기유도 원리로 전압을 변환한다", 
        "전기기사 시험은 필기와 실기로 구성된다",
        "키르히호프 법칙은 회로 해석의 기본이다",
        "삼상 전력 시스템은 효율적인 전력 전송 방식이다"
    ]
    
    # 테스트할 모델 (빠르게 로드되는 것들만)
    models = [
        ("paraphrase-multilingual-MiniLM-L12-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        ("distiluse-base-multilingual-cased", "sentence-transformers/distiluse-base-multilingual-cased"),
        ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2")
    ]
    
    results = {}
    
    for model_name, model_path in models:
        try:
            print(f"\n테스트 중: {model_name}")
            model = SentenceTransformer(model_path)
            
            # 기본 정보
            embedding_dim = model.get_sentence_embedding_dimension()
            max_seq_len = getattr(model, 'max_seq_length', 'Unknown')
            
            print(f"  임베딩 차원: {embedding_dim}")
            print(f"  최대 시퀀스: {max_seq_len}")
            
            # 한국어 임베딩 생성
            embeddings = model.encode(korean_electrical_texts)
            
            # 유사도 행렬 계산
            similarity_matrix = model.similarity(embeddings, embeddings)
            
            # 관련성 점수 (대각선 제외한 평균)
            mask = np.ones_like(similarity_matrix, dtype=bool)
            np.fill_diagonal(mask, 0)
            avg_similarity = similarity_matrix[mask].mean().item()
            
            results[model_name] = {
                'embedding_dim': embedding_dim,
                'max_seq_len': max_seq_len,
                'avg_similarity': avg_similarity,
                'status': 'success'
            }
            
            print(f"  ✅ 성공 - 평균 유사도: {avg_similarity:.3f}")
            
        except Exception as e:
            print(f"  ❌ 실패: {e}")
            results[model_name] = {'status': 'failed', 'error': str(e)}
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("모델 비교 결과")
    print(f"{'='*60}")
    
    for model_name, result in results.items():
        if result['status'] == 'success':
            print(f"{model_name}:")
            print(f"  차원: {result['embedding_dim']}")
            print(f"  시퀀스: {result['max_seq_len']}")
            print(f"  한국어 유사도: {result['avg_similarity']:.3f}")
        else:
            print(f"{model_name}: 실패 ({result['error']})")
        print()
    
    # 권장사항
    successful_models = [(name, res) for name, res in results.items() if res['status'] == 'success']
    if successful_models:
        # 차원수가 높고 유사도가 적절한 모델 우선
        best_model = max(successful_models, key=lambda x: x[1]['embedding_dim'])
        print(f"권장 모델: {best_model[0]}")
        print(f"  이유: 임베딩 차원 {best_model[1]['embedding_dim']}로 풍부한 표현력 제공")

if __name__ == "__main__":
    compare_models()