#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Reranker System with Cross-Encoder
고급 리랭커 시스템 - Cross-Encoder 기반
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-Encoder 기반 리랭커"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        리랭커 초기화
        
        Args:
            model_name: Cross-encoder 모델 이름
        """
        try:
            self.model = CrossEncoder(model_name, max_length=512)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Cross-encoder reranker loaded: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            self.model = None
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        문서 리랭킹
        
        Args:
            query: 검색 쿼리
            documents: 검색된 문서 리스트
            top_k: 상위 k개 반환
        """
        if not self.model or not documents:
            return documents
        
        start_time = time.time()
        
        # 쿼리-문서 쌍 생성
        pairs = []
        for doc in documents:
            doc_text = self._extract_text(doc)
            pairs.append([query, doc_text])
        
        # Cross-encoder 점수 계산
        scores = self.model.predict(pairs, convert_to_numpy=True)
        
        # 점수 기준 정렬
        ranked_indices = np.argsort(scores)[::-1]
        
        # 상위 k개 선택
        reranked_docs = []
        for i in ranked_indices[:top_k]:
            doc = documents[i].copy()
            doc['rerank_score'] = float(scores[i])
            reranked_docs.append(doc)
        
        elapsed = time.time() - start_time
        logger.info(f"Reranked {len(documents)} docs to top {len(reranked_docs)} in {elapsed:.3f}s")
        
        return reranked_docs
    
    def _extract_text(self, doc: Dict[str, Any]) -> str:
        """문서에서 텍스트 추출"""
        if 'documents' in doc and doc['documents']:
            return doc['documents'][0][:500]  # 최대 500자
        elif 'content' in doc:
            return doc['content'][:500]
        elif 'text' in doc:
            return doc['text'][:500]
        else:
            return str(doc)[:500]


class ElectricalEngineeringReranker(CrossEncoderReranker):
    """전기공학 특화 리랭커"""
    
    def __init__(self):
        """전기공학 특화 리랭커 초기화"""
        super().__init__()
        
        # 도메인 특화 가중치
        self.domain_weights = {
            'formula_match': 0.3,      # 수식 일치도
            'concept_match': 0.2,      # 개념 일치도
            'problem_type': 0.2,       # 문제 유형 일치도
            'cross_encoder': 0.3       # Cross-encoder 점수
        }
        
        # 전기공학 핵심 개념
        self.ee_concepts = {
            '전압': ['voltage', 'V', '볼트', '전위차'],
            '전류': ['current', 'I', '암페어', 'A'],
            '저항': ['resistance', 'R', '옴', 'Ω', 'ohm'],
            '전력': ['power', 'P', '와트', 'W', 'kW'],
            '역률': ['power factor', 'PF', 'cosθ', '무효전력'],
            '임피던스': ['impedance', 'Z', '리액턴스'],
            '변압기': ['transformer', '권선비', '철손', '동손'],
            '모터': ['motor', '전동기', '회전자', '고정자'],
            '회로': ['circuit', '직렬', '병렬', 'RLC']
        }
    
    def rerank_with_domain_knowledge(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        query_analysis: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        도메인 지식을 활용한 리랭킹
        
        Args:
            query: 검색 쿼리
            documents: 검색된 문서
            query_analysis: 쿼리 분석 결과 (수식, 개념 등)
            top_k: 상위 k개
        """
        if not documents:
            return documents
        
        # 1. Cross-encoder 점수 계산
        ce_scores = self._get_cross_encoder_scores(query, documents)
        
        # 2. 도메인 특화 점수 계산
        domain_scores = []
        for i, doc in enumerate(documents):
            scores = {
                'cross_encoder': ce_scores[i] if ce_scores else 0.5,
                'formula_match': self._calculate_formula_match(query, doc, query_analysis),
                'concept_match': self._calculate_concept_match(query, doc),
                'problem_type': self._calculate_problem_type_match(query, doc)
            }
            
            # 가중 평균
            final_score = sum(
                scores[key] * self.domain_weights[key] 
                for key in scores
            )
            
            domain_scores.append({
                'index': i,
                'final_score': final_score,
                'detail_scores': scores
            })
        
        # 3. 점수 기준 정렬
        domain_scores.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 4. 상위 k개 선택
        reranked_docs = []
        for score_info in domain_scores[:top_k]:
            doc = documents[score_info['index']].copy()
            doc['rerank_score'] = score_info['final_score']
            doc['score_details'] = score_info['detail_scores']
            reranked_docs.append(doc)
        
        return reranked_docs
    
    def _get_cross_encoder_scores(self, query: str, documents: List[Dict]) -> List[float]:
        """Cross-encoder 점수 계산"""
        if not self.model:
            return []
        
        pairs = [[query, self._extract_text(doc)] for doc in documents]
        scores = self.model.predict(pairs, convert_to_numpy=True)
        
        # 0-1 범위로 정규화
        if len(scores) > 0:
            min_score = np.min(scores)
            max_score = np.max(scores)
            if max_score > min_score:
                scores = (scores - min_score) / (max_score - min_score)
            else:
                scores = np.ones_like(scores) * 0.5
        
        return scores.tolist()
    
    def _calculate_formula_match(
        self, 
        query: str, 
        doc: Dict[str, Any],
        query_analysis: Optional[Dict[str, Any]]
    ) -> float:
        """수식 일치도 계산"""
        score = 0.0
        
        # 쿼리에서 수식 추출
        query_formulas = []
        if query_analysis and 'formulas' in query_analysis:
            query_formulas = query_analysis['formulas']
        
        # 문서에서 수식 추출
        doc_formulas = []
        if 'parsed_content' in doc and 'formulas' in doc['parsed_content']:
            doc_formulas = doc['parsed_content']['formulas']
        elif 'formulas' in doc:
            doc_formulas = doc['formulas']
        
        # 수식 유사도 계산
        if query_formulas and doc_formulas:
            matches = 0
            for qf in query_formulas:
                for df in doc_formulas:
                    if self._formula_similarity(qf, df) > 0.8:
                        matches += 1
                        break
            
            score = matches / len(query_formulas) if query_formulas else 0
        
        return score
    
    def _calculate_concept_match(self, query: str, doc: Dict[str, Any]) -> float:
        """개념 일치도 계산"""
        query_lower = query.lower()
        doc_text = self._extract_text(doc).lower()
        
        matched_concepts = 0
        total_concepts = 0
        
        for concept, variations in self.ee_concepts.items():
            # 쿼리에 개념이 있는지 확인
            query_has_concept = any(var.lower() in query_lower for var in [concept] + variations)
            
            if query_has_concept:
                total_concepts += 1
                # 문서에도 있는지 확인
                doc_has_concept = any(var.lower() in doc_text for var in [concept] + variations)
                if doc_has_concept:
                    matched_concepts += 1
        
        return matched_concepts / total_concepts if total_concepts > 0 else 0.5
    
    def _calculate_problem_type_match(self, query: str, doc: Dict[str, Any]) -> float:
        """문제 유형 일치도 계산"""
        # 문제 유형 키워드
        problem_types = {
            '계산': ['계산', '구하', '산출', '값은'],
            '설명': ['설명', '무엇', '정의', '개념'],
            '차이': ['차이', '비교', '다른점', 'vs'],
            '이유': ['이유', '왜', '원인', '때문'],
            '방법': ['방법', '어떻게', '절차', '과정']
        }
        
        query_lower = query.lower()
        doc_text = self._extract_text(doc).lower()
        
        # 쿼리의 문제 유형 파악
        query_type = None
        for ptype, keywords in problem_types.items():
            if any(kw in query_lower for kw in keywords):
                query_type = ptype
                break
        
        if not query_type:
            return 0.5
        
        # 문서가 해당 유형의 답변을 포함하는지 확인
        type_indicators = {
            '계산': ['계산 과정', '풀이', '답:', '결과:'],
            '설명': ['정의:', '의미:', '개념:', '~입니다'],
            '차이': ['차이점', '반면', '비교하면', '달리'],
            '이유': ['때문', '이유는', '원인은', '~해서'],
            '방법': ['단계', '순서', '먼저', '다음']
        }
        
        if query_type in type_indicators:
            indicators = type_indicators[query_type]
            match_count = sum(1 for ind in indicators if ind in doc_text)
            return min(match_count / len(indicators), 1.0)
        
        return 0.5
    
    def _formula_similarity(self, formula1: str, formula2: str) -> float:
        """수식 유사도 계산 (간단한 버전)"""
        # LaTeX 명령어 제거
        import re
        
        def normalize_formula(f):
            f = re.sub(r'\\[a-zA-Z]+', '', f)
            f = re.sub(r'[{}()[\]]', '', f)
            f = re.sub(r'\s+', '', f)
            return f.lower()
        
        f1_norm = normalize_formula(formula1)
        f2_norm = normalize_formula(formula2)
        
        # 간단한 문자열 유사도
        if f1_norm == f2_norm:
            return 1.0
        
        # 공통 문자 비율
        common_chars = set(f1_norm) & set(f2_norm)
        all_chars = set(f1_norm) | set(f2_norm)
        
        return len(common_chars) / len(all_chars) if all_chars else 0


class HybridReranker:
    """하이브리드 리랭커 (다중 모델 앙상블)"""
    
    def __init__(self):
        """하이브리드 리랭커 초기화"""
        self.rerankers = []
        
        # 1. Cross-encoder 리랭커
        self.ce_reranker = CrossEncoderReranker()
        
        # 2. 전기공학 특화 리랭커
        self.ee_reranker = ElectricalEngineeringReranker()
        
        # 3. 추가 리랭커 (확장 가능)
        # self.bert_reranker = BERTReranker()
        
        logger.info("Hybrid reranker initialized")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        query_analysis: Optional[Dict[str, Any]] = None,
        strategy: str = 'weighted_fusion',
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        하이브리드 리랭킹
        
        Args:
            query: 검색 쿼리
            documents: 검색된 문서
            query_analysis: 쿼리 분석 결과
            strategy: 앙상블 전략 ('weighted_fusion', 'rrf', 'cascade')
            top_k: 상위 k개
        """
        if strategy == 'weighted_fusion':
            return self._weighted_fusion_rerank(query, documents, query_analysis, top_k)
        elif strategy == 'rrf':
            return self._reciprocal_rank_fusion(query, documents, query_analysis, top_k)
        elif strategy == 'cascade':
            return self._cascade_rerank(query, documents, query_analysis, top_k)
        else:
            # 기본: 전기공학 특화 리랭커
            return self.ee_reranker.rerank_with_domain_knowledge(
                query, documents, query_analysis, top_k
            )
    
    def _weighted_fusion_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        query_analysis: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """가중 융합 리랭킹"""
        # 각 리랭커 결과
        ce_results = self.ce_reranker.rerank(query, documents, len(documents))
        ee_results = self.ee_reranker.rerank_with_domain_knowledge(
            query, documents, query_analysis, len(documents)
        )
        
        # 점수 융합 (CE: 0.4, EE: 0.6)
        doc_scores = {}
        
        for doc in ce_results:
            doc_id = id(doc)
            doc_scores[doc_id] = {
                'doc': doc,
                'ce_score': doc.get('rerank_score', 0),
                'ee_score': 0
            }
        
        for doc in ee_results:
            doc_id = id(doc)
            if doc_id in doc_scores:
                doc_scores[doc_id]['ee_score'] = doc.get('rerank_score', 0)
        
        # 최종 점수 계산
        final_results = []
        for doc_id, scores in doc_scores.items():
            final_score = 0.4 * scores['ce_score'] + 0.6 * scores['ee_score']
            doc = scores['doc'].copy()
            doc['final_rerank_score'] = final_score
            doc['score_components'] = {
                'cross_encoder': scores['ce_score'],
                'domain_expert': scores['ee_score']
            }
            final_results.append(doc)
        
        # 정렬 및 상위 k개 선택
        final_results.sort(key=lambda x: x['final_rerank_score'], reverse=True)
        return final_results[:top_k]
    
    def _reciprocal_rank_fusion(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        query_analysis: Optional[Dict[str, Any]],
        top_k: int,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion"""
        # 각 리랭커 결과
        rankings = []
        rankings.append(self.ce_reranker.rerank(query, documents, len(documents)))
        rankings.append(self.ee_reranker.rerank_with_domain_knowledge(
            query, documents, query_analysis, len(documents)
        ))
        
        # RRF 점수 계산
        doc_rrf_scores = {}
        
        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                doc_id = id(doc)
                if doc_id not in doc_rrf_scores:
                    doc_rrf_scores[doc_id] = {'doc': doc, 'rrf_score': 0}
                
                doc_rrf_scores[doc_id]['rrf_score'] += 1 / (k + rank + 1)
        
        # 정렬
        sorted_docs = sorted(
            doc_rrf_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )
        
        # 결과 구성
        results = []
        for item in sorted_docs[:top_k]:
            doc = item['doc'].copy()
            doc['rrf_score'] = item['rrf_score']
            results.append(doc)
        
        return results
    
    def _cascade_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        query_analysis: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """캐스케이드 리랭킹 (단계적 필터링)"""
        # 1단계: Cross-encoder로 상위 2*k개 선택
        stage1_results = self.ce_reranker.rerank(query, documents, min(2 * top_k, len(documents)))
        
        # 2단계: 도메인 특화 리랭커로 최종 k개 선택
        final_results = self.ee_reranker.rerank_with_domain_knowledge(
            query, stage1_results, query_analysis, top_k
        )
        
        return final_results


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    # 리랭커 초기화
    reranker = HybridReranker()
    
    # 테스트 쿼리
    query = "3상 전력 계산 공식"
    
    # 테스트 문서
    documents = [
        {
            'documents': ["3상 전력은 P = √3 × V × I × cosθ로 계산됩니다."],
            'metadata': {'score': 0.8}
        },
        {
            'documents': ["전력은 전압과 전류의 곱입니다."],
            'metadata': {'score': 0.7}
        },
        {
            'documents': ["3상 시스템에서는 선간전압을 사용합니다."],
            'metadata': {'score': 0.75}
        }
    ]
    
    # 리랭킹
    results = reranker.rerank(query, documents, top_k=2)
    
    print("Reranked results:")
    for i, doc in enumerate(results):
        print(f"{i+1}. Score: {doc.get('final_rerank_score', 0):.3f}")
        print(f"   Content: {doc['documents'][0][:100]}...")