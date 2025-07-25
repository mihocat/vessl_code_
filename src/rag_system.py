#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG System Core Module
RAG 시스템 핵심 모듈
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConcreteKoreanElectricalRAG:
    """통합 RAG 시스템"""
    
    def __init__(self, embedding_model_name: str = "jhgan/ko-sroberta-multitask"):
        """
        Args:
            embedding_model_name: 한국어 임베딩 모델 이름
        """
        try:
            os.environ["SAFETENSORS_FAST_GPU"] = "1"
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            
            # 한국어 임베딩 모델 로드
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info(f"한국어 임베딩 모델 로드 완료: {embedding_model_name}")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {str(e)}")
            # 폴백 모델
            self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            logger.info("폴백 임베딩 모델 로드: paraphrase-multilingual-MiniLM")
        
        # ChromaDB 초기화
        self.chroma_client = chromadb.PersistentClient(path="/tmp/chroma_db_korean_electrical")
        self.collection = self.chroma_client.get_or_create_collection(name="electrical_engineering_docs")
        
        # 문서 및 통계
        self.documents = []
        self.user_history = defaultdict(list)
        self.service_stats = {
            "total_queries": 0,
            "successful_answers": 0,
            "db_hits": 0,
            "web_searches": 0,
            "user_satisfaction": []
        }
        
        logger.info("ConcreteKoreanElectricalRAG 초기화 완료")
    
    def load_documents_from_dataset(self, dataset_path: str = "/dataset", max_docs: int = 6000):
        """데이터셋에서 문서 로드 및 벡터화"""
        docs_count = 0
        categories = defaultdict(int)
        
        if not os.path.exists(dataset_path):
            logger.warning(f"데이터셋 경로가 존재하지 않음: {dataset_path}")
            self._load_sample_data()
            return
        
        # JSONL 파일에서 문서 로드
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".jsonl") and docs_count < max_docs:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if docs_count >= max_docs:
                                    break
                                try:
                                    data = json.loads(line)
                                    if "Context" in data and "Response" in data:
                                        context_part = data["Context"]
                                        response_part = data["Response"]
                                        
                                        # 카테고리 자동 분류
                                        category = self._categorize_document(context_part)
                                        categories[category] += 1
                                        
                                        content = f"질문: {context_part} 답변: {response_part} 분류: {category}"
                                        doc_item = {
                                            "id": str(docs_count),
                                            "text": content,
                                            "question": context_part,
                                            "answer": response_part,
                                            "category": category
                                        }
                                        self.documents.append(doc_item)
                                        docs_count += 1
                                except:
                                    continue
                    except:
                        continue
        
        # 카테고리별 통계 로그
        logger.info("문서 카테고리 분포:")
        for cat, count in categories.items():
            logger.info(f"- {cat}: {count}개")
        
        # 벡터화 및 저장
        self._vectorize_documents()
    
    def _categorize_document(self, text: str) -> str:
        """문서 카테고리 자동 분류"""
        if any(word in text for word in ["옴의", "키르히호프", "전자기", "맥스웰"]):
            return "기본이론"
        elif any(word in text for word in ["변압기", "모터", "발전기", "전동기"]):
            return "전기기기"
        elif any(word in text for word in ["송전", "배전", "전력계통", "안정도"]):
            return "전력공학"
        elif any(word in text for word in ["시험", "자격증", "기사", "산업기사"]):
            return "자격증"
        else:
            return "일반"
    
    def _load_sample_data(self):
        """샘플 데이터 로드"""
        sample_docs = [
            {
                "question": "옴의 법칙이 무엇인가요?",
                "answer": "옴의 법칙은 전압(V) = 전류(I) × 저항(R)의 관계를 나타내는 전기공학의 기본 법칙입니다.",
                "category": "기본이론"
            },
            {
                "question": "교류와 직류의 차이점을 설명해주세요.",
                "answer": "직류(DC)는 전류가 한 방향으로만 흐르며, 교류(AC)는 전류의 방향이 주기적으로 바뀝니다.",
                "category": "기본이론"
            },
            {
                "question": "변압기의 동작 원리를 알려주세요.",
                "answer": "변압기는 패러데이의 전자기유도 법칙을 이용하여 권수비에 따라 전압을 변환합니다.",
                "category": "전기기기"
            },
            {
                "question": "전기기사 시험은 어떻게 준비하나요?",
                "answer": "전기기사 시험은 필기와 실기로 구성되며, 기본서 학습 후 기출문제를 반복 풀이하는 것이 효과적입니다.",
                "category": "자격증"
            }
        ]
        
        self.documents = []
        for i, doc_data in enumerate(sample_docs):
            content = f"질문: {doc_data['question']} 답변: {doc_data['answer']} 분류: {doc_data['category']}"
            self.documents.append({
                "id": str(i),
                "text": content,
                "question": doc_data["question"],
                "answer": doc_data["answer"],
                "category": doc_data["category"]
            })
        
        self._vectorize_documents()
    
    def _vectorize_documents(self):
        """문서 벡터화 및 ChromaDB 저장"""
        if not self.documents:
            logger.warning("벡터화할 문서가 없습니다.")
            return
        
        texts = [doc["text"] for doc in self.documents]
        logger.info(f"한국어 벡터 임베딩 시작: {len(texts)}개 전기공학 문서")
        
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_ids = [self.documents[i+j]["id"] for j in range(len(batch_texts))]
            
            embeddings = self.embedding_model.encode(batch_texts, convert_to_tensor=True)
            embeddings_np = embeddings.cpu().numpy()
            
            self.collection.add(
                embeddings=embeddings_np.tolist(),
                documents=batch_texts,
                ids=batch_ids
            )
            
            logger.info(f"배치 {i//batch_size + 1} 완료: {len(batch_texts)}개")
        
        logger.info(f"전기공학 지식베이스 구축 완료: {len(texts)}개 문서")
    
    def search_vector_database(self, query: str, k: int = 5) -> tuple:
        """고도화된 벡터 검색 시스템"""
        try:
            # 쿼리 전처리 및 확장
            enhanced_queries = self._expand_query(query)
            query_category = self._advanced_categorize_document(query)
            
            best_results = []
            
            # 다중 쿼리 전략
            for enhanced_query in enhanced_queries:
                query_embedding = self.embedding_model.encode([enhanced_query], convert_to_tensor=True)
                query_embedding_np = query_embedding.cpu().numpy()
                
                results = self.collection.query(
                    query_embeddings=query_embedding_np.tolist(),
                    n_results=k * 3,  # 더 많이 가져와서 고품질 필터링
                    include=['distances', 'documents', 'metadatas']
                )
                
                if results["documents"] and len(results["documents"]) > 0:
                    documents = results["documents"][0]
                    distances = results["distances"][0]
                    ids = results["ids"][0]
                    
                    # 고도화된 점수 계산
                    for doc, distance, doc_id in zip(documents, distances, ids):
                        doc_info = self._get_document_info(doc_id)
                        if not doc_info:
                            continue
                            
                        # 다중 지표 기반 점수
                        scores = self._calculate_multi_score(query, enhanced_query, doc, doc_info, distance, query_category)
                        
                        # 고품질 결과만 선별
                        if scores['final_score'] > 0.65:  # 더 엄격한 임계값
                            best_results.append({
                                "content": doc,
                                "doc_info": doc_info,
                                "scores": scores,
                                "similarity": scores['cosine_similarity'],
                                "final_score": scores['final_score'],
                                "query_type": enhanced_query
                            })
            
            # 중복 제거 및 정렬
            unique_results = self._deduplicate_results(best_results)
            unique_results.sort(key=lambda x: x["final_score"], reverse=True)
            
            if unique_results:
                self.service_stats["db_hits"] += 1
                return unique_results[:k], True
            
            return [], False
        except Exception as e:
            logger.error(f"벡터 검색 실패: {str(e)}")
            return [], False
    
    def _expand_query(self, query: str) -> List[str]:
        """쿼리 확장 및 다중 버전 생성"""
        queries = [query]  # 원본 쿼리
        
        # 전기공학 동의어 확장
        synonyms = {
            '전압': ['볼트', 'V', '전위차'],
            '전류': ['암페어', 'A', '전류값'],
            '저항': ['오예', 'Ω', 'R'],
            '변압기': ['트랜스포머', '전력변압기'],
            '모터': ['전동기', '유도전동기'],
            '발전기': ['동기', '제너레이터']
        }
        
        for word, alternatives in synonyms.items():
            if word in query:
                for alt in alternatives:
                    queries.append(query.replace(word, alt))
        
        # 기술적 맥락 추가
        if any(term in query for term in ['계산', '구하기', '방법']):
            queries.append(f"{query} 공식 단계")
            queries.append(f"{query} 해결 방법")
        
        return queries[:3]  # 최대 3개로 제한
    
    def _advanced_categorize_document(self, text: str) -> str:
        """고도화된 문서 분류"""
        category_keywords = {
            '기본이론': ['옴의법칙', '키르히호프', '전자기', '맥스웰', '쿨롱의법칙', '렉스의법칙'],
            '전기기기': ['변압기', '모터', '발전기', '전동기', '동기기', '유도전동기', '동기모터'],
            '전력공학': ['송전', '배전', '전력계통', '안정도', '보호계전', '전력품질'],
            '자격증': ['시험', '자격증', '기사', '산업기사', '기능사', '전기기사']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            if score > 0:
                scores[category] = score
        
        return max(scores.items(), key=lambda x: x[1])[0] if scores else '일반'
    
    def _get_document_info(self, doc_id: str) -> Optional[Dict]:
        """문서 정보 고속 검색"""
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        return None
    
    def _calculate_multi_score(self, original_query: str, enhanced_query: str, doc: str, doc_info: Dict, distance: float, query_category: str) -> Dict:
        """다중 지표 기반 점수 계산"""
        cosine_similarity = 1 - distance
        
        # 1. 의미적 유사도
        semantic_score = cosine_similarity
        
        # 2. 카테고리 일치도
        category_score = 0.15 if doc_info["category"] == query_category else 0
        
        # 3. 키워드 매칭 (정밀 검색)
        original_words = set(original_query.split())
        doc_words = set(doc.lower().split())
        keyword_overlap = len(original_words.intersection(doc_words)) / max(len(original_words), 1)
        keyword_score = keyword_overlap * 0.2
        
        # 4. 문서 품질 점수
        quality_score = 0.1 if len(doc_info.get("answer", "")) > 50 else 0
        
        # 5. 길이 기반 정규화
        length_penalty = 0.05 if len(doc) > 1000 else 0  # 너무 긴 문서 페널티
        
        final_score = semantic_score + category_score + keyword_score + quality_score - length_penalty
        
        return {
            'cosine_similarity': cosine_similarity,
            'semantic_score': semantic_score,
            'category_score': category_score,
            'keyword_score': keyword_score,
            'quality_score': quality_score,
            'final_score': min(final_score, 1.0)  # 1.0 최대값 제한
        }
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """중복 결과 제거"""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            doc_id = result["doc_info"]["id"]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(result)
        
        return unique_results
    
    def search_web(self, query: str, max_results: int = 3) -> List[Dict]:
        """지능형 웹 검색 시스템"""
        try:
            with DDGS() as ddgs:
                # 다양한 검색 전략
                search_queries = [
                    f"전기공학 {query}",
                    f"{query} 해설",
                    f"{query} 기초 이론",
                    query  # 원본 쿼리
                ]
                
                all_results = []
                
                for search_query in search_queries[:2]:  # 상위 2개 전략만 사용
                    try:
                        web_results = list(ddgs.text(search_query, region="ko-kr", max_results=max_results))
                
                        for result in web_results:
                            all_results.append(result)
                    except:
                        continue
                
                # 결과 처리 및 필터링
                processed_results = []
                for result in web_results:
                    trust_score = 1.5 if any(domain in result.get("href", "") for domain in [".edu", ".ac.kr", ".go.kr", "kea.kr"]) else 1.0
                    
                    processed_results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("body", "")[:200],
                        "url": result.get("href", ""),
                        "trust_score": trust_score
                    })
                
                # 신뢰도순 정렬
                processed_results.sort(key=lambda x: x["trust_score"], reverse=True)
                self.service_stats["web_searches"] += 1
                return processed_results
        except Exception as e:
            logger.error(f"웹 검색 실패: {str(e)}")
            return []
    
    def check_electrical_relevance(self, query: str) -> bool:
        """전기공학 관련성 확인"""
        electrical_keywords = [
            "전기", "전력", "전압", "전류", "저항", "회로", "변압기", "모터", "발전",
            "배전", "송전", "전자", "에너지", "와트", "암페어", "볼트", "옴",
            "AC", "DC", "교류", "직류", "주파수", "임피던스", "인덕턴스",
            "커패시턴스", "기사", "자격증", "시험", "공부"
        ]
        
        query_lower = query.lower()
        return any(keyword.lower() in query_lower for keyword in electrical_keywords)
    
    def get_service_statistics(self) -> str:
        """서비스 통계 제공"""
        stats = []
        stats.append("📊 **전기공학 통합 서비스 통계**\n")
        stats.append(f"• 총 질의: {self.service_stats['total_queries']}건")
        stats.append(f"• 성공 답변: {self.service_stats['successful_answers']}건")
        stats.append(f"• DB 적중: {self.service_stats['db_hits']}건")
        stats.append(f"• 웹 검색: {self.service_stats['web_searches']}건")
        
        if self.service_stats["total_queries"] > 0:
            success_rate = self.service_stats["successful_answers"] / self.service_stats["total_queries"] * 100
            db_hit_rate = self.service_stats["db_hits"] / self.service_stats["total_queries"] * 100
            stats.append(f"\n• 응답 성공률: {round(success_rate, 1)}%")
            stats.append(f"• DB 활용률: {round(db_hit_rate, 1)}%")
        
        stats.append(f"\n• 지식베이스: {len(self.documents)}개 문서")
        stats.append(f"• 활성 사용자: {len(self.user_history)}명")
        
        return "\n".join(stats)