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
        """지능형 벡터 검색"""
        try:
            # 쿼리 카테고리 추정
            query_category = self._categorize_document(query)
            
            # 벡터 검색
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
            query_embedding_np = query_embedding.cpu().numpy()
            
            results = self.collection.query(
                query_embeddings=query_embedding_np.tolist(),
                n_results=k * 2  # 더 많이 가져와서 필터링
            )
            
            if results["documents"] and len(results["documents"]) > 0:
                documents = results["documents"][0]
                distances = results["distances"][0] if "distances" in results else [1.0] * len(documents)
                ids = results["ids"][0] if "ids" in results else []
                
                # 관련성 점수 계산 및 필터링
                relevant_docs = []
                for doc, distance, doc_id in zip(documents, distances, ids):
                    similarity = 1 - distance
                    
                    # 문서 정보 가져오기
                    doc_info = None
                    for d in self.documents:
                        if d["id"] == doc_id:
                            doc_info = d
                            break
                    
                    # 카테고리 보너스
                    category_bonus = 0.1 if doc_info and doc_info["category"] == query_category else 0
                    
                    # 키워드 매칭 보너스
                    keyword_bonus = sum(0.05 for word in query.split() if len(word) > 1 and word in doc)
                    
                    # 최종 점수
                    final_score = similarity + category_bonus + keyword_bonus
                    
                    # 임계값 이상만 추가
                    if similarity > 0.5 or final_score > 0.6:
                        relevant_docs.append({
                            "content": doc,
                            "similarity": similarity,
                            "final_score": final_score,
                            "doc_info": doc_info
                        })
                
                # 점수순 정렬
                relevant_docs.sort(key=lambda x: x["final_score"], reverse=True)
                
                if relevant_docs:
                    self.service_stats["db_hits"] += 1
                    return relevant_docs[:k], True
            
            return [], False
        except Exception as e:
            logger.error(f"지능형 검색 실패: {str(e)}")
            return [], False
    
    def search_web(self, query: str, max_results: int = 3) -> List[Dict]:
        """웹 검색 보조"""
        try:
            with DDGS() as ddgs:
                enhanced_query = f"다산에듀 {query}"
                web_results = list(ddgs.text(enhanced_query, region="ko-kr", max_results=max_results))
                
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