#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved RAG System
개선된 RAG (Retrieval-Augmented Generation) 시스템
"""

import os
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import RAGConfig, DatasetConfig
from document_loader import DatasetLoader
from llm_client import LLMClient

logger = logging.getLogger(__name__)

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SENTENCE_TRANSFORMERS_TRUST_REMOTE_CODE"] = "true"


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    question: str
    answer: str
    score: float
    category: str = "general"
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ChromaEmbeddingFunction:
    """ChromaDB용 임베딩 함수"""
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        """임베딩 생성"""
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()


class VectorStore:
    """벡터 저장소 관리 클래스"""
    
    def __init__(self, config: RAGConfig, embedding_model: SentenceTransformer):
        self.config = config
        self.embedding_function = ChromaEmbeddingFunction(embedding_model)
        self.client = None
        self.collection = None
        self._initialize()
        
    def _initialize(self):
        """ChromaDB 초기화"""
        try:
            # ChromaDB 클라이언트 생성
            self.client = chromadb.PersistentClient(
                path=self.config.chroma_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 기존 컬렉션 삭제
            try:
                self.client.delete_collection(name=self.config.collection_name)
                logger.info(f"Deleted existing collection: {self.config.collection_name}")
            except Exception:
                pass
            
            # 새 컬렉션 생성
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Created new collection: {self.config.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """문서 벡터화 및 저장"""
        if not documents:
            logger.warning("No documents to vectorize")
            return
            
        total_docs = len(documents)
        logger.info(f"Starting vectorization of {total_docs} documents...")
        
        # 배치 처리
        for i in range(0, total_docs, self.config.batch_size):
            batch = documents[i:i + self.config.batch_size]
            
            # 데이터 준비
            texts = []
            metadatas = []
            ids = []
            
            for idx, doc in enumerate(batch):
                doc_id = i + idx
                texts.append(doc["question"])
                metadatas.append({
                    "question": doc["question"],
                    "answer": doc["answer"],
                    "category": self._categorize(doc["question"]),
                })
                ids.append(str(doc_id))
            
            # ChromaDB에 추가
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            current_batch = (i // self.config.batch_size) + 1
            total_batches = (total_docs + self.config.batch_size - 1) // self.config.batch_size
            logger.info(f"Vectorization progress: {current_batch}/{total_batches} batches")
        
        # 저장된 문서 수 확인
        count = self.collection.count()
        logger.info(f"Total documents in ChromaDB: {count}")
    
    def search(self, query: str, k: int) -> List[SearchResult]:
        """벡터 검색"""
        try:
            count = self.collection.count()
            if count == 0:
                logger.warning("No documents in ChromaDB")
                return []
            
            # 검색 실행
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, count),
                include=["metadatas", "distances", "documents"]
            )
            
            if not results["ids"] or not results["ids"][0]:
                return []
            
            # 결과 변환
            search_results = []
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                
                # 코사인 유사도 계산
                # ChromaDB의 cosine distance 범위 확인을 위한 상세 로깅
                logger.debug(f"Raw distance from ChromaDB: {distance}")
                
                # ChromaDB는 normalized 벡터에 대해 0~2 범위의 cosine distance 반환
                # 하지만 실제로는 다른 범위일 수 있음
                # 안전한 변환을 위해 거리값의 범위 확인
                if distance <= 0:
                    similarity = 1.0  # 완전히 동일
                elif distance >= 2:
                    similarity = 0.0  # 완전히 반대
                else:
                    # 선형 변환: distance 0->1, 2->0
                    similarity = 1.0 - (distance / 2.0)
                
                # 추가 보정: 실제 ChromaDB 동작에 따라 조정
                # 매우 작은 거리는 높은 유사도로, 하지만 완벽한 1.0은 피함
                if distance < 0.1:
                    similarity = min(0.95 + (0.05 * (1.0 - distance/0.1)), 0.99)
                elif distance > 1.5:
                    similarity = max(0.0, similarity * 0.5)  # 큰 거리는 더 낮은 점수
                
                # 결과 로깅
                if i < 3:  # 처음 3개 결과 상세 로그
                    logger.info(f"Result {i}: distance={distance:.6f}, similarity={similarity:.4f}, Q: {metadata['question'][:50]}...")
                
                search_results.append(SearchResult(
                    question=metadata["question"],
                    answer=metadata["answer"],
                    score=similarity,
                    category=metadata.get("category", "general"),
                    metadata=metadata
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _categorize(self, text: str) -> str:
        """간단한 카테고리 분류"""
        # 여기서는 간단한 분류만 수행
        # 필요시 더 복잡한 분류 로직 추가 가능
        return "general"


class SearchRanker:
    """검색 결과 재정렬 클래스"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
    def rerank(self, query: str, results: List[SearchResult], k: int) -> List[SearchResult]:
        """검색 결과 재정렬 및 점수 조정"""
        if not results:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # 점수 재계산
        for result in results:
            # 기본 점수
            base_score = result.score
            
            # 키워드 매칭 보너스
            question_lower = result.question.lower()
            question_words = set(question_lower.split())
            
            # 정확한 단어 매칭
            exact_matches = len(query_words.intersection(question_words))
            keyword_bonus = exact_matches * self.config.keyword_match_bonus
            
            # 부분 문자열 매칭
            substring_bonus = 0
            for word in query_words:
                if len(word) > 2 and word in question_lower:
                    substring_bonus += self.config.substring_match_bonus
            
            # 최종 점수 계산
            final_score = base_score + keyword_bonus + substring_bonus
            final_score = min(final_score, 1.0)
            
            result.score = final_score
        
        # 점수 기준 정렬
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 점수 분포 로깅
        if results:
            scores = [r.score for r in results[:10]]  # 상위 10개
            logger.info(f"Score distribution (top 10): {scores}")
            logger.info(f"Score range: {min(scores):.4f} ~ {max(scores):.4f}")
        
        # 임계값 필터링
        filtered_results = [
            r for r in results 
            if r.score >= self.config.low_confidence_threshold
        ]
        
        return filtered_results[:k]


class RAGSystem:
    """통합 RAG 시스템"""
    
    def __init__(
        self,
        rag_config: Optional[RAGConfig] = None,
        dataset_config: Optional[DatasetConfig] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """
        RAG 시스템 초기화
        
        Args:
            rag_config: RAG 설정
            dataset_config: 데이터셋 설정
            llm_client: LLM 클라이언트
        """
        self.rag_config = rag_config or RAGConfig()
        self.dataset_config = dataset_config or DatasetConfig()
        self.llm_client = llm_client
        
        # 통계
        self.stats = {
            "total_queries": 0,
            "high_confidence_hits": 0,
            "medium_confidence_hits": 0,
            "low_confidence_hits": 0,
            "avg_response_time": 0.0
        }
        
        # 컴포넌트 초기화
        self._initialize_components()
        
        # 문서 로드
        self._load_documents()
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        # 임베딩 모델
        logger.info(f"Loading embedding model: {self.rag_config.embedding_model_name}")
        self.embedding_model = SentenceTransformer(
            self.rag_config.embedding_model_name,
            trust_remote_code=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 임베딩 모델 검증
        try:
            test_embedding = self.embedding_model.encode("테스트 문장")
            logger.info(f"Embedding model loaded successfully. Embedding dim: {len(test_embedding)}")
        except Exception as e:
            logger.error(f"Embedding model test failed: {e}")
        
        # 벡터 저장소
        self.vector_store = VectorStore(self.rag_config, self.embedding_model)
        
        # 검색 랭커
        self.ranker = SearchRanker(self.rag_config)
    
    def _load_documents(self):
        """문서 로드 및 인덱싱"""
        loader = DatasetLoader(
            paths=self.dataset_config.paths,
            file_extensions=self.dataset_config.file_extensions,
            max_documents=self.rag_config.max_documents
        )
        
        documents = loader.load_documents()
        
        if documents:
            self.vector_store.add_documents(documents)
        else:
            logger.warning("No documents loaded")
    
    def search(self, query: str, k: Optional[int] = None) -> Tuple[List[SearchResult], float]:
        """
        향상된 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            (검색 결과 리스트, 최고 점수)
        """
        start_time = time.time()
        k = k or self.rag_config.rerank_k
        
        # 벡터 검색
        results = self.vector_store.search(query, self.rag_config.search_k)
        
        # 재정렬
        final_results = self.ranker.rerank(query, results, k)
        
        # 최고 점수 계산
        max_score = max([r.score for r in final_results]) if final_results else 0.0
        
        # 통계 업데이트
        self._update_stats(max_score, time.time() - start_time)
        
        logger.info(
            f"Search completed: {len(final_results)} results, "
            f"max score: {max_score:.3f}, time: {time.time() - start_time:.2f}s"
        )
        
        return final_results, max_score
    
    def _update_stats(self, max_score: float, response_time: float):
        """통계 업데이트"""
        self.stats["total_queries"] += 1
        
        if max_score >= self.rag_config.high_confidence_threshold:
            self.stats["high_confidence_hits"] += 1
        elif max_score >= self.rag_config.medium_confidence_threshold:
            self.stats["medium_confidence_hits"] += 1
        else:
            self.stats["low_confidence_hits"] += 1
        
        # 평균 응답 시간 계산
        n = self.stats["total_queries"]
        prev_avg = self.stats["avg_response_time"]
        self.stats["avg_response_time"] = (prev_avg * (n - 1) + response_time) / n
    
    def get_stats(self) -> Dict:
        """통계 반환"""
        return self.stats.copy()


# 하위 호환성을 위한 별칭
ImprovedRAGSystem = RAGSystem