#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 RAG 시스템 - 정확도 향상을 위한 전면 재설계
"""

import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import asyncio
import torch
import numpy as np
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from ddgs import DDGS
from llm_client import LLMClient

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedRAGSystem:
    def __init__(
        self,
        embedding_model_name: str = "jinaai/jina-embeddings-v3",
        collection_name: str = "electrical_qa_v3",
        llm_client: Optional[LLMClient] = None
    ):
        """개선된 RAG 시스템 초기화"""
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.llm_client = llm_client
        self.documents = []
        
        # 통계
        self.service_stats = {
            "db_hits": 0,
            "web_searches": 0,
            "llm_responses": 0,
            "avg_response_time": 0,
            "total_queries": 0
        }
        
        # 특수 키워드 사전 (비활성화 - 데이터셋에 의존)
        self.special_keywords = {}
        
        # ChromaDB 초기화
        self._init_chromadb()
        
        # 문서 로드
        self._load_all_documents()
    
    def _init_chromadb(self):
        """ChromaDB 초기화 및 컬렉션 생성"""
        try:
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["SENTENCE_TRANSFORMERS_TRUST_REMOTE_CODE"] = "true"
            
            # jina-embeddings-v3를 위한 커스텀 임베딩 함수
            from sentence_transformers import SentenceTransformer
            
            # trust_remote_code 파라미터로 모델 로드
            self.model = SentenceTransformer(
                self.embedding_model_name,
                trust_remote_code=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # ChromaDB용 커스텀 임베딩 함수 클래스
            class JinaEmbeddingFunction:
                def __init__(self, model):
                    self.model = model
                    
                def __call__(self, input):
                    # ChromaDB는 'input' 파라미터를 기대함
                    embeddings = self.model.encode(input, normalize_embeddings=True)
                    return embeddings.tolist()
            
            self.embedding_function = JinaEmbeddingFunction(self.model)
            
            # ChromaDB 클라이언트 초기화
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 기존 컬렉션 삭제 후 재생성
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                logger.info(f"기존 컬렉션 '{self.collection_name}' 삭제")
            except:
                pass
            
            # 새 컬렉션 생성
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB 컬렉션 '{self.collection_name}' 생성 완료")
            
        except Exception as e:
            logger.error(f"ChromaDB 초기화 실패: {e}")
            raise
    
    def _load_all_documents(self):
        """모든 문서 로드 및 벡터화"""
        # 1. 특수 키워드 먼저 로드
        self._load_special_keywords()
        
        # 2. 데이터셋 문서 로드
        self._load_dataset_documents()
        
        # 3. 벡터화
        if self.documents:
            self._vectorize_documents()
    
    def _load_special_keywords(self):
        """특수 키워드를 우선 문서로 추가"""
        # 특수 키워드 비활성화 - 데이터셋에서 로드
        if self.special_keywords:
            doc_id = 0
            for keyword, info in self.special_keywords.items():
                doc_item = {
                    "id": f"special_{doc_id}",
                    "text": keyword,  # 키워드만 벡터화
                    "question": keyword,
                    "answer": info["answer"],
                    "category": "특수키워드",
                    "is_special": True,
                    "confidence": info["confidence"]
                }
                self.documents.append(doc_item)
                doc_id += 1
            
            logger.info(f"특수 키워드 {len(self.special_keywords)}개 로드 완료")
    
    def _load_dataset_documents(self):
        """데이터셋에서 문서 로드"""
        docs_count = len(self.documents)
        
        # 경로 설정
        paths_to_check = [
            Path("/dataset"),
            Path("./dataset"),
            Path("./2_documents"),
            Path("/data"),
            Path("./data")
        ]
        
        # 현재 작업 디렉토리 확인
        logger.info(f"현재 작업 디렉토리: {os.getcwd()}")
        logger.info(f"디렉토리 내용: {os.listdir('.')}")
        
        # 환경변수 확인
        logger.info(f"DATASET_PATH 환경변수: {os.environ.get('DATASET_PATH', 'Not set')}")
        
        docs_path = None
        for path in paths_to_check:
            logger.info(f"경로 확인 중: {path.absolute()}")
            if path.exists():
                docs_path = path
                logger.info(f"문서 경로 발견: {docs_path.absolute()}")
                # 디렉토리 내용 확인
                if docs_path.is_dir():
                    items = list(docs_path.iterdir())
                    logger.info(f"디렉토리 내 항목 수: {len(items)}")
                    if items:
                        logger.info(f"처음 10개 항목: {[item.name for item in items[:10]]}")
                        # 파일 타입 확인
                        file_types = {}
                        for item in items:
                            if item.is_file():
                                ext = item.suffix
                                file_types[ext] = file_types.get(ext, 0) + 1
                        logger.info(f"파일 타입별 개수: {file_types}")
                break
            else:
                logger.info(f"경로 {path.absolute()} 존재하지 않음")
        
        if not docs_path:
            logger.warning("문서 경로를 찾을 수 없습니다.")
            # 루트 디렉토리 탐색
            logger.info("루트 디렉토리 탐색:")
            for item in Path("/").iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    logger.info(f"  /{item.name}")
            self._load_sample_data()
            return
        
        # 파일 처리 - 다양한 확장자 지원
        all_files = []
        for ext in ['*.txt', '*.json', '*.jsonl', '*.csv', '*.tsv']:
            files = list(docs_path.rglob(ext))
            if files:
                all_files.extend(files)
                logger.info(f"{ext} 파일 {len(files)}개 발견")
        
        logger.info(f"총 발견된 파일 수: {len(all_files)}")
        
        # 파일 이름 샘플 출력
        if all_files:
            logger.info(f"첫 5개 파일: {[f.name for f in all_files[:5]]}")
        
        processed_files = 0
        for file_path in all_files:
            if docs_count >= 10000:  # 최대 문서 수
                break
            
            if file_path.name.startswith("."):
                continue
            
            # 파일 확장자에 따라 처리
            logger.info(f"파일 처리 중: {file_path.name}")
            prev_count = docs_count
            
            if file_path.suffix == '.txt':
                docs_count = self._process_qa_file(file_path, docs_count)
            elif file_path.suffix in ['.json', '.jsonl']:
                docs_count = self._process_json_file(file_path, docs_count)
            elif file_path.suffix in ['.csv', '.tsv']:
                docs_count = self._process_csv_file(file_path, docs_count)
            
            if docs_count > prev_count:
                processed_files += 1
                logger.info(f"파일 {file_path.name}에서 {docs_count - prev_count}개 문서 추가")
        
        logger.info(f"처리된 파일 수: {processed_files}")
        
        logger.info(f"총 {docs_count}개 문서 로드 완료")
    
    def _process_qa_file(self, file_path: Path, start_count: int) -> int:
        """Q&A 형식 파일 처리"""
        docs_count = start_count
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Q&A 패턴 매칭 - 더 유연한 패턴들
            patterns = [
                # 기본 패턴: [1.Q] ... [1.A] ...
                r'\[(\d+)\.Q\]\s*([\s\S]*?)\n\s*\[(\d+)\.A\]\s*([\s\S]*?)(?=\n\s*\[\d+\.Q\]|$)',
                # 변형 패턴: Q1: ... A1: ...
                r'Q(\d+):\s*([\s\S]*?)\n\s*A(\d+):\s*([\s\S]*?)(?=\n\s*Q\d+:|$)',
                # 질문/답변 패턴
                r'질문(\d+):\s*([\s\S]*?)\n\s*답변(\d+):\s*([\s\S]*?)(?=\n\s*질문\d+:|$)'
            ]
            
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
                if found:
                    matches.extend(found)
                    logger.info(f"패턴 '{pattern[:20]}...'에서 {len(found)}개 발견")
            
            if matches:
                logger.info(f"파일 {file_path.name}에서 총 {len(matches)}개 Q&A 쌍 발견")
            
            for match in matches:
                q_num, question, a_num, answer = match
                if q_num == a_num:
                    question = question.strip()
                    answer = answer.strip()
                    
                    # 빈 문서 스킵
                    if not question or not answer:
                        continue
                    
                    # 카테고리 분류
                    category = self._categorize_simple(question)
                    
                    doc_item = {
                        "id": str(docs_count),
                        "text": question,  # 질문만 벡터화
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "is_special": False,
                        "confidence": 0.8  # 기본 신뢰도
                    }
                    self.documents.append(doc_item)
                    docs_count += 1
                    
        except Exception as e:
            logger.warning(f"파일 처리 오류 {file_path}: {e}")
        
        return docs_count
    
    def _process_json_file(self, file_path: Path, start_count: int) -> int:
        """JSON/JSONL 파일 처리"""
        docs_count = start_count
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix == '.jsonl':
                    # JSONL 파일 처리
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if self._process_json_record(data, docs_count):
                                docs_count += 1
                        except json.JSONDecodeError:
                            continue
                else:
                    # JSON 파일 처리
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if self._process_json_record(item, docs_count):
                                docs_count += 1
                    elif isinstance(data, dict):
                        if self._process_json_record(data, docs_count):
                            docs_count += 1
                            
        except Exception as e:
            logger.warning(f"JSON 파일 처리 오류 {file_path}: {e}")
        
        return docs_count
    
    def _process_json_record(self, record: dict, doc_id: int) -> bool:
        """JSON 레코드 처리"""
        # 다양한 필드명 지원
        question_fields = ['question', 'q', 'query', '질문']
        answer_fields = ['answer', 'a', 'response', '답변', 'output']
        
        question = None
        answer = None
        
        for field in question_fields:
            if field in record:
                question = str(record[field]).strip()
                break
                
        for field in answer_fields:
            if field in record:
                answer = str(record[field]).strip()
                break
        
        if question and answer:
            category = self._categorize_simple(question)
            doc_item = {
                "id": str(doc_id),
                "text": question,
                "question": question,
                "answer": answer,
                "category": category,
                "is_special": False,
                "confidence": 0.8
            }
            self.documents.append(doc_item)
            return True
        
        return False
    
    def _process_csv_file(self, file_path: Path, start_count: int) -> int:
        """CSV/TSV 파일 처리"""
        docs_count = start_count
        
        try:
            import pandas as pd
            delimiter = '\t' if file_path.suffix == '.tsv' else ','
            df = pd.read_csv(file_path, delimiter=delimiter)
            
            # 컬럼명 확인
            logger.info(f"CSV 컬럼: {list(df.columns)}")
            
            for _, row in df.iterrows():
                if self._process_json_record(row.to_dict(), docs_count):
                    docs_count += 1
                    
        except Exception as e:
            logger.warning(f"CSV 파일 처리 오류 {file_path}: {e}")
        
        return docs_count
    
    def _categorize_simple(self, text: str) -> str:
        """간단한 카테고리 분류"""
        text_lower = text.lower()
        
        # 주요 카테고리만 사용
        if any(word in text_lower for word in ["전기기사", "기사시험", "기사 시험"]):
            return "전기기사"
        elif any(word in text_lower for word in ["산업기사", "전기산업기사"]):
            return "전기산업기사"
        elif any(word in text_lower for word in ["기능사", "전기기능사"]):
            return "전기기능사"
        elif any(word in text_lower for word in ["회로", "임피던스", "rc회로", "rlc"]):
            return "회로이론"
        elif any(word in text_lower for word in ["변압기", "트랜스포머"]):
            return "변압기"
        elif any(word in text_lower for word in ["전동기", "모터", "서보모터"]):
            return "전동기"
        elif any(word in text_lower for word in ["발전기", "발전소", "댐", "수력"]):
            return "발전시설"
        elif any(word in text_lower for word in ["송전", "배전", "전력계통"]):
            return "전력시스템"
        elif any(word in text_lower for word in ["안전", "접지", "감전"]):
            return "전기안전"
        else:
            return "일반전기"
    
    def _vectorize_documents(self):
        """문서 벡터화 및 ChromaDB 저장"""
        if not self.documents:
            logger.warning("벡터화할 문서가 없습니다.")
            return
        
        try:
            # 배치 처리
            batch_size = 500  # 배치 크기 증가
            total_batches = (len(self.documents) + batch_size - 1) // batch_size
            
            logger.info(f"{len(self.documents)}개 문서 벡터화 시작...")
            
            for i in range(0, len(self.documents), batch_size):
                batch = self.documents[i:i + batch_size]
                
                # 데이터 준비
                texts = [doc["text"] for doc in batch]
                metadatas = [{
                    "question": doc["question"],
                    "answer": doc["answer"],
                    "category": doc["category"],
                    "is_special": doc.get("is_special", False),
                    "confidence": doc.get("confidence", 0.8)
                } for doc in batch]
                ids = [doc["id"] for doc in batch]
                
                # ChromaDB에 추가
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                current_batch = i // batch_size + 1
                logger.info(f"벡터화 진행: {current_batch}/{total_batches} 배치 완료 ({len(batch)}개 문서)")
            
            logger.info(f"벡터화 완료: {len(self.documents)}개 문서")
            
        except Exception as e:
            logger.error(f"벡터화 실패: {e}")
            raise
    
    def search(self, query: str, k: int = 10) -> Tuple[List[Dict], float]:
        """개선된 검색 알고리즘"""
        start_time = time.time()
        
        # 1. 특수 키워드 확인
        special_result = self._check_special_keywords(query)
        if special_result:
            return [special_result], 1.0
        
        # 2. 벡터 검색
        results = self._vector_search(query, k * 2)  # 더 많은 후보 검색
        
        # 3. 결과 재정렬 및 필터링
        final_results = self._rerank_results(query, results, k)
        
        # 4. 최고 점수 계산
        max_score = max([r["score"] for r in final_results]) if final_results else 0.0
        
        elapsed_time = time.time() - start_time
        logger.info(f"검색 완료: {len(final_results)}개 결과, 최고점수: {max_score:.3f}, 소요시간: {elapsed_time:.2f}초")
        
        return final_results, max_score
    
    def _check_special_keywords(self, query: str) -> Optional[Dict]:
        """특수 키워드 확인"""
        query_lower = query.lower()
        
        for keyword, info in self.special_keywords.items():
            if keyword.lower() in query_lower:
                return {
                    "question": keyword,
                    "answer": info["answer"],
                    "score": info["confidence"],
                    "category": "특수키워드"
                }
        
        return None
    
    def _vector_search(self, query: str, k: int) -> List[Dict]:
        """벡터 검색 수행"""
        try:
            # 컬렉션이 비어있는지 확인
            count = self.collection.count()
            logger.info(f"ChromaDB 문서 수: {count}")
            
            if count == 0:
                logger.warning("ChromaDB에 문서가 없습니다.")
                return []
            
            # ChromaDB 검색
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, count),  # 문서 수보다 많이 요청하지 않도록
                include=["metadatas", "distances", "documents"]
            )
            
            if not results["ids"] or not results["ids"][0]:
                logger.warning("검색 결과가 없습니다.")
                return []
            
            # 결과 변환
            search_results = []
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                
                # 코사인 유사도 계산 (거리를 유사도로 변환)
                similarity = 1 - distance
                
                search_results.append({
                    "question": metadata["question"],
                    "answer": metadata["answer"],
                    "score": similarity,
                    "category": metadata["category"],
                    "is_special": metadata.get("is_special", False),
                    "base_confidence": metadata.get("confidence", 0.8)
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            logger.error(f"오류 세부사항: {type(e).__name__}: {str(e)}")
            return []
    
    def _rerank_results(self, query: str, results: List[Dict], k: int) -> List[Dict]:
        """결과 재정렬 및 점수 조정"""
        if not results:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # 점수 재계산
        for result in results:
            # 기본 점수 (코사인 유사도)
            base_score = result["score"]
            
            # 키워드 매칭 보너스
            question_lower = result["question"].lower()
            question_words = set(question_lower.split())
            
            # 정확한 단어 매칭
            exact_matches = len(query_words.intersection(question_words))
            keyword_bonus = exact_matches * 0.1
            
            # 부분 문자열 매칭
            substring_bonus = 0
            for word in query_words:
                if len(word) > 2 and word in question_lower:
                    substring_bonus += 0.05
            
            # 카테고리 보너스
            category_bonus = 0
            if result["category"] == "특수키워드":
                category_bonus = 0.2
            
            # 최종 점수 계산
            final_score = base_score + keyword_bonus + substring_bonus + category_bonus
            final_score = min(final_score, 1.0)  # 최대 1.0
            
            # 신뢰도 조정
            final_score *= result["base_confidence"]
            
            result["score"] = final_score
            result["debug_scores"] = {
                "base": base_score,
                "keyword": keyword_bonus,
                "substring": substring_bonus,
                "category": category_bonus
            }
        
        # 점수 기준 정렬
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # 임계값 필터링 (매우 낮은 값으로 설정)
        threshold = 0.1
        filtered_results = [r for r in results if r["score"] >= threshold]
        
        return filtered_results[:k]
    
    def _load_sample_data(self):
        """샘플 데이터 로드"""
        sample_qa_pairs = [
            {
                "question": "다산에듀는 무엇인가요?",
                "answer": "미호가 다니는 회사입니다!"
            },
            {
                "question": "R-C회로 합성 임피던스에서 -j를 붙이는 이유는?",
                "answer": "커패시터의 용량성 리액턴스 Xc는 전류가 전압보다 90도 앞서기 때문에, 이를 복소평면에서 표현하면 -jXc가 됩니다."
            },
            {
                "question": "과도현상과 인덕턴스 L의 관계는?",
                "answer": "인덕터는 전류가 갑자기 바뀌는 걸 싫어합니다. 스위치를 켜면, 인덕터 전류는 0에서부터 서서히 올라갑니다. 이 '서서히 올라가는 과정'이 바로 과도현상입니다."
            }
        ]
        
        for i, qa in enumerate(sample_qa_pairs):
            doc_item = {
                "id": f"sample_{i}",
                "text": qa["question"],
                "question": qa["question"],
                "answer": qa["answer"],
                "category": "샘플데이터",
                "is_special": False,
                "confidence": 0.9
            }
            self.documents.append(doc_item)
        
        logger.info(f"샘플 데이터 {len(sample_qa_pairs)}개 로드")