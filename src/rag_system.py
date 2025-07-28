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
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/distiluse-base-multilingual-cased"):
        """
        Args:
            embedding_model_name: 한국어 임베딩 모델 이름
        """
        try:
            os.environ["SAFETENSORS_FAST_GPU"] = "1"
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            
            # 한국어 임베딩 모델 로드
            logger.info(f"임베딩 모델 로드 시도: {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
            logger.info(f"한국어 임베딩 모델 로드 완료: {embedding_model_name}")
            
            # 모델 정보 로깅
            try:
                model_info = {
                    "model_name": embedding_model_name,
                    "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
                    "max_seq_length": getattr(self.embedding_model, 'max_seq_length', 'Unknown'),
                    "device": getattr(self.embedding_model, 'device', 'Unknown')
                }
                logger.info(f"로드된 임베딩 모델 정보: {model_info}")
            except Exception as info_e:
                logger.warning(f"모델 정보 수집 실패: {info_e}")
                
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패 - {embedding_model_name}: {str(e)}")
            logger.error(f"에러 타입: {type(e).__name__}")
            logger.info("폴백 모델로 전환 중...")
            
            # 폴백 모델
            fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            try:
                self.embedding_model = SentenceTransformer(fallback_model)
                logger.info(f"폴백 임베딩 모델 로드 성공: {fallback_model}")
                
                # 폴백 모델 정보 로깅
                fallback_info = {
                    "model_name": fallback_model,
                    "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
                    "max_seq_length": getattr(self.embedding_model, 'max_seq_length', 'Unknown'),
                    "device": getattr(self.embedding_model, 'device', 'Unknown')
                }
                logger.info(f"폴백 모델 정보: {fallback_info}")
                
            except Exception as fallback_e:
                logger.error(f"폴백 모델 로드도 실패: {fallback_e}")
                raise RuntimeError(f"모든 임베딩 모델 로드 실패. 원본 에러: {e}, 폴백 에러: {fallback_e}")
        
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
        
        # 임베딩 모델 테스트
        self._test_embedding_model()
    
    def _test_embedding_model(self):
        """임베딩 모델 기능 테스트"""
        try:
            test_texts = ["안녕하세요", "전기공학 테스트"]
            logger.info("임베딩 모델 기능 테스트 시작...")
            
            embeddings = self.embedding_model.encode(test_texts)
            
            test_info = {
                "test_texts": test_texts,
                "embedding_shape": embeddings.shape,
                "embedding_type": type(embeddings).__name__,
                "sample_values": embeddings[0][:5].tolist() if len(embeddings[0]) >= 5 else embeddings[0].tolist()
            }
            logger.info(f"임베딩 모델 테스트 성공: {test_info}")
            
        except Exception as e:
            logger.error(f"임베딩 모델 테스트 실패: {e}")
            raise RuntimeError(f"임베딩 모델이 제대로 작동하지 않습니다: {e}")
    
    def get_current_embedding_model_info(self) -> dict:
        """현재 사용 중인 임베딩 모델 정보 반환"""
        try:
            return {
                "model_class": self.embedding_model.__class__.__name__,
                "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
                "max_seq_length": getattr(self.embedding_model, 'max_seq_length', 'Unknown'),
                "device": str(getattr(self.embedding_model, 'device', 'Unknown')),
                "modules": [str(module) for module in getattr(self.embedding_model, '_modules', {}).keys()],
                "tokenizer_type": type(getattr(self.embedding_model, 'tokenizer', None)).__name__ if hasattr(self.embedding_model, 'tokenizer') else 'Unknown'
            }
        except Exception as e:
            return {"error": str(e)}
    
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
        """문서 카테고리 자동 분류 - 전기 자격증 종목별 세분화"""
        text_lower = text.lower()
        
        # 1. 전기 자격증 종목별 분류 (우선순위)
        if any(word in text_lower for word in ["전기기사", "기사시험", "기사 시험", "기사필기", "기사실기"]):
            return "전기기사"
        elif any(word in text_lower for word in ["전기산업기사", "산업기사시험", "산업기사 시험", "산업기사필기", "산업기사실기"]):
            return "전기산업기사"
        elif any(word in text_lower for word in ["전기기능사", "기능사시험", "기능사 시험", "기능사필기", "기능사실기"]):
            return "전기기능사"
        elif any(word in text_lower for word in ["전기공사기사", "공사기사"]):
            return "전기공사기사"
        elif any(word in text_lower for word in ["전기공사산업기사", "공사산업기사"]):
            return "전기공사산업기사"
        
        # 2. 기본 전기공학 이론 분야
        elif any(word in text_lower for word in ["옴의법칙", "키르히호프", "전자기학", "맥스웰", "쿨롱", "패러데이", "렌츠"]):
            return "기초이론"
        elif any(word in text_lower for word in ["회로이론", "회로해석", "교류회로", "직류회로", "rlc회로", "공진회로"]):
            return "회로이론"
        
        # 3. 전기기기 분야  
        elif any(word in text_lower for word in ["변압기", "트랜스포머", "전력변압기", "배전용변압기"]):
            return "변압기"
        elif any(word in text_lower for word in ["유도전동기", "동기전동기", "직류전동기", "모터", "전동기"]):
            return "전동기"
        elif any(word in text_lower for word in ["발전기", "동기발전기", "유도발전기", "직류발전기"]):
            return "발전기"
        
        # 4. 전력공학 분야
        elif any(word in text_lower for word in ["송전", "송전선로", "송전계통", "고압송전"]):
            return "송전공학"
        elif any(word in text_lower for word in ["배전", "배전선로", "배전계통", "배전용변압기"]):
            return "배전공학"  
        elif any(word in text_lower for word in ["전력계통", "계통운용", "전력품질", "안정도", "조상설비"]):
            return "전력계통"
        elif any(word in text_lower for word in ["보호계전", "계전기", "차단기", "개폐기", "피뢰기"]):
            return "보호제어"
        
        # 5. 전기설비 및 시공 분야
        elif any(word in text_lower for word in ["전기설비", "수변전설비", "배전반", "분전반"]):
            return "전기설비"
        elif any(word in text_lower for word in ["전기공사", "배선공사", "케이블", "전선", "도관"]):
            return "전기공사"
        elif any(word in text_lower for word in ["접지", "피뢰", "전기안전", "감전", "누전"]):
            return "전기안전"
        
        # 6. 신재생에너지 및 최신기술
        elif any(word in text_lower for word in ["태양광", "풍력", "연료전지", "태양전지", "신재생에너지"]):
            return "신재생에너지"
        elif any(word in text_lower for word in ["전기자동차", "ev충전", "배터리", "스마트그리드"]):
            return "최신기술"
        
        # 7. 전자공학 관련
        elif any(word in text_lower for word in ["반도체", "다이오드", "트랜지스터", "ic", "증폭기"]):
            return "전자공학"
        elif any(word in text_lower for word in ["제어공학", "자동제어", "pid제어", "모터제어"]):
            return "제어공학"
        
        # 8. 기타 일반
        else:
            return "기타"
    
    def _load_sample_data(self):
        """샘플 데이터 로드"""
        sample_docs = [
            {
                "question": "옴의 법칙이 무엇인가요?",
                "answer": "옴의 법칙은 전압(V) = 전류(I) × 저항(R)의 관계를 나타내는 전기공학의 기본 법칙입니다.",
                "category": "기초이론"
            },
            {
                "question": "교류와 직류의 차이점을 설명해주세요.",
                "answer": "직류(DC)는 전류가 한 방향으로만 흐르며, 교류(AC)는 전류의 방향이 주기적으로 바뀝니다.",
                "category": "회로이론"
            },
            {
                "question": "변압기의 동작 원리를 알려주세요.",
                "answer": "변압기는 패러데이의 전자기유도 법칙을 이용하여 권수비에 따라 전압을 변환합니다.",
                "category": "변압기"
            },
            {
                "question": "전기기사 시험은 어떻게 준비하나요?",
                "answer": "전기기사 시험은 필기와 실기로 구성되며, 기본서 학습 후 기출문제를 반복 풀이하는 것이 효과적입니다.",
                "category": "전기기사"
            },
            {
                "question": "유도전동기의 동작 원리는?",
                "answer": "유도전동기는 회전자기장에 의해 회전자가 회전하는 원리로 동작하며, 슬립에 따라 토크가 결정됩니다.",
                "category": "전동기"
            },
            {
                "question": "송전선로의 특성 임피던스는?",
                "answer": "송전선로의 특성 임피던스는 √(L/C)로 계산되며, 일반적으로 400-500Ω 범위입니다.",
                "category": "송전공학"
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
        """고도화된 문서 분류 - 확장된 카테고리 지원"""
        category_keywords = {
            # 자격증 종목별
            '전기기사': ['전기기사', '기사시험', '기사필기', '기사실기'],
            '전기산업기사': ['전기산업기사', '산업기사시험', '산업기사필기', '산업기사실기'],
            '전기기능사': ['전기기능사', '기능사시험', '기능사필기', '기능사실기'],
            '전기공사기사': ['전기공사기사', '공사기사'],
            
            # 이론 분야별
            '기초이론': ['옴의법칙', '키르히호프', '전자기학', '맥스웰', '쿨롱', '패러데이'],
            '회로이론': ['회로이론', '회로해석', '교류회로', '직류회로', 'rlc회로'],
            
            # 기기별
            '변압기': ['변압기', '트랜스포머', '전력변압기'],
            '전동기': ['유도전동기', '동기전동기', '직류전동기', '모터'],
            '발전기': ['발전기', '동기발전기', '유도발전기'],
            
            # 시스템별
            '송전공학': ['송전', '송전선로', '송전계통'],
            '배전공학': ['배전', '배전선로', '배전계통'],
            '전력계통': ['전력계통', '계통운용', '전력품질', '안정도'],
            '보호제어': ['보호계전', '계전기', '차단기'],
            
            # 설비/공사별  
            '전기설비': ['전기설비', '수변전설비', '배전반'],
            '전기공사': ['전기공사', '배선공사', '케이블'],
            '전기안전': ['접지', '피뢰', '전기안전', '감전'],
            
            # 신기술별
            '신재생에너지': ['태양광', '풍력', '연료전지', '신재생에너지'],
            '최신기술': ['전기자동차', 'ev충전', '스마트그리드'],
            '전자공학': ['반도체', '다이오드', '트랜지스터'],
            '제어공학': ['제어공학', '자동제어', 'pid제어']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            if score > 0:
                scores[category] = score
        
        return max(scores.items(), key=lambda x: x[1])[0] if scores else '기타'
    
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
                seen_urls = set()
                
                for result in all_results:
                    url = result.get("href", "")
                    if url in seen_urls:  # 중복 URL 제거
                        continue
                    seen_urls.add(url)
                    
                    title = result.get("title", "")
                    body = result.get("body", "")
                    
                    # 전기공학 관련성 검증
                    relevance_score = self._calculate_web_relevance(query, title, body)
                    if relevance_score < 0.3:  # 관련성 낮은 결과 제외
                        continue
                    
                    # 신뢰도 점수 계산
                    trust_domains = [".edu", ".ac.kr", ".go.kr", "kea.kr", "kstec.or.kr", "keit.re.kr"]
                    trust_score = 2.0 if any(domain in url for domain in trust_domains) else 1.0
                    
                    # 최종 점수
                    final_score = relevance_score * trust_score
                    
                    processed_results.append({
                        "title": title[:100],  # 제목 길이 제한
                        "snippet": body[:300],  # 내용 길이 증가
                        "url": url,
                        "trust_score": trust_score,
                        "relevance_score": relevance_score,
                        "final_score": final_score
                    })
                
                # 최종 점수순 정렬
                processed_results.sort(key=lambda x: x["final_score"], reverse=True)
                
                # 상위 결과만 반환
                top_results = processed_results[:max_results]
                if top_results:
                    self.service_stats["web_searches"] += 1
                
                return top_results
        except Exception as e:
            logger.error(f"웹 검색 실패: {str(e)}")
            return []
    
    def _calculate_web_relevance(self, query: str, title: str, body: str) -> float:
        """웹 검색 결과 관련성 계산"""
        try:
            # 확장된 전기공학 핵심 키워드
            electrical_keywords = [
                # 기본 전기 개념
                '전기', '전력', '전압', '전류', '저항', '회로', '임피던스', '인덕턴스', '커패시턴스',
                # 전기기기
                '변압기', '모터', '발전기', '전동기', '동기기', '유도기', '직류기',
                # 전력시스템
                '송전', '배전', '전력계통', '수변전', '보호계전', '차단기', '개폐기',
                # 전기설비/공사
                '전기설비', '배선', '케이블', '전선', '접지', '피뢰', '분전반', '배전반',
                # 제어/전자
                '제어', '자동제어', 'pid', '반도체', '다이오드', '트랜지스터', 'ic',
                # 신기술
                '태양광', '풍력', '신재생', '스마트그리드', '전기자동차', 'ev충전',
                # 단위/측정
                '와트', '암페어', '볼트', '옴', '헤르츠', 'kw', 'kv', 'a', 'v', 'hz',
                # 자격증/시험
                '전기기사', '전기산업기사', '전기기능사', '전기공사기사', '기사', '산업기사', '기능사', '자격증', '시험', '필기', '실기'
            ]
            
            combined_text = f"{title} {body}".lower()
            query_lower = query.lower()
            
            # 1. 직접 쿼리 매칭
            query_match = 0.5 if query_lower in combined_text else 0
            
            # 2. 전기공학 키워드 매칭
            keyword_matches = sum(1 for keyword in electrical_keywords if keyword in combined_text)
            keyword_score = min(keyword_matches * 0.1, 0.4)
            
            # 3. 제목 중요도 가중치
            title_bonus = 0.2 if any(keyword in title.lower() for keyword in electrical_keywords) else 0
            
            # 4. 문서 품질 평가
            quality_score = 0.1 if len(body) > 100 else 0  # 충분한 내용이 있음
            
            total_score = query_match + keyword_score + title_bonus + quality_score
            return min(total_score, 1.0)
            
        except:
            return 0.3  # 기본값
    
    def check_electrical_relevance(self, query: str) -> bool:
        """확장된 전기공학 관련성 확인"""
        electrical_keywords = [
            # 기본 전기 개념
            "전기", "전력", "전압", "전류", "저항", "회로", "임피던스", "인덕턴스", "커패시턴스",
            "교류", "직류", "AC", "DC", "주파수", "위상", "역률", "전력인수",
            
            # 전기기기
            "변압기", "트랜스포머", "모터", "전동기", "발전기", "동기기", "유도기", "직류기",
            "단상", "삼상", "권선", "철심", "자속", "토크", "슬립", "회전수",
            
            # 전력시스템  
            "송전", "배전", "전력계통", "수변전", "변전소", "보호계전", "차단기", "개폐기",
            "안정도", "조상설비", "무효전력", "전력품질", "고조파", "플리커",
            
            # 전기설비/공사
            "전기설비", "수변전설비", "배선", "케이블", "전선", "도관", "덕트",
            "접지", "피뢰", "누전", "감전", "분전반", "배전반", "제어반",
            
            # 제어/전자
            "제어", "자동제어", "pid제어", "시퀀스제어", "프로그래머블로직컨트롤러", "plc",
            "반도체", "다이오드", "트랜지스터", "thyristor", "ic", "증폭기", "인버터",
            
            # 신기술/에너지
            "태양광", "태양전지", "풍력", "연료전지", "신재생에너지", "esg",
            "스마트그리드", "전기자동차", "ev충전", "배터리", "ess", "마이크로그리드",
            
            # 단위/측정
            "와트", "암페어", "볼트", "옴", "헤르츠", "바", "var", "va",
            "kw", "kv", "ka", "mw", "gw", "kva", "mva", "kwh", "mwh",
            "a", "v", "w", "hz", "ω", "°", "φ", "cosφ",
            
            # 자격증/시험/교육
            "전기기사", "전기산업기사", "전기기능사", "전기공사기사", "전기공사산업기사",
            "기사", "산업기사", "기능사", "자격증", "시험", "필기", "실기", "기출문제",
            "전기공학", "전력공학", "전기기기", "회로이론", "제어공학", "전자공학"
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
        
        # 임베딩 모델 정보 추가
        stats.append("\n**🔧 시스템 정보:**")
        model_info = self.get_current_embedding_model_info()
        if "error" not in model_info:
            stats.append(f"• 임베딩 모델: {model_info.get('model_class', 'Unknown')}")
            stats.append(f"• 임베딩 차원: {model_info.get('embedding_dimension', 'Unknown')}")
            stats.append(f"• 최대 시퀀스: {model_info.get('max_seq_length', 'Unknown')}")
            stats.append(f"• 디바이스: {model_info.get('device', 'Unknown')}")
            if model_info.get('tokenizer_type') != 'Unknown':
                stats.append(f"• 토크나이저: {model_info.get('tokenizer_type', 'Unknown')}")
        else:
            stats.append(f"• 모델 정보 오류: {model_info['error']}")
        
        return "\n".join(stats)
    
    def enhanced_search_pipeline(self, query: str) -> tuple:
        """향상된 통합 검색 파이프라인 - 확장된 범위 지원"""
        # 1. 전기공학 관련성 확인 (확장된 키워드로)
        if not self.check_electrical_relevance(query):
            return [], "non_electrical"
        
        # 2. 벡터 검색 실행 (고도화된 다중 지표)
        db_results, db_found = self.search_vector_database(query)
        
        # 3. 신뢰도 기반 검색 전략 결정
        if db_found and len(db_results) > 0:
            highest_score = db_results[0]["final_score"]
            
            if highest_score > 0.80:
                # 매우 고신뢰도 - 직접 DB 답변
                return db_results, "very_high_confidence_db"
            elif highest_score > 0.70:
                # 고신뢰도 - DB 답변 + 보강
                return db_results, "high_confidence_db"
            elif highest_score > 0.60:
                # 중간신뢰도 - LLM 재구성
                return db_results, "medium_confidence_db"
            elif highest_score > 0.45:
                # 저신뢰도 - 하이브리드 시도
                web_results = self.search_web(query)
                if web_results:
                    return (db_results, web_results), "hybrid_search"
                else:
                    return db_results, "low_confidence_db"
        
        # 4. DB 결과가 부족한 경우 웹검색 시도
        web_results = self.search_web(query)
        if web_results:
            if db_found and len(db_results) > 0:
                # DB + 웹 하이브리드
                return (db_results, web_results), "hybrid_search"
            else:
                # 웹검색 전용
                return web_results, "web_only"
        
        # 5. 모든 검색이 실패한 경우
        if db_found and len(db_results) > 0:
            return db_results, "fallback_db"
        else:
            return [], "no_results"