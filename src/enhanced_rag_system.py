#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced RAG System with Multimodal Support
멀티모달 지원을 위한 향상된 RAG 시스템
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import torch
from PIL import Image
import json
import re
from datetime import datetime

# 벡터 DB
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# 임베딩 모델
from sentence_transformers import SentenceTransformer

# 기존 RAG 시스템
from rag_system import SearchResult, VectorStore

# 멀티모달 컴포넌트
from enhanced_image_analyzer import ChatGPTStyleAnalyzer
from chatgpt_response_generator import ChatGPTResponseGenerator

logger = logging.getLogger(__name__)


class MultimodalEmbedder:
    """멀티모달 임베딩 생성기"""
    
    def __init__(self):
        """멀티모달 임베더 초기화"""
        # 텍스트 임베딩 모델
        self.text_embedder = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)
        
        # 수식 임베딩을 위한 전처리기
        self.formula_preprocessor = FormulaPreprocessor()
        
        # 이미지 분석기
        self.image_analyzer = ChatGPTStyleAnalyzer()
        
        logger.info("Multimodal embedder initialized")
    
    def embed_text(self, text: str) -> np.ndarray:
        """텍스트 임베딩"""
        return self.text_embedder.encode(text, convert_to_numpy=True)
    
    def embed_formula(self, formula: str) -> np.ndarray:
        """수식 임베딩 (LaTeX)"""
        # 수식 정규화
        normalized = self.formula_preprocessor.normalize(formula)
        
        # 수식을 텍스트로 변환
        text_repr = self.formula_preprocessor.to_text(normalized)
        
        # 임베딩
        return self.text_embedder.encode(text_repr, convert_to_numpy=True)
    
    def embed_multimodal(self, content: Dict[str, Any]) -> np.ndarray:
        """멀티모달 컨텐츠 통합 임베딩"""
        embeddings = []
        weights = []
        
        # 텍스트 임베딩
        if content.get('text'):
            text_emb = self.embed_text(content['text'])
            embeddings.append(text_emb)
            weights.append(0.5)  # 텍스트 가중치
        
        # 수식 임베딩
        if content.get('formulas'):
            formula_embs = [self.embed_formula(f) for f in content['formulas']]
            if formula_embs:
                avg_formula_emb = np.mean(formula_embs, axis=0)
                embeddings.append(avg_formula_emb)
                weights.append(0.3)  # 수식 가중치
        
        # 다이어그램 설명 임베딩
        if content.get('diagrams'):
            diagram_texts = []
            for diagram in content['diagrams']:
                if isinstance(diagram, dict):
                    components = diagram.get('components', [])
                    diagram_text = ' '.join([c.get('label', '') for c in components])
                    diagram_texts.append(diagram_text)
            
            if diagram_texts:
                diagram_emb = self.embed_text(' '.join(diagram_texts))
                embeddings.append(diagram_emb)
                weights.append(0.2)  # 다이어그램 가중치
        
        # 가중 평균
        if embeddings:
            weights = np.array(weights) / np.sum(weights)
            combined = np.sum([emb * w for emb, w in zip(embeddings, weights)], axis=0)
            return combined
        
        # 빈 컨텐츠의 경우
        return np.zeros(self.text_embedder.get_sentence_embedding_dimension())


class FormulaPreprocessor:
    """수식 전처리기"""
    
    def normalize(self, formula: str) -> str:
        """LaTeX 수식 정규화"""
        # 공백 정리
        formula = re.sub(r'\s+', ' ', formula.strip())
        
        # 분수 정규화
        formula = re.sub(r'\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}', r'(\1)/(\2)', formula)
        
        # 제곱 정규화
        formula = re.sub(r'\^(\d+)', r'^{\1}', formula)
        formula = re.sub(r'\^\{([^}]+)\}', r'^(\1)', formula)
        
        return formula
    
    def to_text(self, formula: str) -> str:
        """수식을 텍스트 표현으로 변환"""
        # LaTeX 명령어를 텍스트로
        replacements = {
            r'\\sqrt': 'square root of',
            r'\\times': 'times',
            r'\\div': 'divided by',
            r'\\pm': 'plus minus',
            r'\\sum': 'sum of',
            r'\\int': 'integral of',
            r'\\alpha': 'alpha',
            r'\\beta': 'beta',
            r'\\gamma': 'gamma',
            r'\\theta': 'theta',
            r'\\omega': 'omega',
            r'\\Omega': 'Omega',
            r'\\pi': 'pi',
            r'\\cos': 'cosine',
            r'\\sin': 'sine',
            r'\\tan': 'tangent',
        }
        
        text = formula
        for latex, plain in replacements.items():
            text = text.replace(latex, plain)
        
        # 특수 기호 제거
        text = re.sub(r'[{}\\]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


class HybridSearchEngine:
    """하이브리드 검색 엔진 (키워드 + 벡터)"""
    
    def __init__(self, vector_db: 'EnhancedVectorDatabase'):
        """하이브리드 검색 엔진 초기화"""
        self.vector_db = vector_db
        self.keyword_index = {}
        
        # 전기공학 키워드 사전
        self.domain_keywords = {
            '전압': ['voltage', 'V', '볼트'],
            '전류': ['current', 'I', '암페어', 'A'],
            '저항': ['resistance', 'R', '옴', 'Ω'],
            '전력': ['power', 'P', '와트', 'W'],
            '임피던스': ['impedance', 'Z'],
            '역률': ['power factor', 'PF', 'cosθ'],
            '변압기': ['transformer', '변환'],
            '모터': ['motor', '전동기'],
        }
    
    def build_keyword_index(self, documents: List[Dict[str, Any]]):
        """키워드 인덱스 구축"""
        self.keyword_index = {}
        
        for i, doc in enumerate(documents):
            text = doc.get('text', '').lower()
            
            # 도메인 키워드 추출
            for main_term, variations in self.domain_keywords.items():
                for term in [main_term] + variations:
                    if term.lower() in text:
                        if term not in self.keyword_index:
                            self.keyword_index[term] = []
                        self.keyword_index[term].append(i)
    
    def search(
        self, 
        query: str, 
        image_analysis: Optional[Dict[str, Any]] = None,
        k: int = 10,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        하이브리드 검색
        
        Args:
            query: 검색 쿼리
            image_analysis: 이미지 분석 결과
            k: 반환할 결과 수
            alpha: 벡터 검색 가중치 (1-alpha는 키워드 검색 가중치)
        """
        # 1. 벡터 검색
        vector_results = self.vector_db.search(query, k=k*2)
        
        # 2. 키워드 검색
        keyword_scores = self._keyword_search(query)
        
        # 3. 이미지 컨텍스트 검색 (있는 경우)
        if image_analysis:
            image_scores = self._image_context_search(image_analysis)
            
            # 점수 결합
            for doc_id, score in image_scores.items():
                if doc_id in keyword_scores:
                    keyword_scores[doc_id] += score * 0.3
                else:
                    keyword_scores[doc_id] = score * 0.3
        
        # 4. 점수 결합
        combined_scores = {}
        
        # 벡터 검색 점수
        for result in vector_results:
            doc_id = result['metadata'].get('doc_id', result['id'])
            combined_scores[doc_id] = alpha * (1 - result['distance'])
        
        # 키워드 검색 점수
        max_keyword_score = max(keyword_scores.values()) if keyword_scores else 1
        for doc_id, score in keyword_scores.items():
            normalized_score = score / max_keyword_score
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - alpha) * normalized_score
            else:
                combined_scores[doc_id] = (1 - alpha) * normalized_score
        
        # 5. 상위 k개 선택
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 결과 구성
        results = []
        for doc_id, score in sorted_docs[:k]:
            # 원본 문서 정보 가져오기
            doc_info = self._get_document_info(doc_id, vector_results)
            if doc_info:
                doc_info['hybrid_score'] = score
                results.append(doc_info)
        
        return results
    
    def _keyword_search(self, query: str) -> Dict[str, float]:
        """키워드 기반 검색"""
        scores = {}
        query_lower = query.lower()
        
        # 쿼리에서 키워드 추출
        for term, doc_ids in self.keyword_index.items():
            if term.lower() in query_lower:
                for doc_id in doc_ids:
                    if doc_id not in scores:
                        scores[doc_id] = 0
                    scores[doc_id] += 1
        
        return scores
    
    def _image_context_search(self, image_analysis: Dict[str, Any]) -> Dict[str, float]:
        """이미지 컨텍스트 기반 검색"""
        scores = {}
        
        # 수식에서 키워드 추출
        if image_analysis.get('formulas'):
            for formula in image_analysis['formulas']:
                # 수식에서 변수/상수 추출
                variables = re.findall(r'[A-Za-z]+', formula)
                for var in variables:
                    if var in self.keyword_index:
                        for doc_id in self.keyword_index[var]:
                            if doc_id not in scores:
                                scores[doc_id] = 0
                            scores[doc_id] += 0.5
        
        # 다이어그램 컴포넌트에서 키워드 추출
        if image_analysis.get('diagrams'):
            for diagram in image_analysis['diagrams']:
                if isinstance(diagram, dict) and 'components' in diagram:
                    for comp in diagram['components']:
                        label = comp.get('label', '')
                        if label in self.keyword_index:
                            for doc_id in self.keyword_index[label]:
                                if doc_id not in scores:
                                    scores[doc_id] = 0
                                scores[doc_id] += 0.3
        
        return scores
    
    def _get_document_info(self, doc_id: str, vector_results: List[Dict]) -> Optional[Dict]:
        """문서 정보 가져오기"""
        for result in vector_results:
            if result['metadata'].get('doc_id', result['id']) == doc_id:
                return result
        return None


class EnhancedVectorDatabase(VectorStore):
    """향상된 벡터 데이터베이스"""
    
    def __init__(self, rag_config=None, embedding_model=None):
        """향상된 벡터 DB 초기화"""
        # 기본 설정 사용
        if rag_config is None:
            from config import RAGConfig
            rag_config = RAGConfig()
            
        # 임베딩 모델이 제공되지 않으면 생성
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(rag_config.embedding_model_name, trust_remote_code=True)
        
        # 부모 클래스 초기화
        super().__init__(rag_config, embedding_model)
        
        # 멀티모달 임베더
        self.multimodal_embedder = MultimodalEmbedder()
        
        # 하이브리드 검색 엔진
        self.hybrid_search = HybridSearchEngine(self)
        
        # 메타데이터 인덱스
        self.metadata_index = {
            'formulas': {},
            'concepts': {},
            'difficulty': {}
        }
    
    def add_multimodal_document(
        self,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """멀티모달 문서 추가"""
        # 멀티모달 임베딩 생성
        embedding = self.multimodal_embedder.embed_multimodal(content)
        
        # 메타데이터 구성
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'has_formulas': len(content.get('formulas', [])) > 0,
            'has_diagrams': len(content.get('diagrams', [])) > 0,
            'content_type': 'multimodal',
            'timestamp': datetime.now().isoformat()
        })
        
        # 문서 ID 생성
        doc_id = f"multimodal_{datetime.now().timestamp()}"
        
        # ChromaDB에 추가
        self.collection.add(
            embeddings=[embedding.tolist()],
            documents=[json.dumps(content, ensure_ascii=False)],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        # 인덱스 업데이트
        self._update_indexes(doc_id, content, metadata)
        
        return doc_id
    
    def _update_indexes(self, doc_id: str, content: Dict[str, Any], metadata: Dict[str, Any]):
        """인덱스 업데이트"""
        # 수식 인덱스
        if content.get('formulas'):
            for formula in content['formulas']:
                formula_key = self._normalize_formula_key(formula)
                if formula_key not in self.metadata_index['formulas']:
                    self.metadata_index['formulas'][formula_key] = []
                self.metadata_index['formulas'][formula_key].append(doc_id)
        
        # 개념 인덱스
        if metadata.get('concepts'):
            for concept in metadata['concepts']:
                if concept not in self.metadata_index['concepts']:
                    self.metadata_index['concepts'][concept] = []
                self.metadata_index['concepts'][concept].append(doc_id)
    
    def _normalize_formula_key(self, formula: str) -> str:
        """수식 키 정규화"""
        # 공백 제거, 소문자 변환
        key = re.sub(r'\s+', '', formula.lower())
        # LaTeX 명령어 제거
        key = re.sub(r'\\[a-zA-Z]+', '', key)
        return key
    
    def search_multimodal(
        self,
        query: str,
        image_analysis: Optional[Dict[str, Any]] = None,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """멀티모달 검색"""
        # 하이브리드 검색 수행
        results = self.hybrid_search.search(
            query=query,
            image_analysis=image_analysis,
            k=k
        )
        
        # 메타데이터 필터링
        if filter_metadata:
            results = [r for r in results if self._match_metadata(r['metadata'], filter_metadata)]
        
        # 결과 후처리
        processed_results = []
        for result in results:
            # JSON 문서 파싱
            if result['documents']:
                try:
                    content = json.loads(result['documents'][0])
                    result['parsed_content'] = content
                except:
                    pass
            
            processed_results.append(result)
        
        return processed_results
    
    def _match_metadata(self, doc_metadata: Dict, filter_metadata: Dict) -> bool:
        """메타데이터 매칭"""
        for key, value in filter_metadata.items():
            if key not in doc_metadata:
                return False
            if isinstance(value, list):
                if doc_metadata[key] not in value:
                    return False
            elif doc_metadata[key] != value:
                return False
        return True


class EnhancedRAGSystem:
    """향상된 RAG 시스템"""
    
    def __init__(self, vector_db: EnhancedVectorDatabase, llm_client):
        """향상된 RAG 시스템 초기화"""
        self.vector_db = vector_db
        self.llm_client = llm_client
        self.response_generator = ChatGPTResponseGenerator()
        self.image_analyzer = ChatGPTStyleAnalyzer()
        
        logger.info("Enhanced RAG system initialized")
    
    def process_query(
        self,
        query: str,
        image: Optional[Union[Image.Image, str, bytes]] = None,
        response_style: str = 'comprehensive'
    ) -> Dict[str, Any]:
        """
        쿼리 처리 (텍스트 + 이미지)
        
        Args:
            query: 사용자 질문
            image: 이미지 (선택적)
            response_style: 응답 스타일
        """
        # 1. 이미지 분석 (있는 경우)
        image_analysis = None
        if image:
            logger.info("Analyzing image for context...")
            
            # 멀티모달 OCR 파이프라인 우선 사용
            try:
                from multimodal_ocr import MultimodalOCRPipeline
                ocr_pipeline = MultimodalOCRPipeline()
                ocr_result = ocr_pipeline.process_image(image)
                
                # OCR 결과를 image_analysis 형식으로 변환
                image_analysis = {
                    'success': True,
                    'ocr_text': ocr_result.get('text_content', ''),
                    'formulas': ocr_result.get('formulas', []),
                    'diagrams': ocr_result.get('diagrams', []),
                    'tables': ocr_result.get('tables', []),
                    'structured_content': ocr_result.get('structured_content', '')
                }
                logger.info(f"Multimodal OCR extracted: {len(image_analysis['ocr_text'])} chars, {len(image_analysis['formulas'])} formulas")
            except Exception as e:
                logger.warning(f"Multimodal OCR failed, falling back to Florence-2: {e}")
                # Florence-2로 폴백
                image_analysis = self.image_analyzer.analyze_for_chatgpt_response(image)
            
            # 이미지에서 추출된 텍스트를 쿼리에 추가
            if image_analysis.get('ocr_text'):
                query = f"{query}\n\n[이미지 내용: {image_analysis['ocr_text'][:200]}...]"
        
        # 2. 관련 문서 검색
        logger.info("Searching for relevant documents...")
        search_results = self.vector_db.search_multimodal(
            query=query,
            image_analysis=image_analysis,
            k=5
        )
        
        # 3. 컨텍스트 구성
        context = self._build_context(search_results, image_analysis)
        
        # 4. LLM 프롬프트 생성
        prompt = self._build_prompt(query, context, response_style)
        
        # 5. LLM 응답 생성
        logger.info("Generating response...")
        llm_response = self.llm_client.generate(prompt)
        
        # 6. ChatGPT 스타일 포맷팅
        formatted_response = self.response_generator.generate_response(
            question=query,
            context=context,
            response_type=response_style
        )
        
        # 7. 최종 응답 통합
        final_response = self._integrate_responses(llm_response, formatted_response, context)
        
        return {
            'success': True,
            'query': query,
            'response': final_response,
            'context': context,
            'search_results': search_results,
            'image_analysis': image_analysis
        }
    
    def _build_context(
        self,
        search_results: List[Dict[str, Any]],
        image_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """컨텍스트 구성"""
        context = {
            'retrieved_documents': [],
            'key_concepts': set(),
            'formulas': [],
            'visual_elements': {},
            'solution_steps': []
        }
        
        # 검색 결과에서 컨텍스트 추출
        for result in search_results:
            doc_content = result.get('parsed_content', {})
            
            # 문서 추가
            context['retrieved_documents'].append({
                'content': doc_content.get('text', ''),
                'score': result.get('hybrid_score', 0),
                'metadata': result.get('metadata', {})
            })
            
            # 수식 수집
            if doc_content.get('formulas'):
                context['formulas'].extend(doc_content['formulas'])
            
            # 개념 수집
            if result.get('metadata', {}).get('concepts'):
                context['key_concepts'].update(result['metadata']['concepts'])
        
        # 이미지 분석 결과 통합
        if image_analysis:
            if image_analysis.get('key_concepts'):
                context['key_concepts'].update(image_analysis['key_concepts'])
            
            if image_analysis.get('solution_steps'):
                context['solution_steps'] = image_analysis['solution_steps']
            
            if image_analysis.get('visual_elements'):
                context['visual_elements'] = image_analysis['visual_elements']
        
        # set을 list로 변환
        context['key_concepts'] = list(context['key_concepts'])
        
        return context
    
    def _build_prompt(self, query: str, context: Dict[str, Any], style: str) -> str:
        """LLM 프롬프트 생성"""
        prompt = f"""당신은 전기공학 분야의 최고 전문가입니다.
다음 질문에 대해 ChatGPT 스타일로 답변해주세요.

질문: {query}

참고 자료:
"""
        
        # 검색된 문서 추가
        for i, doc in enumerate(context['retrieved_documents'][:3]):
            prompt += f"\n[문서 {i+1}]:\n{doc['content'][:500]}...\n"
        
        # 수식 정보 추가
        if context['formulas']:
            prompt += "\n관련 수식:\n"
            for formula in context['formulas'][:5]:
                prompt += f"- ${formula}$\n"
        
        # 스타일 지시
        if style == 'comprehensive':
            prompt += "\n종합적이고 상세한 답변을 제공해주세요."
        elif style == 'step_by_step':
            prompt += "\n단계별로 명확하게 설명해주세요."
        elif style == 'concept':
            prompt += "\n개념을 중심으로 이해하기 쉽게 설명해주세요."
        
        return prompt
    
    def _integrate_responses(
        self,
        llm_response: str,
        formatted_response: str,
        context: Dict[str, Any]
    ) -> str:
        """LLM 응답과 포맷된 응답 통합"""
        # 기본적으로 포맷된 응답 사용
        # LLM 응답에서 추가 정보 추출하여 보강
        
        # 수식이 있으면 LaTeX 형식으로 변환
        integrated = formatted_response
        
        # LLM 응답에서 누락된 정보 추가
        if "예시" in llm_response and "예시" not in integrated:
            # 예시 섹션 추가
            integrated += "\n\n📝 **추가 예시:**\n"
            # 예시 추출 로직...
        
        return integrated


# 기존 RAG 시스템과의 호환성을 위한 어댑터
class RAGSystemAdapter:
    """기존 RAG 시스템 인터페이스와 호환"""
    
    def __init__(self, enhanced_rag: EnhancedRAGSystem):
        self.enhanced_rag = enhanced_rag
    
    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """기존 인터페이스와 호환"""
        result = self.enhanced_rag.process_query(
            query=question,
            response_style='comprehensive'
        )
        
        # 기존 형식으로 변환
        return {
            "answer": result['response'],
            "sources": [doc['content'] for doc in result['context']['retrieved_documents'][:k]],
            "scores": [doc['score'] for doc in result['context']['retrieved_documents'][:k]]
        }


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    # 향상된 벡터 DB 초기화
    vector_db = EnhancedVectorDatabase()
    
    # 테스트 문서 추가
    test_content = {
        'text': '3상 전력 시스템에서 선간전압이 380V이고 역률이 0.8일 때의 전력 계산',
        'formulas': [
            'P = \\sqrt{3} \\times V_L \\times I_L \\times \\cos\\theta',
            'S = \\sqrt{3} \\times V_L \\times I_L'
        ],
        'diagrams': [{
            'type': 'circuit',
            'components': [
                {'type': 'voltage', 'label': 'V1'},
                {'type': 'resistor', 'label': 'R1'},
                {'type': 'inductor', 'label': 'L1'}
            ]
        }]
    }
    
    doc_id = vector_db.add_multimodal_document(
        content=test_content,
        metadata={'concepts': ['3상 전력', '역률', '전력 계산']}
    )
    
    print(f"Added document: {doc_id}")
    
    # 검색 테스트
    results = vector_db.search_multimodal(
        query="3상 전력 계산 방법",
        k=3
    )
    
    print(f"\nSearch results: {len(results)} found")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result.get('hybrid_score', 0):.3f}")