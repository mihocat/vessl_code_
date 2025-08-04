#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modular RAG System
모듈형 RAG 시스템 - 도메인 독립적 설계
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """쿼리 유형"""
    FACTUAL = "factual"  # 사실 기반 질문
    ANALYTICAL = "analytical"  # 분석적 질문
    COMPARATIVE = "comparative"  # 비교 질문
    PROCEDURAL = "procedural"  # 절차/방법 질문
    CONCEPTUAL = "conceptual"  # 개념 설명 질문
    CREATIVE = "creative"  # 창의적 질문
    MULTIMODAL = "multimodal"  # 멀티모달 질문


@dataclass
class ProcessingContext:
    """처리 컨텍스트"""
    query: str
    query_type: QueryType
    domain: Optional[str] = None
    language: str = "ko"
    image_data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseModule(ABC):
    """기본 모듈 인터페이스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def process(self, context: ProcessingContext, data: Any) -> Any:
        """모듈 처리 메서드"""
        pass
    
    def can_handle(self, context: ProcessingContext) -> bool:
        """이 모듈이 주어진 컨텍스트를 처리할 수 있는지 확인"""
        return True


class OCRModule(BaseModule):
    """OCR 모듈 - 범용 OCR 파이프라인 래퍼"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        from universal_ocr_pipeline import UniversalOCRPipeline, DomainAdaptiveOCR
        
        # 도메인 적응형 OCR 사용
        self.ocr_pipeline = DomainAdaptiveOCR()
    
    def process(self, context: ProcessingContext, data: Any) -> Dict[str, Any]:
        """이미지에서 정보 추출"""
        if not context.image_data:
            return {'success': False, 'message': 'No image data provided'}
        
        # 도메인 정보가 있으면 활용
        if context.domain:
            self.ocr_pipeline._adapt_to_domain(context.domain)
        
        # OCR 처리
        results = self.ocr_pipeline.process_adaptive(
            image=context.image_data,
            auto_detect=True
        )
        
        return results
    
    def can_handle(self, context: ProcessingContext) -> bool:
        """멀티모달 쿼리만 처리"""
        return context.query_type == QueryType.MULTIMODAL and context.image_data is not None


class QueryAnalyzerModule(BaseModule):
    """쿼리 분석 모듈"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.query_patterns = self._load_query_patterns()
    
    def _load_query_patterns(self) -> Dict[str, List[str]]:
        """쿼리 패턴 로드"""
        return {
            QueryType.FACTUAL: [
                r'무엇인가\?', r'누구인가\?', r'언제.*\?', r'어디.*\?',
                r'what is', r'who is', r'when', r'where'
            ],
            QueryType.ANALYTICAL: [
                r'분석', r'평가', r'검토', r'이유', r'원인',
                r'analyze', r'evaluate', r'why', r'cause'
            ],
            QueryType.COMPARATIVE: [
                r'비교', r'차이', r'유사', r'대비',
                r'compare', r'difference', r'similar', r'versus'
            ],
            QueryType.PROCEDURAL: [
                r'방법', r'어떻게', r'절차', r'단계',
                r'how to', r'procedure', r'steps', r'method'
            ],
            QueryType.CONCEPTUAL: [
                r'개념', r'정의', r'의미', r'설명',
                r'concept', r'define', r'meaning', r'explain'
            ]
        }
    
    def process(self, context: ProcessingContext, data: str) -> Dict[str, Any]:
        """쿼리 분석"""
        import re
        
        query_lower = data.lower()
        
        # 쿼리 유형 감지
        detected_type = QueryType.FACTUAL  # 기본값
        max_score = 0
        
        for query_type, patterns in self.query_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
            if score > max_score:
                max_score = score
                detected_type = query_type
        
        # 키워드 추출
        keywords = self._extract_keywords(data)
        
        # 의도 분석
        intent = self._analyze_intent(data, detected_type)
        
        return {
            'query_type': detected_type,
            'keywords': keywords,
            'intent': intent,
            'language': self._detect_language(data),
            'complexity': self._estimate_complexity(data)
        }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """키워드 추출"""
        # 간단한 구현 - 실제로는 더 정교한 NLP 사용
        import re
        
        # 불용어 제거
        stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '에서', 
                     'the', 'is', 'at', 'which', 'on', 'a', 'an'}
        
        words = re.findall(r'\w+', query.lower())
        keywords = [w for w in words if len(w) > 1 and w not in stopwords]
        
        return keywords
    
    def _analyze_intent(self, query: str, query_type: QueryType) -> str:
        """의도 분석"""
        intents = {
            QueryType.FACTUAL: "정보 검색",
            QueryType.ANALYTICAL: "분석 요청",
            QueryType.COMPARATIVE: "비교 분석",
            QueryType.PROCEDURAL: "방법 설명",
            QueryType.CONCEPTUAL: "개념 이해",
            QueryType.CREATIVE: "창의적 생성",
            QueryType.MULTIMODAL: "멀티모달 이해"
        }
        return intents.get(query_type, "일반 질의")
    
    def _detect_language(self, text: str) -> str:
        """언어 감지"""
        # 간단한 휴리스틱
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if korean_chars > english_chars:
            return "ko"
        elif english_chars > korean_chars:
            return "en"
        else:
            return "mixed"
    
    def _estimate_complexity(self, query: str) -> float:
        """복잡도 추정 (0-1)"""
        # 간단한 휴리스틱: 길이, 절 개수, 특수 용어 등
        complexity_factors = {
            'length': min(1.0, len(query) / 200),
            'clauses': min(1.0, query.count(',') + query.count('그리고') + query.count('and') / 5),
            'questions': min(1.0, query.count('?') / 3)
        }
        
        return sum(complexity_factors.values()) / len(complexity_factors)


class RetrievalModule(BaseModule):
    """검색 모듈"""
    
    def __init__(self, vector_db: Any, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vector_db = vector_db
        self.retrieval_strategies = self._init_strategies()
    
    def _init_strategies(self) -> Dict[str, Callable]:
        """검색 전략 초기화"""
        return {
            'vector': self._vector_search,
            'keyword': self._keyword_search,
            'hybrid': self._hybrid_search,
            'semantic': self._semantic_search,
            'contextual': self._contextual_search
        }
    
    def process(self, context: ProcessingContext, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """검색 수행"""
        # 쿼리 유형에 따른 검색 전략 선택
        strategy = self._select_strategy(context)
        
        # 검색 수행
        results = self.retrieval_strategies[strategy](
            query=context.query,
            query_analysis=data,
            k=self.config.get('top_k', 10)
        )
        
        return results
    
    def _select_strategy(self, context: ProcessingContext) -> str:
        """컨텍스트에 따른 검색 전략 선택"""
        if context.query_type == QueryType.FACTUAL:
            return 'hybrid'
        elif context.query_type == QueryType.ANALYTICAL:
            return 'semantic'
        elif context.query_type == QueryType.COMPARATIVE:
            return 'contextual'
        else:
            return 'vector'
    
    def _vector_search(self, query: str, query_analysis: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """벡터 검색"""
        return self.vector_db.search(query, k=k)
    
    def _keyword_search(self, query: str, query_analysis: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """키워드 검색"""
        keywords = query_analysis.get('keywords', [])
        # 키워드 기반 검색 로직
        return []
    
    def _hybrid_search(self, query: str, query_analysis: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """하이브리드 검색"""
        vector_results = self._vector_search(query, query_analysis, k=k*2)
        keyword_results = self._keyword_search(query, query_analysis, k=k*2)
        
        # 결과 병합 및 재순위
        merged_results = self._merge_results(vector_results, keyword_results)
        return merged_results[:k]
    
    def _semantic_search(self, query: str, query_analysis: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """의미 기반 검색"""
        # 쿼리 확장
        expanded_query = self._expand_query(query, query_analysis)
        return self._vector_search(expanded_query, query_analysis, k)
    
    def _contextual_search(self, query: str, query_analysis: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """문맥 기반 검색"""
        # 문맥 정보 활용
        return self._vector_search(query, query_analysis, k)
    
    def _expand_query(self, query: str, analysis: Dict[str, Any]) -> str:
        """쿼리 확장"""
        keywords = analysis.get('keywords', [])
        expanded = query + " " + " ".join(keywords)
        return expanded
    
    def _merge_results(self, results1: List[Dict], results2: List[Dict]) -> List[Dict]:
        """검색 결과 병합"""
        # 간단한 병합 로직
        seen_ids = set()
        merged = []
        
        for result in results1 + results2:
            doc_id = result.get('id')
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged.append(result)
        
        return merged


class RerankingModule(BaseModule):
    """재순위 모듈"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.reranking_models = self._init_reranking_models()
    
    def _init_reranking_models(self):
        """재순위 모델 초기화"""
        models = {}
        
        # Cross-Encoder
        try:
            from sentence_transformers import CrossEncoder
            models['cross_encoder'] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except:
            logger.warning("Cross-encoder not available")
        
        return models
    
    def process(self, context: ProcessingContext, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문서 재순위"""
        if not data:
            return data
        
        # 재순위 전략 선택
        if context.query_type in [QueryType.ANALYTICAL, QueryType.COMPARATIVE]:
            return self._deep_rerank(context, data)
        else:
            return self._fast_rerank(context, data)
    
    def _fast_rerank(self, context: ProcessingContext, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """빠른 재순위"""
        # BM25 또는 간단한 스코어링
        for doc in documents:
            doc['rerank_score'] = self._calculate_simple_score(context.query, doc)
        
        return sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
    
    def _deep_rerank(self, context: ProcessingContext, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """심층 재순위"""
        if 'cross_encoder' in self.reranking_models:
            model = self.reranking_models['cross_encoder']
            
            # Cross-encoder 점수 계산
            pairs = [[context.query, doc.get('content', '')] for doc in documents]
            scores = model.predict(pairs)
            
            for i, doc in enumerate(documents):
                doc['rerank_score'] = float(scores[i])
        else:
            # 폴백
            return self._fast_rerank(context, documents)
        
        return sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
    
    def _calculate_simple_score(self, query: str, document: Dict[str, Any]) -> float:
        """간단한 점수 계산"""
        content = document.get('content', '').lower()
        query_lower = query.lower()
        
        # 단순 키워드 매칭
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in content)
        
        return matches / len(query_words) if query_words else 0.0


class ResponseGeneratorModule(BaseModule):
    """응답 생성 모듈"""
    
    def __init__(self, llm_client: Any, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.llm_client = llm_client
        self.response_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[QueryType, str]:
        """응답 템플릿 로드"""
        return {
            QueryType.FACTUAL: """
다음 정보를 바탕으로 질문에 대한 명확하고 정확한 답변을 제공하세요:

질문: {query}
관련 정보: {context}

답변:
""",
            QueryType.ANALYTICAL: """
다음 정보를 분석하여 질문에 대한 심층적인 답변을 제공하세요:

질문: {query}
분석 자료: {context}

분석 결과:
1. 주요 발견사항:
2. 상세 분석:
3. 결론:
""",
            QueryType.COMPARATIVE: """
다음 정보를 바탕으로 비교 분석을 수행하세요:

질문: {query}
비교 대상 정보: {context}

비교 분석:
- 공통점:
- 차이점:
- 종합 평가:
""",
            QueryType.PROCEDURAL: """
다음 정보를 바탕으로 단계별 절차를 설명하세요:

질문: {query}
참고 정보: {context}

절차:
1단계:
2단계:
3단계:
(필요한 만큼 추가)
""",
            QueryType.CONCEPTUAL: """
다음 정보를 바탕으로 개념을 명확히 설명하세요:

질문: {query}
관련 정보: {context}

개념 설명:
- 정의:
- 핵심 특징:
- 예시:
- 관련 개념:
"""
        }
    
    def process(self, context: ProcessingContext, data: Dict[str, Any]) -> str:
        """응답 생성"""
        # 템플릿 선택
        template = self.response_templates.get(
            context.query_type,
            self.response_templates[QueryType.FACTUAL]
        )
        
        # 컨텍스트 준비
        retrieved_context = self._prepare_context(data.get('documents', []))
        
        # 프롬프트 생성
        prompt = template.format(
            query=context.query,
            context=retrieved_context
        )
        
        # 멀티모달 정보 추가
        if 'ocr_results' in data:
            prompt += self._add_multimodal_context(data['ocr_results'])
        
        # LLM 호출
        response = self.llm_client.generate(prompt)
        
        # 후처리
        return self._postprocess_response(response, context)
    
    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """검색된 문서를 컨텍스트로 준비"""
        context_parts = []
        
        for i, doc in enumerate(documents[:5]):  # 상위 5개만
            content = doc.get('content', '')
            source = doc.get('metadata', {}).get('source', '알 수 없음')
            score = doc.get('rerank_score', doc.get('score', 0))
            
            context_parts.append(f"""
[문서 {i+1}] (관련도: {score:.2f})
출처: {source}
내용: {content[:500]}...
""")
        
        return '\n'.join(context_parts)
    
    def _add_multimodal_context(self, ocr_results: Dict[str, Any]) -> str:
        """멀티모달 컨텍스트 추가"""
        multimodal_parts = ["\n\n이미지 분석 결과:"]
        
        if ocr_results.get('text_regions'):
            multimodal_parts.append("- 텍스트: " + ' '.join([r.text for r in ocr_results['text_regions']]))
        
        if ocr_results.get('formulas'):
            multimodal_parts.append("- 수식: " + ', '.join([f.text for f in ocr_results['formulas']]))
        
        if ocr_results.get('tables'):
            multimodal_parts.append(f"- 표: {len(ocr_results['tables'])}개 발견")
        
        if ocr_results.get('charts'):
            multimodal_parts.append("- 차트 분석 완료")
        
        return '\n'.join(multimodal_parts)
    
    def _postprocess_response(self, response: str, context: ProcessingContext) -> str:
        """응답 후처리"""
        # 언어별 처리
        if context.language == "ko":
            response = self._ensure_korean_politeness(response)
        
        # 도메인별 처리
        if context.domain:
            response = self._apply_domain_formatting(response, context.domain)
        
        return response
    
    def _ensure_korean_politeness(self, text: str) -> str:
        """한국어 공손성 확인"""
        # 존댓말 확인 및 수정
        if not text.endswith(('습니다.', '니다.', '세요.', '입니다.')):
            if text.endswith('.'):
                text = text[:-1] + '습니다.'
        
        return text
    
    def _apply_domain_formatting(self, text: str, domain: str) -> str:
        """도메인별 포맷팅"""
        # 도메인별 특수 포맷팅 규칙 적용
        return text


class ModularRAGPipeline:
    """모듈형 RAG 파이프라인"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        모듈형 RAG 파이프라인 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.modules = {}
        self._init_modules()
    
    def _init_modules(self):
        """모듈 초기화"""
        # 쿼리 분석 모듈
        self.modules['query_analyzer'] = QueryAnalyzerModule(self.config.get('query_analyzer'))
        
        # OCR 모듈
        self.modules['ocr'] = OCRModule(self.config.get('ocr'))
        
        # 검색 모듈
        if 'vector_db' in self.config:
            self.modules['retrieval'] = RetrievalModule(
                vector_db=self.config['vector_db'],
                config=self.config.get('retrieval')
            )
        
        # 재순위 모듈
        self.modules['reranking'] = RerankingModule(self.config.get('reranking'))
        
        # 응답 생성 모듈
        if 'llm_client' in self.config:
            self.modules['response_generator'] = ResponseGeneratorModule(
                llm_client=self.config['llm_client'],
                config=self.config.get('response_generator')
            )
    
    def process(
        self,
        query: str,
        image: Optional[Any] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        RAG 파이프라인 처리
        
        Args:
            query: 사용자 질의
            image: 선택적 이미지 입력
            domain: 도메인 힌트
            **kwargs: 추가 파라미터
            
        Returns:
            처리 결과
        """
        # 처리 컨텍스트 생성
        context = ProcessingContext(
            query=query,
            query_type=QueryType.MULTIMODAL if image else QueryType.FACTUAL,
            domain=domain,
            image_data=image,
            metadata=kwargs
        )
        
        # 파이프라인 실행 결과
        results = {
            'success': True,
            'query': query,
            'domain': domain
        }
        
        try:
            # 1. 쿼리 분석
            if 'query_analyzer' in self.modules:
                query_analysis = self.modules['query_analyzer'].process(context, query)
                results['query_analysis'] = query_analysis
                
                # 컨텍스트 업데이트
                context.query_type = query_analysis['query_type']
                context.language = query_analysis['language']
            
            # 2. 멀티모달 처리 (이미지가 있는 경우)
            if image and 'ocr' in self.modules:
                ocr_results = self.modules['ocr'].process(context, image)
                results['ocr_results'] = ocr_results
                
                # OCR 결과를 쿼리에 통합
                if ocr_results.get('success'):
                    context.query = self._enhance_query_with_ocr(query, ocr_results)
            
            # 3. 검색
            if 'retrieval' in self.modules:
                search_results = self.modules['retrieval'].process(
                    context,
                    results.get('query_analysis', {})
                )
                results['search_results'] = search_results
            
            # 4. 재순위
            if 'reranking' in self.modules and 'search_results' in results:
                reranked_results = self.modules['reranking'].process(
                    context,
                    results['search_results']
                )
                results['documents'] = reranked_results
            
            # 5. 응답 생성
            if 'response_generator' in self.modules:
                response = self.modules['response_generator'].process(context, results)
                results['response'] = response
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def _enhance_query_with_ocr(self, query: str, ocr_results: Dict[str, Any]) -> str:
        """OCR 결과로 쿼리 향상"""
        enhanced_parts = [query]
        
        # 주요 텍스트 추가
        if ocr_results.get('unified_content'):
            enhanced_parts.append(f"이미지 내용: {ocr_results['unified_content'][:200]}")
        
        return ' '.join(enhanced_parts)
    
    def add_module(self, name: str, module: BaseModule):
        """모듈 추가"""
        self.modules[name] = module
    
    def remove_module(self, name: str):
        """모듈 제거"""
        if name in self.modules:
            del self.modules[name]
    
    def get_module(self, name: str) -> Optional[BaseModule]:
        """모듈 가져오기"""
        return self.modules.get(name)


# 사용 예시
if __name__ == "__main__":
    # 설정
    config = {
        'query_analyzer': {},
        'ocr': {},
        'retrieval': {'top_k': 10},
        'reranking': {},
        'response_generator': {}
    }
    
    # 파이프라인 생성
    pipeline = ModularRAGPipeline(config)
    
    print("Modular RAG Pipeline initialized")