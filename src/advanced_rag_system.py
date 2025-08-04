#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced RAG System with Reranker and Reasoning Chain
리랭커와 추론 체인을 갖춘 고급 RAG 시스템
"""

import logging
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import time

# 기존 컴포넌트
from enhanced_rag_system import EnhancedVectorDatabase, MultimodalEmbedder
from enhanced_image_analyzer import ChatGPTStyleAnalyzer
from chatgpt_response_generator import ChatGPTResponseGenerator

# 새로운 컴포넌트
from reranker_system import HybridReranker
from rag_reasoning_chain import ElectricalEngineeringReasoner

logger = logging.getLogger(__name__)


class AdvancedRAGSystem:
    """고급 RAG 시스템"""
    
    def __init__(self, vector_db: EnhancedVectorDatabase, llm_client):
        """
        고급 RAG 시스템 초기화
        
        Args:
            vector_db: 향상된 벡터 데이터베이스
            llm_client: LLM 클라이언트
        """
        self.vector_db = vector_db
        self.llm_client = llm_client
        
        # 컴포넌트 초기화
        self.image_analyzer = ChatGPTStyleAnalyzer()
        self.response_generator = ChatGPTResponseGenerator()
        
        # 리랭커 초기화
        self.reranker = HybridReranker()
        
        # 추론 체인 초기화
        self.reasoner = ElectricalEngineeringReasoner(
            llm_client=llm_client,
            vector_db=vector_db,
            reranker=self.reranker
        )
        
        # 범용 OCR 파이프라인
        try:
            from universal_ocr_pipeline import DomainAdaptiveOCR
            self.ocr_pipeline = DomainAdaptiveOCR()
            logger.info("Universal Domain-Adaptive OCR pipeline loaded")
        except:
            logger.warning("Universal OCR pipeline not available, trying Korean OCR")
            try:
                from korean_ocr_pipeline import KoreanElectricalOCR
                self.ocr_pipeline = KoreanElectricalOCR()
                logger.info("Korean Electrical OCR pipeline loaded")
            except:
                logger.warning("Korean OCR pipeline not available, trying multimodal")
                try:
                    from multimodal_ocr import MultimodalOCRPipeline
                    self.ocr_pipeline = MultimodalOCRPipeline()
                    logger.info("Multimodal OCR pipeline loaded")
                except:
                    logger.warning("No OCR pipeline available")
                    self.ocr_pipeline = None
        
        logger.info("Advanced RAG system initialized")
    
    def process_query_advanced(
        self,
        query: str,
        image: Optional[Union[Image.Image, str, bytes]] = None,
        mode: str = 'reasoning',  # 'fast', 'balanced', 'reasoning'
        response_style: str = 'comprehensive'
    ) -> Dict[str, Any]:
        """
        고급 쿼리 처리
        
        Args:
            query: 사용자 질문
            image: 이미지 (선택적)
            mode: 처리 모드
            response_style: 응답 스타일
        """
        start_time = time.time()
        
        try:
            # 1. 이미지 분석 (있는 경우)
            image_analysis = None
            if image:
                image_analysis = self._analyze_image_advanced(image)
                
                # 이미지 컨텍스트를 쿼리에 추가
                if image_analysis.get('ocr_text'):
                    query = f"{query}\n\n[이미지 내용: {image_analysis['ocr_text'][:200]}...]"
            
            # 2. 처리 모드별 실행
            if mode == 'reasoning':
                # 추론 체인 사용
                result = self._process_with_reasoning(query, image_analysis)
            elif mode == 'balanced':
                # 리랭킹만 사용
                result = self._process_with_reranking(query, image_analysis, response_style)
            else:  # fast
                # 기본 검색만 사용
                result = self._process_fast(query, image_analysis, response_style)
            
            # 3. 응답 시간 추가
            result['processing_time'] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Advanced query processing failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _analyze_image_advanced(self, image: Union[Image.Image, str, bytes]) -> Dict[str, Any]:
        """고급 이미지 분석"""
        # 멀티모달 OCR 우선 사용
        if self.ocr_pipeline:
            try:
                if hasattr(self.ocr_pipeline, 'process_adaptive'):
                    # Universal Domain-Adaptive OCR
                    logger.info("Using Universal Domain-Adaptive OCR pipeline...")
                    ocr_result = self.ocr_pipeline.process_adaptive(image, auto_detect=True)
                elif hasattr(self.ocr_pipeline, 'extract_electrical_data'):
                    # Korean Electrical OCR
                    logger.info("Using Korean Electrical OCR pipeline...")
                    ocr_result = self.ocr_pipeline.extract_electrical_data(image)
                else:
                    # Standard multimodal OCR
                    logger.info("Using multimodal OCR pipeline...")
                    ocr_result = self.ocr_pipeline.process_image(image)
                
                # 결과 형식 변환
                return {
                    'success': True,
                    'ocr_text': ocr_result.get('text_content', ''),
                    'formulas': ocr_result.get('formulas', []),
                    'diagrams': ocr_result.get('diagrams', []),
                    'tables': ocr_result.get('tables', []),
                    'structured_content': ocr_result.get('structured_content', ''),
                    'layout_analysis': ocr_result.get('layout_analysis', {}),
                    'electrical_analysis': ocr_result.get('electrical_analysis', {})
                }
            except Exception as e:
                logger.warning(f"Multimodal OCR failed: {e}, falling back to Florence-2")
        
        # Florence-2 폴백
        return self.image_analyzer.analyze_for_chatgpt_response(image)
    
    def _process_with_reasoning(
        self,
        query: str,
        image_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """추론 체인을 사용한 처리"""
        logger.info("Processing with reasoning chain...")
        
        # 추론 수행
        reasoning_result = self.reasoner.reason(
            query=query,
            image_analysis=image_analysis,
            max_iterations=2
        )
        
        if reasoning_result['success']:
            return {
                'success': True,
                'query': query,
                'response': reasoning_result['answer'],
                'reasoning_trace': reasoning_result['reasoning_trace'],
                'evidence': self._format_evidence(reasoning_result['evidence']),
                'confidence': reasoning_result.get('context', {}).get('confidence', 0.8),
                'mode': 'reasoning'
            }
        else:
            # 폴백: 리랭킹 모드로
            logger.warning("Reasoning failed, falling back to reranking mode")
            return self._process_with_reranking(query, image_analysis, 'comprehensive')
    
    def _process_with_reranking(
        self,
        query: str,
        image_analysis: Optional[Dict[str, Any]],
        response_style: str
    ) -> Dict[str, Any]:
        """리랭킹을 사용한 처리"""
        logger.info("Processing with reranking...")
        
        # 1. 초기 검색
        search_results = self.vector_db.search_multimodal(
            query=query,
            image_analysis=image_analysis,
            k=20  # 리랭킹을 위해 많이 검색
        )
        
        # 2. 쿼리 분석
        query_analysis = self._analyze_query(query, image_analysis)
        
        # 3. 리랭킹
        reranked_results = self.reranker.rerank(
            query=query,
            documents=search_results,
            query_analysis=query_analysis,
            strategy='weighted_fusion',
            top_k=5
        )
        
        # 4. 컨텍스트 구성
        context = self._build_enhanced_context(reranked_results, query_analysis)
        
        # 5. 응답 생성
        response = self._generate_enhanced_response(
            query, context, response_style, image_analysis
        )
        
        return {
            'success': True,
            'query': query,
            'response': response,
            'search_results': self._format_search_results(reranked_results),
            'query_analysis': query_analysis,
            'mode': 'reranking'
        }
    
    def _process_fast(
        self,
        query: str,
        image_analysis: Optional[Dict[str, Any]],
        response_style: str
    ) -> Dict[str, Any]:
        """빠른 처리 (기본 검색만)"""
        logger.info("Processing in fast mode...")
        
        # 1. 검색
        search_results = self.vector_db.search_multimodal(
            query=query,
            image_analysis=image_analysis,
            k=5
        )
        
        # 2. 간단한 컨텍스트
        context = {
            'retrieved_documents': [
                {'content': self._extract_text(doc), 'score': doc.get('hybrid_score', 0)}
                for doc in search_results
            ]
        }
        
        # 3. 빠른 응답 생성
        prompt = f"""질문: {query}

참고 자료:
{self._format_context_for_prompt(context)}

간단하고 정확한 답변을 제공해주세요."""
        
        response = self.llm_client.generate(prompt)
        
        return {
            'success': True,
            'query': query,
            'response': response,
            'mode': 'fast'
        }
    
    def _analyze_query(
        self,
        query: str,
        image_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """쿼리 분석"""
        analysis = {
            'query_type': self._determine_query_type(query),
            'key_concepts': self._extract_concepts(query),
            'formulas': [],
            'has_image': image_analysis is not None
        }
        
        # 이미지에서 추가 정보
        if image_analysis:
            if 'formulas' in image_analysis:
                analysis['formulas'].extend(image_analysis['formulas'])
            if 'key_concepts' in image_analysis:
                analysis['key_concepts'].extend(image_analysis['key_concepts'])
        
        return analysis
    
    def _determine_query_type(self, query: str) -> str:
        """쿼리 유형 결정"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['계산', '구하', '값은', '얼마']):
            return 'calculation'
        elif any(word in query_lower for word in ['설명', '무엇', '정의', '의미']):
            return 'explanation'
        elif any(word in query_lower for word in ['차이', '비교', '다른', 'vs']):
            return 'comparison'
        elif any(word in query_lower for word in ['방법', '어떻게', '절차', '과정']):
            return 'method'
        elif any(word in query_lower for word in ['이유', '왜', '원인', '때문']):
            return 'reason'
        else:
            return 'general'
    
    def _extract_concepts(self, query: str) -> List[str]:
        """개념 추출"""
        concepts = []
        ee_keywords = [
            '전압', '전류', '저항', '전력', '임피던스', '역률',
            '변압기', '모터', '전동기', '회로', '콘덴서', '인덕터',
            '3상', '단상', 'RLC', 'PWM', '인버터', '정류기'
        ]
        
        query_lower = query.lower()
        for keyword in ee_keywords:
            if keyword.lower() in query_lower:
                concepts.append(keyword)
        
        return list(set(concepts))
    
    def _build_enhanced_context(
        self,
        reranked_results: List[Dict[str, Any]],
        query_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """향상된 컨텍스트 구성"""
        context = {
            'retrieved_documents': [],
            'key_concepts': query_analysis['key_concepts'],
            'formulas': query_analysis['formulas'],
            'query_type': query_analysis['query_type'],
            'evidence_strength': 0.0
        }
        
        # 리랭킹된 문서 처리
        total_score = 0
        for doc in reranked_results:
            doc_info = {
                'content': self._extract_text(doc),
                'score': doc.get('final_rerank_score', doc.get('rerank_score', 0)),
                'score_details': doc.get('score_details', {})
            }
            
            # 문서에서 추가 정보 추출
            if 'parsed_content' in doc:
                if 'formulas' in doc['parsed_content']:
                    context['formulas'].extend(doc['parsed_content']['formulas'])
            
            context['retrieved_documents'].append(doc_info)
            total_score += doc_info['score']
        
        # 증거 강도 계산
        if reranked_results:
            context['evidence_strength'] = total_score / len(reranked_results)
        
        return context
    
    def _generate_enhanced_response(
        self,
        query: str,
        context: Dict[str, Any],
        response_style: str,
        image_analysis: Optional[Dict[str, Any]]
    ) -> str:
        """향상된 응답 생성"""
        # ChatGPT 스타일 응답 생성
        chatgpt_context = context.copy()
        if image_analysis:
            chatgpt_context['image_analysis'] = image_analysis
        
        response = self.response_generator.generate_response(
            question=query,
            context=chatgpt_context,
            response_type=response_style
        )
        
        # 증거 강도에 따른 신뢰도 표시
        confidence_level = self._get_confidence_level(context['evidence_strength'])
        response += f"\n\n📊 **신뢰도**: {confidence_level}"
        
        # 리랭킹 점수 정보
        if context['retrieved_documents']:
            response += f"\n💡 **검색 품질**: {context['evidence_strength']:.2f}/1.0"
        
        return response
    
    def _get_confidence_level(self, score: float) -> str:
        """신뢰도 레벨 결정"""
        if score >= 0.8:
            return "매우 높음 ⭐⭐⭐⭐⭐"
        elif score >= 0.6:
            return "높음 ⭐⭐⭐⭐"
        elif score >= 0.4:
            return "보통 ⭐⭐⭐"
        elif score >= 0.2:
            return "낮음 ⭐⭐"
        else:
            return "매우 낮음 ⭐"
    
    def _format_evidence(self, evidence_list) -> List[Dict[str, Any]]:
        """증거 포맷팅"""
        formatted = []
        for evidence in evidence_list[:5]:  # 상위 5개
            formatted.append({
                'content': evidence.content[:200] + "...",
                'type': evidence.evidence_type,
                'relevance': evidence.relevance_score
            })
        return formatted
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """검색 결과 포맷팅"""
        formatted = []
        for result in results:
            formatted.append({
                'content': self._extract_text(result)[:200] + "...",
                'score': result.get('final_rerank_score', result.get('rerank_score', 0)),
                'score_details': result.get('score_details', {})
            })
        return formatted
    
    def _extract_text(self, doc: Dict[str, Any]) -> str:
        """문서에서 텍스트 추출"""
        if 'documents' in doc and doc['documents']:
            return doc['documents'][0]
        elif 'content' in doc:
            return doc['content']
        elif 'text' in doc:
            return doc['text']
        else:
            return str(doc)
    
    def _format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """프롬프트용 컨텍스트 포맷팅"""
        formatted = ""
        for i, doc in enumerate(context.get('retrieved_documents', [])[:3]):
            formatted += f"\n[문서 {i+1}] (점수: {doc.get('score', 0):.3f})\n"
            formatted += f"{doc['content'][:300]}...\n"
        return formatted


def create_advanced_rag_system(config, llm_client) -> AdvancedRAGSystem:
    """고급 RAG 시스템 생성"""
    # 향상된 벡터 DB
    vector_db = EnhancedVectorDatabase(
        persist_directory=config.rag.persist_directory
    )
    
    # 고급 RAG 시스템
    return AdvancedRAGSystem(
        vector_db=vector_db,
        llm_client=llm_client
    )


if __name__ == "__main__":
    # 테스트
    from config import Config
    from llm_client import LLMClient
    
    config = Config()
    llm_client = LLMClient(config.llm)
    
    # 시스템 생성
    rag_system = create_advanced_rag_system(config, llm_client)
    
    # 테스트 쿼리
    test_query = "3상 전력 시스템에서 선간전압이 380V이고 역률이 0.8일 때 전력을 계산하는 방법은?"
    
    # 추론 모드로 처리
    result = rag_system.process_query_advanced(
        query=test_query,
        mode='reasoning'
    )
    
    if result['success']:
        print(f"응답:\n{result['response']}")
        print(f"\n처리 시간: {result['processing_time']:.2f}초")
        print(f"처리 모드: {result['mode']}")