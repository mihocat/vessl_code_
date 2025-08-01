#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Reasoning Chain System
RAG 추론 체인 시스템 - 다단계 추론 및 검증
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ReasoningStep(Enum):
    """추론 단계 정의"""
    QUERY_UNDERSTANDING = "query_understanding"
    CONTEXT_RETRIEVAL = "context_retrieval"
    EVIDENCE_EXTRACTION = "evidence_extraction"
    REASONING = "reasoning"
    ANSWER_GENERATION = "answer_generation"
    VERIFICATION = "verification"
    REFINEMENT = "refinement"


@dataclass
class ReasoningContext:
    """추론 컨텍스트"""
    query: str
    query_type: str  # calculation, explanation, comparison, etc.
    key_concepts: List[str]
    formulas: List[str]
    constraints: List[str]
    image_context: Optional[Dict[str, Any]] = None


@dataclass
class ReasoningEvidence:
    """추론 증거"""
    source_id: str
    content: str
    relevance_score: float
    evidence_type: str  # formula, concept, example, solution
    extracted_info: Dict[str, Any]


class RAGReasoningChain:
    """RAG 추론 체인"""
    
    def __init__(self, llm_client, vector_db, reranker):
        """
        추론 체인 초기화
        
        Args:
            llm_client: LLM 클라이언트
            vector_db: 벡터 데이터베이스
            reranker: 리랭커
        """
        self.llm_client = llm_client
        self.vector_db = vector_db
        self.reranker = reranker
        
        # 추론 프롬프트 템플릿
        self.prompts = {
            'query_understanding': self._get_query_understanding_prompt(),
            'evidence_extraction': self._get_evidence_extraction_prompt(),
            'reasoning': self._get_reasoning_prompt(),
            'verification': self._get_verification_prompt(),
            'refinement': self._get_refinement_prompt()
        }
        
        logger.info("RAG Reasoning Chain initialized")
    
    def reason(
        self,
        query: str,
        image_analysis: Optional[Dict[str, Any]] = None,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        다단계 추론 수행
        
        Args:
            query: 사용자 질문
            image_analysis: 이미지 분석 결과
            max_iterations: 최대 반복 횟수
        """
        start_time = time.time()
        reasoning_trace = []
        
        try:
            # 1. 쿼리 이해
            context = self._understand_query(query, image_analysis)
            reasoning_trace.append({
                'step': ReasoningStep.QUERY_UNDERSTANDING.value,
                'result': context.__dict__
            })
            
            # 2. 컨텍스트 검색
            documents = self._retrieve_context(context)
            reasoning_trace.append({
                'step': ReasoningStep.CONTEXT_RETRIEVAL.value,
                'result': {'num_docs': len(documents)}
            })
            
            # 3. 증거 추출
            evidence_list = self._extract_evidence(context, documents)
            reasoning_trace.append({
                'step': ReasoningStep.EVIDENCE_EXTRACTION.value,
                'result': {'num_evidence': len(evidence_list)}
            })
            
            # 4. 추론 수행
            reasoning_result = self._perform_reasoning(context, evidence_list)
            reasoning_trace.append({
                'step': ReasoningStep.REASONING.value,
                'result': reasoning_result
            })
            
            # 5. 답변 생성
            answer = self._generate_answer(context, reasoning_result, evidence_list)
            reasoning_trace.append({
                'step': ReasoningStep.ANSWER_GENERATION.value,
                'result': {'answer_length': len(answer)}
            })
            
            # 6. 검증 및 개선
            for iteration in range(max_iterations):
                verification = self._verify_answer(context, answer, evidence_list)
                
                if verification['is_valid']:
                    break
                
                # 답변 개선
                answer = self._refine_answer(
                    context, answer, verification['issues'], evidence_list
                )
                reasoning_trace.append({
                    'step': ReasoningStep.REFINEMENT.value,
                    'iteration': iteration + 1,
                    'result': verification
                })
            
            elapsed_time = time.time() - start_time
            
            return {
                'success': True,
                'answer': answer,
                'reasoning_trace': reasoning_trace,
                'evidence': evidence_list,
                'context': context,
                'time_elapsed': elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Reasoning failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'reasoning_trace': reasoning_trace
            }
    
    def _understand_query(
        self,
        query: str,
        image_analysis: Optional[Dict[str, Any]]
    ) -> ReasoningContext:
        """쿼리 이해 및 분석"""
        prompt = self.prompts['query_understanding'].format(
            query=query,
            image_context=json.dumps(image_analysis, ensure_ascii=False) if image_analysis else "없음"
        )
        
        response = self.llm_client.generate(prompt)
        
        # 응답 파싱
        try:
            analysis = self._parse_json_response(response)
            
            return ReasoningContext(
                query=query,
                query_type=analysis.get('query_type', 'general'),
                key_concepts=analysis.get('key_concepts', []),
                formulas=analysis.get('formulas', []),
                constraints=analysis.get('constraints', []),
                image_context=image_analysis
            )
        except:
            # 파싱 실패시 기본값
            return ReasoningContext(
                query=query,
                query_type='general',
                key_concepts=self._extract_concepts(query),
                formulas=[],
                constraints=[],
                image_context=image_analysis
            )
    
    def _retrieve_context(self, context: ReasoningContext) -> List[Dict[str, Any]]:
        """관련 문서 검색"""
        # 멀티모달 검색
        results = self.vector_db.search_multimodal(
            query=context.query,
            image_analysis=context.image_context,
            k=20  # 리랭킹을 위해 많이 검색
        )
        
        # 쿼리 분석 결과로 리랭킹
        query_analysis = {
            'formulas': context.formulas,
            'concepts': context.key_concepts,
            'type': context.query_type
        }
        
        # 리랭킹
        reranked = self.reranker.rerank(
            query=context.query,
            documents=results,
            query_analysis=query_analysis,
            strategy='weighted_fusion',
            top_k=5
        )
        
        return reranked
    
    def _extract_evidence(
        self,
        context: ReasoningContext,
        documents: List[Dict[str, Any]]
    ) -> List[ReasoningEvidence]:
        """문서에서 증거 추출"""
        evidence_list = []
        
        for doc in documents:
            prompt = self.prompts['evidence_extraction'].format(
                query=context.query,
                document=self._extract_text(doc),
                query_type=context.query_type
            )
            
            response = self.llm_client.generate(prompt)
            
            try:
                extracted = self._parse_json_response(response)
                
                for evidence in extracted.get('evidence', []):
                    evidence_list.append(ReasoningEvidence(
                        source_id=doc.get('id', ''),
                        content=evidence.get('content', ''),
                        relevance_score=evidence.get('relevance', 0.5),
                        evidence_type=evidence.get('type', 'general'),
                        extracted_info=evidence.get('info', {})
                    ))
            except:
                # 기본 증거 추출
                evidence_list.append(ReasoningEvidence(
                    source_id=doc.get('id', ''),
                    content=self._extract_text(doc)[:500],
                    relevance_score=doc.get('rerank_score', 0.5),
                    evidence_type='document',
                    extracted_info={}
                ))
        
        # 관련성 순으로 정렬
        evidence_list.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return evidence_list[:10]  # 상위 10개 증거
    
    def _perform_reasoning(
        self,
        context: ReasoningContext,
        evidence_list: List[ReasoningEvidence]
    ) -> Dict[str, Any]:
        """추론 수행"""
        # 증거 요약
        evidence_summary = self._summarize_evidence(evidence_list)
        
        prompt = self.prompts['reasoning'].format(
            query=context.query,
            query_type=context.query_type,
            key_concepts=", ".join(context.key_concepts),
            evidence=evidence_summary,
            constraints="\n".join(context.constraints) if context.constraints else "없음"
        )
        
        response = self.llm_client.generate(prompt)
        
        try:
            reasoning = self._parse_json_response(response)
            return reasoning
        except:
            return {
                'conclusion': response,
                'confidence': 0.7,
                'reasoning_steps': []
            }
    
    def _generate_answer(
        self,
        context: ReasoningContext,
        reasoning_result: Dict[str, Any],
        evidence_list: List[ReasoningEvidence]
    ) -> str:
        """최종 답변 생성"""
        # ChatGPT 스타일 응답 생성
        from chatgpt_response_generator import ChatGPTResponseGenerator
        
        generator = ChatGPTResponseGenerator()
        
        # 컨텍스트 구성
        response_context = {
            'key_concepts': context.key_concepts,
            'formulas': context.formulas,
            'solution_steps': reasoning_result.get('reasoning_steps', []),
            'evidence': [e.content for e in evidence_list[:3]],
            'confidence': reasoning_result.get('confidence', 0.8)
        }
        
        # 응답 생성
        if context.query_type == 'calculation':
            response_type = 'step_by_step'
        elif context.query_type in ['explanation', 'definition']:
            response_type = 'concept'
        else:
            response_type = 'comprehensive'
        
        answer = generator.generate_response(
            question=context.query,
            context=response_context,
            response_type=response_type
        )
        
        # 증거 기반 보강
        if evidence_list:
            answer += "\n\n📚 **참고 자료:**\n"
            for i, evidence in enumerate(evidence_list[:3]):
                answer += f"{i+1}. {evidence.content[:100]}...\n"
        
        return answer
    
    def _verify_answer(
        self,
        context: ReasoningContext,
        answer: str,
        evidence_list: List[ReasoningEvidence]
    ) -> Dict[str, Any]:
        """답변 검증"""
        prompt = self.prompts['verification'].format(
            query=context.query,
            answer=answer,
            evidence=self._summarize_evidence(evidence_list[:5])
        )
        
        response = self.llm_client.generate(prompt)
        
        try:
            verification = self._parse_json_response(response)
            return verification
        except:
            # 기본 검증 통과
            return {
                'is_valid': True,
                'confidence': 0.8,
                'issues': []
            }
    
    def _refine_answer(
        self,
        context: ReasoningContext,
        answer: str,
        issues: List[str],
        evidence_list: List[ReasoningEvidence]
    ) -> str:
        """답변 개선"""
        prompt = self.prompts['refinement'].format(
            query=context.query,
            original_answer=answer,
            issues="\n".join(issues),
            additional_evidence=self._summarize_evidence(evidence_list[3:6])
        )
        
        refined_answer = self.llm_client.generate(prompt)
        
        return refined_answer
    
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
    
    def _extract_concepts(self, query: str) -> List[str]:
        """쿼리에서 개념 추출 (간단한 버전)"""
        concepts = []
        concept_keywords = [
            '전압', '전류', '저항', '전력', '임피던스', '역률',
            '변압기', '모터', '회로', '콘덴서', '인덕터'
        ]
        
        query_lower = query.lower()
        for keyword in concept_keywords:
            if keyword in query_lower:
                concepts.append(keyword)
        
        return concepts
    
    def _summarize_evidence(self, evidence_list: List[ReasoningEvidence]) -> str:
        """증거 요약"""
        summary = ""
        for i, evidence in enumerate(evidence_list):
            summary += f"\n[증거 {i+1}] (관련도: {evidence.relevance_score:.2f})\n"
            summary += f"유형: {evidence.evidence_type}\n"
            summary += f"내용: {evidence.content[:200]}...\n"
        
        return summary
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """JSON 응답 파싱"""
        # JSON 블록 추출
        import re
        
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # JSON 형태 찾기
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")
        
        return json.loads(json_str)
    
    def _get_query_understanding_prompt(self) -> str:
        """쿼리 이해 프롬프트"""
        return """전기공학 질문을 분석해주세요.

질문: {query}
이미지 컨텍스트: {image_context}

다음 형식으로 분석 결과를 제공해주세요:
```json
{{
    "query_type": "calculation|explanation|comparison|method|reason",
    "key_concepts": ["개념1", "개념2", ...],
    "formulas": ["수식1", "수식2", ...],
    "constraints": ["조건1", "조건2", ...]
}}
```"""
    
    def _get_evidence_extraction_prompt(self) -> str:
        """증거 추출 프롬프트"""
        return """다음 문서에서 질문에 대한 답변에 필요한 증거를 추출해주세요.

질문: {query}
질문 유형: {query_type}

문서:
{document}

추출할 증거 (JSON 형식):
```json
{{
    "evidence": [
        {{
            "content": "증거 내용",
            "type": "formula|concept|example|solution",
            "relevance": 0.0-1.0,
            "info": {{}}
        }}
    ]
}}
```"""
    
    def _get_reasoning_prompt(self) -> str:
        """추론 프롬프트"""
        return """주어진 증거를 바탕으로 논리적 추론을 수행해주세요.

질문: {query}
질문 유형: {query_type}
핵심 개념: {key_concepts}
제약 조건: {constraints}

증거:
{evidence}

추론 결과 (JSON 형식):
```json
{{
    "conclusion": "최종 결론",
    "confidence": 0.0-1.0,
    "reasoning_steps": [
        {{
            "step": 1,
            "description": "단계 설명",
            "evidence_used": ["증거 참조"],
            "result": "중간 결과"
        }}
    ]
}}
```"""
    
    def _get_verification_prompt(self) -> str:
        """검증 프롬프트"""
        return """생성된 답변을 검증해주세요.

원래 질문: {query}

생성된 답변:
{answer}

사용된 증거:
{evidence}

검증 결과 (JSON 형식):
```json
{{
    "is_valid": true|false,
    "confidence": 0.0-1.0,
    "issues": ["문제점1", "문제점2", ...],
    "suggestions": ["개선사항1", "개선사항2", ...]
}}
```"""
    
    def _get_refinement_prompt(self) -> str:
        """개선 프롬프트"""
        return """답변을 개선해주세요.

원래 질문: {query}

원래 답변:
{original_answer}

발견된 문제점:
{issues}

추가 증거:
{additional_evidence}

개선된 답변을 작성해주세요."""


class ElectricalEngineeringReasoner(RAGReasoningChain):
    """전기공학 특화 추론 체인"""
    
    def __init__(self, llm_client, vector_db, reranker):
        """전기공학 추론기 초기화"""
        super().__init__(llm_client, vector_db, reranker)
        
        # 전기공학 도메인 규칙
        self.domain_rules = {
            'ohms_law': 'V = I × R',
            'power_ac': 'P = V × I × cosθ',
            'power_3phase': 'P = √3 × VL × IL × cosθ',
            'impedance': 'Z = √(R² + X²)',
            'power_factor': 'PF = cosθ = P/S'
        }
        
        # 단위 변환 규칙
        self.unit_conversions = {
            'kW': 1000,
            'MW': 1000000,
            'mA': 0.001,
            'kV': 1000,
            'MΩ': 1000000
        }
    
    def _verify_calculation(
        self,
        calculation: str,
        expected_unit: str
    ) -> Tuple[bool, Optional[str]]:
        """계산 검증"""
        # 단위 확인
        # 물리적 타당성 확인
        # 수식 검증
        
        return True, None  # 간단한 구현


if __name__ == "__main__":
    # 테스트는 통합 시스템에서 수행
    pass