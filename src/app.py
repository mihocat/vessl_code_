#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 Gradio UI 애플리케이션
"""

import sys
import os

import gradio as gr
import time
import logging
from typing import List, Tuple, Optional, Dict
from llm_client import LLMClient
from rag_system import ImprovedRAGSystem
from ddgs import DDGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedRAGService:
    def __init__(self, llm_client: LLMClient):
        """개선된 RAG 서비스 초기화"""
        self.llm_client = llm_client
        self.rag_system = ImprovedRAGSystem(
            embedding_model_name="jinaai/jina-embeddings-v3",
            llm_client=llm_client
        )
        self.conversation_history = []
        
    def process_query(self, question: str, history: List[Tuple[str, str]]) -> str:
        """사용자 질의 처리"""
        start_time = time.time()
        
        # 빈 질문 처리
        if not question or not question.strip():
            return "질문을 입력해주세요."
        
        # 모든 질문에 대해 RAG 검색 수행
        results, max_score = self.rag_system.search(question, k=5)
        
        # 응답 생성
        response = self._generate_response(question, results, max_score)
        
        # 응답 시간 추가
        elapsed_time = time.time() - start_time
        response += f"\n\n응답시간: {elapsed_time:.2f}초"
        
        # 대화 이력 업데이트
        self.conversation_history.append((question, response))
        
        return response
    
    def _is_technical_query(self, query: str) -> bool:
        """기술/전문 분야 관련 질문인지 확인 (범용)"""
        # 모든 질문을 기술 질문으로 간주하여 범용성 확보
        return True
    
    def _generate_response(self, question: str, results: List[Dict], max_score: float) -> str:
        """검색 결과를 기반으로 응답 생성"""
        
        # 점수 기반 응답 전략
        response = f"[질문] {question}\n\n"
        
        if max_score >= 0.8:  # 매우 높은 신뢰도만 직접 답변
            # 높은 신뢰도 - 직접 답변
            best_result = results[0]
            response += f"{best_result['answer']}\n"
            
            if best_result.get('category'):
                response += f"\n[분류: {best_result['category']}]"
            
        elif max_score >= 0.6:  # 중간 신뢰도
            # 중간 신뢰도 - LLM이 RAG 결과를 재구성
            context_parts = []
            for i, result in enumerate(results[:3]):
                if result['score'] >= 0.4:
                    # 답변 길이를 200자로 줄여 토큰 제한 문제 해결
                    context_parts.append(f"참고 {i+1}: {result['answer'][:200]}...")
            
            if context_parts:
                context = "\n\n".join(context_parts)
                # 프롬프트 간소화로 토큰 사용량 감소
                prompt = f"""질문: {question}

참고자료:
{context}

위 질문에 대해 참고자료를 활용하여 답변하세요.
- 2-3문단으로 간결하게
- 전문용어는 정확히, 수식은 텍스트로
- 한국어로 자연스럽게"""
                
                try:
                    llm_response = self.llm_client.query(question, prompt)
                    response += llm_response
                except Exception as e:
                    logger.error(f"LLM 응답 생성 실패: {e}")
                    response += "죄송합니다. 응답 생성 중 오류가 발생했습니다."
            else:
                # _handle_low_confidence_query를 사용하되 중복 질문 제거
                try:
                    prompt = f"""질문: {question}

위 질문에 대해 2-3문단으로 간결하게 답변하세요."""
                    llm_response = self.llm_client.query(question, prompt)
                    response += llm_response
                except Exception as e:
                    logger.error(f"LLM 응답 생성 실패: {e}")
                    response += "죄송합니다. 응답 생성 중 오류가 발생했습니다."
            
        else:
            # 낮은 신뢰도 - 웹 검색 + LLM
            # DB에서 찾은 내용
            db_context = []
            for result in results[:2]:
                if result['score'] >= 0.2:
                    db_context.append(result['answer'])
            
            # 웹 검색
            web_results = self._search_web(question)
            web_context = []
            for web in web_results[:3]:
                web_context.append(f"{web['title']}: {web['snippet']}")
            
            # 통합 프롬프트 - 컨텍스트 길이 제한
            db_context_limited = [ctx[:150] + "..." for ctx in db_context[:2]]
            web_context_limited = [ctx[:100] + "..." for ctx in web_context[:2]]
            all_context = "\n".join(db_context_limited + web_context_limited) if (db_context_limited or web_context_limited) else ""
            
            if all_context:
                prompt = f"""질문: {question}

참고: {all_context[:500]}

위 질문에 대해 간결하게 답변하세요."""
                try:
                    llm_response = self.llm_client.query(question, prompt)
                    response += llm_response
                except Exception as e:
                    logger.error(f"LLM 응답 생성 실패: {e}")
                    response += "죄송합니다. 응답 생성 중 오류가 발생했습니다."
            else:
                response = self._handle_low_confidence_query(question)
        
        # 관련 질문 추천 (높은 신뢰도일 때만)
        if max_score >= 0.7 and len(results) > 1:
            response += "\n\n**관련 질문들:**"
            seen_questions = {question.lower()}
            count = 0
            for result in results[1:]:
                if result['score'] >= 0.6 and result['question'].lower() not in seen_questions and count < 3:
                    response += f"\n- {result['question']}"
                    seen_questions.add(result['question'].lower())
                    count += 1
        
        return response
    
    def _search_web(self, query: str) -> List[Dict]:
        """웹 검색"""
        try:
            with DDGS() as ddgs:
                results = []
                # 전기공학 제한 제거 - 일반 검색
                search_query = query
                
                for r in ddgs.text(search_query, max_results=3):
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", "")[:200],
                        "url": r.get("href", "")
                    })
                
                return results
        except Exception as e:
            logger.error(f"웹 검색 실패: {e}")
            return []
    
    # 전기공학 외 질문 처리 메서드 제거 - 모든 질문을 동일하게 처리
    
    def _handle_low_confidence_query(self, question: str) -> str:
        """낮은 신뢰도 질문 처리"""
        # 프롬프트 간소화
        prompt = f"""질문: {question}

위 질문에 대해 2-3문단으로 간결하게 답변하세요."""
        
        try:
            llm_response = self.llm_client.query(question, prompt)
            return f"[질문] {question}\n\n{llm_response}"
        except Exception as e:
            logger.error(f"낮은 신뢰도 응답 생성 실패: {e}")
            return f"[질문] {question}\n\n죄송합니다. 해당 질문에 대한 정확한 답변을 찾을 수 없습니다."


def create_gradio_app(llm_url: str = "http://localhost:8000") -> gr.Blocks:
    """Gradio 앱 생성"""
    # LLM 클라이언트 초기화
    llm_client = LLMClient(base_url=llm_url)
    
    # RAG 서비스 초기화
    rag_service = ImprovedRAGService(llm_client)
    
    # Gradio 인터페이스
    with gr.Blocks(title="AI 챗봇") as app:
        gr.Markdown(
            """
            # AI 챗봇
            
            **개선사항:**
            - jinaai/jina-embeddings-v3 임베딩 모델 (한국어 처리 1위)
            - 향상된 검색 정확도
            - 특수 키워드 처리
            """
        )
        
        # 채팅 인터페이스
        chatbot = gr.Chatbot(
            label="대화",
            height=500,
            bubble_full_width=False
        )
        
        # 입력창
        msg = gr.Textbox(
            label="질문을 입력하세요",
            placeholder="예: 다산에듀는 무엇인가요?",
            lines=2
        )
        
        # 버튼들
        with gr.Row():
            submit = gr.Button("전송", variant="primary")
            clear = gr.Button("대화 초기화")
        
        # 예제 질문들
        gr.Examples(
            examples=[
                "다산에듀는 무엇인가요?",
                "R-C회로 합성 임피던스에서 -j를 붙이는 이유는?",
                "과도현상과 인덕턴스 L의 관계는?",
                "댐 부속설비 중 수로와 여수로의 차이는?",
                "서보모터의 동작 원리는?"
            ],
            inputs=msg,
            label="예제 질문"
        )
        
        # 이벤트 핸들러
        def respond(message, chat_history):
            response = rag_service.process_query(message, chat_history)
            chat_history.append((message, response))
            return "", chat_history
        
        # 이벤트 바인딩
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: (None, ""), None, [chatbot, msg])
        
        # 통계 표시
        with gr.Accordion("서비스 통계", open=False):
            stats_display = gr.Textbox(
                label="통계",
                interactive=False,
                lines=5
            )
            
            def update_stats():
                stats = rag_service.rag_system.service_stats
                return (
                    f"총 쿼리: {stats['total_queries']}\n"
                    f"DB 히트: {stats['db_hits']}\n"
                    f"웹 검색: {stats['web_searches']}\n"
                    f"LLM 응답: {stats['llm_responses']}\n"
                    f"평균 응답시간: {stats['avg_response_time']:.2f}초"
                )
            
            refresh_stats = gr.Button("통계 새로고침")
            refresh_stats.click(update_stats, None, stats_display)
    
    return app


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )