#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Application for Korean Electrical Engineering RAG System
한국어 전기공학 RAG 시스템 메인 애플리케이션
"""

import os
import sys
import time
import logging
import gradio as gr
from datetime import datetime

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import ConcreteKoreanElectricalRAG
from llm_client import LLMClient

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElectricalRAGService:
    """전기공학 RAG 서비스 통합 클래스"""
    
    def __init__(self):
        # RAG 시스템 초기화
        self.rag_system = ConcreteKoreanElectricalRAG()
        self.llm_client = LLMClient()
        
        # 데이터 로드
        self.rag_system.load_documents_from_dataset()
        
        logger.info("ElectricalRAGService 초기화 완료")
    
    def process_query(self, question: str, user_id: str = "default") -> str:
        """사용자 질의 처리"""
        if not question.strip():
            return self._get_welcome_message()
        
        start_time = time.time()
        self.rag_system.service_stats["total_queries"] += 1
        
        # 사용자 이력 확인
        user_context = ""
        if user_id in self.rag_system.user_history and self.rag_system.user_history[user_id]:
            recent_history = self.rag_system.user_history[user_id][-3:]
            user_context = "이전 대화: " + " | ".join(recent_history)
        
        # 1. 지능형 DB 검색
        db_results, found_in_db = self.rag_system.search_vector_database(question, k=4)
        
        # 2. 전기공학 관련성 체크
        is_electrical = self.rag_system.check_electrical_relevance(question)
        
        response_parts = []
        response_parts.append("🔌 **전기공학 통합 서비스 답변**\n")
        
        if found_in_db and db_results:
            # DB에서 우수한 결과 발견
            best_result = db_results[0]
            if best_result["final_score"] > 0.8:
                # 매우 정확한 매칭
                doc_info = best_result["doc_info"]
                if doc_info:
                    response_parts.append("💡 직접 답변:\n")
                    response_parts.append(doc_info["answer"])
                    response_parts.append(f"\n[분류: {doc_info['category']}]")
            else:
                # 컨텍스트 기반 답변 생성
                context_parts = [result["content"] for result in db_results[:3]]
                context = " | ".join(context_parts)
                
                if user_context:
                    context = user_context + " | " + context
                
                answer = self.llm_client.query(question, context)
                response_parts.append(answer)
                response_parts.append(f"\n📚 참고: 지식베이스 {len(db_results)}건")
                self.rag_system.service_stats["successful_answers"] += 1
            
            # 추가 웹 정보 제공 (최신 정보 요청시)
            if "최신" in question or "현재" in question:
                web_results = self.rag_system.search_web(question, max_results=2)
                if web_results:
                    response_parts.append("\n\n🌐 최신 정보:")
                    for result in web_results[:1]:
                        response_parts.append(f"• {result['title']}")
                        response_parts.append(f"  {result['snippet'][:100]}...")
        
        elif is_electrical:
            # 전기공학 관련이지만 DB에 없음
            response_parts.append("데이터베이스에 정확한 정보가 없어 웹과 전문 지식을 활용합니다.\n")
            
            # 웹 검색
            web_results = self.rag_system.search_web(question, max_results=3)
            web_context = ""
            if web_results:
                web_parts = [f"{r['title']}: {r['snippet']}" for r in web_results]
                web_context = " | ".join(web_parts)
            
            # 전문가 답변 생성
            full_context = web_context
            if user_context:
                full_context = user_context + " | " + full_context
            
            answer = self.llm_client.query(question, full_context)
            response_parts.append(answer)
            self.rag_system.service_stats["successful_answers"] += 1
            
            if web_results:
                response_parts.append("\n\n🔍 참고 출처:")
                for result in web_results[:2]:
                    response_parts.append(f"• {result['title'][:50]}")
        else:
            # 전기공학과 무관한 질문
            response_parts.append(self._handle_non_electrical_query(question))
        
        # 추가 정보
        response_parts.append("\n\n---")
        response_parts.append(f"⏱️ 응답시간: {round(time.time() - start_time, 2)}초")
        
        # 관련 질문 추천
        if found_in_db and db_results:
            response_parts.append("\n💬 관련 질문 추천:")
            categories = set()
            for result in db_results[:3]:
                if result["doc_info"] and result["doc_info"]["category"] not in categories:
                    categories.add(result["doc_info"]["category"])
                    response_parts.append(f"• {result['doc_info']['category']} 관련 더 알아보기")
        
        response_parts.append("\n📞 추가 문의: 언제든지 질문해주세요!")
        
        # 사용자 이력 저장
        full_response = "\n".join(response_parts)
        self.rag_system.user_history[user_id].append(question[:50])
        if len(self.rag_system.user_history[user_id]) > 10:
            self.rag_system.user_history[user_id].pop(0)
        
        return full_response
    
    def _get_welcome_message(self) -> str:
        """환영 메시지"""
        return """🔌 **전기공학 통합 서비스에 오신 것을 환영합니다!**

저는 한국어 전기공학 전문 AI 상담사입니다.
다음과 같은 서비스를 제공합니다:

📚 **지식 기반 답변**: 3000개 전문 문서 데이터베이스
🌐 **최신 정보 검색**: 실시간 웹 정보 통합
🤖 **AI 전문가 상담**: 파인튜닝된 한국어 모델
📊 **맞춤형 서비스**: 사용자 이력 기반 상담

무엇을 도와드릴까요?"""
    
    def _handle_non_electrical_query(self, query: str) -> str:
        """비전기공학 질문 처리"""
        return """죄송합니다. 이 질문은 전기공학 분야가 아닌 것 같습니다.

저는 전기공학 전문 서비스로서 다음과 같은 분야를 도와드릴 수 있습니다:

✅ **기본 이론**: 옴의 법칙, 키르히호프 법칙, 전자기학
✅ **전기 기기**: 변압기, 전동기, 발전기
✅ **전력 공학**: 송배전, 전력계통, 안정도
✅ **제어 공학**: PID 제어, 자동제어
✅ **전기 설비**: 배선, 접지, 보호계전
✅ **자격증**: 전기기사, 전기산업기사 시험 준비

전기공학 관련 질문을 해주시면 전문적으로 도와드리겠습니다!"""


def create_gradio_interface(service: ElectricalRAGService):
    """Gradio 인터페이스 생성"""
    
    def handle_query(message, history):
        """질의 처리"""
        if message.startswith("/통계"):
            return service.rag_system.get_service_statistics()
        elif message.startswith("/도움"):
            return service._get_welcome_message()
        else:
            # 간단한 사용자 ID 생성
            user_id = f"user_{len(history) % 100}"
            return service.process_query(message, user_id)
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔌 전기공학 통합 서비스")
        gr.Markdown("### LLM(파인튜닝) + RAG(DB+웹검색) 완성형 AI 서비스")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    elem_id="chatbot"
                )
                msg = gr.Textbox(
                    label="질문 입력",
                    placeholder="전기공학 관련 질문을 입력하세요...",
                    lines=2
                )
                with gr.Row():
                    submit = gr.Button("전송", variant="primary")
                    clear = gr.Button("대화 초기화")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📌 주요 기능")
                gr.Markdown("""
                - **지능형 검색**: 의미+카테고리 기반
                - **실시간 웹검색**: 최신 정보 통합
                - **전문가 AI**: 파인튜닝 KoLlama
                - **사용자 맞춤**: 대화 이력 활용
                """)
                
                gr.Markdown("### 💡 예시 질문")
                examples = gr.Examples(
                    examples=[
                        "옴의 법칙을 쉽게 설명해주세요",
                        "변압기의 원리와 종류를 알려주세요",
                        "전기기사 시험 준비는 어떻게 하나요?",
                        "최신 전력계통 기술 동향은?",
                        "15-1-6 Spot 네트워크 방식이란?",
                        "/통계",
                        "/도움"
                    ],
                    inputs=msg
                )
        
        def user_submit(message, history):
            history = history or []
            response = handle_query(message, history)
            history.append([message, response])
            return "", history
        
        msg.submit(user_submit, [msg, chatbot], [msg, chatbot])
        submit.click(user_submit, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
        
        gr.Markdown("---")
        gr.Markdown("© 2025 Korean Electrical Engineering AI Service | Powered by KoLlama + ChromaDB + DuckDuckGo")
    
    return demo


def main():
    """메인 함수"""
    logger.info("=== Korean Electrical Engineering RAG Service 시작 ===")
    
    # 서비스 초기화
    service = ElectricalRAGService()
    
    # vLLM 서버 대기
    service.llm_client.wait_for_server()
    
    # Gradio 인터페이스 실행
    app = create_gradio_interface(service)
    logger.info("Gradio 인터페이스 시작...")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()