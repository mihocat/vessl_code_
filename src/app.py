#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Application for RAG System
RAG 시스템 메인 애플리케이션
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


class RAGService:
    """RAG 서비스 통합 클래스"""
    
    def __init__(self):
        # RAG 시스템 초기화
        self.rag_system = ConcreteKoreanElectricalRAG(embedding_model_name="jinaai/jina-embeddings-v3")
        self.llm_client = LLMClient()
        
        # 데이터 로드
        self.rag_system.load_documents_from_dataset()
        
        logger.info("RAGService 초기화 완료")
    
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
        
        # 향상된 통합 검색 파이프라인 실행
        search_results, search_type = self.rag_system.enhanced_search_pipeline(question)
        
        response_parts = []
        
        # 검색 결과 유형별 처리
        if search_type == "high_confidence_db":
            # 고신뢰도 DB 결과 - 직접 답변 제공
            best_result = search_results[0]
            doc_info = best_result["doc_info"]
            
            response_parts.append("**전문 지식베이스 답변:**\n")
            response_parts.append(doc_info["answer"])
            response_parts.append(f"\n*[분류: {doc_info['category']} | 신뢰도: {best_result['final_score']:.2f}]*")
            self.rag_system.service_stats["successful_answers"] += 1
            
        elif search_type == "medium_confidence_db":
            # 중간 신뢰도 DB 결과 - LLM으로 재구성
            context_parts = []
            for result in search_results[:2]:
                doc_info = result["doc_info"]
                context_parts.append(f"참고자료 {len(context_parts)+1}: {doc_info['answer']}")
            
            context = "\n---\n".join(context_parts)
            if user_context:
                context = f"이전 대화:\n{user_context}\n---\n{context}"
            
            answer = self.llm_client.query(question, context)
            response_parts.append(answer)
            response_parts.append(f"\n*[참고: 지식베이스 {len(search_results)}건 종합]*")
            self.rag_system.service_stats["successful_answers"] += 1
            
        elif search_type == "hybrid_search":
            # DB + 웹 검색 하이브리드
            db_results, web_results = search_results
            
            # DB 컨텍스트 구성
            db_context = []
            for result in db_results[:2]:
                doc_info = result["doc_info"]
                db_context.append(f"전문자료: {doc_info['answer']}")
            
            # 웹 컨텍스트 구성
            web_context = []
            for result in web_results[:2]:
                web_context.append(f"웹자료: {result['title']} - {result['snippet'][:150]}")
            
            # 통합 컨텍스트
            full_context = "\n".join(db_context + web_context)
            if user_context:
                full_context = f"이전 대화: {user_context}\n---\n{full_context}"
            
            answer = self.llm_client.query(question, full_context)
            response_parts.append(answer)
            response_parts.append(f"\n*[통합검색: DB {len(db_results)}건 + 웹 {len(web_results)}건]*")
            self.rag_system.service_stats["successful_answers"] += 1
            
        elif search_type == "web_only":
            # 웹 검색만 사용
            web_context = []
            for result in search_results[:2]:
                web_context.append(f"{result['title']}: {result['snippet']}")
            
            context = "\n---\n".join(web_context)
            if user_context:
                context = f"이전 대화: {user_context}\n---\n{context}"
            
            answer = self.llm_client.query(question, context)
            response_parts.append(answer)
            response_parts.append(f"\n*[웹검색 기반: {len(search_results)}건 참조]*")
            self.rag_system.service_stats["successful_answers"] += 1
            
        elif search_type == "low_confidence_db":
            # 저품질 DB 결과라도 시도
            if search_results:
                context_parts = [result["doc_info"]["answer"] for result in search_results[:2]]
                context = " | ".join(context_parts)
                
                answer = self.llm_client.query(question, context)
                response_parts.append(answer)
                response_parts.append("\n*[참고: 부분적 일치 정보 활용]*")
            else:
                response_parts.append(self._handle_non_electrical_query(question))
                
        else:
            # 검색 결과 없음
            response_parts.append(self._handle_non_electrical_query(question))
        
        # 추가 정보
        response_parts.append("\n\n---")
        response_parts.append(f"응답시간: {round(time.time() - start_time, 2)}초")
        
        # 스마트 관련 질문 추천
        if search_type in ["high_confidence_db", "medium_confidence_db", "hybrid_search"]:
            db_results = search_results if search_type != "hybrid_search" else search_results[0]
            response_parts.append("\n**관련 질문 추천:**")
            categories = set()
            for result in db_results[:3]:
                if result["doc_info"] and result["doc_info"]["category"] not in categories:
                    categories.add(result["doc_info"]["category"])
                    response_parts.append(f"• {result['doc_info']['category']} 분야 심화 학습")
        
        response_parts.append("\n언제든지 질문해주세요.")
        
        # 사용자 이력 저장
        full_response = "\n".join(response_parts)
        self.rag_system.user_history[user_id].append(question[:50])
        if len(self.rag_system.user_history[user_id]) > 10:
            self.rag_system.user_history[user_id].pop(0)
        
        return full_response
    
    def _get_welcome_message(self) -> str:
        """환영 메시지"""
        return """🔌 **전기공학 전문 AI 상담사**입니다.

**주요 서비스:**
• 전기공학 이론 및 실무 상담
• 전기기사/산업기사 시험 지도  
• 회로 해석 및 설계 조언
• 전력시스템 관련 문의

무엇을 도와드릴까요? 구체적인 질문을 해주시면 정확한 답변을 드리겠습니다."""
    
    def _handle_non_electrical_query(self, query: str) -> str:
        """전기공학 외 질문 지능형 처리"""
        # 일반 인사말이나 간단한 질문
        greetings = ["안녕", "반가워", "고마워", "감사"]
        if any(greeting in query for greeting in greetings):
            return "안녕하세요! 전기공학 관련 질문이 있으시면 언제든 도움드리겠습니다."
        
        # 도움말 요청
        if any(word in query for word in ["도움", "사용법", "어떻게"]):
            return """전기공학 전문 상담 서비스입니다.

**질문 예시:**
• 옴의 법칙이 무엇인가요?
• 변압기의 동작 원리를 설명해주세요
• 전기기사 시험 준비 방법은?
• AC와 DC의 차이점은?

**/통계** 명령으로 서비스 현황을 확인할 수 있습니다."""
        
        return "죄송합니다. 전기공학 관련 질문에만 답변드릴 수 있습니다. 전기, 전력, 회로, 자격증 등에 대해 질문해 주세요."


def create_gradio_interface(service: RAGService):
    """Gradio 인터페이스 생성"""
    
    def handle_query(message, history):
        """질의 처리"""
        if message.startswith("/통계"):
            return service.rag_system.get_service_statistics()
        elif message.startswith("/도움") or message.startswith("/help"):
            return service._get_welcome_message()
        else:
            user_id = f"user_{len(history) % 100}"
            return service.process_query(message, user_id)
    
    demo = gr.ChatInterface(
        fn=handle_query,
        title="⚡ 전기공학 전문 AI 상담사",
        description="**전기공학 이론부터 실무까지, 정확한 전문 답변을 제공합니다**\n\n• 전기기사/산업기사 시험 대비 • 회로 해석 및 설계 • 전력시스템 상담 • 전기기기 원리 설명",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"),
        chatbot=gr.Chatbot(
            height=600, 
            show_copy_button=True, 
            type="messages",
            avatar_images=(None, "⚡")
        ),
        textbox=gr.Textbox(
            placeholder="전기공학 관련 질문을 입력하세요... (예: 옴의 법칙, 변압기 원리, 전기기사 시험)", 
            container=False, 
            scale=7
        ),
        type="messages",
        examples=[
            "옴의 법칙에 대해 설명해주세요",
            "변압기의 동작 원리가 궁금합니다", 
            "전기기사 시험 준비 방법을 알려주세요",
            "AC와 DC의 차이점은 무엇인가요?",
            "키르히호프 법칙을 쉽게 설명해주세요"
        ]
    )
    
    return demo


def main():
    """메인 함수"""
    logger.info("=== RAG Service 시작 ===")
    
    try:
        # 서비스 초기화
        service = RAGService()
        logger.info("RAG 서비스 초기화 완료")
        
        # vLLM 서버 연결 필수 확인
        logger.info("vLLM 서버 연결 대기 중...")
        if not service.llm_client.check_health():
            # 서버 대기 로직 (wait_for_server 메서드가 없으므로 직접 구현)
            import time
            max_attempts = 30
            for i in range(max_attempts):
                if service.llm_client.check_health():
                    logger.info("vLLM 서버 연결 성공")
                    break
                time.sleep(3)
                if i % 5 == 4:
                    logger.info(f"vLLM 서버 대기 중... ({i+1}/{max_attempts})")
            else:
                logger.error("vLLM 서버 연결 실패 - 서비스를 시작할 수 없습니다")
                raise RuntimeError("vLLM 서버에 연결할 수 없습니다")
        else:
            logger.info("vLLM 서버 연결 성공")
        
        # Gradio 인터페이스 실행
        app = create_gradio_interface(service)
        logger.info("Gradio 인터페이스 시작...")
        
        port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
        logger.info(f"서버 포트: {port}")
        logger.info("=== 전기공학 전문 RAG 시스템 준비 완료 ===")
        
        app.launch(
            server_name="0.0.0.0", 
            server_port=port, 
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=False,
            prevent_thread_lock=False,
            favicon_path=None,
            ssl_verify=False,
            allowed_paths=[],
            app_kwargs={"docs_url": None, "redoc_url": None}
        )
        
    except Exception as e:
        logger.error(f"서비스 시작 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()