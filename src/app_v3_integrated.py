#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 파이프라인 Gradio 앱 v3
OpenAI 분석(1회) → RAG → 파인튜닝 LLM 파이프라인
"""

import sys
import os
import time
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import gradio as gr

from config import Config
from integrated_pipeline import IntegratedPipeline

# 상세한 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 앱 시작 시간 기록
APP_START_TIME = datetime.now()


class IntegratedChatService:
    """통합 파이프라인 채팅 서비스"""
    
    def __init__(self, config: Config):
        """서비스 초기화"""
        init_start_time = time.time()
        logger.info("🚀 [INIT] IntegratedChatService 초기화 시작")
        
        logger.info("🔧 [INIT-1] 설정 객체 저장 중...")
        self.config = config
        
        logger.info("🔧 [INIT-2] 통합 파이프라인 생성 중...")
        pipeline_start = time.time()
        self.pipeline = IntegratedPipeline(config)
        pipeline_time = time.time() - pipeline_start
        logger.info(f"✅ [INIT-2] 통합 파이프라인 생성 완료 ({pipeline_time:.2f}초)")
        
        logger.info("🔧 [INIT-3] 대화 이력 초기화 중...")
        self.conversation_history = []
        
        init_total_time = time.time() - init_start_time
        logger.info(f"✅ [INIT] IntegratedChatService 초기화 완료 (총 {init_total_time:.2f}초)")
    
    def process_query(
        self, 
        question: str, 
        history: List[Tuple[str, str]],
        image: Optional[Image.Image] = None
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        질의 처리 - 전체 플로우 로깅 포함
        
        Args:
            question: 사용자 질문
            history: 대화 이력
            image: 선택적 이미지
            
        Returns:
            (응답 메시지, 업데이트된 대화 이력)
        """
        query_start_time = time.time()
        query_id = int(time.time() * 1000) % 100000  # 5자리 쿼리 ID
        
        logger.info(f"\n🎯 ====== [QUERY-{query_id}] 사용자 질의 처리 시작 ======")
        logger.info(f"📝 [QUERY-{query_id}] 질문: {question[:100]}{'...' if len(question) > 100 else ''}")
        logger.info(f"🖼️ [QUERY-{query_id}] 이미지: {'있음' if image else '없음'}")
        logger.info(f"📚 [QUERY-{query_id}] 대화 이력: {len(history)}개")
        
        if not question.strip():
            logger.warning(f"⚠️ [QUERY-{query_id}] 빈 질문 입력됨")
            return "질문을 입력해주세요.", history
        
        logger.info(f"🔄 [QUERY-{query_id}] 통합 파이프라인 처리 시작")
        pipeline_start = time.time()
        
        # SKIP_VLLM 환경변수 확인
        skip_vllm = os.getenv("SKIP_VLLM", "false").lower() == "true"
        use_llm_flag = not skip_vllm
        
        # 파이프라인 처리 - 경로별 상세 로깅
        logger.info(f"📋 [QUERY-{query_id}] 파이프라인 입력 파라미터:")
        logger.info(f"   - use_rag: True")
        logger.info(f"   - use_llm: {use_llm_flag}")
        logger.info(f"   - SKIP_VLLM: {skip_vllm}")
        logger.info(f"   - 질문 길이: {len(question)}자")
        logger.info(f"   - 이미지: {'포함' if image else '없음'}")
        
        # 파이프라인 경로 확인 및 로깅
        if image is not None:
            logger.info(f"🖼️ [QUERY-{query_id}] 선택된 파이프라인: OpenAI → RAG → LLM (이미지 포함)")
        else:
            logger.info(f"📝 [QUERY-{query_id}] 선택된 파이프라인: RAG → LLM (텍스트 전용)")
        
        result = self.pipeline.process_query(
            question=question,
            image=image,
            use_rag=True,
            use_llm=use_llm_flag
        )
        
        # 파이프라인 결과 상세 로깅
        logger.info(f"📊 [QUERY-{query_id}] 파이프라인 결과 분석:")
        if result.success:
            logger.info(f"   ✅ 처리 성공")
            logger.info(f"   📝 최종 답변 길이: {len(result.final_answer)}자")
            
            # OpenAI 분석 결과 (이미지 포함 질의인 경우에만)
            if result.analysis_result:
                analysis = result.analysis_result
                logger.info(f"   🔍 OpenAI 분석: 성공")
                if analysis.get('token_usage'):
                    tokens = analysis['token_usage']
                    logger.info(f"     - 토큰: {tokens.get('total_tokens', 0)}개")
                if analysis.get('cost'):
                    logger.info(f"     - 비용: ${analysis['cost']:.4f}")
            else:
                # 텍스트 전용 질의인 경우
                logger.info(f"   📝 OpenAI 분석: 건너뜀 (텍스트 전용 질의)")
            
            # RAG 검색 결과
            if result.rag_results:
                logger.info(f"   📚 RAG 검색: {len(result.rag_results)}개 문서")
            
            # 처리 시간 분석
            if result.processing_times:
                times = result.processing_times
                logger.info(f"   ⏱️ 단계별 시간:")
                for step, duration in times.items():
                    if step != 'total':
                        logger.info(f"     - {step}: {duration:.2f}초")
        else:
            logger.error(f"   ❌ 처리 실패: {result.error_message}")
        
        pipeline_time = time.time() - pipeline_start
        logger.info(f"{'✅' if result.success else '❌'} [QUERY-{query_id}] 파이프라인 처리 완료 ({pipeline_time:.2f}초)")
        
        if result.success:
            logger.info(f"✅ [QUERY-{query_id}] 파이프라인 성공 - 응답 포맷팅 시작")
            format_start = time.time()
            response = self._format_response(result, question, image)
            format_time = time.time() - format_start
            logger.info(f"📝 [QUERY-{query_id}] 응답 포맷팅 완료 ({format_time:.3f}초)")
            logger.info(f"📄 [QUERY-{query_id}] 최종 응답 요약:")
            logger.info(f"   - 응답 총 길이: {len(response)}자")
            logger.info(f"   - 메인 답변: {result.final_answer[:100]}{'...' if len(result.final_answer) > 100 else ''}")
        else:
            logger.error(f"❌ [QUERY-{query_id}] 파이프라인 실패: {result.error_message}")
            response = f"처리 중 오류가 발생했습니다: {result.error_message}"
        
        # 대화 이력 업데이트
        logger.info(f"📚 [QUERY-{query_id}] 대화 이력 업데이트 중...")
        history.append((question, response))
        self.conversation_history.append({
            'query_id': query_id,
            'question': question,
            'response': response,
            'has_image': image is not None,
            'timestamp': time.time(),
            'pipeline_result': result,
            'processing_time': time.time() - query_start_time
        })
        
        total_time = time.time() - query_start_time
        logger.info(f"🏁 [QUERY-{query_id}] 전체 질의 처리 완료 (총 {total_time:.2f}초)")
        logger.info(f"📈 [QUERY-{query_id}] 성능 요약:")
        logger.info(f"   - 파이프라인: {pipeline_time:.2f}초 ({pipeline_time/total_time*100:.1f}%)")
        if result.success and result.processing_times:
            times = result.processing_times
            # OpenAI 분석 시간 (이미지 포함 질의인 경우에만)
            if 'openai_analysis' in times:
                openai_time = times['openai_analysis']
                logger.info(f"   - OpenAI 분석: {openai_time:.2f}초 ({openai_time/total_time*100:.1f}%)")
            # RAG 검색 시간
            if 'rag_search' in times:
                rag_time = times['rag_search']
                logger.info(f"   - RAG 검색: {rag_time:.2f}초 ({rag_time/total_time*100:.1f}%)")
            # LLM 생성 시간
            if 'llm_generation' in times:
                llm_time = times['llm_generation']
                logger.info(f"   - LLM 생성: {llm_time:.2f}초 ({llm_time/total_time*100:.1f}%)")
        logger.info(f"🎯 ====== [QUERY-{query_id}] 처리 종료 ======\n")
        
        return response, history
    
    def _format_response(self, result, question: str, image: Optional[Image.Image]) -> str:
        """응답 포맷팅"""
        response_parts = []
        
        # 메인 답변
        response_parts.append(result.final_answer)
        
        # 처리 정보 추가
        if result.analysis_result or result.processing_times:
            response_parts.append("\n" + "="*50)
            response_parts.append("📊 **처리 정보**")
            
            # OpenAI 분석 결과
            if result.analysis_result:
                analysis = result.analysis_result
                if analysis.get('formulas'):
                    response_parts.append(f"📐 감지된 수식: {len(analysis['formulas'])}개")
                if analysis.get('key_concepts'):
                    response_parts.append(f"🔑 핵심 개념: {', '.join(analysis['key_concepts'][:3])}")
                if analysis.get('token_usage'):
                    tokens = analysis['token_usage']
                    response_parts.append(f"🪙 토큰 사용: {tokens['total_tokens']}개 (입력: {tokens['prompt_tokens']}, 출력: {tokens['completion_tokens']})")
                if analysis.get('cost'):
                    response_parts.append(f"💰 OpenAI 비용: ${analysis['cost']:.4f}")
            
            # RAG 검색 결과
            if result.rag_results:
                response_parts.append(f"📚 RAG 검색: {len(result.rag_results)}개 문서 활용")
            
            # 처리 시간
            if result.processing_times:
                times = result.processing_times
                response_parts.append(f"⏱️ 처리 시간: {times.get('total', 0):.2f}초")
                if 'openai_analysis' in times:
                    response_parts.append(f"  - OpenAI 분석: {times['openai_analysis']:.2f}초")
                if 'rag_search' in times:
                    response_parts.append(f"  - RAG 검색: {times['rag_search']:.2f}초")
                if 'llm_generation' in times:
                    response_parts.append(f"  - LLM 생성: {times['llm_generation']:.2f}초")
            
            # 파이프라인 단계
            if result.pipeline_steps:
                response_parts.append(f"🔄 파이프라인: {' → '.join(result.pipeline_steps)}")
        
        return "\n".join(response_parts)
    
    def get_system_status(self) -> str:
        """시스템 상태 반환"""
        try:
            # 파이프라인 상태 확인
            health_status = self.pipeline.health_check()
            stats = self.pipeline.get_statistics()
            
            status_parts = [
                "🏥 **시스템 상태**",
                f"OpenAI 프로세서: {'✅' if health_status['openai_processor'] else '❌'}",
                f"RAG 시스템: {'✅' if health_status['rag_system'] else '❌'}",
                f"파인튜닝 LLM: {'✅' if health_status['llm_client'] else '❌'}",
                "",
                "📈 **처리 통계**",
                f"총 질의 수: {stats['total_queries']}",
                f"성공률: {stats['success_rate']:.1%}",
                f"평균 비용: ${stats['average_cost_per_query']:.4f}",
                f"OpenAI 호출 효율: {stats['openai_call_efficiency']}",
                f"총 비용: ${stats['total_cost']:.4f}"
            ]
            
            return "\n".join(status_parts)
            
        except Exception as e:
            return f"상태 확인 중 오류 발생: {e}"


def create_gradio_interface(service: IntegratedChatService) -> gr.Interface:
    """Gradio 인터페이스 생성"""
    
    def chat_function(question: str, history: List[Tuple[str, str]], image: Optional[Image.Image]):
        """채팅 함수"""
        return service.process_query(question, history, image)
    
    def status_function():
        """상태 확인 함수"""
        return service.get_system_status()
    
    # 채팅 인터페이스
    with gr.Blocks(
        title="🚀 통합 AI 파이프라인",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto;
        }
        .chat-container {
            height: 600px;
        }
        """
    ) as iface:
        gr.Markdown("""
        # 🚀 통합 AI 파이프라인 채팅봇
        
        **새로운 아키텍처:**
        1. 🔍 **OpenAI GPT-4.1** - 이미지+텍스트 분석 (1회 호출)
        2. 📚 **RAG 검색** - ChromaDB 전문 문서 활용
        3. 🤖 **파인튜닝 LLM** - KoLlama 한국어 전문 모델
        
        **특징:**
        - OpenAI API는 분석 전용 (최종 답변 금지)
        - 질의당 1회만 OpenAI 호출
        - 최종 답변은 파인튜닝된 한국어 LLM만 담당
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # 채팅 인터페이스
                chatbot = gr.Chatbot(
                    label="대화",
                    height=600,
                    show_label=True,
                    container=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        label="질문",
                        placeholder="질문을 입력하세요...",
                        lines=2,
                        scale=4
                    )
                    image_input = gr.Image(
                        label="이미지 (선택사항)",
                        type="pil",
                        scale=1
                    )
                
                with gr.Row():
                    submit_btn = gr.Button("전송", variant="primary", scale=1)
                    clear_btn = gr.Button("대화 초기화", scale=1)
            
            with gr.Column(scale=1):
                # 시스템 상태 패널
                status_output = gr.Textbox(
                    label="시스템 상태",
                    lines=20,
                    interactive=False
                )
                status_btn = gr.Button("상태 새로고침")
                
                # 예제 질문들
                gr.Markdown("### 💡 예제 질문")
                example_questions = [
                    "Pr'을 구할 때 왜 Pr에서 Qc를 빼는 건가요?",
                    "미분할 때 d/da를 적용하면 왜 s는 1이고 a는 0이 되는 건가요?",
                    "두 점 사이의 거리 단위벡터를 구하는 방식을 알려주세요",
                    "라플라스 변환의 정의와 성질을 설명해주세요"
                ]
                
                for i, example in enumerate(example_questions, 1):
                    gr.Button(f"{i}. {example[:30]}...", size="sm").click(
                        lambda x=example: (x, []),
                        outputs=[question_input, chatbot]
                    )
        
        # 이벤트 핸들러
        submit_btn.click(
            chat_function,
            inputs=[question_input, chatbot, image_input],
            outputs=[question_input, chatbot]
        ).then(
            lambda: ("", None),
            outputs=[question_input, image_input]
        )
        
        question_input.submit(
            chat_function,
            inputs=[question_input, chatbot, image_input],
            outputs=[question_input, chatbot]
        ).then(
            lambda: ("", None),
            outputs=[question_input, image_input]
        )
        
        clear_btn.click(
            lambda: ([], "", None),
            outputs=[chatbot, question_input, image_input]
        )
        
        status_btn.click(
            status_function,
            outputs=status_output
        )
        
        # 초기 상태 로드
        iface.load(
            status_function,
            outputs=status_output
        )
    
    return iface


def main():
    """메인 함수 - 앱 로딩 단계별 상세 로깅"""
    app_start_time = time.time()
    logger.info("\n🎉 ========================================")
    logger.info("🎉    통합 AI 파이프라인 앱 시작")
    logger.info("🎉 ========================================")
    
    try:
        # Step 1: 설정 로드
        logger.info("🔧 [STEP-1] 시스템 설정 로드 시작...")
        config_start = time.time()
        config = Config()
        config_time = time.time() - config_start
        logger.info(f"✅ [STEP-1] 시스템 설정 로드 완료 ({config_time:.2f}초)")
        logger.info(f"📋 [STEP-1] 설정 요약:")
        logger.info(f"   - OpenAI 모델: {getattr(config.openai, 'unified_model', 'N/A')}")
        logger.info(f"   - 서버 주소: {config.app.server_name}:{config.app.server_port}")
        logger.info(f"   - RAG 활성화: {hasattr(config, 'rag')}")
        
        # Step 2: 통합 서비스 초기화
        logger.info("🔧 [STEP-2] IntegratedChatService 초기화 시작...")
        service_start = time.time()
        service = IntegratedChatService(config)
        service_time = time.time() - service_start
        logger.info(f"✅ [STEP-2] IntegratedChatService 초기화 완료 ({service_time:.2f}초)")
        
        # Step 3: Gradio 인터페이스 생성
        logger.info("🔧 [STEP-3] Gradio 웹 인터페이스 생성 시작...")
        iface_start = time.time()
        iface = create_gradio_interface(service)
        iface_time = time.time() - iface_start
        logger.info(f"✅ [STEP-3] Gradio 인터페이스 생성 완료 ({iface_time:.2f}초)")
        
        # Step 4: 시스템 상태 확인
        logger.info("🔧 [STEP-4] 시스템 상태 최종 확인...")
        try:
            status_check = service.get_system_status()
            logger.info("✅ [STEP-4] 시스템 상태 확인 완료")
        except Exception as status_e:
            logger.warning(f"⚠️ [STEP-4] 시스템 상태 확인 실패: {status_e}")
        
        # 전체 초기화 시간 계산
        total_init_time = time.time() - app_start_time
        logger.info(f"\n🎊 전체 앱 초기화 완료!")
        logger.info(f"📊 초기화 시간 분석:")
        logger.info(f"   - 설정 로드: {config_time:.2f}초")
        logger.info(f"   - 서비스 초기화: {service_time:.2f}초")
        logger.info(f"   - 인터페이스 생성: {iface_time:.2f}초")
        logger.info(f"   - 전체 소요시간: {total_init_time:.2f}초")
        
        # Step 5: 서버 시작
        logger.info(f"\n🚀 [LAUNCH] 웹 서버 시작 중...")
        logger.info(f"🌐 [LAUNCH] 서버 주소: http://{config.app.server_name}:{config.app.server_port}")
        logger.info(f"🔗 [LAUNCH] 공유 링크: {'활성화' if config.app.share else '비활성화'}")
        logger.info(f"🎯 [LAUNCH] 서버 시작 후 질의 처리 로깅이 시작됩니다...")
        
        # Gradio 서버 런치
        iface.launch(
            server_name=config.app.server_name,
            server_port=config.app.server_port,
            share=config.app.share,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"❌ [ERROR] 앱 시작 실패: {e}")
        logger.error(f"❌ [ERROR] 오류 타입: {type(e).__name__}")
        logger.error(f"❌ [ERROR] 오류 상세: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()