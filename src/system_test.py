#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
차세대 AI 챗봇 시스템 검증 테스트
Next-Generation AI Chatbot System Validation Test

전체 시스템 기능 테스트 및 성능 검증
Full System Functionality Test and Performance Validation
"""

import os
import sys
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# 테스트 대상 시스템들
try:
    from integrated_service import IntegratedAIService, ServiceConfig, ServiceRequest
    from advanced_ai_system import AdvancedAISystem, ReasoningType
    from multimodal_pipeline import MultimodalPipeline, ProcessingMode
    from query_intent_analyzer import QueryIntentAnalyzer, QueryType
    from openai_vision_client import OpenAIVisionClient
    from ncp_ocr_client import NCPOCRClient
    from enhanced_rag_system import EnhancedRAGSystem
    from enhanced_llm_system import EnhancedLLMSystem
except ImportError as e:
    print(f"시스템 임포트 실패: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemTestSuite:
    """시스템 테스트 스위트"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = None
        self.end_time = None
        
        # 테스트 질문들
        self.test_queries = [
            {
                'id': 'basic_electrical',
                'query': '전압과 전류의 관계를 설명해주세요.',
                'expected_keywords': ['옴의 법칙', '전압', '전류', '저항'],
                'complexity': 'simple'
            },
            {
                'id': 'calculation_query',
                'query': '12V 전압과 4Ω 저항이 있을 때 전류를 계산해주세요.',
                'expected_keywords': ['3A', '3암페어', '계산', 'I=V/R'],
                'complexity': 'medium'
            },
            {
                'id': 'complex_reasoning',
                'query': '병렬 회로에서 각 브랜치의 저항이 다를 때 전류 분배 원리를 설명하고 예시를 들어주세요.',
                'expected_keywords': ['병렬', '키르히호프', '전류분배', '역비례'],
                'complexity': 'complex'
            },
            {
                'id': 'domain_knowledge',
                'query': '3상 교류 시스템에서 상전압과 선전압의 관계는?',
                'expected_keywords': ['3상', '상전압', '선전압', '√3'],
                'complexity': 'domain_specific'
            }
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        self.start_time = datetime.now()
        logger.info("=== 차세대 AI 챗봇 시스템 검증 테스트 시작 ===")
        
        # 컴포넌트 테스트
        await self._test_individual_components()
        
        # 통합 시스템 테스트  
        await self._test_integrated_system()
        
        # 성능 테스트
        await self._test_performance()
        
        # 오류 처리 테스트
        await self._test_error_handling()
        
        self.end_time = datetime.now()
        
        # 결과 분석
        return self._analyze_results()
    
    async def _test_individual_components(self):
        """개별 컴포넌트 테스트"""
        logger.info("🔧 개별 컴포넌트 테스트 시작")
        
        # 1. 질의 의도 분석기 테스트
        await self._test_query_analyzer()
        
        # 2. ChatGPT Vision 클라이언트 테스트
        await self._test_vision_client()
        
        # 3. NCP OCR 클라이언트 테스트
        await self._test_ocr_client()
        
        # 4. 향상된 RAG 시스템 테스트
        await self._test_rag_system()
        
        # 5. 향상된 LLM 시스템 테스트
        await self._test_llm_system()
    
    async def _test_query_analyzer(self):
        """질의 의도 분석기 테스트"""
        try:
            analyzer = QueryIntentAnalyzer()
            
            for test_case in self.test_queries:
                result = analyzer.analyze_query(test_case['query'])
                
                self.test_results.append({
                    'component': 'QueryIntentAnalyzer',
                    'test_case': test_case['id'],
                    'success': result.confidence > 0.5,
                    'details': {
                        'query_type': result.query_type.value,
                        'complexity': result.complexity.value,
                        'confidence': result.confidence,
                        'domain': result.domain
                    }
                })
            
            logger.info("✅ 질의 의도 분석기 테스트 완료")
            
        except Exception as e:
            logger.error(f"❌ 질의 의도 분석기 테스트 실패: {e}")
            self.test_results.append({
                'component': 'QueryIntentAnalyzer',
                'test_case': 'initialization',
                'success': False,
                'error': str(e)
            })
    
    async def _test_vision_client(self):
        """ChatGPT Vision 클라이언트 테스트"""
        try:
            client = OpenAIVisionClient()
            
            # API 가용성 테스트
            available = client.api_available
            
            self.test_results.append({
                'component': 'OpenAIVisionClient',
                'test_case': 'availability',
                'success': available,
                'details': {
                    'api_available': available,
                    'model': client.config.get('vision_model', 'unknown')
                }
            })
            
            logger.info(f"✅ ChatGPT Vision 클라이언트 테스트 완료 (사용 가능: {available})")
            
        except Exception as e:
            logger.error(f"❌ ChatGPT Vision 클라이언트 테스트 실패: {e}")
            self.test_results.append({
                'component': 'OpenAIVisionClient',
                'test_case': 'initialization',
                'success': False,
                'error': str(e)
            })
    
    async def _test_ocr_client(self):
        """NCP OCR 클라이언트 테스트"""
        try:
            client = NCPOCRClient()
            
            # API 가용성 테스트
            available = client.api_available
            connection_test = client.test_connection()
            
            self.test_results.append({
                'component': 'NCPOCRClient',
                'test_case': 'availability',
                'success': available,
                'details': {
                    'api_available': available,
                    'connection_test': connection_test
                }
            })
            
            logger.info(f"✅ NCP OCR 클라이언트 테스트 완료 (사용 가능: {available})")
            
        except Exception as e:
            logger.error(f"❌ NCP OCR 클라이언트 테스트 실패: {e}")
            self.test_results.append({
                'component': 'NCPOCRClient',
                'test_case': 'initialization',
                'success': False,
                'error': str(e)
            })
    
    async def _test_rag_system(self):
        """향상된 RAG 시스템 테스트"""
        try:
            rag_system = EnhancedRAGSystem()
            
            # 검색 테스트
            for test_case in self.test_queries[:2]:  # 처음 2개만 테스트
                results = rag_system.search(test_case['query'])
                
                self.test_results.append({
                    'component': 'EnhancedRAGSystem',
                    'test_case': test_case['id'],
                    'success': len(results) > 0,
                    'details': {
                        'results_count': len(results),
                        'top_score': results[0]['score'] if results else 0.0
                    }
                })
            
            logger.info("✅ 향상된 RAG 시스템 테스트 완료")
            
        except Exception as e:
            logger.error(f"❌ 향상된 RAG 시스템 테스트 실패: {e}")
            self.test_results.append({
                'component': 'EnhancedRAGSystem',
                'test_case': 'initialization',
                'success': False,
                'error': str(e)
            })
    
    async def _test_llm_system(self):
        """향상된 LLM 시스템 테스트"""
        try:
            # 기본 설정으로 시스템 생성 시도
            from enhanced_llm_system import create_enhanced_llm_system
            
            try:
                llm_system = create_enhanced_llm_system()
                health_status = llm_system.check_system_health()
                
                self.test_results.append({
                    'component': 'EnhancedLLMSystem',
                    'test_case': 'health_check',
                    'success': any(health_status.values()),
                    'details': health_status
                })
                
                logger.info("✅ 향상된 LLM 시스템 테스트 완료")
                
            except Exception as e:
                logger.warning(f"LLM 시스템 초기화 실패 (정상적일 수 있음): {e}")
                self.test_results.append({
                    'component': 'EnhancedLLMSystem',
                    'test_case': 'initialization',
                    'success': False,
                    'error': str(e),
                    'note': 'LLM 서버가 실행 중이지 않을 수 있음'
                })
                
        except Exception as e:
            logger.error(f"❌ 향상된 LLM 시스템 테스트 실패: {e}")
            self.test_results.append({
                'component': 'EnhancedLLMSystem',
                'test_case': 'import',
                'success': False,
                'error': str(e)
            })
    
    async def _test_integrated_system(self):
        """통합 시스템 테스트"""
        logger.info("🚀 통합 시스템 테스트 시작")
        
        try:
            # 서비스 설정
            config = ServiceConfig(
                service_mode="hybrid",
                enable_openai_vision=True,
                enable_ncp_ocr=True,
                enable_rag=True,
                enable_fine_tuned_llm=False,  # LLM 서버가 없을 수 있으므로
                enable_reasoning=True,
                enable_memory=True,
                log_detailed_processing=True
            )
            
            # 통합 서비스 초기화
            service = IntegratedAIService(config)
            
            # 시스템 상태 확인
            status = service.get_service_status()
            
            self.test_results.append({
                'component': 'IntegratedAIService',
                'test_case': 'initialization',
                'success': status['status'] == 'ready',
                'details': {
                    'status': status['status'],
                    'available_systems': status['available_systems'],
                    'capabilities': status['capabilities']
                }
            })
            
            # 실제 쿼리 테스트
            for test_case in self.test_queries[:2]:  # 처음 2개 테스트
                request = ServiceRequest(
                    request_id=f"test_{test_case['id']}",
                    query=test_case['query']
                )
                
                start_time = time.time()
                response = await service.process_request(request)
                processing_time = time.time() - start_time
                
                # 키워드 체크
                keywords_found = sum(1 for keyword in test_case['expected_keywords'] 
                                   if keyword.lower() in response.response.lower())
                keyword_score = keywords_found / len(test_case['expected_keywords'])
                
                self.test_results.append({
                    'component': 'IntegratedAIService',
                    'test_case': f"query_{test_case['id']}",
                    'success': response.success and response.confidence_score >= 0.5,
                    'details': {
                        'success': response.success,
                        'confidence_score': response.confidence_score,
                        'processing_time': processing_time,
                        'response_length': len(response.response),
                        'keyword_score': keyword_score,
                        'processing_system': response.metadata.get('processing_system') if response.metadata else None
                    }
                })
            
            logger.info("✅ 통합 시스템 테스트 완료")
            
        except Exception as e:
            logger.error(f"❌ 통합 시스템 테스트 실패: {e}")
            self.test_results.append({
                'component': 'IntegratedAIService',
                'test_case': 'initialization',
                'success': False,
                'error': str(e)
            })
    
    async def _test_performance(self):
        """성능 테스트"""
        logger.info("⚡ 성능 테스트 시작")
        
        try:
            # 간단한 성능 테스트
            from integrated_service import process_query_with_service
            
            test_query = "전압과 전류의 관계는?"
            
            # 여러 번 실행하여 평균 시간 측정
            times = []
            for i in range(3):
                start_time = time.time()
                try:
                    response = await process_query_with_service(test_query)
                    processing_time = time.time() - start_time
                    times.append(processing_time)
                except Exception as e:
                    logger.warning(f"성능 테스트 {i+1}번째 실행 실패: {e}")
            
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                
                self.test_results.append({
                    'component': 'Performance',
                    'test_case': 'response_time',
                    'success': avg_time < 30.0,  # 30초 이내
                    'details': {
                        'average_time': avg_time,
                        'max_time': max_time,
                        'min_time': min_time,
                        'runs': len(times)
                    }
                })
            
            logger.info("✅ 성능 테스트 완료")
            
        except Exception as e:
            logger.error(f"❌ 성능 테스트 실패: {e}")
            self.test_results.append({
                'component': 'Performance',
                'test_case': 'response_time',
                'success': False,
                'error': str(e)
            })
    
    async def _test_error_handling(self):
        """오류 처리 테스트"""
        logger.info("🛡️ 오류 처리 테스트 시작")
        
        try:
            from integrated_service import IntegratedAIService, ServiceConfig, ServiceRequest
            
            config = ServiceConfig(service_mode="basic")
            service = IntegratedAIService(config)
            
            # 빈 쿼리 테스트
            empty_request = ServiceRequest(
                request_id="test_empty",
                query=""
            )
            
            response = await service.process_request(empty_request)
            
            self.test_results.append({
                'component': 'ErrorHandling',
                'test_case': 'empty_query',
                'success': not response.success,  # 실패해야 정상
                'details': {
                    'response': response.response[:100],
                    'error_handled': response.error_message is not None
                }
            })
            
            logger.info("✅ 오류 처리 테스트 완료")
            
        except Exception as e:
            logger.error(f"❌ 오류 처리 테스트 실패: {e}")
            self.test_results.append({
                'component': 'ErrorHandling',
                'test_case': 'empty_query',
                'success': False,
                'error': str(e)
            })
    
    def _analyze_results(self) -> Dict[str, Any]:
        """결과 분석"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - successful_tests
        
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # 컴포넌트별 성공률
        components = {}
        for result in self.test_results:
            component = result['component']
            if component not in components:
                components[component] = {'total': 0, 'success': 0}
            
            components[component]['total'] += 1
            if result['success']:
                components[component]['success'] += 1
        
        component_success_rates = {}
        for comp, stats in components.items():
            component_success_rates[comp] = (stats['success'] / stats['total']) * 100
        
        total_time = (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0
        
        return {
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate,
                'total_time': total_time
            },
            'component_success_rates': component_success_rates,
            'detailed_results': self.test_results,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        # 컴포넌트별 분석
        failed_components = []
        for result in self.test_results:
            if not result['success']:
                failed_components.append(result['component'])
        
        if 'OpenAIVisionClient' in failed_components:
            recommendations.append("OpenAI API 키를 확인하고 Vision API 접근 권한을 검토하세요.")
        
        if 'NCPOCRClient' in failed_components:
            recommendations.append("NCP OCR API 키를 확인하고 서비스 설정을 검토하세요.")
        
        if 'EnhancedLLMSystem' in failed_components:
            recommendations.append("LLM 서버가 실행 중인지 확인하거나 대체 LLM 서비스를 설정하세요.")
        
        if 'EnhancedRAGSystem' in failed_components:
            recommendations.append("문서 데이터베이스와 임베딩 모델 설정을 확인하세요.")
        
        # 성능 권장사항
        performance_results = [r for r in self.test_results if r['component'] == 'Performance']
        if performance_results and not performance_results[0]['success']:
            recommendations.append("시스템 성능이 기대치에 못 미칩니다. 하드웨어 리소스를 확인하세요.")
        
        if not recommendations:
            recommendations.append("모든 시스템이 정상적으로 작동하고 있습니다.")
        
        return recommendations


def print_test_report(results: Dict[str, Any]):
    """테스트 결과 출력"""
    print("\n" + "="*80)
    print("🤖 차세대 AI 챗봇 시스템 검증 결과 리포트")
    print("="*80)
    
    summary = results['summary']
    
    print(f"\n📊 전체 테스트 요약:")
    print(f"   총 테스트 수: {summary['total_tests']}")
    print(f"   성공한 테스트: {summary['successful_tests']}")
    print(f"   실패한 테스트: {summary['failed_tests']}")
    print(f"   성공률: {summary['success_rate']:.1f}%")
    print(f"   총 소요 시간: {summary['total_time']:.2f}초")
    
    print(f"\n🔧 컴포넌트별 성공률:")
    for component, success_rate in results['component_success_rates'].items():
        status = "✅" if success_rate == 100 else "⚠️" if success_rate >= 50 else "❌"
        print(f"   {status} {component}: {success_rate:.1f}%")
    
    print(f"\n💡 권장사항:")
    for rec in results['recommendations']:
        print(f"   • {rec}")
    
    print(f"\n📝 상세 결과:")
    for result in results['detailed_results']:
        status = "✅" if result['success'] else "❌"
        component = result['component']
        test_case = result['test_case']
        print(f"   {status} {component}.{test_case}")
        
        if 'details' in result:
            details = result['details']
            if isinstance(details, dict):
                for key, value in details.items():
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            print(f"      {key}: {value:.3f}")
                        else:
                            print(f"      {key}: {value}")
                    else:
                        print(f"      {key}: {value}")
        
        if 'error' in result:
            print(f"      오류: {result['error']}")
        
        if 'note' in result:
            print(f"      참고: {result['note']}")
    
    print("\n" + "="*80)


async def main():
    """메인 테스트 함수"""
    print("🚀 차세대 AI 챗봇 시스템 검증 테스트를 시작합니다...")
    
    test_suite = SystemTestSuite()
    results = await test_suite.run_all_tests()
    
    print_test_report(results)
    
    # 결과를 파일로 저장
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    filename = f"logs/{timestamp}_system_test_result.json"
    
    os.makedirs("logs", exist_ok=True)
    
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n💾 상세 결과가 {filename}에 저장되었습니다.")
    
    # 성공률에 따른 종료 코드
    success_rate = results['summary']['success_rate']
    if success_rate >= 80:
        print("\n🎉 시스템이 성공적으로 검증되었습니다!")
        return 0
    elif success_rate >= 50:
        print("\n⚠️ 시스템에 일부 문제가 있지만 기본 기능은 작동합니다.")
        return 1
    else:
        print("\n❌ 시스템에 심각한 문제가 있습니다. 설정을 확인해주세요.")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())