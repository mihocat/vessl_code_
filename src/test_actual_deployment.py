#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실제 배포된 VESSL 서비스 테스트
"""

import requests
import json
import time
from datetime import datetime

def test_gradio_api(base_url, message, history=[]):
    """Gradio API를 통한 실제 테스트"""
    
    # Gradio ChatInterface API endpoint
    api_url = f"{base_url}/run/predict"
    
    data = {
        "data": [
            message,  # 현재 메시지
            history   # 대화 히스토리
        ],
        "fn_index": 0  # ChatInterface의 기본 함수 인덱스
    }
    
    headers = {
        "Content-Type": "application/json",
    }
    
    try:
        start_time = time.time()
        response = requests.post(api_url, json=data, headers=headers, timeout=60)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            # Gradio ChatInterface는 [response, updated_history] 형태로 반환
            chat_response = result.get("data", [None, []])[0]
            return {
                "success": True,
                "response": chat_response,
                "elapsed_time": elapsed,
                "status_code": response.status_code
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "elapsed_time": elapsed,
                "status_code": response.status_code
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
            "status_code": None
        }

def main():
    # VESSL Gradio 엔드포인트
    gradio_url = "https://run-execution-x3loqiv27br1-run-exec-7860.sanjose.oracle-cluster.vessl.ai"
    
    # EXAMPLE.md 기반 테스트 질문
    test_questions = [
        {
            "id": 1,
            "question": "다산에듀는 무엇인가요?",
            "expected": "미호가 다니는 회사입니다!"
        },
        {
            "id": 2,
            "question": "R-C회로 합성 임피던스에서 -j를 붙이는 이유는?",
            "expected": "커패시터의 용량성 리액턴스 Xc는 전류가 전압보다 90도 앞서기 때문"
        },
        {
            "id": 3,
            "question": "과도현상과 인덕턴스 L의 관계는?",
            "expected": "인덕터는 전류가 갑자기 바뀌는 걸 싫어합니다"
        },
        {
            "id": 4,
            "question": "댐 부속설비 중 수로와 여수로의 차이는?",
            "expected": "수로는 여수로를 의미하는 경우가 많습니다"
        },
        {
            "id": 5,
            "question": "서보모터의 동작 원리는?",
            "expected": "오일 피스톤에 의해 동작"
        }
    ]
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"=== VESSL 배포 시스템 실제 테스트 ===")
    print(f"시간: {timestamp}")
    print(f"Gradio URL: {gradio_url}")
    print("=" * 60)
    
    results = []
    
    for test in test_questions:
        print(f"\n테스트 {test['id']}: {test['question']}")
        print(f"예상 답변 힌트: {test['expected'][:30]}...")
        
        result = test_gradio_api(gradio_url, test["question"])
        
        if result["success"]:
            print(f"✅ 응답 성공 ({result['elapsed_time']:.2f}초)")
            print(f"\n실제 응답:")
            print("-" * 40)
            print(result['response'])
            print("-" * 40)
        else:
            print(f"❌ 오류 발생: {result['error']}")
        
        # 결과 저장
        test_result = {
            "test_id": test["id"],
            "question": test["question"],
            "expected_hint": test["expected"],
            "actual_response": result.get("response", ""),
            "success": result["success"],
            "elapsed_time": result["elapsed_time"],
            "error": result.get("error", None)
        }
        results.append(test_result)
        
        # 요청 간 대기
        time.sleep(3)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 요약:")
    success_count = sum(1 for r in results if r["success"])
    print(f"- 총 테스트: {len(test_questions)}개")
    print(f"- 성공: {success_count}개")
    print(f"- 실패: {len(test_questions) - success_count}개")
    
    if success_count > 0:
        avg_time = sum(r['elapsed_time'] for r in results if r['success']) / success_count
        print(f"- 평균 응답 시간: {avg_time:.2f}초")
    
    return results

if __name__ == "__main__":
    results = main()