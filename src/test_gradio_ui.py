#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio UI를 통한 실제 질문 테스트
"""

import requests
import json
import time
from datetime import datetime

def test_gradio_chat(base_url, message):
    """Gradio 채팅 API 테스트"""
    
    # Gradio API 엔드포인트
    api_url = f"{base_url}/run/predict"
    
    # 요청 데이터
    data = {
        "data": [
            message,  # 메시지
            []       # 히스토리
        ],
        "fn_index": 0
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
            return {
                "success": True,
                "response": result.get("data", ["응답 없음"])[0],
                "elapsed_time": elapsed
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "elapsed_time": elapsed
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }

def main():
    # VESSL 서비스 URL (포트 7860)
    service_url = "http://localhost:7860"
    
    # EXAMPLE.md 기반 테스트 질문
    test_questions = [
        "다산에듀는 무엇인가요?",
        "R-C회로 합성 임피던스에서 -j를 붙이는 이유는?",
        "과도현상과 인덕턴스 L의 관계는?",
        "댐 부속설비 중 수로와 여수로의 차이는?",
        "서보모터의 동작 원리는?"
    ]
    
    print(f"=== Gradio UI 실제 테스트 시작 ===")
    print(f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"URL: {service_url}")
    print("=" * 60)
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n테스트 {i}: {question}")
        
        result = test_gradio_chat(service_url, question)
        
        if result["success"]:
            print(f"✅ 응답 시간: {result['elapsed_time']:.2f}초")
            print(f"응답:\n{result['response']}")
        else:
            print(f"❌ 오류: {result['error']}")
        
        results.append({
            "question": question,
            "result": result
        })
        
        # 요청 간 대기
        time.sleep(2)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 요약:")
    success_count = sum(1 for r in results if r["result"]["success"])
    print(f"- 성공: {success_count}/{len(test_questions)}")
    print(f"- 평균 응답 시간: {sum(r['result']['elapsed_time'] for r in results if r['result']['success']) / max(success_count, 1):.2f}초")
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"test_results_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과 저장: test_results_{timestamp}.json")

if __name__ == "__main__":
    main()