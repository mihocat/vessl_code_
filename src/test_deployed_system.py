#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VESSL 배포 시스템 테스트 스크립트
EXAMPLE.md의 실제 질문으로 배포된 시스템 테스트
"""

import requests
import json
import time
from datetime import datetime
import sys

def test_deployed_system(base_url):
    """배포된 시스템에 대한 테스트 수행"""
    
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
    
    results = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"=== VESSL 배포 시스템 테스트 시작 ({timestamp}) ===")
    print(f"테스트 URL: {base_url}")
    print("=" * 60)
    
    for test in test_questions:
        print(f"\n테스트 {test['id']}: {test['question']}")
        
        start_time = time.time()
        
        try:
            # API 엔드포인트로 요청
            response = requests.post(
                f"{base_url}/api/chat",
                json={"message": test["question"]},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                test_result = {
                    "test_id": test["id"],
                    "question": test["question"],
                    "expected_hint": test["expected"][:50] + "...",
                    "actual_response": result.get("response", "응답 없음"),
                    "search_results": result.get("search_results", 0),
                    "confidence_score": result.get("confidence", 0),
                    "response_time": f"{elapsed_time:.2f}초",
                    "status": "성공"
                }
                
                print(f"✅ 응답 성공 ({elapsed_time:.2f}초)")
                print(f"응답: {test_result['actual_response'][:100]}...")
                
            else:
                test_result = {
                    "test_id": test["id"],
                    "question": test["question"],
                    "status": "실패",
                    "error": f"HTTP {response.status_code}",
                    "response_time": f"{elapsed_time:.2f}초"
                }
                print(f"❌ 응답 실패: HTTP {response.status_code}")
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            test_result = {
                "test_id": test["id"],
                "question": test["question"],
                "status": "오류",
                "error": str(e),
                "response_time": f"{elapsed_time:.2f}초"
            }
            print(f"❌ 오류 발생: {e}")
        
        results.append(test_result)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 요약:")
    success_count = sum(1 for r in results if r.get("status") == "성공")
    print(f"- 총 테스트: {len(test_questions)}개")
    print(f"- 성공: {success_count}개")
    print(f"- 실패: {len(test_questions) - success_count}개")
    
    return results

def generate_test_report(results):
    """테스트 결과를 마크다운 형식으로 생성"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = f"""# VESSL 배포 테스트 결과

## 테스트 정보
- **테스트 시간**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **테스트 질문 수**: {len(results)}개 (EXAMPLE.md 기반)

## 상세 테스트 결과

"""
    
    for result in results:
        report += f"### 테스트 {result['test_id']}: {result['question']}\n"
        report += f"- **상태**: {result['status']}\n"
        report += f"- **응답 시간**: {result['response_time']}\n"
        
        if result['status'] == "성공":
            report += f"- **검색 결과 수**: {result.get('search_results', 'N/A')}\n"
            report += f"- **신뢰도 점수**: {result.get('confidence_score', 'N/A')}\n"
            report += f"- **예상 답변 힌트**: {result['expected_hint']}\n"
            report += f"- **실제 응답**:\n```\n{result['actual_response']}\n```\n"
        else:
            report += f"- **오류**: {result.get('error', 'Unknown')}\n"
        
        report += "\n"
    
    # 테스트 요약
    success_count = sum(1 for r in results if r.get("status") == "성공")
    report += f"""## 테스트 요약

| 항목 | 결과 |
|------|------|
| 총 테스트 | {len(results)}개 |
| 성공 | {success_count}개 |
| 실패 | {len(results) - success_count}개 |
| 성공률 | {(success_count/len(results)*100):.1f}% |

## 평가

"""
    
    if success_count == len(results):
        report += "✅ **모든 테스트 통과**: RAG 시스템이 정상적으로 작동하고 있습니다.\n"
    elif success_count > len(results) * 0.7:
        report += "⚠️ **부분 성공**: 대부분의 테스트가 통과했으나 일부 개선이 필요합니다.\n"
    else:
        report += "❌ **개선 필요**: 많은 테스트가 실패했습니다. 시스템 점검이 필요합니다.\n"
    
    # 파일 저장
    filename = f"logs/{timestamp}_배포테스트결과.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n테스트 보고서 저장: {filename}")
    
    return filename

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python test_deployed_system.py <service_url>")
        print("예시: python test_deployed_system.py https://service-xxx.vessl.com")
        sys.exit(1)
    
    service_url = sys.argv[1].rstrip('/')
    
    # 테스트 실행
    results = test_deployed_system(service_url)
    
    # 보고서 생성
    report_file = generate_test_report(results)
    
    print("\n테스트 완료!")