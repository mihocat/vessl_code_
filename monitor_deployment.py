#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VESSL Deployment Monitor with Testing
VESSL 배포 모니터링 및 테스트
"""

import subprocess
import time
import sys
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ANSI 색상 코드
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


class VESSLMonitor:
    """VESSL 배포 모니터"""
    
    def __init__(self, run_id: Optional[str] = None):
        """모니터 초기화"""
        self.run_id = run_id
        self.start_time = time.time()
        self.test_results = []
        self.gradio_url = None
        
    def create_deployment(self, yaml_path: str = "vessl_configs/run_enhanced.yaml") -> str:
        """VESSL 배포 생성"""
        print(f"\n{BLUE}=== VESSL 배포 시작 ==={RESET}")
        print(f"설정 파일: {yaml_path}")
        
        try:
            result = subprocess.run(
                ["vessl", "run", "create", "-f", yaml_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # 실행 ID 추출
                output = result.stdout.strip()
                for line in output.split('\n'):
                    if 'RUN-' in line:
                        self.run_id = line.split()[-1] if ' ' in line else line
                        break
                
                print(f"{GREEN}✓ 배포 생성 성공{RESET}")
                print(f"Run ID: {BOLD}{self.run_id}{RESET}")
                return self.run_id
            else:
                print(f"{RED}✗ 배포 생성 실패{RESET}")
                print(f"에러: {result.stderr}")
                sys.exit(1)
                
        except Exception as e:
            print(f"{RED}✗ 배포 명령 실행 실패: {e}{RESET}")
            sys.exit(1)
    
    def monitor_logs(self, max_duration: int = 600):
        """실시간 로그 모니터링"""
        if not self.run_id:
            print(f"{RED}Run ID가 없습니다{RESET}")
            return False
        
        print(f"\n{BLUE}=== 로그 모니터링 시작 ==={RESET}")
        print(f"최대 모니터링 시간: {max_duration}초")
        
        try:
            # 로그 스트리밍 프로세스 시작
            process = subprocess.Popen(
                ["vessl", "run", "logs", self.run_id, "-f"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            success_indicators = [
                "✓ 고급 모드로 실행",
                "✓ 향상된 모드로 실행",
                "✓ 표준 모드로 실행",
                "Running on local URL",
                "Gradio app running"
            ]
            
            error_indicators = [
                "workload failed",
                "Error",
                "FAIL",
                "OutOfMemoryError",
                "Traceback"
            ]
            
            start_monitor = time.time()
            deployment_successful = False
            
            while True:
                # 시간 초과 확인
                if time.time() - start_monitor > max_duration:
                    print(f"\n{YELLOW}⚠ 모니터링 시간 초과 ({max_duration}초){RESET}")
                    break
                
                # 로그 읽기
                line = process.stdout.readline()
                if not line:
                    # 프로세스 종료 확인
                    if process.poll() is not None:
                        break
                    continue
                
                # 로그 출력
                line = line.strip()
                if line:
                    # 색상 처리
                    if any(indicator in line for indicator in success_indicators):
                        print(f"{GREEN}{line}{RESET}")
                        deployment_successful = True
                        
                        # Gradio URL 추출
                        if "Running on local URL" in line or "http" in line:
                            import re
                            urls = re.findall(r'http[s]?://[^\s]+', line)
                            if urls:
                                self.gradio_url = urls[0].rstrip('/')
                                print(f"\n{GREEN}✓ Gradio URL: {self.gradio_url}{RESET}")
                    
                    elif any(indicator in line for indicator in error_indicators):
                        print(f"{RED}{line}{RESET}")
                    
                    elif "===" in line:
                        print(f"{BLUE}{line}{RESET}")
                    
                    else:
                        print(line)
                
                # 배포 성공 후 테스트 시작
                if deployment_successful and self.gradio_url:
                    print(f"\n{GREEN}✓ 배포 성공! 테스트를 시작합니다...{RESET}")
                    break
            
            # 프로세스 정리
            process.terminate()
            
            return deployment_successful
            
        except KeyboardInterrupt:
            print(f"\n{YELLOW}모니터링 중단됨{RESET}")
            if 'process' in locals():
                process.terminate()
            return False
        except Exception as e:
            print(f"{RED}모니터링 오류: {e}{RESET}")
            return False
    
    def test_deployment(self):
        """배포된 시스템 테스트"""
        print(f"\n{BLUE}=== 배포 테스트 시작 ==={RESET}")
        
        if not self.gradio_url:
            # VESSL 서비스 URL 가져오기
            self.get_service_url()
        
        if not self.gradio_url:
            print(f"{RED}✗ Gradio URL을 찾을 수 없습니다{RESET}")
            return
        
        # 테스트 질문들
        test_questions = [
            {
                "question": "3상 전력 시스템에서 선간전압이 380V이고 부하전류가 10A일 때 전력을 구하시오.",
                "expected_keywords": ["전력", "3상", "380", "10"]
            },
            {
                "question": "RLC 직렬회로에서 공진주파수를 구하는 공식은?",
                "expected_keywords": ["공진", "주파수", "LC", "2π"]
            },
            {
                "question": "변압기의 철손과 동손의 차이점을 설명하세요.",
                "expected_keywords": ["철손", "동손", "변압기"]
            },
            {
                "question": "옴의 법칙을 설명해주세요.",
                "expected_keywords": ["전압", "전류", "저항", "V=IR"]
            },
            {
                "question": "유도전동기의 슬립이 0.05일 때 회전속도는?",
                "expected_keywords": ["슬립", "회전속도", "동기속도"]
            }
        ]
        
        print(f"\n테스트할 질문: {len(test_questions)}개")
        print(f"Gradio URL: {self.gradio_url}")
        
        # API 엔드포인트 확인
        api_url = f"{self.gradio_url}/api/predict"
        
        success_count = 0
        
        for i, test in enumerate(test_questions, 1):
            print(f"\n{BOLD}테스트 {i}/{len(test_questions)}{RESET}")
            print(f"질문: {test['question'][:50]}...")
            
            try:
                # Gradio API 호출
                response = requests.post(
                    api_url,
                    json={
                        "data": [test['question'], None, []]  # message, image, history
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get('data', [''])[0] if isinstance(result.get('data'), list) else str(result)
                    
                    # 키워드 확인
                    keywords_found = sum(1 for keyword in test['expected_keywords'] if keyword.lower() in answer.lower())
                    keyword_ratio = keywords_found / len(test['expected_keywords'])
                    
                    if keyword_ratio >= 0.5:
                        print(f"{GREEN}✓ 테스트 통과 (키워드 매칭: {keywords_found}/{len(test['expected_keywords'])}){RESET}")
                        success_count += 1
                        self.test_results.append({
                            'question': test['question'],
                            'status': 'PASS',
                            'keywords_matched': f"{keywords_found}/{len(test['expected_keywords'])}"
                        })
                    else:
                        print(f"{YELLOW}⚠ 키워드 매칭 부족 ({keywords_found}/{len(test['expected_keywords'])}){RESET}")
                        self.test_results.append({
                            'question': test['question'],
                            'status': 'WARN',
                            'keywords_matched': f"{keywords_found}/{len(test['expected_keywords'])}"
                        })
                    
                    # 응답 일부 출력
                    print(f"응답: {answer[:200]}...")
                    
                else:
                    print(f"{RED}✗ API 호출 실패 (상태 코드: {response.status_code}){RESET}")
                    self.test_results.append({
                        'question': test['question'],
                        'status': 'FAIL',
                        'error': f"HTTP {response.status_code}"
                    })
                
            except requests.exceptions.Timeout:
                print(f"{RED}✗ 응답 시간 초과{RESET}")
                self.test_results.append({
                    'question': test['question'],
                    'status': 'FAIL',
                    'error': 'Timeout'
                })
            except Exception as e:
                print(f"{RED}✗ 테스트 오류: {e}{RESET}")
                self.test_results.append({
                    'question': test['question'],
                    'status': 'FAIL',
                    'error': str(e)
                })
            
            # 과부하 방지
            time.sleep(2)
        
        # 테스트 결과 요약
        print(f"\n{BLUE}=== 테스트 결과 요약 ==={RESET}")
        print(f"총 테스트: {len(test_questions)}")
        print(f"성공: {success_count} ({success_count/len(test_questions)*100:.1f}%)")
        print(f"실패: {len(test_questions) - success_count}")
    
    def get_service_url(self):
        """VESSL 서비스 URL 가져오기"""
        try:
            result = subprocess.run(
                ["vessl", "run", "read", self.run_id, "--json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # 서비스 URL 추출
                services = data.get('services', {})
                for service_name, service_info in services.items():
                    if 'gradio' in service_name.lower():
                        self.gradio_url = service_info.get('url', '')
                        break
        except:
            pass
    
    def terminate_deployment(self):
        """배포 종료"""
        if not self.run_id:
            return
        
        print(f"\n{BLUE}=== 배포 종료 ==={RESET}")
        
        try:
            result = subprocess.run(
                ["vessl", "run", "terminate", self.run_id],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"{GREEN}✓ 배포가 종료되었습니다{RESET}")
            else:
                print(f"{YELLOW}⚠ 배포 종료 실패: {result.stderr}{RESET}")
                
        except Exception as e:
            print(f"{RED}✗ 배포 종료 오류: {e}{RESET}")
    
    def save_report(self):
        """테스트 보고서 저장"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'run_id': self.run_id,
            'duration': time.time() - self.start_time,
            'gradio_url': self.gradio_url,
            'test_results': self.test_results,
            'summary': {
                'total_tests': len(self.test_results),
                'passed': sum(1 for r in self.test_results if r['status'] == 'PASS'),
                'failed': sum(1 for r in self.test_results if r['status'] == 'FAIL'),
                'warnings': sum(1 for r in self.test_results if r['status'] == 'WARN')
            }
        }
        
        filename = f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n{GREEN}✓ 보고서 저장: {filename}{RESET}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VESSL Deployment Monitor")
    parser.add_argument(
        "--yaml",
        type=str,
        default="vessl_configs/run_enhanced.yaml",
        help="VESSL run configuration file"
    )
    parser.add_argument(
        "--no-terminate",
        action="store_true",
        help="Do not terminate deployment after testing"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Existing run ID to monitor"
    )
    
    args = parser.parse_args()
    
    # 모니터 생성
    monitor = VESSLMonitor(run_id=args.run_id)
    
    try:
        # 새 배포 생성 (run-id가 없는 경우)
        if not args.run_id:
            monitor.create_deployment(args.yaml)
        
        # 로그 모니터링
        success = monitor.monitor_logs()
        
        if success:
            # 테스트 실행
            time.sleep(5)  # 서비스 안정화 대기
            monitor.test_deployment()
        
        # 보고서 저장
        monitor.save_report()
        
        # 배포 종료 (옵션)
        if not args.no_terminate:
            monitor.terminate_deployment()
        else:
            print(f"\n{YELLOW}배포가 계속 실행 중입니다. 수동으로 종료하세요:{RESET}")
            print(f"vessl run terminate {monitor.run_id}")
            
    except KeyboardInterrupt:
        print(f"\n{YELLOW}모니터링 중단됨{RESET}")
        if not args.no_terminate:
            monitor.terminate_deployment()


if __name__ == "__main__":
    main()