# vLLM 서버 시작 문제 해결
날짜: 2025-08-05 13:24

## 문제 진단

### 1. 첫 번째 배포 실패 (369367213652)
- **문제**: LLM 서버 연결 실패
- **원인**: vLLM 서버가 시작되지 않음
- **로그**:
  ```
  INFO:llm_client:Waiting for LLM server...
  ERROR:llm_client:LLM server failed to start
  ERROR:app_v2:Failed to connect to LLM server
  ```

### 2. 포트 불일치 문제
- **config.py**: `base_url: str = "http://localhost:8000"`
- **YAML**: vLLM 서버는 8088 포트로 설정됨
- **해결**: config.py의 포트를 8088로 변경

## 수정 사항

### 1. config.py 포트 변경
```python
# 변경 전
base_url: str = "http://localhost:8000"

# 변경 후
base_url: str = "http://localhost:8088"
```

### 2. vLLM 서버 시작 개선
- PID 추적 추가
- 헬스체크 기반 대기 로직 구현
- 시작 실패 시 로그 확인 기능 추가
- 대기 시간 증가 (30초 → 최대 5분)

### 3. 디버깅 개선
```bash
# vLLM 서버 PID 저장
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# 헬스체크 기반 대기
for i in {1..60}; do
    sleep 5
    if curl -s http://localhost:8088/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    echo "Waiting for vLLM server... ($i/60)"
    # 30초, 60초 시점에 로그 확인
    if [ $i -eq 30 ] || [ $i -eq 60 ]; then
        echo "=== vLLM log (last 20 lines) ==="
        tail -20 /tmp/vllm.log || echo "No vLLM log found"
        echo "=== End of vLLM log ==="
    fi
done
```

## 기대 효과

1. **안정적인 서버 시작**
   - 헬스체크로 실제 서버 준비 상태 확인
   - 충분한 대기 시간 제공

2. **향상된 디버깅**
   - vLLM 로그 확인 가능
   - 시작 실패 시 원인 파악 용이

3. **포트 일치**
   - 클라이언트와 서버 포트 일치로 연결 문제 해결

## 다음 단계

1. 수정사항 GitHub 푸시
2. 재배포 및 모니터링
3. vLLM 서버 정상 시작 확인
4. Gradio UI 접근 테스트