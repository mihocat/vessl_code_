# 개선된 VESSL 배포 모니터링 프로세스

## 배포 절차 개선안

### 1. 배포 시작
```bash
vessl run create -f vessl_configs/run.yaml
```

### 2. 효율적인 모니터링 방법

#### 방법 1: 준비 완료 메시지 대기
```bash
# "준비 완료" 메시지가 나올 때까지 모니터링
vessl run logs <run-id> -f | grep -E "(준비 완료|Running on|7860)"
```

#### 방법 2: 단계별 상태 확인
```bash
# 1단계: 의존성 설치 확인 (약 2-3분)
vessl run logs <run-id> | grep "Successfully installed"

# 2단계: 임베딩 모델 로드 확인
vessl run logs <run-id> | grep "한국어 임베딩 모델 로드 완료"

# 3단계: 문서 벡터화 완료 확인
vessl run logs <run-id> | grep "전기공학 지식베이스 구축 완료"

# 4단계: 서비스 시작 확인
vessl run logs <run-id> | grep "전기공학 전문 RAG 시스템 준비 완료"
```

#### 방법 3: 자동화 스크립트
```bash
#!/bin/bash
RUN_ID=$1
TIMEOUT=600  # 10분

echo "배포 모니터링 시작: $RUN_ID"

# 준비 완료까지 대기
timeout $TIMEOUT bash -c "
while true; do
    if vessl run logs $RUN_ID | grep -q '준비 완료'; then
        echo '✅ 시스템 준비 완료!'
        break
    fi
    sleep 10
done
"
```

### 3. 주요 확인 포인트

| 단계 | 로그 메시지 | 예상 시간 |
|------|------------|----------|
| 의존성 설치 | "Successfully installed" | 2-3분 |
| 모델 로드 | "jinaai/jina-embeddings-v3" | 30초 |
| 문서 벡터화 | "6000개 문서" | 90초 |
| 서비스 시작 | "Running on local URL" | 즉시 |

### 4. 에러 감지
```bash
# 에러 모니터링
vessl run logs <run-id> -f | grep -E "(ERROR|Failed|Exception)"
```

### 5. 상태 기반 자동 처리

```python
import subprocess
import time

def monitor_deployment(run_id, timeout=600):
    """배포 상태 모니터링 및 자동 처리"""
    
    stages = {
        "installing": "pip install",
        "model_loading": "임베딩 모델 로드",
        "vectorizing": "문서 벡터화",
        "ready": "준비 완료"
    }
    
    start_time = time.time()
    current_stage = "installing"
    
    while time.time() - start_time < timeout:
        # 로그 확인
        result = subprocess.run(
            f"vessl run logs {run_id} | tail -100",
            shell=True, capture_output=True, text=True
        )
        
        logs = result.stdout
        
        # 단계 확인
        if "준비 완료" in logs:
            print("✅ 배포 완료! 테스트 가능")
            return "ready"
        elif "전기공학 지식베이스 구축" in logs:
            current_stage = "vectorizing"
        elif "임베딩 모델 로드 완료" in logs:
            current_stage = "model_loading"
            
        # 에러 확인
        if "ERROR" in logs or "Failed" in logs:
            print(f"❌ 에러 발생: {current_stage}")
            return "error"
            
        print(f"⏳ 현재 단계: {current_stage}")
        time.sleep(10)
    
    print("⏱️ 타임아웃")
    return "timeout"
```

## 권장사항

1. **배포 시작 후 즉시 모니터링 시작**
   - 타임아웃 없이 전체 로그 스트리밍 대신
   - 주요 체크포인트만 확인

2. **병렬 작업 활용**
   - 모니터링 중 다른 터미널에서 테스트 준비
   - 테스트 스크립트 미리 준비

3. **자동 알림 설정**
   ```bash
   # macOS 알림 예시
   vessl run logs <run-id> -f | grep "준비 완료" && osascript -e 'display notification "RAG 시스템 준비 완료" with title "VESSL"'
   ```

4. **로그 저장**
   ```bash
   vessl run logs <run-id> > logs/deployment_$(date +%Y%m%d_%H%M%S).log
   ```