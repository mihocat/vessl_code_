# VESSL 배포 가이드

## 개요

VESSL에서 Gradio 기반 AI 챗봇을 배포하는 방법을 설명합니다.

## 배포 유형

### 1. Batch Mode (기본)
- 단순 실행 후 종료
- Gradio 접근 불가
- 테스트 및 디버깅용

### 2. Service Mode (권장)
- 지속적인 서비스 제공
- 외부에서 Gradio UI 접근 가능
- 프로덕션 배포용

## 배포 방법

### 서비스 모드로 배포

```bash
# Gradio 서비스로 배포
vessl run create -f vessl_configs/run_with_gradio_service.yaml
```

### 배포 상태 확인

```bash
# 실행 중인 서비스 목록
vessl run list

# 로그 모니터링
vessl run logs <run-id> -f
```

### 서비스 접근

배포 완료 후 VESSL 콘솔에서:
1. 해당 Run 클릭
2. "Service" 탭 선택
3. "gradio" 포트의 URL 클릭
4. Gradio UI 접근

## 환경 변수

### OpenAI Vision API 사용

```yaml
env:
  USE_OPENAI_VISION: "true"      # OpenAI Vision API 사용
  OPENAI_VISION_MODEL: "gpt-4o-mini"  # 모델 선택
```

### OCR 엔진 사용 (로컬)

```yaml
env:
  USE_OPENAI_VISION: "false"     # 로컬 OCR 사용
```

## 문제 해결

### Gradio 접근 불가

1. **mode: service** 설정 확인
2. **ports** 섹션에 gradio 포트(7860) 정의 확인
3. 배포 상태가 "running"인지 확인
4. healthcheck 통과 여부 확인

### API 키 오류

1. Storage에 `/dataset/o_api` 파일 존재 확인
2. 파일 내용이 올바른 API 키인지 확인
3. 환경 변수 `OPENAI_API_KEY` 설정 확인

### 메모리 부족

GPU 메모리 부족 시:
- `USE_OPENAI_VISION: "true"`로 설정하여 OCR 모델 로드 스킵
- 더 큰 GPU preset 사용 (`gpu-a10-large`)

## 최적화 팁

### 1. 시작 시간 단축

OpenAI Vision API 사용 시:
- OCR 패키지 설치 스킵
- OCR 모델 로드 스킵
- 약 2-3분 단축

### 2. 안정성 향상

```yaml
healthcheck:
  initial_delay_seconds: 120  # 충분한 초기화 시간
  failure_threshold: 3        # 재시작 전 실패 허용
```

### 3. 로그 레벨 조정

```yaml
env:
  LOG_LEVEL: "INFO"  # DEBUG for troubleshooting
```

## 배포 시나리오

### 개발/테스트
```bash
vessl run create -f vessl_configs/run_with_openai.yaml
```

### 프로덕션
```bash
vessl run create -f vessl_configs/run_with_gradio_service.yaml
```

### 비용 최적화 (Storage API Key)
```bash
vessl run create -f vessl_configs/run_with_storage_apikey.yaml
```