# VESSL Storage를 사용한 API 키 관리

## 개요

VESSL Storage에 API 키를 저장하여 안전하게 관리하는 방법입니다.

## Storage 구조

```
volume://vessl-storage/APIkey/
└── o_api  # OpenAI API 키 파일
```

## 사용 방법

### 1. API 키 파일 준비

로컬에서 `o_api` 파일 생성:
```bash
echo "sk-your-openai-api-key" > o_api
```

### 2. VESSL Storage에 업로드

VESSL CLI 또는 웹 콘솔을 통해 업로드:
```bash
# VESSL CLI 사용 (예시)
vessl storage upload o_api volume://vessl-storage/APIkey/
```

### 3. 배포

Storage 기반 배포 설정 사용:
```bash
vessl run create -f vessl_configs/run_with_storage_apikey.yaml
```

## 작동 방식

### 1. 런타임 시 자동 로드

배포 시 다음 순서로 API 키를 로드합니다:

1. **Shell 스크립트에서 환경 변수 설정**
   ```bash
   export OPENAI_API_KEY=$(cat /dataset/o_api)
   ```

2. **Python에서 자동 감지**
   ```python
   # api_key_loader.py가 자동으로 실행되어
   # /dataset/o_api 파일에서 키를 읽음
   ```

### 2. 우선순위

1. 환경 변수 (`OPENAI_API_KEY`)
2. Storage 파일 (`/dataset/o_api`)
3. 로컬 `.env` 파일

## 장점

### ✅ 보안
- GitHub에 키가 노출되지 않음
- Storage는 조직 내부에서만 접근 가능
- 파일 권한으로 추가 보호 가능

### ✅ 편의성
- 한 번 업로드하면 모든 배포에서 사용
- Secret 생성 없이 바로 사용
- 팀원 간 쉽게 공유

### ✅ 유연성
- 여러 API 키 관리 가능
- 환경별로 다른 키 사용 가능
- 백업 및 버전 관리 용이

## 주의사항

### ⚠️ Storage 권한
- Storage 접근 권한이 있는 사용자만 키 확인 가능
- 적절한 권한 설정 필요

### ⚠️ 파일 형식
- 텍스트 파일로 저장
- 줄바꿈 없이 키만 포함
- UTF-8 인코딩 사용

### ⚠️ 경로 확인
- `/dataset/` 경로는 YAML의 import 설정에 따름
- 정확한 파일명 사용 (`o_api`)

## 문제 해결

### API 키를 찾을 수 없음
```
✗ API key file not found at /dataset/o_api
```
→ Storage 마운트 확인, 파일명 확인

### API 키 형식 오류
```
Invalid API key format
```
→ 파일 내용이 'sk-'로 시작하는지 확인

### 권한 오류
```
Permission denied
```
→ Storage 접근 권한 확인

## 다른 API 키 추가

다른 서비스의 API 키도 같은 방식으로 관리:

```
volume://vessl-storage/APIkey/
├── o_api          # OpenAI
├── anthropic_api  # Anthropic
└── google_api     # Google
```

`api_key_loader.py`를 확장하여 여러 키 지원 가능