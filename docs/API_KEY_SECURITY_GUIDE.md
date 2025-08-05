# API 키 보안 가이드

## 🔐 중요: API 키를 GitHub에 절대 커밋하지 마세요!

## 로컬 개발 환경

### 1. 환경 변수 파일 사용 (.env)
```bash
# .env 파일 생성 (이 파일은 .gitignore에 포함됨)
cp .env.example .env

# .env 파일 편집
nano .env
```

### 2. 시스템 환경 변수 설정
```bash
# 임시 설정 (터미널 세션 동안만 유효)
export OPENAI_API_KEY="sk-your-api-key-here"

# 영구 설정 (~/.bashrc 또는 ~/.zshrc에 추가)
echo 'export OPENAI_API_KEY="sk-your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## VESSL 배포 시 API 키 설정

### 방법 1: VESSL Secrets 사용 (권장)

1. VESSL 웹 콘솔에서 Secret 생성:
```bash
vessl secret create openai-api-key --value "sk-your-api-key-here"
```

2. run.yaml에서 Secret 참조:
```yaml
name: RAG_Bllossom_with_OpenAI
env:
  OPENAI_API_KEY:
    secret: openai-api-key
  USE_OPENAI_VISION: "true"
  OPENAI_VISION_MODEL: "gpt-4o-mini"
```

### 방법 2: VESSL 환경 변수 직접 설정

```bash
# VESSL run 생성 시 환경 변수 전달
vessl run create -f vessl_configs/run.yaml \
  --env OPENAI_API_KEY="sk-your-api-key-here" \
  --env USE_OPENAI_VISION="true"
```

### 방법 3: VESSL Organization Secrets

조직 레벨에서 Secret 관리:
```bash
# 조직 Secret 생성
vessl organization secret create openai-api-key \
  --value "sk-your-api-key-here"

# run.yaml에서 참조
env:
  OPENAI_API_KEY:
    organization-secret: openai-api-key
```

## 보안 체크리스트

### ✅ 해야 할 일
1. `.gitignore`에 `.env` 파일 추가 확인
2. 커밋 전 `git status`로 `.env` 파일이 포함되지 않았는지 확인
3. VESSL Secrets 사용하여 프로덕션 환경에서 키 관리
4. API 키 정기적으로 로테이션
5. 키별 사용량 모니터링

### ❌ 하지 말아야 할 일
1. `config.py`에 API 키 하드코딩
2. `.env` 파일을 Git에 커밋
3. 로그에 API 키 출력
4. 공개 저장소에 키 포함
5. 키를 평문으로 저장

## 실수로 커밋한 경우

### 즉시 조치사항
1. **OpenAI 대시보드에서 즉시 키 무효화**
2. 새 키 생성
3. Git 히스토리에서 제거:
```bash
# BFG Repo-Cleaner 사용
bfg --delete-files .env
git push --force

# 또는 git filter-branch 사용
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all
```

## 코드에서 안전하게 사용하기

```python
import os
from config import config

# 환경 변수에서 읽기 (안전)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")

# config 객체 사용 (환경 변수 자동 로드)
if config.openai.api_key:
    client = OpenAI(api_key=config.openai.api_key)
else:
    raise ValueError("OpenAI API key not configured")
```

## 개발 워크플로우

1. **초기 설정**
   ```bash
   cp .env.example .env
   # .env 파일에 실제 API 키 입력
   ```

2. **개발**
   ```bash
   # 로컬 테스트
   python test_openai_api.py
   ```

3. **커밋 전 확인**
   ```bash
   git status  # .env가 없는지 확인
   git diff    # API 키가 포함되지 않았는지 확인
   ```

4. **배포**
   ```bash
   # VESSL Secret 생성 (한 번만)
   vessl secret create openai-api-key --value "sk-..."
   
   # 배포
   vessl run create -f vessl_configs/run.yaml
   ```

## 모니터링

### API 사용량 확인
- OpenAI 대시보드: https://platform.openai.com/usage
- 일일/월간 한도 설정
- 이상 사용 알림 설정

### 로그 검토
```python
# 로그에 API 키가 노출되지 않도록 주의
logger.info(f"Using OpenAI model: {model}")  # ✅ OK
logger.info(f"API Key: {api_key}")  # ❌ 절대 금지
```