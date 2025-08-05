# OpenAI API 설정 가이드

## 🚀 빠른 시작

### 1. 로컬 개발
```bash
# 환경 파일 복사
cp .env.example .env

# .env 파일에 API 키 추가
# OPENAI_API_KEY=sk-your-actual-key-here

# 테스트 실행
python test_openai_api.py
```

### 2. VESSL 배포

#### Step 1: VESSL Secret 생성
```bash
# OpenAI API 키를 VESSL Secret으로 저장
vessl secret create openai-api-key --value "sk-your-actual-key-here"

# Secret 확인
vessl secret list
```

#### Step 2: 배포
```bash
# OpenAI 통합 버전으로 배포
vessl run create -f vessl_configs/run_with_openai.yaml
```

## 📝 환경 변수 설정

### 필수 설정
- `OPENAI_API_KEY`: OpenAI API 키 (Secret으로 관리)

### 선택 설정
- `OPENAI_VISION_MODEL`: Vision 모델 선택
  - `gpt-4o` (최고 성능, $2.50/$10.00 per 1M)
  - `gpt-4o-mini` (권장, $0.15/$0.60 per 1M)
- `USE_OPENAI_VISION`: OpenAI Vision 사용 여부 (`true`/`false`)
- `USE_OPENAI_LLM`: LLM 응답도 OpenAI 사용 (`true`/`false`)

## 💰 비용 관리

### 예상 비용 (GPT-4o-mini 기준)
- 이미지 분석: ~$0.00044/요청
- 월 1,000회: ~$13.2
- 월 10,000회: ~$132

### 비용 절감 팁
1. `gpt-4o-mini` 사용 (gpt-4o 대비 94% 저렴)
2. 이미지 크기 최적화 (512x512 권장)
3. 캐싱 구현
4. 필요한 경우만 OpenAI 사용

## 🔐 보안 주의사항

### ⚠️ 절대 하지 마세요
- ❌ API 키를 코드에 하드코딩
- ❌ `.env` 파일을 Git에 커밋
- ❌ 로그에 API 키 출력

### ✅ 항상 하세요
- ✅ VESSL Secrets 사용
- ✅ `.gitignore` 확인
- ✅ 커밋 전 `git status` 확인

## 🧪 테스트

### 로컬 테스트
```bash
# API 키 설정
export OPENAI_API_KEY="sk-..."

# 테스트 실행
python test_openai_api.py
```

### VESSL에서 테스트
```bash
# 로그 확인
vessl run logs <run-id> -f

# OpenAI API 연결 확인 메시지 찾기
# "✓ OpenAI API 연결 성공"
```

## 📊 모니터링

### OpenAI 사용량 확인
- https://platform.openai.com/usage
- 일일/월간 한도 설정 권장

### VESSL 로그에서 확인
```bash
# 토큰 사용량 확인
vessl run logs <run-id> | grep "토큰 사용량"
```

## 🛠️ 문제 해결

### API 키 오류
```
Error: OpenAI API key not found
```
→ VESSL Secret이 제대로 생성되었는지 확인

### 연결 오류
```
Error: Connection error
```
→ 네트워크 연결 및 API 키 유효성 확인

### 비용 초과
→ OpenAI 대시보드에서 사용량 한도 설정

## 📚 추가 자료

- [OpenAI API 문서](https://platform.openai.com/docs)
- [VESSL Secrets 문서](https://docs.vessl.ai/guides/secrets)
- [API 키 보안 가이드](./docs/API_KEY_SECURITY_GUIDE.md)
- [OpenAI 가격 정보](./docs/OPENAI_PRICING_GUIDE.md)