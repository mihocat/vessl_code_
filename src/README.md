# RAG System Source Code

리팩토링된 RAG 시스템 소스 코드입니다.

## 구조

```
src/
├── __init__.py          # 패키지 초기화
├── config.py            # 중앙 설정 관리
├── llm_client.py        # LLM 클라이언트
├── document_loader.py   # 문서 로더 모듈
├── rag_system.py        # RAG 코어 시스템
├── services.py          # 웹 검색 및 응답 생성 서비스
└── app.py              # Gradio UI 애플리케이션
```

## 주요 개선사항

1. **모듈화**: 각 기능을 독립된 모듈로 분리
2. **설정 중앙화**: `Config` 클래스로 모든 설정 통합 관리
3. **타입 안전성**: 데이터클래스와 타입 힌트 사용
4. **확장성**: 새로운 기능 추가가 용이한 구조
5. **에러 처리**: 각 모듈별 강화된 에러 처리

## 사용법

```python
from config import Config
from llm_client import LLMClient
from app import create_gradio_app

# 설정 로드
config = Config()

# 앱 실행
app = create_gradio_app(config)
app.launch()
```

## 설정

환경 변수로 설정 오버라이드 가능:
- `LLM_BASE_URL`: LLM 서버 URL
- `EMBEDDING_MODEL`: 임베딩 모델명
- `DATASET_PATH`: 데이터셋 경로
- `SERVER_PORT`: 서버 포트