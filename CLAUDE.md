# CLAUDE.md

이 파일은 이 저장소에서 코드 작업을 할 때 Claude Code (claude.ai/code)에 대한 통합 가이드를 제공합니다.

## 프로젝트 개요
VESSL AI 플랫폼 기반 LLM+RAG+웹검색 AI 챗봇 서버

### 핵심 기술
- **VESSL AI**: 클라우드 ML 워크플로우 관리
- **KoLlama**: 파인튜닝된 한국어 전기공학 전문 모델
- **ChromaDB**: 벡터 데이터베이스, 3000개 전문 문서
- **vLLM**: 고성능 LLM 서빙
- **DuckDuckGo**: 실시간 웹검색 통합

## 프로젝트 구조
```
vessl_code_/
├── src/                         # 모듈화된 소스 코드
│   ├── app.py                       # 메인 애플리케이션
│   ├── rag_system.py                # RAG 핵심 시스템
│   └── llm_client.py                # LLM 클라이언트
├── requirements.txt             # Python 의존성
├── run-v12.yaml                 # GitHub import 방식 VESSL 설정
└── CLAUDE.md                    # 개발 가이드
```

## 핵심 명령어
```bash
# GitHub 기반 모듈화 배포 (권장)
vessl run create -f run-v12.yaml

# 모니터링
vessl run list
vessl run logs <run-id> -f
```

## 현재 상황

### 완료된 작업
1. **✅ v12 GitHub 배포**: run-v12.yaml GitHub import 방식 검증 완료
2. **✅ GitHub 저장소**: https://github.com/mihocat/vessl_code_.git 구축 완료
3. **✅ 모듈화 코드**: app.py, rag_system.py, llm_client.py 분리 완료
4. **✅ 의존성 해결**: OpenCV libgl1-mesa-glx 문제 해결

### GitHub import 방식 검증 완료
✅ **저장소 연동**: https://github.com/mihocat/vessl_code_.git  
✅ **코드 다운로드**: GitHub import 성공  
✅ **의존성 해결**: libgl1-mesa-glx 추가로 OpenCV 문제 해결  
✅ **모듈화 구조**: src/ 폴더 기반 정상 작동

### 최종 배포 방식 확정
```bash
# GitHub 기반 모듈화 배포 (권장)
vessl run create -f run-v12.yaml
```

## 주요 목표
- ✅ LLM+RAG+웹검색 AI 챗봇 서버 완성
- ✅ GitHub 기반 모듈화 배포 방식 검증 완료
- 🔄 수식/이미지 처리 기능 추가 (향후 계획)

## 개발 가이드라인

### MCP 도구 활용
- **Context7**: 공식 문서 검색
- **GitHub**: 자동 커밋 및 푸시
- **TaskMaster**: 복잡한 작업 분할 관리

## RULES
- **모든 채팅은 CLAUDE.md 지침을 지속 준수**
- **CLAUDE.md 파일은 채팅이 업데이트될 때마다 최신화하라.**
- **웹검색 시 Context7 mcp를 이용하여 공식 문서에 접근하라.**
- **작업 완료 시 github mcp를 통해 commit/push하라.**