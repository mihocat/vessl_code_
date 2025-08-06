# 시스템 플로우 단순화 및 2분기 구조 구현

## 업데이트 정보
- **날짜**: 2025-08-06 11:31:00  
- **작업**: 사용자 요구사항에 따른 시스템 플로우 단순화
- **상태**: 완료

## 수정된 시스템 플로우 (2분기 구조)

### ✅ **최종 플로우 구조**

| 단계 | 구성 요소 | 사용 모델 | 조건 | 목적 |
|------|-----------|-----------|------|------|
| **1** | **사용자 입력** | - | 모든 질문 | 질문/이미지 접수 |
| **2** | **ChatGPT API 호출** | `gpt-4o-mini-2024-07-18` | 이미지 있는 경우만 | **질의당 1회만 호출** |
| **3** | **RAG 검색** | jinaai/jina-embeddings-v3 | 모든 질문 | 벡터 유사도 검색 |
| **4** | **신뢰도 평가** | - | 모든 질문 | 0.75 기준 판단 |
| **5a** | **RAG 직접 응답** | RAG 시스템 | **신뢰도 ≥ 0.75** | 검색 결과 직접 사용 |
| **5b** | **자체 LLM 응답** | vLLM 또는 ChatGPT | **신뢰도 < 0.75** | 고품질 LLM 생성 |

### 🚫 **제거된 중간 단계**
- ~~**5b 중간 LLM 보강 단계**~~ → **완전 제거**
- ~~**3단계 신뢰도 분기**~~ (`high`/`medium`/`low`) → **2분기로 단순화**
- ~~**중복 ChatGPT API 호출**~~ → **1회 호출로 제한**

## 코드 수정 내용

### 1. app.py `_generate_response` 메서드 핵심 변경

#### 변경 전 (3단계 분기)
```python
if max_score >= self.config.rag.high_confidence_threshold:
    confidence_level = "high"
elif max_score >= self.config.rag.medium_confidence_threshold:
    confidence_level = "medium"  # 중간 LLM 보강 단계
else:
    confidence_level = "low"
```

#### 변경 후 (2분기 구조)
```python
# 신뢰도 0.75 이상: RAG 직접 답변 사용
if max_score >= self.config.rag.high_confidence_threshold and results:
    best_result = results[0]
    response = response_header + best_result.answer
    confidence_level = "high"

# 신뢰도 0.75 미만: 자체 LLM 사용
else:
    confidence_level = "low"
    # ChatGPT API는 이미지가 있을 때만 1회 호출
    # 그 외에는 vLLM 사용
```

### 2. ChatGPT API 호출 최적화

| 상황 | 이전 | 현재 | 효과 |
|------|------|------|------|
| **이미지 + 텍스트** | Vision API + Chat API (2회) | **통합 API 1회** | 50% 호출 감소 |
| **텍스트만** | Chat API (1회) | **vLLM 우선** | 비용 절감 |
| **RAG 고신뢰도** | 필요시 LLM 호출 | **RAG 직접** | API 호출 0회 |

## 성능 개선 효과

### A. 호출 횟수 감소
- **이전**: 질의당 최대 2회 ChatGPT API 호출
- **현재**: 질의당 최대 1회 ChatGPT API 호출  
- **개선**: 50% 호출 횟수 감소

### B. 응답 속도 향상  
- **RAG 직접 응답**: API 호출 없이 즉시 응답
- **자체 LLM 우선**: vLLM이 ChatGPT보다 빠름
- **단순화된 로직**: 복잡한 분기 제거로 처리 속도 향상

### C. 비용 효율성
| 신뢰도 구간 | 시스템 | 비용 | 품질 |
|-------------|--------|------|------|
| **≥ 0.75** | RAG 직접 | **무료** | 높음 |
| **< 0.75** | vLLM 우선 | **저비용** | 높음 |
| **이미지 있음** | ChatGPT 1회 | $0.15/$0.60 | 최고 |

## 사용자 요구사항 준수 확인

✅ **ChatGPT API 질의당 1회만 호출**: 이미지가 있을 때만 1회 호출  
✅ **5b 중간 단계 제거**: 중간 LLM 보강 로직 완전 제거  
✅ **0.75 임계값 기준 2분기**: RAG 직접 vs 자체 LLM  
✅ **올바른 플로우**: API 호출 → RAG 검색 → 2분기 선택  
✅ **단순한 구조**: 복잡한 다단계 로직 제거

## 시스템 응답 예시

### 1. 고신뢰도 (≥ 0.75) - RAG 직접 응답
```
답변: [검색된 전문 내용을 직접 제공]

[점수: 0.856, 시스템: RAG 직접응답]
```

### 2. 저신뢰도 (< 0.75) - 자체 LLM 응답  
```
답변: [vLLM 또는 ChatGPT가 생성한 고품질 답변]

[점수: 0.432, 시스템: 자체 LLM]
```

## 다음 단계
- 변경사항 커밋  
- 기존 배포에서 새 로직 자동 적용
- 성능 및 비용 모니터링