# Story 1.1: 모델 로딩 최적화 구현 계획

**Story ID**: 1.1
**Epic**: 성능 최적화 및 안정성 강화
**상태**: Draft
**예상 소요 시간**: 4시간

## Story Statement
"시스템 관리자로서, 나는 서비스 시작 시간을 30% 단축하기를 원한다. 왜냐하면 빠른 재시작이 배포와 장애 복구에 중요하기 때문이다."

## Acceptance Criteria
1. [ ] 현재 모델 로딩 시간 측정 및 기록
2. [ ] Lazy loading 메커니즘 구현
3. [ ] 모델 캐싱 시스템 구현
4. [ ] 시작 시간 30% 이상 단축 검증
5. [ ] 기능 회귀 테스트 통과

## Technical Context

### 현재 상태 분석
**파일**: `src/simple_image_analyzer.py`
- BLIP 모델: ~1.5GB (Salesforce/blip-image-captioning-base)
- EasyOCR 모델: ~64MB (한국어) + ~64MB (영어)
- 초기화 시점: 클래스 인스턴스 생성 시

**파일**: `src/app.py`
- Florence2ImageAnalyzer 초기화: 3회 재시도 로직
- 동기적 로딩으로 인한 블로킹

### 최적화 전략

#### 1. Lazy Loading 구현
```python
class SimpleImageAnalyzer:
    def __init__(self):
        self._blip_model = None
        self._ocr_reader = None
        self._initialized = False
    
    @property
    def blip_model(self):
        if self._blip_model is None:
            self._load_blip_model()
        return self._blip_model
    
    @property
    def ocr_reader(self):
        if self._ocr_reader is None:
            self._load_ocr_reader()
        return self._ocr_reader
```

#### 2. 모델 캐싱 전략
- 디스크 캐싱: HuggingFace 캐시 활용
- 메모리 매핑: torch.load with mmap=True
- 사전 다운로드: 도커 이미지에 포함

#### 3. 병렬 로딩
```python
import concurrent.futures

def initialize_models_parallel():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_blip = executor.submit(load_blip_model)
        future_ocr = executor.submit(load_ocr_reader)
        
        blip_model = future_blip.result()
        ocr_reader = future_ocr.result()
```

## Implementation Tasks

### Task 1: 프로파일링 도구 설정 (AC: 1)
- [ ] 시간 측정 데코레이터 구현
- [ ] 모델별 로딩 시간 측정
- [ ] 메모리 사용량 추적
- [ ] 기준선(baseline) 기록

### Task 2: Lazy Loading 구현 (AC: 2)
- [ ] SimpleImageAnalyzer 클래스 리팩토링
- [ ] Property 기반 지연 로딩 구현
- [ ] 초기화 플래그 관리
- [ ] 단위 테스트 작성

### Task 3: 모델 캐싱 구현 (AC: 3)
- [ ] 캐시 디렉토리 구조 설계
- [ ] 모델 버전 관리 로직
- [ ] 캐시 유효성 검증
- [ ] 디스크 공간 관리

### Task 4: 병렬 로딩 최적화 (AC: 2, 3)
- [ ] ThreadPoolExecutor 구현
- [ ] 에러 핸들링 추가
- [ ] 타임아웃 설정
- [ ] 로깅 통합

### Task 5: 성능 검증 (AC: 4)
- [ ] 로딩 시간 벤치마크
- [ ] 메모리 사용량 비교
- [ ] 다양한 환경에서 테스트
- [ ] 성능 리포트 작성

### Task 6: 통합 테스트 (AC: 5)
- [ ] 기존 기능 테스트
- [ ] 엣지 케이스 검증
- [ ] 동시성 테스트
- [ ] 회귀 테스트 스위트 실행

## 리스크 및 고려사항

### 리스크
1. **메모리 매핑 호환성**: 일부 환경에서 mmap 미지원
2. **캐시 무효화**: 모델 업데이트 시 캐시 관리
3. **동시성 이슈**: 여러 요청 시 초기화 경쟁 상태

### 완화 전략
1. Fallback 메커니즘 구현
2. 캐시 버전 관리 시스템
3. Thread-safe 초기화 보장

## 성공 지표
- 시작 시간: 현재 ~15초 → 목표 ~10초 (33% 단축)
- 메모리 피크: 현재 대비 20% 이하 증가
- 첫 요청 응답 시간: 5초 이내

## 종속성
- Python 3.8+
- PyTorch 2.0+
- HuggingFace Transformers 4.30+

## 참고 자료
- [PyTorch Lazy Loading Guide](https://pytorch.org/docs/stable/notes/serialization.html)
- [HuggingFace Model Caching](https://huggingface.co/docs/transformers/installation#cache-setup)
- [Python Concurrent Futures](https://docs.python.org/3/library/concurrent.futures.html)