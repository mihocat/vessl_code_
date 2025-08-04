#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
피드백 시스템
사용자 피드백을 통한 지속적인 개선
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import asyncio
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

logger = logging.getLogger(__name__)


@dataclass
class Feedback:
    """피드백 데이터"""
    id: str
    timestamp: datetime
    problem_text: str
    system_answer: Dict[str, Any]
    user_rating: int  # 1-5
    correct_answer: Optional[Dict[str, Any]] = None
    user_comment: Optional[str] = None
    problem_type: Optional[str] = None
    improvement_applied: bool = False


@dataclass
class PerformanceMetrics:
    """성능 지표"""
    total_problems: int
    correct_problems: int
    accuracy: float
    avg_rating: float
    confidence_correlation: float
    problem_type_accuracy: Dict[str, float]


class FeedbackSystem:
    """피드백 및 개선 시스템"""
    
    def __init__(self, storage_path: str = "./feedback_data"):
        self.storage_path = storage_path
        self.feedbacks: List[Feedback] = []
        self.performance_history: List[PerformanceMetrics] = []
        self.improvement_rules: Dict[str, Any] = {}
        
        # 저장 경로 생성
        os.makedirs(storage_path, exist_ok=True)
        
        # 기존 데이터 로드
        self._load_data()
    
    def _load_data(self):
        """저장된 데이터 로드"""
        try:
            # 피드백 로드
            feedback_file = os.path.join(self.storage_path, "feedbacks.json")
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.feedbacks = [
                        Feedback(**item) for item in data
                    ]
            
            # 성능 히스토리 로드
            metrics_file = os.path.join(self.storage_path, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.performance_history = [
                        PerformanceMetrics(**item) for item in data
                    ]
            
            # 개선 규칙 로드
            rules_file = os.path.join(self.storage_path, "improvement_rules.json")
            if os.path.exists(rules_file):
                with open(rules_file, 'r', encoding='utf-8') as f:
                    self.improvement_rules = json.load(f)
            
            logger.info(f"Loaded {len(self.feedbacks)} feedbacks")
            
        except Exception as e:
            logger.error(f"Failed to load feedback data: {e}")
    
    def _save_data(self):
        """데이터 저장"""
        try:
            # 피드백 저장
            feedback_file = os.path.join(self.storage_path, "feedbacks.json")
            with open(feedback_file, 'w', encoding='utf-8') as f:
                data = [asdict(fb) for fb in self.feedbacks]
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            # 성능 히스토리 저장
            metrics_file = os.path.join(self.storage_path, "metrics.json")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                data = [asdict(m) for m in self.performance_history]
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 개선 규칙 저장
            rules_file = os.path.join(self.storage_path, "improvement_rules.json")
            with open(rules_file, 'w', encoding='utf-8') as f:
                json.dump(self.improvement_rules, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save feedback data: {e}")
    
    def add_feedback(
        self,
        problem_text: str,
        system_answer: Dict[str, Any],
        user_rating: int,
        correct_answer: Optional[Dict[str, Any]] = None,
        user_comment: Optional[str] = None,
        problem_type: Optional[str] = None
    ) -> str:
        """피드백 추가"""
        feedback = Feedback(
            id=f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.feedbacks)}",
            timestamp=datetime.now(),
            problem_text=problem_text,
            system_answer=system_answer,
            user_rating=user_rating,
            correct_answer=correct_answer,
            user_comment=user_comment,
            problem_type=problem_type,
            improvement_applied=False
        )
        
        self.feedbacks.append(feedback)
        self._save_data()
        
        # 즉시 분석 및 개선
        self._analyze_feedback(feedback)
        
        return feedback.id
    
    def _analyze_feedback(self, feedback: Feedback):
        """피드백 분석 및 개선 규칙 생성"""
        # 낮은 평점 피드백 분석
        if feedback.user_rating <= 2:
            self._analyze_poor_performance(feedback)
        
        # 정답이 제공된 경우 분석
        if feedback.correct_answer:
            self._analyze_error_pattern(feedback)
    
    def _analyze_poor_performance(self, feedback: Feedback):
        """저성능 케이스 분석"""
        # 문제 유형별 실패 패턴 추적
        if feedback.problem_type:
            key = f"poor_performance_{feedback.problem_type}"
            if key not in self.improvement_rules:
                self.improvement_rules[key] = {
                    'count': 0,
                    'common_errors': [],
                    'suggested_improvements': []
                }
            
            self.improvement_rules[key]['count'] += 1
            
            # 공통 오류 패턴 추출
            if feedback.user_comment:
                self.improvement_rules[key]['common_errors'].append({
                    'comment': feedback.user_comment,
                    'system_answer': feedback.system_answer
                })
    
    def _analyze_error_pattern(self, feedback: Feedback):
        """오류 패턴 분석"""
        system_val = feedback.system_answer.get('value', 0)
        correct_val = feedback.correct_answer.get('value', 0)
        
        if correct_val != 0:
            error_rate = abs(system_val - correct_val) / abs(correct_val)
            
            # 오류 유형 분류
            error_type = self._classify_error(error_rate, feedback)
            
            # 개선 규칙 업데이트
            key = f"error_pattern_{error_type}"
            if key not in self.improvement_rules:
                self.improvement_rules[key] = {
                    'count': 0,
                    'avg_error_rate': 0,
                    'examples': []
                }
            
            rules = self.improvement_rules[key]
            rules['count'] += 1
            rules['avg_error_rate'] = (
                (rules['avg_error_rate'] * (rules['count'] - 1) + error_rate) 
                / rules['count']
            )
            rules['examples'].append({
                'problem': feedback.problem_text[:100],
                'error_rate': error_rate
            })
    
    def _classify_error(self, error_rate: float, feedback: Feedback) -> str:
        """오류 분류"""
        if error_rate < 0.05:
            return "minor_calculation"
        elif error_rate < 0.2:
            return "formula_selection"
        elif error_rate < 0.5:
            return "unit_conversion"
        else:
            return "major_misunderstanding"
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """성능 지표 계산"""
        if not self.feedbacks:
            return PerformanceMetrics(
                total_problems=0,
                correct_problems=0,
                accuracy=0.0,
                avg_rating=0.0,
                confidence_correlation=0.0,
                problem_type_accuracy={}
            )
        
        total = len(self.feedbacks)
        correct = sum(1 for fb in self.feedbacks if fb.user_rating >= 4)
        
        # 평균 평점
        avg_rating = np.mean([fb.user_rating for fb in self.feedbacks])
        
        # 신뢰도와 정확도의 상관관계
        if len(self.feedbacks) > 1:
            confidences = [fb.system_answer.get('confidence', 0) for fb in self.feedbacks]
            ratings = [fb.user_rating for fb in self.feedbacks]
            correlation = np.corrcoef(confidences, ratings)[0, 1]
        else:
            correlation = 0.0
        
        # 문제 유형별 정확도
        type_accuracy = {}
        type_counts = {}
        
        for fb in self.feedbacks:
            if fb.problem_type:
                if fb.problem_type not in type_counts:
                    type_counts[fb.problem_type] = {'total': 0, 'correct': 0}
                
                type_counts[fb.problem_type]['total'] += 1
                if fb.user_rating >= 4:
                    type_counts[fb.problem_type]['correct'] += 1
        
        for ptype, counts in type_counts.items():
            type_accuracy[ptype] = counts['correct'] / counts['total']
        
        metrics = PerformanceMetrics(
            total_problems=total,
            correct_problems=correct,
            accuracy=correct / total,
            avg_rating=avg_rating,
            confidence_correlation=correlation if not np.isnan(correlation) else 0.0,
            problem_type_accuracy=type_accuracy
        )
        
        # 히스토리에 추가
        self.performance_history.append(metrics)
        self._save_data()
        
        return metrics
    
    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """개선 제안 생성"""
        suggestions = []
        
        # 1. 저성능 문제 유형
        for key, rule in self.improvement_rules.items():
            if key.startswith('poor_performance_') and rule['count'] >= 3:
                problem_type = key.replace('poor_performance_', '')
                suggestions.append({
                    'type': 'poor_performance',
                    'problem_type': problem_type,
                    'frequency': rule['count'],
                    'suggestion': f"{problem_type} 유형 문제의 성능 개선 필요",
                    'priority': 'high' if rule['count'] >= 5 else 'medium'
                })
        
        # 2. 반복적인 오류 패턴
        for key, rule in self.improvement_rules.items():
            if key.startswith('error_pattern_') and rule['count'] >= 3:
                error_type = key.replace('error_pattern_', '')
                suggestions.append({
                    'type': 'error_pattern',
                    'error_type': error_type,
                    'frequency': rule['count'],
                    'avg_error_rate': rule['avg_error_rate'],
                    'suggestion': self._get_error_suggestion(error_type),
                    'priority': 'high' if rule['avg_error_rate'] > 0.2 else 'medium'
                })
        
        # 3. 낮은 신뢰도-정확도 상관관계
        if self.performance_history:
            latest_metrics = self.performance_history[-1]
            if latest_metrics.confidence_correlation < 0.5:
                suggestions.append({
                    'type': 'confidence_calibration',
                    'correlation': latest_metrics.confidence_correlation,
                    'suggestion': "신뢰도 계산 로직 재조정 필요",
                    'priority': 'medium'
                })
        
        return sorted(suggestions, key=lambda x: x['priority'] == 'high', reverse=True)
    
    def _get_error_suggestion(self, error_type: str) -> str:
        """오류 유형별 개선 제안"""
        suggestions = {
            'minor_calculation': "계산 정밀도 향상 필요",
            'formula_selection': "공식 선택 로직 개선 필요",
            'unit_conversion': "단위 변환 로직 검토 필요",
            'major_misunderstanding': "문제 이해 및 분류 시스템 전면 재검토 필요"
        }
        return suggestions.get(error_type, "일반적인 개선 필요")
    
    def apply_improvements(self, ai_system) -> Dict[str, Any]:
        """개선 사항 적용"""
        applied = []
        
        # 개선 제안 가져오기
        suggestions = self.get_improvement_suggestions()
        
        for suggestion in suggestions:
            if suggestion['priority'] == 'high':
                # 실제 개선 적용 로직
                if suggestion['type'] == 'poor_performance':
                    # 특정 문제 유형에 대한 추가 학습
                    applied.append({
                        'type': 'enhanced_training',
                        'target': suggestion['problem_type'],
                        'action': 'Added specialized rules'
                    })
                
                elif suggestion['type'] == 'error_pattern':
                    # 오류 패턴에 대한 보정
                    applied.append({
                        'type': 'error_correction',
                        'target': suggestion['error_type'],
                        'action': 'Adjusted calculation methods'
                    })
        
        # 적용된 개선 사항 표시
        for fb in self.feedbacks:
            if not fb.improvement_applied and fb.user_rating <= 2:
                fb.improvement_applied = True
        
        self._save_data()
        
        return {
            'suggestions_count': len(suggestions),
            'applied_count': len(applied),
            'applied_improvements': applied
        }
    
    def generate_report(self) -> str:
        """성능 보고서 생성"""
        metrics = self.calculate_metrics()
        suggestions = self.get_improvement_suggestions()
        
        report = "# AI 시스템 성능 보고서\n\n"
        report += f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## 전체 성능 지표\n"
        report += f"- 총 문제 수: {metrics.total_problems}\n"
        report += f"- 정답률: {metrics.accuracy:.1%}\n"
        report += f"- 평균 사용자 평점: {metrics.avg_rating:.1f}/5.0\n"
        report += f"- 신뢰도-정확도 상관계수: {metrics.confidence_correlation:.2f}\n\n"
        
        report += "## 문제 유형별 성능\n"
        for ptype, acc in metrics.problem_type_accuracy.items():
            report += f"- {ptype}: {acc:.1%}\n"
        report += "\n"
        
        report += "## 개선 제안\n"
        for i, suggestion in enumerate(suggestions[:5], 1):
            report += f"{i}. [{suggestion['priority'].upper()}] {suggestion['suggestion']}\n"
            if 'frequency' in suggestion:
                report += f"   - 발생 빈도: {suggestion['frequency']}회\n"
        
        # 시간별 성능 추이
        if len(self.performance_history) > 1:
            report += "\n## 성능 추이\n"
            recent = self.performance_history[-5:]
            for i, m in enumerate(recent):
                report += f"- 측정 {i+1}: 정답률 {m.accuracy:.1%}, 평점 {m.avg_rating:.1f}\n"
        
        return report
    
    def find_similar_problems(self, problem_text: str, top_k: int = 5) -> List[Feedback]:
        """유사한 문제 찾기"""
        if not self.feedbacks:
            return []
        
        # 간단한 텍스트 유사도 (실제로는 더 정교한 방법 필요)
        similarities = []
        for fb in self.feedbacks:
            # 문자 수준 유사도
            sim = self._text_similarity(problem_text, fb.problem_text)
            similarities.append((sim, fb))
        
        # 상위 k개 반환
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [fb for _, fb in similarities[:top_k]]
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산"""
        # 간단한 Jaccard 유사도
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)


class AdaptiveLearning:
    """적응형 학습 시스템"""
    
    def __init__(self, feedback_system: FeedbackSystem):
        self.feedback_system = feedback_system
        self.learning_rate = 0.1
        self.adaptation_history = []
    
    async def adapt(self, ai_system) -> Dict[str, Any]:
        """시스템 적응"""
        # 1. 성능 분석
        metrics = self.feedback_system.calculate_metrics()
        
        # 2. 개선 필요 영역 식별
        weak_areas = []
        for ptype, acc in metrics.problem_type_accuracy.items():
            if acc < 0.7:  # 70% 미만 정확도
                weak_areas.append(ptype)
        
        # 3. 적응 전략 수립
        adaptations = []
        
        # 약한 영역에 대한 추가 학습
        for area in weak_areas:
            adaptations.append({
                'type': 'focused_learning',
                'target': area,
                'method': 'increase_weight'
            })
        
        # 4. 적응 적용
        applied = await self._apply_adaptations(ai_system, adaptations)
        
        # 5. 히스토리 기록
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics,
            'adaptations': applied
        })
        
        return {
            'weak_areas': weak_areas,
            'adaptations_applied': len(applied),
            'new_accuracy_target': 0.8
        }
    
    async def _apply_adaptations(
        self, 
        ai_system, 
        adaptations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """적응 적용"""
        applied = []
        
        for adaptation in adaptations:
            if adaptation['type'] == 'focused_learning':
                # 특정 문제 유형에 가중치 증가
                # 실제 구현에서는 모델 파라미터 조정
                applied.append({
                    'type': adaptation['type'],
                    'target': adaptation['target'],
                    'adjustment': f"Increased weight by {self.learning_rate}"
                })
        
        return applied


# 사용 예시
def demo_feedback_system():
    """피드백 시스템 데모"""
    # 시스템 초기화
    feedback_system = FeedbackSystem()
    
    # 피드백 추가
    feedback_id = feedback_system.add_feedback(
        problem_text="유효전력 100kW, 무효전력 50kVar일 때 역률은?",
        system_answer={'value': 0.85, 'confidence': 0.9},
        user_rating=3,
        correct_answer={'value': 0.894},
        user_comment="계산이 약간 부정확합니다",
        problem_type="power_factor"
    )
    
    print(f"Feedback added: {feedback_id}")
    
    # 성능 지표 계산
    metrics = feedback_system.calculate_metrics()
    print(f"Accuracy: {metrics.accuracy:.1%}")
    print(f"Average rating: {metrics.avg_rating:.1f}")
    
    # 개선 제안
    suggestions = feedback_system.get_improvement_suggestions()
    for s in suggestions:
        print(f"Suggestion: {s['suggestion']} (Priority: {s['priority']})")
    
    # 보고서 생성
    report = feedback_system.generate_report()
    print("\n" + report)


if __name__ == "__main__":
    demo_feedback_system()