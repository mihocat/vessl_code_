#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Autonomous System
신경망 기반 자율 학습 시스템 - 인간 뇌 구조 모방
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict
import asyncio
import time
import json
import pickle
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class NeuronType(Enum):
    """뉴런 타입"""
    SENSORY = "sensory"  # 감각 뉴런
    MOTOR = "motor"  # 운동 뉴런
    INTER = "inter"  # 연결 뉴런
    MIRROR = "mirror"  # 거울 뉴런
    PYRAMIDAL = "pyramidal"  # 피라미드 뉴런
    PURKINJE = "purkinje"  # 푸르킨예 뉴런
    DOPAMINE = "dopamine"  # 도파민 뉴런
    SEROTONIN = "serotonin"  # 세로토닌 뉴런


class BrainRegion(Enum):
    """뇌 영역"""
    FRONTAL = "frontal"  # 전두엽
    PARIETAL = "parietal"  # 두정엽
    TEMPORAL = "temporal"  # 측두엽
    OCCIPITAL = "occipital"  # 후두엽
    CEREBELLUM = "cerebellum"  # 소뇌
    HIPPOCAMPUS = "hippocampus"  # 해마
    AMYGDALA = "amygdala"  # 편도체
    THALAMUS = "thalamus"  # 시상
    HYPOTHALAMUS = "hypothalamus"  # 시상하부
    BASAL_GANGLIA = "basal_ganglia"  # 기저핵


@dataclass
class Neuron:
    """개별 뉴런"""
    id: str
    type: NeuronType
    region: BrainRegion
    position: Tuple[float, float, float]  # 3D 위치
    activation: float = 0.0
    threshold: float = 0.5
    connections: Dict[str, float] = field(default_factory=dict)  # 연결 가중치
    neurotransmitters: Dict[str, float] = field(default_factory=dict)
    plasticity: float = 1.0  # 가소성
    fatigue: float = 0.0  # 피로도
    history: deque = field(default_factory=lambda: deque(maxlen=100))


class NeuralLayer(nn.Module):
    """신경층"""
    
    def __init__(self, input_size: int, output_size: int, 
                 neuron_type: NeuronType, dropout: float = 0.1):
        super().__init__()
        self.neuron_type = neuron_type
        
        # 기본 레이어
        self.linear = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        
        # 뉴런 타입별 특화 처리
        if neuron_type == NeuronType.PYRAMIDAL:
            # 피라미드 뉴런: 복잡한 처리
            self.complex_layer = nn.Sequential(
                nn.Linear(output_size, output_size * 2),
                nn.GELU(),
                nn.Linear(output_size * 2, output_size),
                nn.LayerNorm(output_size)
            )
        elif neuron_type == NeuronType.DOPAMINE:
            # 도파민 뉴런: 보상 신호
            self.reward_layer = nn.Linear(output_size, 1)
        elif neuron_type == NeuronType.MIRROR:
            # 거울 뉴런: 모방 학습
            self.mirror_attention = nn.MultiheadAttention(output_size, 4)
        
        # 가소성 파라미터
        self.plasticity = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 기본 처리
        out = self.linear(x)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        # 뉴런 타입별 처리
        if self.neuron_type == NeuronType.PYRAMIDAL and hasattr(self, 'complex_layer'):
            out = out + self.complex_layer(out)
        elif self.neuron_type == NeuronType.MIRROR and context is not None:
            attn_out, _ = self.mirror_attention(out, context, context)
            out = out + attn_out
        
        # 가소성 적용
        out = out * self.plasticity
        
        return out


class BrainModule(nn.Module):
    """뇌 모듈 - 특정 뇌 영역 시뮬레이션"""
    
    def __init__(self, region: BrainRegion, input_size: int, hidden_size: int):
        super().__init__()
        self.region = region
        
        # 영역별 특화 구조
        if region == BrainRegion.FRONTAL:
            # 전두엽: 실행 기능, 계획
            self.layers = nn.ModuleList([
                NeuralLayer(input_size, hidden_size, NeuronType.PYRAMIDAL),
                NeuralLayer(hidden_size, hidden_size, NeuronType.INTER),
                NeuralLayer(hidden_size, hidden_size, NeuronType.DOPAMINE)
            ])
            
        elif region == BrainRegion.HIPPOCAMPUS:
            # 해마: 기억 형성
            self.layers = nn.ModuleList([
                NeuralLayer(input_size, hidden_size * 2, NeuronType.PYRAMIDAL),
                NeuralLayer(hidden_size * 2, hidden_size, NeuronType.INTER)
            ])
            self.memory_gate = nn.GRU(hidden_size, hidden_size, batch_first=True)
            
        elif region == BrainRegion.AMYGDALA:
            # 편도체: 감정 처리
            self.layers = nn.ModuleList([
                NeuralLayer(input_size, hidden_size, NeuronType.SENSORY),
                NeuralLayer(hidden_size, hidden_size // 2, NeuronType.SEROTONIN)
            ])
            self.emotion_classifier = nn.Linear(hidden_size // 2, 8)  # 8가지 기본 감정
            
        elif region == BrainRegion.CEREBELLUM:
            # 소뇌: 운동 조정, 학습
            self.layers = nn.ModuleList([
                NeuralLayer(input_size, hidden_size * 3, NeuronType.PURKINJE),
                NeuralLayer(hidden_size * 3, hidden_size, NeuronType.MOTOR)
            ])
            
        else:
            # 기본 구조
            self.layers = nn.ModuleList([
                NeuralLayer(input_size, hidden_size, NeuronType.INTER),
                NeuralLayer(hidden_size, hidden_size, NeuronType.INTER)
            ])
    
    def forward(self, x: torch.Tensor, state: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = x
        new_state = {}
        
        # 레이어 통과
        for layer in self.layers:
            out = layer(out)
        
        # 영역별 특수 처리
        if self.region == BrainRegion.HIPPOCAMPUS and hasattr(self, 'memory_gate'):
            if state and 'memory' in state:
                out, new_memory = self.memory_gate(out.unsqueeze(1), state['memory'])
                new_state['memory'] = new_memory
            else:
                out, new_memory = self.memory_gate(out.unsqueeze(1))
                new_state['memory'] = new_memory
            out = out.squeeze(1)
            
        elif self.region == BrainRegion.AMYGDALA and hasattr(self, 'emotion_classifier'):
            emotions = self.emotion_classifier(out)
            new_state['emotions'] = F.softmax(emotions, dim=-1)
        
        return out, new_state


class NeuralAutonomousSystem(nn.Module):
    """신경망 기반 자율 시스템"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 차원 설정
        self.input_dim = config.get('input_dim', 768)
        self.hidden_dim = config.get('hidden_dim', 1024)
        self.output_dim = config.get('output_dim', 768)
        
        # 뇌 영역 초기화
        self.brain_regions = nn.ModuleDict({
            region.value: BrainModule(region, self.input_dim, self.hidden_dim)
            for region in BrainRegion
        })
        
        # 영역 간 연결
        self.inter_regional_connections = nn.ModuleDict({
            f"{r1.value}_to_{r2.value}": nn.Linear(self.hidden_dim, self.hidden_dim)
            for r1 in BrainRegion
            for r2 in BrainRegion
            if r1 != r2
        })
        
        # 통합 처리
        self.integration_network = nn.Sequential(
            nn.Linear(self.hidden_dim * len(BrainRegion), self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # 의식 네트워크 (Global Workspace Theory)
        self.consciousness_network = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=8,
                dim_feedforward=self.hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=6
        )
        
        # 메타 학습 네트워크
        self.meta_learner = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 신경조절물질 시스템
        self.neuromodulators = nn.ModuleDict({
            'dopamine': nn.Linear(self.hidden_dim, 1),  # 보상/동기
            'serotonin': nn.Linear(self.hidden_dim, 1),  # 기분/안정
            'norepinephrine': nn.Linear(self.hidden_dim, 1),  # 각성/주의
            'acetylcholine': nn.Linear(self.hidden_dim, 1),  # 학습/기억
            'gaba': nn.Linear(self.hidden_dim, 1),  # 억제
            'glutamate': nn.Linear(self.hidden_dim, 1)  # 흥분
        })
        
        # 자율 학습 시스템
        self.hebbian_learning = config.get('hebbian_learning', True)
        self.spike_timing = config.get('spike_timing', True)
        
        # 상태 관리
        self.brain_state = {}
        self.learning_history = deque(maxlen=1000)
        self.consciousness_buffer = deque(maxlen=100)
        
        # 에너지 시스템
        self.energy_level = 1.0
        self.fatigue_rate = config.get('fatigue_rate', 0.001)
        
        logger.info("Neural Autonomous System initialized")
    
    def forward(self, x: torch.Tensor, 
                task: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """전방향 처리"""
        batch_size = x.shape[0] if x.dim() > 1 else 1
        
        # 입력 정규화
        x = F.normalize(x, p=2, dim=-1)
        
        # 1. 병렬 뇌 영역 처리
        regional_outputs = {}
        regional_states = {}
        
        for region_name, region_module in self.brain_regions.items():
            region_out, region_state = region_module(
                x, self.brain_state.get(region_name)
            )
            regional_outputs[region_name] = region_out
            regional_states[region_name] = region_state
        
        # 2. 영역 간 통신
        interconnected = {}
        for connection_name, connection_layer in self.inter_regional_connections.items():
            source, target = connection_name.split('_to_')
            if source in regional_outputs and target in regional_outputs:
                signal = connection_layer(regional_outputs[source])
                interconnected[f"{source}_to_{target}"] = signal
        
        # 3. 의식 통합 (Global Workspace)
        all_signals = list(regional_outputs.values()) + list(interconnected.values())
        consciousness_input = torch.stack(all_signals, dim=1)  # [batch, num_signals, hidden_dim]
        conscious_output = self.consciousness_network(consciousness_input)
        
        # 4. 통합 처리
        integrated = torch.cat(list(regional_outputs.values()), dim=-1)
        final_output = self.integration_network(integrated)
        
        # 5. 신경조절물질 계산
        neuromodulator_levels = {}
        for name, modulator in self.neuromodulators.items():
            level = torch.sigmoid(modulator(conscious_output.mean(dim=1)))
            neuromodulator_levels[name] = level
        
        # 6. 메타 학습
        if hasattr(self, 'meta_learner'):
            meta_input = conscious_output
            meta_output, _ = self.meta_learner(meta_input)
            meta_output = meta_output.mean(dim=1)  # 평균 풀링
        else:
            meta_output = conscious_output.mean(dim=1)
        
        # 7. 자율 학습 (Hebbian)
        if self.training and self.hebbian_learning:
            self._apply_hebbian_learning(regional_outputs, interconnected)
        
        # 8. 에너지 관리
        self._update_energy_level(neuromodulator_levels)
        
        # 상태 업데이트
        self.brain_state = regional_states
        self.consciousness_buffer.append(conscious_output.detach())
        
        return {
            'output': final_output,
            'consciousness': conscious_output,
            'regional_outputs': regional_outputs,
            'neuromodulators': neuromodulator_levels,
            'meta_learning': meta_output,
            'energy_level': torch.tensor([self.energy_level]),
            'brain_state': self.brain_state
        }
    
    def _apply_hebbian_learning(self, regional_outputs: Dict[str, torch.Tensor],
                                interconnected: Dict[str, torch.Tensor]):
        """헤비안 학습 적용 - 함께 활성화되는 뉴런은 강화"""
        with torch.no_grad():
            # 영역 간 연결 강화
            for connection_name, signal in interconnected.items():
                source, target = connection_name.split('_to_')
                if source in regional_outputs and target in regional_outputs:
                    # 상관관계 계산
                    correlation = torch.matmul(
                        regional_outputs[source].T,
                        regional_outputs[target]
                    ) / regional_outputs[source].shape[0]
                    
                    # 가중치 업데이트
                    connection_layer = self.inter_regional_connections[connection_name]
                    if hasattr(connection_layer, 'weight'):
                        connection_layer.weight.data += 0.01 * correlation
    
    def _update_energy_level(self, neuromodulator_levels: Dict[str, torch.Tensor]):
        """에너지 레벨 업데이트"""
        # 도파민은 에너지 증가, GABA는 감소
        dopamine = neuromodulator_levels.get('dopamine', torch.tensor([0.5])).item()
        gaba = neuromodulator_levels.get('gaba', torch.tensor([0.5])).item()
        
        energy_change = (dopamine - gaba) * 0.01 - self.fatigue_rate
        self.energy_level = max(0.1, min(1.0, self.energy_level + energy_change))
    
    def dream_mode(self, num_cycles: int = 10) -> List[torch.Tensor]:
        """꿈 모드 - 자율적 패턴 생성 및 재조직"""
        dreams = []
        
        with torch.no_grad():
            # 랜덤 노이즈에서 시작
            current_state = torch.randn(1, self.input_dim)
            
            for _ in range(num_cycles):
                # 뇌 활동 시뮬레이션
                result = self.forward(current_state)
                
                # 의식 상태를 다음 입력으로
                current_state = result['consciousness'].mean(dim=1)
                
                # 꿈 기록
                dreams.append(result['output'].detach())
                
                # 에너지 회복
                self.energy_level = min(1.0, self.energy_level + 0.05)
        
        return dreams
    
    def consolidate_memory(self):
        """기억 공고화 - 중요한 패턴 강화"""
        if len(self.learning_history) < 10:
            return
        
        with torch.no_grad():
            # 최근 학습 이력에서 패턴 추출
            recent_patterns = list(self.learning_history)[-100:]
            
            # 해마 영역 강화
            hippocampus = self.brain_regions['hippocampus']
            for pattern in recent_patterns:
                if isinstance(pattern, dict) and 'input' in pattern:
                    # 패턴 재생
                    _, state = hippocampus(pattern['input'])
                    
                    # 시냅스 강화
                    if hasattr(hippocampus, 'layers'):
                        for layer in hippocampus.layers:
                            if hasattr(layer, 'plasticity'):
                                layer.plasticity.data *= 1.01
    
    def introspect(self) -> Dict[str, Any]:
        """내성 - 자기 상태 분석"""
        introspection = {
            'energy_level': self.energy_level,
            'active_regions': [],
            'dominant_neuromodulator': None,
            'consciousness_coherence': 0.0,
            'learning_rate': 0.0,
            'memory_capacity': len(self.learning_history) / 1000
        }
        
        # 활성 영역 분석
        if self.brain_state:
            for region, state in self.brain_state.items():
                if isinstance(state, dict):
                    for key, tensor in state.items():
                        if torch.is_tensor(tensor) and tensor.mean().item() > 0.5:
                            introspection['active_regions'].append(region)
        
        # 의식 일관성 분석
        if len(self.consciousness_buffer) > 2:
            recent_states = list(self.consciousness_buffer)[-10:]
            similarities = []
            for i in range(1, len(recent_states)):
                sim = F.cosine_similarity(
                    recent_states[i-1].flatten(),
                    recent_states[i].flatten(),
                    dim=0
                )
                similarities.append(sim.item())
            introspection['consciousness_coherence'] = np.mean(similarities)
        
        return introspection
    
    def evolve(self, fitness_scores: List[float]):
        """진화 - 적응도에 따른 구조 변경"""
        if len(fitness_scores) < 10:
            return
        
        avg_fitness = np.mean(fitness_scores)
        
        with torch.no_grad():
            # 높은 적응도 -> 현재 구조 강화
            if avg_fitness > 0.8:
                for module in self.modules():
                    if hasattr(module, 'weight'):
                        module.weight.data *= 1.01
            
            # 낮은 적응도 -> 구조 변이
            elif avg_fitness < 0.3:
                for module in self.modules():
                    if hasattr(module, 'weight'):
                        # 작은 랜덤 변이 추가
                        noise = torch.randn_like(module.weight) * 0.01
                        module.weight.data += noise
    
    def save_state(self, path: Path):
        """상태 저장"""
        state = {
            'model_state': self.state_dict(),
            'brain_state': self.brain_state,
            'energy_level': self.energy_level,
            'learning_history': list(self.learning_history)[-100:],
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, path: Path):
        """상태 로드"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.load_state_dict(state['model_state'])
        self.brain_state = state['brain_state']
        self.energy_level = state['energy_level']
        self.learning_history = deque(state['learning_history'], maxlen=1000)


class AutonomousAgent:
    """자율 에이전트"""
    
    def __init__(self, neural_system: NeuralAutonomousSystem):
        self.neural_system = neural_system
        self.goals = []
        self.beliefs = {}
        self.plans = []
        self.experiences = deque(maxlen=10000)
        
    async def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """자율적 사고"""
        # 지각 -> 텐서 변환
        input_tensor = self._perception_to_tensor(perception)
        
        # 신경망 처리
        with torch.no_grad():
            neural_output = self.neural_system(input_tensor)
        
        # 의사결정
        decision = self._make_decision(neural_output, perception)
        
        # 경험 저장
        self.experiences.append({
            'perception': perception,
            'neural_state': neural_output,
            'decision': decision,
            'timestamp': time.time()
        })
        
        # 학습 (비동기)
        if len(self.experiences) > 100:
            asyncio.create_task(self._autonomous_learning())
        
        return decision
    
    def _perception_to_tensor(self, perception: Dict[str, Any]) -> torch.Tensor:
        """지각 정보를 텐서로 변환"""
        # 간단한 구현 - 실제로는 더 복잡한 인코딩 필요
        features = []
        
        for key, value in perception.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # 문자열 해시
                features.append(float(int(hashlib.md5(value.encode()).hexdigest()[:8], 16)) / 1e8)
            elif isinstance(value, list):
                features.extend([float(v) if isinstance(v, (int, float)) else 0.0 for v in value[:10]])
        
        # 패딩
        while len(features) < self.neural_system.input_dim:
            features.append(0.0)
        
        return torch.tensor(features[:self.neural_system.input_dim], dtype=torch.float32).unsqueeze(0)
    
    def _make_decision(self, neural_output: Dict[str, torch.Tensor], 
                      perception: Dict[str, Any]) -> Dict[str, Any]:
        """의사결정"""
        decision = {
            'action': None,
            'confidence': 0.0,
            'reasoning': [],
            'emotion': None
        }
        
        # 신경조절물질 기반 행동 선택
        neuromodulators = neural_output['neuromodulators']
        
        # 도파민 높음 -> 탐험
        if neuromodulators['dopamine'].item() > 0.7:
            decision['action'] = 'explore'
            decision['reasoning'].append('High dopamine - seeking novelty')
        
        # 세로토닌 낮음 -> 안전 추구
        elif neuromodulators['serotonin'].item() < 0.3:
            decision['action'] = 'retreat'
            decision['reasoning'].append('Low serotonin - seeking safety')
        
        # GABA 높음 -> 휴식
        elif neuromodulators['gaba'].item() > 0.8:
            decision['action'] = 'rest'
            decision['reasoning'].append('High GABA - need rest')
        
        else:
            # 의식 상태 기반 결정
            consciousness = neural_output['consciousness']
            decision['action'] = 'process'
            decision['confidence'] = consciousness.std().item()
        
        # 감정 상태
        if 'amygdala' in neural_output['brain_state']:
            emotions = neural_output['brain_state']['amygdala'].get('emotions')
            if emotions is not None:
                emotion_idx = emotions.argmax().item()
                emotion_names = ['joy', 'sadness', 'anger', 'fear', 
                               'surprise', 'disgust', 'trust', 'anticipation']
                decision['emotion'] = emotion_names[emotion_idx]
        
        return decision
    
    async def _autonomous_learning(self):
        """자율 학습"""
        # 최근 경험에서 학습
        recent_experiences = list(self.experiences)[-500:]
        
        # 패턴 분석
        patterns = self._analyze_patterns(recent_experiences)
        
        # 신경망 업데이트
        if patterns:
            # 여기서는 간단히 진화 메서드 호출
            fitness_scores = [p.get('success_rate', 0.5) for p in patterns]
            self.neural_system.evolve(fitness_scores)
        
        # 기억 공고화
        self.neural_system.consolidate_memory()
    
    def _analyze_patterns(self, experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """경험에서 패턴 분석"""
        patterns = []
        
        # 행동-결과 패턴 분석
        action_outcomes = defaultdict(list)
        for i in range(1, len(experiences)):
            prev_exp = experiences[i-1]
            curr_exp = experiences[i]
            
            if 'decision' in prev_exp and 'perception' in curr_exp:
                action = prev_exp['decision'].get('action')
                # 간단한 성공 척도 (실제로는 더 복잡한 평가 필요)
                success = curr_exp['perception'].get('reward', 0.5)
                action_outcomes[action].append(success)
        
        # 패턴 정리
        for action, outcomes in action_outcomes.items():
            if outcomes:
                patterns.append({
                    'action': action,
                    'success_rate': np.mean(outcomes),
                    'frequency': len(outcomes)
                })
        
        return patterns


# 사용 예시
if __name__ == "__main__":
    # 설정
    config = {
        'input_dim': 768,
        'hidden_dim': 1024,
        'output_dim': 768,
        'hebbian_learning': True,
        'spike_timing': True,
        'fatigue_rate': 0.001
    }
    
    # 시스템 초기화
    neural_system = NeuralAutonomousSystem(config)
    agent = AutonomousAgent(neural_system)
    
    # 비동기 실행 예시
    async def test_autonomous_thinking():
        # 다양한 지각 입력
        perceptions = [
            {
                'visual': 'bright light',
                'audio': 'soft music',
                'temperature': 22.5,
                'time': 14.30,
                'energy': 0.8
            },
            {
                'visual': 'dark room',
                'audio': 'silence',
                'temperature': 18.0,
                'time': 22.45,
                'energy': 0.3
            },
            {
                'visual': 'complex pattern',
                'audio': 'conversation',
                'temperature': 21.0,
                'time': 10.15,
                'energy': 0.9,
                'social': ['friend', 'colleague']
            }
        ]
        
        for perception in perceptions:
            decision = await agent.think(perception)
            print(f"Perception: {perception}")
            print(f"Decision: {decision}")
            print(f"Introspection: {neural_system.introspect()}")
            print("-" * 50)
        
        # 꿈 모드
        print("\nEntering dream mode...")
        dreams = neural_system.dream_mode(num_cycles=5)
        print(f"Generated {len(dreams)} dream sequences")
        
        # 상태 저장
        save_path = Path("neural_state.pkl")
        neural_system.save_state(save_path)
        print(f"\nNeural state saved to {save_path}")
    
    # asyncio.run(test_autonomous_thinking())