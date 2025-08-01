#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Components for Enhanced UI
향상된 UI를 위한 시각화 컴포넌트
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import numpy as np
import io
import base64
from PIL import Image
import json

logger = logging.getLogger(__name__)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class CircuitDiagramGenerator:
    """회로도 생성기"""
    
    def __init__(self):
        """회로도 생성기 초기화"""
        self.component_symbols = {
            'resistor': self._draw_resistor,
            'capacitor': self._draw_capacitor,
            'inductor': self._draw_inductor,
            'voltage_source': self._draw_voltage_source,
            'current_source': self._draw_current_source,
            'ground': self._draw_ground,
            'switch': self._draw_switch,
            'diode': self._draw_diode,
            'transistor': self._draw_transistor
        }
    
    def generate_circuit(self, components: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> Image.Image:
        """
        회로도 생성
        
        Args:
            components: 컴포넌트 리스트 [{'type': 'resistor', 'label': 'R1', 'position': (x, y)}]
            connections: 연결 리스트 [{'from': (x1, y1), 'to': (x2, y2)}]
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 그리드 표시 (선택적)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 연결선 그리기
        for conn in connections:
            self._draw_wire(ax, conn['from'], conn['to'])
        
        # 컴포넌트 그리기
        for comp in components:
            comp_type = comp.get('type', 'resistor')
            position = comp.get('position', (5, 4))
            label = comp.get('label', '')
            orientation = comp.get('orientation', 'horizontal')
            
            if comp_type in self.component_symbols:
                self.component_symbols[comp_type](ax, position, label, orientation)
        
        # 이미지로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img
    
    def _draw_wire(self, ax, start: Tuple[float, float], end: Tuple[float, float]):
        """전선 그리기"""
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2)
    
    def _draw_resistor(self, ax, pos: Tuple[float, float], label: str, orientation: str):
        """저항 그리기"""
        x, y = pos
        if orientation == 'horizontal':
            # 지그재그 패턴
            zigzag_x = np.linspace(x - 0.3, x + 0.3, 7)
            zigzag_y = [y, y + 0.1, y - 0.1, y + 0.1, y - 0.1, y + 0.1, y]
            ax.plot(zigzag_x, zigzag_y, 'k-', linewidth=2)
            # 연결 단자
            ax.plot([x - 0.5, x - 0.3], [y, y], 'k-', linewidth=2)
            ax.plot([x + 0.3, x + 0.5], [y, y], 'k-', linewidth=2)
        else:  # vertical
            zigzag_y = np.linspace(y - 0.3, y + 0.3, 7)
            zigzag_x = [x, x + 0.1, x - 0.1, x + 0.1, x - 0.1, x + 0.1, x]
            ax.plot(zigzag_x, zigzag_y, 'k-', linewidth=2)
            ax.plot([x, x], [y - 0.5, y - 0.3], 'k-', linewidth=2)
            ax.plot([x, x], [y + 0.3, y + 0.5], 'k-', linewidth=2)
        
        # 라벨
        if label:
            ax.text(x, y - 0.3, label, ha='center', va='top', fontsize=10)
    
    def _draw_capacitor(self, ax, pos: Tuple[float, float], label: str, orientation: str):
        """커패시터 그리기"""
        x, y = pos
        if orientation == 'horizontal':
            # 두 평행선
            ax.plot([x - 0.05, x - 0.05], [y - 0.2, y + 0.2], 'k-', linewidth=3)
            ax.plot([x + 0.05, x + 0.05], [y - 0.2, y + 0.2], 'k-', linewidth=3)
            # 연결 단자
            ax.plot([x - 0.5, x - 0.05], [y, y], 'k-', linewidth=2)
            ax.plot([x + 0.05, x + 0.5], [y, y], 'k-', linewidth=2)
        else:  # vertical
            ax.plot([x - 0.2, x + 0.2], [y - 0.05, y - 0.05], 'k-', linewidth=3)
            ax.plot([x - 0.2, x + 0.2], [y + 0.05, y + 0.05], 'k-', linewidth=3)
            ax.plot([x, x], [y - 0.5, y - 0.05], 'k-', linewidth=2)
            ax.plot([x, x], [y + 0.05, y + 0.5], 'k-', linewidth=2)
        
        if label:
            ax.text(x, y - 0.3, label, ha='center', va='top', fontsize=10)
    
    def _draw_inductor(self, ax, pos: Tuple[float, float], label: str, orientation: str):
        """인덕터 그리기"""
        x, y = pos
        if orientation == 'horizontal':
            # 코일 모양
            t = np.linspace(0, 4*np.pi, 100)
            coil_x = x + 0.3 * t / (4*np.pi) - 0.15
            coil_y = y + 0.05 * np.sin(t)
            ax.plot(coil_x, coil_y, 'k-', linewidth=2)
            # 연결 단자
            ax.plot([x - 0.5, x - 0.15], [y, y], 'k-', linewidth=2)
            ax.plot([x + 0.15, x + 0.5], [y, y], 'k-', linewidth=2)
        else:  # vertical
            t = np.linspace(0, 4*np.pi, 100)
            coil_y = y + 0.3 * t / (4*np.pi) - 0.15
            coil_x = x + 0.05 * np.sin(t)
            ax.plot(coil_x, coil_y, 'k-', linewidth=2)
            ax.plot([x, x], [y - 0.5, y - 0.15], 'k-', linewidth=2)
            ax.plot([x, x], [y + 0.15, y + 0.5], 'k-', linewidth=2)
        
        if label:
            ax.text(x, y - 0.3, label, ha='center', va='top', fontsize=10)
    
    def _draw_voltage_source(self, ax, pos: Tuple[float, float], label: str, orientation: str):
        """전압원 그리기"""
        x, y = pos
        # 원 그리기
        circle = Circle((x, y), 0.2, fill=False, linewidth=2)
        ax.add_patch(circle)
        
        if orientation == 'horizontal':
            # + 기호
            ax.text(x - 0.1, y, '+', ha='center', va='center', fontsize=12, weight='bold')
            # - 기호
            ax.text(x + 0.1, y, '-', ha='center', va='center', fontsize=12, weight='bold')
            # 연결 단자
            ax.plot([x - 0.5, x - 0.2], [y, y], 'k-', linewidth=2)
            ax.plot([x + 0.2, x + 0.5], [y, y], 'k-', linewidth=2)
        else:  # vertical
            ax.text(x, y + 0.1, '+', ha='center', va='center', fontsize=12, weight='bold')
            ax.text(x, y - 0.1, '-', ha='center', va='center', fontsize=12, weight='bold')
            ax.plot([x, x], [y - 0.5, y - 0.2], 'k-', linewidth=2)
            ax.plot([x, x], [y + 0.2, y + 0.5], 'k-', linewidth=2)
        
        if label:
            ax.text(x, y - 0.4, label, ha='center', va='top', fontsize=10)
    
    def _draw_current_source(self, ax, pos: Tuple[float, float], label: str, orientation: str):
        """전류원 그리기"""
        x, y = pos
        # 원 그리기
        circle = Circle((x, y), 0.2, fill=False, linewidth=2)
        ax.add_patch(circle)
        
        # 화살표
        if orientation == 'horizontal':
            arrow = FancyArrowPatch((x - 0.15, y), (x + 0.15, y),
                                  arrowstyle='->', mutation_scale=20, linewidth=2)
            ax.add_patch(arrow)
            # 연결 단자
            ax.plot([x - 0.5, x - 0.2], [y, y], 'k-', linewidth=2)
            ax.plot([x + 0.2, x + 0.5], [y, y], 'k-', linewidth=2)
        else:  # vertical
            arrow = FancyArrowPatch((x, y - 0.15), (x, y + 0.15),
                                  arrowstyle='->', mutation_scale=20, linewidth=2)
            ax.add_patch(arrow)
            ax.plot([x, x], [y - 0.5, y - 0.2], 'k-', linewidth=2)
            ax.plot([x, x], [y + 0.2, y + 0.5], 'k-', linewidth=2)
        
        if label:
            ax.text(x, y - 0.4, label, ha='center', va='top', fontsize=10)
    
    def _draw_ground(self, ax, pos: Tuple[float, float], label: str, orientation: str):
        """접지 그리기"""
        x, y = pos
        # 세 개의 수평선
        ax.plot([x - 0.2, x + 0.2], [y, y], 'k-', linewidth=2)
        ax.plot([x - 0.15, x + 0.15], [y - 0.1, y - 0.1], 'k-', linewidth=2)
        ax.plot([x - 0.1, x + 0.1], [y - 0.2, y - 0.2], 'k-', linewidth=2)
        # 수직 연결선
        ax.plot([x, x], [y, y + 0.3], 'k-', linewidth=2)
        
        if label:
            ax.text(x + 0.3, y, label, ha='left', va='center', fontsize=10)
    
    def _draw_switch(self, ax, pos: Tuple[float, float], label: str, orientation: str):
        """스위치 그리기"""
        x, y = pos
        if orientation == 'horizontal':
            # 접점
            ax.plot(x - 0.2, y, 'ko', markersize=8)
            ax.plot(x + 0.2, y, 'ko', markersize=8)
            # 스위치 암
            ax.plot([x - 0.2, x + 0.15], [y, y + 0.15], 'k-', linewidth=2)
            # 연결 단자
            ax.plot([x - 0.5, x - 0.2], [y, y], 'k-', linewidth=2)
            ax.plot([x + 0.2, x + 0.5], [y, y], 'k-', linewidth=2)
        
        if label:
            ax.text(x, y - 0.3, label, ha='center', va='top', fontsize=10)
    
    def _draw_diode(self, ax, pos: Tuple[float, float], label: str, orientation: str):
        """다이오드 그리기"""
        x, y = pos
        if orientation == 'horizontal':
            # 삼각형
            triangle = np.array([
                [x - 0.15, y - 0.15],
                [x - 0.15, y + 0.15],
                [x + 0.15, y]
            ])
            triangle_patch = plt.Polygon(triangle, fill=False, linewidth=2)
            ax.add_patch(triangle_patch)
            # 막대
            ax.plot([x + 0.15, x + 0.15], [y - 0.15, y + 0.15], 'k-', linewidth=2)
            # 연결 단자
            ax.plot([x - 0.5, x - 0.15], [y, y], 'k-', linewidth=2)
            ax.plot([x + 0.15, x + 0.5], [y, y], 'k-', linewidth=2)
        
        if label:
            ax.text(x, y - 0.3, label, ha='center', va='top', fontsize=10)
    
    def _draw_transistor(self, ax, pos: Tuple[float, float], label: str, orientation: str):
        """트랜지스터 그리기 (NPN)"""
        x, y = pos
        # 원 그리기
        circle = Circle((x, y), 0.25, fill=False, linewidth=2)
        ax.add_patch(circle)
        
        # 베이스 라인
        ax.plot([x - 0.1, x - 0.1], [y - 0.15, y + 0.15], 'k-', linewidth=3)
        
        # 이미터 (화살표 포함)
        ax.plot([x - 0.1, x + 0.1], [y - 0.1, y - 0.2], 'k-', linewidth=2)
        arrow = FancyArrowPatch((x + 0.05, y - 0.17), (x + 0.1, y - 0.2),
                              arrowstyle='->', mutation_scale=15, linewidth=2)
        ax.add_patch(arrow)
        
        # 컬렉터
        ax.plot([x - 0.1, x + 0.1], [y + 0.1, y + 0.2], 'k-', linewidth=2)
        
        # 연결 단자
        ax.plot([x - 0.25, x - 0.35], [y, y], 'k-', linewidth=2)  # 베이스
        ax.plot([x + 0.1, x + 0.1], [y - 0.2, y - 0.35], 'k-', linewidth=2)  # 이미터
        ax.plot([x + 0.1, x + 0.1], [y + 0.2, y + 0.35], 'k-', linewidth=2)  # 컬렉터
        
        if label:
            ax.text(x + 0.3, y, label, ha='left', va='center', fontsize=10)


class FormulaRenderer:
    """수식 렌더러"""
    
    @staticmethod
    def render_latex(formula: str, fontsize: int = 16) -> Image.Image:
        """LaTeX 수식을 이미지로 렌더링"""
        try:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.text(0.5, 0.5, f'${formula}$', 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   fontsize=fontsize)
            ax.axis('off')
            
            # 이미지로 변환
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            img = Image.open(buf)
            plt.close()
            
            return img
            
        except Exception as e:
            logger.error(f"Failed to render LaTeX: {e}")
            # 폴백: 텍스트 이미지
            return FormulaRenderer._render_text_fallback(formula)
    
    @staticmethod
    def _render_text_fallback(text: str) -> Image.Image:
        """텍스트 폴백 렌더링"""
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, text,
               horizontalalignment='center',
               verticalalignment='center',
               transform=ax.transAxes,
               fontsize=14,
               family='monospace')
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img


class GraphPlotter:
    """그래프 플로터"""
    
    @staticmethod
    def plot_function(
        x_data: np.ndarray,
        y_data: np.ndarray,
        title: str = "",
        xlabel: str = "x",
        ylabel: str = "y",
        grid: bool = True
    ) -> Image.Image:
        """함수 그래프 플롯"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(x_data, y_data, 'b-', linewidth=2)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        if grid:
            ax.grid(True, alpha=0.3)
        
        # 이미지로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img
    
    @staticmethod
    def plot_bode(
        frequencies: np.ndarray,
        magnitude_db: np.ndarray,
        phase_deg: np.ndarray,
        title: str = "Bode Plot"
    ) -> Image.Image:
        """보드 선도 플롯"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        
        # 크기 플롯
        ax1.semilogx(frequencies, magnitude_db, 'b-', linewidth=2)
        ax1.set_title(title, fontsize=14)
        ax1.set_ylabel('Magnitude (dB)', fontsize=12)
        ax1.grid(True, which='both', alpha=0.3)
        
        # 위상 플롯
        ax2.semilogx(frequencies, phase_deg, 'r-', linewidth=2)
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Phase (degrees)', fontsize=12)
        ax2.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        
        # 이미지로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img
    
    @staticmethod
    def plot_phasor(
        phasors: List[Dict[str, Any]],
        title: str = "Phasor Diagram"
    ) -> Image.Image:
        """페이저 다이어그램 플롯"""
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        colors = ['b', 'r', 'g', 'm', 'c', 'y']
        
        for i, phasor in enumerate(phasors):
            magnitude = phasor.get('magnitude', 1)
            angle_deg = phasor.get('angle', 0)
            label = phasor.get('label', f'V{i+1}')
            
            angle_rad = np.radians(angle_deg)
            color = colors[i % len(colors)]
            
            # 페이저 화살표
            ax.arrow(0, 0, angle_rad, magnitude, 
                    head_width=0.1, head_length=magnitude*0.1,
                    fc=color, ec=color, linewidth=2)
            
            # 라벨
            ax.text(angle_rad, magnitude*1.1, label, 
                   ha='center', va='center', fontsize=10)
        
        ax.set_title(title, fontsize=14, pad=20)
        ax.grid(True)
        
        # 이미지로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img


class VisualizationManager:
    """시각화 관리자"""
    
    def __init__(self):
        """시각화 관리자 초기화"""
        self.circuit_generator = CircuitDiagramGenerator()
        self.formula_renderer = FormulaRenderer()
        self.graph_plotter = GraphPlotter()
        
        logger.info("Visualization manager initialized")
    
    def create_visualization(self, viz_type: str, data: Dict[str, Any]) -> Optional[Image.Image]:
        """
        시각화 생성
        
        Args:
            viz_type: 시각화 타입 ('circuit', 'formula', 'graph', 'bode', 'phasor')
            data: 시각화 데이터
        """
        try:
            if viz_type == 'circuit':
                return self.circuit_generator.generate_circuit(
                    components=data.get('components', []),
                    connections=data.get('connections', [])
                )
            
            elif viz_type == 'formula':
                return self.formula_renderer.render_latex(
                    formula=data.get('formula', ''),
                    fontsize=data.get('fontsize', 16)
                )
            
            elif viz_type == 'graph':
                return self.graph_plotter.plot_function(
                    x_data=np.array(data.get('x', [])),
                    y_data=np.array(data.get('y', [])),
                    title=data.get('title', ''),
                    xlabel=data.get('xlabel', 'x'),
                    ylabel=data.get('ylabel', 'y')
                )
            
            elif viz_type == 'bode':
                return self.graph_plotter.plot_bode(
                    frequencies=np.array(data.get('frequencies', [])),
                    magnitude_db=np.array(data.get('magnitude_db', [])),
                    phase_deg=np.array(data.get('phase_deg', [])),
                    title=data.get('title', 'Bode Plot')
                )
            
            elif viz_type == 'phasor':
                return self.graph_plotter.plot_phasor(
                    phasors=data.get('phasors', []),
                    title=data.get('title', 'Phasor Diagram')
                )
            
            else:
                logger.warning(f"Unknown visualization type: {viz_type}")
                return None
                
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return None
    
    def image_to_base64(self, img: Image.Image) -> str:
        """이미지를 base64 문자열로 변환"""
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    
    def create_example_visualizations(self) -> Dict[str, Image.Image]:
        """예제 시각화 생성"""
        examples = {}
        
        # 1. RLC 회로
        examples['rlc_circuit'] = self.create_visualization('circuit', {
            'components': [
                {'type': 'voltage_source', 'label': 'V', 'position': (2, 4)},
                {'type': 'resistor', 'label': 'R', 'position': (4, 4)},
                {'type': 'inductor', 'label': 'L', 'position': (6, 4)},
                {'type': 'capacitor', 'label': 'C', 'position': (8, 4)},
                {'type': 'ground', 'label': 'GND', 'position': (5, 2)}
            ],
            'connections': [
                {'from': (2.5, 4), 'to': (3.5, 4)},
                {'from': (4.5, 4), 'to': (5.5, 4)},
                {'from': (6.5, 4), 'to': (7.5, 4)},
                {'from': (8.5, 4), 'to': (8.5, 2)},
                {'from': (8.5, 2), 'to': (5.3, 2)},
                {'from': (4.7, 2), 'to': (1.5, 2)},
                {'from': (1.5, 2), 'to': (1.5, 4)}
            ]
        })
        
        # 2. 전력 공식
        examples['power_formula'] = self.create_visualization('formula', {
            'formula': r'P = \sqrt{3} \times V_L \times I_L \times \cos\theta'
        })
        
        # 3. 사인파
        x = np.linspace(0, 2*np.pi, 1000)
        examples['sine_wave'] = self.create_visualization('graph', {
            'x': x,
            'y': np.sin(x),
            'title': 'Sine Wave',
            'xlabel': 'Time (s)',
            'ylabel': 'Amplitude (V)'
        })
        
        return examples


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    viz_manager = VisualizationManager()
    
    # 예제 시각화 생성
    examples = viz_manager.create_example_visualizations()
    
    # 저장
    for name, img in examples.items():
        if img:
            img.save(f"test_{name}.png")
            logger.info(f"Saved test_{name}.png")