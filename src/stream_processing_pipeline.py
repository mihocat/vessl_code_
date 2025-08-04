#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Stream Processing Pipeline
실시간 스트림 처리 파이프라인
"""

import logging
from typing import Dict, List, Any, Optional, Callable, AsyncIterator, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from collections import deque
import numpy as np
import torch
from abc import ABC, abstractmethod
import websockets
import json
import aioredis
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """스트림 상태"""
    IDLE = "idle"
    STREAMING = "streaming"
    BUFFERING = "buffering"
    PROCESSING = "processing"
    ERROR = "error"
    COMPLETED = "completed"


class EventType(Enum):
    """이벤트 타입"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CONTROL = "control"
    METADATA = "metadata"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamEvent:
    """스트림 이벤트"""
    id: str
    type: EventType
    data: Any
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    
    def __lt__(self, other):
        """우선순위 비교"""
        return self.priority > other.priority  # 높은 우선순위가 먼저


class StreamProcessor(ABC):
    """스트림 프로세서 인터페이스"""
    
    @abstractmethod
    async def process(self, event: StreamEvent) -> Optional[StreamEvent]:
        """이벤트 처리"""
        pass
    
    @abstractmethod
    def can_process(self, event: StreamEvent) -> bool:
        """처리 가능 여부"""
        pass


class TextStreamProcessor(StreamProcessor):
    """텍스트 스트림 프로세서"""
    
    def __init__(self, tokenizer=None, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer = []
        
    async def process(self, event: StreamEvent) -> Optional[StreamEvent]:
        if event.type != EventType.TEXT:
            return None
        
        text = event.data
        
        # 토큰화 (있는 경우)
        if self.tokenizer:
            tokens = await asyncio.to_thread(
                self.tokenizer.encode, text, max_length=self.max_length
            )
            processed_data = {
                'text': text,
                'tokens': tokens,
                'token_count': len(tokens)
            }
        else:
            processed_data = {
                'text': text,
                'length': len(text)
            }
        
        return StreamEvent(
            id=f"processed_{event.id}",
            type=EventType.TEXT,
            data=processed_data,
            metadata={**event.metadata, 'processor': 'text'}
        )
    
    def can_process(self, event: StreamEvent) -> bool:
        return event.type == EventType.TEXT


class ImageStreamProcessor(StreamProcessor):
    """이미지 스트림 프로세서"""
    
    def __init__(self, transform_func: Optional[Callable] = None):
        self.transform_func = transform_func
        
    async def process(self, event: StreamEvent) -> Optional[StreamEvent]:
        if event.type != EventType.IMAGE:
            return None
        
        image_data = event.data
        
        # 변환 함수 적용
        if self.transform_func:
            processed_image = await asyncio.to_thread(
                self.transform_func, image_data
            )
        else:
            processed_image = image_data
        
        return StreamEvent(
            id=f"processed_{event.id}",
            type=EventType.IMAGE,
            data=processed_image,
            metadata={**event.metadata, 'processor': 'image'}
        )
    
    def can_process(self, event: StreamEvent) -> bool:
        return event.type == EventType.IMAGE


class StreamBuffer:
    """스트림 버퍼"""
    
    def __init__(self, max_size: int = 1000, max_age_seconds: float = 60.0):
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def add(self, event: StreamEvent):
        """이벤트 추가"""
        with self.lock:
            self.buffer.append(event)
            self._cleanup_old_events()
    
    def get_batch(self, batch_size: int) -> List[StreamEvent]:
        """배치 가져오기"""
        with self.lock:
            batch = []
            for _ in range(min(batch_size, len(self.buffer))):
                if self.buffer:
                    batch.append(self.buffer.popleft())
            return batch
    
    def _cleanup_old_events(self):
        """오래된 이벤트 정리"""
        current_time = time.time()
        while self.buffer:
            if current_time - self.buffer[0].timestamp > self.max_age_seconds:
                self.buffer.popleft()
            else:
                break
    
    def size(self) -> int:
        """버퍼 크기"""
        with self.lock:
            return len(self.buffer)


class StreamPipeline:
    """스트림 처리 파이프라인"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = StreamState.IDLE
        
        # 프로세서 체인
        self.processors: List[StreamProcessor] = []
        
        # 버퍼
        self.input_buffer = StreamBuffer(
            max_size=config.get('buffer_size', 1000),
            max_age_seconds=config.get('buffer_max_age', 60.0)
        )
        self.output_buffer = StreamBuffer(
            max_size=config.get('buffer_size', 1000),
            max_age_seconds=config.get('buffer_max_age', 60.0)
        )
        
        # 우선순위 큐
        self.priority_queue = asyncio.PriorityQueue(maxsize=100)
        
        # 처리 통계
        self.stats = {
            'events_received': 0,
            'events_processed': 0,
            'events_dropped': 0,
            'processing_time_avg': 0.0,
            'throughput': 0.0
        }
        
        # 백프레셔 설정
        self.backpressure_threshold = config.get('backpressure_threshold', 0.8)
        self.is_backpressure_active = False
        
        # 실행기
        self.executor = ThreadPoolExecutor(
            max_workers=config.get('num_workers', 4)
        )
        
        # 처리 태스크
        self.processing_tasks = []
        
        logger.info("Stream Pipeline initialized")
    
    def add_processor(self, processor: StreamProcessor):
        """프로세서 추가"""
        self.processors.append(processor)
        logger.info(f"Added processor: {processor.__class__.__name__}")
    
    async def start(self):
        """파이프라인 시작"""
        if self.state != StreamState.IDLE:
            logger.warning(f"Cannot start pipeline in state: {self.state}")
            return
        
        self.state = StreamState.STREAMING
        logger.info("Stream Pipeline started")
        
        # 처리 워커 시작
        num_workers = self.config.get('num_workers', 4)
        for i in range(num_workers):
            task = asyncio.create_task(self._process_worker(i))
            self.processing_tasks.append(task)
        
        # 통계 업데이트 태스크
        asyncio.create_task(self._update_stats())
        
        # 백프레셔 모니터 태스크
        asyncio.create_task(self._monitor_backpressure())
    
    async def stop(self):
        """파이프라인 중지"""
        self.state = StreamState.COMPLETED
        
        # 모든 태스크 취소
        for task in self.processing_tasks:
            task.cancel()
        
        # 태스크 완료 대기
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        logger.info("Stream Pipeline stopped")
    
    async def input(self, event: StreamEvent):
        """이벤트 입력"""
        if self.state != StreamState.STREAMING:
            logger.warning(f"Pipeline not streaming, dropping event: {event.id}")
            self.stats['events_dropped'] += 1
            return
        
        self.stats['events_received'] += 1
        
        # 백프레셔 확인
        if self.is_backpressure_active:
            if event.priority < 5:  # 낮은 우선순위 이벤트 드롭
                logger.warning(f"Backpressure active, dropping low priority event: {event.id}")
                self.stats['events_dropped'] += 1
                return
        
        # 우선순위 큐에 추가
        try:
            await asyncio.wait_for(
                self.priority_queue.put(event),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            # 큐가 가득 찬 경우 버퍼에 추가
            self.input_buffer.add(event)
    
    async def output(self) -> AsyncIterator[StreamEvent]:
        """처리된 이벤트 출력 (스트림)"""
        while self.state == StreamState.STREAMING:
            # 출력 버퍼에서 배치 가져오기
            batch = self.output_buffer.get_batch(10)
            
            for event in batch:
                yield event
            
            if not batch:
                # 버퍼가 비어있으면 잠시 대기
                await asyncio.sleep(0.01)
    
    async def _process_worker(self, worker_id: int):
        """처리 워커"""
        logger.info(f"Worker {worker_id} started")
        
        while self.state == StreamState.STREAMING:
            try:
                # 우선순위 큐에서 이벤트 가져오기
                event = await asyncio.wait_for(
                    self.priority_queue.get(),
                    timeout=0.1
                )
                
                # 이벤트 처리
                start_time = time.time()
                processed_event = await self._process_event(event)
                processing_time = time.time() - start_time
                
                if processed_event:
                    self.output_buffer.add(processed_event)
                    self.stats['events_processed'] += 1
                    
                    # 처리 시간 업데이트 (이동 평균)
                    alpha = 0.1
                    self.stats['processing_time_avg'] = (
                        alpha * processing_time +
                        (1 - alpha) * self.stats['processing_time_avg']
                    )
                
            except asyncio.TimeoutError:
                # 큐가 비어있으면 입력 버퍼 확인
                if self.input_buffer.size() > 0:
                    batch = self.input_buffer.get_batch(5)
                    for event in batch:
                        await self.priority_queue.put(event)
            
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_event(self, event: StreamEvent) -> Optional[StreamEvent]:
        """단일 이벤트 처리"""
        current_event = event
        
        # 프로세서 체인 통과
        for processor in self.processors:
            if processor.can_process(current_event):
                try:
                    processed = await processor.process(current_event)
                    if processed:
                        current_event = processed
                    else:
                        # 프로세서가 None 반환하면 처리 중단
                        return None
                except Exception as e:
                    logger.error(f"Processor error: {e}")
                    return None
        
        return current_event
    
    async def _update_stats(self):
        """통계 업데이트"""
        last_processed = 0
        last_time = time.time()
        
        while self.state == StreamState.STREAMING:
            await asyncio.sleep(1.0)
            
            current_time = time.time()
            current_processed = self.stats['events_processed']
            
            # 처리량 계산
            time_delta = current_time - last_time
            events_delta = current_processed - last_processed
            
            if time_delta > 0:
                self.stats['throughput'] = events_delta / time_delta
            
            last_processed = current_processed
            last_time = current_time
    
    async def _monitor_backpressure(self):
        """백프레셔 모니터링"""
        while self.state == StreamState.STREAMING:
            # 큐 사용률 확인
            queue_usage = self.priority_queue.qsize() / self.priority_queue.maxsize
            
            if queue_usage > self.backpressure_threshold:
                if not self.is_backpressure_active:
                    logger.warning("Backpressure activated")
                    self.is_backpressure_active = True
            elif queue_usage < self.backpressure_threshold * 0.5:
                if self.is_backpressure_active:
                    logger.info("Backpressure deactivated")
                    self.is_backpressure_active = False
            
            await asyncio.sleep(0.5)
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        return {
            **self.stats,
            'state': self.state.value,
            'input_buffer_size': self.input_buffer.size(),
            'output_buffer_size': self.output_buffer.size(),
            'priority_queue_size': self.priority_queue.qsize(),
            'backpressure_active': self.is_backpressure_active,
            'num_processors': len(self.processors)
        }


class WebSocketStreamServer:
    """WebSocket 스트림 서버"""
    
    def __init__(self, pipeline: StreamPipeline, host: str = 'localhost', port: int = 8765):
        self.pipeline = pipeline
        self.host = host
        self.port = port
        self.clients = set()
        
    async def handler(self, websocket, path):
        """WebSocket 핸들러"""
        # 클라이언트 추가
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        
        try:
            # 입력 처리 태스크
            input_task = asyncio.create_task(self._handle_input(websocket))
            
            # 출력 처리 태스크
            output_task = asyncio.create_task(self._handle_output(websocket))
            
            # 둘 중 하나가 완료될 때까지 대기
            await asyncio.gather(input_task, output_task)
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        finally:
            self.clients.remove(websocket)
    
    async def _handle_input(self, websocket):
        """입력 처리"""
        async for message in websocket:
            try:
                # JSON 파싱
                data = json.loads(message)
                
                # StreamEvent 생성
                event = StreamEvent(
                    id=data.get('id', str(time.time())),
                    type=EventType(data.get('type', 'text')),
                    data=data.get('data'),
                    metadata=data.get('metadata', {}),
                    priority=data.get('priority', 0)
                )
                
                # 파이프라인에 입력
                await self.pipeline.input(event)
                
            except Exception as e:
                logger.error(f"Input handling error: {e}")
                await websocket.send(json.dumps({
                    'error': str(e),
                    'type': 'error'
                }))
    
    async def _handle_output(self, websocket):
        """출력 처리"""
        async for event in self.pipeline.output():
            try:
                # JSON 직렬화
                message = json.dumps({
                    'id': event.id,
                    'type': event.type.value,
                    'data': event.data if isinstance(event.data, (str, int, float, list, dict)) else str(event.data),
                    'metadata': event.metadata,
                    'timestamp': event.timestamp
                })
                
                # 클라이언트로 전송
                await websocket.send(message)
                
            except Exception as e:
                logger.error(f"Output handling error: {e}")
    
    async def start(self):
        """서버 시작"""
        logger.info(f"WebSocket server starting on {self.host}:{self.port}")
        
        # 파이프라인 시작
        await self.pipeline.start()
        
        # WebSocket 서버 시작
        async with websockets.serve(self.handler, self.host, self.port):
            await asyncio.Future()  # 무한 대기


class StreamAggregator(StreamProcessor):
    """스트림 집계 프로세서"""
    
    def __init__(self, window_size: int = 10, aggregation_func: str = 'mean'):
        self.window_size = window_size
        self.aggregation_func = aggregation_func
        self.windows = defaultdict(deque)
        
    async def process(self, event: StreamEvent) -> Optional[StreamEvent]:
        # 윈도우에 추가
        window_key = f"{event.type.value}_{event.metadata.get('source', 'default')}"
        window = self.windows[window_key]
        
        # 윈도우 크기 제한
        if len(window) >= self.window_size:
            window.popleft()
        
        window.append(event.data)
        
        # 집계 수행
        if len(window) >= self.window_size:
            if self.aggregation_func == 'mean' and all(isinstance(d, (int, float)) for d in window):
                aggregated = np.mean(list(window))
            elif self.aggregation_func == 'sum' and all(isinstance(d, (int, float)) for d in window):
                aggregated = np.sum(list(window))
            else:
                aggregated = list(window)
            
            return StreamEvent(
                id=f"agg_{event.id}",
                type=event.type,
                data={
                    'aggregated': aggregated,
                    'window_size': len(window),
                    'function': self.aggregation_func
                },
                metadata={**event.metadata, 'processor': 'aggregator'}
            )
        
        return None
    
    def can_process(self, event: StreamEvent) -> bool:
        return True  # 모든 이벤트 처리 가능


# 사용 예시
if __name__ == "__main__":
    import uuid
    
    # 파이프라인 설정
    config = {
        'buffer_size': 1000,
        'buffer_max_age': 60.0,
        'num_workers': 4,
        'backpressure_threshold': 0.8
    }
    
    # 파이프라인 생성
    pipeline = StreamPipeline(config)
    
    # 프로세서 추가
    pipeline.add_processor(TextStreamProcessor())
    pipeline.add_processor(ImageStreamProcessor())
    pipeline.add_processor(StreamAggregator(window_size=5))
    
    # 테스트 실행
    async def test_streaming():
        # 파이프라인 시작
        await pipeline.start()
        
        # 출력 처리 태스크
        async def consume_output():
            async for event in pipeline.output():
                print(f"Output: {event.id} - {event.type.value} - {event.data}")
        
        output_task = asyncio.create_task(consume_output())
        
        # 테스트 이벤트 생성
        for i in range(20):
            event_type = EventType.TEXT if i % 2 == 0 else EventType.IMAGE
            event = StreamEvent(
                id=str(uuid.uuid4()),
                type=event_type,
                data=f"Test data {i}" if event_type == EventType.TEXT else f"Image_{i}.jpg",
                priority=i % 3
            )
            
            await pipeline.input(event)
            await asyncio.sleep(0.1)
        
        # 통계 출력
        stats = pipeline.get_stats()
        print(f"\nPipeline Stats: {json.dumps(stats, indent=2)}")
        
        # 정리
        await asyncio.sleep(2)
        await pipeline.stop()
        output_task.cancel()
    
    # WebSocket 서버 테스트
    async def test_websocket_server():
        # 파이프라인 생성
        pipeline = StreamPipeline(config)
        pipeline.add_processor(TextStreamProcessor())
        
        # WebSocket 서버 생성
        server = WebSocketStreamServer(pipeline)
        
        # 서버 시작
        await server.start()
    
    # 실행 선택
    # asyncio.run(test_streaming())
    # asyncio.run(test_websocket_server())