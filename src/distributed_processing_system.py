#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distributed Processing System
분산 처리 및 확장성 시스템
"""

import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import hashlib
import pickle
import json
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import ray
from ray import serve
import dask
from dask.distributed import Client, as_completed
import redis
import aioredis
from kafka import KafkaProducer, KafkaConsumer
import grpc
from concurrent import futures
import threading
import multiprocessing as mp
from collections import defaultdict
import socket
import uuid

logger = logging.getLogger(__name__)


class NodeRole(Enum):
    """노드 역할"""
    MASTER = "master"
    WORKER = "worker"
    ROUTER = "router"
    STORAGE = "storage"
    COMPUTE = "compute"
    INFERENCE = "inference"


class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class ComputeNode:
    """계산 노드"""
    id: str
    role: NodeRole
    host: str
    port: int
    capacity: Dict[str, float] = field(default_factory=dict)  # CPU, GPU, memory
    current_load: Dict[str, float] = field(default_factory=dict)
    status: str = "active"
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedTask:
    """분산 작업"""
    id: str
    type: str
    data: Any
    requirements: Dict[str, float] = field(default_factory=dict)
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


class LoadBalancer:
    """로드 밸런서"""
    
    def __init__(self, strategy: str = "least_loaded"):
        self.strategy = strategy
        self.nodes: Dict[str, ComputeNode] = {}
        self.task_history = defaultdict(list)
        
    def register_node(self, node: ComputeNode):
        """노드 등록"""
        self.nodes[node.id] = node
        logger.info(f"Registered node: {node.id} ({node.role.value})")
    
    def unregister_node(self, node_id: str):
        """노드 등록 해제"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Unregistered node: {node_id}")
    
    def select_node(self, task: DistributedTask) -> Optional[ComputeNode]:
        """작업에 적합한 노드 선택"""
        available_nodes = [
            node for node in self.nodes.values()
            if node.status == "active" and self._can_handle_task(node, task)
        ]
        
        if not available_nodes:
            return None
        
        if self.strategy == "least_loaded":
            return self._select_least_loaded(available_nodes)
        elif self.strategy == "round_robin":
            return self._select_round_robin(available_nodes, task.type)
        elif self.strategy == "capacity_aware":
            return self._select_capacity_aware(available_nodes, task)
        else:
            return available_nodes[0]
    
    def _can_handle_task(self, node: ComputeNode, task: DistributedTask) -> bool:
        """노드가 작업을 처리할 수 있는지 확인"""
        for resource, required in task.requirements.items():
            available = node.capacity.get(resource, 0) - node.current_load.get(resource, 0)
            if available < required:
                return False
        return True
    
    def _select_least_loaded(self, nodes: List[ComputeNode]) -> ComputeNode:
        """가장 부하가 적은 노드 선택"""
        def load_score(node):
            total_capacity = sum(node.capacity.values())
            total_load = sum(node.current_load.values())
            return total_load / total_capacity if total_capacity > 0 else 1.0
        
        return min(nodes, key=load_score)
    
    def _select_round_robin(self, nodes: List[ComputeNode], task_type: str) -> ComputeNode:
        """라운드 로빈 방식 선택"""
        history = self.task_history[task_type]
        if not history:
            selected = nodes[0]
        else:
            last_index = next(
                (i for i, node in enumerate(nodes) if node.id == history[-1]),
                -1
            )
            selected = nodes[(last_index + 1) % len(nodes)]
        
        self.task_history[task_type].append(selected.id)
        return selected
    
    def _select_capacity_aware(self, nodes: List[ComputeNode], task: DistributedTask) -> ComputeNode:
        """용량 인식 선택"""
        scores = []
        
        for node in nodes:
            score = 0
            for resource, required in task.requirements.items():
                available = node.capacity.get(resource, 0) - node.current_load.get(resource, 0)
                score += available / required if required > 0 else 1
            scores.append((score, node))
        
        return max(scores, key=lambda x: x[0])[1]
    
    def update_node_load(self, node_id: str, load: Dict[str, float]):
        """노드 부하 업데이트"""
        if node_id in self.nodes:
            self.nodes[node_id].current_load = load
            self.nodes[node_id].last_heartbeat = time.time()


class DistributedOrchestrator:
    """분산 오케스트레이터"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.node_id = str(uuid.uuid4())
        self.role = NodeRole(config.get('role', 'master'))
        
        # 컴포넌트 초기화
        self.load_balancer = LoadBalancer(
            strategy=config.get('load_balancing_strategy', 'least_loaded')
        )
        
        # 작업 큐
        self.task_queue = asyncio.Queue()
        self.pending_tasks: Dict[str, DistributedTask] = {}
        self.running_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}
        
        # Redis 연결 (상태 공유)
        self.redis_client = None
        self.pubsub = None
        
        # Ray 초기화 (있는 경우)
        self.use_ray = config.get('use_ray', False)
        if self.use_ray:
            try:
                ray.init(address=config.get('ray_address'))
                logger.info("Ray initialized")
            except:
                logger.warning("Ray initialization failed")
                self.use_ray = False
        
        # Dask 클라이언트 (있는 경우)
        self.dask_client = None
        if config.get('use_dask', False):
            try:
                self.dask_client = Client(config.get('dask_scheduler'))
                logger.info("Dask client connected")
            except:
                logger.warning("Dask connection failed")
        
        # 작업 실행기
        self.executors = {}
        self._init_executors()
        
        logger.info(f"Distributed Orchestrator initialized as {self.role.value}")
    
    def _init_executors(self):
        """실행기 초기화"""
        # 로컬 실행기
        self.executors['local'] = LocalExecutor()
        
        # Ray 실행기
        if self.use_ray:
            self.executors['ray'] = RayExecutor()
        
        # Dask 실행기
        if self.dask_client:
            self.executors['dask'] = DaskExecutor(self.dask_client)
    
    async def connect_redis(self, redis_url: str):
        """Redis 연결"""
        self.redis_client = await aioredis.create_redis_pool(redis_url)
        self.pubsub = self.redis_client.pubsub()
        
        # 채널 구독
        await self.pubsub.subscribe('tasks', 'nodes', 'results')
        
        # 노드 등록
        await self._register_self()
        
        logger.info("Connected to Redis")
    
    async def _register_self(self):
        """자신을 노드로 등록"""
        node = ComputeNode(
            id=self.node_id,
            role=self.role,
            host=socket.gethostname(),
            port=self.config.get('port', 8000),
            capacity={
                'cpu': mp.cpu_count(),
                'memory': 16.0,  # GB
                'gpu': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        )
        
        # Redis에 등록
        if self.redis_client:
            await self.redis_client.hset(
                'nodes',
                node.id,
                pickle.dumps(node)
            )
        
        # 로드 밸런서에 등록
        self.load_balancer.register_node(node)
    
    async def submit_task(self, task: DistributedTask) -> str:
        """작업 제출"""
        # 작업 ID 생성
        if not task.id:
            task.id = str(uuid.uuid4())
        
        # 대기 큐에 추가
        self.pending_tasks[task.id] = task
        await self.task_queue.put(task)
        
        # Redis에 발행
        if self.redis_client:
            await self.redis_client.publish(
                'tasks',
                pickle.dumps(task)
            )
        
        logger.info(f"Task submitted: {task.id}")
        return task.id
    
    async def start(self):
        """오케스트레이터 시작"""
        # 작업 처리 워커
        asyncio.create_task(self._task_processor())
        
        # 하트비트 모니터
        asyncio.create_task(self._heartbeat_monitor())
        
        # Redis 메시지 처리
        if self.pubsub:
            asyncio.create_task(self._message_handler())
        
        # 결과 수집기
        asyncio.create_task(self._result_collector())
        
        logger.info("Orchestrator started")
    
    async def _task_processor(self):
        """작업 처리기"""
        while True:
            try:
                # 작업 가져오기
                task = await self.task_queue.get()
                
                # 노드 선택
                node = self.load_balancer.select_node(task)
                
                if not node:
                    # 재시도 큐에 추가
                    await asyncio.sleep(1)
                    await self.task_queue.put(task)
                    continue
                
                # 작업 할당
                task.status = TaskStatus.ASSIGNED
                task.assigned_node = node.id
                self.running_tasks[task.id] = task
                
                # 작업 실행
                if node.id == self.node_id:
                    # 로컬 실행
                    asyncio.create_task(self._execute_local(task))
                else:
                    # 원격 실행
                    await self._execute_remote(task, node)
                
            except Exception as e:
                logger.error(f"Task processing error: {e}")
    
    async def _execute_local(self, task: DistributedTask):
        """로컬 작업 실행"""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        try:
            # 실행기 선택
            executor = self._select_executor(task)
            
            # 작업 실행
            result = await executor.execute(task)
            
            # 성공 처리
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            
            # 결과 발행
            if self.redis_client:
                await self.redis_client.publish(
                    'results',
                    pickle.dumps({
                        'task_id': task.id,
                        'result': result,
                        'node_id': self.node_id
                    })
                )
            
        except Exception as e:
            # 실패 처리
            logger.error(f"Task execution failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            
            # 재시도 확인
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                await self.task_queue.put(task)
        
        finally:
            # 정리
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
            self.completed_tasks[task.id] = task
    
    async def _execute_remote(self, task: DistributedTask, node: ComputeNode):
        """원격 작업 실행"""
        # gRPC 또는 HTTP를 통한 원격 실행
        # 여기서는 Redis를 통한 간단한 구현
        if self.redis_client:
            await self.redis_client.hset(
                f'node_tasks:{node.id}',
                task.id,
                pickle.dumps(task)
            )
    
    def _select_executor(self, task: DistributedTask):
        """작업 유형에 따른 실행기 선택"""
        if task.type == 'ray' and 'ray' in self.executors:
            return self.executors['ray']
        elif task.type == 'dask' and 'dask' in self.executors:
            return self.executors['dask']
        else:
            return self.executors['local']
    
    async def _heartbeat_monitor(self):
        """하트비트 모니터"""
        while True:
            current_time = time.time()
            
            # 노드 상태 확인
            for node in list(self.load_balancer.nodes.values()):
                if current_time - node.last_heartbeat > 30:  # 30초 타임아웃
                    node.status = "inactive"
                    logger.warning(f"Node {node.id} is inactive")
            
            # 자신의 하트비트 전송
            if self.redis_client:
                await self.redis_client.hset(
                    'heartbeats',
                    self.node_id,
                    str(current_time)
                )
            
            await asyncio.sleep(10)
    
    async def _message_handler(self):
        """Redis 메시지 처리"""
        async for message in self.pubsub.listen():
            if message['type'] == 'message':
                channel = message['channel'].decode()
                data = pickle.loads(message['data'])
                
                if channel == 'tasks' and self.role == NodeRole.WORKER:
                    # 새 작업 확인
                    if isinstance(data, DistributedTask):
                        await self.task_queue.put(data)
                
                elif channel == 'nodes':
                    # 노드 업데이트
                    if isinstance(data, ComputeNode):
                        self.load_balancer.register_node(data)
                
                elif channel == 'results':
                    # 결과 처리
                    if isinstance(data, dict) and 'task_id' in data:
                        await self._handle_result(data)
    
    async def _handle_result(self, result_data: Dict[str, Any]):
        """결과 처리"""
        task_id = result_data['task_id']
        
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.result = result_data['result']
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            del self.running_tasks[task_id]
            self.completed_tasks[task_id] = task
    
    async def _result_collector(self):
        """결과 수집기"""
        while True:
            # 완료된 작업 정리
            old_tasks = []
            current_time = time.time()
            
            for task_id, task in self.completed_tasks.items():
                if current_time - task.completed_at > 3600:  # 1시간 후 정리
                    old_tasks.append(task_id)
            
            for task_id in old_tasks:
                del self.completed_tasks[task_id]
            
            await asyncio.sleep(60)
    
    def get_task_status(self, task_id: str) -> Optional[DistributedTask]:
        """작업 상태 조회"""
        if task_id in self.pending_tasks:
            return self.pending_tasks[task_id]
        elif task_id in self.running_tasks:
            return self.running_tasks[task_id]
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        return None
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """클러스터 상태 조회"""
        return {
            'orchestrator_id': self.node_id,
            'role': self.role.value,
            'nodes': len(self.load_balancer.nodes),
            'active_nodes': sum(1 for n in self.load_balancer.nodes.values() if n.status == 'active'),
            'pending_tasks': len(self.pending_tasks),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'executors': list(self.executors.keys())
        }


class LocalExecutor:
    """로컬 실행기"""
    
    async def execute(self, task: DistributedTask) -> Any:
        """작업 실행"""
        # 간단한 시뮬레이션
        if task.type == 'compute':
            # 계산 작업
            data = task.data
            if isinstance(data, dict) and 'operation' in data:
                op = data['operation']
                values = data.get('values', [])
                
                if op == 'sum':
                    return sum(values)
                elif op == 'mean':
                    return np.mean(values)
                elif op == 'matmul' and len(values) == 2:
                    return np.matmul(values[0], values[1]).tolist()
        
        elif task.type == 'ml_inference':
            # ML 추론
            model_data = task.data.get('model')
            input_data = task.data.get('input')
            
            # 시뮬레이션된 추론
            return {'prediction': np.random.randn(10).tolist()}
        
        # 기본 처리
        await asyncio.sleep(1)  # 작업 시뮬레이션
        return f"Processed task {task.id}"


class RayExecutor:
    """Ray 실행기"""
    
    def __init__(self):
        self.ray_tasks = {}
    
    async def execute(self, task: DistributedTask) -> Any:
        """Ray를 사용한 작업 실행"""
        @ray.remote
        def ray_task(data):
            # Ray 작업 함수
            import time
            time.sleep(1)
            return f"Ray processed: {data}"
        
        # Ray 작업 제출
        future = ray_task.remote(task.data)
        
        # 비동기 대기
        result = await asyncio.to_thread(ray.get, future)
        
        return result


class DaskExecutor:
    """Dask 실행기"""
    
    def __init__(self, client: Client):
        self.client = client
    
    async def execute(self, task: DistributedTask) -> Any:
        """Dask를 사용한 작업 실행"""
        def dask_task(data):
            # Dask 작업 함수
            import time
            time.sleep(1)
            return f"Dask processed: {data}"
        
        # Dask 작업 제출
        future = self.client.submit(dask_task, task.data)
        
        # 비동기 대기
        result = await asyncio.to_thread(future.result)
        
        return result


@ray.remote
class RayActor:
    """Ray 액터 - 상태를 가진 분산 객체"""
    
    def __init__(self, actor_id: str):
        self.actor_id = actor_id
        self.state = {}
        self.processed_count = 0
    
    def process(self, data: Any) -> Any:
        """데이터 처리"""
        self.processed_count += 1
        result = f"Actor {self.actor_id} processed item {self.processed_count}"
        self.state[f'item_{self.processed_count}'] = data
        return result
    
    def get_state(self) -> Dict[str, Any]:
        """상태 조회"""
        return {
            'actor_id': self.actor_id,
            'processed_count': self.processed_count,
            'state_size': len(self.state)
        }


# Kubernetes 연동
class KubernetesDeployer:
    """Kubernetes 배포 관리자"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # kubernetes 클라이언트는 별도 설치 필요
        self.api_client = None
    
    def deploy_worker(self, worker_config: Dict[str, Any]) -> str:
        """워커 노드 배포"""
        deployment_spec = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f"worker-{worker_config['id']}",
                'labels': {'app': 'distributed-ai-worker'}
            },
            'spec': {
                'replicas': worker_config.get('replicas', 1),
                'selector': {'matchLabels': {'app': 'distributed-ai-worker'}},
                'template': {
                    'metadata': {'labels': {'app': 'distributed-ai-worker'}},
                    'spec': {
                        'containers': [{
                            'name': 'worker',
                            'image': worker_config['image'],
                            'resources': {
                                'requests': {
                                    'memory': f"{worker_config.get('memory', 4)}Gi",
                                    'cpu': str(worker_config.get('cpu', 2))
                                },
                                'limits': {
                                    'nvidia.com/gpu': str(worker_config.get('gpu', 0))
                                }
                            }
                        }]
                    }
                }
            }
        }
        
        # 실제 배포는 kubernetes 클라이언트 필요
        logger.info(f"Deployment spec created for worker {worker_config['id']}")
        return worker_config['id']


# 사용 예시
if __name__ == "__main__":
    # 마스터 노드 설정
    master_config = {
        'role': 'master',
        'port': 8000,
        'load_balancing_strategy': 'capacity_aware',
        'use_ray': False,  # Ray 설치 필요
        'use_dask': False,  # Dask 설치 필요
    }
    
    # 오케스트레이터 생성
    orchestrator = DistributedOrchestrator(master_config)
    
    # 테스트 실행
    async def test_distributed_processing():
        # 오케스트레이터 시작
        await orchestrator.start()
        
        # 테스트 작업 생성
        tasks = []
        
        # 계산 작업
        compute_task = DistributedTask(
            id=str(uuid.uuid4()),
            type='compute',
            data={
                'operation': 'matmul',
                'values': [
                    np.random.randn(100, 100).tolist(),
                    np.random.randn(100, 100).tolist()
                ]
            },
            requirements={'cpu': 2, 'memory': 1},
            priority=1
        )
        tasks.append(compute_task)
        
        # ML 추론 작업
        ml_task = DistributedTask(
            id=str(uuid.uuid4()),
            type='ml_inference',
            data={
                'model': 'resnet50',
                'input': np.random.randn(1, 3, 224, 224).tolist()
            },
            requirements={'gpu': 1, 'memory': 2},
            priority=2
        )
        tasks.append(ml_task)
        
        # 작업 제출
        task_ids = []
        for task in tasks:
            task_id = await orchestrator.submit_task(task)
            task_ids.append(task_id)
            print(f"Submitted task: {task_id}")
        
        # 결과 대기
        await asyncio.sleep(5)
        
        # 상태 확인
        for task_id in task_ids:
            task = orchestrator.get_task_status(task_id)
            if task:
                print(f"Task {task_id}: {task.status.value}")
                if task.result:
                    print(f"Result: {task.result[:100]}...")  # 결과 일부만 출력
        
        # 클러스터 상태
        cluster_status = orchestrator.get_cluster_status()
        print(f"\nCluster Status: {json.dumps(cluster_status, indent=2)}")
    
    # 실행
    # asyncio.run(test_distributed_processing())