#!/usr/bin/env python3
"""
Agent Coordination Framework for CrewAI

This module provides enhanced agent coordination capabilities including:
- Dynamic agent orchestration
- Task dependency management
- Resource allocation and scheduling
- Inter-agent communication
- Performance monitoring
- Error handling and recovery
- Workflow optimization

Author: Assistant
Date: 2024
Phase: Enhanced Agent Coordination
"""

import os
import sys
import json
import yaml
import asyncio
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, PriorityQueue

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from .tool_registry import ToolRegistry, get_tool_registry, ToolCategory

# Import unified config loader
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class AgentStatus(Enum):
    """Agent status enumeration."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"


class CoordinationStrategy(Enum):
    """Coordination strategy enumeration."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    ADAPTIVE = "adaptive"
    PRIORITY_BASED = "priority_based"


@dataclass
class TaskMetadata:
    """Metadata for coordinated tasks."""
    task_id: str
    name: str
    description: str
    priority: int = 1
    estimated_duration: Optional[float] = None
    required_tools: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    agent_requirements: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    error_message: Optional[str] = None
    result: Optional[Any] = None
    
    def __lt__(self, other):
        """Less than comparison for priority queue ordering."""
        if not isinstance(other, TaskMetadata):
            return NotImplemented
        return self.task_id < other.task_id
    
    def __le__(self, other):
        """Less than or equal comparison."""
        if not isinstance(other, TaskMetadata):
            return NotImplemented
        return self.task_id <= other.task_id
    
    def __gt__(self, other):
        """Greater than comparison."""
        if not isinstance(other, TaskMetadata):
            return NotImplemented
        return self.task_id > other.task_id
    
    def __ge__(self, other):
        """Greater than or equal comparison."""
        if not isinstance(other, TaskMetadata):
            return NotImplemented
        return self.task_id >= other.task_id


@dataclass
class AgentMetadata:
    """Metadata for coordinated agents."""
    agent_id: str
    name: str
    role: str
    capabilities: List[str] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 1
    current_tasks: List[str] = field(default_factory=list)
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_task_duration: float = 0.0
    last_activity: Optional[str] = None
    status: AgentStatus = AgentStatus.IDLE
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    health_score: float = 1.0


@dataclass
class CoordinationPlan:
    """Execution plan for coordinated tasks."""
    plan_id: str
    tasks: List[TaskMetadata]
    strategy: CoordinationStrategy
    estimated_total_duration: float
    resource_allocation: Dict[str, Any]
    agent_assignments: Dict[str, List[str]]  # agent_id -> task_ids
    execution_order: List[List[str]]  # Stages of parallel task groups
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "planned"


class ResourceManager:
    """Manages resources for agent coordination."""
    
    def __init__(self):
        self.resources: Dict[str, Any] = {
            'cpu_cores': os.cpu_count() or 4,
            'memory_gb': 8,  # Default assumption
            'gpu_available': False,
            'max_concurrent_tasks': 10,
            'tool_instances': {}
        }
        self.allocated_resources: Dict[str, Dict[str, Any]] = {}
        self.resource_lock = threading.Lock()
    
    def allocate_resources(self, task_id: str, requirements: Dict[str, Any]) -> bool:
        """Allocate resources for a task."""
        with self.resource_lock:
            # Check if resources are available
            if not self._check_resource_availability(requirements):
                return False
            
            # Allocate resources
            self.allocated_resources[task_id] = requirements
            
            # Update available resources
            for resource, amount in requirements.items():
                if resource in self.resources:
                    self.resources[resource] -= amount
            
            return True
    
    def release_resources(self, task_id: str):
        """Release resources allocated to a task."""
        with self.resource_lock:
            if task_id in self.allocated_resources:
                requirements = self.allocated_resources[task_id]
                
                # Release resources
                for resource, amount in requirements.items():
                    if resource in self.resources:
                        self.resources[resource] += amount
                
                del self.allocated_resources[task_id]
    
    def _check_resource_availability(self, requirements: Dict[str, Any]) -> bool:
        """Check if required resources are available."""
        for resource, amount in requirements.items():
            if resource in self.resources:
                if self.resources[resource] < amount:
                    return False
        return True
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        return {
            'available': self.resources.copy(),
            'allocated': self.allocated_resources.copy(),
            'utilization': self._calculate_utilization()
        }
    
    def _calculate_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization percentages."""
        utilization = {}
        total_allocated = {}
        
        # Sum up allocated resources
        for task_resources in self.allocated_resources.values():
            for resource, amount in task_resources.items():
                total_allocated[resource] = total_allocated.get(resource, 0) + amount
        
        # Calculate utilization percentages
        for resource in self.resources:
            resource_value = self.resources[resource]
            # Skip non-numeric resources like tool_instances dict
            if not isinstance(resource_value, (int, float)):
                utilization[resource] = 0.0
                continue
                
            total = resource_value + total_allocated.get(resource, 0)
            if total > 0:
                utilization[resource] = (total_allocated.get(resource, 0) / total) * 100
            else:
                utilization[resource] = 0.0
        
        return utilization


class TaskScheduler:
    """Schedules and manages task execution."""
    
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.task_queue = PriorityQueue()
        self.running_tasks: Dict[str, Future] = {}
        self.completed_tasks: Dict[str, TaskMetadata] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.scheduler_lock = threading.Lock()
    
    def schedule_task(self, task_metadata: TaskMetadata, agent: Agent, 
                     task: Task) -> bool:
        """Schedule a task for execution."""
        with self.scheduler_lock:
            # Check resource requirements
            if not self.resource_manager.allocate_resources(
                task_metadata.task_id, 
                task_metadata.resource_requirements
            ):
                logger.warning(f"Insufficient resources for task {task_metadata.task_id}")
                return False
            
            # Add to queue with priority (lower number = higher priority)
            priority = -task_metadata.priority  # Negative for max-heap behavior
            self.task_queue.put((priority, task_metadata, agent, task))
            task_metadata.status = TaskStatus.QUEUED
            
            logger.info(f"Task {task_metadata.task_id} scheduled with priority {task_metadata.priority}")
            return True
    
    def execute_next_task(self) -> Optional[str]:
        """Execute the next task in the queue."""
        try:
            if self.task_queue.empty():
                return None
            
            priority, task_metadata, agent, task = self.task_queue.get_nowait()
            
            # Submit task for execution
            future = self.executor.submit(self._execute_task, task_metadata, agent, task)
            self.running_tasks[task_metadata.task_id] = future
            
            task_metadata.status = TaskStatus.RUNNING
            task_metadata.started_at = datetime.now().isoformat()
            
            logger.info(f"Started executing task {task_metadata.task_id}")
            return task_metadata.task_id
            
        except Exception as e:
            logger.error(f"Error executing next task: {e}")
            return None
    
    def _execute_task(self, task_metadata: TaskMetadata, agent: Agent, task: Task) -> Any:
        """Execute a single task."""
        try:
            logger.info(f"Executing task {task_metadata.task_id} with agent {agent.role}")
            
            # Set timeout if specified
            if task_metadata.timeout:
                # Note: CrewAI doesn't have built-in timeout, so we'd need to implement this
                pass
            
            # Execute the task
            result = agent.execute_task(task)
            
            # Update task metadata
            task_metadata.status = TaskStatus.COMPLETED
            task_metadata.completed_at = datetime.now().isoformat()
            task_metadata.result = result
            
            # Move to completed tasks
            self.completed_tasks[task_metadata.task_id] = task_metadata
            
            # Release resources
            self.resource_manager.release_resources(task_metadata.task_id)
            
            logger.info(f"Task {task_metadata.task_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Task {task_metadata.task_id} failed: {e}")
            
            # Update task metadata
            task_metadata.status = TaskStatus.FAILED
            task_metadata.error_message = str(e)
            task_metadata.completed_at = datetime.now().isoformat()
            
            # Handle retry logic
            if task_metadata.retry_count < task_metadata.max_retries:
                task_metadata.retry_count += 1
                task_metadata.status = TaskStatus.RETRYING
                logger.info(f"Retrying task {task_metadata.task_id} (attempt {task_metadata.retry_count})")
                # Re-schedule the task
                self.task_queue.put((-task_metadata.priority, task_metadata, agent, task))
            else:
                self.completed_tasks[task_metadata.task_id] = task_metadata
                self.resource_manager.release_resources(task_metadata.task_id)
            
            raise e
        finally:
            # Remove from running tasks
            if task_metadata.task_id in self.running_tasks:
                del self.running_tasks[task_metadata.task_id]
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a specific task."""
        # Check running tasks
        if task_id in self.running_tasks:
            return TaskStatus.RUNNING
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].status
        
        # Check queue
        # Note: This is inefficient for PriorityQueue, but needed for status checking
        temp_queue = PriorityQueue()
        found_status = None
        
        while not self.task_queue.empty():
            item = self.task_queue.get()
            temp_queue.put(item)
            if item[1].task_id == task_id:
                found_status = item[1].status
        
        # Restore queue
        while not temp_queue.empty():
            self.task_queue.put(temp_queue.get())
        
        return found_status
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        # Cancel running task
        if task_id in self.running_tasks:
            future = self.running_tasks[task_id]
            if future.cancel():
                del self.running_tasks[task_id]
                self.resource_manager.release_resources(task_id)
                return True
        
        # Remove from queue (inefficient but necessary)
        temp_queue = PriorityQueue()
        cancelled = False
        
        while not self.task_queue.empty():
            item = self.task_queue.get()
            if item[1].task_id == task_id:
                item[1].status = TaskStatus.CANCELLED
                self.resource_manager.release_resources(task_id)
                cancelled = True
            else:
                temp_queue.put(item)
        
        # Restore queue
        while not temp_queue.empty():
            self.task_queue.put(temp_queue.get())
        
        return cancelled
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            'queued_tasks': self.task_queue.qsize(),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'resource_status': self.resource_manager.get_resource_status()
        }


class AgentCoordinator:
    """
    Main coordinator class for managing agents and tasks.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the agent coordinator.
        
        Args:
            config_path: Path to coordinator configuration file
        """
        self.config = self._load_config(config_path)
        self.tool_registry = get_tool_registry()
        self.resource_manager = ResourceManager()
        self.task_scheduler = TaskScheduler(self.resource_manager)
        
        # Agent management
        self.agents: Dict[str, Agent] = {}
        self.agent_metadata: Dict[str, AgentMetadata] = {}
        
        # Task management
        self.tasks: Dict[str, TaskMetadata] = {}
        self.coordination_plans: Dict[str, CoordinationPlan] = {}
        
        # Communication and monitoring
        self.message_queue = Queue()
        self.performance_metrics: Dict[str, Any] = {}
        
        # Initialize coordinator
        self._initialize_coordinator()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load coordinator configuration."""
        try:
            config_loader = get_config()
            coordinator_config = config_loader.get_section('coordination', {})
            
            # If coordination section doesn't exist, try to get it from the old location
            if not coordinator_config:
                # Fallback to old config file for backward compatibility
                if config_path is None:
                    config_path = os.path.join(
                        os.path.dirname(__file__), 
                        '../config/coordinator.yaml'
                    )
                
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    return config
                except FileNotFoundError:
                    logger.warning(f"Config file not found: {config_path}. Using defaults.")
                    return self._get_default_config()
            
            return coordinator_config
        except Exception as e:
            logger.warning(f"Could not load coordinator config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default coordinator configuration."""
        return {
            'coordinator': {
                'max_concurrent_tasks': 10,
                'default_task_timeout': 300,  # 5 minutes
                'health_check_interval': 60,  # 1 minute
                'performance_monitoring': True,
                'auto_recovery': True,
                'load_balancing': True
            },
            'strategies': {
                'default': 'adaptive',
                'fallback': 'sequential',
                'optimization_enabled': True
            },
            'agents': {
                'video_researcher': {
                    'max_concurrent_tasks': 2,
                    'capabilities': ['search', 'research'],
                    'tools': ['youtube_search']
                },
                'transcriber': {
                    'max_concurrent_tasks': 1,
                    'capabilities': ['transcription', 'audio_processing'],
                    'tools': ['transcriber']
                },
                'vector_indexer': {
                    'max_concurrent_tasks': 3,
                    'capabilities': ['embedding', 'indexing', 'storage'],
                    'tools': ['chunker', 'embedder', 'faiss_store']
                },
                'content_qa': {
                    'max_concurrent_tasks': 2,
                    'capabilities': ['query', 'generation', 'qa'],
                    'tools': ['faiss_query', 'summarizer', 'answer_generator']
                }
            }
        }
    
    def _initialize_coordinator(self):
        """Initialize the coordinator."""
        logger.info("Initializing agent coordinator...")
        
        # Update resource manager with config
        if 'coordinator' in self.config:
            coord_config = self.config['coordinator']
            self.resource_manager.resources['max_concurrent_tasks'] = coord_config.get(
                'max_concurrent_tasks', 10
            )
        
        logger.info("Agent coordinator initialized")
    
    def register_agent(self, agent_id: str, agent: Agent, 
                      metadata: Optional[AgentMetadata] = None,
                      capabilities: Optional[List[str]] = None, 
                      available_tools: Optional[List[str]] = None) -> bool:
        """Register an agent with the coordinator."""
        try:
            if agent_id in self.agents:
                logger.warning(f"Agent {agent_id} already registered, updating...")
            
            # Create metadata if not provided
            if metadata is None:
                agent_config = self.config.get('agents', {}).get(agent_id, {})
                metadata = AgentMetadata(
                    agent_id=agent_id,
                    name=getattr(agent, 'role', agent_id),
                    role=getattr(agent, 'role', 'Agent'),
                    capabilities=capabilities or agent_config.get('capabilities', []),
                    available_tools=available_tools or agent_config.get('tools', []),
                    max_concurrent_tasks=agent_config.get('max_concurrent_tasks', 1)
                )
            
            self.agents[agent_id] = agent
            self.agent_metadata[agent_id] = metadata
            
            logger.info(f"Agent {agent_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False
    
    def create_task(self, task_id: str, name: str, description: str, 
                   agent_id: str, task: Task, **kwargs) -> bool:
        """Create and register a task."""
        try:
            # Create task metadata
            task_metadata = TaskMetadata(
                task_id=task_id,
                name=name,
                description=description,
                priority=kwargs.get('priority', 1),
                estimated_duration=kwargs.get('estimated_duration'),
                required_tools=kwargs.get('required_tools', []),
                dependencies=kwargs.get('dependencies', []),
                agent_requirements=kwargs.get('agent_requirements', {}),
                resource_requirements=kwargs.get('resource_requirements', {}),
                max_retries=kwargs.get('max_retries', 3),
                timeout=kwargs.get('timeout')
            )
            
            self.tasks[task_id] = task_metadata
            
            # Schedule the task
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                success = self.task_scheduler.schedule_task(task_metadata, agent, task)
                
                if success:
                    # Update agent metadata
                    self.agent_metadata[agent_id].current_tasks.append(task_id)
                    logger.info(f"Created and scheduled task: {task_id}")
                    return True
                else:
                    logger.error(f"Failed to schedule task: {task_id}")
                    return False
            else:
                logger.error(f"Agent not found: {agent_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating task {task_id}: {e}")
            return False
    
    def create_coordination_plan(self, plan_id: str, task_ids: List[str], 
                               strategy: CoordinationStrategy = CoordinationStrategy.ADAPTIVE) -> Optional[CoordinationPlan]:
        """Create a coordination plan for multiple tasks."""
        try:
            # Get task metadata
            plan_tasks = [self.tasks[tid] for tid in task_ids if tid in self.tasks]
            
            if len(plan_tasks) != len(task_ids):
                logger.error("Some tasks not found for coordination plan")
                return None
            
            # Analyze dependencies and create execution plan
            execution_order = self._analyze_task_dependencies(plan_tasks)
            agent_assignments = self._assign_agents_to_tasks(plan_tasks)
            estimated_duration = self._estimate_plan_duration(plan_tasks, execution_order)
            
            # Create coordination plan
            plan = CoordinationPlan(
                plan_id=plan_id,
                tasks=plan_tasks,
                strategy=strategy,
                estimated_total_duration=estimated_duration,
                resource_allocation={},
                agent_assignments=agent_assignments,
                execution_order=execution_order
            )
            
            self.coordination_plans[plan_id] = plan
            logger.info(f"Created coordination plan: {plan_id} with {len(plan_tasks)} tasks")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating coordination plan {plan_id}: {e}")
            return None
    
    def execute_coordination_plan(self, plan_id: str) -> bool:
        """Execute a coordination plan."""
        if plan_id not in self.coordination_plans:
            logger.error(f"Coordination plan not found: {plan_id}")
            return False
        
        plan = self.coordination_plans[plan_id]
        plan.status = "executing"
        
        try:
            if plan.strategy == CoordinationStrategy.SEQUENTIAL:
                return self._execute_sequential_plan(plan)
            elif plan.strategy == CoordinationStrategy.PARALLEL:
                return self._execute_parallel_plan(plan)
            elif plan.strategy == CoordinationStrategy.PIPELINE:
                return self._execute_pipeline_plan(plan)
            elif plan.strategy == CoordinationStrategy.ADAPTIVE:
                return self._execute_adaptive_plan(plan)
            else:
                logger.error(f"Unknown coordination strategy: {plan.strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing coordination plan {plan_id}: {e}")
            plan.status = "failed"
            return False
    
    def _analyze_task_dependencies(self, tasks: List[TaskMetadata]) -> List[List[str]]:
        """Analyze task dependencies and create execution order."""
        # Simple topological sort
        task_map = {task.task_id: task for task in tasks}
        in_degree = {task.task_id: 0 for task in tasks}
        
        # Calculate in-degrees
        for task in tasks:
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[task.task_id] += 1
        
        # Generate execution order
        execution_order = []
        remaining_tasks = set(task.task_id for task in tasks)
        
        while remaining_tasks:
            # Find tasks with no dependencies
            ready_tasks = [tid for tid in remaining_tasks if in_degree[tid] == 0]
            
            if not ready_tasks:
                # Circular dependency or error
                logger.warning("Circular dependency detected, using remaining tasks")
                ready_tasks = list(remaining_tasks)
            
            execution_order.append(ready_tasks)
            
            # Remove ready tasks and update in-degrees
            for tid in ready_tasks:
                remaining_tasks.remove(tid)
                task = task_map[tid]
                
                # Update dependent tasks
                for other_task in tasks:
                    if tid in other_task.dependencies:
                        in_degree[other_task.task_id] -= 1
        
        return execution_order
    
    def _assign_agents_to_tasks(self, tasks: List[TaskMetadata]) -> Dict[str, List[str]]:
        """Assign agents to tasks based on capabilities and load."""
        assignments = {agent_id: [] for agent_id in self.agents.keys()}
        
        for task in tasks:
            # Find suitable agents
            suitable_agents = []
            
            for agent_id, metadata in self.agent_metadata.items():
                # Check if agent has required tools
                if all(tool in metadata.available_tools for tool in task.required_tools):
                    # Check current load
                    current_load = len(metadata.current_tasks)
                    if current_load < metadata.max_concurrent_tasks:
                        suitable_agents.append((agent_id, current_load))
            
            if suitable_agents:
                # Assign to agent with lowest load
                suitable_agents.sort(key=lambda x: x[1])
                chosen_agent = suitable_agents[0][0]
                assignments[chosen_agent].append(task.task_id)
            else:
                logger.warning(f"No suitable agent found for task {task.task_id}")
        
        return assignments
    
    def _estimate_plan_duration(self, tasks: List[TaskMetadata], 
                              execution_order: List[List[str]]) -> float:
        """Estimate total duration for coordination plan."""
        total_duration = 0.0
        
        for stage in execution_order:
            stage_duration = 0.0
            for task_id in stage:
                task = next(t for t in tasks if t.task_id == task_id)
                if task.estimated_duration:
                    stage_duration = max(stage_duration, task.estimated_duration)
                else:
                    # Use default estimate
                    stage_duration = max(stage_duration, 60.0)  # 1 minute default
            
            total_duration += stage_duration
        
        return total_duration
    
    def _execute_sequential_plan(self, plan: CoordinationPlan) -> bool:
        """Execute tasks sequentially."""
        logger.info(f"Executing sequential plan: {plan.plan_id}")
        
        for stage in plan.execution_order:
            for task_id in stage:
                # Execute one task at a time
                executed_task_id = self.task_scheduler.execute_next_task()
                if executed_task_id != task_id:
                    logger.warning(f"Task execution order mismatch: expected {task_id}, got {executed_task_id}")
                
                # Wait for completion
                while self.task_scheduler.get_task_status(task_id) == TaskStatus.RUNNING:
                    import time
                    time.sleep(1)
                
                # Check if task failed
                if self.task_scheduler.get_task_status(task_id) == TaskStatus.FAILED:
                    logger.error(f"Task {task_id} failed, stopping sequential execution")
                    plan.status = "failed"
                    return False
        
        plan.status = "completed"
        return True
    
    def _execute_parallel_plan(self, plan: CoordinationPlan) -> bool:
        """Execute tasks in parallel."""
        logger.info(f"Executing parallel plan: {plan.plan_id}")
        
        # Execute all tasks in parallel
        for stage in plan.execution_order:
            # Start all tasks in the stage
            for task_id in stage:
                self.task_scheduler.execute_next_task()
            
            # Wait for all tasks in stage to complete
            while any(self.task_scheduler.get_task_status(tid) == TaskStatus.RUNNING 
                     for tid in stage):
                import time
                time.sleep(1)
            
            # Check for failures
            failed_tasks = [tid for tid in stage 
                          if self.task_scheduler.get_task_status(tid) == TaskStatus.FAILED]
            
            if failed_tasks:
                logger.error(f"Tasks failed in parallel execution: {failed_tasks}")
                plan.status = "failed"
                return False
        
        plan.status = "completed"
        return True
    
    def _execute_pipeline_plan(self, plan: CoordinationPlan) -> bool:
        """Execute tasks in pipeline mode."""
        logger.info(f"Executing pipeline plan: {plan.plan_id}")
        
        # Pipeline execution: start next stage as soon as previous stage completes
        active_stages = []
        
        for i, stage in enumerate(plan.execution_order):
            # Start current stage
            for task_id in stage:
                self.task_scheduler.execute_next_task()
            
            active_stages.append((i, stage))
            
            # Check if we can start the next stage
            if i < len(plan.execution_order) - 1:
                # Wait for current stage to complete before starting next
                while any(self.task_scheduler.get_task_status(tid) == TaskStatus.RUNNING 
                         for tid in stage):
                    import time
                    time.sleep(0.5)
        
        # Wait for all stages to complete
        all_tasks = [tid for stage in plan.execution_order for tid in stage]
        while any(self.task_scheduler.get_task_status(tid) == TaskStatus.RUNNING 
                 for tid in all_tasks):
            import time
            time.sleep(1)
        
        # Check for failures
        failed_tasks = [tid for tid in all_tasks 
                       if self.task_scheduler.get_task_status(tid) == TaskStatus.FAILED]
        
        if failed_tasks:
            logger.error(f"Tasks failed in pipeline execution: {failed_tasks}")
            plan.status = "failed"
            return False
        
        plan.status = "completed"
        return True
    
    def _execute_adaptive_plan(self, plan: CoordinationPlan) -> bool:
        """Execute tasks using adaptive strategy."""
        logger.info(f"Executing adaptive plan: {plan.plan_id}")
        
        # Adaptive execution: dynamically choose between strategies based on conditions
        resource_status = self.resource_manager.get_resource_status()
        utilization = resource_status['utilization']
        
        # Choose strategy based on resource utilization
        if utilization.get('cpu_cores', 0) < 50:  # Low utilization
            return self._execute_parallel_plan(plan)
        elif len(plan.execution_order) > 3:  # Many stages
            return self._execute_pipeline_plan(plan)
        else:  # Default to sequential
            return self._execute_sequential_plan(plan)
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get comprehensive coordinator status."""
        return {
            'agents': {
                'total': len(self.agents),
                'active': sum(1 for m in self.agent_metadata.values() 
                            if m.status != AgentStatus.OFFLINE),
                'details': {
                    agent_id: {
                        'status': metadata.status.value,
                        'current_tasks': len(metadata.current_tasks),
                        'total_completed': metadata.total_tasks_completed,
                        'health_score': metadata.health_score
                    }
                    for agent_id, metadata in self.agent_metadata.items()
                }
            },
            'tasks': {
                'total': len(self.tasks),
                'by_status': {
                    status.value: sum(1 for t in self.tasks.values() if t.status == status)
                    for status in TaskStatus
                }
            },
            'coordination_plans': {
                'total': len(self.coordination_plans),
                'active': sum(1 for p in self.coordination_plans.values() 
                            if p.status == 'executing')
            },
            'scheduler': self.task_scheduler.get_scheduler_status(),
            'performance': self.performance_metrics
        }
    
    def export_coordination_report(self, output_path: str):
        """Export coordination report to file."""
        report = {
            'report_info': {
                'generated_at': datetime.now().isoformat(),
                'coordinator_version': '1.0.0'
            },
            'status': self.get_coordinator_status(),
            'agents': {
                agent_id: {
                    'metadata': {
                        'name': metadata.name,
                        'role': metadata.role,
                        'capabilities': metadata.capabilities,
                        'available_tools': metadata.available_tools,
                        'status': metadata.status.value
                    },
                    'performance': {
                        'total_tasks_completed': metadata.total_tasks_completed,
                        'total_tasks_failed': metadata.total_tasks_failed,
                        'average_task_duration': metadata.average_task_duration,
                        'health_score': metadata.health_score
                    }
                }
                for agent_id, metadata in self.agent_metadata.items()
            },
            'tasks': {
                task_id: {
                    'metadata': {
                        'name': task.name,
                        'description': task.description,
                        'status': task.status.value,
                        'priority': task.priority
                    },
                    'execution': {
                        'created_at': task.created_at,
                        'started_at': task.started_at,
                        'completed_at': task.completed_at,
                        'retry_count': task.retry_count
                    }
                }
                for task_id, task in self.tasks.items()
            },
            'coordination_plans': {
                plan_id: {
                    'strategy': plan.strategy.value,
                    'status': plan.status,
                    'estimated_duration': plan.estimated_total_duration,
                    'task_count': len(plan.tasks)
                }
                for plan_id, plan in self.coordination_plans.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(report, f, default_flow_style=False, indent=2)
        
        logger.info(f"Coordination report exported to: {output_path}")


# Global coordinator instance
_global_coordinator: Optional[AgentCoordinator] = None


def get_agent_coordinator() -> AgentCoordinator:
    """
    Get the global agent coordinator instance.
    
    Returns:
        Global agent coordinator
    """
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = AgentCoordinator()
    return _global_coordinator


if __name__ == "__main__":
    # Demo usage
    print("Agent Coordinator Demo")
    print("=" * 50)
    
    # Initialize coordinator
    coordinator = AgentCoordinator()
    
    # Print status
    status = coordinator.get_coordinator_status()
    print(f"Total agents: {status['agents']['total']}")
    print(f"Total tasks: {status['tasks']['total']}")
    print(f"Active coordination plans: {status['coordination_plans']['active']}")
    
    # Export report
    output_path = "/tmp/coordination_report.yaml"
    coordinator.export_coordination_report(output_path)
    print(f"\nCoordination report exported to: {output_path}")