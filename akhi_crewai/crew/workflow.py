#!/usr/bin/env python3
"""
Workflow Orchestration for Akhi Pipeline Crew - Phase 4

Advanced workflow management with:
- Task dependency handling
- Parallel processing capabilities
- Error recovery and retry logic
- Progress monitoring and checkpointing
- Resource management

Author: Assistant
Date: December 2024
Phase: 4 - Crew Orchestration
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Callable, Union

from crewai import Crew, Process
from pydantic import BaseModel, Field

from .tasks import (
    BaseTaskDefinition, 
    TaskResult, 
    VideoSearchTask, 
    TranscriptionTask, 
    IndexingTask, 
    QATask,
    TaskOrchestrator
)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskDependency(BaseModel):
    """Task dependency definition."""
    task_id: str
    depends_on: List[str] = Field(default_factory=list)
    required_status: str = "completed"  # Status required from dependencies
    failure_action: str = "stop"  # "stop", "skip", "retry"


class RetryPolicy(BaseModel):
    """Retry policy for failed tasks."""
    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    backoff_multiplier: float = 2.0
    max_delay: float = 300.0  # 5 minutes
    retry_on_errors: List[str] = Field(default_factory=lambda: ["timeout", "network", "temporary"])


class CheckpointData(BaseModel):
    """Workflow checkpoint data."""
    workflow_id: str
    timestamp: str
    status: str
    completed_tasks: List[str]
    failed_tasks: List[str]
    current_task: Optional[str] = None
    task_results: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowMetrics(BaseModel):
    """Workflow execution metrics."""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    total_duration: Optional[float] = None
    task_count: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    retried_tasks: int = 0
    average_task_duration: Optional[float] = None
    resource_usage: Dict[str, Any] = Field(default_factory=dict)


class WorkflowOrchestrator:
    """
    Advanced workflow orchestrator for the Akhi Pipeline.
    
    Features:
    - Task dependency management
    - Parallel execution where possible
    - Error recovery and retry logic
    - Progress monitoring and checkpointing
    - Resource management and optimization
    """
    
    def __init__(
        self, 
        workflow_id: str,
        output_dir: Path,
        config: Dict[str, Any] = None,
        max_workers: int = 4
    ):
        """
        Initialize the workflow orchestrator.
        
        Args:
            workflow_id: Unique identifier for this workflow
            output_dir: Directory for workflow outputs
            config: Workflow configuration
            max_workers: Maximum number of parallel workers
        """
        self.workflow_id = workflow_id
        self.output_dir = output_dir
        self.config = config or {}
        self.max_workers = max_workers
        
        # Initialize components
        self.logger = logging.getLogger(f'Workflow.{workflow_id}')
        self.task_orchestrator = TaskOrchestrator(output_dir)
        
        # Workflow state
        self.status = WorkflowStatus.PENDING
        self.dependencies: Dict[str, TaskDependency] = {}
        self.retry_policies: Dict[str, RetryPolicy] = {}
        self.metrics = WorkflowMetrics()
        
        # Execution tracking
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.running_tasks: Set[str] = set()
        self.task_attempts: Dict[str, int] = {}
        
        # Checkpointing
        self.checkpoint_interval = 30  # seconds
        self.last_checkpoint = None
        self.checkpoint_file = output_dir / f"workflow_{workflow_id}_checkpoint.json"
        
        # Callbacks
        self.progress_callbacks: List[Callable] = []
        self.completion_callbacks: List[Callable] = []
        
        self.logger.info(f"Workflow orchestrator initialized: {workflow_id}")
    
    def add_task_dependency(self, task_id: str, depends_on: List[str], **kwargs):
        """
        Add task dependency.
        
        Args:
            task_id: ID of the dependent task
            depends_on: List of task IDs this task depends on
            **kwargs: Additional dependency options
        """
        self.dependencies[task_id] = TaskDependency(
            task_id=task_id,
            depends_on=depends_on,
            **kwargs
        )
        self.logger.info(f"Added dependency: {task_id} depends on {depends_on}")
    
    def set_retry_policy(self, task_id: str, policy: RetryPolicy):
        """
        Set retry policy for a task.
        
        Args:
            task_id: Task ID
            policy: Retry policy
        """
        self.retry_policies[task_id] = policy
        self.logger.info(f"Set retry policy for {task_id}: {policy.max_attempts} attempts")
    
    def add_progress_callback(self, callback: Callable):
        """
        Add progress callback function.
        
        Args:
            callback: Function to call on progress updates
        """
        self.progress_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable):
        """
        Add completion callback function.
        
        Args:
            callback: Function to call on workflow completion
        """
        self.completion_callbacks.append(callback)
    
    def _check_dependencies(self, task_id: str) -> bool:
        """
        Check if task dependencies are satisfied.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            True if dependencies are satisfied
        """
        if task_id not in self.dependencies:
            return True  # No dependencies
        
        dependency = self.dependencies[task_id]
        
        for dep_task_id in dependency.depends_on:
            if dep_task_id not in self.completed_tasks:
                return False
            
            # Check if dependency completed successfully
            dep_result = self.task_orchestrator.tasks.get(dep_task_id)
            if dep_result and dep_result.result.status != dependency.required_status:
                return False
        
        return True
    
    def _get_ready_tasks(self) -> List[str]:
        """
        Get list of tasks ready for execution.
        
        Returns:
            List of task IDs ready to run
        """
        ready_tasks = []
        
        for task_id in self.task_orchestrator.tasks:
            if (
                task_id not in self.completed_tasks and
                task_id not in self.failed_tasks and
                task_id not in self.running_tasks and
                self._check_dependencies(task_id)
            ):
                ready_tasks.append(task_id)
        
        return ready_tasks
    
    def _should_retry_task(self, task_id: str, error: str) -> bool:
        """
        Determine if a task should be retried.
        
        Args:
            task_id: Task ID
            error: Error message
            
        Returns:
            True if task should be retried
        """
        if task_id not in self.retry_policies:
            return False
        
        policy = self.retry_policies[task_id]
        attempts = self.task_attempts.get(task_id, 0)
        
        if attempts >= policy.max_attempts:
            return False
        
        # Check if error type is retryable
        error_lower = error.lower()
        for retry_error in policy.retry_on_errors:
            if retry_error.lower() in error_lower:
                return True
        
        return False
    
    def _calculate_retry_delay(self, task_id: str) -> float:
        """
        Calculate delay before retry.
        
        Args:
            task_id: Task ID
            
        Returns:
            Delay in seconds
        """
        if task_id not in self.retry_policies:
            return 0
        
        policy = self.retry_policies[task_id]
        attempts = self.task_attempts.get(task_id, 0)
        
        delay = policy.initial_delay * (policy.backoff_multiplier ** attempts)
        return min(delay, policy.max_delay)
    
    async def _execute_task_async(self, task_id: str, crew: Crew) -> bool:
        """
        Execute a single task asynchronously.
        
        Args:
            task_id: Task ID to execute
            crew: CrewAI crew instance
            
        Returns:
            True if task completed successfully
        """
        task_def = self.task_orchestrator.tasks[task_id]
        
        try:
            self.running_tasks.add(task_id)
            task_def.start_execution()
            
            self.logger.info(f"Executing task: {task_id}")
            
            # Execute the task (this would be integrated with CrewAI)
            # For now, we'll simulate task execution
            await asyncio.sleep(1)  # Simulate work
            
            # Mark as completed
            task_def.complete_execution(
                result_data={"status": "completed", "task_id": task_id},
                output_files=[]
            )
            
            self.completed_tasks.add(task_id)
            self.running_tasks.discard(task_id)
            
            self.logger.info(f"Task completed successfully: {task_id}")
            self._notify_progress()
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Task failed: {task_id} - {error_msg}")
            
            # Check if we should retry
            if self._should_retry_task(task_id, error_msg):
                self.task_attempts[task_id] = self.task_attempts.get(task_id, 0) + 1
                delay = self._calculate_retry_delay(task_id)
                
                self.logger.info(f"Retrying task {task_id} in {delay:.1f}s (attempt {self.task_attempts[task_id]})")
                
                await asyncio.sleep(delay)
                self.running_tasks.discard(task_id)
                return await self._execute_task_async(task_id, crew)
            else:
                # Mark as failed
                task_def.fail_execution(error_msg)
                self.failed_tasks.add(task_id)
                self.running_tasks.discard(task_id)
                
                return False
    
    def _notify_progress(self):
        """
        Notify progress callbacks.
        """
        progress_data = {
            'workflow_id': self.workflow_id,
            'status': self.status.value,
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'running_tasks': len(self.running_tasks),
            'total_tasks': len(self.task_orchestrator.tasks),
            'progress_percentage': len(self.completed_tasks) / len(self.task_orchestrator.tasks) * 100
        }
        
        for callback in self.progress_callbacks:
            try:
                callback(progress_data)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
    
    def _notify_completion(self, success: bool):
        """
        Notify completion callbacks.
        
        Args:
            success: Whether workflow completed successfully
        """
        completion_data = {
            'workflow_id': self.workflow_id,
            'success': success,
            'metrics': self.metrics,
            'task_results': self.task_orchestrator.get_task_results()
        }
        
        for callback in self.completion_callbacks:
            try:
                callback(completion_data)
            except Exception as e:
                self.logger.warning(f"Completion callback failed: {e}")
    
    def _save_checkpoint(self):
        """
        Save workflow checkpoint.
        """
        checkpoint = CheckpointData(
            workflow_id=self.workflow_id,
            timestamp=datetime.now().isoformat(),
            status=self.status.value,
            completed_tasks=list(self.completed_tasks),
            failed_tasks=list(self.failed_tasks),
            current_task=list(self.running_tasks)[0] if self.running_tasks else None,
            task_results={tid: task.result.dict() for tid, task in self.task_orchestrator.tasks.items()},
            metadata={
                'task_attempts': self.task_attempts,
                'metrics': self.metrics.dict()
            }
        )
        
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.dict(), f, indent=2, ensure_ascii=False)
            
            self.last_checkpoint = datetime.now()
            self.logger.debug(f"Checkpoint saved: {self.checkpoint_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> bool:
        """
        Load workflow checkpoint.
        
        Returns:
            True if checkpoint was loaded successfully
        """
        try:
            if not self.checkpoint_file.exists():
                return False
            
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            checkpoint = CheckpointData(**checkpoint_data)
            
            # Restore state
            self.status = WorkflowStatus(checkpoint.status)
            self.completed_tasks = set(checkpoint.completed_tasks)
            self.failed_tasks = set(checkpoint.failed_tasks)
            self.task_attempts = checkpoint.metadata.get('task_attempts', {})
            
            if checkpoint.metadata.get('metrics'):
                self.metrics = WorkflowMetrics(**checkpoint.metadata['metrics'])
            
            self.logger.info(f"Checkpoint loaded: {len(self.completed_tasks)} completed, {len(self.failed_tasks)} failed")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return False
    
    async def execute_workflow(self, crew: Crew, max_parallel_tasks: int = None) -> bool:
        """
        Execute the complete workflow.
        
        Args:
            crew: CrewAI crew instance
            max_parallel_tasks: Maximum number of parallel tasks
            
        Returns:
            True if workflow completed successfully
        """
        if max_parallel_tasks is None:
            max_parallel_tasks = self.max_workers
        
        self.logger.info(f"Starting workflow execution: {self.workflow_id}")
        
        # Initialize metrics
        self.status = WorkflowStatus.RUNNING
        self.metrics.start_time = datetime.now().isoformat()
        self.metrics.task_count = len(self.task_orchestrator.tasks)
        
        try:
            # Main execution loop
            while True:
                # Check if we should save checkpoint
                if (
                    self.last_checkpoint is None or 
                    datetime.now() - self.last_checkpoint > timedelta(seconds=self.checkpoint_interval)
                ):
                    self._save_checkpoint()
                
                # Get ready tasks
                ready_tasks = self._get_ready_tasks()
                
                if not ready_tasks and not self.running_tasks:
                    # No more tasks to run
                    break
                
                # Execute tasks in parallel (up to max_parallel_tasks)
                if ready_tasks:
                    tasks_to_run = ready_tasks[:max_parallel_tasks - len(self.running_tasks)]
                    
                    # Create async tasks
                    async_tasks = [
                        self._execute_task_async(task_id, crew)
                        for task_id in tasks_to_run
                    ]
                    
                    if async_tasks:
                        # Wait for at least one task to complete
                        done, pending = await asyncio.wait(
                            async_tasks,
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        
                        # Cancel pending tasks if needed
                        for task in pending:
                            task.cancel()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
            
            # Finalize metrics
            self.metrics.end_time = datetime.now().isoformat()
            self.metrics.completed_tasks = len(self.completed_tasks)
            self.metrics.failed_tasks = len(self.failed_tasks)
            
            if self.metrics.start_time and self.metrics.end_time:
                start = datetime.fromisoformat(self.metrics.start_time)
                end = datetime.fromisoformat(self.metrics.end_time)
                self.metrics.total_duration = (end - start).total_seconds()
            
            # Determine success
            success = len(self.failed_tasks) == 0 and len(self.completed_tasks) == self.metrics.task_count
            
            if success:
                self.status = WorkflowStatus.COMPLETED
                self.logger.info(f"Workflow completed successfully: {self.workflow_id}")
            else:
                self.status = WorkflowStatus.FAILED
                self.logger.error(f"Workflow failed: {len(self.failed_tasks)} failed tasks")
            
            # Final checkpoint
            self._save_checkpoint()
            
            # Notify completion
            self._notify_completion(success)
            
            return success
            
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            self.logger.error(f"Workflow execution failed: {e}")
            self._save_checkpoint()
            self._notify_completion(False)
            return False
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get current workflow status.
        
        Returns:
            Workflow status information
        """
        return {
            'workflow_id': self.workflow_id,
            'status': self.status.value,
            'metrics': self.metrics.dict(),
            'task_summary': self.task_orchestrator.get_execution_summary(),
            'dependencies': {tid: dep.dict() for tid, dep in self.dependencies.items()},
            'retry_policies': {tid: policy.dict() for tid, policy in self.retry_policies.items()}
        }


class WorkflowBuilder:
    """
    Builder class for creating complex workflows.
    """
    
    def __init__(self, workflow_id: str, output_dir: Path):
        self.workflow_id = workflow_id
        self.output_dir = output_dir
        self.orchestrator = WorkflowOrchestrator(workflow_id, output_dir)
        
    def add_video_search(
        self, 
        task_id: str,
        search_query: str,
        max_results: int = 10,
        **kwargs
    ) -> 'WorkflowBuilder':
        """
        Add video search task to workflow.
        
        Args:
            task_id: Unique task identifier
            search_query: Search query for videos
            max_results: Maximum number of videos
            **kwargs: Additional task parameters
            
        Returns:
            Self for method chaining
        """
        task = VideoSearchTask(task_id, self.output_dir)
        self.orchestrator.task_orchestrator.add_task(task)
        return self
    
    def add_transcription(
        self, 
        task_id: str,
        depends_on: List[str] = None,
        **kwargs
    ) -> 'WorkflowBuilder':
        """
        Add transcription task to workflow.
        
        Args:
            task_id: Unique task identifier
            depends_on: List of task IDs this depends on
            **kwargs: Additional task parameters
            
        Returns:
            Self for method chaining
        """
        task = TranscriptionTask(task_id, self.output_dir)
        self.orchestrator.task_orchestrator.add_task(task)
        
        if depends_on:
            self.orchestrator.add_task_dependency(task_id, depends_on)
        
        return self
    
    def add_indexing(
        self, 
        task_id: str,
        depends_on: List[str] = None,
        **kwargs
    ) -> 'WorkflowBuilder':
        """
        Add indexing task to workflow.
        
        Args:
            task_id: Unique task identifier
            depends_on: List of task IDs this depends on
            **kwargs: Additional task parameters
            
        Returns:
            Self for method chaining
        """
        task = IndexingTask(task_id, self.output_dir)
        self.orchestrator.task_orchestrator.add_task(task)
        
        if depends_on:
            self.orchestrator.add_task_dependency(task_id, depends_on)
        
        return self
    
    def add_qa(
        self, 
        task_id: str,
        questions: List[str],
        depends_on: List[str] = None,
        **kwargs
    ) -> 'WorkflowBuilder':
        """
        Add QA task to workflow.
        
        Args:
            task_id: Unique task identifier
            questions: Questions to answer
            depends_on: List of task IDs this depends on
            **kwargs: Additional task parameters
            
        Returns:
            Self for method chaining
        """
        task = QATask(task_id, self.output_dir)
        self.orchestrator.task_orchestrator.add_task(task)
        
        if depends_on:
            self.orchestrator.add_task_dependency(task_id, depends_on)
        
        return self
    
    def with_retry_policy(
        self, 
        task_id: str, 
        max_attempts: int = 3,
        **kwargs
    ) -> 'WorkflowBuilder':
        """
        Add retry policy to a task.
        
        Args:
            task_id: Task ID
            max_attempts: Maximum retry attempts
            **kwargs: Additional retry policy parameters
            
        Returns:
            Self for method chaining
        """
        policy = RetryPolicy(max_attempts=max_attempts, **kwargs)
        self.orchestrator.set_retry_policy(task_id, policy)
        return self
    
    def build(self) -> WorkflowOrchestrator:
        """
        Build and return the workflow orchestrator.
        
        Returns:
            Configured workflow orchestrator
        """
        return self.orchestrator