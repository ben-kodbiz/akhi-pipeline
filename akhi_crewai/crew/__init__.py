#!/usr/bin/env python3
"""
Akhi Pipeline Crew - Phase 4 Implementation

Crew orchestration module for Islamic Content Processing Pipeline.
Provides advanced workflow management with task coordination,
dependency handling, and error recovery.

Author: Assistant
Date: December 2024
Phase: 4 - Crew Orchestration
"""

from .akhi_pipeline import AkhiPipelineCrew
from .tasks import (
    BaseTaskDefinition,
    TaskResult,
    VideoSearchTask,
    TranscriptionTask,
    IndexingTask,
    QATask,
    TaskOrchestrator
)
from .workflow import (
    WorkflowOrchestrator,
    WorkflowBuilder,
    WorkflowStatus,
    TaskDependency,
    RetryPolicy,
    CheckpointData,
    WorkflowMetrics
)

__all__ = [
    # Main crew class
    'AkhiPipelineCrew',
    
    # Task definitions
    'BaseTaskDefinition',
    'TaskResult',
    'VideoSearchTask',
    'TranscriptionTask',
    'IndexingTask',
    'QATask',
    'TaskOrchestrator',
    
    # Workflow orchestration
    'WorkflowOrchestrator',
    'WorkflowBuilder',
    'WorkflowStatus',
    'TaskDependency',
    'RetryPolicy',
    'CheckpointData',
    'WorkflowMetrics'
]

__version__ = "1.0.0"
__author__ = "Assistant"
__description__ = "Advanced crew orchestration for Islamic Content Processing Pipeline"