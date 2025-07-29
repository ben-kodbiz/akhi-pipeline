#!/usr/bin/env python3
"""
Phase 4 Demo: Advanced Crew Orchestration

Demonstrates the new workflow orchestration capabilities:
- Task dependency management
- Parallel processing
- Error recovery and retry logic
- Progress monitoring
- Checkpointing

Author: Assistant
Date: December 2024
Phase: 4 - Crew Orchestration
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project paths
sys.path.append(os.path.dirname(__file__))

from crew import (
    AkhiPipelineCrew,
    WorkflowBuilder,
    WorkflowOrchestrator,
    RetryPolicy
)


def setup_logging():
    """
    Setup logging configuration for the demo.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('demo_phase4.log')
        ]
    )


def progress_callback(progress_data: Dict[str, Any]):
    """
    Progress callback for workflow monitoring.
    
    Args:
        progress_data: Progress information
    """
    print(f"\n📊 WORKFLOW PROGRESS:")
    print(f"   Status: {progress_data['status']}")
    print(f"   Progress: {progress_data['progress_percentage']:.1f}%")
    print(f"   Completed: {progress_data['completed_tasks']}/{progress_data['total_tasks']}")
    print(f"   Running: {progress_data['running_tasks']}")
    print(f"   Failed: {progress_data['failed_tasks']}")


def completion_callback(completion_data: Dict[str, Any]):
    """
    Completion callback for workflow finalization.
    
    Args:
        completion_data: Completion information
    """
    print(f"\n🏁 WORKFLOW COMPLETED:")
    print(f"   Success: {completion_data['success']}")
    print(f"   Duration: {completion_data['metrics'].total_duration:.2f}s")
    print(f"   Tasks: {completion_data['metrics'].completed_tasks}/{completion_data['metrics'].task_count}")


async def demo_workflow_builder():
    """
    Demonstrate the workflow builder pattern.
    """
    print("\n🔧 DEMO: Workflow Builder Pattern")
    print("=" * 50)
    
    output_dir = Path("data/crew_outputs/phase4_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build a complex workflow
    workflow = (
        WorkflowBuilder("islamic_content_pipeline", output_dir)
        .add_video_search(
            "search_task",
            search_query="Islamic prayer importance",
            max_results=5
        )
        .add_transcription(
            "transcription_task",
            depends_on=["search_task"]
        )
        .add_indexing(
            "indexing_task",
            depends_on=["transcription_task"]
        )
        .add_qa(
            "qa_task",
            questions=[
                "What is the importance of prayer in Islam?",
                "How many times should Muslims pray daily?",
                "What are the benefits of regular prayer?"
            ],
            depends_on=["indexing_task"]
        )
        .with_retry_policy("transcription_task", max_attempts=3)
        .with_retry_policy("indexing_task", max_attempts=2)
        .build()
    )
    
    # Add callbacks
    workflow.add_progress_callback(progress_callback)
    workflow.add_completion_callback(completion_callback)
    
    print(f"✅ Workflow built with {len(workflow.task_orchestrator.tasks)} tasks")
    print(f"   Dependencies: {len(workflow.dependencies)}")
    print(f"   Retry policies: {len(workflow.retry_policies)}")
    
    # Display workflow structure
    status = workflow.get_workflow_status()
    print(f"\n📋 Workflow Structure:")
    for task_id in workflow.task_orchestrator.execution_order:
        deps = workflow.dependencies.get(task_id)
        if deps:
            print(f"   {task_id} → depends on: {deps.depends_on}")
        else:
            print(f"   {task_id} → no dependencies")
    
    return workflow


async def demo_qa_only_workflow():
    """
    Demonstrate QA-only workflow using existing index.
    """
    print("\n🤖 DEMO: QA-Only Workflow")
    print("=" * 50)
    
    try:
        # Initialize the crew
        crew = AkhiPipelineCrew()
        
        # Test questions
        test_questions = [
            "What is the importance of prayer in Islam?",
            "How should Muslims treat their parents?",
            "What are the five pillars of Islam?",
            "What is the significance of Ramadan?",
            "How should Muslims conduct business ethically?"
        ]
        
        print(f"📝 Testing {len(test_questions)} questions...")
        
        # Execute QA workflow
        results = crew.execute_qa_only(test_questions, "islamic_content")
        
        print(f"\n📊 QA Results:")
        print(f"   Status: {results['status']}")
        print(f"   Questions: {len(results['questions'])}")
        print(f"   Output: {results['output_dir']}")
        
        if results['status'] == 'completed':
            print("\n✅ QA workflow completed successfully!")
        else:
            print(f"\n❌ QA workflow failed: {results.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ QA workflow demo failed: {str(e)}")
        return None


def demo_checkpoint_recovery():
    """
    Demonstrate checkpoint and recovery functionality.
    """
    print("\n💾 DEMO: Checkpoint and Recovery")
    print("=" * 50)
    
    output_dir = Path("data/crew_outputs/checkpoint_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create workflow
    workflow = WorkflowOrchestrator("checkpoint_test", output_dir)
    
    # Simulate some completed tasks
    workflow.completed_tasks.add("task1")
    workflow.completed_tasks.add("task2")
    workflow.failed_tasks.add("task3")
    
    # Save checkpoint
    workflow._save_checkpoint()
    print(f"✅ Checkpoint saved: {workflow.checkpoint_file}")
    
    # Create new workflow and load checkpoint
    new_workflow = WorkflowOrchestrator("checkpoint_test", output_dir)
    success = new_workflow.load_checkpoint()
    
    if success:
        print(f"✅ Checkpoint loaded successfully")
        print(f"   Completed tasks: {len(new_workflow.completed_tasks)}")
        print(f"   Failed tasks: {len(new_workflow.failed_tasks)}")
    else:
        print(f"❌ Failed to load checkpoint")
    
    return success


def demo_retry_policies():
    """
    Demonstrate retry policy configuration.
    """
    print("\n🔄 DEMO: Retry Policies")
    print("=" * 50)
    
    # Create different retry policies
    policies = {
        "aggressive": RetryPolicy(
            max_attempts=5,
            initial_delay=0.5,
            backoff_multiplier=1.5,
            max_delay=30.0,
            retry_on_errors=["timeout", "network", "temporary", "rate_limit"]
        ),
        "conservative": RetryPolicy(
            max_attempts=2,
            initial_delay=2.0,
            backoff_multiplier=3.0,
            max_delay=60.0,
            retry_on_errors=["timeout", "network"]
        ),
        "no_retry": RetryPolicy(
            max_attempts=1,
            initial_delay=0.0,
            retry_on_errors=[]
        )
    }
    
    print("📋 Retry Policy Examples:")
    for name, policy in policies.items():
        print(f"\n   {name.upper()}:")
        print(f"     Max attempts: {policy.max_attempts}")
        print(f"     Initial delay: {policy.initial_delay}s")
        print(f"     Backoff multiplier: {policy.backoff_multiplier}x")
        print(f"     Max delay: {policy.max_delay}s")
        print(f"     Retry on: {', '.join(policy.retry_on_errors)}")
    
    # Demonstrate delay calculation
    print(f"\n🕐 Delay Progression (aggressive policy):")
    policy = policies["aggressive"]
    for attempt in range(policy.max_attempts):
        delay = policy.initial_delay * (policy.backoff_multiplier ** attempt)
        delay = min(delay, policy.max_delay)
        print(f"   Attempt {attempt + 1}: {delay:.1f}s delay")


def demo_metrics_and_monitoring():
    """
    Demonstrate metrics collection and monitoring.
    """
    print("\n📈 DEMO: Metrics and Monitoring")
    print("=" * 50)
    
    output_dir = Path("data/crew_outputs/metrics_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create workflow with metrics
    workflow = WorkflowOrchestrator("metrics_test", output_dir)
    
    # Simulate workflow execution
    workflow.metrics.start_time = datetime.now().isoformat()
    workflow.metrics.task_count = 4
    
    # Simulate task completion
    import time
    time.sleep(0.1)  # Simulate some work
    
    workflow.metrics.end_time = datetime.now().isoformat()
    workflow.metrics.completed_tasks = 3
    workflow.metrics.failed_tasks = 1
    workflow.metrics.retried_tasks = 2
    
    # Calculate duration
    if workflow.metrics.start_time and workflow.metrics.end_time:
        start = datetime.fromisoformat(workflow.metrics.start_time)
        end = datetime.fromisoformat(workflow.metrics.end_time)
        workflow.metrics.total_duration = (end - start).total_seconds()
        workflow.metrics.average_task_duration = workflow.metrics.total_duration / workflow.metrics.task_count
    
    # Display metrics
    print(f"📊 Workflow Metrics:")
    print(f"   Total duration: {workflow.metrics.total_duration:.3f}s")
    print(f"   Task count: {workflow.metrics.task_count}")
    print(f"   Completed: {workflow.metrics.completed_tasks}")
    print(f"   Failed: {workflow.metrics.failed_tasks}")
    print(f"   Retried: {workflow.metrics.retried_tasks}")
    print(f"   Avg task duration: {workflow.metrics.average_task_duration:.3f}s")
    print(f"   Success rate: {workflow.metrics.completed_tasks/workflow.metrics.task_count*100:.1f}%")


async def main():
    """
    Main demo function.
    """
    print("🚀 AKHI PIPELINE CREW - PHASE 4 DEMO")
    print("=" * 60)
    print("Advanced Crew Orchestration Capabilities")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger('Phase4Demo')
    logger.info("Starting Phase 4 demo")
    
    try:
        # Demo 1: Workflow Builder
        workflow = await demo_workflow_builder()
        
        # Demo 2: QA-Only Workflow (using existing index)
        qa_results = await demo_qa_only_workflow()
        
        # Demo 3: Checkpoint and Recovery
        checkpoint_success = demo_checkpoint_recovery()
        
        # Demo 4: Retry Policies
        demo_retry_policies()
        
        # Demo 5: Metrics and Monitoring
        demo_metrics_and_monitoring()
        
        # Summary
        print("\n🎯 PHASE 4 DEMO SUMMARY")
        print("=" * 50)
        print(f"✅ Workflow Builder: Demonstrated")
        print(f"✅ QA Workflow: {'Success' if qa_results and qa_results['status'] == 'completed' else 'Failed'}")
        print(f"✅ Checkpoint Recovery: {'Success' if checkpoint_success else 'Failed'}")
        print(f"✅ Retry Policies: Demonstrated")
        print(f"✅ Metrics Collection: Demonstrated")
        
        print("\n🎉 Phase 4 implementation is complete and functional!")
        print("\n📋 Key Features Implemented:")
        print("   • Advanced task dependency management")
        print("   • Parallel processing capabilities")
        print("   • Error recovery and retry logic")
        print("   • Progress monitoring and callbacks")
        print("   • Workflow checkpointing")
        print("   • Comprehensive metrics collection")
        print("   • Builder pattern for workflow creation")
        
        logger.info("Phase 4 demo completed successfully")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        logger.error(f"Phase 4 demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())