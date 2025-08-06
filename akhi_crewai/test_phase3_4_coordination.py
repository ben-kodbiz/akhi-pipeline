#!/usr/bin/env python3
"""
Phase 3-4 Agent Coordination System Test Suite

Comprehensive testing for:
- Individual agent functionality
- Crew orchestration and workflow management
- Task dependency handling
- Error recovery and retry logic
- Performance validation

Author: Assistant
Date: December 2024
Phase: 3-4 Testing
"""

import os
import sys
import json
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'crew'))

# Import agents
from agents import (
    create_video_researcher_agent,
    create_transcriber_agent,
    create_vector_indexer_agent,
    create_content_qa_agent
)

# Import crew components
from crew.akhi_pipeline import AkhiPipelineCrew
from crew.tasks import (
    VideoSearchTask,
    TranscriptionTask,
    IndexingTask,
    QATask,
    TaskOrchestrator
)
from crew.workflow import (
    WorkflowOrchestrator,
    WorkflowBuilder,
    RetryPolicy,
    WorkflowStatus
)


class Phase34TestSuite:
    """
    Comprehensive test suite for Phase 3-4 Agent Coordination System.
    """
    
    def __init__(self):
        self.test_results = {
            'agent_tests': {},
            'crew_tests': {},
            'workflow_tests': {},
            'integration_tests': {},
            'performance_tests': {}
        }
        self.temp_dir = None
        
    def setup_test_environment(self):
        """Setup temporary test environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix='akhi_phase34_test_'))
        print(f"📁 Test environment: {self.temp_dir}")
        
        # Create test directories
        (self.temp_dir / 'crew_outputs').mkdir(exist_ok=True)
        (self.temp_dir / 'embeddings').mkdir(exist_ok=True)
        (self.temp_dir / 'transcripts').mkdir(exist_ok=True)
        (self.temp_dir / 'audio').mkdir(exist_ok=True)
        
    def cleanup_test_environment(self):
        """Cleanup test environment."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up test environment")
    
    # ========================================
    # Phase 3: Individual Agent Tests
    # ========================================
    
    def test_video_researcher_agent(self) -> bool:
        """Test VideoResearcherAgent functionality."""
        print("\n🔍 Testing VideoResearcherAgent...")
        try:
            # Create agent
            agent = create_video_researcher_agent()
            
            # Validate agent properties
            assert "Research" in agent.role or "YouTube" in agent.role
            assert "YouTube" in agent.goal or "video" in agent.goal.lower()
            assert len(agent.tools) > 0
            
            # Test tool availability
            tool_names = [tool.__class__.__name__ for tool in agent.tools]
            assert "YouTubeSearchTool" in tool_names
            
            print("  ✅ Agent creation and configuration")
            print(f"  ✅ Role: {agent.role}")
            print(f"  ✅ Tools: {tool_names}")
            
            self.test_results['agent_tests']['video_researcher'] = {
                'status': 'passed',
                'details': {
                    'role': agent.role,
                    'tools': tool_names,
                    'goal_length': len(agent.goal)
                }
            }
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results['agent_tests']['video_researcher'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_transcriber_agent(self) -> bool:
        """Test TranscriberAgent functionality."""
        print("\n🎤 Testing TranscriberAgent...")
        try:
            # Create agent
            agent = create_transcriber_agent()
            
            # Validate agent properties
            assert "Transcriber" in agent.role
            assert "transcrib" in agent.goal.lower()
            assert len(agent.tools) >= 2  # Should have downloader and transcriber tools
            
            # Test tool availability
            tool_names = [tool.__class__.__name__ for tool in agent.tools]
            expected_tools = ["YouTubeDownloaderTool", "TranscriptionTool"]
            
            for expected_tool in expected_tools:
                assert expected_tool in tool_names, f"Missing tool: {expected_tool}"
            
            print("  ✅ Agent creation and configuration")
            print(f"  ✅ Role: {agent.role}")
            print(f"  ✅ Tools: {tool_names}")
            
            self.test_results['agent_tests']['transcriber'] = {
                'status': 'passed',
                'details': {
                    'role': agent.role,
                    'tools': tool_names,
                    'tool_count': len(agent.tools)
                }
            }
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results['agent_tests']['transcriber'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_vector_indexer_agent(self) -> bool:
        """Test VectorIndexerAgent functionality."""
        print("\n🔢 Testing VectorIndexerAgent...")
        try:
            # Create agent
            agent = create_vector_indexer_agent()
            
            # Validate agent properties
            assert "Indexer" in agent.role or "Engineer" in agent.role
            assert "embedding" in agent.goal.lower() or "vector" in agent.goal.lower()
            assert len(agent.tools) >= 3  # Should have chunker, embedder, storage tools
            
            # Test tool availability
            tool_names = [tool.__class__.__name__ for tool in agent.tools]
            expected_tools = ["TextChunkerTool", "EmbedderTool", "FAISSStorageTool"]
            
            for expected_tool in expected_tools:
                assert expected_tool in tool_names, f"Missing tool: {expected_tool}"
            
            print("  ✅ Agent creation and configuration")
            print(f"  ✅ Role: {agent.role}")
            print(f"  ✅ Tools: {tool_names}")
            
            self.test_results['agent_tests']['vector_indexer'] = {
                'status': 'passed',
                'details': {
                    'role': agent.role,
                    'tools': tool_names,
                    'tool_count': len(agent.tools)
                }
            }
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results['agent_tests']['vector_indexer'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_content_qa_agent(self) -> bool:
        """Test ContentQAAgent functionality."""
        print("\n❓ Testing ContentQAAgent...")
        try:
            # Create agent
            agent = create_content_qa_agent()
            
            # Validate agent properties
            assert "QA" in agent.role or "Assistant" in agent.role
            assert "question" in agent.goal.lower() or "answer" in agent.goal.lower()
            assert len(agent.tools) >= 3  # Should have query, summarizer, answer tools
            
            # Test tool availability
            tool_names = [tool.__class__.__name__ for tool in agent.tools]
            expected_tools = ["FAISSQueryTool", "SummarizerTool", "AnswerGeneratorTool"]
            
            for expected_tool in expected_tools:
                assert expected_tool in tool_names, f"Missing tool: {expected_tool}"
            
            print("  ✅ Agent creation and configuration")
            print(f"  ✅ Role: {agent.role}")
            print(f"  ✅ Tools: {tool_names}")
            
            self.test_results['agent_tests']['content_qa'] = {
                'status': 'passed',
                'details': {
                    'role': agent.role,
                    'tools': tool_names,
                    'tool_count': len(agent.tools)
                }
            }
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results['agent_tests']['content_qa'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    # ========================================
    # Phase 4: Crew Orchestration Tests
    # ========================================
    
    def test_akhi_pipeline_crew_creation(self) -> bool:
        """Test AkhiPipelineCrew creation and initialization."""
        print("\n🚀 Testing AkhiPipelineCrew creation...")
        try:
            # Create crew with test configuration
            test_config = {
                'output_dir': str(self.temp_dir / 'crew_outputs'),
                'logging': {
                    'level': 'INFO',
                    'file': str(self.temp_dir / 'crew.log')
                }
            }
            
            # Save test config
            config_path = self.temp_dir / 'test_crew_config.yaml'
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(test_config, f)
            
            # Create crew
            crew = AkhiPipelineCrew(config_path=str(config_path))
            
            # Validate crew properties
            assert crew.agents is not None
            assert len(crew.agents) >= 4  # Should have all 4 main agents
            assert crew.output_dir.exists()
            
            # Check agent availability
            expected_agents = ['researcher', 'transcriber', 'indexer', 'qa_agent']
            for agent_name in expected_agents:
                assert agent_name in crew.agents, f"Missing agent: {agent_name}"
            
            print("  ✅ Crew creation and initialization")
            print(f"  ✅ Agents: {list(crew.agents.keys())}")
            print(f"  ✅ Output directory: {crew.output_dir}")
            
            self.test_results['crew_tests']['pipeline_creation'] = {
                'status': 'passed',
                'details': {
                    'agent_count': len(crew.agents),
                    'agents': list(crew.agents.keys()),
                    'output_dir': str(crew.output_dir)
                }
            }
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['crew_tests']['pipeline_creation'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_task_definitions(self) -> bool:
        """Test individual task definitions."""
        print("\n📋 Testing Task Definitions...")
        try:
            output_dir = self.temp_dir / 'tasks'
            output_dir.mkdir(exist_ok=True)
            
            # Test VideoSearchTask
            video_task = VideoSearchTask("test_video_search", output_dir)
            assert video_task.task_id == "test_video_search"
            assert video_task.result.task_type == "VideoSearchTask"
            print("  ✅ VideoSearchTask creation")
            
            # Test TranscriptionTask
            transcription_task = TranscriptionTask("test_transcription", output_dir)
            assert transcription_task.task_id == "test_transcription"
            assert transcription_task.result.task_type == "TranscriptionTask"
            print("  ✅ TranscriptionTask creation")
            
            # Test IndexingTask
            indexing_task = IndexingTask("test_indexing", output_dir)
            assert indexing_task.task_id == "test_indexing"
            assert indexing_task.result.task_type == "IndexingTask"
            print("  ✅ IndexingTask creation")
            
            # Test QATask
            qa_task = QATask("test_qa", output_dir)
            assert qa_task.task_id == "test_qa"
            assert qa_task.result.task_type == "QATask"
            print("  ✅ QATask creation")
            
            # Test TaskOrchestrator
            orchestrator = TaskOrchestrator(output_dir)
            task_id = orchestrator.add_task(video_task)
            assert task_id == "test_video_search"
            print("  ✅ TaskOrchestrator functionality")
            
            self.test_results['crew_tests']['task_definitions'] = {
                'status': 'passed',
                'details': {
                    'video_search': 'created',
                    'transcription': 'created',
                    'indexing': 'created',
                    'qa': 'created',
                    'orchestrator': 'functional'
                }
            }
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results['crew_tests']['task_definitions'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_workflow_orchestration(self) -> bool:
        """Test workflow orchestration capabilities."""
        print("\n🔄 Testing Workflow Orchestration...")
        try:
            workflow_dir = self.temp_dir / 'workflow'
            workflow_dir.mkdir(exist_ok=True)
            
            # Create workflow orchestrator
            orchestrator = WorkflowOrchestrator(
                workflow_id="test_workflow",
                output_dir=workflow_dir,
                max_workers=2
            )
            
            # Test workflow builder
            builder = WorkflowBuilder("test_builder", workflow_dir)
            
            # Add tasks with dependencies
            builder.add_video_search(
                "search_task",
                search_query="Islamic prayer",
                max_results=3
            )
            
            builder.add_transcription(
                "transcribe_task",
                depends_on=["search_task"]
            )
            
            builder.add_indexing(
                "index_task",
                depends_on=["transcribe_task"]
            )
            
            builder.add_qa(
                "qa_task",
                questions=["What is the importance of prayer?"],
                depends_on=["index_task"]
            )
            
            # Build workflow
            workflow = builder.build()
            
            # Validate workflow structure
            assert len(workflow.task_orchestrator.tasks) == 4
            assert len(workflow.dependencies) >= 3  # Should have dependency chain
            
            # Test dependency checking
            assert not workflow._check_dependencies("transcribe_task")  # Should wait for search
            assert workflow._check_dependencies("search_task")  # Should be ready
            
            print("  ✅ Workflow orchestrator creation")
            print("  ✅ Workflow builder functionality")
            print("  ✅ Task dependency management")
            print(f"  ✅ Tasks: {list(workflow.task_orchestrator.tasks.keys())}")
            
            self.test_results['workflow_tests']['orchestration'] = {
                'status': 'passed',
                'details': {
                    'task_count': len(workflow.task_orchestrator.tasks),
                    'dependency_count': len(workflow.dependencies),
                    'tasks': list(workflow.task_orchestrator.tasks.keys())
                }
            }
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['workflow_tests']['orchestration'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_retry_and_error_handling(self) -> bool:
        """Test retry policies and error handling."""
        print("\n🔄 Testing Retry and Error Handling...")
        try:
            # Create retry policy
            retry_policy = RetryPolicy(
                max_attempts=3,
                initial_delay=0.1,
                backoff_multiplier=2.0,
                max_delay=1.0
            )
            
            # Validate retry policy
            assert retry_policy.max_attempts == 3
            assert retry_policy.initial_delay == 0.1
            assert retry_policy.backoff_multiplier == 2.0
            
            # Test workflow with retry policy
            workflow_dir = self.temp_dir / 'retry_test'
            workflow_dir.mkdir(exist_ok=True)
            
            orchestrator = WorkflowOrchestrator(
                workflow_id="retry_test",
                output_dir=workflow_dir
            )
            
            # Add task with retry policy
            orchestrator.set_retry_policy("test_task", retry_policy)
            
            # Test retry delay calculation
            delay1 = orchestrator._calculate_retry_delay("test_task")
            assert delay1 == 0.1
            
            # Simulate retry attempt
            orchestrator.task_attempts["test_task"] = 1
            delay2 = orchestrator._calculate_retry_delay("test_task")
            assert delay2 == 0.2  # 0.1 * 2^1
            
            print("  ✅ Retry policy creation")
            print("  ✅ Retry delay calculation")
            print("  ✅ Error handling setup")
            
            self.test_results['workflow_tests']['retry_handling'] = {
                'status': 'passed',
                'details': {
                    'retry_policy': 'functional',
                    'delay_calculation': 'correct',
                    'error_handling': 'configured'
                }
            }
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results['workflow_tests']['retry_handling'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    # ========================================
    # Integration Tests
    # ========================================
    
    def test_end_to_end_integration(self) -> bool:
        """Test end-to-end integration (simplified)."""
        print("\n🔗 Testing End-to-End Integration...")
        try:
            # Create minimal test configuration
            test_config = {
                'output_dir': str(self.temp_dir / 'integration'),
                'youtube_search': {
                    'max_results': 2,
                    'duration': 'short'
                },
                'transcription': {
                    'model_size': 'tiny',
                    'include_timestamps': True
                },
                'embedding': {
                    'model_name': 'all-MiniLM-L6-v2',
                    'batch_size': 2
                },
                'faiss': {
                    'index_type': 'flat'
                }
            }
            
            # Save test config
            config_path = self.temp_dir / 'integration_config.yaml'
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(test_config, f)
            
            # Create crew
            crew = AkhiPipelineCrew(config_path=str(config_path))
            
            # Validate all components are available
            assert len(crew.agents) >= 4
            assert crew.output_dir.exists()
            
            # Test workflow creation
            builder = WorkflowBuilder("integration_test", crew.output_dir)
            builder.add_video_search(
                "search",
                search_query="Islamic basics",
                max_results=1
            )
            
            workflow = builder.build()
            assert len(workflow.task_orchestrator.tasks) == 1
            
            print("  ✅ Integration configuration")
            print("  ✅ Crew initialization")
            print("  ✅ Workflow creation")
            print("  ✅ Component integration")
            
            self.test_results['integration_tests']['end_to_end'] = {
                'status': 'passed',
                'details': {
                    'crew_agents': len(crew.agents),
                    'workflow_tasks': len(workflow.task_orchestrator.tasks),
                    'configuration': 'valid'
                }
            }
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['integration_tests']['end_to_end'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    # ========================================
    # Performance Tests
    # ========================================
    
    def test_performance_metrics(self) -> bool:
        """Test performance monitoring and metrics."""
        print("\n⚡ Testing Performance Metrics...")
        try:
            import time
            
            # Test workflow metrics
            workflow_dir = self.temp_dir / 'performance'
            workflow_dir.mkdir(exist_ok=True)
            
            orchestrator = WorkflowOrchestrator(
                workflow_id="performance_test",
                output_dir=workflow_dir
            )
            
            # Test metrics initialization
            assert orchestrator.metrics.task_count == 0
            assert orchestrator.metrics.completed_tasks == 0
            assert orchestrator.metrics.failed_tasks == 0
            
            # Test checkpoint functionality
            orchestrator._save_checkpoint()
            checkpoint_file = workflow_dir / 'workflow_performance_test_checkpoint.json'
            assert checkpoint_file.exists()
            
            # Test status reporting
            status = orchestrator.get_workflow_status()
            assert 'workflow_id' in status
            assert 'status' in status
            assert 'metrics' in status
            
            print("  ✅ Metrics initialization")
            print("  ✅ Checkpoint functionality")
            print("  ✅ Status reporting")
            
            self.test_results['performance_tests']['metrics'] = {
                'status': 'passed',
                'details': {
                    'checkpoint_created': checkpoint_file.exists(),
                    'status_fields': list(status.keys()),
                    'metrics_tracking': 'functional'
                }
            }
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results['performance_tests']['metrics'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    # ========================================
    # Test Execution and Reporting
    # ========================================
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 3-4 tests."""
        print("\n" + "="*60)
        print("🚀 PHASE 3-4 AGENT COORDINATION SYSTEM TESTS")
        print("="*60)
        
        # Setup test environment
        self.setup_test_environment()
        
        try:
            # Phase 3: Individual Agent Tests
            print("\n📋 PHASE 3: INDIVIDUAL AGENT TESTS")
            print("-" * 40)
            
            agent_tests = [
                self.test_video_researcher_agent,
                self.test_transcriber_agent,
                self.test_vector_indexer_agent,
                self.test_content_qa_agent
            ]
            
            agent_results = []
            for test in agent_tests:
                result = test()
                agent_results.append(result)
            
            # Phase 4: Crew Orchestration Tests
            print("\n🚀 PHASE 4: CREW ORCHESTRATION TESTS")
            print("-" * 40)
            
            crew_tests = [
                self.test_akhi_pipeline_crew_creation,
                self.test_task_definitions,
                self.test_workflow_orchestration,
                self.test_retry_and_error_handling
            ]
            
            crew_results = []
            for test in crew_tests:
                result = test()
                crew_results.append(result)
            
            # Integration Tests
            print("\n🔗 INTEGRATION TESTS")
            print("-" * 40)
            
            integration_result = self.test_end_to_end_integration()
            
            # Performance Tests
            print("\n⚡ PERFORMANCE TESTS")
            print("-" * 40)
            
            performance_result = self.test_performance_metrics()
            
            # Generate summary
            return self.generate_test_summary(
                agent_results, crew_results, integration_result, performance_result
            )
            
        finally:
            # Cleanup
            self.cleanup_test_environment()
    
    def generate_test_summary(self, agent_results: List[bool], crew_results: List[bool], 
                            integration_result: bool, performance_result: bool) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        
        # Calculate statistics
        total_tests = len(agent_results) + len(crew_results) + 2  # +2 for integration and performance
        passed_tests = sum(agent_results) + sum(crew_results) + int(integration_result) + int(performance_result)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100
        
        # Generate summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'phase': '3-4 Agent Coordination System',
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': success_rate,
            'test_categories': {
                'agent_tests': {
                    'total': len(agent_results),
                    'passed': sum(agent_results),
                    'success_rate': (sum(agent_results) / len(agent_results)) * 100
                },
                'crew_tests': {
                    'total': len(crew_results),
                    'passed': sum(crew_results),
                    'success_rate': (sum(crew_results) / len(crew_results)) * 100
                },
                'integration_tests': {
                    'total': 1,
                    'passed': int(integration_result),
                    'success_rate': int(integration_result) * 100
                },
                'performance_tests': {
                    'total': 1,
                    'passed': int(performance_result),
                    'success_rate': int(performance_result) * 100
                }
            },
            'detailed_results': self.test_results
        }
        
        # Print summary
        print("\n" + "="*60)
        print("📊 PHASE 3-4 TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        print("\nBy Category:")
        for category, stats in summary['test_categories'].items():
            print(f"  {category}: {stats['passed']}/{stats['total']} ({stats['success_rate']:.1f}%)")
        
        # Status indicator
        if success_rate >= 90:
            print("\n🎉 PHASE 3-4 READY FOR PRODUCTION!")
        elif success_rate >= 75:
            print("\n⚠️ PHASE 3-4 MOSTLY READY - Minor issues to address")
        else:
            print("\n❌ PHASE 3-4 NEEDS ATTENTION - Critical issues found")
        
        print("\nNext Steps:")
        if failed_tests == 0:
            print("1. Proceed to Phase 5: QLoRA Production Testing")
            print("2. Run performance benchmarks")
            print("3. Validate with real Islamic content")
        else:
            print("1. Address failed test cases")
            print("2. Re-run test suite")
            print("3. Review error logs and fix issues")
        
        print("="*60)
        
        return summary


def main():
    """Main test execution function."""
    test_suite = Phase34TestSuite()
    results = test_suite.run_all_tests()
    
    # Save results to file
    results_file = Path('test_reports') / f'phase34_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main()