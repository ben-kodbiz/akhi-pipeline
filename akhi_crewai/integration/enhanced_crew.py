#!/usr/bin/env python3
"""
Enhanced CrewAI Integration

This module provides an enhanced CrewAI integration that combines the existing
Islamic content processing tools with the new framework for:

- Seamless tool integration
- Enhanced agent coordination
- Intelligent workflow orchestration
- Performance optimization
- Error handling and recovery

Author: Assistant
Date: 2024
Phase: Enhanced Integration
"""

import os
import sys
import json
import yaml
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union, Callable
from datetime import datetime
from dataclasses import dataclass

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Import framework components
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from framework import (
    FrameworkManager,
    ToolRegistry,
    AgentCoordinator,
    ToolMetadata,
    AgentMetadata,
    CoordinationStrategy,
    ToolCategory,
    initialize_framework,
    get_framework_manager
)

# Import existing tools
from ..tools import (
    YouTubeSearchTool,
    YouTubeDownloaderTool,
    TranscriptionTool,
    TextChunkerTool,
    EmbedderTool,
    FAISSStorageTool,
    FAISSQueryTool,
    SummarizerTool,
    AnswerGeneratorTool
)

# Import existing agents
from ..agents import (
    VideoResearcherAgent,
    TranscriberAgent,
    VectorIndexerAgent,
    ContentQAAgent
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """Represents a step in the enhanced workflow."""
    step_id: str
    name: str
    description: str
    agent_type: str
    tool_name: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    dependencies: List[str]
    optional: bool = False
    retry_count: int = 3
    timeout: int = 300


class EnhancedIslamicContentCrew:
    """
    Enhanced Islamic Content Processing Crew with framework integration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the enhanced crew.
        
        Args:
            config_path: Path to crew configuration file
        """
        self.config = self._load_config(config_path)
        self.framework_manager: Optional[FrameworkManager] = None
        
        # Crew components
        self.agents: Dict[str, Agent] = {}
        self.tools: Dict[str, BaseTool] = {}
        self.workflows: Dict[str, List[WorkflowStep]] = {}
        
        # Performance tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Initialize the enhanced crew
        self._initialize_crew()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load crew configuration."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                '../config/enhanced_crew.yaml'
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default crew configuration."""
        return {
            'crew': {
                'name': 'Enhanced Islamic Content Crew',
                'version': '2.0.0',
                'description': 'Enhanced crew for Islamic educational content processing',
                'max_concurrent_workflows': 3,
                'enable_performance_monitoring': True,
                'enable_error_recovery': True
            },
            'workflows': {
                'full_pipeline': {
                    'description': 'Complete video processing pipeline',
                    'steps': [
                        'search_videos',
                        'download_video',
                        'transcribe_audio',
                        'chunk_text',
                        'generate_embeddings',
                        'store_vectors',
                        'setup_query_system'
                    ]
                },
                'qa_pipeline': {
                    'description': 'Question answering pipeline',
                    'steps': [
                        'query_vectors',
                        'generate_summary',
                        'generate_answer'
                    ]
                },
                'content_analysis': {
                    'description': 'Content analysis and summarization',
                    'steps': [
                        'query_vectors',
                        'analyze_content',
                        'generate_insights'
                    ]
                }
            },
            'agents': {
                'video_researcher': {
                    'enabled': True,
                    'max_concurrent_tasks': 2,
                    'tools': ['youtube_search', 'youtube_downloader']
                },
                'transcriber': {
                    'enabled': True,
                    'max_concurrent_tasks': 1,
                    'tools': ['transcriber']
                },
                'vector_indexer': {
                    'enabled': True,
                    'max_concurrent_tasks': 2,
                    'tools': ['chunker', 'embedder', 'faiss_store']
                },
                'content_qa': {
                    'enabled': True,
                    'max_concurrent_tasks': 3,
                    'tools': ['faiss_query', 'summarizer', 'answer_generator']
                }
            }
        }
    
    def _initialize_crew(self):
        """Initialize the enhanced crew with framework integration."""
        logger.info("Initializing Enhanced Islamic Content Crew...")
        
        try:
            # Initialize framework
            framework_config_dir = os.path.join(
                os.path.dirname(__file__), '../config'
            )
            
            if not initialize_framework(framework_config_dir):
                raise Exception("Failed to initialize framework")
            
            self.framework_manager = get_framework_manager()
            logger.info("Framework initialized successfully")
            
            # Register tools with framework
            self._register_tools()
            
            # Create and register agents
            self._create_agents()
            
            # Define workflows
            self._define_workflows()
            
            logger.info("Enhanced crew initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced crew: {e}")
            raise
    
    def _register_tools(self):
        """Register all tools with the framework."""
        logger.info("Registering tools with framework...")
        
        # Tool registration mapping
        tool_classes = {
            'youtube_search': (YouTubeSearchTool, ToolCategory.SEARCH),
            'youtube_downloader': (YouTubeDownloaderTool, ToolCategory.DOWNLOAD),
            'transcriber': (TranscriptionTool, ToolCategory.TRANSCRIPTION),
            'chunker': (TextChunkerTool, ToolCategory.PROCESSING),
            'embedder': (EmbedderTool, ToolCategory.EMBEDDING),
            'faiss_store': (FAISSStorageTool, ToolCategory.STORAGE),
            'faiss_query': (FAISSQueryTool, ToolCategory.QUERY),
            'summarizer': (SummarizerTool, ToolCategory.GENERATION),
            'answer_generator': (AnswerGeneratorTool, ToolCategory.GENERATION)
        }
        
        for tool_name, (tool_class, category) in tool_classes.items():
            try:
                # Create tool metadata
                metadata = ToolMetadata(
                    name=tool_name,
                    description=getattr(tool_class, 'description', f'{tool_name} tool'),
                    category=category,
                    version='1.0.0',
                    author='Enhanced Crew'
                )
                
                # Register with framework
                success = self.framework_manager.register_tool_from_class(
                    tool_name, tool_class, metadata
                )
                
                if success:
                    # Get tool instance
                    tool_instance = self.framework_manager.get_tool_instance(tool_name)
                    if tool_instance:
                        self.tools[tool_name] = tool_instance
                        logger.info(f"Registered tool: {tool_name}")
                    else:
                        logger.warning(f"Failed to get instance for tool: {tool_name}")
                else:
                    logger.error(f"Failed to register tool: {tool_name}")
                    
            except Exception as e:
                logger.error(f"Error registering tool {tool_name}: {e}")
        
        logger.info(f"Registered {len(self.tools)} tools")
    
    def _create_agents(self):
        """Create and register agents with the framework."""
        logger.info("Creating and registering agents...")
        
        agent_configs = self.config.get('agents', {})
        
        # Video Researcher Agent
        if agent_configs.get('video_researcher', {}).get('enabled', True):
            try:
                video_researcher = VideoResearcherAgent()
                
                # Add tools to agent
                researcher_tools = []
                for tool_name in agent_configs['video_researcher'].get('tools', []):
                    if tool_name in self.tools:
                        researcher_tools.append(self.tools[tool_name])
                
                if researcher_tools:
                    video_researcher.tools = researcher_tools
                
                # Register with framework
                metadata = AgentMetadata(
                    agent_id='video_researcher',
                    name='Video Researcher',
                    role=video_researcher.role,
                    capabilities=['search', 'research', 'download'],
                    available_tools=agent_configs['video_researcher'].get('tools', []),
                    max_concurrent_tasks=agent_configs['video_researcher'].get('max_concurrent_tasks', 2)
                )
                
                self.framework_manager.register_agent('video_researcher', video_researcher, metadata)
                self.agents['video_researcher'] = video_researcher
                logger.info("Created Video Researcher Agent")
                
            except Exception as e:
                logger.error(f"Error creating video researcher agent: {e}")
        
        # Transcriber Agent
        if agent_configs.get('transcriber', {}).get('enabled', True):
            try:
                transcriber = TranscriberAgent()
                
                # Add tools to agent
                transcriber_tools = []
                for tool_name in agent_configs['transcriber'].get('tools', []):
                    if tool_name in self.tools:
                        transcriber_tools.append(self.tools[tool_name])
                
                if transcriber_tools:
                    transcriber.tools = transcriber_tools
                
                # Register with framework
                metadata = AgentMetadata(
                    agent_id='transcriber',
                    name='Transcriber',
                    role=transcriber.role,
                    capabilities=['transcription', 'audio_processing'],
                    available_tools=agent_configs['transcriber'].get('tools', []),
                    max_concurrent_tasks=agent_configs['transcriber'].get('max_concurrent_tasks', 1)
                )
                
                self.framework_manager.register_agent('transcriber', transcriber, metadata)
                self.agents['transcriber'] = transcriber
                logger.info("Created Transcriber Agent")
                
            except Exception as e:
                logger.error(f"Error creating transcriber agent: {e}")
        
        # Vector Indexer Agent
        if agent_configs.get('vector_indexer', {}).get('enabled', True):
            try:
                vector_indexer = VectorIndexerAgent()
                
                # Add tools to agent
                indexer_tools = []
                for tool_name in agent_configs['vector_indexer'].get('tools', []):
                    if tool_name in self.tools:
                        indexer_tools.append(self.tools[tool_name])
                
                if indexer_tools:
                    vector_indexer.tools = indexer_tools
                
                # Register with framework
                metadata = AgentMetadata(
                    agent_id='vector_indexer',
                    name='Vector Indexer',
                    role=vector_indexer.role,
                    capabilities=['embedding', 'indexing', 'storage'],
                    available_tools=agent_configs['vector_indexer'].get('tools', []),
                    max_concurrent_tasks=agent_configs['vector_indexer'].get('max_concurrent_tasks', 2)
                )
                
                self.framework_manager.register_agent('vector_indexer', vector_indexer, metadata)
                self.agents['vector_indexer'] = vector_indexer
                logger.info("Created Vector Indexer Agent")
                
            except Exception as e:
                logger.error(f"Error creating vector indexer agent: {e}")
        
        # Content QA Agent
        if agent_configs.get('content_qa', {}).get('enabled', True):
            try:
                content_qa = ContentQAAgent()
                
                # Add tools to agent
                qa_tools = []
                for tool_name in agent_configs['content_qa'].get('tools', []):
                    if tool_name in self.tools:
                        qa_tools.append(self.tools[tool_name])
                
                if qa_tools:
                    content_qa.tools = qa_tools
                
                # Register with framework
                metadata = AgentMetadata(
                    agent_id='content_qa',
                    name='Content QA',
                    role=content_qa.role,
                    capabilities=['query', 'generation', 'qa', 'summarization'],
                    available_tools=agent_configs['content_qa'].get('tools', []),
                    max_concurrent_tasks=agent_configs['content_qa'].get('max_concurrent_tasks', 3)
                )
                
                self.framework_manager.register_agent('content_qa', content_qa, metadata)
                self.agents['content_qa'] = content_qa
                logger.info("Created Content QA Agent")
                
            except Exception as e:
                logger.error(f"Error creating content qa agent: {e}")
        
        logger.info(f"Created and registered {len(self.agents)} agents")
    
    def _define_workflows(self):
        """Define workflow steps for different pipelines."""
        logger.info("Defining workflows...")
        
        # Full Pipeline Workflow
        self.workflows['full_pipeline'] = [
            WorkflowStep(
                step_id='search_videos',
                name='Search Videos',
                description='Search for Islamic educational videos on YouTube',
                agent_type='video_researcher',
                tool_name='youtube_search',
                input_schema={'query': 'str', 'max_results': 'int'},
                output_schema={'videos': 'list'},
                dependencies=[]
            ),
            WorkflowStep(
                step_id='download_video',
                name='Download Video',
                description='Download selected video for processing',
                agent_type='video_researcher',
                tool_name='youtube_downloader',
                input_schema={'video_url': 'str', 'output_dir': 'str'},
                output_schema={'audio_file': 'str'},
                dependencies=['search_videos']
            ),
            WorkflowStep(
                step_id='transcribe_audio',
                name='Transcribe Audio',
                description='Transcribe audio to text using Whisper',
                agent_type='transcriber',
                tool_name='transcriber',
                input_schema={'audio_file': 'str'},
                output_schema={'transcript': 'str'},
                dependencies=['download_video']
            ),
            WorkflowStep(
                step_id='chunk_text',
                name='Chunk Text',
                description='Split transcript into semantic chunks',
                agent_type='vector_indexer',
                tool_name='chunker',
                input_schema={'text': 'str', 'chunk_size': 'int'},
                output_schema={'chunks': 'list'},
                dependencies=['transcribe_audio']
            ),
            WorkflowStep(
                step_id='generate_embeddings',
                name='Generate Embeddings',
                description='Generate vector embeddings for text chunks',
                agent_type='vector_indexer',
                tool_name='embedder',
                input_schema={'chunks': 'list'},
                output_schema={'embeddings': 'list'},
                dependencies=['chunk_text']
            ),
            WorkflowStep(
                step_id='store_vectors',
                name='Store Vectors',
                description='Store embeddings in FAISS index',
                agent_type='vector_indexer',
                tool_name='faiss_store',
                input_schema={'embeddings': 'list', 'metadata': 'list'},
                output_schema={'index_name': 'str'},
                dependencies=['generate_embeddings']
            ),
            WorkflowStep(
                step_id='setup_query_system',
                name='Setup Query System',
                description='Initialize query system for the indexed content',
                agent_type='content_qa',
                tool_name='faiss_query',
                input_schema={'index_name': 'str'},
                output_schema={'query_system_ready': 'bool'},
                dependencies=['store_vectors']
            )
        ]
        
        # QA Pipeline Workflow
        self.workflows['qa_pipeline'] = [
            WorkflowStep(
                step_id='query_vectors',
                name='Query Vectors',
                description='Search for relevant content using vector similarity',
                agent_type='content_qa',
                tool_name='faiss_query',
                input_schema={'query': 'str', 'k': 'int'},
                output_schema={'relevant_chunks': 'list'},
                dependencies=[]
            ),
            WorkflowStep(
                step_id='generate_summary',
                name='Generate Summary',
                description='Generate summary of relevant content',
                agent_type='content_qa',
                tool_name='summarizer',
                input_schema={'text': 'str'},
                output_schema={'summary': 'str'},
                dependencies=['query_vectors'],
                optional=True
            ),
            WorkflowStep(
                step_id='generate_answer',
                name='Generate Answer',
                description='Generate comprehensive answer to the question',
                agent_type='content_qa',
                tool_name='answer_generator',
                input_schema={'question': 'str', 'context': 'list'},
                output_schema={'answer': 'str', 'citations': 'list'},
                dependencies=['query_vectors']
            )
        ]
        
        logger.info(f"Defined {len(self.workflows)} workflows")
    
    def execute_workflow(self, workflow_name: str, input_data: Dict[str, Any], 
                        strategy: CoordinationStrategy = CoordinationStrategy.ADAPTIVE) -> Dict[str, Any]:
        """Execute a workflow using the framework coordination."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        logger.info(f"Executing workflow: {workflow_name}")
        
        try:
            # Create tasks for workflow steps
            tasks = []
            task_ids = []
            
            for step in self.workflows[workflow_name]:
                # Create CrewAI task
                task = Task(
                    description=step.description,
                    agent=self.agents[step.agent_type],
                    tools=[self.tools[step.tool_name]]
                )
                
                # Create task with framework
                task_id = f"{workflow_name}_{step.step_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                success = self.framework_manager.agent_coordinator.create_task(
                    task_id=task_id,
                    name=step.name,
                    description=step.description,
                    agent_id=step.agent_type,
                    task=task,
                    priority=len(self.workflows[workflow_name]) - len(tasks),  # Earlier steps have higher priority
                    dependencies=step.dependencies,
                    required_tools=[step.tool_name],
                    timeout=step.timeout,
                    max_retries=step.retry_count
                )
                
                if success:
                    tasks.append(task)
                    task_ids.append(task_id)
                else:
                    logger.error(f"Failed to create task for step: {step.step_id}")
            
            # Create coordination plan
            plan_id = f"{workflow_name}_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            plan = self.framework_manager.create_coordination_plan(plan_id, task_ids, strategy)
            
            if not plan:
                raise Exception("Failed to create coordination plan")
            
            # Execute the plan
            execution_start = datetime.now()
            success = self.framework_manager.execute_plan(plan_id)
            
            if not success:
                raise Exception("Failed to execute coordination plan")
            
            # Wait for completion and collect results
            results = self._wait_for_workflow_completion(plan_id, task_ids)
            
            execution_time = (datetime.now() - execution_start).total_seconds()
            
            # Record execution history
            execution_record = {
                'workflow_name': workflow_name,
                'plan_id': plan_id,
                'task_ids': task_ids,
                'strategy': strategy.value,
                'execution_time': execution_time,
                'success': True,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            self.execution_history.append(execution_record)
            
            logger.info(f"Workflow {workflow_name} completed successfully in {execution_time:.2f} seconds")
            
            return {
                'success': True,
                'workflow_name': workflow_name,
                'execution_time': execution_time,
                'results': results,
                'plan_id': plan_id
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            
            # Record failed execution
            execution_record = {
                'workflow_name': workflow_name,
                'plan_id': plan_id if 'plan_id' in locals() else None,
                'strategy': strategy.value,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            self.execution_history.append(execution_record)
            
            return {
                'success': False,
                'workflow_name': workflow_name,
                'error': str(e)
            }
    
    def _wait_for_workflow_completion(self, plan_id: str, task_ids: List[str], 
                                    timeout: int = 3600) -> Dict[str, Any]:
        """Wait for workflow completion and collect results."""
        import time
        
        start_time = time.time()
        results = {}
        
        while time.time() - start_time < timeout:
            # Check task statuses
            all_completed = True
            
            for task_id in task_ids:
                status = self.framework_manager.agent_coordinator.task_scheduler.get_task_status(task_id)
                
                if status and status.value in ['completed', 'failed']:
                    if task_id not in results:
                        # Get task result
                        if task_id in self.framework_manager.agent_coordinator.task_scheduler.completed_tasks:
                            task_metadata = self.framework_manager.agent_coordinator.task_scheduler.completed_tasks[task_id]
                            results[task_id] = {
                                'status': status.value,
                                'result': task_metadata.result,
                                'error': task_metadata.error_message
                            }
                        else:
                            results[task_id] = {
                                'status': status.value,
                                'result': None,
                                'error': 'Task not found in completed tasks'
                            }
                else:
                    all_completed = False
            
            if all_completed:
                break
            
            time.sleep(1)  # Check every second
        
        return results
    
    def process_islamic_content(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Process Islamic content using the full pipeline."""
        input_data = {
            'query': query,
            'max_results': max_results
        }
        
        return self.execute_workflow('full_pipeline', input_data)
    
    def answer_question(self, question: str, index_name: Optional[str] = None) -> Dict[str, Any]:
        """Answer a question using the QA pipeline."""
        input_data = {
            'question': question,
            'index_name': index_name
        }
        
        return self.execute_workflow('qa_pipeline', input_data)
    
    def get_crew_status(self) -> Dict[str, Any]:
        """Get comprehensive crew status."""
        framework_status = self.framework_manager.get_status() if self.framework_manager else {}
        
        return {
            'crew_info': {
                'name': self.config['crew']['name'],
                'version': self.config['crew']['version'],
                'agents_count': len(self.agents),
                'tools_count': len(self.tools),
                'workflows_count': len(self.workflows)
            },
            'framework_status': framework_status,
            'execution_history': {
                'total_executions': len(self.execution_history),
                'successful_executions': sum(1 for e in self.execution_history if e['success']),
                'recent_executions': self.execution_history[-5:] if self.execution_history else []
            },
            'performance_metrics': self.performance_metrics
        }
    
    def export_crew_report(self, output_path: str):
        """Export comprehensive crew report."""
        report = {
            'report_info': {
                'generated_at': datetime.now().isoformat(),
                'crew_version': self.config['crew']['version']
            },
            'crew_status': self.get_crew_status(),
            'workflows': {
                name: [
                    {
                        'step_id': step.step_id,
                        'name': step.name,
                        'description': step.description,
                        'agent_type': step.agent_type,
                        'tool_name': step.tool_name,
                        'dependencies': step.dependencies
                    }
                    for step in steps
                ]
                for name, steps in self.workflows.items()
            },
            'execution_history': self.execution_history
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(report, f, default_flow_style=False, indent=2)
        
        logger.info(f"Crew report exported to: {output_path}")


if __name__ == "__main__":
    # Demo usage
    print("Enhanced Islamic Content Crew Demo")
    print("=" * 50)
    
    try:
        # Initialize enhanced crew
        crew = EnhancedIslamicContentCrew()
        
        # Get status
        status = crew.get_crew_status()
        print(f"Crew: {status['crew_info']['name']} v{status['crew_info']['version']}")
        print(f"Agents: {status['crew_info']['agents_count']}")
        print(f"Tools: {status['crew_info']['tools_count']}")
        print(f"Workflows: {status['crew_info']['workflows_count']}")
        
        # Test workflow execution (commented out for demo)
        # print("\nTesting QA workflow...")
        # result = crew.answer_question("What is the importance of prayer in Islam?")
        # print(f"QA Result: {result['success']}")
        
        # Export report
        output_path = "/tmp/enhanced_crew_report.yaml"
        crew.export_crew_report(output_path)
        print(f"\nCrew report exported to: {output_path}")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()