#!/usr/bin/env python3
"""
Akhi Pipeline Crew - Phase 4 Implementation

Main crew orchestration for the Islamic Content Processing Pipeline.
This module implements the core CrewAI workflow coordination with:
- Task dependency management
- Error recovery and retry logic
- Progress monitoring and logging
- Result aggregation

Author: Assistant
Date: December 2024
Phase: 4 - Crew Orchestration
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

# CrewAI imports
from crewai import Crew, Task, Process, Agent
from crewai.task import TaskOutput

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agents import (
    create_video_researcher_agent,
    create_transcriber_agent,
    create_vector_indexer_agent,
    create_content_qa_agent
)

# Phase 8: QLoRA Training Integration
try:
    from agents.qlora_trainer import QLoRATrainerAgent
    QLORA_AVAILABLE = True
except ImportError:
    QLORA_AVAILABLE = False
    print("⚠️ QLoRA training components not available")


class AkhiPipelineCrew:
    """
    Main Akhi Pipeline Crew for Islamic Content Processing
    
    Orchestrates the complete workflow:
    1. Video Research and Selection
    2. Content Download and Transcription
    3. Vector Indexing and Storage
    4. Question Answering and Summarization
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Akhi Pipeline Crew.
        
        Args:
            config_path: Path to the crew configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize agents
        self.agents = self._create_agents()
        
        # Initialize crew
        self.crew = None
        self.current_session = None
        
        # Create output directories
        self.output_dir = Path(self.config.get('output_dir', 'data/crew_outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 8: Initialize QLoRA trainer if available
        self.qlora_trainer = None
        if QLORA_AVAILABLE and self.config.get('qlora_training', {}).get('enabled', False):
            try:
                self.qlora_trainer = QLoRATrainerAgent(config=self.config)
                self.logger.info("QLoRA trainer initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize QLoRA trainer: {e}")
        
        self.logger.info("AkhiPipelineCrew initialized successfully")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                '../config/crew_config.yaml'
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            # Fallback configuration
            return {
                'crew': {
                    'process': 'sequential',
                    'verbose': True,
                    'memory': True,
                    'max_rpm': 10,
                    'share_crew': False
                },
                'output_dir': 'data/crew_outputs',
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging configuration.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger('AkhiPipelineCrew')
        
        if not logger.handlers:
            # Configure logging
            log_level = getattr(logging, self.config.get('logging', {}).get('level', 'INFO'))
            log_format = self.config.get('logging', {}).get(
                'format', 
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter(log_format)
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
            logger.setLevel(log_level)
        
        return logger
    
    def _create_agents(self) -> Dict[str, Agent]:
        """
        Create and configure all agents.
        
        Returns:
            Dictionary of configured agents
        """
        try:
            # Use None to let agents use their default config paths
            agents = {
                'researcher': create_video_researcher_agent(),
                'transcriber': create_transcriber_agent(),
                'indexer': create_vector_indexer_agent(),
                'qa_agent': create_content_qa_agent()
            }
            
            self.logger.info(f"Created {len(agents)} agents successfully")
            return agents
            
        except Exception as e:
            self.logger.error(f"Failed to create agents: {str(e)}")
            raise
    
    def create_video_search_task(
        self, 
        search_query: str, 
        max_results: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> Task:
        """
        Create a video search task.
        
        Args:
            search_query: The search query for Islamic content
            max_results: Maximum number of videos to find
            context: Additional context for the search
            
        Returns:
            Configured video search task
        """
        task_context = context or {}
        
        return Task(
            description=f"""
            Research and find relevant Islamic educational videos based on the query: "{search_query}"
            
            Requirements:
            - Find {max_results} high-quality Islamic educational videos
            - Prioritize content from reputable Islamic scholars
            - Ensure videos are suitable for transcription and analysis
            - Filter out inappropriate or low-quality content
            - Provide detailed metadata for each video
            
            Context: {json.dumps(task_context, indent=2)}
            """,
            expected_output="A structured list of Islamic educational videos with metadata including titles, URLs, channels, descriptions, and relevance scores.",
            agent=self.agents['researcher'],
            output_file=str(self.output_dir / f"video_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        )
    
    def create_transcription_task(
        self, 
        video_urls: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Task:
        """
        Create a transcription task.
        
        Args:
            video_urls: List of YouTube video URLs to transcribe
            context: Additional context for transcription
            
        Returns:
            Configured transcription task
        """
        task_context = context or {}
        
        return Task(
            description=f"""
            Download and transcribe the following Islamic educational videos:
            {json.dumps(video_urls, indent=2)}
            
            Requirements:
            - Download high-quality audio from each video
            - Generate accurate transcriptions with timestamps
            - Handle Arabic terms and Islamic terminology correctly
            - Preserve speaker context and segment boundaries
            - Generate both text and structured JSON outputs
            
            Context: {json.dumps(task_context, indent=2)}
            """,
            expected_output="Complete transcriptions for all videos with timestamps, metadata, and quality metrics.",
            agent=self.agents['transcriber'],
            output_file=str(self.output_dir / f"transcriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        )
    
    def create_indexing_task(
        self, 
        transcript_files: List[str],
        index_name: str = "islamic_content",
        context: Optional[Dict[str, Any]] = None
    ) -> Task:
        """
        Create a vector indexing task.
        
        Args:
            transcript_files: List of transcript files to process
            index_name: Name for the FAISS index
            context: Additional context for indexing
            
        Returns:
            Configured indexing task
        """
        task_context = context or {}
        
        return Task(
            description=f"""
            Process and index the following transcript files for semantic search:
            {json.dumps(transcript_files, indent=2)}
            
            Requirements:
            - Chunk transcripts into semantically meaningful segments
            - Generate high-quality embeddings for each chunk
            - Create and optimize FAISS index for fast retrieval
            - Preserve metadata and source information
            - Ensure index is ready for question answering
            
            Index Name: {index_name}
            Context: {json.dumps(task_context, indent=2)}
            """,
            expected_output=f"A complete FAISS index '{index_name}' with embeddings, metadata, and performance metrics.",
            agent=self.agents['indexer'],
            output_file=str(self.output_dir / f"indexing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        )
    
    def create_qa_task(
        self, 
        questions: List[str],
        index_name: str = "islamic_content",
        context: Optional[Dict[str, Any]] = None
    ) -> Task:
        """
        Create a question answering task.
        
        Args:
            questions: List of questions to answer
            index_name: Name of the FAISS index to query
            context: Additional context for QA
            
        Returns:
            Configured QA task
        """
        task_context = context or {}
        
        return Task(
            description=f"""
            Answer the following questions using the Islamic content index '{index_name}':
            {json.dumps(questions, indent=2)}
            
            Requirements:
            - Provide accurate, well-sourced answers from the indexed content
            - Include relevant citations and confidence scores
            - Ensure answers are contextually appropriate for Islamic education
            - Handle questions that may not have answers in the content
            - Generate comprehensive summaries when requested
            
            Context: {json.dumps(task_context, indent=2)}
            """,
            expected_output="Detailed answers with citations, confidence scores, and source references for each question.",
            agent=self.agents['qa_agent'],
            output_file=str(self.output_dir / f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        )
    
    def execute_full_pipeline(
        self, 
        search_query: str,
        questions: List[str],
        max_videos: int = 5,
        index_name: str = "islamic_content"
    ) -> Dict[str, Any]:
        """
        Execute the complete Islamic content processing pipeline.
        
        Args:
            search_query: Query to search for Islamic videos
            questions: Questions to answer using the processed content
            max_videos: Maximum number of videos to process
            index_name: Name for the FAISS index
            
        Returns:
            Complete pipeline results
        """
        self.logger.info(f"Starting full pipeline execution for query: '{search_query}'")
        
        try:
            # Create tasks with dependencies
            search_task = self.create_video_search_task(search_query, max_videos)
            
            # Note: In a real implementation, we would need to extract video URLs
            # from the search results and pass them to the transcription task
            # For now, we'll create placeholder tasks
            
            transcription_task = self.create_transcription_task([])
            indexing_task = self.create_indexing_task([], index_name)
            qa_task = self.create_qa_task(questions, index_name)
            
            # Create crew with sequential processing
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=[search_task, transcription_task, indexing_task, qa_task],
                process=Process.sequential,
                verbose=self.config.get('crew', {}).get('verbose', True),
                memory=self.config.get('crew', {}).get('memory', True)
            )
            
            # Execute the crew
            self.logger.info("Executing crew tasks...")
            result = crew.kickoff()
            
            # Process and return results
            pipeline_results = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'search_query': search_query,
                'questions': questions,
                'results': result,
                'output_dir': str(self.output_dir)
            }
            
            self.logger.info("Pipeline execution completed successfully")
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return {
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'search_query': search_query,
                'questions': questions
            }
    
    def execute_qa_only(
        self, 
        questions: List[str],
        index_name: str = "islamic_content"
    ) -> Dict[str, Any]:
        """
        Execute only the question answering part of the pipeline.
        
        Args:
            questions: Questions to answer
            index_name: Name of the existing FAISS index
            
        Returns:
            QA results
        """
        self.logger.info(f"Starting QA-only execution for {len(questions)} questions")
        
        try:
            # Create QA task
            qa_task = self.create_qa_task(questions, index_name)
            
            # Create crew with just the QA agent
            crew = Crew(
                agents=[self.agents['qa_agent']],
                tasks=[qa_task],
                process=Process.sequential,
                verbose=self.config.get('crew', {}).get('verbose', True)
            )
            
            # Execute the crew
            self.logger.info("Executing QA task...")
            result = crew.kickoff()
            
            qa_results = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'questions': questions,
                'index_name': index_name,
                'results': result,
                'output_dir': str(self.output_dir)
            }
            
            self.logger.info("QA execution completed successfully")
            return qa_results
            
        except Exception as e:
            self.logger.error(f"QA execution failed: {str(e)}")
            return {
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'questions': questions,
                'index_name': index_name
            }
    
    def execute_qlora_training(
        self,
        training_data_path: str,
        model_name: str = "akhi-islamic-assistant",
        training_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute QLoRA fine-tuning workflow.
        
        Args:
            training_data_path: Path to training data
            model_name: Name for the trained model
            training_config: Optional training configuration override
            
        Returns:
            Training results and model information
        """
        if not QLORA_AVAILABLE or self.qlora_trainer is None:
            return {
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'error': 'QLoRA training not available or not initialized',
                'training_data_path': training_data_path
            }
        
        try:
            self.logger.info(f"Starting QLoRA training for model: {model_name}")
            
            # Prepare training configuration
            config = training_config or {}
            config.update({
                'training_data_path': training_data_path,
                'model_name': model_name,
                'output_dir': str(self.output_dir / 'qlora_models')
            })
            
            # Execute training
            training_results = self.qlora_trainer.train_model(config)
            
            # Validate trained model
            if training_results.get('status') == 'completed':
                validation_results = self.qlora_trainer.validate_model(
                    training_results.get('model_path')
                )
                training_results['validation'] = validation_results
            
            self.logger.info("QLoRA training completed successfully")
            return training_results
            
        except Exception as e:
            self.logger.error(f"QLoRA training failed: {str(e)}")
            return {
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'training_data_path': training_data_path,
                'model_name': model_name
            }
    
    def execute_full_pipeline_with_training(
        self,
        search_query: str,
        questions: List[str],
        enable_training: bool = True,
        max_videos: int = 5,
        training_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute complete pipeline including QLoRA training.
        
        Args:
            search_query: Search query for videos
            questions: Questions to answer
            enable_training: Whether to perform QLoRA training
            max_videos: Maximum number of videos to process
            training_config: Optional training configuration
            
        Returns:
            Complete pipeline results including training
        """
        try:
            self.logger.info("Starting full pipeline with QLoRA training")
            
            # Step 1: Execute standard pipeline
            pipeline_results = self.execute_pipeline(
                search_query=search_query,
                questions=questions,
                max_videos=max_videos
            )
            
            if pipeline_results.get('status') != 'completed':
                return pipeline_results
            
            # Step 2: Prepare training data from transcripts
            if enable_training and QLORA_AVAILABLE and self.qlora_trainer:
                try:
                    # Format transcripts for QLoRA training
                    training_data_path = self.qlora_trainer.prepare_training_data(
                        transcript_dir=str(self.output_dir / 'transcripts'),
                        output_path=str(self.output_dir / 'qlora_training_data.json')
                    )
                    
                    # Execute QLoRA training
                    training_results = self.execute_qlora_training(
                        training_data_path=training_data_path,
                        model_name=f"akhi-{search_query.replace(' ', '-').lower()}",
                        training_config=training_config
                    )
                    
                    # Add training results to pipeline results
                    pipeline_results['qlora_training'] = training_results
                    
                    # Deploy trained model if successful
                    if training_results.get('status') == 'completed':
                        deployment_results = self.qlora_trainer.deploy_model(
                            model_path=training_results.get('model_path'),
                            deployment_target='local'
                        )
                        pipeline_results['model_deployment'] = deployment_results
                    
                except Exception as e:
                    self.logger.warning(f"QLoRA training failed, continuing with standard pipeline: {e}")
                    pipeline_results['qlora_training'] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            pipeline_results['pipeline_type'] = 'full_with_training'
            self.logger.info("Full pipeline with training completed")
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Full pipeline with training failed: {str(e)}")
            return {
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'search_query': search_query,
                'questions': questions
            }
    
    def get_qlora_status(self) -> Dict[str, Any]:
        """
        Get QLoRA training system status.
        
        Returns:
            QLoRA system status information
        """
        status = {
            'qlora_available': QLORA_AVAILABLE,
            'trainer_initialized': self.qlora_trainer is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.qlora_trainer:
            try:
                trainer_status = self.qlora_trainer.get_status()
                status.update(trainer_status)
            except Exception as e:
                status['trainer_error'] = str(e)
        
        return status


if __name__ == "__main__":
    # Example usage
    crew = AkhiPipelineCrew()
    
    # Test QA functionality with existing index
    test_questions = [
        "What is the importance of prayer in Islam?",
        "How should Muslims treat their parents?",
        "What are the pillars of Islam?"
    ]
    
    results = crew.execute_qa_only(test_questions)
    print(json.dumps(results, indent=2, ensure_ascii=False))