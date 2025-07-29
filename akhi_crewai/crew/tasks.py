#!/usr/bin/env python3
"""
Task Definitions for Akhi Pipeline Crew - Phase 4

Individual task classes for structured workflow management:
- VideoSearchTask: Research and select Islamic videos
- TranscriptionTask: Download and transcribe content
- IndexingTask: Process and store embeddings
- QATask: Answer questions and generate summaries

Author: Assistant
Date: December 2024
Phase: 4 - Crew Orchestration
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from crewai import Task, Agent
from pydantic import BaseModel, Field


class TaskResult(BaseModel):
    """
    Standard task result structure.
    """
    task_id: str
    task_type: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    output_files: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseTaskDefinition(ABC):
    """
    Base class for all task definitions.
    """
    
    def __init__(self, task_id: str, output_dir: Path):
        self.task_id = task_id
        self.output_dir = output_dir
        self.logger = logging.getLogger(f'Task.{self.__class__.__name__}')
        self.result = TaskResult(
            task_id=task_id,
            task_type=self.__class__.__name__,
            status='pending'
        )
    
    @abstractmethod
    def create_task(self, agent: Agent, **kwargs) -> Task:
        """
        Create the CrewAI Task instance.
        
        Args:
            agent: The agent to assign to this task
            **kwargs: Task-specific parameters
            
        Returns:
            Configured CrewAI Task
        """
        pass
    
    def _generate_output_file(self, suffix: str = "") -> str:
        """
        Generate a unique output file path.
        
        Args:
            suffix: Optional suffix for the filename
            
        Returns:
            Full path to the output file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.task_id}_{timestamp}{suffix}.json"
        return str(self.output_dir / filename)
    
    def start_execution(self):
        """Mark task as started."""
        self.result.status = 'running'
        self.result.start_time = datetime.now().isoformat()
        self.logger.info(f"Task {self.task_id} started")
    
    def complete_execution(self, result_data: Dict[str, Any], output_files: List[str] = None):
        """Mark task as completed."""
        self.result.status = 'completed'
        self.result.end_time = datetime.now().isoformat()
        self.result.result_data = result_data
        self.result.output_files = output_files or []
        
        if self.result.start_time:
            start = datetime.fromisoformat(self.result.start_time)
            end = datetime.fromisoformat(self.result.end_time)
            self.result.duration_seconds = (end - start).total_seconds()
        
        self.logger.info(f"Task {self.task_id} completed in {self.result.duration_seconds:.2f}s")
    
    def fail_execution(self, error_message: str):
        """Mark task as failed."""
        self.result.status = 'failed'
        self.result.end_time = datetime.now().isoformat()
        self.result.error_message = error_message
        
        if self.result.start_time:
            start = datetime.fromisoformat(self.result.start_time)
            end = datetime.fromisoformat(self.result.end_time)
            self.result.duration_seconds = (end - start).total_seconds()
        
        self.logger.error(f"Task {self.task_id} failed: {error_message}")


class VideoSearchTask(BaseTaskDefinition):
    """
    Task for researching and selecting Islamic educational videos.
    """
    
    def create_task(
        self, 
        agent: Agent, 
        search_query: str,
        max_results: int = 10,
        scholar_preferences: List[str] = None,
        topic_filters: List[str] = None,
        **kwargs
    ) -> Task:
        """
        Create a video search task.
        
        Args:
            agent: Video researcher agent
            search_query: The search query for Islamic content
            max_results: Maximum number of videos to find
            scholar_preferences: Preferred Islamic scholars
            topic_filters: Topic-based filters
            **kwargs: Additional parameters
            
        Returns:
            Configured video search task
        """
        scholar_preferences = scholar_preferences or []
        topic_filters = topic_filters or []
        
        # Store task parameters
        self.result.metadata.update({
            'search_query': search_query,
            'max_results': max_results,
            'scholar_preferences': scholar_preferences,
            'topic_filters': topic_filters
        })
        
        output_file = self._generate_output_file('_video_search')
        
        description = f"""
        Research and find relevant Islamic educational videos based on the query: "{search_query}"
        
        SEARCH REQUIREMENTS:
        - Find up to {max_results} high-quality Islamic educational videos
        - Prioritize content from reputable Islamic scholars and institutions
        - Ensure videos are suitable for transcription and educational analysis
        - Filter out inappropriate, low-quality, or non-educational content
        - Focus on authentic Islamic teachings and scholarly discussions
        
        PREFERRED SCHOLARS: {', '.join(scholar_preferences) if scholar_preferences else 'Any reputable Islamic scholar'}
        TOPIC FILTERS: {', '.join(topic_filters) if topic_filters else 'General Islamic education'}
        
        QUALITY CRITERIA:
        - Clear audio quality for transcription
        - Educational or scholarly content
        - Appropriate length (5-60 minutes preferred)
        - Recent uploads (within last 5 years preferred)
        - High engagement metrics (views, likes, comments)
        
        OUTPUT FORMAT:
        Provide a structured JSON list with the following information for each video:
        - video_id: YouTube video ID
        - title: Video title
        - url: Full YouTube URL
        - channel: Channel name
        - channel_url: Channel URL
        - description: Video description (first 500 characters)
        - duration: Video duration in seconds
        - view_count: Number of views
        - upload_date: Upload date
        - relevance_score: Relevance to search query (0.0-1.0)
        - islamic_relevance: Islamic content relevance (0.0-1.0)
        - quality_score: Overall quality assessment (0.0-1.0)
        - scholar_match: Whether video features preferred scholars
        - topic_tags: Relevant Islamic topic tags
        """
        
        expected_output = f"A JSON array of {max_results} Islamic educational videos with complete metadata, relevance scores, and quality assessments, sorted by overall relevance and quality."
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            output_file=output_file
        )


class TranscriptionTask(BaseTaskDefinition):
    """
    Task for downloading and transcribing video content.
    """
    
    def create_task(
        self, 
        agent: Agent,
        video_data: List[Dict[str, Any]],
        audio_quality: str = 'best',
        transcription_model: str = 'base',
        include_timestamps: bool = True,
        **kwargs
    ) -> Task:
        """
        Create a transcription task.
        
        Args:
            agent: Transcriber agent
            video_data: List of video metadata from search task
            audio_quality: Audio download quality
            transcription_model: Whisper model size
            include_timestamps: Whether to include timestamps
            **kwargs: Additional parameters
            
        Returns:
            Configured transcription task
        """
        # Store task parameters
        self.result.metadata.update({
            'video_count': len(video_data),
            'audio_quality': audio_quality,
            'transcription_model': transcription_model,
            'include_timestamps': include_timestamps
        })
        
        output_file = self._generate_output_file('_transcriptions')
        
        video_list = json.dumps(video_data, indent=2, ensure_ascii=False)
        
        description = f"""
        Download and transcribe the following Islamic educational videos:
        
        VIDEO LIST:
        {video_list}
        
        DOWNLOAD REQUIREMENTS:
        - Download high-quality audio ({audio_quality} available quality)
        - Convert to appropriate format for transcription (MP3/WAV)
        - Verify audio integrity and quality
        - Handle download failures gracefully with retry logic
        - Respect YouTube's terms of service
        
        TRANSCRIPTION REQUIREMENTS:
        - Use Whisper model: {transcription_model}
        - Generate accurate transcriptions with proper punctuation
        - Handle Arabic terms and Islamic terminology correctly
        - Preserve speaker context and natural speech patterns
        - Include timestamps: {include_timestamps}
        - Generate confidence scores for transcription quality
        
        ISLAMIC CONTENT CONSIDERATIONS:
        - Properly transcribe Arabic phrases and Quranic verses
        - Maintain respectful formatting for religious content
        - Preserve context for Islamic terminology and concepts
        - Handle multilingual content (Arabic/English mix)
        
        OUTPUT FORMAT:
        For each video, provide:
        - video_id: YouTube video ID
        - title: Video title
        - download_status: Success/failure status
        - audio_file_path: Path to downloaded audio
        - transcription_status: Success/failure status
        - transcript_text: Full transcript text
        - transcript_segments: Timestamped segments (if enabled)
        - word_count: Total word count
        - duration_seconds: Audio duration
        - confidence_score: Average transcription confidence
        - language_detected: Detected primary language
        - arabic_content_detected: Whether Arabic content was found
        - processing_time: Time taken for transcription
        - error_messages: Any errors encountered
        """
        
        expected_output = f"Complete transcription results for all {len(video_data)} videos with full text, timestamps, metadata, and quality metrics in structured JSON format."
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            output_file=output_file
        )


class IndexingTask(BaseTaskDefinition):
    """
    Task for processing transcripts into searchable vector embeddings.
    """
    
    def create_task(
        self, 
        agent: Agent,
        transcript_data: List[Dict[str, Any]],
        index_name: str = "islamic_content",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        **kwargs
    ) -> Task:
        """
        Create a vector indexing task.
        
        Args:
            agent: Vector indexer agent
            transcript_data: Transcription results from previous task
            index_name: Name for the FAISS index
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
            embedding_model: Sentence transformer model name
            **kwargs: Additional parameters
            
        Returns:
            Configured indexing task
        """
        # Store task parameters
        self.result.metadata.update({
            'transcript_count': len(transcript_data),
            'index_name': index_name,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'embedding_model': embedding_model
        })
        
        output_file = self._generate_output_file('_indexing')
        
        transcript_summary = [
            {
                'video_id': t.get('video_id'),
                'title': t.get('title'),
                'word_count': t.get('word_count', 0),
                'status': t.get('transcription_status')
            }
            for t in transcript_data
        ]
        
        description = f"""
        Process and index the following transcript data for semantic search:
        
        TRANSCRIPT SUMMARY:
        {json.dumps(transcript_summary, indent=2, ensure_ascii=False)}
        
        CHUNKING REQUIREMENTS:
        - Split transcripts into {chunk_size}-token chunks with {chunk_overlap}-token overlap
        - Preserve semantic boundaries (sentences, paragraphs)
        - Maintain context across chunk boundaries
        - Handle Arabic text and Islamic terminology appropriately
        - Preserve timestamp and source metadata for each chunk
        
        EMBEDDING REQUIREMENTS:
        - Use embedding model: {embedding_model}
        - Generate high-quality vector representations
        - Ensure consistent embedding dimensions
        - Batch process for efficiency
        - Handle multilingual content (Arabic/English)
        
        FAISS INDEX REQUIREMENTS:
        - Create optimized FAISS index: {index_name}
        - Use appropriate index type for dataset size
        - Implement efficient similarity search
        - Store metadata alongside embeddings
        - Enable fast retrieval for question answering
        
        ISLAMIC CONTENT CONSIDERATIONS:
        - Preserve Islamic terminology and concepts
        - Maintain context for Quranic references
        - Handle Arabic transliterations consistently
        - Ensure respectful processing of religious content
        
        OUTPUT FORMAT:
        Provide comprehensive indexing results:
        - index_name: Name of created FAISS index
        - index_file_path: Path to FAISS index file
        - metadata_file_path: Path to metadata JSON file
        - total_chunks: Number of text chunks created
        - total_embeddings: Number of embeddings generated
        - embedding_dimensions: Vector dimensions
        - index_size_mb: Index file size in MB
        - processing_time: Total processing time
        - chunk_statistics: Statistics about chunk sizes and content
        - embedding_statistics: Statistics about embedding quality
        - video_coverage: Coverage statistics per video
        - arabic_content_ratio: Ratio of Arabic to English content
        - performance_metrics: Search performance benchmarks
        """
        
        expected_output = f"A complete FAISS index '{index_name}' with embeddings, metadata, performance metrics, and detailed processing statistics."
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            output_file=output_file
        )


class QATask(BaseTaskDefinition):
    """
    Task for answering questions and generating summaries using indexed content.
    """
    
    def create_task(
        self, 
        agent: Agent,
        questions: List[str],
        index_name: str = "islamic_content",
        max_context_chunks: int = 5,
        confidence_threshold: float = 0.7,
        include_citations: bool = True,
        **kwargs
    ) -> Task:
        """
        Create a question answering task.
        
        Args:
            agent: Content QA agent
            questions: List of questions to answer
            index_name: Name of the FAISS index to query
            max_context_chunks: Maximum context chunks per question
            confidence_threshold: Minimum confidence for answers
            include_citations: Whether to include source citations
            **kwargs: Additional parameters
            
        Returns:
            Configured QA task
        """
        # Store task parameters
        self.result.metadata.update({
            'question_count': len(questions),
            'index_name': index_name,
            'max_context_chunks': max_context_chunks,
            'confidence_threshold': confidence_threshold,
            'include_citations': include_citations
        })
        
        output_file = self._generate_output_file('_qa_results')
        
        questions_formatted = json.dumps(questions, indent=2, ensure_ascii=False)
        
        description = f"""
        Answer the following questions using the Islamic content index '{index_name}':
        
        QUESTIONS:
        {questions_formatted}
        
        RETRIEVAL REQUIREMENTS:
        - Query the FAISS index for relevant content chunks
        - Retrieve up to {max_context_chunks} most relevant chunks per question
        - Use semantic similarity for context selection
        - Filter results by relevance threshold
        - Preserve source metadata and timestamps
        
        ANSWER GENERATION REQUIREMENTS:
        - Provide accurate, well-sourced answers from indexed content
        - Maintain Islamic scholarly accuracy and sensitivity
        - Include confidence scores (minimum {confidence_threshold})
        - Handle questions without sufficient context gracefully
        - Preserve Arabic terms and Islamic terminology
        - Provide contextually appropriate responses
        
        CITATION REQUIREMENTS (if enabled: {include_citations}):
        - Include specific source references for each answer
        - Provide video titles, timestamps, and channel information
        - Link answers to original content segments
        - Maintain traceability to source material
        
        ISLAMIC CONTENT CONSIDERATIONS:
        - Ensure answers align with authentic Islamic teachings
        - Respect religious sensitivities and terminology
        - Provide balanced perspectives when appropriate
        - Handle controversial topics with scholarly care
        - Maintain respectful tone for religious content
        
        OUTPUT FORMAT:
        For each question, provide:
        - question: Original question text
        - answer: Comprehensive answer based on indexed content
        - confidence_score: Confidence in answer accuracy (0.0-1.0)
        - sources_found: Number of relevant sources found
        - context_chunks: Retrieved context chunks with metadata
        - citations: Source citations (if enabled)
        - processing_time: Time taken to answer question
        - status: Success/partial/failed status
        - error_message: Any errors or limitations
        - related_topics: Suggested related topics or questions
        
        SUMMARY STATISTICS:
        - total_questions: Number of questions processed
        - successful_answers: Number of successful answers
        - average_confidence: Average confidence score
        - total_processing_time: Total time for all questions
        - index_performance: Index query performance metrics
        """
        
        expected_output = f"Detailed answers with citations, confidence scores, and source references for all {len(questions)} questions, plus comprehensive performance statistics."
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            output_file=output_file
        )


class TaskOrchestrator:
    """
    Orchestrates task execution and manages dependencies.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.tasks: Dict[str, BaseTaskDefinition] = {}
        self.execution_order: List[str] = []
        self.logger = logging.getLogger('TaskOrchestrator')
    
    def add_task(self, task: BaseTaskDefinition) -> str:
        """
        Add a task to the orchestrator.
        
        Args:
            task: Task definition to add
            
        Returns:
            Task ID
        """
        self.tasks[task.task_id] = task
        self.execution_order.append(task.task_id)
        self.logger.info(f"Added task: {task.task_id}")
        return task.task_id
    
    def get_task_results(self) -> Dict[str, TaskResult]:
        """
        Get results from all tasks.
        
        Returns:
            Dictionary of task results
        """
        return {task_id: task.result for task_id, task in self.tasks.items()}
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of task execution.
        
        Returns:
            Execution summary
        """
        results = self.get_task_results()
        
        summary = {
            'total_tasks': len(results),
            'completed': len([r for r in results.values() if r.status == 'completed']),
            'failed': len([r for r in results.values() if r.status == 'failed']),
            'pending': len([r for r in results.values() if r.status == 'pending']),
            'running': len([r for r in results.values() if r.status == 'running']),
            'total_duration': sum(r.duration_seconds or 0 for r in results.values()),
            'execution_order': self.execution_order,
            'task_details': results
        }
        
        return summary