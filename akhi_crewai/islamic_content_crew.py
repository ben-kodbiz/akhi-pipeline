#!/usr/bin/env python3
"""
Islamic Content Processing Crew

This module orchestrates a team of specialized agents to process Islamic educational content:
1. Video Researcher: Finds relevant Islamic YouTube videos
2. Transcriber: Downloads and transcribes video content
3. Vector Indexer: Processes and indexes content for search
4. Content QA: Answers questions and generates summaries

Author: Assistant
Date: 2024
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# CrewAI imports
from crewai import Crew, Task, Process

# Add agents directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
from agents import (
    create_video_researcher_agent,
    create_transcriber_agent,
    create_vector_indexer_agent,
    create_content_qa_agent
)


class IslamicContentCrew:
    """
    Islamic Content Processing Crew
    
    Orchestrates a team of agents to:
    1. Research and find Islamic educational videos
    2. Download and transcribe video content
    3. Process and index content for semantic search
    4. Answer questions and generate summaries
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Islamic Content Crew.
        
        Args:
            config_path: Path to the crew configuration file
        """
        self.config = self._load_config(config_path)
        self.agents = self._create_agents()
        self.crew = None
        
        # Create output directory
        self.output_dir = Path(self.config.get('output_dir', 'data/crew_outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
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
                'config/crew_config.yaml'
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
                'local_llm': {
                    'model_name': 'lm_studio/qwen-3-14b',
                    'base_url': 'http://192.168.0.74:1234/v1',
                    'api_key': None,
                    'temperature': 0.7,
                    'max_tokens': 2048
                }
            }
    
    def _create_agents(self) -> Dict[str, Any]:
        """
        Create all agents for the crew.
        
        Returns:
            Dictionary of agent instances
        """
        config_path = os.path.join(
            os.path.dirname(__file__), 
            'config/crew_config.yaml'
        )
        
        return {
            'video_researcher': create_video_researcher_agent(config_path),
            'transcriber': create_transcriber_agent(config_path),
            'vector_indexer': create_vector_indexer_agent(config_path),
            'content_qa': create_content_qa_agent(config_path)
        }
    
    def create_research_task(self, search_query: str, max_results: int = 10) -> Task:
        """
        Create a video research task.
        
        Args:
            search_query: Query to search for Islamic videos
            max_results: Maximum number of videos to find
            
        Returns:
            Task instance for video research
        """
        return Task(
            description=f"""
            Research and find relevant Islamic educational videos on YouTube.
            
            Search Query: {search_query}
            Maximum Results: {max_results}
            
            Requirements:
            1. Search for Islamic educational content related to the query
            2. Filter results to ensure content is appropriate and educational
            3. Prioritize videos from reputable Islamic scholars and institutions
            4. Extract video metadata including title, description, duration, and URL
            5. Provide a ranked list of the most relevant videos
            
            Output Format:
            - JSON list of video metadata
            - Each entry should include: title, url, description, duration, channel
            - Videos should be ranked by relevance and educational value
            """,
            agent=self.agents['video_researcher'],
            expected_output="JSON list of Islamic educational videos with metadata"
        )
    
    def create_transcription_task(self, video_urls: List[str]) -> Task:
        """
        Create a transcription task.
        
        Args:
            video_urls: List of YouTube video URLs to transcribe
            
        Returns:
            Task instance for video transcription
        """
        urls_text = "\n".join([f"- {url}" for url in video_urls])
        
        return Task(
            description=f"""
            Download and transcribe Islamic educational videos.
            
            Video URLs to process:
            {urls_text}
            
            Requirements:
            1. Download each video using the YouTube downloader tool
            2. Extract audio and transcribe using the transcription tool
            3. Clean and format transcripts for readability
            4. Handle Arabic terms and Islamic terminology appropriately
            5. Include timestamps and speaker identification if available
            
            Output Format:
            - JSON object with transcripts for each video
            - Include video metadata and transcript text
            - Ensure proper formatting and structure
            """,
            agent=self.agents['transcriber'],
            expected_output="JSON object containing transcripts for all processed videos"
        )
    
    def create_indexing_task(self, transcripts: Dict[str, Any]) -> Task:
        """
        Create a vector indexing task.
        
        Args:
            transcripts: Dictionary of video transcripts
            
        Returns:
            Task instance for content indexing
        """
        return Task(
            description=f"""
            Process and index Islamic educational content for semantic search.
            
            Input: Transcripts from {len(transcripts)} videos
            
            Requirements:
            1. Chunk transcripts into semantically meaningful segments
            2. Generate embeddings for each text chunk
            3. Store embeddings in FAISS vector database
            4. Create metadata index for efficient retrieval
            5. Optimize for Islamic terminology and concepts
            
            Processing Steps:
            1. Use TextChunkerTool to segment transcripts
            2. Use EmbedderTool to generate embeddings
            3. Use FAISSStorageTool to store in vector database
            4. Verify indexing with test queries
            
            Output Format:
            - Confirmation of successful indexing
            - Statistics on processed chunks and embeddings
            - Index metadata and configuration details
            """,
            agent=self.agents['vector_indexer'],
            expected_output="Indexing completion report with statistics and metadata"
        )
    
    def create_qa_task(self, questions: List[str], 
                      index_name: str = "islamic_content") -> Task:
        """
        Create a question answering task.
        
        Args:
            questions: List of questions to answer
            index_name: Name of the FAISS index to query
            
        Returns:
            Task instance for question answering
        """
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        
        return Task(
            description=f"""
            Answer questions about Islamic content using RAG-based approach.
            
            Questions to answer:
            {questions_text}
            
            Index to query: {index_name}
            
            Requirements:
            1. Use FAISS query tool to retrieve relevant context
            2. Generate accurate answers using retrieved content
            3. Provide proper citations and references
            4. Ensure answers are Islamically accurate and appropriate
            5. Include confidence scores for each answer
            
            Processing Steps:
            1. Query vector database for relevant context
            2. Use AnswerGeneratorTool to generate responses
            3. Validate answers for Islamic accuracy
            4. Format responses with citations
            
            Output Format:
            - JSON object with answers for each question
            - Include answer text, citations, and confidence scores
            - Provide source references and metadata
            """,
            agent=self.agents['content_qa'],
            expected_output="JSON object containing answers with citations and confidence scores"
        )
    
    def create_summarization_task(self, content: str, 
                                 summary_type: str = "abstractive") -> Task:
        """
        Create a content summarization task.
        
        Args:
            content: Text content to summarize
            summary_type: Type of summary (extractive, abstractive, hybrid)
            
        Returns:
            Task instance for content summarization
        """
        return Task(
            description=f"""
            Generate a comprehensive summary of Islamic educational content.
            
            Content Length: {len(content)} characters
            Summary Type: {summary_type}
            
            Requirements:
            1. Create a {summary_type} summary of the provided content
            2. Focus on key Islamic concepts and teachings
            3. Maintain accuracy of religious terminology
            4. Structure summary with clear sections and points
            5. Include important quotes and references
            
            Processing Steps:
            1. Use SummarizerTool with Islamic focus keywords
            2. Ensure proper handling of Arabic terms
            3. Validate summary for completeness and accuracy
            4. Format for readability and structure
            
            Output Format:
            - Well-structured summary with clear sections
            - Key points and important concepts highlighted
            - Proper formatting and Islamic terminology
            """,
            agent=self.agents['content_qa'],
            expected_output="Comprehensive summary of Islamic educational content"
        )
    
    def process_islamic_content(self, search_query: str, 
                              questions: Optional[List[str]] = None,
                              max_videos: int = 5) -> Dict[str, Any]:
        """
        Complete workflow to process Islamic content.
        
        Args:
            search_query: Query to search for Islamic videos
            questions: Optional list of questions to answer
            max_videos: Maximum number of videos to process
            
        Returns:
            Results from the complete workflow
        """
        print(f"🚀 Starting Islamic Content Processing Workflow")
        print(f"📝 Search Query: {search_query}")
        print(f"🎥 Max Videos: {max_videos}")
        
        # Create tasks
        tasks = []
        
        # Task 1: Research videos
        research_task = self.create_research_task(search_query, max_videos)
        tasks.append(research_task)
        
        # Task 2: Transcribe videos (depends on research results)
        # Note: In a real implementation, we'd need to pass results between tasks
        # For now, we'll create a placeholder
        sample_urls = ["https://youtube.com/watch?v=sample1"]  # Placeholder
        transcription_task = self.create_transcription_task(sample_urls)
        tasks.append(transcription_task)
        
        # Task 3: Index content
        sample_transcripts = {"video1": "sample transcript"}  # Placeholder
        indexing_task = self.create_indexing_task(sample_transcripts)
        tasks.append(indexing_task)
        
        # Task 4: Answer questions (if provided)
        if questions:
            qa_task = self.create_qa_task(questions)
            tasks.append(qa_task)
        
        # Create and run crew
        crew_config = self.config.get('crew', {})
        
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=tasks,
            process=Process.sequential if crew_config.get('process') == 'sequential' else Process.hierarchical,
            verbose=crew_config.get('verbose', True),
            memory=crew_config.get('memory', True),
            max_rpm=crew_config.get('max_rpm', 10),
            share_crew=crew_config.get('share_crew', False)
        )
        
        try:
            print("🔄 Executing crew workflow...")
            result = self.crew.kickoff()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"islamic_content_processing_{timestamp}.json"
            
            output_data = {
                'timestamp': timestamp,
                'search_query': search_query,
                'max_videos': max_videos,
                'questions': questions,
                'result': str(result),
                'crew_config': crew_config
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Workflow completed successfully!")
            print(f"📄 Results saved to: {output_file}")
            
            return {
                'success': True,
                'result': result,
                'output_file': str(output_file),
                'timestamp': timestamp
            }
            
        except Exception as e:
            print(f"❌ Workflow failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def answer_questions_only(self, questions: List[str], 
                            index_name: str = "islamic_content") -> Dict[str, Any]:
        """
        Answer questions using existing indexed content.
        
        Args:
            questions: List of questions to answer
            index_name: Name of the FAISS index to query
            
        Returns:
            Results from question answering
        """
        print(f"❓ Answering {len(questions)} questions using index: {index_name}")
        
        # Create QA task
        qa_task = self.create_qa_task(questions, index_name)
        
        # Create crew with only QA agent
        crew_config = self.config.get('crew', {})
        
        self.crew = Crew(
            agents=[self.agents['content_qa']],
            tasks=[qa_task],
            process=Process.sequential,
            verbose=crew_config.get('verbose', True),
            memory=crew_config.get('memory', True)
        )
        
        try:
            print("🔄 Processing questions...")
            result = self.crew.kickoff()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"qa_results_{timestamp}.json"
            
            output_data = {
                'timestamp': timestamp,
                'questions': questions,
                'index_name': index_name,
                'result': str(result)
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Questions answered successfully!")
            print(f"📄 Results saved to: {output_file}")
            
            return {
                'success': True,
                'result': result,
                'output_file': str(output_file),
                'timestamp': timestamp
            }
            
        except Exception as e:
            print(f"❌ Question answering failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_crew_info(self) -> Dict[str, Any]:
        """
        Get information about the crew and its agents.
        
        Returns:
            Crew information dictionary
        """
        return {
            'agents': {
                name: {
                    'role': agent.role,
                    'goal': agent.goal,
                    'tools': [tool.__class__.__name__ for tool in agent.tools]
                }
                for name, agent in self.agents.items()
            },
            'config': self.config,
            'output_dir': str(self.output_dir)
        }


def create_islamic_content_crew(config_path: Optional[str] = None) -> IslamicContentCrew:
    """
    Factory function to create an Islamic Content Crew.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured IslamicContentCrew instance
    """
    return IslamicContentCrew(config_path)


if __name__ == "__main__":
    # Demo usage
    print("🤖 Creating Islamic Content Processing Crew...")
    
    try:
        crew = create_islamic_content_crew()
        
        print("✅ Crew created successfully!")
        print("\n👥 Crew Information:")
        
        info = crew.get_crew_info()
        for agent_name, agent_info in info['agents'].items():
            print(f"\n🤖 {agent_name.title()}:")
            print(f"   Role: {agent_info['role']}")
            print(f"   Tools: {', '.join(agent_info['tools'])}")
        
        print("\n🚀 Crew ready for Islamic content processing!")
        print("\nAvailable workflows:")
        print("- process_islamic_content(search_query, questions, max_videos)")
        print("- answer_questions_only(questions, index_name)")
        
        # Example usage
        print("\n📝 Example: Answer questions about existing content")
        sample_questions = [
            "What are the five pillars of Islam?",
            "Explain the concept of Tawhid"
        ]
        
        print(f"Sample questions: {sample_questions}")
        print("Use: crew.answer_questions_only(sample_questions)")
        
    except Exception as e:
        print(f"❌ Error creating crew: {e}")
        import traceback
        traceback.print_exc()