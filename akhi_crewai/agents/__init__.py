#!/usr/bin/env python3
"""
CrewAI Agents Package for Islamic Content Processing

This package contains specialized agents for processing Islamic educational content:
- VideoResearcherAgent: Finds relevant Islamic YouTube videos
- TranscriberAgent: Downloads and transcribes video content
- VectorIndexerAgent: Processes and indexes content for search
- ContentQAAgent: Answers questions and generates summaries

Author: Assistant
Date: 2024
"""

from .video_researcher import VideoResearcherAgent, create_video_researcher_agent
from .transcriber_agent import TranscriberAgent, create_transcriber_agent
from .vector_indexer import VectorIndexerAgent, create_vector_indexer_agent
from .content_qa import ContentQAAgent, create_content_qa_agent

__all__ = [
    # Agent Classes
    'VideoResearcherAgent',
    'TranscriberAgent', 
    'VectorIndexerAgent',
    'ContentQAAgent',
    
    # Factory Functions
    'create_video_researcher_agent',
    'create_transcriber_agent',
    'create_vector_indexer_agent',
    'create_content_qa_agent'
]

__version__ = '1.0.0'
__author__ = 'Assistant'
__description__ = 'CrewAI agents for Islamic content processing and analysis'