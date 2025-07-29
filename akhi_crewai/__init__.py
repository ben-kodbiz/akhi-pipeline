#!/usr/bin/env python3
"""
Akhi CrewAI Package - Islamic Content Processing System

A comprehensive CrewAI-based system for processing Islamic educational content.
Includes specialized agents, tools, and workflows for:

- Video research and discovery
- Content transcription and processing
- Vector indexing and semantic search
- Question answering and summarization
- Islamic content analysis and citation

Components:
- Tools: Custom CrewAI tools for Islamic content processing
- Agents: Specialized agents for different workflow stages
- Crew: Orchestrated workflows for complete content processing

Author: Assistant
Date: 2024
Version: 1.0.0
"""

# Import tools
from .tools import (
    YouTubeSearchTool,
    YouTubeDownloaderTool, 
    TranscriptionTool,
    TextChunkerTool,
    EmbedderTool,
    FAISSStorageTool,
    FAISSQueryTool,
    SummarizerTool,
    AnswerGeneratorTool,
    # Input schemas
    YouTubeSearchInput,
    YouTubeDownloadInput,
    TranscriptionInput,
    ChunkingInput,
    EmbeddingInput,
    FAISSStoreInput,
    FAISSQueryInput,
    SummarizationInput,
    AnswerGenerationInput
)

# Import agents
from .agents import (
    VideoResearcherAgent,
    TranscriberAgent,
    VectorIndexerAgent,
    ContentQAAgent,
    # Factory functions
    create_video_researcher_agent,
    create_transcriber_agent,
    create_vector_indexer_agent,
    create_content_qa_agent
)

# Import crew orchestrator
from .islamic_content_crew import (
    IslamicContentCrew,
    create_islamic_content_crew
)

# Package metadata
__version__ = '1.0.0'
__author__ = 'Assistant'
__description__ = 'CrewAI-based Islamic content processing system'
__license__ = 'MIT'

# Public API
__all__ = [
    # Tools
    'YouTubeSearchTool',
    'YouTubeDownloaderTool',
    'TranscriptionTool', 
    'TextChunkerTool',
    'EmbedderTool',
    'FAISSStorageTool',
    'FAISSQueryTool',
    'SummarizerTool',
    'AnswerGeneratorTool',
    
    # Tool Input Schemas
    'YouTubeSearchInput',
    'YouTubeDownloadInput',
    'TranscriptionInput',
    'ChunkingInput',
    'EmbeddingInput',
    'FAISSStoreInput',
    'FAISSQueryInput',
    'SummarizationInput',
    'AnswerGenerationInput',
    
    # Agents
    'VideoResearcherAgent',
    'TranscriberAgent',
    'VectorIndexerAgent', 
    'ContentQAAgent',
    
    # Agent Factory Functions
    'create_video_researcher_agent',
    'create_transcriber_agent',
    'create_vector_indexer_agent',
    'create_content_qa_agent',
    
    # Crew Orchestrator
    'IslamicContentCrew',
    'create_islamic_content_crew'
]

# Package information
def get_package_info():
    """
    Get package information.
    
    Returns:
        Dictionary with package metadata
    """
    return {
        'name': 'akhi_crewai',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'license': __license__,
        'components': {
            'tools': 9,
            'agents': 4,
            'workflows': 1
        },
        'features': [
            'YouTube video research and discovery',
            'Audio transcription with Islamic terminology support',
            'Text chunking and embedding generation', 
            'FAISS vector storage and retrieval',
            'RAG-based question answering',
            'Content summarization (extractive/abstractive/hybrid)',
            'Citation generation and confidence scoring',
            'Islamic content optimization',
            'Multi-agent workflow orchestration'
        ]
    }

# Convenience functions
def create_complete_workflow(config_path=None):
    """
    Create a complete Islamic content processing workflow.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured IslamicContentCrew instance
    """
    return create_islamic_content_crew(config_path)

def get_available_tools():
    """
    Get list of available tools.
    
    Returns:
        List of tool class names
    """
    return [
        'YouTubeSearchTool',
        'YouTubeDownloaderTool', 
        'TranscriptionTool',
        'TextChunkerTool',
        'EmbedderTool',
        'FAISSStorageTool',
        'FAISSQueryTool',
        'SummarizerTool',
        'AnswerGeneratorTool'
    ]

def get_available_agents():
    """
    Get list of available agents.
    
    Returns:
        List of agent class names
    """
    return [
        'VideoResearcherAgent',
        'TranscriberAgent',
        'VectorIndexerAgent',
        'ContentQAAgent'
    ]

# Welcome message
def print_welcome():
    """
    Print welcome message with package information.
    """
    info = get_package_info()
    
    print(f"🕌 Welcome to {info['name']} v{info['version']}")
    print(f"📖 {info['description']}")
    print(f"\n🛠️  Available Components:")
    print(f"   📦 Tools: {info['components']['tools']}")
    print(f"   🤖 Agents: {info['components']['agents']}")
    print(f"   🔄 Workflows: {info['components']['workflows']}")
    print(f"\n✨ Key Features:")
    for feature in info['features']:
        print(f"   • {feature}")
    print(f"\n🚀 Ready for Islamic content processing!")

# Auto-print welcome message when imported
if __name__ != '__main__':
    # Only print welcome in interactive environments
    import sys
    if hasattr(sys, 'ps1') or 'jupyter' in sys.modules:
        print_welcome()