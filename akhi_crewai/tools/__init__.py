"""CrewAI Tools Package

This package contains custom tools for the CrewAI agentic system,
specialized for Islamic educational content processing.
"""

from .youtube_search import YouTubeSearchTool, YouTubeSearchInput
from .youtube_downloader import YouTubeDownloaderTool, YouTubeDownloadInput
from .transcriber import TranscriptionTool, TranscriptionInput
from .chunker import TextChunkerTool, ChunkingInput
from .embedder import EmbedderTool, EmbeddingInput
from .faiss_store import FAISSStorageTool, FAISSStorageInput
from .faiss_query import FAISSQueryTool, FAISSQueryInput
from .summarizer import SummarizerTool, SummarizationInput
from .answer_generator import AnswerGeneratorTool, AnswerGenerationInput

__all__ = [
    'YouTubeSearchTool',
    'YouTubeSearchInput', 
    'YouTubeDownloaderTool',
    'YouTubeDownloadInput',
    'TranscriptionTool',
    'TranscriptionInput',
    'TextChunkerTool',
    'ChunkingInput',
    'EmbedderTool',
    'EmbeddingInput',
    'FAISSStorageTool',
    'FAISSStorageInput',
    'FAISSQueryTool',
    'FAISSQueryInput',
    'SummarizerTool',
    'SummarizationInput',
    'AnswerGeneratorTool',
    'AnswerGenerationInput'
]

__version__ = '1.0.0'
__author__ = 'Akhi Data Builder Team'
__description__ = 'Custom CrewAI tools for Islamic educational content processing'