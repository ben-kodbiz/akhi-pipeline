"""CrewAI Tools Package

This package contains custom tools for the CrewAI agentic system,
specialized for Islamic educational content processing.
"""

from .youtube_search import YouTubeSearchTool, YouTubeSearchInput
from .youtube_downloader import YouTubeDownloaderTool, YouTubeDownloadInput

__all__ = [
    'YouTubeSearchTool',
    'YouTubeSearchInput', 
    'YouTubeDownloaderTool',
    'YouTubeDownloadInput'
]

__version__ = '1.0.0'
__author__ = 'Akhi Data Builder Team'
__description__ = 'Custom CrewAI tools for Islamic educational content processing'