#!/usr/bin/env python3
"""
Transcriber Agent for CrewAI Agentic System

This agent specializes in downloading YouTube videos and transcribing them
into text using Whisper models. It handles the complete audio processing
pipeline from URL to transcript.

Author: Assistant
Date: 2024
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

# CrewAI imports
from crewai import Agent, LLM

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
from transcriber import TranscriptionTool
from youtube_downloader import YouTubeDownloaderTool

# Import unified config loader
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.config_loader import get_config


class TranscriberAgent:
    """
    Transcriber Agent for YouTube video processing and transcription.
    
    This agent is responsible for:
    - Downloading YouTube videos as audio files
    - Transcribing audio to text using Whisper models
    - Managing transcription quality and output formats
    - Handling multiple video processing workflows
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Transcriber Agent.
        
        Args:
            config_path: Path to the crew configuration file
        """
        self.config = self._load_config(config_path)
        self.llm = self._setup_llm()
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from unified config.yaml.
        
        Args:
            config_path: Path to configuration file (ignored, using unified config)
            
        Returns:
            Configuration dictionary
        """
        try:
            config_loader = get_config()
            return {
                'llm': config_loader.get_llm_config(),
                'agents': config_loader.get_crewai_config().get('agents', {}),
                'transcription': config_loader.get_transcription_config()
            }
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
        
        # Fallback configuration
        return {
            'llm': {
                'model_name': 'lm_studio/qwen-3-14b',
                'base_url': 'http://localhost:1234/v1',
                'api_key': None,
                'temperature': 0.7,
                'max_tokens': 2048
            },
            'agents': {
                'transcriber': {
                    'role': 'Audio Transcription Specialist',
                    'goal': 'Convert audio content to accurate text transcriptions',
                    'backstory': 'Expert in speech recognition and audio processing',
                    'verbose': True,
                    'allow_delegation': False,
                    'max_iter': 3,
                    'memory': True
                }
            },
            'transcription': {
                'model_name': 'openai/whisper-large-v3',
                'device': 'auto',
                'language': 'auto',
                'task': 'transcribe',
                'batch_size': 1,
                'chunk_length': 30,
                'return_timestamps': True,
                'output_dir': 'data/transcripts',
                'output_format': 'json'
            }
        }
    
    def _setup_llm(self):
        """
        Setup the local LLM for the agent.
        Uses local GGUF model when available, falls back to HTTP API.
        
        Returns:
            Configured LLM instance (LocalGGUFLLM or LLM)
        """
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from utils.llm_config import LLMConfig
        
        llm_config = LLMConfig()
        return llm_config.get_local_llm()
    
    def _setup_tools(self) -> list:
        """
        Setup tools for the agent.
        
        Returns:
            List of tools for the agent
        """
        return [
            YouTubeDownloaderTool(),
            TranscriptionTool()
        ]
    
    def _create_agent(self) -> Agent:
        """
        Create the CrewAI agent instance.
        
        Returns:
            Configured Agent instance
        """
        agent_config = self.config.get('agents', {}).get('transcriber', {})
        
        return Agent(
            role=agent_config.get(
                'role', 
                'Audio-to-Text Transcriber'
            ),
            goal=agent_config.get(
                'goal',
                'Download and transcribe YouTube videos with high accuracy'
            ),
            backstory=agent_config.get(
                'backstory',
                'You are a professional audio processing specialist with expertise '
                'in speech recognition and transcription. You ensure accurate and '
                'clean transcripts.'
            ),
            tools=self.tools,
            llm=self.llm,
            max_iter=agent_config.get('max_iter', 3),
            max_execution_time=agent_config.get('max_execution_time', 600),
            verbose=agent_config.get('verbose', True),
            allow_delegation=agent_config.get('allow_delegation', False)
        )
    
    def get_agent(self) -> Agent:
        """
        Get the configured agent instance.
        
        Returns:
            The CrewAI Agent instance
        """
        return self.agent
    
    def download_and_transcribe(self, video_url: str, 
                               model_size: str = "base",
                               include_timestamps: bool = True) -> Dict[str, Any]:
        """
        Download a YouTube video and transcribe it.
        
        Args:
            video_url: YouTube video URL
            model_size: Whisper model size to use
            include_timestamps: Whether to include timestamps
            
        Returns:
            Transcription results dictionary
        """
        try:
            # Step 1: Download video as audio
            downloader_tool = self.tools[0]  # YouTubeDownloaderTool
            download_result = downloader_tool._run(
                video_url=video_url,
                download_audio=True,
                download_video=False
            )
            
            if not download_result.get('success', False):
                return {
                    'success': False,
                    'error': f"Failed to download video: {download_result.get('error', 'Unknown error')}"
                }
            
            audio_file = download_result.get('audio_file')
            if not audio_file:
                return {
                    'success': False,
                    'error': 'No audio file path returned from download'
                }
            
            # Step 2: Transcribe audio
            transcription_tool = self.tools[1]  # TranscriptionTool
            transcription_result = transcription_tool._run(
                audio_file_path=audio_file,
                model_size=model_size,
                device="auto",
                include_timestamps=include_timestamps,
                output_format="json"
            )
            
            return {
                'success': True,
                'video_url': video_url,
                'audio_file': audio_file,
                'transcription': transcription_result,
                'model_size': model_size
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Transcription pipeline failed: {str(e)}"
            }
    
    def batch_transcribe(self, video_urls: List[str], 
                        model_size: str = "base") -> List[Dict[str, Any]]:
        """
        Transcribe multiple videos in batch.
        
        Args:
            video_urls: List of YouTube video URLs
            model_size: Whisper model size to use
            
        Returns:
            List of transcription results
        """
        results = []
        
        for i, url in enumerate(video_urls, 1):
            print(f"Processing video {i}/{len(video_urls)}: {url}")
            
            result = self.download_and_transcribe(
                video_url=url,
                model_size=model_size
            )
            
            results.append(result)
            
            if not result.get('success', False):
                print(f"❌ Failed to process {url}: {result.get('error', 'Unknown error')}")
            else:
                print(f"✅ Successfully processed {url}")
        
        return results


def create_transcriber_agent(config_path: Optional[str] = None) -> Agent:
    """
    Factory function to create a Transcriber Agent.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured CrewAI Agent instance
    """
    transcriber = TranscriberAgent(config_path)
    return transcriber.get_agent()


if __name__ == "__main__":
    # Demo usage
    print("🎤 Creating Transcriber Agent...")
    
    try:
        agent = create_transcriber_agent()
        print(f"✅ Agent created successfully!")
        print(f"Role: {agent.role}")
        print(f"Goal: {agent.goal}")
        print(f"Tools: {[tool.__class__.__name__ for tool in agent.tools]}")
        
        # Test transcription workflow
        transcriber = TranscriberAgent()
        print("\n🎤 Transcriber Agent ready for video processing")
        print("Use download_and_transcribe(video_url) to process videos")
        
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        import traceback
        traceback.print_exc()