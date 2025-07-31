#!/usr/bin/env python3
"""
Video Researcher Agent for CrewAI Agentic System

This agent specializes in finding relevant Islamic YouTube videos based on topics
and keywords. It uses the YouTubeSearchTool to discover high-quality educational
content from Islamic scholars and educators.

Author: Assistant
Date: 2024
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# CrewAI imports
from crewai import Agent, LLM

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
from tools.youtube_search import YouTubeSearchTool


class VideoResearcherAgent:
    """
    Video Researcher Agent for Islamic content discovery on YouTube.
    
    This agent is responsible for:
    - Searching YouTube for relevant Islamic educational content
    - Filtering and ranking videos based on quality and relevance
    - Selecting appropriate videos for transcription and processing
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Video Researcher Agent.
        
        Args:
            config_path: Path to the crew configuration file
        """
        self.config = self._load_config(config_path)
        self.llm = self._setup_llm()
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
    
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
                'local_llm': {
                    'model_name': 'lm_studio/qwen-3-14b',
                    'base_url': 'http://192.168.0.74:1234/v1',
                    'api_key': None,
                    'temperature': 0.7,
                    'max_tokens': 2048
                },
                'agents': {
                    'video_researcher': {
                        'role': 'YouTube Research Assistant',
                        'goal': 'Find relevant Islamic YouTube videos based on topics and keywords',
                        'backstory': 'You are an expert Islamic content researcher with deep knowledge of Islamic scholars, topics, and YouTube content. You excel at finding high-quality educational videos.',
                        'max_iter': 5,
                        'max_execution_time': 300,
                        'verbose': True,
                        'allow_delegation': False
                    }
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
        return [YouTubeSearchTool()]
    
    def _create_agent(self) -> Agent:
        """
        Create the CrewAI agent instance.
        
        Returns:
            Configured Agent instance
        """
        agent_config = self.config.get('agents', {}).get('video_researcher', {})
        
        return Agent(
            role=agent_config.get(
                'role', 
                'YouTube Research Assistant'
            ),
            goal=agent_config.get(
                'goal',
                'Find relevant Islamic YouTube videos based on topics and keywords'
            ),
            backstory=agent_config.get(
                'backstory',
                'You are an expert Islamic content researcher with deep knowledge '
                'of Islamic scholars, topics, and YouTube content. You excel at '
                'finding high-quality educational videos.'
            ),
            tools=self.tools,
            llm=self.llm,
            max_iter=agent_config.get('max_iter', 5),
            max_execution_time=agent_config.get('max_execution_time', 300),
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
    
    def search_videos(self, topic: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search for videos on a specific topic.
        
        Args:
            topic: The topic to search for
            max_results: Maximum number of results to return
            
        Returns:
            Search results dictionary
        """
        search_tool = self.tools[0]  # YouTubeSearchTool
        
        return search_tool._run(
            query=topic,
            max_results=max_results,
            duration_filter="medium",
            quality_filter="high"
        )


def create_video_researcher_agent(config_path: Optional[str] = None) -> Agent:
    """
    Factory function to create a Video Researcher Agent.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured CrewAI Agent instance
    """
    researcher = VideoResearcherAgent(config_path)
    return researcher.get_agent()


if __name__ == "__main__":
    # Demo usage
    print("🔍 Creating Video Researcher Agent...")
    
    try:
        agent = create_video_researcher_agent()
        print(f"✅ Agent created successfully!")
        print(f"Role: {agent.role}")
        print(f"Goal: {agent.goal}")
        print(f"Tools: {[tool.__class__.__name__ for tool in agent.tools]}")
        
        # Test video search
        researcher = VideoResearcherAgent()
        print("\n🔍 Testing video search...")
        results = researcher.search_videos("Islamic prayer", max_results=3)
        print(f"Found {len(results.get('videos', []))} videos")
        
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        import traceback
        traceback.print_exc()