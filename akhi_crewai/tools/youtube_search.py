"""YouTube Search Tool for CrewAI

This tool provides YouTube search capabilities for the CrewAI agentic system.
It integrates with the existing YouTube search logic from the pipeline.
"""

import os
import sys
import yaml
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

# Add the pipeline directory to the path to import existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../pipeline'))

try:
    from agents.youtube_agent import YouTubeAgent
except ImportError:
    # Fallback implementation if pipeline module is not available
    pass

try:
    from youtube_search import YoutubeSearch
except ImportError:
    try:
        from youtubesearchpython import VideosSearch
    except ImportError:
        VideosSearch = None
        YoutubeSearch = None


class YouTubeSearchInput(BaseModel):
    """Input schema for YouTube search tool."""
    query: str = Field(..., description="Search query for YouTube videos")
    max_results: int = Field(default=10, description="Maximum number of results to return")
    duration: str = Field(default="medium", description="Video duration filter: short, medium, long, or any")
    upload_date: str = Field(default="any", description="Upload date filter: hour, today, week, month, year, or any")
    sort_by: str = Field(default="relevance", description="Sort results by: relevance, upload_date, view_count, rating")
    language: str = Field(default="en", description="Language filter for videos")
    region: str = Field(default="US", description="Region filter for videos")


class YouTubeSearchTool(BaseTool):
    """YouTube Search Tool for finding Islamic educational content."""
    
    name: str = "YouTube Search Tool"
    description: str = (
        "Search for YouTube videos based on keywords, topics, and filters. "
        "Specializes in finding Islamic educational content from reputable scholars. "
        "Returns structured video information including title, URL, duration, channel, and metadata."
    )
    args_schema: type[BaseModel] = YouTubeSearchInput
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the YouTube Search Tool.
        
        Args:
            config_path: Path to the configuration file
        """
        super().__init__()
        self._config = self._load_config(config_path)
        self._youtube_agent = None
        self._initialize_agent()
    
    @property
    def config(self):
        """Get the configuration."""
        return self._config
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
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
            # Return default configuration if file not found
            return {
                'youtube': {
                    'search': {
                        'max_results': 50,
                        'default_duration': 'medium',
                        'quality_filter': 'high',
                        'language': 'en',
                        'region': 'US'
                    }
                },
                'islamic_content': {
                    'preferred_scholars': [
                        'Nouman Ali Khan', 'Yasir Qadhi', 'Omar Suleiman',
                        'Mufti Menk', 'Abdur Raheem Green'
                    ],
                    'topics': [
                        'Quran', 'Hadith', 'Seerah', 'Fiqh', 'Aqeedah',
                        'Islamic History', 'Tafseer', 'Islamic Ethics'
                    ]
                }
            }
    
    def _initialize_agent(self):
        """Initialize the YouTube agent if available."""
        try:
            # Try to use existing YouTube agent
            pipeline_config_path = os.path.join(
                os.path.dirname(__file__), 
                '../../pipeline/config.yaml'
            )
            if os.path.exists(pipeline_config_path):
                self._youtube_agent = YouTubeAgent(pipeline_config_path)
        except Exception as e:
            print(f"Warning: Could not initialize YouTube agent: {e}")
            self._youtube_agent = None
    
    def _search_with_agent(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using the existing YouTube agent.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of video information dictionaries
        """
        if self._youtube_agent is None:
            raise Exception("YouTube agent not available")
        
        try:
            # Use the existing agent's search functionality
            results = self._youtube_agent.search_videos(query, max_results)
            return results
        except Exception as e:
            raise Exception(f"Search failed with agent: {e}")
    
    def _search_fallback(self, query: str, max_results: int, **kwargs) -> List[Dict[str, Any]]:
        """Fallback search implementation using youtubesearchpython.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            **kwargs: Additional search parameters
            
        Returns:
            List of video information dictionaries
        """
        if VideosSearch is None:
            raise Exception("youtubesearchpython not available for fallback search")
        
        try:
            # Create search object and get results
            videos_search = VideosSearch(query, limit=max_results)
            results = videos_search.result()['result']
            
            # Format results
            formatted_results = []
            for i, video in enumerate(results):
                formatted_video = {
                    'title': video.get('title', ''),
                    'url': video.get('link', ''),
                    'video_id': video.get('id', ''),
                    'duration': video.get('duration', ''),
                    'channel': video.get('channel', {}).get('name', ''),
                    'channel_url': video.get('channel', {}).get('link', ''),
                    'views': video.get('viewCount', {}).get('text', ''),
                    'published': video.get('publishedTime', ''),
                    'description': video.get('descriptionSnippet', [{}])[0].get('text', '') if video.get('descriptionSnippet') else '',
                    'thumbnails': video.get('thumbnails', []),
                    'metadata': {
                        'search_query': query,
                        'search_rank': i + 1
                    }
                }
                formatted_results.append(formatted_video)
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Fallback search failed: {e}")
    
    def _filter_islamic_content(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and prioritize Islamic content.
        
        Args:
            results: List of video results
            
        Returns:
            Filtered and sorted results
        """
        islamic_config = self.config.get('islamic_content', {})
        preferred_scholars = islamic_config.get('preferred_scholars', [])
        islamic_topics = islamic_config.get('topics', [])
        
        # Score videos based on Islamic content relevance
        scored_results = []
        for video in results:
            score = 0
            title = (video.get('title') or '').lower()
            channel = (video.get('channel') or '').lower()
            description = (video.get('description') or '').lower()
            
            # Check for preferred scholars
            for scholar in preferred_scholars:
                if scholar.lower() in channel or scholar.lower() in title:
                    score += 10
            
            # Check for Islamic topics
            for topic in islamic_topics:
                if topic.lower() in title or topic.lower() in description:
                    score += 5
            
            # Check for Islamic keywords
            islamic_keywords = ['islam', 'muslim', 'quran', 'allah', 'prophet', 'hadith']
            for keyword in islamic_keywords:
                if keyword in title or keyword in description:
                    score += 2
            
            video['islamic_relevance_score'] = score
            scored_results.append(video)
        
        # Sort by Islamic relevance score (descending)
        scored_results.sort(key=lambda x: x['islamic_relevance_score'], reverse=True)
        
        return scored_results
    
    async def search_videos(self, query: str, max_results: int = 10, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search for videos and return structured results.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            filters: Additional search filters
            
        Returns:
            Dictionary with success status and video results
        """
        try:
            # Use the _run method to get formatted results
            formatted_output = self._run(query, max_results)
            
            # Parse the formatted output to extract video URLs and metadata
            videos = []
            if "Found" in formatted_output and "videos" in formatted_output:
                lines = formatted_output.split('\n')
                current_video = {}
                
                for line in lines:
                    line = line.strip()
                    if line.startswith(tuple('123456789')):
                        # New video entry
                        if current_video:
                            videos.append(current_video)
                        title = line.split('**')[1] if '**' in line else line.split('. ')[1]
                        current_video = {'title': title}
                    elif line.startswith('URL:'):
                        current_video['url'] = line.replace('URL: ', '')
                    elif line.startswith('Channel:'):
                        current_video['channel'] = line.replace('Channel: ', '')
                    elif line.startswith('Duration:'):
                        current_video['duration'] = line.replace('Duration: ', '')
                
                # Add the last video
                if current_video:
                    videos.append(current_video)
            
            return {
                'success': True,
                'videos': videos,
                'total_found': len(videos)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'videos': []
            }
    
    def _run(self, query: str, max_results: int = 10, duration: str = "medium", 
             upload_date: str = "any", sort_by: str = "relevance", 
             language: str = "en", region: str = "US") -> str:
        """Execute the YouTube search.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            duration: Duration filter
            upload_date: Upload date filter
            sort_by: Sort criteria
            language: Language filter
            region: Region filter
            
        Returns:
            Formatted search results as string
        """
        try:
            # Enhance query for Islamic content if not already specific
            enhanced_query = self._enhance_query_for_islamic_content(query)
            
            # Try to search with existing agent first
            results = []
            if self._youtube_agent:
                try:
                    results = self._search_with_agent(enhanced_query, max_results)
                except Exception as e:
                    print(f"Agent search failed, using fallback: {e}")
            
            # Use fallback if agent search failed or agent not available
            if not results:
                results = self._search_fallback(
                    enhanced_query, max_results,
                    duration=duration, upload_date=upload_date,
                    sort_by=sort_by, language=language, region=region
                )
            
            # Filter and prioritize Islamic content
            filtered_results = self._filter_islamic_content(results)
            
            # Limit to requested number of results
            final_results = filtered_results[:max_results]
            
            # Format results for output
            return self._format_results(final_results, query)
            
        except Exception as e:
            return f"Error searching YouTube: {str(e)}"
    
    def _enhance_query_for_islamic_content(self, query: str) -> str:
        """Enhance search query to better find Islamic content.
        
        Args:
            query: Original search query
            
        Returns:
            Enhanced search query
        """
        # Check if query already contains Islamic terms
        islamic_terms = ['islam', 'muslim', 'quran', 'allah', 'prophet', 'hadith', 'islamic']
        query_lower = query.lower()
        
        has_islamic_term = any(term in query_lower for term in islamic_terms)
        
        if not has_islamic_term:
            # Add Islamic context to the query
            enhanced_query = f"{query} islamic"
        else:
            enhanced_query = query
        
        return enhanced_query
    
    def _format_results(self, results: List[Dict[str, Any]], original_query: str) -> str:
        """Format search results for output.
        
        Args:
            results: List of video results
            original_query: Original search query
            
        Returns:
            Formatted results string
        """
        if not results:
            return f"No videos found for query: '{original_query}'"
        
        output = f"Found {len(results)} videos for query: '{original_query}'\n\n"
        
        for i, video in enumerate(results, 1):
            output += f"{i}. **{video.get('title', 'Unknown Title')}**\n"
            output += f"   Channel: {video.get('channel', 'Unknown Channel')}\n"
            output += f"   Duration: {video.get('duration', 'Unknown')}\n"
            output += f"   URL: {video.get('url', 'No URL')}\n"
            output += f"   Views: {video.get('views', 'Unknown')}\n"
            output += f"   Published: {video.get('published', 'Unknown')}\n"
            
            if 'islamic_relevance_score' in video:
                output += f"   Islamic Relevance Score: {video['islamic_relevance_score']}\n"
            
            description = video.get('description', '')
            if description:
                # Truncate description to first 100 characters
                desc_preview = description[:100] + "..." if len(description) > 100 else description
                output += f"   Description: {desc_preview}\n"
            
            output += "\n"
        
        return output


# Example usage and testing
if __name__ == "__main__":
    # Test the YouTube Search Tool
    tool = YouTubeSearchTool()
    
    # Test search
    test_query = "Nouman Ali Khan Quran tafseer"
    result = tool._run(test_query, max_results=5)
    print(result)