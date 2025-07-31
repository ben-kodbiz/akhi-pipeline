"""YouTube Downloader Tool for CrewAI

This tool provides YouTube video downloading capabilities for the CrewAI agentic system.
It downloads audio from YouTube videos using yt-dlp and converts to MP3 format.
"""

import os
import sys
import yaml
import subprocess
import tempfile
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from urllib.parse import urlparse, parse_qs

# Add the pipeline directory to the path to import existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../pipeline'))

try:
    from config_manager import ConfigManager
except ImportError:
    ConfigManager = None


class YouTubeDownloadInput(BaseModel):
    """Input schema for YouTube download tool."""
    url: str = Field(..., description="YouTube video URL to download")
    output_dir: str = Field(default="./downloads", description="Directory to save downloaded audio files")
    audio_format: str = Field(default="mp3", description="Audio format: mp3, wav, m4a, etc.")
    audio_quality: str = Field(default="0", description="Audio quality: 0 (best), 1-9 (worst), or bitrate like 128K")
    filename_template: str = Field(default="%(title)s.%(ext)s", description="Output filename template")
    extract_info: bool = Field(default=True, description="Whether to extract video metadata")


class YouTubeDownloaderTool(BaseTool):
    """YouTube Downloader Tool for downloading Islamic educational content."""
    
    name: str = "YouTube Downloader Tool"
    description: str = (
        "Download audio from YouTube videos using yt-dlp. "
        "Extracts high-quality audio in MP3 format from YouTube URLs. "
        "Supports custom output directories, filename templates, and audio quality settings. "
        "Returns download status and file information."
    )
    args_schema: type[BaseModel] = YouTubeDownloadInput
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the YouTube Downloader Tool.
        
        Args:
            config_path: Path to the configuration file
        """
        super().__init__()
        self._config = self._load_config(config_path)
        self._validate_dependencies()
    
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
                    'download': {
                        'audio_format': 'mp3',
                        'audio_quality': '0',
                        'timeout': 300,
                        'retry_attempts': 3,
                        'output_template': '%(title)s.%(ext)s',
                        'max_filesize': '500M'
                    }
                }
            }
    
    def _validate_dependencies(self):
        """Validate that required dependencies are available."""
        try:
            # Check if yt-dlp is available
            result = subprocess.run(
                ['yt-dlp', '--version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("yt-dlp is not working properly")
        except FileNotFoundError:
            raise RuntimeError(
                "yt-dlp not found. Please install it: pip install yt-dlp"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("yt-dlp check timed out")
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID or None if not found
        """
        try:
            parsed_url = urlparse(url)
            
            if 'youtube.com' in parsed_url.netloc:
                if 'watch' in parsed_url.path:
                    query_params = parse_qs(parsed_url.query)
                    return query_params.get('v', [None])[0]
                elif '/embed/' in parsed_url.path:
                    return parsed_url.path.split('/embed/')[-1].split('?')[0]
            elif 'youtu.be' in parsed_url.netloc:
                return parsed_url.path.lstrip('/')
            
            return None
        except Exception:
            return None
    
    def _validate_youtube_url(self, url: str) -> bool:
        """Validate if the URL is a valid YouTube URL.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid YouTube URL, False otherwise
        """
        video_id = self._extract_video_id(url)
        return video_id is not None and len(video_id) == 11
    
    def _get_video_info(self, url: str) -> Dict[str, Any]:
        """Extract video information without downloading.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Video information dictionary
        """
        try:
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-warnings',
                url
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                import json
                return json.loads(result.stdout.strip())
            else:
                return {}
                
        except Exception as e:
            print(f"Warning: Could not extract video info: {e}")
            return {}
    
    def _download_audio(self, url: str, output_dir: str, audio_format: str = "mp3", 
                       audio_quality: str = "0", filename_template: str = "%(title)s.%(ext)s") -> Dict[str, Any]:
        """Download audio from YouTube video.
        
        Args:
            url: YouTube video URL
            output_dir: Output directory
            audio_format: Audio format (mp3, wav, etc.)
            audio_quality: Audio quality (0-9 or bitrate)
            filename_template: Output filename template
            
        Returns:
            Download result dictionary
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get download configuration
        download_config = self.config.get('youtube', {}).get('download', {})
        timeout = download_config.get('timeout', 300)
        max_filesize = download_config.get('max_filesize', '500M')
        
        # Construct yt-dlp command
        output_path = os.path.join(output_dir, filename_template)
        cmd = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', audio_format,
            '--audio-quality', audio_quality,
            '--output', output_path,
            '--no-warnings',
            '--max-filesize', max_filesize,
            url
        ]
        
        try:
            print(f"Downloading audio from: {url}")
            print(f"Output directory: {output_dir}")
            print(f"Format: {audio_format}, Quality: {audio_quality}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                # Find the downloaded file
                downloaded_files = []
                for file in os.listdir(output_dir):
                    if file.endswith(f'.{audio_format}'):
                        file_path = os.path.join(output_dir, file)
                        if os.path.getmtime(file_path) > (os.path.getmtime(output_dir) - 60):  # Modified in last minute
                            downloaded_files.append(file_path)
                
                if downloaded_files:
                    # Get the most recently created file
                    latest_file = max(downloaded_files, key=os.path.getmtime)
                    file_size = os.path.getsize(latest_file)
                    
                    return {
                        'success': True,
                        'file_path': latest_file,
                        'file_size': file_size,
                        'format': audio_format,
                        'message': f"Successfully downloaded audio to {latest_file}"
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Download completed but file not found',
                        'stderr': result.stderr
                    }
            else:
                return {
                    'success': False,
                    'error': f"yt-dlp failed with return code {result.returncode}",
                    'stderr': result.stderr,
                    'stdout': result.stdout
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Download timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error during download: {str(e)}"
            }
    
    async def download_video(self, url: str, output_dir: str = "./downloads", 
                           extract_audio: bool = True, audio_dir: str = None, 
                           quality: str = "best[height<=720]", audio_format: str = "mp3",
                           audio_quality: str = "0") -> Dict[str, Any]:
        """Download video and optionally extract audio in MP3 format.
        
        Args:
            url: YouTube video URL
            output_dir: Directory for video files
            extract_audio: Whether to extract audio
            audio_dir: Directory for audio files (defaults to output_dir/audio)
            quality: Video quality selector
            audio_format: Audio format (always MP3 for consistency)
            audio_quality: Audio quality setting
            
        Returns:
            Download result dictionary
        """
        try:
            # Validate URL
            if not self._validate_youtube_url(url):
                return {'success': False, 'error': f'Invalid YouTube URL: {url}'}
            
            # Get video info
            video_info = self._get_video_info(url)
            title = video_info.get('title', 'Unknown')
            
            # Set audio directory
            if audio_dir is None:
                audio_dir = os.path.join(output_dir, 'audio')
            
            # Ensure MP3 format for consistency
            audio_format = "mp3"
            
            # Download audio only (more efficient for our use case)
            if extract_audio:
                download_result = self._download_audio(
                    url, audio_dir, audio_format, audio_quality, "%(title)s.%(ext)s"
                )
                
                if download_result['success']:
                    return {
                        'success': True,
                        'title': title,
                        'audio_path': download_result['file_path'],
                        'audio_file': download_result['file_path'],  # Keep for backward compatibility
                        'file_size': download_result['file_size'],
                        'format': audio_format,
                        'video_info': video_info
                    }
                else:
                    return {
                        'success': False,
                        'error': download_result['error'],
                        'title': title
                    }
            else:
                return {'success': False, 'error': 'Audio extraction is required'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run(self, url: str, output_dir: str = "./downloads", audio_format: str = "mp3", 
             audio_quality: str = "0", filename_template: str = "%(title)s.%(ext)s", 
             extract_info: bool = True) -> str:
        """Execute the YouTube download.
        
        Args:
            url: YouTube video URL
            output_dir: Output directory
            audio_format: Audio format
            audio_quality: Audio quality
            filename_template: Filename template
            extract_info: Whether to extract video info
            
        Returns:
            Download result as formatted string
        """
        try:
            # Validate URL
            if not self._validate_youtube_url(url):
                return f"Error: Invalid YouTube URL: {url}"
            
            # Extract video information if requested
            video_info = {}
            if extract_info:
                video_info = self._get_video_info(url)
            
            # Download audio
            download_result = self._download_audio(
                url, output_dir, audio_format, audio_quality, filename_template
            )
            
            # Format response
            if download_result['success']:
                output = f"✅ **Download Successful**\n\n"
                output += f"**File:** {download_result['file_path']}\n"
                output += f"**Size:** {download_result['file_size']:,} bytes\n"
                output += f"**Format:** {download_result['format']}\n"
                
                if video_info:
                    output += f"\n**Video Information:**\n"
                    output += f"Title: {video_info.get('title', 'Unknown')}\n"
                    output += f"Channel: {video_info.get('uploader', 'Unknown')}\n"
                    output += f"Duration: {video_info.get('duration_string', 'Unknown')}\n"
                    output += f"Views: {video_info.get('view_count', 'Unknown'):,}\n" if video_info.get('view_count') else "Views: Unknown\n"
                    output += f"Upload Date: {video_info.get('upload_date', 'Unknown')}\n"
                
                output += f"\n{download_result['message']}"
                return output
            else:
                output = f"❌ **Download Failed**\n\n"
                output += f"**URL:** {url}\n"
                output += f"**Error:** {download_result['error']}\n"
                
                if 'stderr' in download_result and download_result['stderr']:
                    output += f"**Details:** {download_result['stderr']}\n"
                
                return output
                
        except Exception as e:
            return f"❌ **Unexpected Error:** {str(e)}"


# Example usage and testing
if __name__ == "__main__":
    # Test the YouTube Downloader Tool
    tool = YouTubeDownloaderTool()
    
    # Test download (replace with actual YouTube URL)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Example URL
    result = tool._run(
        url=test_url,
        output_dir="./test_downloads",
        audio_format="mp3",
        audio_quality="0"
    )
    print(result)