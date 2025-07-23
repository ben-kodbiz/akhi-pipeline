#!/usr/bin/env python3

import os
import json
import subprocess
import re
import shlex
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..state_manager import StateManager, VideoStatus, VideoRecord
from ..config_manager import ConfigManager

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIPELINE_DIR = os.path.join(BASE_DIR, "pipeline")
OUTPUT_DIR = os.path.join(PIPELINE_DIR, "output")
DB_DIR = os.path.join(PIPELINE_DIR, "db")

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Status tracker file
STATUS_FILE = os.path.join(DB_DIR, "status_tracker.json")

# Default search terms for Islamic scholars
DEFAULT_SEARCH_TERMS = [
    "Nouman Ali Khan tafsir",
    "Mufti Menk lecture",
    "Yasir Qadhi hadith",
    "Omar Suleiman Islamic lecture",
    "Islamic scholar Quran explanation"
]

class YouTubeAgent:
    def __init__(self, config_manager: Optional[ConfigManager] = None, 
                 state_manager: Optional[StateManager] = None):
        """
        Initialize the YouTube Agent.
        
        Args:
            config_manager: Configuration manager instance
            state_manager: State manager instance
        """
        self.config = config_manager or ConfigManager()
        self.state_manager = state_manager or StateManager()
        self.search_config = self.config.get_search_config()
    
    def _validate_search_query(self, yt_search_query: str) -> str:
        """
        Validate and sanitize the YouTube search query.
        
        Args:
            yt_search_query: Raw search query
        
        Returns:
            Sanitized search query
        
        Raises:
            ValueError: If query is invalid
        """
        if not yt_search_query or not yt_search_query.strip():
            raise ValueError("Search query cannot be empty")
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[;&|`$(){}\[\]<>]', '', yt_search_query.strip())
        
        if not sanitized:
            raise ValueError("Search query contains only invalid characters")
        
        return sanitized
    
    def search_youtube_videos(self, yt_search_query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for YouTube videos using yt-dlp with robust error handling.
        
        Args:
            yt_search_query: Search query (e.g., "trauma nouman ali khan")
            max_results: Maximum number of results to return (uses config default if None)
        
        Returns:
            List of video dictionaries with metadata
        
        Raises:
            ValueError: If search query is invalid
            RuntimeError: If yt-dlp is not available or fails
        """
        try:
            # Validate and sanitize query
            sanitized_query = self._validate_search_query(yt_search_query)
            
            # Use config default if max_results not specified
            if max_results is None:
                max_results = self.search_config.max_results
            
            # Construct the yt-dlp search query
            search_query = f"ytsearch{max_results}:{sanitized_query}"
            
            # Prepare command with proper escaping
            cmd = ["yt-dlp", "--dump-json", "--no-warnings", search_query]
            
            print(f"Searching YouTube for: {sanitized_query} (max {max_results} results)")
            
            # Run yt-dlp to get video metadata
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.search_config.timeout,
                check=False  # Don't raise on non-zero exit
            )
            
            if result.returncode != 0:
                error_msg = f"yt-dlp failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr.strip()}"
                print(f"❌ {error_msg}")
                
                # Check if it's a network issue
                if "network" in result.stderr.lower() or "connection" in result.stderr.lower():
                    raise RuntimeError(f"Network error during search: {result.stderr.strip()}")
                
                return []
            
            # Parse JSON output
            raw_video_list = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        video_metadata = json.loads(line)
                        raw_video_list.append(video_metadata)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Error parsing JSON line: {e}")
                        continue
            
            print(f"✅ Found {len(raw_video_list)} videos")
            return raw_video_list
        
        except subprocess.TimeoutExpired:
            error_msg = f"yt-dlp search timed out after {self.search_config.timeout} seconds"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        except FileNotFoundError:
            error_msg = "yt-dlp not found. Please install yt-dlp: pip install yt-dlp"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during YouTube search: {e}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def _parse_duration(self, duration_str: str) -> int:
        """
        Parse ISO 8601 duration format to seconds.
        
        Args:
            duration_str: Duration string in ISO 8601 format (e.g., 'PT1H30M15S')
            
        Returns:
            Duration in seconds
        """
        import re
        import datetime
        
        # Remove 'PT' prefix
        duration_str = duration_str[2:]
        
        # Initialize duration components
        hours = 0
        minutes = 0
        seconds = 0
        
        # Extract hours, minutes, seconds
        hours_match = re.search(r'(\d+)H', duration_str)
        if hours_match:
            hours = int(hours_match.group(1))
        
        minutes_match = re.search(r'(\d+)M', duration_str)
        if minutes_match:
            minutes = int(minutes_match.group(1))
        
        seconds_match = re.search(r'(\d+)S', duration_str)
        if seconds_match:
            seconds = int(seconds_match.group(1))
        
        return hours * 3600 + minutes * 60 + seconds
    
    def filter_videos_by_duration(self, raw_video_list: List[Dict[str, Any]], 
                                  min_duration: Optional[int] = None, 
                                  max_duration: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Filter videos by duration using configuration defaults.
        
        Args:
            raw_video_list: List of raw video dictionaries from yt-dlp
            min_duration: Minimum duration in seconds (uses config default if None)
            max_duration: Maximum duration in seconds (uses config default if None)
        
        Returns:
            Filtered list of videos
        """
        if min_duration is None:
            min_duration = self.search_config.min_duration
        if max_duration is None:
            max_duration = self.search_config.max_duration
        
        filtered_video_list = []
        
        for video_metadata in raw_video_list:
            duration_seconds = video_metadata.get('duration', 0)
            if duration_seconds and min_duration <= duration_seconds <= max_duration:
                filtered_video_list.append(video_metadata)
        
        print(f"✅ Filtered to {len(filtered_video_list)} videos (duration: {min_duration}-{max_duration}s)")
        return filtered_video_list
    
    def process_and_store_videos(self, raw_video_list: List[Dict[str, Any]], 
                                yt_search_query: str) -> List[VideoRecord]:
        """
        Process raw video data and store in state manager.
        
        Args:
            raw_video_list: List of raw video dictionaries from yt-dlp
            yt_search_query: Original search query used
        
        Returns:
            List of processed VideoRecord objects
        """
        processed_videos = []
        new_videos_count = 0
        
        for video_metadata in raw_video_list:
            try:
                video_id = video_metadata.get('id', '')
                if not video_id:
                    continue
                
                # Check if video already exists
                existing_video = self.state_manager.get_video(video_id)
                if existing_video:
                    processed_videos.append(existing_video)
                    continue
                
                # Process new video
                duration_seconds = video_metadata.get('duration', 0)
                
                structured_video_data = {
                    'video_id': video_id,
                    'title': video_metadata.get('title', ''),
                    'channel': video_metadata.get('uploader', '') or video_metadata.get('channel', ''),
                    'description': video_metadata.get('description', ''),
                    'published_at': video_metadata.get('upload_date', ''),
                    'duration': duration_seconds,
                    'duration_str': self._format_duration(duration_seconds),
                    'view_count': video_metadata.get('view_count', 0),
                    'url': video_metadata.get('webpage_url', ''),
                    'search_term': yt_search_query
                }
                
                # Add to state manager
                if self.state_manager.add_video(structured_video_data):
                    new_videos_count += 1
                    video_record = self.state_manager.get_video(video_id)
                    processed_videos.append(video_record)
                
            except Exception as e:
                print(f"⚠️ Error processing video {video_metadata.get('id', 'unknown')}: {e}")
                continue
        
        print(f"✅ Processed {len(processed_videos)} videos ({new_videos_count} new)")
        return processed_videos
    
    def _format_duration(self, seconds: int) -> str:
        """
        Format duration from seconds to HH:MM:SS or MM:SS format.
        
        Args:
            seconds: Duration in seconds
        
        Returns:
            Formatted duration string
        """
        if seconds <= 0:
            return "0:00"
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"
    
    def search_and_filter_videos(self, yt_search_query: str, 
                                max_results: Optional[int] = None,
                                min_duration: Optional[int] = None,
                                max_duration: Optional[int] = None) -> List[VideoRecord]:
        """
        Complete search and filter workflow for YouTube videos.
        
        Args:
            yt_search_query: Search query
            max_results: Maximum number of results (uses config default if None)
            min_duration: Minimum duration in seconds (uses config default if None)
            max_duration: Maximum duration in seconds (uses config default if None)
        
        Returns:
            List of processed and filtered VideoRecord objects
        """
        try:
            # Step 1: Search YouTube
            raw_video_list = self.search_youtube_videos(yt_search_query, max_results)
            if not raw_video_list:
                print("No videos found")
                return []
            
            # Step 2: Filter by duration
            filtered_video_list = self.filter_videos_by_duration(
                raw_video_list, min_duration, max_duration
            )
            if not filtered_video_list:
                print("No videos passed duration filter")
                return []
            
            # Step 3: Process and store
            processed_videos = self.process_and_store_videos(filtered_video_list, yt_search_query)
            
            return processed_videos
            
        except Exception as e:
            print(f"❌ Error in search and filter workflow: {e}")
            return []
    
    def get_videos_to_download(self, limit: Optional[int] = None) -> List[VideoRecord]:
        """
        Get videos that are ready to be downloaded.
        
        Args:
            limit: Maximum number of videos to return (None for all)
        
        Returns:
            List of VideoRecord objects ready for download
        """
        videos_to_download = self.state_manager.get_videos_by_status(VideoStatus.DISCOVERED)
        
        if limit:
            videos_to_download = videos_to_download[:limit]
        
        return videos_to_download
    
    def get_download_list(self, max_videos: int = 10) -> List[Dict[str, Any]]:
        """
        Generate a list of videos to download with essential information.
        
        Args:
            max_videos: Maximum number of videos to include
        
        Returns:
            List of video dictionaries ready for download
        """
        videos_to_download = self.get_videos_to_download(limit=max_videos)
        
        download_list = []
        for video_record in videos_to_download:
            download_list.append({
                "video_id": video_record.video_id,
                "title": video_record.title,
                "url": video_record.url,
                "duration_str": video_record.duration_str,
                "channel": video_record.channel
            })
        
        return download_list
    
    def generate_download_links_file(self, output_file: str = "video_links.txt", max_videos: int = 10) -> str:
        """
        Generate a file with video URLs for downloading (legacy method).
        
        Args:
            output_file: Name of the output file
            max_videos: Maximum number of videos to include
        
        Returns:
            Path to the generated file
        """
        videos_to_download = self.get_videos_to_download(limit=max_videos)
        return self.create_download_file(videos_to_download, output_file)
    
    def update_video_status(self, video_id: str, new_status: VideoStatus, 
                           error_message: Optional[str] = None,
                           file_path: Optional[str] = None,
                           file_type: Optional[str] = None) -> bool:
        """
        Update the status of a specific video.
        
        Args:
            video_id: Video ID to update
            new_status: New status for the video
            error_message: Error message if status indicates failure
            file_path: Path to generated file (audio, transcript, etc.)
            file_type: Type of file (audio, transcript, json)
        
        Returns:
            True if update was successful, False if video not found
        """
        return self.state_manager.update_video_status(
            video_id, new_status, error_message, file_path, file_type
        )
    
    def mark_videos_as_downloaded(self, video_ids: List[str]) -> None:
        """
        Mark multiple videos as downloaded.
        
        Args:
            video_ids: List of video IDs to mark as downloaded
        """
        for video_id in video_ids:
            self.update_video_status(video_id, VideoStatus.DOWNLOADED)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics from state manager.
        
        Returns:
            Dictionary containing current stats
        """
        return self.state_manager.get_stats()
    
    def get_videos_by_status(self, status: VideoStatus) -> List[VideoRecord]:
        """
        Get all videos with a specific status.
        
        Args:
            status: Status to filter by
        
        Returns:
            List of VideoRecord objects
        """
        return self.state_manager.get_videos_by_status(status)
    
    def get_failed_videos(self, max_attempts: int = 3) -> List[VideoRecord]:
        """
        Get videos that have failed but can be retried.
        
        Args:
            max_attempts: Maximum number of attempts before giving up
        
        Returns:
            List of VideoRecord objects that can be retried
        """
        return self.state_manager.get_failed_videos(max_attempts)
    
    def export_summary(self) -> Dict[str, Any]:
        """
        Export a summary of the current state.
        
        Returns:
            Dictionary with summary information
        """
        return self.state_manager.export_summary()


if __name__ == "__main__":
    import argparse
    
    def main():
        parser = argparse.ArgumentParser(
            description="YouTube Agent for Islamic Scholar Videos",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python youtube_agent.py --search --query "nouman ali khan trauma"
  python youtube_agent.py --search --query "omar suleiman" --max-results 15
  python youtube_agent.py --download-list --max-downloads 5
  python youtube_agent.py --stats
  python youtube_agent.py --summary
            """
        )
        
        parser.add_argument("--search", action="store_true", 
                          help="Search for new videos")
        parser.add_argument("--query", type=str, 
                          help="Custom search query (required with --search)")
        parser.add_argument("--max-results", type=int, 
                          help="Maximum results per search (uses config default if not specified)")
        parser.add_argument("--min-duration", type=int, 
                          help="Minimum video duration in seconds (uses config default if not specified)")
        parser.add_argument("--max-duration", type=int, 
                          help="Maximum video duration in seconds (uses config default if not specified)")
        parser.add_argument("--download-list", action="store_true", 
                          help="Generate download list")
        parser.add_argument("--max-downloads", type=int, default=10, 
                          help="Maximum videos in download list")
        parser.add_argument("--stats", action="store_true", 
                          help="Show current statistics")
        parser.add_argument("--summary", action="store_true", 
                          help="Show detailed summary")
        parser.add_argument("--failed", action="store_true", 
                          help="Show failed videos that can be retried")
        parser.add_argument("--config", type=str, 
                          help="Path to config file")
        
        args = parser.parse_args()
        
        try:
            # Initialize components
            config_manager = ConfigManager(args.config) if args.config else ConfigManager()
            agent = YouTubeAgent(config_manager=config_manager)
            
            if args.search:
                if not args.query:
                    print("❌ Error: --query is required when using --search")
                    return 1
                
                print(f"🔍 Searching for videos: {args.query}")
                videos = agent.search_and_filter_videos(
                    yt_search_query=args.query,
                    max_results=args.max_results,
                    min_duration=args.min_duration,
                    max_duration=args.max_duration
                )
                print(f"✅ Found and processed {len(videos)} videos")
                
                # Show some details about found videos
                if videos:
                    print("\n📋 Recent videos:")
                    for i, video in enumerate(videos[:5], 1):
                        print(f"  {i}. {video.title[:60]}... ({video.duration_str})")
                    if len(videos) > 5:
                        print(f"  ... and {len(videos) - 5} more")
            
            if args.download_list:
                print(f"📝 Generating download list...")
                videos_to_download = agent.get_videos_to_download(limit=args.max_downloads)
                if videos_to_download:
                    download_file = agent.create_download_file(videos_to_download)
                    print(f"✅ Download file created: {download_file}")
                else:
                    print("ℹ️ No videos available for download")
            
            if args.stats:
                stats = agent.get_stats()
                print("\n📊 Current Statistics:")
                for key, value in stats.items():
                    formatted_key = key.replace('_', ' ').title()
                    print(f"  {formatted_key}: {value}")
            
            if args.summary:
                summary = agent.export_summary()
                print("\n📋 Detailed Summary:")
                print(f"  Total Videos: {summary['total_videos']}")
                print(f"  Created: {summary['metadata'].get('created_at', 'Unknown')}")
                print(f"  Last Updated: {summary['metadata'].get('last_updated', 'Unknown')}")
                
                print("\n📊 Statistics:")
                for key, value in summary['stats'].items():
                    formatted_key = key.replace('_', ' ').title()
                    print(f"    {formatted_key}: {value}")
                
                if summary['recent_videos']:
                    print("\n🕒 Recent Videos:")
                    for video in summary['recent_videos']:
                        print(f"    {video['title']} ({video['status']})")
            
            if args.failed:
                failed_videos = agent.get_failed_videos()
                if failed_videos:
                    print(f"\n⚠️ Failed Videos ({len(failed_videos)} can be retried):")
                    for video in failed_videos:
                        print(f"  {video.title[:60]}... ({video.status.value})")
                        if video.error_message:
                            print(f"    Error: {video.error_message}")
                else:
                    print("\n✅ No failed videos to retry")
            
            # If no action specified, show help
            if not any([args.search, args.download_list, args.stats, args.summary, args.failed]):
                parser.print_help()
                return 1
            
            return 0
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    exit(main())