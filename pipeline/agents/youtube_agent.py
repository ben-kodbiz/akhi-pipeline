import os
import json
import subprocess
import re
from typing import List, Dict, Any, Optional

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
    def __init__(self):
        """
        Initialize the YouTube Agent using yt-dlp for video discovery and downloading.
        """
        # Load existing status tracker if it exists
        self.status_tracker = self._load_status_tracker()
    
    def _load_status_tracker(self) -> Dict[str, Any]:
        """
        Load the status tracker from file or create a new one.
        
        Returns:
            Dict containing the status tracker data
        """
        if os.path.exists(STATUS_FILE):
            try:
                with open(STATUS_FILE, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading status tracker file. Creating new one.")
        
        # Create new status tracker
        return {
            "videos": {},
            "last_search": None,
            "stats": {
                "total_discovered": 0,
                "downloaded": 0,
                "transcribed": 0,
                "json_ready": 0
            }
        }
    
    def _save_status_tracker(self) -> None:
        """
        Save the status tracker to file.
        """
        with open(STATUS_FILE, "w") as f:
            json.dump(self.status_tracker, f, indent=2)
    
    def search_videos(self, search_terms: Optional[List[str]] = None, 
                      max_results_per_term: int = 20,
                      min_duration: int = 300,  # 5 minutes
                      max_duration: int = 3600) -> List[Dict[str, Any]]:
        """
        Search for videos on YouTube using yt-dlp and filter by duration.
        
        Args:
            search_terms: List of search terms to use. If None, uses DEFAULT_SEARCH_TERMS.
            max_results_per_term: Maximum number of results to search per term.
            min_duration: Minimum video duration in seconds.
            max_duration: Maximum video duration in seconds.
            
        Returns:
            List of video metadata dictionaries.
        """
        search_terms = search_terms or DEFAULT_SEARCH_TERMS
        all_videos = []
        
        for term in search_terms:
            try:
                print(f"Searching for: {term}")
                
                # Use yt-dlp to search and get video metadata
                cmd = [
                    "yt-dlp",
                    "--dump-json",
                    "--no-download",
                    f"ytsearch{max_results_per_term}:{term}"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Parse each line as JSON (yt-dlp outputs one JSON per line)
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            video_data = json.loads(line)
                            
                            # Extract duration
                            duration_seconds = video_data.get('duration', 0)
                            if not duration_seconds:
                                continue
                            
                            # Filter by duration
                            if min_duration <= duration_seconds <= max_duration:
                                video_id = video_data.get('id', '')
                                
                                # Skip if already in status tracker
                                if video_id in self.status_tracker["videos"]:
                                    continue
                                
                                processed_video = {
                                    "video_id": video_id,
                                    "title": video_data.get('title', ''),
                                    "channel": video_data.get('uploader', ''),
                                    "description": video_data.get('description', ''),
                                    "published_at": video_data.get('upload_date', ''),
                                    "duration": duration_seconds,
                                    "duration_str": self._format_duration(duration_seconds),
                                    "view_count": video_data.get('view_count', 0),
                                    "url": video_data.get('webpage_url', ''),
                                    "search_term": term,
                                    "status": {
                                        "discovered": True,
                                        "downloaded": False,
                                        "transcribed": False,
                                        "json_ready": False
                                    }
                                }
                                
                                all_videos.append(processed_video)
                                self.status_tracker["videos"][video_id] = processed_video
                                self.status_tracker["stats"]["total_discovered"] += 1
                                
                        except json.JSONDecodeError as e:
                            print(f"Error parsing video metadata: {e}")
                            continue
            
            except subprocess.CalledProcessError as e:
                print(f"Error searching for '{term}' with yt-dlp: {e}")
                print(f"stderr: {e.stderr}")
            except Exception as e:
                print(f"Unexpected error while searching for '{term}': {e}")
        
        # Save updated status tracker
        self._save_status_tracker()
        
        return all_videos
    
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
    
    def _format_duration(self, seconds: int) -> str:
        """
        Format seconds to HH:MM:SS string.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"
    
    def get_download_list(self, max_videos: int = 10) -> List[Dict[str, Any]]:
        """
        Get a list of videos to download.
        
        Args:
            max_videos: Maximum number of videos to include in the list
            
        Returns:
            List of video metadata dictionaries for videos that haven't been downloaded yet
        """
        to_download = []
        
        for video_id, video_data in self.status_tracker["videos"].items():
            if not video_data["status"]["downloaded"] and len(to_download) < max_videos:
                to_download.append(video_data)
        
        return to_download
    
    def generate_download_links_file(self, output_file: str = "video_links.txt", max_videos: int = 10) -> str:
        """
        Generate a file with YouTube links for downloading.
        
        Args:
            output_file: Path to the output file
            max_videos: Maximum number of videos to include
            
        Returns:
            Path to the generated file
        """
        download_list = self.get_download_list(max_videos)
        
        # Write links to file
        output_path = os.path.join(PIPELINE_DIR, output_file)
        with open(output_path, "w") as f:
            for video in download_list:
                f.write(f"{video['url']}\n")
        
        print(f"Generated download links file with {len(download_list)} videos: {output_path}")
        return output_path
    
    def update_video_status(self, video_id: str, status_key: str, value: bool) -> None:
        """
        Update the status of a video in the status tracker.
        
        Args:
            video_id: YouTube video ID
            status_key: Status key to update (downloaded, transcribed, json_ready)
            value: New status value
        """
        if video_id in self.status_tracker["videos"]:
            # Update video status
            self.status_tracker["videos"][video_id]["status"][status_key] = value
            
            # Update stats counter if status changed to True
            if value and status_key in self.status_tracker["stats"]:
                self.status_tracker["stats"][status_key] += 1
            
            # Save updated status tracker
            self._save_status_tracker()
    
    def mark_videos_as_downloaded(self, video_ids: List[str]) -> None:
        """
        Mark multiple videos as downloaded.
        
        Args:
            video_ids: List of YouTube video IDs
        """
        for video_id in video_ids:
            self.update_video_status(video_id, "downloaded", True)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the discovered and processed videos.
        
        Returns:
            Dictionary with statistics
        """
        return self.status_tracker["stats"]


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube Agent for Islamic Scholar Videos using yt-dlp")
    parser.add_argument("--search", action="store_true", help="Search for videos")
    parser.add_argument("--generate-links", action="store_true", help="Generate download links file")
    parser.add_argument("--max-videos", type=int, default=10, help="Maximum number of videos to process")
    parser.add_argument("--max-results-per-term", type=int, default=20, help="Maximum search results per term")
    
    args = parser.parse_args()
    
    agent = YouTubeAgent()
    
    if args.search:
        videos = agent.search_videos(max_results_per_term=args.max_results_per_term)
        print(f"Found {len(videos)} new videos")
    
    if args.generate_links:
        links_file = agent.generate_download_links_file(max_videos=args.max_videos)
        print(f"Generated links file: {links_file}")
    
    # Print stats
    stats = agent.get_stats()
    print("\nCurrent Stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")