#!/usr/bin/env python3

import os
import json
import time
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum

class VideoStatus(Enum):
    """Enum for video processing status."""
    DISCOVERED = "discovered"
    DOWNLOAD_PENDING = "download_pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    DOWNLOAD_FAILED = "download_failed"
    TRANSCRIPTION_PENDING = "transcription_pending"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    TRANSCRIPTION_FAILED = "transcription_failed"
    FORMATTING_PENDING = "formatting_pending"
    FORMATTING = "formatting"
    FORMATTED = "formatted"
    FORMATTING_FAILED = "formatting_failed"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class VideoRecord:
    """Data class for video record."""
    video_id: str
    title: str
    channel: str
    description: str
    published_at: str
    duration: int
    duration_str: str
    view_count: int
    url: str
    search_term: str
    status: VideoStatus
    discovered_at: str
    last_updated: str
    download_attempts: int = 0
    transcription_attempts: int = 0
    formatting_attempts: int = 0
    error_message: Optional[str] = None
    file_paths: Dict[str, str] = None
    
    def __post_init__(self):
        if self.file_paths is None:
            self.file_paths = {}
        if isinstance(self.status, str):
            self.status = VideoStatus(self.status)

class StateManager:
    """Manages the state of videos in the pipeline."""
    
    def __init__(self, state_file: Optional[str] = None, backup_enabled: bool = True):
        """
        Initialize the state manager.
        
        Args:
            state_file: Path to the state file. If None, uses default.
            backup_enabled: Whether to create backups of the state file.
        """
        if state_file is None:
            # Default to status_tracker.json in the db directory
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_dir = os.path.join(base_dir, "pipeline", "db")
            os.makedirs(db_dir, exist_ok=True)
            state_file = os.path.join(db_dir, "video_state.json")
        
        self.state_file = state_file
        self.backup_enabled = backup_enabled
        self.state = self._load_state()
        self._operation_count = 0
    
    def _load_state(self) -> Dict[str, Any]:
        """
        Load state from file.
        
        Returns:
            Dictionary containing the state data
        """
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert video records back to VideoRecord objects
                videos = {}
                for video_id, video_data in data.get('videos', {}).items():
                    try:
                        # Handle legacy format
                        if 'status' in video_data and isinstance(video_data['status'], dict):
                            # Convert old status format
                            old_status = video_data['status']
                            if old_status.get('json_ready', False):
                                new_status = VideoStatus.COMPLETED
                            elif old_status.get('transcribed', False):
                                new_status = VideoStatus.TRANSCRIBED
                            elif old_status.get('downloaded', False):
                                new_status = VideoStatus.DOWNLOADED
                            else:
                                new_status = VideoStatus.DISCOVERED
                            video_data['status'] = new_status.value
                        
                        # Ensure required fields exist
                        video_data.setdefault('discovered_at', datetime.now().isoformat())
                        video_data.setdefault('last_updated', datetime.now().isoformat())
                        video_data.setdefault('download_attempts', 0)
                        video_data.setdefault('transcription_attempts', 0)
                        video_data.setdefault('formatting_attempts', 0)
                        video_data.setdefault('error_message', None)
                        video_data.setdefault('file_paths', {})
                        
                        videos[video_id] = VideoRecord(**video_data)
                    except Exception as e:
                        print(f"Error loading video record {video_id}: {e}")
                        continue
                
                return {
                    'videos': videos,
                    'stats': data.get('stats', self._get_default_stats()),
                    'metadata': data.get('metadata', {
                        'created_at': datetime.now().isoformat(),
                        'version': '2.0'
                    })
                }
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error loading state file: {e}")
                print("Creating new state.")
        
        return {
            'videos': {},
            'stats': self._get_default_stats(),
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '2.0'
            }
        }
    
    def _get_default_stats(self) -> Dict[str, int]:
        """Get default statistics dictionary."""
        return {
            'total_discovered': 0,
            'download_pending': 0,
            'downloaded': 0,
            'download_failed': 0,
            'transcription_pending': 0,
            'transcribed': 0,
            'transcription_failed': 0,
            'formatting_pending': 0,
            'formatted': 0,
            'formatting_failed': 0,
            'completed': 0,
            'failed': 0
        }
    
    def _save_state(self, force: bool = False) -> None:
        """
        Save state to file.
        
        Args:
            force: Force save even if auto-save interval hasn't been reached
        """
        self._operation_count += 1
        
        # Auto-save every 10 operations unless forced
        if not force and self._operation_count % 10 != 0:
            return
        
        try:
            # Create backup if enabled
            if self.backup_enabled and os.path.exists(self.state_file):
                backup_file = f"{self.state_file}.backup"
                shutil.copy2(self.state_file, backup_file)
            
            # Convert VideoRecord objects to dictionaries
            data_to_save = {
                'videos': {},
                'stats': self.state['stats'],
                'metadata': {
                    **self.state['metadata'],
                    'last_updated': datetime.now().isoformat()
                }
            }
            
            for video_id, video_record in self.state['videos'].items():
                video_dict = asdict(video_record)
                video_dict['status'] = video_record.status.value
                data_to_save['videos'][video_id] = video_dict
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = f"{self.state_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            
            os.rename(temp_file, self.state_file)
            
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def add_video(self, video_data: Dict[str, Any]) -> bool:
        """
        Add a new video to the state.
        
        Args:
            video_data: Dictionary containing video information
        
        Returns:
            True if video was added, False if it already exists
        """
        video_id = video_data.get('video_id')
        if not video_id:
            raise ValueError("video_id is required")
        
        if video_id in self.state['videos']:
            return False
        
        # Create VideoRecord
        now = datetime.now().isoformat()
        video_record = VideoRecord(
            video_id=video_id,
            title=video_data.get('title', ''),
            channel=video_data.get('channel', ''),
            description=video_data.get('description', ''),
            published_at=video_data.get('published_at', ''),
            duration=video_data.get('duration', 0),
            duration_str=video_data.get('duration_str', ''),
            view_count=video_data.get('view_count', 0),
            url=video_data.get('url', ''),
            search_term=video_data.get('search_term', ''),
            status=VideoStatus.DISCOVERED,
            discovered_at=now,
            last_updated=now
        )
        
        self.state['videos'][video_id] = video_record
        self.state['stats']['total_discovered'] += 1
        self._save_state()
        
        return True
    
    def update_video_status(self, video_id: str, new_status: VideoStatus, 
                           error_message: Optional[str] = None,
                           file_path: Optional[str] = None,
                           file_type: Optional[str] = None) -> bool:
        """
        Update the status of a video.
        
        Args:
            video_id: Video ID
            new_status: New status
            error_message: Error message if status indicates failure
            file_path: Path to generated file (audio, transcript, etc.)
            file_type: Type of file (audio, transcript, json)
        
        Returns:
            True if update was successful, False if video not found
        """
        if video_id not in self.state['videos']:
            return False
        
        video_record = self.state['videos'][video_id]
        old_status = video_record.status
        
        # Update status
        video_record.status = new_status
        video_record.last_updated = datetime.now().isoformat()
        
        # Update error message
        if error_message:
            video_record.error_message = error_message
        elif new_status not in [VideoStatus.DOWNLOAD_FAILED, VideoStatus.TRANSCRIPTION_FAILED, VideoStatus.FORMATTING_FAILED, VideoStatus.FAILED]:
            video_record.error_message = None
        
        # Update file paths
        if file_path and file_type:
            video_record.file_paths[file_type] = file_path
        
        # Update attempt counters
        if new_status == VideoStatus.DOWNLOADING:
            video_record.download_attempts += 1
        elif new_status == VideoStatus.TRANSCRIBING:
            video_record.transcription_attempts += 1
        elif new_status == VideoStatus.FORMATTING:
            video_record.formatting_attempts += 1
        
        # Update statistics
        self._update_stats(old_status, new_status)
        
        self._save_state()
        return True
    
    def _update_stats(self, old_status: VideoStatus, new_status: VideoStatus) -> None:
        """
        Update statistics when a video status changes.
        
        Args:
            old_status: Previous status
            new_status: New status
        """
        # Decrement old status count
        old_key = self._status_to_stat_key(old_status)
        if old_key and old_key in self.state['stats']:
            self.state['stats'][old_key] = max(0, self.state['stats'][old_key] - 1)
        
        # Increment new status count
        new_key = self._status_to_stat_key(new_status)
        if new_key and new_key in self.state['stats']:
            self.state['stats'][new_key] += 1
    
    def _status_to_stat_key(self, status: VideoStatus) -> Optional[str]:
        """
        Convert video status to statistics key.
        
        Args:
            status: Video status
        
        Returns:
            Statistics key or None
        """
        status_map = {
            VideoStatus.DOWNLOAD_PENDING: 'download_pending',
            VideoStatus.DOWNLOADED: 'downloaded',
            VideoStatus.DOWNLOAD_FAILED: 'download_failed',
            VideoStatus.TRANSCRIPTION_PENDING: 'transcription_pending',
            VideoStatus.TRANSCRIBED: 'transcribed',
            VideoStatus.TRANSCRIPTION_FAILED: 'transcription_failed',
            VideoStatus.FORMATTING_PENDING: 'formatting_pending',
            VideoStatus.FORMATTED: 'formatted',
            VideoStatus.FORMATTING_FAILED: 'formatting_failed',
            VideoStatus.COMPLETED: 'completed',
            VideoStatus.FAILED: 'failed'
        }
        return status_map.get(status)
    
    def get_videos_by_status(self, status: VideoStatus) -> List[VideoRecord]:
        """
        Get all videos with a specific status.
        
        Args:
            status: Status to filter by
        
        Returns:
            List of video records
        """
        return [video for video in self.state['videos'].values() if video.status == status]
    
    def get_video(self, video_id: str) -> Optional[VideoRecord]:
        """
        Get a specific video record.
        
        Args:
            video_id: Video ID
        
        Returns:
            VideoRecord or None if not found
        """
        return self.state['videos'].get(video_id)
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get current statistics.
        
        Returns:
            Dictionary with statistics
        """
        return self.state['stats'].copy()
    
    def get_failed_videos(self, max_attempts: int = 3) -> List[VideoRecord]:
        """
        Get videos that have failed and haven't exceeded max attempts.
        
        Args:
            max_attempts: Maximum number of attempts before giving up
        
        Returns:
            List of video records that can be retried
        """
        failed_videos = []
        
        for video in self.state['videos'].values():
            if video.status == VideoStatus.DOWNLOAD_FAILED and video.download_attempts < max_attempts:
                failed_videos.append(video)
            elif video.status == VideoStatus.TRANSCRIPTION_FAILED and video.transcription_attempts < max_attempts:
                failed_videos.append(video)
            elif video.status == VideoStatus.FORMATTING_FAILED and video.formatting_attempts < max_attempts:
                failed_videos.append(video)
        
        return failed_videos
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """
        Remove old failed records.
        
        Args:
            days: Number of days after which to remove failed records
        
        Returns:
            Number of records removed
        """
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        removed_count = 0
        
        videos_to_remove = []
        for video_id, video in self.state['videos'].items():
            if video.status in [VideoStatus.FAILED, VideoStatus.DOWNLOAD_FAILED, 
                              VideoStatus.TRANSCRIPTION_FAILED, VideoStatus.FORMATTING_FAILED]:
                try:
                    last_updated = datetime.fromisoformat(video.last_updated).timestamp()
                    if last_updated < cutoff_time:
                        videos_to_remove.append(video_id)
                except ValueError:
                    # Invalid timestamp, remove it
                    videos_to_remove.append(video_id)
        
        for video_id in videos_to_remove:
            del self.state['videos'][video_id]
            removed_count += 1
        
        if removed_count > 0:
            self._save_state(force=True)
        
        return removed_count
    
    def force_save(self) -> None:
        """
        Force save the current state.
        """
        self._save_state(force=True)
    
    def export_summary(self) -> Dict[str, Any]:
        """
        Export a summary of the current state.
        
        Returns:
            Dictionary with summary information
        """
        return {
            'total_videos': len(self.state['videos']),
            'stats': self.get_stats(),
            'metadata': self.state['metadata'],
            'recent_videos': [
                {
                    'video_id': video.video_id,
                    'title': video.title[:50] + '...' if len(video.title) > 50 else video.title,
                    'status': video.status.value,
                    'last_updated': video.last_updated
                }
                for video in sorted(self.state['videos'].values(), 
                                  key=lambda x: x.last_updated, reverse=True)[:10]
            ]
        }

if __name__ == "__main__":
    # Test the state manager
    state_manager = StateManager()
    
    # Add a test video
    test_video = {
        'video_id': 'test123',
        'title': 'Test Video',
        'channel': 'Test Channel',
        'description': 'Test description',
        'published_at': '20231201',
        'duration': 600,
        'duration_str': '10:00',
        'view_count': 1000,
        'url': 'https://youtube.com/watch?v=test123',
        'search_term': 'test'
    }
    
    state_manager.add_video(test_video)
    print("Added test video")
    
    # Update status
    state_manager.update_video_status('test123', VideoStatus.DOWNLOADED)
    print("Updated video status to downloaded")
    
    # Get stats
    stats = state_manager.get_stats()
    print(f"Stats: {stats}")
    
    # Export summary
    summary = state_manager.export_summary()
    print(f"Summary: {summary}")