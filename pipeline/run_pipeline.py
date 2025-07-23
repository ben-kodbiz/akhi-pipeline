#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add the parent directory to the path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agents and managers
from pipeline.agents.youtube_agent import YouTubeAgent
from pipeline.agents.transcriber import Transcriber
from pipeline.agents.qlora_formatter import QLoRAFormatter
from pipeline.config_manager import ConfigManager
from pipeline.state_manager import StateManager, VideoStatus

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(BASE_DIR, "pipeline")
OUTPUT_DIR = os.path.join(PIPELINE_DIR, "output")
CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips")
TRANSCRIPTS_DIR = os.path.join(OUTPUT_DIR, "transcripts")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")
DB_DIR = os.path.join(PIPELINE_DIR, "db")

# Create directories if they don't exist
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Status tracker file
STATUS_FILE = os.path.join(DB_DIR, "status_tracker.json")

# Pipeline log file
LOG_FILE = os.path.join(DB_DIR, "pipeline_log.txt")

def setup_logging(config_manager: ConfigManager) -> None:
    """
    Set up logging configuration using config manager.
    
    Args:
        config_manager: Configuration manager instance
    """
    logging_config = config_manager.get_logging_config()
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=getattr(logging, logging_config.level.upper()),
        format=logging_config.format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Log level: {logging_config.level}")
    logger.info(f"Max file size: {logging_config.max_file_size}MB")

def run_search(config_manager: ConfigManager, 
              state_manager: StateManager,
              search_queries: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run the search step of the pipeline.
    
    Args:
        config_manager: Configuration manager instance
        state_manager: State manager instance
        search_queries: List of search queries to use
        
    Returns:
        Dictionary with search results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting search step")
    
    try:
        # Initialize YouTube agent
        youtube_agent = YouTubeAgent(config_manager=config_manager, state_manager=state_manager)
        search_config = config_manager.get_search_config()
        
        # Use default search terms if none provided
        if not search_queries:
            search_queries = search_config.default_terms
            logger.info(f"Using default search terms: {search_queries}")
        
        all_videos = []
        
        # Search for each query
        for query in search_queries:
            logger.info(f"Searching for: {query}")
            
            videos = youtube_agent.search_and_filter_videos(
                yt_search_query=query,
                max_results=search_config.max_results,
                min_duration=search_config.min_duration,
                max_duration=search_config.max_duration
            )
            
            all_videos.extend(videos)
            logger.info(f"Found {len(videos)} videos for query: {query}")
        
        logger.info(f"Search completed. Total found: {len(all_videos)} videos")
        
        # Generate download list if videos found
        if all_videos:
            download_config = config_manager.get_download_config()
            videos_to_download = youtube_agent.get_videos_to_download(limit=download_config.batch_size)
            if videos_to_download:
                download_file = youtube_agent.create_download_file(videos_to_download)
                logger.info(f"Generated download file: {download_file}")
        
        return {
            "success": True,
            "videos_found": len(all_videos),
            "videos": all_videos
        }
        
    except Exception as e:
        logger.error(f"Error in search step: {e}")
        return {
            "success": False,
            "videos_found": 0,
            "error": str(e)
        }

def run_download(config_manager: ConfigManager, 
                state_manager: StateManager,
                max_videos: Optional[int] = None) -> Dict[str, Any]:
    """
    Run the download step of the pipeline.
    
    Args:
        config_manager: Configuration manager instance
        state_manager: State manager instance
        max_videos: Maximum number of videos to download (overrides config)
        
    Returns:
        Dictionary with download results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting download step")
    
    try:
        download_config = config_manager.get_download_config()
        
        # Get videos ready for download
        videos_to_download = state_manager.get_videos_by_status(VideoStatus.DISCOVERED)
        
        if not videos_to_download:
            logger.warning("No videos found for download. Run search first.")
            return {
                "success": False,
                "videos_downloaded": 0,
                "error": "No videos found for download"
            }
        
        # Limit the number of videos to download
        limit = max_videos or download_config.batch_size
        videos_to_download = videos_to_download[:limit]
        
        logger.info(f"Downloading {len(videos_to_download)} videos...")
        
        downloaded_count = 0
        failed_count = 0
        
        for i, video in enumerate(videos_to_download, 1):
            logger.info(f"Downloading video {i}/{len(videos_to_download)}: {video.title}")
            
            try:
                # Mark as downloading
                state_manager.update_video_status(video.video_id, VideoStatus.DOWNLOADING)
                
                # Use yt-dlp to download audio only
                cmd = [
                    "yt-dlp",
                    "--extract-audio",
                    "--audio-format", download_config.audio_format,
                    "--audio-quality", "0",
                    "--output", os.path.join(CLIPS_DIR, "%(title)s.%(ext)s"),
                    video.url
                ]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=download_config.timeout
                )
                
                if result.returncode == 0:
                    downloaded_count += 1
                    state_manager.update_video_status(video.video_id, VideoStatus.DOWNLOADED)
                    logger.info(f"Successfully downloaded: {video.title}")
                else:
                    failed_count += 1
                    state_manager.update_video_status(video.video_id, VideoStatus.FAILED)
                    logger.error(f"Failed to download: {video.title} - {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                failed_count += 1
                state_manager.update_video_status(video.video_id, VideoStatus.FAILED)
                logger.error(f"Download timeout for: {video.title}")
            except Exception as e:
                failed_count += 1
                state_manager.update_video_status(video.video_id, VideoStatus.FAILED)
                logger.error(f"Download error for {video.title}: {e}")
        
        logger.info(f"Download completed. Success: {downloaded_count}, Failed: {failed_count}")
        
        return {
            "success": True,
            "videos_downloaded": downloaded_count,
            "videos_failed": failed_count
        }
        
    except Exception as e:
        logger.error(f"Error in download step: {e}")
        return {
            "success": False,
            "videos_downloaded": 0,
            "error": str(e)
        }

def run_transcribe(config_manager: ConfigManager, 
                  state_manager: StateManager,
                  max_files: Optional[int] = None) -> Dict[str, Any]:
    """
    Run the transcription step of the pipeline.
    
    Args:
        config_manager: Configuration manager instance
        state_manager: State manager instance
        max_files: Maximum number of files to transcribe (overrides config)
        
    Returns:
        Dictionary with transcription results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting transcription step")
    
    try:
        transcription_config = config_manager.get_transcription_config()
        
        # Get videos ready for transcription
        videos_to_transcribe = state_manager.get_videos_by_status(VideoStatus.DOWNLOADED)
        
        if not videos_to_transcribe:
            logger.warning("No videos found for transcription. Run download first.")
            return {
                "success": False,
                "files_transcribed": 0,
                "error": "No videos found for transcription"
            }
        
        # Limit the number of files to transcribe
        limit = max_files or transcription_config.batch_size
        videos_to_transcribe = videos_to_transcribe[:limit]
        
        logger.info(f"Transcribing {len(videos_to_transcribe)} audio files...")
        
        transcriber = Transcriber(config_manager=config_manager)
        transcribed_count = 0
        failed_count = 0
        
        for i, video in enumerate(videos_to_transcribe, 1):
            logger.info(f"Transcribing file {i}/{len(videos_to_transcribe)}: {video.title}")
            
            try:
                # Mark as transcribing
                state_manager.update_video_status(video.video_id, VideoStatus.TRANSCRIBING)
                
                # Find the audio file (assuming it was downloaded with the video title)
                audio_files = [f for f in os.listdir(CLIPS_DIR) 
                             if f.startswith(video.title.replace('/', '_').replace('\\', '_')[:50]) 
                             and f.endswith(('.mp3', '.wav', '.m4a'))]
                
                if not audio_files:
                    logger.warning(f"Audio file not found for: {video.title}")
                    state_manager.update_video_status(video.video_id, VideoStatus.FAILED)
                    failed_count += 1
                    continue
                
                audio_path = os.path.join(CLIPS_DIR, audio_files[0])
                transcript = transcriber.transcribe_audio(audio_path)
                
                if transcript:
                    # Save transcript
                    safe_title = video.title.replace('/', '_').replace('\\', '_')[:100]
                    transcript_file = os.path.join(TRANSCRIPTS_DIR, f"{safe_title}.txt")
                    with open(transcript_file, "w", encoding="utf-8") as f:
                        f.write(transcript)
                    
                    transcribed_count += 1
                    state_manager.update_video_status(video.video_id, VideoStatus.TRANSCRIBED)
                    logger.info(f"Successfully transcribed: {video.title}")
                else:
                    failed_count += 1
                    state_manager.update_video_status(video.video_id, VideoStatus.FAILED)
                    logger.error(f"Failed to transcribe: {video.title}")
                    
            except Exception as e:
                failed_count += 1
                state_manager.update_video_status(video.video_id, VideoStatus.FAILED)
                logger.error(f"Transcription error for {video.title}: {e}")
        
        logger.info(f"Transcription completed. Success: {transcribed_count}, Failed: {failed_count}")
        
        return {
            "success": True,
            "files_transcribed": transcribed_count,
            "files_failed": failed_count
        }
        
    except Exception as e:
        logger.error(f"Error in transcription step: {e}")
        return {
            "success": False,
            "files_transcribed": 0,
            "error": str(e)
        }

def run_format_json(config_manager: ConfigManager, 
                    state_manager: StateManager,
                    max_files: Optional[int] = None) -> Dict[str, Any]:
    """
    Run the JSON formatting step of the pipeline.
    
    Args:
        config_manager: Configuration manager instance
        state_manager: State manager instance
        max_files: Maximum number of files to format (overrides config)
        
    Returns:
        Dictionary with formatting results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting JSON formatting step")
    
    try:
        qlora_config = config_manager.get_qlora_config()
        
        # Get videos ready for formatting
        videos_to_format = state_manager.get_videos_by_status(VideoStatus.TRANSCRIBED)
        
        if not videos_to_format:
            logger.warning("No videos found for formatting. Run transcription first.")
            return {
                "success": False,
                "files_formatted": 0,
                "error": "No videos found for formatting"
            }
        
        # Limit the number of files to format
        limit = max_files or qlora_config.batch_size
        videos_to_format = videos_to_format[:limit]
        
        logger.info(f"Formatting {len(videos_to_format)} transcript files...")
        
        formatter = QLoRAFormatter(config_manager=config_manager)
        formatted_count = 0
        failed_count = 0
        
        for i, video in enumerate(videos_to_format, 1):
            logger.info(f"Formatting file {i}/{len(videos_to_format)}: {video.title}")
            
            try:
                # Mark as formatting
                state_manager.update_video_status(video.video_id, VideoStatus.FORMATTING)
                
                # Find the transcript file
                safe_title = video.title.replace('/', '_').replace('\\', '_')[:100]
                transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{safe_title}.txt")
                
                if not os.path.exists(transcript_path):
                    logger.warning(f"Transcript file not found for: {video.title}")
                    state_manager.update_video_status(video.video_id, VideoStatus.FAILED)
                    failed_count += 1
                    continue
                
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript_text = f.read()
                
                formatted_data = formatter.format_transcript(
                    transcript_text, 
                    video_metadata={
                        'title': video.title,
                        'channel': video.channel,
                        'url': video.url,
                        'duration': video.duration
                    }
                )
                
                if formatted_data:
                    # Save formatted JSON
                    json_file = os.path.join(JSON_DIR, f"{safe_title}.json")
                    with open(json_file, "w", encoding="utf-8") as f:
                        import json
                        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
                    
                    formatted_count += 1
                    state_manager.update_video_status(video.video_id, VideoStatus.COMPLETED)
                    logger.info(f"Successfully formatted: {video.title}")
                else:
                    failed_count += 1
                    state_manager.update_video_status(video.video_id, VideoStatus.FAILED)
                    logger.error(f"Failed to format: {video.title}")
                    
            except Exception as e:
                failed_count += 1
                state_manager.update_video_status(video.video_id, VideoStatus.FAILED)
                logger.error(f"Formatting error for {video.title}: {e}")
        
        logger.info(f"Formatting completed. Success: {formatted_count}, Failed: {failed_count}")
        
        return {
            "success": True,
            "files_formatted": formatted_count,
            "files_failed": failed_count
        }
        
    except Exception as e:
        logger.error(f"Error in formatting step: {e}")
        return {
            "success": False,
            "files_formatted": 0,
            "error": str(e)
        }

def run_full_pipeline(config_manager: ConfigManager, 
                     state_manager: StateManager,
                     args: argparse.Namespace) -> None:
    """
    Run the full pipeline from search to JSON formatting.
    
    Args:
        config_manager: Configuration manager instance
        state_manager: State manager instance
        args: Command line arguments
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting full pipeline")
    
    pipeline_config = config_manager.get_pipeline_config()
    
    try:
        # Step 1: Search
        if args.search:
            logger.info("Running search step")
            search_result = run_search(
                config_manager=config_manager,
                state_manager=state_manager,
                search_queries=args.search_terms
            )
            if not search_result["success"]:
                logger.error("Search step failed, stopping pipeline")
                if not pipeline_config.continue_on_error:
                    return
        
        # Step 2: Download
        if args.download:
            logger.info("Running download step")
            download_result = run_download(
                config_manager=config_manager,
                state_manager=state_manager,
                max_videos=args.max_videos
            )
            if not download_result["success"]:
                logger.error("Download step failed")
                if not pipeline_config.continue_on_error:
                    logger.error("Stopping pipeline")
                    return
        
        # Step 3: Transcribe
        if args.transcribe:
            logger.info("Running transcription step")
            transcribe_result = run_transcribe(
                config_manager=config_manager,
                state_manager=state_manager,
                max_files=args.max_files
            )
            if not transcribe_result["success"]:
                logger.error("Transcription step failed")
                if not pipeline_config.continue_on_error:
                    logger.error("Stopping pipeline")
                    return
        
        # Step 4: Format JSON
        if args.format:
            logger.info("Running JSON formatting step")
            format_result = run_format_json(
                config_manager=config_manager,
                state_manager=state_manager,
                max_files=args.max_files
            )
            if not format_result["success"]:
                logger.error("JSON formatting step failed")
                if not pipeline_config.continue_on_error:
                    logger.error("Stopping pipeline")
                    return
        
        logger.info("Full pipeline completed successfully")
        
        # Print summary
        stats = state_manager.get_statistics()
        logger.info(f"Pipeline Summary: {stats}")
        
    except Exception as e:
        logger.error(f"Error in full pipeline: {e}")
        raise

def main():
    """
    Main function to parse arguments and run the pipeline.
    """
    parser = argparse.ArgumentParser(description="YouTube to QLoRA Data Pipeline")
    
    # Configuration
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    
    # Pipeline steps
    parser.add_argument("--search", action="store_true", help="Run search step")
    parser.add_argument("--download", action="store_true", help="Run download step")
    parser.add_argument("--transcribe", action="store_true", help="Run transcription step")
    parser.add_argument("--format", action="store_true", help="Run JSON formatting step")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    # Override parameters
    parser.add_argument("--search-terms", nargs="+", help="Search terms to use (overrides config)")
    parser.add_argument("--max-videos", type=int, help="Maximum videos to download (overrides config)")
    parser.add_argument("--max-files", type=int, help="Maximum files to process (overrides config)")
    
    # State management
    parser.add_argument("--status", action="store_true", help="Show pipeline status")
    parser.add_argument("--summary", action="store_true", help="Show pipeline summary")
    parser.add_argument("--failed", action="store_true", help="Show failed videos")
    parser.add_argument("--reset", action="store_true", help="Reset pipeline state")
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration and state managers
        config_path = os.path.join(os.path.dirname(__file__), args.config)
        config_manager = ConfigManager(config_path)
        
        # Setup logging
        setup_logging(config_manager)
        logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        ensure_directories()
        
        # Initialize state manager
        state_file = os.path.join(DB_DIR, "video_state.jsonl")
        state_manager = StateManager(state_file)
        
        # Handle status commands
        if args.status:
            stats = state_manager.get_statistics()
            logger.info(f"Pipeline Status: {stats}")
            return
        
        if args.summary:
            from pipeline.agents.youtube_agent import YouTubeAgent
            youtube_agent = YouTubeAgent(config_manager=config_manager, state_manager=state_manager)
            summary = youtube_agent.export_summary()
            logger.info(f"Pipeline Summary: {summary}")
            return
        
        if args.failed:
            from pipeline.agents.youtube_agent import YouTubeAgent
            youtube_agent = YouTubeAgent(config_manager=config_manager, state_manager=state_manager)
            failed_videos = youtube_agent.get_failed_videos()
            logger.info(f"Failed Videos ({len(failed_videos)}):")
            for video in failed_videos:
                logger.info(f"  - {video.title} ({video.url})")
            return
        
        if args.reset:
            state_manager.reset_state()
            logger.info("Pipeline state reset")
            return
        
        # If --all is specified, run all steps
        if args.all:
            args.search = True
            args.download = True
            args.transcribe = True
            args.format = True
        
        # If no steps are specified, show help
        if not (args.search or args.download or args.transcribe or args.format):
            parser.print_help()
            return
        
        # Run the pipeline
        run_full_pipeline(config_manager, state_manager, args)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()