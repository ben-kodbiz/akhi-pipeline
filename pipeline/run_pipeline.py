#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add the parent directory to the path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agents
from pipeline.agents.youtube_agent import YouTubeAgent
from pipeline.agents.transcriber import Transcriber
from pipeline.agents.qlora_formatter import QLoRAFormatter

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

def log_message(message: str) -> None:
    """
    Log a message to the console and log file.
    
    Args:
        message: Message to log
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    print(log_entry)
    
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def run_search(max_results: int = 20, search_terms: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run the search step of the pipeline using yt-dlp.
    
    Args:
        max_results: Maximum number of results per search term
        search_terms: List of search terms
        
    Returns:
        Dictionary with search results
    """
    log_message("Starting search step")
    
    agent = YouTubeAgent()
    videos = agent.search_videos(search_terms=search_terms, max_results_per_term=max_results)
    
    log_message(f"Found {len(videos)} new videos")
    
    return {
        "success": True,
        "videos_found": len(videos),
        "videos": videos
    }

def run_download(max_videos: int = 10) -> Dict[str, Any]:
    """
    Run the download step of the pipeline.
    
    Args:
        max_videos: Maximum number of videos to download
        
    Returns:
        Dictionary with download results
    """
    log_message("Starting download step")
    
    agent = YouTubeAgent()
    links_file = agent.generate_download_links_file(max_videos=max_videos)
    
    if not os.path.exists(links_file) or os.path.getsize(links_file) == 0:
        log_message("No videos to download")
        return {
            "success": True,
            "videos_downloaded": 0,
            "message": "No videos to download"
        }
    
    # Count number of links
    with open(links_file, "r") as f:
        links = f.read().strip().split("\n")
    
    num_links = len(links)
    log_message(f"Downloading {num_links} videos")
    
    # Run yt-dlp command
    try:
        result = subprocess.run([
            "yt-dlp", "-a", links_file, 
            "-f", "ba",  # Best audio format
            "-x",  # Extract audio
            "--audio-format", "mp3", 
            "--audio-quality", "0",
            "--no-playlist", 
            "--verbose", 
            "--no-check-certificate",
            "--geo-bypass",
            "-o", f"{CLIPS_DIR}/%(title)s.%(ext)s"
        ], capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            log_message(f"Successfully downloaded {num_links} videos")
            return {
                "success": True,
                "videos_downloaded": num_links,
                "message": f"Successfully downloaded {num_links} videos"
            }
        else:
            log_message(f"Error downloading videos: {result.stderr}")
            return {
                "success": False,
                "videos_downloaded": 0,
                "message": f"Error downloading videos: {result.stderr}"
            }
    except Exception as e:
        log_message(f"Exception during download: {str(e)}")
        return {
            "success": False,
            "videos_downloaded": 0,
            "message": f"Exception during download: {str(e)}"
        }

def run_transcribe(model_size: str = "base", device: str = "cpu", max_files: Optional[int] = None) -> Dict[str, Any]:
    """
    Run the transcription step of the pipeline.
    
    Args:
        model_size: Size of the Whisper model
        device: Device to run the model on
        max_files: Maximum number of files to transcribe
        
    Returns:
        Dictionary with transcription results
    """
    log_message("Starting transcription step")
    
    transcriber = Transcriber(model_size=model_size, device=device)
    result = transcriber.transcribe_all(max_files=max_files)
    
    if result["success"]:
        log_message(f"Successfully transcribed {result['files_transcribed']} files")
    else:
        log_message(f"Error during transcription: {result['message']}")
    
    return result

def run_format_json(min_words: int = 50, max_words: int = 500) -> Dict[str, Any]:
    """
    Run the JSON formatting step of the pipeline.
    
    Args:
        min_words: Minimum number of words for a segment to be included
        max_words: Maximum number of words for a segment
        
    Returns:
        Dictionary with formatting results
    """
    log_message("Starting JSON formatting step")
    
    formatter = QLoRAFormatter(min_segment_words=min_words, max_segment_words=max_words)
    result = formatter.generate_json()
    
    if result["success"]:
        log_message(f"Successfully generated {result['examples_generated']} examples from {result['files_processed']} files")
    else:
        log_message(f"Error during JSON formatting: {result['message']}")
    
    return result

def run_full_pipeline(args: argparse.Namespace) -> None:
    """
    Run the full pipeline.
    
    Args:
        args: Command line arguments
    """
    log_message("Starting full pipeline")
    
    # Step 1: Search
    if args.search:
        search_result = run_search(max_results=args.max_results, search_terms=args.search_terms)
        if not search_result["success"]:
            log_message("Search step failed. Stopping pipeline.")
            return
    
    # Step 2: Download
    if args.download:
        download_result = run_download(max_videos=args.max_videos)
        if not download_result["success"]:
            log_message("Download step failed. Stopping pipeline.")
            return
    
    # Step 3: Transcribe
    if args.transcribe:
        transcribe_result = run_transcribe(model_size=args.model, device=args.device, max_files=args.max_files)
        if not transcribe_result["success"]:
            log_message("Transcription step failed. Stopping pipeline.")
            return
    
    # Step 4: Format JSON
    if args.format:
        format_result = run_format_json(min_words=args.min_words, max_words=args.max_words)
        if not format_result["success"]:
            log_message("JSON formatting step failed. Stopping pipeline.")
            return
    
    log_message("Pipeline completed successfully")

def main():
    parser = argparse.ArgumentParser(description="Akhi Data Builder Pipeline")
    
    # Pipeline steps
    parser.add_argument("--search", action="store_true", help="Run the search step")
    parser.add_argument("--download", action="store_true", help="Run the download step")
    parser.add_argument("--transcribe", action="store_true", help="Run the transcription step")
    parser.add_argument("--format", action="store_true", help="Run the JSON formatting step")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    # Search options
    parser.add_argument("--max-results", type=int, default=20, help="Maximum number of results per search term")
    parser.add_argument("--search-terms", type=str, nargs="+", help="Search terms")
    
    # Download options
    parser.add_argument("--max-videos", type=int, default=10, help="Maximum number of videos to download")
    
    # Transcription options
    parser.add_argument("--model", type=str, default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (cpu, cuda)")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to transcribe")
    
    # JSON formatting options
    parser.add_argument("--min-words", type=int, default=50, help="Minimum number of words for a segment to be included")
    parser.add_argument("--max-words", type=int, default=500, help="Maximum number of words for a segment")
    
    args = parser.parse_args()
    
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
    run_full_pipeline(args)

if __name__ == "__main__":
    main()