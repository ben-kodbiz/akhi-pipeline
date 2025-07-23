import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import subprocess
from datetime import datetime

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIPELINE_DIR = os.path.join(BASE_DIR, "pipeline")
OUTPUT_DIR = os.path.join(PIPELINE_DIR, "output")
CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips")
TRANSCRIPTS_DIR = os.path.join(OUTPUT_DIR, "transcripts")
DB_DIR = os.path.join(PIPELINE_DIR, "db")

# Create directories if they don't exist
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Status tracker file
STATUS_FILE = os.path.join(DB_DIR, "status_tracker.json")

class Transcriber:
    def __init__(self, model_size: str = "base", compute_type: str = "int8", device: str = "cpu"):
        """
        Initialize the Transcriber with Whisper model settings.
        
        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
            compute_type: Compute type for the model (float16, int8)
            device: Device to run the model on (cpu, cuda)
        """
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        
        # Transcription status
        self.transcription_status = {
            "is_running": False,
            "current_file": None,
            "progress": 0,
            "total_files": 0,
            "completed_files": 0,
            "error": None,
            "last_updated": None
        }
        
        # Load status tracker
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
    
    def get_untranscribed_files(self) -> List[Tuple[str, str]]:
        """
        Get a list of files that haven't been transcribed yet.
        
        Returns:
            List of tuples (file_path, video_id) for untranscribed files
        """
        untranscribed = []
        
        # Get all MP3 files in the clips directory
        mp3_files = [f for f in os.listdir(CLIPS_DIR) if f.endswith(".mp3")]
        
        # Check which files haven't been transcribed yet
        for file in mp3_files:
            file_path = os.path.join(CLIPS_DIR, file)
            base_name = os.path.splitext(file)[0]
            transcript_file = os.path.join(TRANSCRIPTS_DIR, f"{base_name}.txt")
            
            # If transcript file doesn't exist, add to untranscribed list
            if not os.path.exists(transcript_file):
                # Try to find video_id in status tracker
                video_id = None
                for vid_id, video_data in self.status_tracker["videos"].items():
                    if base_name in video_data.get("title", ""):
                        video_id = vid_id
                        break
                
                untranscribed.append((file_path, video_id))
        
        return untranscribed
    
    def transcribe_all(self, max_files: Optional[int] = None) -> Dict[str, Any]:
        """
        Transcribe all untranscribed files.
        
        Args:
            max_files: Maximum number of files to transcribe
            
        Returns:
            Dictionary with transcription results
        """
        # Get untranscribed files
        untranscribed = self.get_untranscribed_files()
        
        if max_files is not None:
            untranscribed = untranscribed[:max_files]
        
        total_files = len(untranscribed)
        
        if total_files == 0:
            print("No files to transcribe.")
            return {
                "success": True,
                "files_transcribed": 0,
                "message": "No files to transcribe"
            }
        
        # Update transcription status
        self.transcription_status.update({
            "is_running": True,
            "current_file": None,
            "progress": 0,
            "total_files": total_files,
            "completed_files": 0,
            "error": None,
            "last_updated": datetime.now().isoformat()
        })
        
        print(f"Starting transcription of {total_files} files")
        
        # Import faster-whisper here to avoid import errors if not installed
        try:
            from faster_whisper import WhisperModel
            
            # Initialize the Whisper model
            model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            
            # Transcribe each file
            for i, (file_path, video_id) in enumerate(untranscribed):
                file_name = os.path.basename(file_path)
                
                try:
                    # Update current file status
                    self.transcription_status.update({
                        "current_file": file_name,
                        "progress": int((i / total_files) * 100),
                        "last_updated": datetime.now().isoformat()
                    })
                    
                    print(f"Transcribing file {i+1}/{total_files}: {file_name}")
                    
                    # Transcribe using faster-whisper Python API with word-level timestamps
                    segments, info = model.transcribe(
                        file_path, 
                        beam_size=5,
                        word_timestamps=True,  # Enable word-level timestamps
                        vad_filter=True,       # Voice activity detection
                        vad_parameters=dict(min_silence_duration_ms=500)  # Adjust for speaker pauses
                    )
                    
                    # Process segments with timestamps and potential speaker detection
                    processed_segments = []
                    current_speaker = "Speaker 1"  # Default speaker
                    
                    for segment in segments:
                        # Simple heuristic for speaker change: significant pause between segments
                        if processed_segments and (segment.start - processed_segments[-1]["end"]) > 2.0:
                            # Alternate between speakers on significant pauses
                            if current_speaker == "Speaker 1":
                                current_speaker = "Speaker 2"
                            else:
                                current_speaker = "Speaker 1"
                        
                        processed_segments.append({
                            "speaker": current_speaker,
                            "text": segment.text,
                            "start": segment.start,
                            "end": segment.end,
                            "words": [
                                {"word": word.word, "start": word.start, "end": word.end, "probability": word.probability}
                                for word in segment.words
                            ] if segment.words else []
                        })
                    
                    # Generate enhanced transcript with timestamps and speakers
                    transcript_text = ""
                    json_transcript = {
                        "metadata": {
                            "file": file_name,
                            "duration": processed_segments[-1]["end"] if processed_segments else 0,
                            "language": info.language,
                            "language_probability": info.language_probability
                        },
                        "segments": processed_segments
                    }
                    
                    # Generate readable transcript with timestamps and speakers
                    for segment in processed_segments:
                        timestamp = self._format_timestamp(segment["start"]) + " - " + self._format_timestamp(segment["end"])
                        transcript_text += f"[{timestamp}] {segment['speaker']}: {segment['text']}\n\n"
                    
                    # Save transcription to text file
                    base_name = os.path.splitext(file_name)[0]
                    transcript_file = os.path.join(TRANSCRIPTS_DIR, f"{base_name}.txt")
                    json_transcript_file = os.path.join(TRANSCRIPTS_DIR, f"{base_name}.json")
                    
                    with open(transcript_file, "w", encoding="utf-8") as f:
                        f.write(transcript_text.strip())
                    
                    with open(json_transcript_file, "w", encoding="utf-8") as f:
                        json.dump(json_transcript, f, indent=2, ensure_ascii=False)
                    
                    # Update video status if video_id is available
                    if video_id:
                        self.update_video_status(video_id, "transcribed", True)
                    
                    # Update completed files count
                    self.transcription_status.update({
                        "completed_files": i + 1,
                        "progress": int(((i + 1) / total_files) * 100),
                        "last_updated": datetime.now().isoformat()
                    })
                    
                    print(f"Successfully transcribed: {file_name}")
                    
                except Exception as e:
                    error_message = f"Error transcribing {file_name}: {str(e)}"
                    print(error_message)
                    self.transcription_status.update({
                        "error": error_message,
                        "last_updated": datetime.now().isoformat()
                    })
            
            # Update transcription status
            self.transcription_status.update({
                "is_running": False,
                "current_file": None,
                "progress": 100,
                "last_updated": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "files_transcribed": self.transcription_status["completed_files"],
                "message": f"Successfully transcribed {self.transcription_status['completed_files']} files"
            }
            
        except ImportError:
            error_message = "faster-whisper not installed. Please install it with 'pip install faster-whisper'."
            print(error_message)
            self.transcription_status.update({
                "is_running": False,
                "error": error_message,
                "last_updated": datetime.now().isoformat()
            })
            
            return {
                "success": False,
                "files_transcribed": 0,
                "message": error_message
            }
        except Exception as e:
            error_message = f"Unexpected error during transcription: {str(e)}"
            print(error_message)
            self.transcription_status.update({
                "is_running": False,
                "error": error_message,
                "last_updated": datetime.now().isoformat()
            })
            
            return {
                "success": False,
                "files_transcribed": self.transcription_status["completed_files"],
                "message": error_message
            }
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds to HH:MM:SS.MS string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{int(hours):02d}:{int(minutes):02d}:{seconds:.2f}"
        else:
            return f"{int(minutes):02d}:{seconds:.2f}"
    
    def get_transcription_status(self) -> Dict[str, Any]:
        """
        Get the current transcription status.
        
        Returns:
            Dictionary with transcription status
        """
        return self.transcription_status


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcriber for Islamic Scholar Videos")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (cpu, cuda)")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to transcribe")
    
    args = parser.parse_args()
    
    transcriber = Transcriber(model_size=args.model, device=args.device)
    result = transcriber.transcribe_all(max_files=args.max_files)
    
    print(f"\nTranscription completed: {result['message']}")