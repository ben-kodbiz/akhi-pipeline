import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIPELINE_DIR = os.path.join(BASE_DIR, "pipeline")
OUTPUT_DIR = os.path.join(PIPELINE_DIR, "output")
TRANSCRIPTS_DIR = os.path.join(OUTPUT_DIR, "transcripts")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")
DB_DIR = os.path.join(PIPELINE_DIR, "db")

# Create directories if they don't exist
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Status tracker file
STATUS_FILE = os.path.join(DB_DIR, "status_tracker.json")

# Output JSON file
OUTPUT_JSON_FILE = os.path.join(JSON_DIR, "akhi_lora.json")

class QLoRAFormatter:
    def __init__(self, min_segment_words: int = 50, max_segment_words: int = 500):
        """
        Initialize the QLoRA Formatter.
        
        Args:
            min_segment_words: Minimum number of words for a segment to be included
            max_segment_words: Maximum number of words for a segment (longer will be split)
        """
        self.min_segment_words = min_segment_words
        self.max_segment_words = max_segment_words
        
        # JSON generation status
        self.json_generation_status = {
            "is_running": False,
            "current_step": None,
            "progress": 0,
            "total_files": 0,
            "processed_files": 0,
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
    
    def _get_transcript_files(self) -> List[Tuple[str, Optional[str]]]:
        """
        Get a list of transcript files and their associated video IDs.
        
        Returns:
            List of tuples (file_path, video_id) for transcript files
        """
        transcript_files = []
        
        # Get all transcript files
        for file in os.listdir(TRANSCRIPTS_DIR):
            if file.endswith(".txt"):
                file_path = os.path.join(TRANSCRIPTS_DIR, file)
                base_name = os.path.splitext(file)[0]
                
                # Try to find video_id in status tracker
                video_id = None
                for vid_id, video_data in self.status_tracker["videos"].items():
                    if base_name in video_data.get("title", ""):
                        video_id = vid_id
                        break
                
                transcript_files.append((file_path, video_id))
        
        return transcript_files
    
    def _parse_transcript(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a transcript file into segments.
        
        Args:
            file_path: Path to the transcript file
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        
        try:
            # Check if JSON transcript exists (enhanced format)
            json_path = os.path.splitext(file_path)[0] + ".json"
            
            if os.path.exists(json_path):
                # Parse the enhanced JSON transcript
                with open(json_path, "r", encoding="utf-8") as f:
                    transcript_data = json.load(f)
                
                # Process segments from JSON
                return self._process_json_transcript(transcript_data)
            else:
                # Parse the plain text transcript
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                
                # Process plain text transcript
                return self._process_text_transcript(content)
        
        except Exception as e:
            print(f"Error parsing transcript {file_path}: {str(e)}")
            return []
    
    def _process_json_transcript(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process an enhanced JSON transcript into segments.
        
        Args:
            transcript_data: JSON transcript data
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        current_segment = {
            "speaker": None,
            "text": "",
            "start": None,
            "end": None
        }
        
        # Group by speaker for dialogue format
        for segment in transcript_data.get("segments", []):
            speaker = segment.get("speaker")
            text = segment.get("text", "").strip()
            
            if not text:
                continue
            
            # Start a new segment if speaker changes or first segment
            if current_segment["speaker"] != speaker or current_segment["speaker"] is None:
                # Save previous segment if it exists
                if current_segment["speaker"] is not None and current_segment["text"]:
                    segments.append(current_segment.copy())
                
                # Start new segment
                current_segment = {
                    "speaker": speaker,
                    "text": text,
                    "start": segment.get("start"),
                    "end": segment.get("end")
                }
            else:
                # Continue current segment
                current_segment["text"] += " " + text
                current_segment["end"] = segment.get("end")
        
        # Add the last segment
        if current_segment["speaker"] is not None and current_segment["text"]:
            segments.append(current_segment)
        
        return segments
    
    def _process_text_transcript(self, content: str) -> List[Dict[str, Any]]:
        """
        Process a plain text transcript into segments.
        
        Args:
            content: Transcript content
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        
        # Try to parse timestamp format: [MM:SS.MS - MM:SS.MS] Speaker: Text
        timestamp_pattern = r'\[(\d+:\d+\.\d+) - (\d+:\d+\.\d+)\] ([^:]+): (.+)'
        matches = re.findall(timestamp_pattern, content)
        
        if matches:
            for start_time, end_time, speaker, text in matches:
                segments.append({
                    "speaker": speaker.strip(),
                    "text": text.strip(),
                    "start": self._parse_timestamp(start_time),
                    "end": self._parse_timestamp(end_time)
                })
        else:
            # If no timestamp format, try to identify speakers with Speaker: Text pattern
            speaker_pattern = r'([^:]+): (.+)'
            matches = re.findall(speaker_pattern, content)
            
            if matches:
                for speaker, text in matches:
                    segments.append({
                        "speaker": speaker.strip(),
                        "text": text.strip(),
                        "start": None,
                        "end": None
                    })
            else:
                # If no speaker format, treat as a single segment
                segments.append({
                    "speaker": "Unknown",
                    "text": content,
                    "start": None,
                    "end": None
                })
        
        return segments
    
    def _parse_timestamp(self, timestamp: str) -> float:
        """
        Parse a timestamp string to seconds.
        
        Args:
            timestamp: Timestamp string (MM:SS.MS or HH:MM:SS.MS)
            
        Returns:
            Time in seconds
        """
        parts = timestamp.split(":")
        
        if len(parts) == 2:  # MM:SS.MS
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        elif len(parts) == 3:  # HH:MM:SS.MS
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        else:
            return 0.0
    
    def _generate_training_examples(self, segments: List[Dict[str, Any]], file_name: str) -> List[Dict[str, Any]]:
        """
        Generate training examples from transcript segments.
        
        Args:
            segments: List of transcript segments
            file_name: Name of the transcript file
            
        Returns:
            List of training examples
        """
        examples = []
        
        # If no segments, return empty list
        if not segments:
            return examples
        
        # Extract base name without extension
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        
        # Group segments for different training formats
        
        # 1. Individual segments as examples (for longer segments)
        for segment in segments:
            text = segment["text"]
            word_count = len(text.split())
            
            if word_count >= self.min_segment_words:
                # For longer segments, create individual examples
                if word_count <= self.max_segment_words:
                    examples.append({
                        "instruction": "Summarize and offer Islamic advice based on this:",
                        "input": text,
                        "output": "Remember, Allah is always with those who are patient and sincere. This teaching reminds us that [insert relevant Islamic principle based on the content].",
                        "metadata": {
                            "source": base_name,
                            "speaker": segment["speaker"],
                            "type": "segment"
                        }
                    })
                else:
                    # Split very long segments
                    words = text.split()
                    for i in range(0, len(words), self.max_segment_words):
                        chunk = " ".join(words[i:i+self.max_segment_words])
                        if len(chunk.split()) >= self.min_segment_words:
                            examples.append({
                                "instruction": "Summarize and offer Islamic advice based on this:",
                                "input": chunk,
                                "output": "Remember, Allah is always with those who are patient and sincere. This teaching reminds us that [insert relevant Islamic principle based on the content].",
                                "metadata": {
                                    "source": base_name,
                                    "speaker": segment["speaker"],
                                    "type": "segment_chunk"
                                }
                            })
        
        # 2. Dialogue format (for segments with different speakers)
        if len(segments) >= 2:
            # Group segments into dialogues (2-4 segments per dialogue)
            for i in range(0, len(segments), 3):  # Overlap dialogues for more examples
                dialogue_segments = segments[i:i+4]  # Get up to 4 segments
                
                if len(dialogue_segments) >= 2:  # Need at least 2 segments for a dialogue
                    dialogue_text = ""
                    for seg in dialogue_segments:
                        dialogue_text += f"{seg['speaker']}: {seg['text']}\n\n"
                    
                    word_count = len(dialogue_text.split())
                    if word_count >= self.min_segment_words:
                        examples.append({
                            "instruction": "Based on this Islamic dialogue, provide a thoughtful response that captures the key teachings:",
                            "input": dialogue_text,
                            "output": "This dialogue highlights important Islamic principles including [insert relevant principles]. As Muslims, we should reflect on these teachings by [insert practical advice].",
                            "metadata": {
                                "source": base_name,
                                "type": "dialogue"
                            }
                        })
        
        # 3. Full lecture summary (combine all segments)
        full_text = ""
        for segment in segments:
            full_text += f"{segment['text']}\n\n"
        
        word_count = len(full_text.split())
        if word_count >= self.min_segment_words:
            # For very long lectures, create a summary example
            examples.append({
                "instruction": "Summarize the key points from this Islamic lecture:",
                "input": full_text[:2000],  # Limit to first 2000 chars for summary
                "output": "This lecture covers several important Islamic concepts including [insert key points]. The scholar emphasizes [insert main emphasis], which reminds us of our duty to [insert relevant duty/practice].",
                "metadata": {
                    "source": base_name,
                    "type": "lecture_summary"
                }
            })
        
        return examples
    
    def generate_json(self) -> Dict[str, Any]:
        """
        Generate QLoRA-compatible JSON from transcripts.
        
        Returns:
            Dictionary with generation results
        """
        # Update JSON generation status
        self.json_generation_status.update({
            "is_running": True,
            "current_step": "Gathering transcript files",
            "progress": 0,
            "last_updated": datetime.now().isoformat()
        })
        
        # Get transcript files
        transcript_files = self._get_transcript_files()
        total_files = len(transcript_files)
        
        if total_files == 0:
            self.json_generation_status.update({
                "is_running": False,
                "error": "No transcript files found",
                "last_updated": datetime.now().isoformat()
            })
            
            return {
                "success": False,
                "message": "No transcript files found"
            }
        
        self.json_generation_status.update({
            "total_files": total_files,
            "processed_files": 0,
            "current_step": "Processing transcripts",
            "last_updated": datetime.now().isoformat()
        })
        
        # Process each transcript file
        all_examples = []
        
        for i, (file_path, video_id) in enumerate(transcript_files):
            try:
                file_name = os.path.basename(file_path)
                
                self.json_generation_status.update({
                    "current_step": f"Processing {file_name}",
                    "progress": int((i / total_files) * 100),
                    "last_updated": datetime.now().isoformat()
                })
                
                print(f"Processing transcript {i+1}/{total_files}: {file_name}")
                
                # Parse transcript into segments
                segments = self._parse_transcript(file_path)
                
                # Generate training examples
                examples = self._generate_training_examples(segments, file_name)
                
                # Add examples to the list
                all_examples.extend(examples)
                
                # Update video status if video_id is available
                if video_id:
                    self.update_video_status(video_id, "json_ready", True)
                
                # Update processed files count
                self.json_generation_status.update({
                    "processed_files": i + 1,
                    "progress": int(((i + 1) / total_files) * 100),
                    "last_updated": datetime.now().isoformat()
                })
                
                print(f"Generated {len(examples)} examples from {file_name}")
                
            except Exception as e:
                error_message = f"Error processing {file_path}: {str(e)}"
                print(error_message)
        
        # Save all examples to JSON file
        self.json_generation_status.update({
            "current_step": "Saving JSON file",
            "last_updated": datetime.now().isoformat()
        })
        
        try:
            with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
                json.dump(all_examples, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(all_examples)} examples to {OUTPUT_JSON_FILE}")
            
            # Update JSON generation status
            self.json_generation_status.update({
                "is_running": False,
                "current_step": "Completed",
                "progress": 100,
                "last_updated": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "examples_generated": len(all_examples),
                "files_processed": self.json_generation_status["processed_files"],
                "message": f"Successfully generated {len(all_examples)} examples from {self.json_generation_status['processed_files']} files"
            }
            
        except Exception as e:
            error_message = f"Error saving JSON file: {str(e)}"
            print(error_message)
            
            self.json_generation_status.update({
                "is_running": False,
                "error": error_message,
                "last_updated": datetime.now().isoformat()
            })
            
            return {
                "success": False,
                "message": error_message
            }
    
    def get_json_generation_status(self) -> Dict[str, Any]:
        """
        Get the current JSON generation status.
        
        Returns:
            Dictionary with JSON generation status
        """
        return self.json_generation_status


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QLoRA Formatter for Islamic Scholar Videos")
    parser.add_argument("--min-words", type=int, default=50, help="Minimum number of words for a segment to be included")
    parser.add_argument("--max-words", type=int, default=500, help="Maximum number of words for a segment")
    
    args = parser.parse_args()
    
    formatter = QLoRAFormatter(min_segment_words=args.min_words, max_segment_words=args.max_words)
    result = formatter.generate_json()
    
    print(f"\nJSON generation completed: {result['message']}")