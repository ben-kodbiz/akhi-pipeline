from fastapi import FastAPI, BackgroundTasks, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import subprocess
import json
import shutil
from datetime import datetime

app = FastAPI(title="Akhi Data Builder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIPELINE_DIR = os.path.join(BASE_DIR, "pipeline")
OUTPUT_DIR = os.path.join(PIPELINE_DIR, "output")
CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips")
TRANSCRIPTS_DIR = os.path.join(OUTPUT_DIR, "transcripts")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")

# Global transcription status tracking
transcription_status = {
    "is_running": False,
    "current_file": None,
    "progress": 0,
    "total_files": 0,
    "completed_files": 0,
    "error": None,
    "last_updated": None
}

# Global JSON generation status tracking
json_generation_status = {
    "is_running": False,
    "current_step": None,
    "progress": 0,
    "total_files": 0,
    "processed_files": 0,
    "error": None,
    "last_updated": None
}

# Create directories if they don't exist
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# Models
class VideoLink(BaseModel):
    url: str
    title: Optional[str] = None

class VideoLinks(BaseModel):
    links: List[VideoLink]

class TranscriptEdit(BaseModel):
    file_name: str
    content: str

# Background tasks
def download_videos(links: List[str]):
    # Write links to temporary file
    links_file = os.path.join(PIPELINE_DIR, "temp_links.txt")
    with open(links_file, "w") as f:
        for link in links:
            f.write(f"{link}\n")
    
    try:
        # Run yt-dlp command with better error handling and debugging
        print(f"Starting download of {len(links)} videos with yt-dlp")
        print(f"Clips directory: {CLIPS_DIR}")
        print(f"Links: {links}")
        
        # Use the parameters that we've confirmed work with the upgraded yt-dlp
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
        ], capture_output=True, text=True, check=False)  # Changed to check=False to handle errors manually
        
        # Check if the command was successful
        if result.returncode == 0:
            print(f"Videos downloaded successfully")
            print(f"yt-dlp output: {result.stdout}")
            return "Videos downloaded successfully"
        else:
            # Handle specific error cases
            if "HTTP Error 403: Forbidden" in result.stderr:
                error_message = "YouTube is blocking the download. This is a common issue with YouTube's restrictions."
                print(f"YouTube blocking error: {result.stderr}")
                raise Exception(error_message)
            else:
                error_message = f"Error running yt-dlp (code {result.returncode}):\nOutput: {result.stdout}\nError: {result.stderr}"
                print(error_message)
                raise Exception(f"Error downloading videos: {result.stderr}")
    except subprocess.CalledProcessError as e:
        error_message = f"Error running yt-dlp: {e}\nOutput: {e.stdout}\nError: {e.stderr}"
        print(error_message)
        raise Exception(f"Error downloading videos: {e.stderr}")
    except Exception as e:
        if str(e).startswith("YouTube is blocking"):
            # Pass through our custom error message
            raise
        error_message = f"Unexpected error downloading videos: {str(e)}"
        print(error_message)
        raise Exception(error_message)
    finally:
        # Clean up
        if os.path.exists(links_file):
            os.remove(links_file)

def transcribe_audio():
    global transcription_status
    
    try:
        # Import faster-whisper here to avoid import errors if not installed
        from faster_whisper import WhisperModel
        
        # Reset and initialize transcription status
        transcription_status.update({
            "is_running": True,
            "current_file": None,
            "progress": 0,
            "completed_files": 0,
            "error": None,
            "last_updated": datetime.now().isoformat()
        })
        
        # Get all MP3 files
        mp3_files = [f for f in os.listdir(CLIPS_DIR) if f.endswith(".mp3")]
        total_files = len(mp3_files)
        
        if total_files == 0:
            transcription_status.update({
                "is_running": False,
                "error": "No MP3 files found in clips directory",
                "last_updated": datetime.now().isoformat()
            })
            print("No MP3 files found for transcription")
            return
        
        transcription_status["total_files"] = total_files
        print(f"Starting transcription of {total_files} files")
        
        # Initialize the Whisper model
        model = WhisperModel("base", device="cpu", compute_type="int8")
        
        # Transcribe each file
        for i, file in enumerate(mp3_files):
            try:
                # Update current file status
                transcription_status.update({
                    "current_file": file,
                    "progress": int((i / total_files) * 100),
                    "last_updated": datetime.now().isoformat()
                })
                
                print(f"Transcribing file {i+1}/{total_files}: {file}")
                file_path = os.path.join(CLIPS_DIR, file)
                
                # Transcribe using faster-whisper Python API
                segments, info = model.transcribe(file_path, beam_size=5)
                
                # Collect all transcribed text
                transcribed_text = ""
                for segment in segments:
                    transcribed_text += segment.text + " "
                
                # Save transcription to text file
                base_name = os.path.splitext(file)[0]
                transcript_file = os.path.join(TRANSCRIPTS_DIR, f"{base_name}.txt")
                
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(transcribed_text.strip())
                
                # Update completed files count
                transcription_status.update({
                    "completed_files": i + 1,
                    "progress": int(((i + 1) / total_files) * 100),
                    "last_updated": datetime.now().isoformat()
                })
                
                print(f"Successfully transcribed: {file}")
                
            except Exception as e:
                error_msg = f"Unexpected error transcribing {file}: {str(e)}"
                print(error_msg)
                transcription_status.update({
                    "error": error_msg,
                    "last_updated": datetime.now().isoformat()
                })
                continue
        
        # Mark transcription as complete
        transcription_status.update({
            "is_running": False,
            "current_file": None,
            "progress": 100,
            "last_updated": datetime.now().isoformat()
        })
        
        print(f"Transcription completed. {transcription_status['completed_files']}/{total_files} files processed successfully")
        
    except ImportError as e:
        error_msg = f"faster-whisper not properly installed: {str(e)}"
        print(error_msg)
        transcription_status.update({
            "is_running": False,
            "error": error_msg,
            "last_updated": datetime.now().isoformat()
        })
    except Exception as e:
        error_msg = f"Fatal error in transcription process: {str(e)}"
        print(error_msg)
        transcription_status.update({
            "is_running": False,
            "error": error_msg,
            "last_updated": datetime.now().isoformat()
        })

def generate_json():
    global json_generation_status
    
    try:
        # Initialize status
        json_generation_status.update({
            "is_running": True,
            "current_step": "Initializing",
            "progress": 0,
            "total_files": 0,
            "processed_files": 0,
            "error": None,
            "last_updated": datetime.now().isoformat()
        })
        
        # Count transcript files
        transcript_files = [f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".txt")]
        total_files = len(transcript_files)
        
        if total_files == 0:
            raise Exception("No transcript files found to process")
        
        json_generation_status.update({
            "total_files": total_files,
            "current_step": "Processing transcripts",
            "progress": 10,
            "last_updated": datetime.now().isoformat()
        })
        
        # Process files manually for better progress tracking
        results = []
        for i, file in enumerate(transcript_files):
            json_generation_status.update({
                "current_step": f"Processing {file}",
                "processed_files": i,
                "progress": 10 + int((i / total_files) * 80),  # 10-90% for processing
                "last_updated": datetime.now().isoformat()
            })
            
            try:
                file_path = os.path.join(TRANSCRIPTS_DIR, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    
                # Only include files with substantial content
                if len(content.split()) > 50:
                    results.append({
                        "instruction": "Summarize and offer Islamic advice based on this:",
                        "input": content,
                        "output": "Remember, Allah is always with those who are patient and sincere."
                    })
                    
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        # Update progress for saving
        json_generation_status.update({
            "current_step": "Saving JSON file",
            "processed_files": total_files,
            "progress": 90,
            "last_updated": datetime.now().isoformat()
        })
        
        # Save the JSON file
        json_file_path = os.path.join(JSON_DIR, "akhi_lora.json")
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Mark as complete
        json_generation_status.update({
            "is_running": False,
            "current_step": "Complete",
            "progress": 100,
            "last_updated": datetime.now().isoformat()
        })
        
        print(f"JSON generation completed. Generated {len(results)} entries from {total_files} transcript files.")
        
    except Exception as e:
        error_msg = f"Error generating JSON: {str(e)}"
        print(error_msg)
        json_generation_status.update({
            "is_running": False,
            "error": error_msg,
            "last_updated": datetime.now().isoformat()
        })

# API Endpoints
@app.get("/")
def read_root():
    return {"msg": "Akhi Backend API Ready"}

@app.post("/api/videos/download")
async def download_videos_endpoint(request_data: dict = Body(...)):
    try:
        print(f"Raw request data: {request_data}")
        
        # Handle both formats: direct VideoLinks model or dict with links key
        if isinstance(request_data, dict) and 'links' in request_data:
            # Extract links from the request data
            links_data = request_data['links']
            links = [link['url'] for link in links_data if 'url' in link]
        else:
            # Fallback to the old format
            video_links = VideoLinks.parse_obj(request_data)
            links = [link.url for link in video_links.links]
            
        print(f"Extracted links: {links}")
        
        if not links:
            raise ValueError("No valid links provided")
            
        # Run synchronously for better error handling
        result = download_videos(links)
        return {"message": result}
    except Exception as e:
        # Return error with status code 500
        error_msg = f"Error in download_videos_endpoint: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/transcribe")
async def transcribe_videos(background_tasks: BackgroundTasks):
    background_tasks.add_task(transcribe_audio)
    return {"msg": "Started transcription process"}

@app.post("/api/generate-json")
async def create_json(background_tasks: BackgroundTasks):
    background_tasks.add_task(generate_json)
    return {"msg": "Started JSON generation"}

@app.get("/api/status")
def get_status():
    # Count files in each directory
    clips_count = len([f for f in os.listdir(CLIPS_DIR) if f.endswith(".mp3")])
    transcripts_count = len([f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".txt")])
    
    # Check if JSON exists
    json_file = os.path.join(JSON_DIR, "akhi_lora.json")
    json_exists = os.path.exists(json_file)
    json_count = 0
    
    if json_exists:
        with open(json_file, "r") as f:
            try:
                data = json.load(f)
                json_count = len(data)
            except json.JSONDecodeError:
                json_exists = False
    
    return {
        "clips": clips_count,
        "transcripts": transcripts_count,
        "json_exists": json_exists,
        "json_count": json_count,
        "transcription_status": transcription_status,
        "json_generation_status": json_generation_status
    }

@app.get("/api/transcription/status")
def get_transcription_status():
    """Get detailed transcription progress and status"""
    return transcription_status

@app.get("/api/json-generation/status")
def get_json_generation_status():
    """Get detailed JSON generation progress and status"""
    return json_generation_status

@app.get("/api/transcripts")
def list_transcripts():
    transcripts = []
    for file in os.listdir(TRANSCRIPTS_DIR):
        if file.endswith(".txt"):
            file_path = os.path.join(TRANSCRIPTS_DIR, file)
            with open(file_path, "r") as f:
                content = f.read()
                word_count = len(content.split())
                transcripts.append({
                    "file_name": file,
                    "word_count": word_count,
                    "preview": content[:200] + "..." if len(content) > 200 else content
                })
    
    return {"transcripts": transcripts}

@app.get("/api/transcripts/{file_name}")
def get_transcript(file_name: str):
    file_path = os.path.join(TRANSCRIPTS_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    with open(file_path, "r") as f:
        content = f.read()
    
    return {"file_name": file_name, "content": content}

@app.post("/api/transcripts/{file_name}")
def update_transcript(file_name: str, transcript: TranscriptEdit):
    file_path = os.path.join(TRANSCRIPTS_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    with open(file_path, "w") as f:
        f.write(transcript.content)
    
    return {"msg": "Transcript updated successfully"}

@app.get("/api/json")
def get_json():
    json_file = os.path.join(JSON_DIR, "akhi_lora.json")
    if not os.path.exists(json_file):
        raise HTTPException(status_code=404, detail="JSON file not found")
    
    with open(json_file, "r") as f:
        data = json.load(f)
    
    return {"data": data}

@app.delete("/api/reset")
def reset_data():
    # Clear all directories
    for dir_path in [CLIPS_DIR, TRANSCRIPTS_DIR, JSON_DIR]:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    return {"msg": "All data has been reset"}
