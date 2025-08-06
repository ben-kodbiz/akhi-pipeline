"""Transcription Tool for CrewAI

This tool provides audio transcription capabilities for the CrewAI agentic system.
It integrates with the existing transcriber logic from the pipeline and supports
multiple Whisper model sizes, device selection, and advanced features.
"""

import os
import sys
import yaml
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from datetime import datetime

# Add the pipeline directory to the path to import existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../pipeline'))

# Import unified config loader
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.config_loader import get_config

try:
    from agents.transcriber import Transcriber
except ImportError:
    # Fallback implementation if pipeline module is not available
    Transcriber = None


class TranscriptionInput(BaseModel):
    """Input schema for transcription tool."""
    audio_file_path: str = Field(..., description="Path to the audio file to transcribe")
    model_size: str = Field(default="base", description="Whisper model size: tiny, base, small, medium, large")
    device: str = Field(default="auto", description="Device to run on: auto, cpu, cuda")
    language: str = Field(default="auto", description="Language code or 'auto' for detection")
    include_timestamps: bool = Field(default=True, description="Include timestamps in output")
    word_timestamps: bool = Field(default=False, description="Include word-level timestamps")
    output_format: str = Field(default="both", description="Output format: text, json, or both")
    output_dir: Optional[str] = Field(default=None, description="Custom output directory")


class TranscriptionTool(BaseTool):
    """Transcription Tool for converting audio to text using Whisper models."""
    
    name: str = "Transcription Tool"
    description: str = (
        "Transcribe audio files to text using OpenAI Whisper models. "
        "Supports multiple model sizes, device selection (CPU/GPU), language detection, "
        "timestamp generation, and various output formats. Optimized for Islamic educational content."
    )
    args_schema: type[BaseModel] = TranscriptionInput
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Transcription Tool.
        
        Args:
            config_path: Path to the configuration file
        """
        super().__init__()
        self._config = self._load_config(config_path)
        self._transcriber = None
        self._validate_dependencies()
    
    @property
    def config(self):
        """Get the configuration."""
        return self._config
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from unified config.yaml.
        
        Args:
            config_path: Path to configuration file (ignored, using unified config)
            
        Returns:
            Configuration dictionary
        """
        try:
            config_loader = get_config()
            return {
                'transcription': config_loader.get_transcription_config()
            }
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            # Return default configuration if file not found
            return {
                'transcription': {
                    'model_name': 'openai/whisper-large-v3',
                    'model_size': 'base',
                    'device': 'auto',
                    'language': 'auto',
                    'task': 'transcribe',
                    'batch_size': 1,
                    'chunk_length': 30,
                    'return_timestamps': True,
                    'output_dir': 'data/transcripts',
                    'output_format': 'json',
                    'include_timestamps': True,
                    'word_timestamps': False
                }
            }
    
    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are available."""
        try:
            import faster_whisper
        except ImportError:
            raise ImportError(
                "faster-whisper is required for transcription. "
                "Install it with: pip install faster-whisper"
            )
    
    def _validate_audio_file(self, file_path: str) -> bool:
        """Validate that the audio file exists and is accessible.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if file is valid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Check file extension
        valid_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in valid_extensions:
            raise ValueError(
                f"Unsupported audio format: {file_ext}. "
                f"Supported formats: {', '.join(valid_extensions)}"
            )
        
        return True
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use for transcription.
        
        Args:
            device: Device preference ('auto', 'cpu', 'cuda')
            
        Returns:
            Device string to use
        """
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device
    
    def _get_output_directory(self, custom_dir: Optional[str] = None) -> str:
        """Get the output directory for transcripts.
        
        Args:
            custom_dir: Custom output directory
            
        Returns:
            Output directory path
        """
        if custom_dir:
            output_dir = custom_dir
        else:
            output_dir = self._config.get('transcription', {}).get('output_dir', 'data/transcripts')
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _transcribe_single_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe a single audio file.
        
        Args:
            file_path: Path to the audio file
            **kwargs: Transcription parameters
            
        Returns:
            Transcription results
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster-whisper not available")
        
        # Extract parameters
        model_size = kwargs.get('model_size', 'base')
        device = self._determine_device(kwargs.get('device', 'auto'))
        language = kwargs.get('language', 'auto')
        include_timestamps = kwargs.get('include_timestamps', True)
        word_timestamps = kwargs.get('word_timestamps', False)
        
        # Initialize Whisper model with local cache directory
        # Set up local model cache to avoid downloading from Hugging Face
        cache_dir = os.path.join(os.path.dirname(__file__), '../../models/whisper_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Force offline mode to prevent any network connections
        import os as env_os
        env_os.environ['HF_HUB_OFFLINE'] = '1'
        env_os.environ['TRANSFORMERS_OFFLINE'] = '1'
        env_os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        try:
            model = WhisperModel(
                model_size, 
                device=device, 
                compute_type="int8" if device == "cpu" else "float16",
                download_root=cache_dir,  # Cache models locally to minimize future downloads
                local_files_only=True  # Force using only local files
            )
        except Exception as e:
            # If local_files_only fails, try without it but with offline env vars
            print(f"Warning: local_files_only failed, trying without it: {e}")
            model = WhisperModel(
                model_size, 
                device=device, 
                compute_type="int8" if device == "cpu" else "float16",
                download_root=cache_dir
            )
        
        # Transcription parameters
        transcribe_params = {
            'beam_size': 5,
            'word_timestamps': word_timestamps,
            'vad_filter': True,
            'vad_parameters': dict(min_silence_duration_ms=500)
        }
        
        if language != 'auto':
            transcribe_params['language'] = language
        
        # Perform transcription
        segments, info = model.transcribe(file_path, **transcribe_params)
        
        # Process segments
        processed_segments = []
        transcript_text = ""
        
        for segment in segments:
            segment_data = {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end
            }
            
            if word_timestamps and segment.words:
                segment_data["words"] = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    }
                    for word in segment.words
                ]
            
            processed_segments.append(segment_data)
            
            # Build text transcript
            if include_timestamps:
                timestamp = self._format_timestamp(segment.start)
                transcript_text += f"[{timestamp}] {segment.text}\n\n"
            else:
                transcript_text += f"{segment.text} "
        
        return {
            "transcript_text": transcript_text.strip(),
            "segments": processed_segments,
            "metadata": {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": processed_segments[-1]["end"] if processed_segments else 0,
                "model_size": model_size,
                "device": device
            }
        }
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS or HH:MM:SS string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        else:
            return f"{int(minutes):02d}:{int(seconds):02d}"
    
    def _save_transcription(self, result: Dict[str, Any], file_path: str, 
                          output_format: str, output_dir: str) -> Dict[str, str]:
        """Save transcription results to files.
        
        Args:
            result: Transcription results
            file_path: Original audio file path
            output_format: Output format ('text', 'json', 'both')
            output_dir: Output directory
            
        Returns:
            Dictionary with saved file paths
        """
        base_name = Path(file_path).stem
        saved_files = {}
        
        # Save text transcript
        if output_format in ['text', 'both']:
            text_file = os.path.join(output_dir, f"{base_name}.txt")
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(result['transcript_text'])
            saved_files['text'] = text_file
        
        # Save JSON transcript
        if output_format in ['json', 'both']:
            json_file = os.path.join(output_dir, f"{base_name}.json")
            json_data = {
                "metadata": result['metadata'],
                "segments": result['segments'],
                "transcript": result['transcript_text']
            }
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            saved_files['json'] = json_file
        
        return saved_files
    
    async def transcribe_audio(self, audio_path: str, output_dir: str, model_size: str = 'base', language: str = 'auto', device: str = 'auto') -> Dict[str, Any]:
        """Transcribe audio file and return structured result for pipeline integration.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Output directory for transcripts
            model_size: Whisper model size
            language: Language code or 'auto'
            
        Returns:
            Dictionary with transcription results
        """
        try:
            # Validate audio file
            self._validate_audio_file(audio_path)
            
            # Perform transcription
            result = self._transcribe_single_file(
                audio_path,
                model_size=model_size,
                device=device,
                language=language,
                include_timestamps=True,
                word_timestamps=False
            )
            
            # Save results
            saved_files = self._save_transcription(
                result, audio_path, 'both', output_dir
            )
            
            return {
                'success': True,
                'transcript_path': saved_files.get('text'),
                'transcript_json': saved_files.get('json'),
                'transcript_text': result['transcript_text'],
                'duration': result['metadata']['duration'],
                'language': result['metadata']['language'],
                'confidence': result['metadata']['language_probability']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _run(self, **kwargs) -> str:
        """Execute the transcription tool.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            Formatted transcription results
        """
        try:
            # Extract and validate inputs
            audio_file_path = kwargs.get('audio_file_path')
            if not audio_file_path:
                return "❌ **Error**: audio_file_path is required"
            
            # Validate audio file
            self._validate_audio_file(audio_file_path)
            
            # Get parameters with defaults from config
            transcription_config = self._config.get('transcription', {})
            
            model_size = kwargs.get('model_size', transcription_config.get('model_size', 'base'))
            device = kwargs.get('device', transcription_config.get('device', 'auto'))
            language = kwargs.get('language', transcription_config.get('language', 'auto'))
            include_timestamps = kwargs.get('include_timestamps', 
                                          transcription_config.get('include_timestamps', True))
            word_timestamps = kwargs.get('word_timestamps', 
                                       transcription_config.get('word_timestamps', False))
            output_format = kwargs.get('output_format', 'both')
            output_dir = self._get_output_directory(kwargs.get('output_dir'))
            
            # Perform transcription
            result = self._transcribe_single_file(
                audio_file_path,
                model_size=model_size,
                device=device,
                language=language,
                include_timestamps=include_timestamps,
                word_timestamps=word_timestamps
            )
            
            # Save results
            saved_files = self._save_transcription(
                result, audio_file_path, output_format, output_dir
            )
            
            # Format output
            file_name = Path(audio_file_path).name
            duration = result['metadata']['duration']
            language_detected = result['metadata']['language']
            confidence = result['metadata']['language_probability']
            
            output = f"✅ **Transcription Successful**\n\n"
            output += f"**File:** {file_name}\n"
            output += f"**Duration:** {self._format_timestamp(duration)}\n"
            output += f"**Language:** {language_detected} (confidence: {confidence:.2f})\n"
            output += f"**Model:** {model_size} on {result['metadata']['device']}\n\n"
            
            output += f"**Saved Files:**\n"
            for format_type, file_path in saved_files.items():
                output += f"- {format_type.upper()}: {file_path}\n"
            
            output += f"\n**Preview:**\n"
            preview_text = result['transcript_text'][:300]
            if len(result['transcript_text']) > 300:
                preview_text += "..."
            output += f"{preview_text}"
            
            return output
            
        except FileNotFoundError as e:
            return f"❌ **File Error**: {str(e)}"
        except ValueError as e:
            return f"❌ **Validation Error**: {str(e)}"
        except ImportError as e:
            return f"❌ **Dependency Error**: {str(e)}"
        except Exception as e:
            return f"❌ **Transcription Error**: {str(e)}"


# Example usage
if __name__ == "__main__":
    # Initialize the tool
    tool = TranscriptionTool()
    
    # Example transcription
    test_file = "path/to/audio/file.mp3"
    if os.path.exists(test_file):
        result = tool._run(
            audio_file_path=test_file,
            model_size="base",
            device="auto",
            include_timestamps=True
        )
        print(result)
    else:
        print("Test audio file not found. Tool is ready for use.")