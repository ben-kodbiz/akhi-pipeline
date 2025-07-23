# Task 2: Agentic Pipeline Implementation

## Original Feedback

Thanks for the context — that makes your project way more meaningful. You're not just building a pipeline; you're crafting a **data refinery for Islamic AI specialization**. With that in mind, here's a sharper review tailored to your real goal:

---

## 💡 What You're Actually Building

> A semi-automated agentic pipeline that:

1. **Discovers** Islamic scholar videos on YouTube
2. **Transcribes** them using Whisper
3. **Structures** them into QLoRA-compatible JSON (prompt + response format)
4. **Feeds** into model finetuning for your Islamic AI companion (Quran, Hadith, quotes, etc.)

---

## ✅ Implementation Complete

All suggested components have been successfully implemented:

### 1. YouTube Discovery Agent
- **File**: `pipeline/agents/youtube_agent.py`
- **Purpose**: Discovers and filters Islamic scholar videos from YouTube
- **Features**:
  - yt-dlp integration for video search and metadata extraction
  - Filtering by duration, language, and content relevance
  - Persistent tracking of discovered videos
  - Automatic generation of download lists

### 2. Transcriber Agent
- **File**: `pipeline/agents/transcriber.py`
- **Purpose**: Transcribes audio files with enhanced features
- **Features**:
  - Integration with faster-whisper for efficient transcription
  - Word-level timestamps for precise text alignment
  - Simple speaker detection heuristics
  - Dual output format (plain text and enhanced JSON)
  - Progress tracking and state management

### 3. QLoRA Formatter Agent
- **File**: `pipeline/agents/qlora_formatter.py`
- **Purpose**: Formats transcripts into AI training data
- **Features**:
  - Multiple training example formats (segments, dialogues, summaries)
  - Content filtering based on word count and relevance
  - Metadata enrichment for training examples
  - Structured JSON output for QLoRA fine-tuning

### 4. Pipeline Orchestrator
- **File**: `pipeline/run_pipeline.py`
- **Purpose**: Coordinates the entire pipeline workflow
- **Features**:
  - Command-line interface with configurable options
  - Step-by-step or full pipeline execution
  - Comprehensive logging and error handling
  - Integration of all agent components

## Status Tracking System

A key innovation in this implementation is the persistent status tracking system that maintains state across pipeline runs:

```json
{
  "videos": {
    "video_id_1": {
      "title": "Understanding Surah Al-Fatiha",
      "url": "https://www.youtube.com/watch?v=abc123",
      "duration": "00:45:30",
      "status": {
        "discovered": true,
        "downloaded": true,
        "transcribed": true,
        "json_ready": true
      },
      "file_paths": {
        "audio": "pipeline/output/clips/video_id_1.mp3",
        "transcript": "pipeline/output/transcripts/video_id_1.txt",
        "enhanced_transcript": "pipeline/output/transcripts/video_id_1.json"
      },
      "metadata": {
        "channel": "Islamic Scholar Channel",
        "published_at": "2023-01-15",
        "language": "english"
      }
    }
  },
  "last_search": "2023-06-01T12:34:56Z",
  "stats": {
    "total_discovered": 50,
    "total_downloaded": 30,
    "total_transcribed": 25,
    "total_json_ready": 20
  }
}
```

## Enhanced Training Data Formats

The QLoRA Formatter generates multiple training example formats in a single JSON file:

### 1. Individual Segments
```json
[
  {
    "instruction": "Transcribe and explain this segment from an Islamic lecture:",
    "input": "[Segment from Islamic lecture with speaker identification]",
    "output": "[Same segment text, formatted as a response]",
    "metadata": {
      "source": "youtube",
      "video_id": "abc123xyz",
      "segment_id": 1,
      "speaker": "Scholar",
      "word_count": 75
    }
  }
]
```

### 2. Dialogue Format
```json
[
  {
    "instruction": "Continue this Islamic dialogue based on the lecture context:",
    "input": "Scholar: [First part of lecture]\nHost: [Question or comment]\nScholar: [Beginning of response]",
    "output": "[Continuation of Scholar's response]",
    "metadata": {
      "source": "youtube",
      "video_id": "abc123xyz",
      "dialogue_id": 1,
      "speakers": ["Scholar", "Host"],
      "word_count": 120
    }
  }
]
```

### 3. Lecture Summary
```json
[
  {
    "instruction": "Summarize the key points from this Islamic lecture:",
    "input": "[First 2000 characters of lecture transcript]",
    "output": "The main points of this lecture are: 1) [Point one], 2) [Point two], 3) [Point three]...",
    "metadata": {
      "source": "youtube",
      "video_id": "abc123xyz",
      "title": "Understanding Surah Al-Fatiha",
      "word_count": 350
    }
  }
]
```

## Usage

### Command Line Interface

```bash
# Run the full pipeline with default settings
python pipeline/run_pipeline.py --all

# Run specific steps of the pipeline
python pipeline/run_pipeline.py --search --search_terms "Islamic lecture,Quran tafsir" --max_results 10
python pipeline/run_pipeline.py --download --max_videos 5
python pipeline/run_pipeline.py --transcribe --model_size medium --device cuda
python pipeline/run_pipeline.py --format --min_words 50 --max_words 500

# Get help on available options
python pipeline/run_pipeline.py --help
```

## Documentation Updates

The README.md file has been comprehensively updated to reflect the new agentic pipeline architecture, including:

- Updated data flow diagram
- Detailed component descriptions
- Enhanced project structure
- New pipeline stages
- Training data format examples
- Error handling and monitoring information
- Status tracking system documentation
- Future enhancement possibilities

## Conclusion

The implementation of the agentic pipeline significantly enhances the Akhi Data Builder project by adding intelligent automation, improved data quality, and a more robust architecture. The system now provides a comprehensive solution for discovering, processing, and formatting Islamic knowledge from YouTube lectures into high-quality training data for fine-tuning Islamic LLMs.

```bash
python3 run_pipeline.py --agent youtube_to_qlora --max-videos 10
```

Which will do:

1. Search for videos
2. Download
3. Transcribe
4. Format
5. Track all stages

---

