# Agentic YouTube to QLoRA Data Pipeline

An intelligent, CrewAI-powered agentic pipeline for converting YouTube videos from Islamic scholars into high-quality QLoRA training data. The system uses specialized AI agents for discovery, processing, and formatting.

## Features

### Core Capabilities
- **Intelligent Discovery**: AI agents find relevant Islamic content from trusted scholars
- **Agentic Processing**: CrewAI agents handle transcription and formatting
- **State Management**: Persistent tracking with `status_tracker.json`
- **Error Recovery**: Robust error handling with automatic retry logic
- **Resumable Operations**: Continue processing from interruption points
- **Real-time Monitoring**: Live progress tracking and status updates

### Advanced Features
- **Multi-format Output**: Segments, dialogues, and summaries for training
- **Speaker Detection**: Identify and preserve speaker information
- **Content Filtering**: Quality control and Islamic content validation
- **Metadata Enrichment**: Enhanced context and categorization
- **Performance Optimization**: Efficient processing with resource management

## Quick Start

### 1. Configuration

Copy and customize the configuration file:
```bash
cp config.yaml my_config.yaml
# Edit my_config.yaml with your preferred settings
```

### 2. Run the Agentic Pipeline

**Full Agentic Pipeline:**
```bash
python run_pipeline.py
```

**Legacy CLI Mode:**
```bash
# Run with shell script
./run_pipeline.sh

# Individual steps (legacy)
python run_pipeline.py --search
python run_pipeline.py --download
python run_pipeline.py --transcribe
python run_pipeline.py --format
```

**Web Interface Integration:**
```bash
# Start backend API for web interface
cd ../akhi_crewai
python api.py

# Access via web interface at http://localhost:3000
```

### 3. Monitor Progress

```bash
# Check pipeline status
python run_pipeline.py --status

# View detailed summary
python run_pipeline.py --summary

# See failed videos
python run_pipeline.py --failed

# Reset pipeline state
python run_pipeline.py --reset
```

## Configuration

The pipeline uses a YAML configuration file (`config.yaml`) with the following sections:

- **search**: Video search parameters (terms, duration filters, quality)
- **download**: Download settings (batch size, format, timeouts)
- **transcription**: Whisper model settings (size, device, language)
- **qlora**: QLoRA formatting options (segment length, filtering)
- **logging**: Log levels and file management
- **pipeline**: Overall behavior (error handling, performance)

## Agentic State Management

The pipeline tracks video processing status in `db/status_tracker.json`:

### Processing States
- **discovered**: Found by YouTube Discovery Agent
- **downloaded**: Successfully downloaded by YouTube Agent
- **transcribed**: Processed by Transcriber Agent
- **json_ready**: Formatted by QLoRA Formatter Agent

### Agent Coordination
- **YouTube Agent**: Discovers and downloads Islamic lecture videos
- **Transcriber Agent**: Generates enhanced transcripts with speaker detection
- **QLoRA Formatter Agent**: Creates multiple training data formats
- **Pipeline Orchestrator**: Coordinates agent workflows and state management

### Status Tracking Structure
```json
{
  "videos": {
    "video_id": {
      "title": "Lecture Title",
      "status": {
        "discovered": true,
        "downloaded": true,
        "transcribed": true,
        "json_ready": true
      },
      "file_paths": {
        "audio": "output/clips/video_id.mp3",
        "transcript": "output/transcripts/video_id.txt",
        "enhanced_transcript": "output/transcripts/video_id.json"
      }
    }
  },
  "stats": {
    "total_discovered": 50,
    "total_downloaded": 30,
    "total_transcribed": 25,
    "total_json_ready": 20
  }
}

## Directory Structure

```
pipeline/
├── agents/
│   ├── youtube_agent.py      # YouTube Discovery & Download Agent
│   ├── transcriber.py        # Transcriber Agent (faster-whisper)
│   └── qlora_formatter.py    # QLoRA Formatter Agent
├── config_manager.py         # Configuration management
├── state_manager.py          # Agentic state tracking
├── run_pipeline.py           # Agentic pipeline orchestrator
├── run_pipeline.sh           # Legacy CLI script
├── config.yaml               # Pipeline configuration
├── video_links.txt           # Input video URLs
└── README.md                 # This file

output/
├── clips/                    # Downloaded MP3 files
├── transcripts/              # Enhanced transcripts (txt + json)
└── json/                     # QLoRA training data

db/
├── status_tracker.json       # Agentic state management
└── pipeline_log.txt          # Comprehensive logging

scripts/
└── make_quran_lora_json.py   # Quran data preparation
```

## Error Handling

The pipeline includes comprehensive error handling:

- **Retry Logic**: Configurable retry attempts for downloads
- **Timeout Protection**: Prevents hanging on long operations
- **State Persistence**: Resume from failures without losing progress
- **Detailed Logging**: Track errors and performance metrics
- **Graceful Degradation**: Continue processing other videos when one fails

## Advanced Usage

### Custom Configuration

```bash
python run_pipeline.py --config my_config.yaml --all
```

### Batch Processing

```bash
# Process in smaller batches
python run_pipeline.py --download --max-videos 5
python run_pipeline.py --transcribe --max-files 3
```

### Development and Testing

```bash
# Test with minimal settings
python run_pipeline.py --search --search-terms "test query" --max-videos 1
```

## Troubleshooting

### Common Issues

1. **No videos found**: Check search terms and duration filters in config
2. **Download failures**: Verify internet connection and yt-dlp installation
3. **Transcription errors**: Ensure Whisper dependencies are installed
4. **Permission errors**: Check write permissions for data directories

### Logs

Check the logs directory for detailed error information:
```bash
tail -f pipeline/logs/pipeline_*.log
```

### Reset State

If you encounter persistent issues:
```bash
python run_pipeline.py --reset
```

## Dependencies

### Core Requirements
- Python 3.8+
- CrewAI framework
- yt-dlp (YouTube downloading)
- faster-whisper (transcription)
- PyYAML (configuration)

### Agent Dependencies
- langchain (agent framework)
- openai (LLM integration)
- sentence-transformers (embeddings)
- faiss-cpu (vector search)

## Integration Points

### Web Interface
- FastAPI backend at `../akhi_crewai/api.py`
- React frontend at `../frontend_web/`
- Real-time status polling and progress tracking

### Desktop Application
- Tauri-based app at `../desktop_app/`
- Direct pipeline integration via command API
- Native file system access

### QLoRA Training
- Axolotl integration in `../akhi_crewai/axolotl_data/`
- Direct training data consumption
- Model fine-tuning workflows

## Contributing

The agentic pipeline is designed for extensibility:

- **New Agents**: Add specialized processing agents
- **Enhanced Discovery**: Improve content discovery algorithms
- **Quality Control**: Add content validation agents
- **Performance**: Optimize agent coordination and resource usage
- **Integration**: Expand API and interface capabilities