# YouTube to QLoRA Data Pipeline

A robust, configurable pipeline for converting YouTube videos from Islamic scholars into QLoRA training data.

## Features

- **Configurable**: YAML-based configuration for all pipeline settings
- **State Management**: Persistent tracking of video processing status
- **Error Handling**: Robust error handling with retry logic and failure tracking
- **Modular**: Run individual steps or the full pipeline
- **Logging**: Comprehensive logging with configurable levels
- **Resume Support**: Continue processing from where you left off

## Quick Start

### 1. Configuration

Copy and customize the configuration file:
```bash
cp config.yaml my_config.yaml
# Edit my_config.yaml with your preferred settings
```

### 2. Run the Pipeline

**Full Pipeline:**
```bash
python run_pipeline.py --all
```

**Individual Steps:**
```bash
# Search for videos
python run_pipeline.py --search

# Download found videos
python run_pipeline.py --download

# Transcribe downloaded audio
python run_pipeline.py --transcribe

# Format transcripts for QLoRA
python run_pipeline.py --format
```

**With Custom Parameters:**
```bash
python run_pipeline.py --search --search-terms "nouman ali khan" "omar suleiman" --max-videos 20
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

## State Management

The pipeline tracks video processing status in `video_state.jsonl`:

- **DISCOVERED**: Found during search
- **DOWNLOADING**: Currently being downloaded
- **DOWNLOADED**: Successfully downloaded
- **TRANSCRIBING**: Currently being transcribed
- **TRANSCRIBED**: Successfully transcribed
- **FORMATTING**: Currently being formatted
- **COMPLETED**: Fully processed
- **FAILED**: Processing failed

## Directory Structure

```
pipeline/
├── agents/
│   ├── youtube_agent.py      # YouTube search and management
│   ├── transcriber.py        # Audio transcription
│   └── qlora_formatter.py    # QLoRA data formatting
├── config_manager.py         # Configuration management
├── state_manager.py          # Video state tracking
├── run_pipeline.py           # Main pipeline orchestrator
├── config.yaml               # Default configuration
└── README.md                 # This file

data/
├── clips/                    # Downloaded audio files
├── transcripts/              # Generated transcripts
├── json/                     # QLoRA-formatted data
└── db/                       # State and log files
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

- Python 3.8+
- yt-dlp
- whisper (OpenAI)
- PyYAML
- dataclasses (Python 3.7+)

## Contributing

The pipeline is designed to be modular and extensible. Key areas for enhancement:

- Additional video sources beyond YouTube
- Alternative transcription services
- Enhanced QLoRA formatting options
- Performance optimizations
- UI/web interface