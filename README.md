# Akhi Data Builder

A comprehensive Islamic dataset builder for LLM training that processes YouTube lectures into clean, structured training data using a semi-automated agentic pipeline.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              AKHI DATA BUILDER                                 │
│                         Islamic Dataset Pipeline System                        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend Web  │    │  Desktop App    │    │   Backend API   │
│   (React + UI)  │    │  (Tauri + Rust) │    │  (FastAPI + Py) │
│                 │    │                 │    │                 │
│ • Dashboard     │◄──►│ • Native UI     │◄──►│ • REST API      │
│ • Progress UI   │    │ • File Access   │    │ • Status Track  │
│ • Transcript    │    │ • CLI Commands  │    │ • Background    │
│   Editor        │    │ • Cross-platform│    │   Tasks         │
│ • JSON Preview  │    │                 │    │ • CORS Enabled  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE CORE                                     │
│                          (Agentic Python Pipeline)                             │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   STEP 1:       │    │   STEP 2:       │    │   STEP 3:       │    │   STEP 4:       │
│   DISCOVER      │───►│   DOWNLOAD      │───►│   TRANSCRIBE    │───►│   GENERATE JSON │
│                 │    │                 │    │                 │    │                 │
│ • YouTube Agent │    │ • yt-dlp        │    │ • faster-whisper│    │ • QLoRA Format │
│ • API Search    │    │ • YouTube URLs  │    │ • Audio → Text  │    │ • Content Filter│
│ • Scholar Filter│    │ • MP3 Extract   │    │ • Speaker Det.  │    │ • Training Data │
│ • Duration Filt.│    │ • Batch Process │    │ • Word Timestamp│    │ • JSON Export   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │                        │
        ▼                        ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ pipeline/db/    │    │ output/clips/   │    │output/transcripts│    │ output/json/    │
│                 │    │                 │    │                 │    │                 │
│ • status_tracker│    │ • *.mp3 files   │    │ • *.txt files   │    │ • akhi_lora.json│
│ • pipeline_log  │    │ • Audio clips   │    │ • *.json files  │    │ • Training data │
│ • State tracking│    │ • Extracted     │    │ • Timestamped   │    │ • Instruction   │
│ • Progress data │    │   from YouTube  │    │ • Speaker labels│    │   format        │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 System Components

### 1. Frontend Web Application
- **Technology**: React 18 + Tailwind CSS
- **Port**: http://localhost:3000
- **Features**:
  - Interactive dashboard with pipeline controls
  - Real-time progress indicators for transcription and JSON generation
  - YouTube URL submission interface
  - Transcript viewing and editing capabilities
  - JSON preview and export functionality
  - Error handling and status monitoring

### 2. Desktop Application
- **Technology**: Tauri (Rust + React)
- **Features**:
  - Native desktop experience
  - Direct CLI tool integration
  - Cross-platform compatibility
  - File system access
  - Offline functionality

### 3. Backend API
- **Technology**: FastAPI + Python
- **Port**: http://localhost:8001
- **Endpoints**:
  - `POST /api/videos/download` - Download YouTube videos
  - `POST /api/transcribe` - Start transcription process
  - `POST /api/generate-json` - Generate training JSON
  - `GET /api/status` - Get overall pipeline status
  - `GET /api/transcription/status` - Detailed transcription progress
  - `GET /api/json-generation/status` - Detailed JSON generation progress
  - `GET /api/transcripts/{file_name}` - Get specific transcript
  - `DELETE /api/reset` - Reset all data

### 4. Agentic Pipeline Core
- **Technology**: Python + Specialized Agents
- **Components**:
  - **YouTube Agent**: Discovers and filters Islamic scholar videos
  - **Transcriber Agent**: Enhanced audio transcription with speaker detection
  - **QLoRA Formatter Agent**: Advanced JSON formatting for AI training
  - **State Tracking**: Persistent pipeline state management

### 4. Pipeline Tools
- **Technology**: Python + External Libraries
- **Components**:
  - **yt-dlp**: YouTube video downloading and discovery
  - **faster-whisper**: Audio transcription with word-level timestamps
  - **JSON Processing**: Structured data generation for model training

## 📊 Data Flow

```
YouTube Agent ──► Discover Videos ──► Download (yt-dlp) ──► MP3 Files ──► Transcribe (Whisper) ──► Enhanced Transcripts ──► QLoRA Format ──► Training JSON
      │                 │                  │                   │                │                      │                    │                │
      └─────────────────┴──────────────────┴───────────────────┴────────────────┴──────────────────────┴────────────────────┴────────────────┘
                                                                     │
                                                                     ▼
                                                            Status Tracking System
                                                            (Persistent State Management)
```

## 🤖 Agentic Pipeline Components

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

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 14+
- FFmpeg
- Git
- No API keys required (uses yt-dlp for discovery)
- CUDA-compatible GPU (recommended for faster transcription)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd akhi_data_builder
```bash
# Install Python dependencies
pip install faster-whisper langdetect fastapi uvicorn yt-dlp tqdm

# Install frontend dependencies
cd frontend_web
npm install
cd ..

# Install desktop app dependencies (optional)
cd desktop_app
npm install
cd ..
```

### Running the Application

1. **Start the Backend API**:
```bash
cd backend/api
uvicorn main:app --reload --port 8001
```

2. **Start the Frontend Web App**:
```bash
cd frontend_web
npm start
```

3. **Access the Application**:
   - Web Interface: http://localhost:3000
   - API Documentation: http://localhost:8001/docs

### Using the Pipeline

#### Web Interface
1. **Add YouTube URLs**: Enter YouTube lecture URLs in the dashboard
2. **Download Videos**: Click "Download" to extract audio as MP3 files
3. **Transcribe Audio**: Click "Transcribe" to convert audio to text
4. **Generate JSON**: Click "Generate JSON" to create training data
5. **Review & Export**: View transcripts and download the final JSON

#### Command Line (Agentic Pipeline)

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

## 📁 Project Structure

```
akhi_data_builder/
├── backend/
│   └── api/
│       └── main.py              # FastAPI backend server
├── frontend_web/
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   ├── pages/              # Page components
│   │   └── services/           # API services
│   └── package.json
├── desktop_app/
│   ├── src/                    # React frontend
│   ├── src-tauri/             # Rust backend
│   └── package.json
├── pipeline/
│   ├── agents/
│   │   ├── youtube_agent.py    # YouTube discovery agent
│   │   ├── transcriber.py      # Audio transcription agent
│   │   └── qlora_formatter.py  # JSON formatting agent
│   ├── db/
│   │   └── status_tracker.json # Pipeline state tracking
│   ├── output/
│   │   ├── clips/              # Downloaded MP3 files
│   │   ├── transcripts/        # Generated text & JSON transcripts
│   │   └── json/               # Training data JSON
│   ├── scripts/
│   │   └── make_quran_lora_json.py
│   ├── run_pipeline.py         # Agentic pipeline orchestrator
│   └── run_pipeline.sh         # Legacy CLI pipeline script
└── README.md
```

## 🔄 Pipeline Stages

### Stage 1: Video Discovery
- **Agent**: YouTube Agent (`youtube_agent.py`)
- **Input**: Search terms, filters
- **Output**: Curated list of Islamic lecture videos
- **Location**: `pipeline/db/status_tracker.json`
- **Features**: yt-dlp integration, duration filtering, status tracking

### Stage 2: Video Download
- **Tool**: yt-dlp (integrated with YouTube Agent)
- **Input**: Discovered video URLs
- **Output**: MP3 audio files
- **Location**: `pipeline/output/clips/`
- **Features**: Batch processing, error handling, progress tracking

### Stage 3: Audio Transcription
- **Agent**: Transcriber Agent (`transcriber.py`)
- **Tool**: faster-whisper
- **Input**: MP3 files
- **Output**: Enhanced transcripts (plain text and JSON)
- **Location**: `pipeline/output/transcripts/`
- **Features**: Word-level timestamps, speaker detection, dual output formats

### Stage 4: JSON Generation
- **Agent**: QLoRA Formatter Agent (`qlora_formatter.py`)
- **Input**: Enhanced transcripts
- **Output**: LoRA training JSON
- **Location**: `pipeline/output/json/akhi_lora.json`
- **Features**: Multiple training formats (segments, dialogues, summaries), content filtering, metadata enrichment

## 🎯 Training Data Format

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

## 🔧 Configuration

### Backend Configuration
- **API Port**: 8001 (configurable in uvicorn command)
- **CORS**: Enabled for all origins (development mode)
- **File Paths**: Automatically configured relative to project root

### Frontend Configuration
- **API URL**: `http://localhost:8001` (configurable in `src/services/api.js`)
- **Polling Interval**: 2 seconds for status updates
- **UI Theme**: Tailwind CSS with blue primary colors

## 🚨 Error Handling

### Web Interface
- **Download Errors**: Invalid URLs, network issues, yt-dlp failures
- **Transcription Errors**: Audio format issues, whisper model problems
- **JSON Generation Errors**: File access issues, content filtering problems
- **API Errors**: Server connectivity, endpoint failures
- **UI Errors**: Real-time error display with detailed messages

### Agentic Pipeline
- **Discovery Errors**: yt-dlp search failures, search term issues, network failures
- **Download Errors**: Video unavailability, format restrictions, network timeouts
- **Transcription Errors**: Audio quality issues, model compatibility, resource constraints
- **Formatting Errors**: Transcript parsing failures, content filtering edge cases
- **State Management**: Automatic recovery from interruptions, error logging

## 📈 Monitoring & Progress

### Web Interface
- **Real-time Progress Bars**: Visual indicators for all pipeline stages
- **File Counters**: Track processed vs. total files
- **Status Endpoints**: Detailed progress and error information

### Agentic Pipeline
- **Status Tracker**: Persistent JSON-based state tracking (`status_tracker.json`)
- **Command-line Progress**: Real-time updates during pipeline execution
- **Detailed Logging**: Comprehensive logging to `pipeline_log.txt`
- **Statistics**: Video counts by processing stage (discovered, downloaded, transcribed, JSON-ready)
- **Resumable Operations**: Continue from last successful state after interruptions
- **Background Processing**: Non-blocking operations with status polling

## 🔄 Status Tracking System

The agentic pipeline uses a persistent status tracking system to maintain state across runs:

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

### Key Features

- **Persistent State**: Maintains pipeline state between runs
- **Progress Tracking**: Records which videos have completed each stage
- **File Path Management**: Stores paths to all generated files
- **Statistics**: Maintains counts for monitoring and reporting
- **Resumable Operations**: Allows pipeline to continue from interruptions
- **Shared State**: Common data structure used by all agent components

## 🔮 Future Enhancements

### Agentic Pipeline Improvements
- [ ] **Advanced Speaker Identification**: ML-based speaker recognition
- [ ] **Topic Classification**: Automatic categorization of lecture content
- [ ] **Content Quality Scoring**: Ranking system for transcript quality
- [ ] **Parallel Processing**: Multi-threaded pipeline execution
- [ ] **Adaptive Segmentation**: Context-aware text segmentation
- [ ] **Enhanced Metadata**: Extraction of additional video context
- [ ] **Checkpoint System**: Finer-grained recovery points
- [ ] **Pipeline Analytics**: Performance metrics and optimization

### Integration Enhancements
- [ ] **Axolotl Integration**: Direct connection to LoRA training
- [ ] **Cloud Processing**: Support for distributed execution
- [ ] **API Expansion**: Comprehensive REST API for all pipeline functions
- [ ] **Multi-language Support**: Processing for various Islamic languages
- [ ] **Advanced Transcript Editor**: Web-based correction interface
- [ ] **Quality Assurance Tools**: Validation and verification modules
- [ ] **Automated Testing**: CI/CD pipeline for agent components
- [ ] **Containerization**: Docker support for easy deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Goal

Build an intelligent, agentic pipeline to discover, process, and format Islamic knowledge from YouTube lectures into high-quality training data for fine-tuning Islamic LLMs (like Akhi). The system aims to:

1. **Automate Discovery**: Intelligently find relevant Islamic content from trusted scholars
2. **Ensure Quality**: Process audio with state-of-the-art transcription and formatting
3. **Maximize Learning**: Generate diverse training examples from the same content
4. **Maintain Context**: Preserve speaker information and dialogue structure
5. **Scale Efficiently**: Process large volumes of content with minimal human intervention
6. **Support Research**: Provide a foundation for Islamic AI research and development

Ultimately, this project serves the broader goal of making authentic Islamic knowledge more accessible through AI while maintaining the highest standards of accuracy and fidelity to traditional Islamic scholarship.