# Phase 6: Main Application & CLI - Implementation Guide

## Overview

Phase 6 implements the main application interface for the Akhi CrewAI Pipeline, providing both CLI and web-based access to the Islamic content processing system.

## Components Implemented

### 1. Command Line Interface (`main.py`)

The CLI provides direct access to all pipeline functionality:

```bash
# Search for videos
python main.py search "Islamic finance principles"

# Process videos with full pipeline
python main.py process "Quran recitation" --max-videos 5

# Ask questions about indexed content
python main.py query "What does the Quran say about charity?"

# Generate summaries
python main.py summarize --content-type recent --limit 10

# Check system status
python main.py status

# Interactive chat mode
python main.py interactive
```

### 2. Web API Interface (`api.py`)

RESTful API with WebSocket support:

```bash
# Start the web server
python api.py --host 0.0.0.0 --port 8000

# Or with auto-reload for development
python api.py --reload
```

**API Endpoints:**
- `GET /` - Web interface
- `GET /api` - API information
- `POST /api/search` - Search videos
- `POST /api/process` - Process pipeline
- `POST /api/query` - Ask questions
- `POST /api/summarize` - Generate summaries
- `GET /api/status` - System status
- `WebSocket /ws` - Real-time updates
- `GET /docs` - Swagger documentation

### 3. Web Interface (`static/index.html`)

Modern, responsive web interface with:
- Video search functionality
- Pipeline processing controls
- Question answering interface
- Real-time WebSocket updates
- System status monitoring

## Usage Examples

### CLI Usage

1. **Search for Islamic content:**
   ```bash
   python main.py search "Islamic history" --max-results 10
   ```

2. **Process content with full pipeline:**
   ```bash
   python main.py process "Hadith explanation" --max-videos 3 --full-process
   ```

3. **Interactive Q&A session:**
   ```bash
   python main.py interactive
   # Then ask questions interactively
   ```

### Web API Usage

1. **Start the server:**
   ```bash
   python api.py
   ```

2. **Access web interface:**
   - Open browser to `http://localhost:8000`
   - Use the interactive web interface

3. **API calls with curl:**
   ```bash
   # Search videos
   curl -X POST "http://localhost:8000/api/search" \
        -H "Content-Type: application/json" \
        -d '{"query": "Islamic finance", "max_results": 5}'
   
   # Ask a question
   curl -X POST "http://localhost:8000/api/query" \
        -H "Content-Type: application/json" \
        -d '{"question": "What is Zakat?", "index_name": "default"}'
   ```

### WebSocket Usage

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// Send a question
ws.send(JSON.stringify({
    type: 'query',
    question: 'What are the five pillars of Islam?'
}));

// Receive real-time updates
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## Features

### CLI Features
- **Search**: Find Islamic videos by topic
- **Process**: Run full pipeline (search → download → transcribe → embed)
- **Query**: Ask questions about indexed content
- **Summarize**: Generate content summaries
- **Status**: Check system health
- **Interactive**: Real-time chat mode

### Web Features
- **Modern UI**: Responsive, mobile-friendly interface
- **Real-time Updates**: WebSocket integration for live feedback
- **API Documentation**: Auto-generated Swagger docs
- **CORS Support**: Cross-origin requests enabled
- **Error Handling**: Comprehensive error management

### API Features
- **RESTful Design**: Standard HTTP methods and status codes
- **JSON Responses**: Consistent response format
- **Background Processing**: Long-running tasks via background jobs
- **WebSocket Support**: Real-time bidirectional communication
- **Health Checks**: System monitoring endpoints

## Configuration

The application uses the existing `crew_config.yaml` for all configuration:

```yaml
# LLM Configuration
llm:
  provider: "lm_studio"
  model_name: "lm_studio/qwen-3-14b"
  base_url: "http://192.168.0.74:1234/v1"
  model_path: "qwen-3-8b-chat.gguf"

# Embedding Configuration
embedding:
  model_name: "all-MiniLM-L6-v2"
  device: "cpu"

# Agent Configuration
agents:
  youtube_agent:
    role: "YouTube Content Researcher"
    goal: "Find and analyze Islamic educational content"
    # ... more configuration
```

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify configuration:**
   ```bash
   python -c "import yaml; print(yaml.safe_load(open('crew_config.yaml')))"
   ```

3. **Test CLI:**
   ```bash
   python main.py status
   ```

4. **Test Web API:**
   ```bash
   python api.py
   # Open http://localhost:8000 in browser
   ```

## Development

### Running in Development Mode

```bash
# CLI with debug logging
LOG_LEVEL=DEBUG python main.py interactive

# API with auto-reload
python api.py --reload --host 127.0.0.1 --port 8000
```

### Testing the Implementation

```bash
# Test CLI functionality
python main.py status
python main.py search "test query" --max-results 3

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/status
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │    │  Web Interface  │    │   API Clients   │
│    (main.py)    │    │ (index.html)    │    │   (External)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴───────────┐
                    │     FastAPI Server      │
                    │       (api.py)          │
                    └─────────────┬───────────┘
                                  │
                    ┌─────────────┴───────────┐
                    │   AkhiPipelineApp       │
                    │     (main.py)           │
                    └─────────────┬───────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                        │                         │
┌───────┴────────┐    ┌─────────┴────────┐    ┌──────────┴─────────┐
│   CrewAI       │    │   Tools & Utils  │    │   Configuration    │
│   Agents       │    │   (embedder,     │    │   (crew_config.    │
│   (Phase 4)    │    │   transcriber)   │    │    yaml)           │
└────────────────┘    └──────────────────┘    └────────────────────┘
```

## Next Steps

Phase 6 is now complete! The system provides:

✅ **CLI Interface** - Full command-line access
✅ **Web API** - RESTful API with WebSocket support
✅ **Web Interface** - Modern, responsive UI
✅ **Documentation** - Comprehensive usage guide
✅ **Integration** - Seamless connection to existing pipeline

Ready for **Phase 7: Testing & Validation**!