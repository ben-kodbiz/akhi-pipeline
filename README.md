# Akhi Data Builder

A comprehensive Islamic dataset builder for LLM training that processes YouTube lectures into clean, structured training data.

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
│                          (CLI + Python Scripts)                                │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   STEP 1:       │    │   STEP 2:       │    │   STEP 3:       │
│   DOWNLOAD      │───►│   TRANSCRIBE    │───►│   GENERATE JSON │
│                 │    │                 │    │                 │
│ • yt-dlp        │    │ • faster-whisper│    │ • Content Filter│
│ • YouTube URLs  │    │ • Audio → Text  │    │ • LoRA Format   │
│ • MP3 Extract   │    │ • Language Det. │    │ • Training Data │
│ • Batch Process │    │ • Progress Track│    │ • JSON Export   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ output/clips/   │    │output/transcripts│    │ output/json/    │
│                 │    │                 │    │                 │
│ • *.mp3 files   │    │ • *.txt files   │    │ • akhi_lora.json│
│ • Audio clips   │    │ • Raw transcripts│    │ • Training data │
│ • Extracted     │    │ • Text content  │    │ • Instruction   │
│   from YouTube  │    │ • Timestamped   │    │   format        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
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

### 4. Pipeline Core
- **Technology**: Python + CLI tools
- **Components**:
  - **yt-dlp**: YouTube video downloading
  - **faster-whisper**: Audio transcription
  - **Custom scripts**: JSON formatting and filtering

## 📊 Data Flow

```
YouTube URLs ──► Download (yt-dlp) ──► MP3 Files ──► Transcribe (Whisper) ──► Text Files ──► Process ──► Training JSON
     │                │                    │                │                     │              │
     │                │                    │                │                     │              │
     ▼                ▼                    ▼                ▼                     ▼              ▼
[User Input]    [output/clips/]    [Audio Processing]  [output/transcripts/]  [Filtering]  [output/json/]
                   *.mp3                                    *.txt              >50 words   akhi_lora.json
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 14+
- FFmpeg
- Git

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd akhi_data_builder

# Install Python dependencies
pip install faster-whisper langdetect fastapi uvicorn yt-dlp

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

1. **Add YouTube URLs**: Enter YouTube lecture URLs in the dashboard
2. **Download Videos**: Click "Download" to extract audio as MP3 files
3. **Transcribe Audio**: Click "Transcribe" to convert audio to text
4. **Generate JSON**: Click "Generate JSON" to create training data
5. **Review & Export**: View transcripts and download the final JSON

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
│   ├── output/
│   │   ├── clips/             # Downloaded MP3 files
│   │   ├── transcripts/       # Generated text files
│   │   └── json/              # Training data JSON
│   ├── scripts/
│   │   └── make_quran_lora_json.py
│   └── run_pipeline.sh        # CLI pipeline script
└── README.md
```

## 🔄 Pipeline Stages

### Stage 1: Video Download
- **Tool**: yt-dlp
- **Input**: YouTube URLs
- **Output**: MP3 audio files
- **Location**: `pipeline/output/clips/`
- **Features**: Batch processing, error handling, progress tracking

### Stage 2: Audio Transcription
- **Tool**: faster-whisper
- **Input**: MP3 files
- **Output**: Text transcripts
- **Location**: `pipeline/output/transcripts/`
- **Features**: Language detection, real-time progress, error recovery

### Stage 3: JSON Generation
- **Tool**: Custom Python script
- **Input**: Text transcripts
- **Output**: LoRA training JSON
- **Location**: `pipeline/output/json/akhi_lora.json`
- **Features**: Content filtering (>50 words), instruction formatting

## 🎯 Training Data Format

The generated JSON follows the LoRA training format:

```json
[
  {
    "instruction": "Summarize and offer Islamic advice based on this:",
    "input": "[Transcript content from Islamic lecture]",
    "output": "Remember, Allah is always with those who are patient and sincere."
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

- **Download Errors**: Invalid URLs, network issues, yt-dlp failures
- **Transcription Errors**: Audio format issues, whisper model problems
- **JSON Generation Errors**: File access issues, content filtering problems
- **API Errors**: Server connectivity, endpoint failures
- **UI Errors**: Real-time error display with detailed messages

## 📈 Monitoring & Progress

- **Real-time Progress Bars**: Visual indicators for all pipeline stages
- **File Counters**: Track processed vs. total files
- **Status Endpoints**: Detailed progress and error information
- **Background Processing**: Non-blocking operations with status polling

## 🔮 Future Enhancements

- [ ] Speaker identification and classification
- [ ] Topic categorization module
- [ ] Advanced content filtering
- [ ] Integration with Axolotl LoRA trainer
- [ ] Batch processing optimization
- [ ] Cloud deployment support
- [ ] Multi-language support
- [ ] Advanced transcript editing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Goal

Build a clean, repeatable dataset pipeline to fine-tune Islamic LLMs (like Akhi) using YouTube lectures, providing high-quality training data for Islamic AI applications.