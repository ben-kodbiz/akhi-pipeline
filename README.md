# Akhi Data Builder

🕌 **An intelligent agentic pipeline for discovering, processing, and formatting Islamic knowledge from YouTube lectures into high-quality training data for fine-tuning Islamic LLMs.**

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- FFmpeg (for audio processing)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd akhi_data_builder
   ```

2. **Install Python dependencies**
   ```bash
   cd akhi_crewai
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd frontend_web
   npm install
   ```

### Running the Applications

#### Option 1: Web Interface (Recommended)

1. **Start the backend API**
   ```bash
   cd akhi_crewai
   python api.py
   ```
   Backend runs on: `http://localhost:8001`

2. **Start the frontend**
   ```bash
   cd frontend_web
   npm start
   ```
   Frontend runs on: `http://localhost:3000`

#### Option 2: Agentic Pipeline (Command Line)

```bash
cd pipeline
python run_pipeline.py
```

#### Option 3: Desktop Application

```bash
cd desktop_app
npm run tauri dev
```

## 🏗️ System Architecture

### Core Components

- **🌐 Frontend Web**: React-based web interface with real-time progress tracking
- **🖥️ Desktop App**: Tauri-based cross-platform desktop application
- **🔧 Backend API**: FastAPI server for pipeline orchestration
- **🤖 Agentic Pipeline**: CrewAI-powered intelligent agents for content processing
- **🛠️ Pipeline Tools**: Specialized tools for YouTube discovery, transcription, and QLoRA formatting

### Key Features

✅ **Intelligent Video Discovery**: AI agents find relevant Islamic content from trusted scholars  
✅ **High-Quality Transcription**: faster-whisper integration with speaker detection  
✅ **QLoRA Training Data**: Multiple training formats (segments, dialogues, summaries)  
✅ **Real-time Progress Tracking**: Web interface with live status updates  
✅ **Resumable Operations**: Persistent state management across pipeline runs  
✅ **Error Handling**: Comprehensive error recovery and logging  
✅ **Multi-format Output**: Plain text, JSON, and structured training data  

## 📊 Pipeline Stages

1. **🔍 Video Discovery**: AI agents search and curate Islamic lecture videos
2. **⬇️ Video Download**: Batch download of audio content using yt-dlp
3. **📝 Audio Transcription**: High-quality transcription with speaker identification
4. **📋 QLoRA Formatting**: Generate multiple training data formats for fine-tuning

## 🎯 Training Data Formats

The system generates diverse training examples:

- **Individual Segments**: Transcription and explanation tasks
- **Dialogue Format**: Conversational Islamic Q&A
- **Lecture Summaries**: Key points extraction and summarization
- **Metadata Enrichment**: Speaker info, timestamps, and content categorization

## 🔧 Configuration

- **Backend API**: Port 8001 (configurable)
- **Frontend**: Port 3000 with API integration
- **Pipeline**: YAML-based configuration in `pipeline/config.yaml`
- **Agents**: CrewAI configuration in `akhi_crewai/config/`

## 📁 Project Structure

```
akhi_data_builder/
├── frontend_web/          # React web interface
├── desktop_app/           # Tauri desktop application
├── backend/               # FastAPI backend services
├── akhi_crewai/          # CrewAI agents and tools
├── pipeline/             # Agentic pipeline core
├── docs/                 # Comprehensive documentation
└── README.md             # This file
```

## 📚 Documentation

For detailed documentation, visit the [`docs/`](./docs/) directory:

- **[📖 Project Overview](./docs/project/README.md)**: Complete system architecture and features
- **[👤 User Guide](./docs/guides/USER_GUIDE.md)**: Step-by-step usage instructions
- **[🔗 Integration Guide](./docs/integration/INTEGRATION_GUIDE.md)**: QLoRA and Axolotl integration
- **[🧩 Components](./docs/components/)**: Individual component documentation
- **[📈 Development Phases](./docs/phases/)**: Project development history
- **[🎯 QLoRA Documentation](./docs/qlora/)**: Fine-tuning and training guides

## 🚨 Troubleshooting

### Common Issues

- **FFmpeg not found**: Install FFmpeg for audio processing
- **API connection errors**: Ensure backend is running on port 8001
- **Transcription failures**: Check audio file formats and whisper model installation
- **Download errors**: Verify internet connection and YouTube URL validity

### Getting Help

1. Check the [User Guide](./docs/guides/USER_GUIDE.md) for detailed instructions
2. Review error logs in `pipeline/db/pipeline_log.txt`
3. Check the status tracker in `pipeline/db/status_tracker.json`

## 🔮 Recent Updates

### Latest Features (Phase 8)

- ✅ **QLoRA Integration**: Complete integration with Axolotl training framework
- ✅ **Documentation Organization**: Structured docs directory with comprehensive guides
- ✅ **Enhanced Error Handling**: Improved error recovery and user feedback
- ✅ **Performance Optimization**: Faster processing and reduced resource usage
- ✅ **Testing Cleanup**: Removed outdated test files, retained essential QLoRA tests

### Operational Improvements

- 🔄 **Resumable Pipeline**: Continue processing from interruption points
- 📊 **Real-time Monitoring**: Live progress tracking in web interface
- 🛡️ **Error Recovery**: Automatic retry mechanisms and graceful degradation
- 📝 **Enhanced Logging**: Detailed logs for debugging and monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Mission

**Making authentic Islamic knowledge more accessible through AI while maintaining the highest standards of accuracy and fidelity to traditional Islamic scholarship.**

---

**🔗 Quick Links:**
- [📖 Full Documentation](./docs/README.md)
- [🚀 User Guide](./docs/guides/USER_GUIDE.md)
- [🔧 Integration Guide](./docs/integration/INTEGRATION_GUIDE.md)
- [🎯 QLoRA Setup](./docs/qlora/QLoRAReadme.md)