# Akhi Data Builder - User Guide

A comprehensive guide to using the Akhi Data Builder system for creating Islamic AI training datasets from YouTube lectures.

## 🎯 Overview

The Akhi Data Builder provides three ways to interact with the system:
1. **Web Interface** - User-friendly browser-based dashboard
2. **Desktop App** - Native desktop application
3. **Command Line** - Direct pipeline control for advanced users

---

## 🌐 Web Interface Guide

### Getting Started

1. **Start the Backend API**:
   ```bash
   cd backend/api
   uvicorn main:app --reload --port 8001
   ```

2. **Start the Web Frontend**:
   ```bash
   cd frontend_web
   npm install  # First time only
   npm start
   ```

3. **Access the Application**:
   - Open your browser to: http://localhost:3000
   - API documentation: http://localhost:8001/docs

### Using the Dashboard

#### Step 1: Download Videos
1. **Add YouTube URLs**:
   - Paste YouTube lecture URLs in the text area
   - One URL per line
   - Example:
     ```
     https://www.youtube.com/watch?v=abc123
     https://www.youtube.com/watch?v=def456
     ```

2. **Click "Download Videos"**:
   - The system will extract audio as MP3 files
   - Progress will be shown in real-time
   - Files are saved to `pipeline/output/clips/`

#### Step 2: Transcribe Audio
1. **Click "Start Transcription"**:
   - Uses faster-whisper for accurate transcription
   - Shows progress with current file being processed
   - Creates both text and JSON transcripts

2. **Monitor Progress**:
   - Real-time progress bar
   - Current file indicator
   - Completion status

#### Step 3: Generate Training Data
1. **Click "Generate JSON"**:
   - Formats transcripts for AI training
   - Creates QLoRA-compatible JSON
   - Applies content filtering

2. **Review Results**:
   - Navigate to "Transcripts" page to review
   - Edit transcripts if needed
   - Download final JSON from "JSON" page

### Navigation

- **Dashboard**: Main control panel
- **Transcripts**: View and edit transcribed content
- **JSON**: Preview and download training data
- **Status**: Monitor overall system status

### Error Handling

- **Red error messages**: Check console for details
- **Network errors**: Ensure backend is running
- **Download failures**: Verify YouTube URLs are valid
- **Transcription errors**: Check audio file quality

---

## 🖥️ Desktop App Guide

### Installation

1. **Install Prerequisites**:
   ```bash
   # Install Rust and Cargo
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   
   # Install Tauri CLI
   npm install -g @tauri-apps/cli
   ```

2. **Build the App**:
   ```bash
   cd desktop_app
   npm install
   npm run tauri build
   ```

3. **Run Development Version**:
   ```bash
   npm run tauri dev
   ```

### Using the Desktop App

#### Features
- **Native Performance**: Faster than web interface
- **File System Access**: Direct access to output files
- **Offline Capability**: Works without internet for local files
- **Cross-Platform**: Windows, macOS, Linux support

#### Workflow
1. **Upload Video Links**: Drag and drop or paste URLs
2. **Start Processing**: Click to begin download and transcription
3. **Monitor Progress**: Native progress indicators
4. **Export Results**: Save JSON files to desired location

---

## ⌨️ Command Line Guide

### Quick Start

```bash
# Navigate to pipeline directory
cd pipeline

# Run full pipeline with default settings
python run_pipeline.py --all
```

### Step-by-Step Usage

#### Step 1: Search and Discover Videos
```bash
# Search for Islamic lectures
python run_pipeline.py --search \
  --search_terms "Islamic lecture,Quran tafsir,Hadith explanation" \
  --max_results 10
```

**Options**:
- `--search_terms`: Comma-separated search terms
- `--max_results`: Maximum videos per search term

#### Step 2: Download Videos
```bash
# Download discovered videos
python run_pipeline.py --download --max_videos 5

# Or generate download links file
python agents/youtube_agent.py --generate-links --max-videos 10
```

**Options**:
- `--max_videos`: Limit number of videos to download

#### Step 3: Transcribe Audio
```bash
# Transcribe with default settings
python run_pipeline.py --transcribe

# Transcribe with custom options
python run_pipeline.py --transcribe \
  --model_size medium \
  --device cuda \
  --language english
```

**Options**:
- `--model_size`: tiny, base, small, medium, large
- `--device`: cpu, cuda
- `--language`: auto, english, arabic, etc.

#### Step 4: Generate Training JSON
```bash
# Generate QLoRA-compatible JSON
python run_pipeline.py --format \
  --min_words 50 \
  --max_words 500
```

**Options**:
- `--min_words`: Minimum words per training example
- `--max_words`: Maximum words per training example

### Advanced Usage

#### Custom Search Terms
```bash
# Use specific scholars
python agents/youtube_agent.py --search \
  --search_terms "Nouman Ali Khan,Mufti Menk,Omar Suleiman" \
  --max_results_per_term 5
```

#### Batch Processing
```bash
# Process specific video list
echo "https://www.youtube.com/watch?v=abc123" > video_links.txt
echo "https://www.youtube.com/watch?v=def456" >> video_links.txt
python run_pipeline.py --download
```

#### Status Monitoring
```bash
# Check current status
python agents/youtube_agent.py --search "" --max_results_per_term 0

# View detailed logs
tail -f db/pipeline_log.txt
```

---

## 📁 Output Structure

```
pipeline/output/
├── clips/              # Downloaded MP3 audio files
│   ├── video1.mp3
│   └── video2.mp3
├── transcripts/        # Transcribed text files
│   ├── video1.txt      # Plain text transcript
│   ├── video1.json     # Enhanced JSON with timestamps
│   └── video2.txt
└── json/              # Training data
    └── akhi_lora.json # QLoRA-compatible training data
```

---

## 🔧 Configuration

### Search Terms
Edit default search terms in `pipeline/agents/youtube_agent.py`:
```python
DEFAULT_SEARCH_TERMS = [
    "Nouman Ali Khan tafsir",
    "Mufti Menk lecture",
    "Yasir Qadhi hadith",
    "Omar Suleiman Islamic lecture",
    "Islamic scholar Quran explanation"
]
```

### Transcription Settings
Modify transcription parameters in `pipeline/agents/transcriber.py`:
- Model size: tiny, base, small, medium, large
- Language detection: auto or specific language
- Device: CPU or CUDA GPU

### JSON Formatting
Adjust training data format in `pipeline/agents/qlora_formatter.py`:
- Word count limits
- Content filtering rules
- Training example formats

---

## 🚨 Troubleshooting

### Common Issues

#### "No videos found"
- Check internet connection
- Verify search terms are relevant
- Try different search terms

#### "Download failed"
- Video may be private or deleted
- Check YouTube URL format
- Ensure yt-dlp is updated: `pip install -U yt-dlp`

#### "Transcription error"
- Check audio file exists and is valid
- Ensure sufficient disk space
- For GPU: verify CUDA installation

#### "JSON generation failed"
- Ensure transcripts exist
- Check file permissions
- Verify transcript content quality

### Performance Tips

1. **Use GPU for transcription**: Add `--device cuda`
2. **Batch processing**: Process multiple videos at once
3. **Monitor disk space**: Audio and transcript files can be large
4. **Regular cleanup**: Remove processed files when done

### Getting Help

- **Logs**: Check `pipeline/db/pipeline_log.txt`
- **Status**: Use `--help` flag with any command
- **Debug**: Add `--verbose` flag for detailed output

---

## 📊 Best Practices

### Content Selection
- Choose high-quality audio lectures
- Prefer longer lectures (10+ minutes)
- Select diverse topics and speakers
- Avoid music or background noise

### Quality Control
- Review transcripts before generating JSON
- Edit obvious transcription errors
- Remove irrelevant segments
- Verify speaker identification

### Training Data
- Aim for 1000+ training examples
- Balance different topics and speakers
- Include various lecture formats
- Maintain consistent quality standards

---

## 🎯 Next Steps

After generating your training data:

1. **Review Quality**: Check `output/json/akhi_lora.json`
2. **Fine-tune Model**: Use with Axolotl or similar tools
3. **Test Results**: Evaluate model performance
4. **Iterate**: Refine data and retrain as needed

For advanced AI training workflows, refer to QLoRA and Axolotl documentation.