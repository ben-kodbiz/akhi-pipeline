# Search Feature Implementation - Agent Documentation

## Overview

This document outlines all the search-related features implemented in the `search_feature` branch of the Akhi Data Builder project. The primary focus was to replace Google API dependency with `yt-dlp` for video discovery and downloading.

---

## 🔍 Core Search Features

### 1. **yt-dlp Integration for Video Discovery**
- **File**: `pipeline/agents/youtube_agent.py`
- **Description**: Complete rewrite of the YouTube agent to use `yt-dlp` instead of Google YouTube Data API
- **Features**:
  - Search YouTube videos using natural language queries
  - Extract video metadata (title, channel, duration, URL)
  - Programmatic duration filtering (e.g., videos between 10-60 minutes)
  - No API key requirements
  - Robust error handling and retry mechanisms

### 2. **Advanced Search Parameters**
- **Search Terms**: Configurable search queries for Islamic content
- **Duration Filtering**: Automatic filtering by video length
- **Result Limiting**: Control maximum results per search term
- **Content Validation**: Verify video accessibility before processing

### 3. **Default Search Terms for Islamic Content**
```python
DEFAULT_SEARCH_TERMS = [
    "Nouman Ali Khan tafsir",
    "Mufti Menk lecture", 
    "Yasir Qadhi hadith",
    "Omar Suleiman Islamic lecture",
    "Islamic scholar Quran explanation"
]
```

---

## 🛠️ Technical Implementation

### 1. **YouTube Agent Rewrite**
- **Location**: `pipeline/agents/youtube_agent.py`
- **Key Changes**:
  - Removed `googleapiclient` dependency
  - Implemented `yt-dlp` for search and metadata extraction
  - Added comprehensive error handling
  - Implemented status tracking and logging

### 2. **Pipeline Integration**
- **File**: `pipeline/run_pipeline.py`
- **Features**:
  - Removed `--api_key` parameter requirement
  - Integrated yt-dlp search functionality
  - Added search step to pipeline workflow
  - Command-line interface for search operations

### 3. **Status Tracking System**
- **File**: `pipeline/db/status_tracker.json`
- **Purpose**: Track discovered, downloaded, transcribed, and processed videos
- **Features**:
  - Real-time status updates
  - Video metadata storage
  - Progress tracking across pipeline stages

---

## 📋 Command Line Interface

### Search Commands

```bash
# Basic search with default terms
python run_pipeline.py --search

# Custom search terms
python run_pipeline.py --search --search_terms "Islamic lecture,Quran tafsir"

# Limit results per term
python run_pipeline.py --search --max_results 10

# Generate video links file
python agents/youtube_agent.py --generate-links --max-videos 10

# Search with specific parameters
python agents/youtube_agent.py --search \
  --search_terms "Nouman Ali Khan,Mufti Menk" \
  --max_results_per_term 5
```

### Pipeline Integration

```bash
# Run full pipeline with search
python run_pipeline.py --all

# Step-by-step execution
python run_pipeline.py --search
python run_pipeline.py --download
python run_pipeline.py --transcribe
python run_pipeline.py --format
```

---

## 🔧 Configuration Options

### Search Parameters
- **search_terms**: List of search queries
- **max_results_per_term**: Maximum videos per search term
- **min_duration**: Minimum video duration (seconds)
- **max_duration**: Maximum video duration (seconds)
- **max_videos**: Total video limit for processing

### Duration Filtering
- **Default Range**: 600-3600 seconds (10-60 minutes)
- **Configurable**: Adjustable via command-line parameters
- **Purpose**: Focus on substantial lecture content

---

## 📊 Output and Results

### Generated Files
1. **video_links.txt**: List of discovered video URLs
2. **status_tracker.json**: Comprehensive video metadata and status
3. **pipeline_log.txt**: Detailed operation logs

### Video Discovery Results
- **Total Videos Discovered**: 9 videos in test run
- **Channels**: Nouman Ali Khan, Mufti Menk, Omar Suleiman, Yasir Qadhi
- **Content Types**: Tafsir, lectures, Islamic guidance
- **Duration Range**: 10-60 minutes of quality content

---

## 🚀 Performance Improvements

### 1. **No API Rate Limits**
- Eliminated Google API quota restrictions
- Unlimited search operations
- No API key management required

### 2. **Direct Video Access**
- `yt-dlp` provides direct video stream access
- Faster metadata extraction
- More reliable video availability checking

### 3. **Robust Error Handling**
- Graceful handling of unavailable videos
- Automatic retry mechanisms
- Comprehensive logging for debugging

---

## 🔄 Integration with Existing Features

### 1. **Download Pipeline**
- Seamless integration with existing download functionality
- Maintains compatibility with manual URL input
- Supports both discovered and manually provided videos

### 2. **Transcription System**
- Works with faster-whisper transcription
- Maintains existing transcript formats
- Preserves speaker detection and timestamps

### 3. **JSON Generation**
- Compatible with QLoRA formatting
- Maintains training data quality
- Supports existing filtering and processing

---

## 📝 Documentation Updates

### Files Updated
1. **README.md**: Removed Google API references, updated installation
2. **task2.md**: Updated usage examples and API requirements
3. **USER_GUIDE.md**: Comprehensive user guide for all interfaces

### Key Changes
- Removed all Google API key references
- Updated installation instructions
- Modified command-line examples
- Updated architecture descriptions

---

## 🧪 Testing and Validation

### Functionality Tests
- ✅ Video search and discovery
- ✅ Metadata extraction
- ✅ Duration filtering
- ✅ Link generation
- ✅ Status tracking
- ✅ Error handling

### Integration Tests
- ✅ Pipeline workflow
- ✅ Command-line interface
- ✅ File generation
- ✅ Backend API compatibility

---

## 🎯 Future Enhancements

### Potential Improvements
1. **Advanced Filtering**: Content quality scoring
2. **Playlist Support**: Discover videos from specific playlists
3. **Channel Monitoring**: Track new uploads from favorite scholars
4. **Language Detection**: Multi-language search capabilities
5. **Duplicate Detection**: Avoid processing duplicate content

---

## 📈 Impact Summary

### Benefits Achieved
- **Zero API Dependencies**: Complete removal of Google API requirements
- **Unlimited Discovery**: No rate limits or quota restrictions
- **Enhanced Reliability**: More robust video discovery and access
- **Simplified Setup**: No API key configuration needed
- **Better Performance**: Faster metadata extraction and processing

### Technical Debt Resolved
- Eliminated external API dependency
- Simplified authentication requirements
- Reduced configuration complexity
- Improved error handling and logging

This search feature implementation represents a significant improvement in the system's autonomy, reliability, and ease of use while maintaining full compatibility with existing functionality.