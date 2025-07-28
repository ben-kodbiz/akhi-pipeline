# Complete Pipeline Example: YouTube to QLoRA Training Data

This example demonstrates a complete end-to-end workflow using the Akhi Data Builder pipeline to convert Islamic YouTube lectures into QLoRA training data.

## 🎯 Goal
Create QLoRA training data from Nouman Ali Khan's lectures on patience and gratitude.

## 📋 Prerequisites

```bash
# Ensure you're in the project directory
cd /path/to/akhi_data_builder

# Install dependencies
pip install yt-dlp openai-whisper PyYAML

# Verify installation
python -c "import whisper; print('Whisper installed successfully')"
yt-dlp --version
```

## 🚀 Step-by-Step Walkthrough

### Step 1: Configure the Pipeline

First, let's customize our configuration for this specific example:

```bash
cd pipeline
cp config.yaml example_config.yaml
```

Edit `example_config.yaml` to focus on our specific needs:

```yaml
# Example Configuration for Nouman Ali Khan Lectures
search:
  default_terms:
    - "Nouman Ali Khan patience"
    - "Nouman Ali Khan gratitude"
    - "Nouman Ali Khan sabr"
  max_results_per_term: 3
  min_duration: 300  # 5 minutes minimum
  max_duration: 3600 # 1 hour maximum
  quality_filter: "best[height<=720]"

download:
  batch_size: 2
  audio_format: "mp3"
  max_retries: 3
  timeout: 300

transcription:
  model_size: "base"  # Good balance of speed/accuracy
  device: "cpu"       # Change to "cuda" if you have GPU
  language: "en"
  batch_size: 1

qlora:
  min_segment_length: 50
  max_segment_length: 300
  overlap_words: 10
  filter_music: true
  filter_short_segments: true

logging:
  level: "INFO"
  file_enabled: true
  console_enabled: true

pipeline:
  continue_on_error: true
  max_concurrent: 2
  state_backup_interval: 10
```

### Step 2: Search for Videos

```bash
# Search for videos using our custom configuration
python run_pipeline.py --config example_config.yaml --search
```

**Expected Output:**
```
2024-01-15 10:30:15 - INFO - Starting video search...
2024-01-15 10:30:16 - INFO - Searching for: Nouman Ali Khan patience
2024-01-15 10:30:18 - INFO - Found 3 videos for term: Nouman Ali Khan patience
2024-01-15 10:30:19 - INFO - Searching for: Nouman Ali Khan gratitude
2024-01-15 10:30:21 - INFO - Found 3 videos for term: Nouman Ali Khan gratitude
2024-01-15 10:30:22 - INFO - Searching for: Nouman Ali Khan sabr
2024-01-15 10:30:24 - INFO - Found 2 videos for term: Nouman Ali Khan sabr
2024-01-15 10:30:25 - INFO - Total videos discovered: 8
2024-01-15 10:30:25 - INFO - Videos saved to state manager
2024-01-15 10:30:25 - INFO - Search completed successfully
```

**Check the results:**
```bash
# View discovered videos
python run_pipeline.py --status
```

**Expected Status Output:**
```
=== Pipeline Status ===
Total videos: 8

Status breakdown:
- DISCOVERED: 8 videos
- DOWNLOADING: 0 videos
- DOWNLOADED: 0 videos
- TRANSCRIBING: 0 videos
- TRANSCRIBED: 0 videos
- FORMATTING: 0 videos
- COMPLETED: 0 videos
- FAILED: 0 videos

Recent videos:
1. "The Power of Patience in Islam" (15:30) - DISCOVERED
2. "Gratitude: The Key to Happiness" (22:45) - DISCOVERED
3. "Sabr: More Than Just Patience" (18:20) - DISCOVERED
...
```

### Step 3: Download Videos

```bash
# Download the discovered videos
python run_pipeline.py --config example_config.yaml --download
```

**Expected Output:**
```
2024-01-15 10:35:00 - INFO - Starting download process...
2024-01-15 10:35:01 - INFO - Found 8 videos to download
2024-01-15 10:35:02 - INFO - Downloading: "The Power of Patience in Islam"
2024-01-15 10:35:45 - INFO - Successfully downloaded: patience_islam_abc123.mp3
2024-01-15 10:35:46 - INFO - Downloading: "Gratitude: The Key to Happiness"
2024-01-15 10:36:30 - INFO - Successfully downloaded: gratitude_happiness_def456.mp3
...
2024-01-15 10:42:15 - INFO - Download completed: 8/8 successful, 0 failed
```

**Verify downloads:**
```bash
# Check downloaded files
ls -la output/clips/
```

**Expected File Structure:**
```
output/clips/
├── patience_islam_abc123.mp3          (15.2 MB)
├── gratitude_happiness_def456.mp3     (22.8 MB)
├── sabr_patience_ghi789.mp3           (18.5 MB)
├── islamic_patience_jkl012.mp3        (20.1 MB)
├── gratitude_quran_mno345.mp3         (16.7 MB)
├── patience_trials_pqr678.mp3         (19.3 MB)
├── thankfulness_islam_stu901.mp3      (21.4 MB)
└── sabr_strength_vwx234.mp3           (17.9 MB)
```

### Step 4: Transcribe Audio

```bash
# Transcribe all downloaded audio files
python run_pipeline.py --config example_config.yaml --transcribe
```

**Expected Output:**
```
2024-01-15 10:45:00 - INFO - Starting transcription process...
2024-01-15 10:45:01 - INFO - Found 8 audio files to transcribe
2024-01-15 10:45:02 - INFO - Loading Whisper model: base
2024-01-15 10:45:05 - INFO - Transcribing: patience_islam_abc123.mp3
2024-01-15 10:47:30 - INFO - Transcription completed: patience_islam_abc123.txt
2024-01-15 10:47:31 - INFO - Transcribing: gratitude_happiness_def456.mp3
2024-01-15 10:50:15 - INFO - Transcription completed: gratitude_happiness_def456.txt
...
2024-01-15 11:15:45 - INFO - All transcriptions completed successfully
```

**Check transcription results:**
```bash
# View transcript files
ls -la output/transcripts/

# Preview a transcript
head -20 output/transcripts/patience_islam_abc123.txt
```

**Expected Transcript Preview:**
```
Assalamu alaikum wa rahmatullahi wa barakatuh. Today we're going to talk about one of the most important concepts in Islam, and that is the concept of patience, or as we call it in Arabic, sabr.

Sabr is not just about waiting. It's not just about enduring difficulty. Sabr is a comprehensive attitude towards life that Allah subhanahu wa ta'ala wants us to develop.

When we look at the Quran, we find that Allah mentions sabr over and over again. In fact, the word sabr and its derivatives appear more than 100 times in the Quran.

This tells us something very important about how central this concept is to our faith...
```

### Step 5: Generate QLoRA Training Data

```bash
# Format transcripts into QLoRA-compatible JSON
python run_pipeline.py --config example_config.yaml --format
```

**Expected Output:**
```
2024-01-15 11:20:00 - INFO - Starting QLoRA formatting...
2024-01-15 11:20:01 - INFO - Found 8 transcripts to process
2024-01-15 11:20:02 - INFO - Processing: patience_islam_abc123.txt
2024-01-15 11:20:03 - INFO - Generated 12 training segments
2024-01-15 11:20:04 - INFO - Processing: gratitude_happiness_def456.txt
2024-01-15 11:20:05 - INFO - Generated 18 training segments
...
2024-01-15 11:20:25 - INFO - QLoRA formatting completed
2024-01-15 11:20:25 - INFO - Total training examples: 127
2024-01-15 11:20:25 - INFO - Output saved to: output/json/akhi_lora_training.json
```

### Step 6: Verify Final Results

```bash
# Check final status
python run_pipeline.py --summary
```

**Expected Summary:**
```
=== Pipeline Summary ===
Total videos processed: 8
Successful completions: 8
Failed videos: 0

Processing breakdown:
✅ COMPLETED: 8 videos
❌ FAILED: 0 videos

Output files:
📁 Audio files: 8 (152.9 MB total)
📄 Transcripts: 8 (1.2 MB total)
🎯 Training examples: 127
📊 Final JSON: output/json/akhi_lora_training.json (890 KB)

Average processing time per video: 3.2 minutes
Total pipeline runtime: 45 minutes
```

**Inspect the final training data:**
```bash
# Preview the QLoRA JSON structure
head -50 output/json/akhi_lora_training.json
```

**Expected JSON Structure:**
```json
[
  {
    "instruction": "Explain the Islamic concept of patience (sabr) and its importance in a believer's life.",
    "input": "",
    "output": "Sabr is not just about waiting or enduring difficulty. Sabr is a comprehensive attitude towards life that Allah subhanahu wa ta'ala wants us to develop. When we look at the Quran, we find that Allah mentions sabr over and over again, appearing more than 100 times. This tells us how central this concept is to our faith. True sabr involves maintaining faith during trials, being grateful during ease, and always trusting in Allah's wisdom.",
    "metadata": {
      "source": "The Power of Patience in Islam",
      "speaker": "Nouman Ali Khan",
      "topic": "patience",
      "duration": "15:30",
      "segment_id": 1,
      "word_count": 89
    }
  },
  {
    "instruction": "What does the Quran teach us about gratitude and thankfulness?",
    "input": "",
    "output": "Gratitude in Islam is not just saying 'Alhamdulillah' with our tongues. True gratitude, or shukr, involves recognizing Allah's blessings with our hearts, expressing thanks with our tongues, and showing appreciation through our actions. The Quran tells us that if we are grateful, Allah will certainly increase us in His blessings. This is a divine promise that connects our attitude of gratitude directly to receiving more from Allah.",
    "metadata": {
      "source": "Gratitude: The Key to Happiness",
      "speaker": "Nouman Ali Khan",
      "topic": "gratitude",
      "duration": "22:45",
      "segment_id": 1,
      "word_count": 95
    }
  }
]
```

## 🎉 Success! Complete Pipeline Results

### Final Output Structure:
```
pipeline/
├── output/
│   ├── clips/                          # 8 MP3 files (152.9 MB)
│   ├── transcripts/                    # 8 text files + 8 JSON files
│   └── json/
│       └── akhi_lora_training.json     # 127 training examples (890 KB)
├── db/
│   ├── video_state.jsonl               # State tracking
│   └── pipeline.log                    # Detailed logs
└── example_config.yaml                 # Our custom configuration
```

### Training Data Statistics:
- **Total Examples**: 127
- **Average Words per Example**: 85
- **Topics Covered**: Patience, Gratitude, Trials, Faith
- **Speaker**: Nouman Ali Khan
- **Quality**: High (filtered and validated)

## 🔧 Alternative: One-Command Pipeline

For future runs, you can execute the entire pipeline in one command:

```bash
# Run complete pipeline with custom config
python run_pipeline.py --config example_config.yaml --all

# Or with inline parameters
python run_pipeline.py --all --search-terms "Nouman Ali Khan patience,Nouman Ali Khan gratitude" --max-videos 5
```

## 🚨 Troubleshooting Common Issues

### Issue 1: No videos found
```bash
# Check search terms
python run_pipeline.py --search --search-terms "test query" --max-videos 1

# Verify internet connection
yt-dlp --list-formats "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### Issue 2: Download failures
```bash
# Check specific video
yt-dlp --extract-audio --audio-format mp3 "https://youtube.com/watch?v=VIDEO_ID"

# Update yt-dlp
pip install -U yt-dlp
```

### Issue 3: Transcription errors
```bash
# Test Whisper installation
python -c "import whisper; model = whisper.load_model('base'); print('Whisper working')"

# Check audio file
file output/clips/your_audio_file.mp3
```

### Issue 4: Low-quality transcriptions
```bash
# Use larger model for better accuracy
python run_pipeline.py --transcribe --model-size medium

# Or with GPU acceleration
python run_pipeline.py --transcribe --model-size large --device cuda
```

## 📊 Performance Optimization

### For Faster Processing:
```yaml
# In your config.yaml
transcription:
  model_size: "tiny"    # Fastest, lower accuracy
  device: "cuda"        # Use GPU if available
  batch_size: 2         # Process multiple files

download:
  batch_size: 3         # Download multiple videos
  quality_filter: "worst[height>=360]"  # Lower quality, faster download
```

### For Better Quality:
```yaml
# In your config.yaml
transcription:
  model_size: "large"   # Best accuracy, slower
  device: "cuda"        # GPU recommended for large model

qlora:
  min_segment_length: 100  # Longer, more coherent segments
  filter_music: true       # Remove non-speech content
```

## 🎯 Next Steps

1. **Review the training data**: Check `output/json/akhi_lora_training.json`
2. **Quality control**: Edit any transcription errors
3. **Expand dataset**: Run with different scholars or topics
4. **Train your model**: Use with Axolotl, QLoRA, or similar frameworks

**Example Axolotl usage:**
```bash
# Use the generated JSON with Axolotl
axolotl train --config your_axolotl_config.yaml --data output/json/akhi_lora_training.json
```

Congratulations! You've successfully created a high-quality Islamic AI training dataset using the Akhi Data Builder pipeline! 🎉