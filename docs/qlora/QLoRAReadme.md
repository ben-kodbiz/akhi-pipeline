# QLoRA Integration Guide for Akhi CrewAI Pipeline

## Overview

This comprehensive guide covers the complete QLoRA (Quantized Low-Rank Adaptation) integration within the Akhi CrewAI Pipeline. The system transforms YouTube Islamic content into high-quality training datasets for specialized AI model fine-tuning using Axolotl, featuring a complete pipeline from content discovery to model deployment.

## 🚀 Complete Pipeline Architecture

```
YouTube Search → Download → Transcription → Content Analysis → QLoRA Formatting → Axolotl Training → Model Deployment
     ↓              ↓           ↓              ↓                ↓                ↓                ↓
  Smart Filter   Audio Extract  Whisper AI   Islamic RAG    Standard Format   Config Generation  API Server
     ↓              ↓           ↓              ↓                ↓                ↓                ↓
 run_pipeline.py → Audio Tools → OpenAI API → enhanced_rag → QLoRAFormatter → prepare_axolotl → Model Deploy
```

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Pipeline Components](#pipeline-components)
5. [Configuration Management](#configuration-management)
6. [CLI Operations](#cli-operations)
7. [Data Processing Workflow](#data-processing-workflow)
8. [QLoRA Training with Axolotl](#qlora-training-with-axolotl)
9. [Monitoring & Logging](#monitoring--logging)
10. [Production Deployment](#production-deployment)
11. [Troubleshooting](#troubleshooting)
12. [Advanced Features](#advanced-features)
13. [Best Practices](#best-practices)

## Prerequisites

### System Requirements

- **Python**: 3.8+ (3.10+ recommended)
- **CUDA**: 11.8+ or 12.1+ (for GPU acceleration)
- **Memory**: Minimum 16GB RAM (32GB+ recommended for production)
- **Storage**: 100GB+ free space (datasets + models)
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3080/4080+ recommended)
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### Hardware Recommendations

| Component | Minimum | Recommended | Production |
|-----------|---------|-------------|------------|
| RAM | 16GB | 32GB | 64GB+ |
| GPU VRAM | 8GB | 12GB | 24GB+ |
| Storage | 50GB | 100GB | 500GB+ SSD |
| CPU Cores | 4 | 8 | 16+ |

### Required API Keys

```bash
# Required environment variables
export OPENAI_API_KEY="your_openai_key_here"
export HUGGINGFACE_TOKEN="your_hf_token_here"
export WANDB_API_KEY="your_wandb_key_here"  # Optional for experiment tracking
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/ben-kodbiz/akhi-pipeline.git
cd akhi-pipeline/akhi_crewai
git checkout Qlora_integration  # Switch to QLoRA branch
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip setuptools wheel
```

### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import crewai; print('CrewAI installed successfully')"
```

### 4. Setup Environment Variables

```bash
# Create environment file
cp .env.example .env

# Add required environment variables
echo "OPENAI_API_KEY=your_openai_key_here" >> .env
echo "HUGGINGFACE_TOKEN=your_hf_token_here" >> .env
echo "WANDB_API_KEY=your_wandb_key_here" >> .env

# Optional configuration
echo "AKHI_LOG_LEVEL=INFO" >> .env
echo "AKHI_MAX_WORKERS=8" >> .env
echo "AKHI_CACHE_DIR=./cache" >> .env
echo "AKHI_GPU_MEMORY_FRACTION=0.9" >> .env
```

### 5. Initialize System

```bash
# Create necessary directories
mkdir -p data/{audio,transcripts,embeddings,crew_outputs}
mkdir -p logs outputs models cache temp

# Initialize Islamic knowledge base
python enhanced_rag_checker.py --build-knowledge-base

# Test installation
python run_pipeline.py --help
python prepare_axolotl_dataset.py --help
python enhanced_rag_checker.py --help
python cli_logger.py --help
```

## Quick Start

### 1. Complete Pipeline Execution

```bash
# Run the complete pipeline from YouTube search to QLoRA training
python run_pipeline.py --query "mufti menk trauma healing" --max-videos 5

# With custom configuration
python run_pipeline.py \
  --query "islamic finance principles" \
  --max-videos 10 \
  --config config/qlora_config.yaml \
  --output-dir ./custom_output
```

### 2. Step-by-Step Execution

```bash
# Step 1: Search and download videos
python run_pipeline.py --query "quran recitation" --stage search_download

# Step 2: Transcribe audio
python run_pipeline.py --stage transcribe --input-dir ./data/audio

# Step 3: Format for QLoRA
python run_pipeline.py --stage format_qlora --input-dir ./data/transcripts

# Step 4: Prepare Axolotl dataset
python prepare_axolotl_dataset.py --input-dir ./outputs --output-dir ./axolotl_data
```

### 3. Real-time Monitoring

```bash
# Start monitoring dashboard
python cli_logger.py --monitor-pipeline

# Analyze existing logs
python cli_logger.py --analyze-logs --log-dir ./logs

# View current status
python cli_logger.py --dashboard
```

## Pipeline Components

### Core Components Overview

```
📁 akhi_crewai/
├── 🚀 run_pipeline.py           # Main orchestrator
├── 🔧 prepare_axolotl_dataset.py # Dataset preparation
├── 🛡️ enhanced_rag_checker.py   # Islamic content validation
├── 📊 cli_logger.py             # Monitoring & logging
├── 📋 config/                   # Configuration files
├── 🤖 agents/                   # CrewAI agents
├── 🛠️ tools/                    # Processing tools
└── 📈 logs/                     # System logs
```

### 1. Unified Pipeline Orchestrator (`run_pipeline.py`)

**Purpose**: Central command center for the entire QLoRA pipeline

**Key Features**:
- YouTube search and content collection
- Audio extraction and transcription
- Content quality assessment
- QLoRA data formatting
- Axolotl dataset preparation

**Usage Examples**:
```bash
# Basic usage
python run_pipeline.py --query "islamic ethics" --max-videos 5

# Advanced usage with all options
python run_pipeline.py \
  --query "mufti menk advice" \
  --max-videos 10 \
  --min-duration 300 \
  --max-duration 3600 \
  --language en \
  --quality-threshold 0.8 \
  --output-dir ./custom_output \
  --config config/qlora_config.yaml \
  --enable-rag-validation \
  --parallel-workers 4 \
  --verbose
```

### 2. Axolotl Dataset Preparation (`prepare_axolotl_dataset.py`)

**Purpose**: Merge QLoRA JSONs and auto-generate Axolotl configuration

**Key Features**:
- Multi-format support (Alpaca, prompt-response, Q&A)
- Data validation and quality filtering
- Train/validation split
- Automatic config.yml generation
- Dataset statistics and analysis

**Usage Examples**:
```bash
# Basic dataset preparation
python prepare_axolotl_dataset.py --input-dir ./outputs --output-dir ./axolotl_data

# Advanced preparation with custom settings
python prepare_axolotl_dataset.py \
  --input-dir ./outputs \
  --output-dir ./axolotl_data \
  --train-split 0.85 \
  --min-length 50 \
  --max-length 2048 \
  --base-model "meta-llama/Llama-2-7b-hf" \
  --lora-r 32 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --epochs 5 \
  --validate-islamic-content
```

### 3. Enhanced RAG Checker (`enhanced_rag_checker.py`)

**Purpose**: SQLite + FAISS backend for Islamic content validation

**Key Features**:
- Quran and Hadith knowledge base
- Semantic similarity checking
- Hallucination detection
- Content authenticity validation

**Usage Examples**:
```bash
# Build knowledge base
python enhanced_rag_checker.py --build-knowledge-base

# Check content authenticity
python enhanced_rag_checker.py \
  --check-content "The Prophet said about patience..." \
  --threshold 0.7

# Batch validation
python enhanced_rag_checker.py \
  --validate-file ./data/content.jsonl \
  --output-report ./validation_report.json
```

### 4. CLI Logger & Monitor (`cli_logger.py`)

**Purpose**: Comprehensive logging, monitoring, and system health tracking

**Key Features**:
- Real-time pipeline monitoring
- Resource usage tracking
- Error analysis and reporting
- Performance metrics

**Usage Examples**:
```bash
# Start real-time monitoring
python cli_logger.py --monitor-pipeline

# Analyze logs
python cli_logger.py --analyze-logs --log-dir ./logs

# View dashboard
python cli_logger.py --dashboard

# Cleanup old logs
python cli_logger.py --cleanup-logs --max-log-files 50
```

## Configuration Management

### Configuration File Structure

#### 1. QLoRA Configuration (`config/qlora_config.yaml`)

**Model Selection**:
```yaml
qlora:
  base_model: "microsoft/DialoGPT-medium"  # Options:
  # - "microsoft/DialoGPT-medium" (2.7B params, good for testing)
  # - "meta-llama/Llama-2-7b-hf" (7B params, production ready)
  # - "mistralai/Mistral-7B-v0.1" (7B params, efficient)
  # - "microsoft/DialoGPT-large" (762M params, lightweight)
```

**LoRA Parameters Tuning**:
```yaml
# Memory vs Performance Trade-offs
lora_r: 16        # 8=lightweight, 16=balanced, 32=high-quality
lora_alpha: 32    # Typically 2x lora_r
lora_dropout: 0.1 # 0.05=less regularization, 0.2=more regularization

# Target modules (model-specific)
target_modules:
  # For Llama/Mistral:
  - "q_proj"
  - "v_proj"
  - "k_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"
  
  # For DialoGPT:
  - "c_attn"      # Attention projection
  - "c_proj"      # Output projection
```

**Training Optimization**:
```yaml
# Batch size calculation: effective_batch = batch_size * gradient_accumulation_steps
batch_size: 4                    # Adjust based on GPU memory
gradient_accumulation_steps: 4   # Total effective batch size = 16

# Learning rate scheduling
learning_rate: 2e-4             # Conservative for stability
warmup_steps: 100               # 10% of total steps recommended
weight_decay: 0.01              # L2 regularization
max_grad_norm: 1.0              # Gradient clipping

# Memory optimization
gradient_checkpointing: true     # Trade compute for memory
optim: "paged_adamw_32bit"      # Memory-efficient optimizer
```

#### 2. Crew Configuration (`config/crew_config.yaml`)

```yaml
crew_settings:
  # Agent configuration
  max_agents: 5
  agent_timeout: 300
  
  # QLoRA integration
  enable_qlora_training: true
  auto_format_data: true
  validate_before_training: true
  
  # Enhanced features
  use_enhanced_rag: true
  axolotl_integration: true
  
  # Processing settings
  parallel_processing: true
  max_concurrent_tasks: 4
  
  # Quality control
  min_content_quality: 0.7
  islamic_content_validation: true
  
  # Output settings
  save_intermediate_results: true
  cleanup_temp_files: true
```

#### 3. Orchestration Configuration (`config/crew_orchestration.yaml`)

```yaml
orchestration:
  # Pipeline stages
  stages:
    - search_collect
    - download
    - transcribe
    - summarize
    - format_qlora
    - prepare_dataset
    - generate_config
    - training
  
  # Stage-specific settings
  search_collect:
    max_results: 50
    quality_filter: true
    duration_range: [300, 3600]  # 5 min to 1 hour
  
  transcribe:
    model: "whisper-large-v2"
    language: "auto"
    chunk_length: 30
  
  format_qlora:
    instruction_format: "alpaca"
    max_sequence_length: 2048
    include_metadata: true
  
  # Error handling
  retry_attempts: 3
  timeout_per_stage: 1800  # 30 minutes
  
  # Logging
  log_level: "INFO"
  detailed_logging: true
```

### Configuration Best Practices

#### GPU Memory Optimization

```yaml
# For 8GB GPU (RTX 3070/4060 Ti)
qlora:
  batch_size: 2
  gradient_accumulation_steps: 8
  max_seq_length: 1024
  load_in_4bit: true

# For 12GB GPU (RTX 3080/4070)
qlora:
  batch_size: 4
  gradient_accumulation_steps: 4
  max_seq_length: 2048
  load_in_4bit: true

# For 24GB GPU (RTX 3090/4090)
qlora:
  batch_size: 8
  gradient_accumulation_steps: 2
  max_seq_length: 4096
  load_in_4bit: false  # Can use full precision
```

#### Islamic Content Optimization

```yaml
data:
  # Arabic text support
  arabic_support: true
  transliteration: true
  
  # Citation formats
  verse_citation_format: "quran_hadith"  # "Quran 2:255" format
  hadith_citation_format: "bukhari_muslim"  # "Bukhari 1234" format
  
  # Content validation
  validate_arabic_text: true
  check_islamic_authenticity: true
  min_islamic_relevance: 0.8
```

## CLI Operations

### Primary Commands

#### 1. Complete Pipeline Execution

```bash
# Basic execution
python run_pipeline.py --query "islamic ethics" --max-videos 5

# Production execution with all features
python run_pipeline.py \
  --query "mufti menk advice" \
  --max-videos 20 \
  --min-duration 300 \
  --max-duration 3600 \
  --language en \
  --quality-threshold 0.8 \
  --output-dir ./production_output \
  --config config/qlora_config.yaml \
  --enable-rag-validation \
  --parallel-workers 8 \
  --batch-size 4 \
  --verbose
```

#### 2. Stage-Specific Execution

```bash
# Search and download only
python run_pipeline.py \
  --query "quran tafseer" \
  --stage search_download \
  --max-videos 10

# Transcription only
python run_pipeline.py \
  --stage transcribe \
  --input-dir ./data/audio \
  --whisper-model large-v2

# QLoRA formatting only
python run_pipeline.py \
  --stage format_qlora \
  --input-dir ./data/transcripts \
  --format alpaca

# Dataset preparation only
python run_pipeline.py \
  --stage prepare_dataset \
  --input-dir ./outputs \
  --output-dir ./axolotl_ready
```

#### 3. Axolotl Dataset Preparation

```bash
# Basic dataset preparation
python prepare_axolotl_dataset.py \
  --input-dir ./outputs \
  --output-dir ./axolotl_data

# Advanced preparation with custom model
python prepare_axolotl_dataset.py \
  --input-dir ./outputs \
  --output-dir ./axolotl_data \
  --base-model "meta-llama/Llama-2-7b-hf" \
  --train-split 0.85 \
  --validation-split 0.15 \
  --min-length 50 \
  --max-length 2048 \
  --lora-r 32 \
  --lora-alpha 64 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --epochs 5 \
  --validate-islamic-content \
  --generate-stats
```

#### 4. Enhanced RAG Validation

```bash
# Build Islamic knowledge base
python enhanced_rag_checker.py --build-knowledge-base

# Validate single content
python enhanced_rag_checker.py \
  --check-content "The Prophet (PBUH) said about patience..." \
  --threshold 0.7 \
  --detailed-report

# Batch validation
python enhanced_rag_checker.py \
  --validate-file ./data/content.jsonl \
  --output-report ./validation_report.json \
  --threshold 0.8 \
  --parallel-workers 4

# Interactive mode
python enhanced_rag_checker.py --interactive
```

#### 5. Monitoring and Logging

```bash
# Real-time monitoring dashboard
python cli_logger.py --monitor-pipeline

# Analyze existing logs
python cli_logger.py \
  --analyze-logs \
  --log-dir ./logs \
  --time-range "last-24h"

# Generate performance report
python cli_logger.py \
  --dashboard \
  --export-report ./performance_report.json

# Cleanup old logs
python cli_logger.py \
  --cleanup-logs \
  --max-log-files 50 \
  --older-than-days 30
```

### Command Line Arguments Reference

#### run_pipeline.py Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--query` | str | Required | YouTube search query |
| `--max-videos` | int | 10 | Maximum videos to process |
| `--min-duration` | int | 60 | Minimum video duration (seconds) |
| `--max-duration` | int | 3600 | Maximum video duration (seconds) |
| `--language` | str | "en" | Content language |
| `--quality-threshold` | float | 0.7 | Minimum content quality score |
| `--output-dir` | str | "./outputs" | Output directory |
| `--config` | str | "config/qlora_config.yaml" | Configuration file |
| `--stage` | str | "all" | Pipeline stage to run |
| `--enable-rag-validation` | flag | False | Enable RAG content validation |
| `--parallel-workers` | int | 4 | Number of parallel workers |
| `--batch-size` | int | 4 | Processing batch size |
| `--verbose` | flag | False | Verbose logging |

#### prepare_axolotl_dataset.py Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input-dir` | str | Required | Input directory with JSON files |
| `--output-dir` | str | Required | Output directory for Axolotl data |
| `--base-model` | str | "microsoft/DialoGPT-medium" | Base model for training |
| `--train-split` | float | 0.9 | Training data split ratio |
| `--validation-split` | float | 0.1 | Validation data split ratio |
| `--min-length` | int | 10 | Minimum text length |
| `--max-length` | int | 2048 | Maximum text length |
| `--lora-r` | int | 16 | LoRA rank parameter |
| `--lora-alpha` | int | 32 | LoRA alpha parameter |
| `--batch-size` | int | 4 | Training batch size |
| `--learning-rate` | float | 2e-4 | Learning rate |
| `--epochs` | int | 3 | Number of training epochs |
| `--validate-islamic-content` | flag | False | Enable Islamic content validation |
| `--generate-stats` | flag | False | Generate dataset statistics |

### Environment Variables

```bash
# Required API keys
export OPENAI_API_KEY="your_openai_key_here"
export HUGGINGFACE_TOKEN="your_hf_token_here"
export WANDB_API_KEY="your_wandb_key_here"

# Optional configuration
export AKHI_LOG_LEVEL="INFO"              # DEBUG, INFO, WARNING, ERROR
export AKHI_MAX_WORKERS="8"               # Maximum parallel workers
export AKHI_CACHE_DIR="./cache"           # Cache directory
export AKHI_TEMP_DIR="./temp"             # Temporary files directory
export AKHI_GPU_MEMORY_FRACTION="0.9"     # GPU memory usage limit
```

## Data Processing Workflow

### Input Data Sources

The system supports multiple input formats and sources:

#### 1. YouTube Search Queries
```bash
# Single query
python run_pipeline.py --query "islamic finance principles"

# Multiple queries (batch processing)
echo "islamic ethics\nquran tafseer\nhadith collection" > queries.txt
python run_pipeline.py --query-file queries.txt
```

#### 2. Direct YouTube URLs
```bash
# Single URL
python run_pipeline.py --url "https://youtube.com/watch?v=example"

# Multiple URLs
echo "https://youtube.com/watch?v=url1\nhttps://youtube.com/watch?v=url2" > urls.txt
python run_pipeline.py --url-file urls.txt
```

#### 3. Existing Audio Files
```bash
# Process local audio files
python run_pipeline.py --stage transcribe --input-dir ./audio_files
```

#### 4. Raw Text Data
```bash
# Process existing transcripts
python run_pipeline.py --stage format_qlora --input-dir ./transcripts
```

### Data Processing Stages

#### Stage 1: Content Collection & Filtering

**Smart Filtering Criteria**:
- Duration: 5 minutes to 1 hour (configurable)
- Language: Auto-detection with preference settings
- Quality: Video resolution and audio clarity
- Relevance: Islamic content scoring
- Authenticity: Scholar verification (when available)

**Output**: Filtered video metadata and download queue

```json
{
  "video_id": "abc123",
  "title": "Islamic Ethics in Modern Times",
  "duration": 1800,
  "quality_score": 0.85,
  "islamic_relevance": 0.92,
  "language": "en",
  "scholar": "Mufti Menk",
  "download_url": "..."
}
```

#### Stage 2: Audio Extraction & Processing

**Audio Processing Pipeline**:
1. Download video using yt-dlp
2. Extract audio (MP3, 44.1kHz)
3. Noise reduction and normalization
4. Chunk into manageable segments
5. Quality validation

**Configuration**:
```yaml
audio_processing:
  format: "mp3"
  sample_rate: 44100
  channels: 1  # Mono for speech
  bitrate: "128k"
  noise_reduction: true
  normalize_volume: true
  chunk_duration: 30  # seconds
```

#### Stage 3: Transcription with Whisper AI

**Whisper Model Selection**:
- `tiny`: Fastest, lower accuracy
- `base`: Balanced speed/accuracy
- `small`: Good accuracy, reasonable speed
- `medium`: High accuracy (recommended)
- `large-v2`: Highest accuracy, slower

**Transcription Features**:
- Automatic language detection
- Speaker diarization (when multiple speakers)
- Timestamp alignment
- Confidence scoring
- Arabic text support

**Output Format**:
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Assalamu alaikum, today we discuss...",
      "confidence": 0.95,
      "speaker": "speaker_1"
    }
  ],
  "language": "en",
  "duration": 1800.5,
  "word_count": 2500
}
```

#### Stage 4: Content Analysis & Validation

**Islamic Content Validation**:
- Quran verse verification
- Hadith authenticity checking
- Scholar attribution validation
- Topic classification
- Content quality scoring

**Enhanced RAG Checker Process**:
1. Extract Islamic references
2. Query knowledge base (SQLite + FAISS)
3. Calculate semantic similarity
4. Verify authenticity
5. Generate confidence scores

**Quality Metrics**:
```json
{
  "islamic_relevance": 0.92,
  "authenticity_score": 0.88,
  "content_quality": 0.85,
  "language_quality": 0.90,
  "references_verified": 15,
  "references_total": 17,
  "topics": ["ethics", "family", "worship"]
}
```

#### Stage 5: QLoRA Data Formatting

**Supported Formats**:

1. **Alpaca Format** (Recommended):
```json
{
  "instruction": "Explain the Islamic perspective on patience during trials.",
  "input": "",
  "output": "In Islam, patience (sabr) during trials is considered one of the highest virtues..."
}
```

2. **Prompt-Response Format**:
```json
{
  "prompt": "What does Islam teach about forgiveness?",
  "response": "Islam emphasizes forgiveness as a noble quality..."
}
```

3. **Q&A Format**:
```json
{
  "question": "How should Muslims handle financial difficulties?",
  "answer": "Islamic teachings provide guidance for financial hardships..."
}
```

4. **Conversational Format**:
```json
{
  "conversations": [
    {"from": "human", "value": "Tell me about Islamic charity."},
    {"from": "assistant", "value": "Islamic charity, known as Zakat..."}
  ]
}
```

**Data Enhancement**:
- Context preservation
- Metadata inclusion
- Citation formatting
- Arabic transliteration
- Topic tagging

#### Stage 6: Dataset Preparation for Axolotl

**Automatic Processing**:
1. Merge all JSON/JSONL files
2. Normalize data formats
3. Quality filtering and validation
4. Train/validation split
5. Generate Axolotl config.yml
6. Create training scripts
7. Generate dataset statistics

**Output Structure**:
```
axolotl_data/
├── train.jsonl              # Training dataset
├── validation.jsonl         # Validation dataset
├── config.yml              # Axolotl configuration
├── train.sh                # Training script
├── dataset_stats.json      # Dataset statistics
├── README.md               # Documentation
└── logs/
    └── preparation.log     # Processing logs
```

### Data Quality Assurance

#### Validation Checks

1. **Format Validation**:
   - JSON structure integrity
   - Required fields presence
   - Data type consistency

2. **Content Validation**:
   - Minimum/maximum length checks
   - Language detection
   - Islamic content relevance
   - Authenticity verification

3. **Quality Scoring**:
   - Text coherence
   - Grammar and spelling
   - Information completeness
   - Citation accuracy

#### Error Handling

```python
# Example error handling in pipeline
try:
    result = process_video(video_url)
except TranscriptionError as e:
    logger.error(f"Transcription failed: {e}")
    # Retry with different model or skip
except ValidationError as e:
    logger.warning(f"Content validation failed: {e}")
    # Mark for manual review
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Continue with next video
```

## QLoRA Training with Axolotl

### Axolotl Integration

The system automatically generates Axolotl-compatible configurations and datasets:

#### 1. Generated Axolotl Configuration

```yaml
# Auto-generated config.yml
base_model: meta-llama/Llama-2-7b-hf
model_type: LlamaForCausalLM
tokenizer_type: LlamaTokenizer

# Dataset configuration
datasets:
  - path: train.jsonl
    type: alpaca
  - path: validation.jsonl
    type: alpaca
    test_split: 0.1

# LoRA configuration
adapter: lora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.1
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Training parameters
sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

# Optimization
load_in_8bit: false
load_in_4bit: true
strict: false

# Training hyperparameters
micro_batch_size: 4
gradient_accumulation_steps: 4
num_epochs: 3
optimizer: paged_adamw_32bit
lr_scheduler: cosine
learning_rate: 0.0002

# Logging and saving
logging_steps: 1
save_steps: 500
eval_steps: 100
save_total_limit: 3

# Output
output_dir: ./qlora-out
hub_model_id: akhi-islamic-model
```

#### 2. Training Execution

```bash
# Navigate to Axolotl data directory
cd axolotl_data

# Install Axolotl (if not already installed)
pip install axolotl[flash-attn,deepspeed]

# Run training
axolotl train config.yml

# Or use the generated training script
./train.sh
```

#### 3. Training Script (`train.sh`)

```bash
#!/bin/bash
# Auto-generated training script

echo "Starting QLoRA training with Axolotl..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="akhi-islamic-model"

# Check GPU memory
nvidia-smi

# Start training
axolotl train config.yml \
  --deepspeed deepspeed_configs/zero2.json \
  --logging_steps 10 \
  --save_steps 500 \
  --eval_steps 100

echo "Training completed!"
echo "Model saved to: ./qlora-out"
echo "Logs available in: ./logs"
```

### Training Monitoring

#### 1. Real-time Monitoring

```bash
# Monitor training progress
python cli_logger.py --monitor-training --training-dir ./axolotl_data

# View GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f ./axolotl_data/logs/training.log
```

#### 2. Weights & Biases Integration

```bash
# Login to W&B (if not already done)
wandb login

# Training will automatically log to W&B
# View at: https://wandb.ai/your-username/akhi-islamic-model
```

#### 3. Training Metrics

```python
# Check training progress programmatically
import json

with open('./axolotl_data/logs/trainer_state.json', 'r') as f:
    state = json.load(f)
    
print(f"Current epoch: {state['epoch']}")
print(f"Global step: {state['global_step']}")
print(f"Training loss: {state['log_history'][-1]['train_loss']}")
print(f"Learning rate: {state['log_history'][-1]['learning_rate']}")
```

### Model Validation and Testing

#### 1. Automatic Validation

```bash
# Run validation after training
python enhanced_rag_checker.py \
  --validate-model ./axolotl_data/qlora-out \
  --test-prompts-file ./test_prompts.txt \
  --output-report ./validation_report.json
```

#### 2. Interactive Testing

```python
# Test the trained model interactively
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the trained model
model_path = "./axolotl_data/qlora-out"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Test prompts
test_prompts = [
    "What is the concept of zakat in Islam?",
    "Explain the importance of prayer in Islamic practice.",
    "How does Islam view the treatment of parents?"
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response[len(prompt):].strip()}")
    print("-" * 50)
```

## Monitoring & Logging

### Comprehensive Logging System

The `cli_logger.py` provides a complete monitoring and logging solution:

#### 1. Real-time Pipeline Monitoring

```bash
# Start monitoring dashboard
python cli_logger.py --monitor-pipeline

# Monitor specific components
python cli_logger.py --monitor-component transcription
python cli_logger.py --monitor-component qlora_formatting
python cli_logger.py --monitor-component training
```

**Dashboard Features**:
- Real-time progress tracking
- Resource usage (CPU, GPU, Memory)
- Error rate monitoring
- Performance metrics
- Stage completion status

#### 2. Log Analysis and Reporting

```bash
# Analyze logs from last 24 hours
python cli_logger.py --analyze-logs --time-range "last-24h"

# Generate comprehensive report
python cli_logger.py --generate-report --output ./performance_report.html

# Export metrics to JSON
python cli_logger.py --export-metrics --format json --output ./metrics.json
```

#### 3. Error Tracking and Debugging

```bash
# View error summary
python cli_logger.py --error-summary

# Debug specific error
python cli_logger.py --debug-error --error-id "ERR_001"

# View error trends
python cli_logger.py --error-trends --days 7
```

### Log File Structure

```
logs/
├── pipeline/
│   ├── search_download.log     # YouTube search and download
│   ├── transcription.log       # Audio transcription
│   ├── qlora_formatting.log    # Data formatting
│   └── dataset_preparation.log # Axolotl dataset prep
├── training/
│   ├── axolotl_training.log    # Training logs
│   ├── validation.log          # Model validation
│   └── metrics.log             # Performance metrics
├── system/
│   ├── resource_usage.log      # System resources
│   ├── errors.log              # Error tracking
│   └── performance.log         # Performance data
└── aggregated/
    ├── daily_summary.log       # Daily summaries
    └── weekly_report.log       # Weekly reports
```

### Performance Metrics

#### Key Performance Indicators (KPIs)

1. **Pipeline Efficiency**:
   - Videos processed per hour
   - Average processing time per video
   - Success rate by stage
   - Error rate and recovery time

2. **Data Quality**:
   - Islamic content relevance score
   - Transcription accuracy
   - QLoRA format compliance
   - Validation pass rate

3. **Training Performance**:
   - Training loss convergence
   - Validation accuracy
   - Model perplexity
   - Training time per epoch

4. **Resource Utilization**:
   - GPU utilization percentage
   - Memory usage patterns
   - Disk I/O performance
   - Network bandwidth usage

#### Monitoring Alerts

```yaml
# Alert configuration in cli_logger.py
alerts:
  error_rate_threshold: 0.1      # Alert if error rate > 10%
  gpu_memory_threshold: 0.9      # Alert if GPU memory > 90%
  disk_space_threshold: 0.8      # Alert if disk usage > 80%
  training_stall_threshold: 300  # Alert if no progress for 5 min
  
  notification_channels:
    - email: admin@example.com
    - slack: "#akhi-pipeline-alerts"
    - webhook: "https://hooks.slack.com/..."
```

## Production Deployment

### Deployment Options

#### 1. Local API Server

```bash
# Deploy trained model as local API
python -c "
from tools.model_deployer import ModelDeployerTool
deployer = ModelDeployerTool()
deployer._run(
    model_path='./axolotl_data/qlora-out',
    deployment_target='api_server',
    port=8080,
    host='0.0.0.0'
)
"
```

#### 2. Docker Deployment

```dockerfile
# Dockerfile for production deployment
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Expose port
EXPOSE 8080

# Start API server
CMD ["python3", "api_server.py"]
```

```bash
# Build and run Docker container
docker build -t akhi-islamic-model .
docker run -p 8080:8080 --gpus all akhi-islamic-model
```

#### 3. Cloud Deployment (AWS/GCP/Azure)

```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: akhi-islamic-model
spec:
  replicas: 2
  selector:
    matchLabels:
      app: akhi-islamic-model
  template:
    metadata:
      labels:
        app: akhi-islamic-model
    spec:
      containers:
      - name: model-server
        image: akhi-islamic-model:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
```

#### 4. HuggingFace Hub Deployment

```python
# Deploy to HuggingFace Hub
from huggingface_hub import HfApi

api = HfApi()

# Upload model
api.upload_folder(
    folder_path="./axolotl_data/qlora-out",
    repo_id="your-username/akhi-islamic-model",
    repo_type="model",
    private=True
)

print("Model deployed to HuggingFace Hub!")
print("Access at: https://huggingface.co/your-username/akhi-islamic-model")
```

### Production Configuration

#### 1. Environment Setup

```bash
# Production environment variables
export ENVIRONMENT="production"
export LOG_LEVEL="INFO"
export MAX_CONCURRENT_REQUESTS="10"
export MODEL_CACHE_SIZE="2GB"
export API_RATE_LIMIT="100/minute"
export ENABLE_METRICS="true"
export ENABLE_AUTHENTICATION="true"
```

#### 2. Security Configuration

```yaml
# security_config.yaml
security:
  authentication:
    enabled: true
    method: "jwt"  # or "api_key"
    token_expiry: 3600  # 1 hour
  
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst_limit: 20
  
  input_validation:
    max_prompt_length: 1000
    allowed_languages: ["en", "ar"]
    content_filtering: true
  
  logging:
    log_requests: true
    log_responses: false  # Don't log sensitive content
    audit_trail: true
```

#### 3. Monitoring and Health Checks

```python
# health_check.py
import requests
import time

def health_check():
    """Comprehensive health check for production deployment"""
    
    checks = {
        "api_server": check_api_server(),
        "model_loading": check_model_loading(),
        "gpu_availability": check_gpu_availability(),
        "disk_space": check_disk_space(),
        "memory_usage": check_memory_usage()
    }
    
    all_healthy = all(checks.values())
    
    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "checks": checks,
        "timestamp": time.time()
    }

def check_api_server():
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Add other check functions...
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
python prepare_axolotl_dataset.py --batch-size 2  # Instead of 4

# Enable gradient checkpointing
# In config.yml:
gradient_checkpointing: true

# Use smaller model
python prepare_axolotl_dataset.py --base-model "microsoft/DialoGPT-small"

# Enable 4-bit quantization
load_in_4bit: true
```

#### 2. Transcription Failures

**Error**: `WhisperError: Failed to transcribe audio`

**Solutions**:
```bash
# Try different Whisper model
python run_pipeline.py --whisper-model base  # Instead of large-v2

# Check audio quality
ffprobe audio_file.mp3

# Re-encode audio
ffmpeg -i input.mp4 -ar 16000 -ac 1 output.wav

# Skip problematic files
python run_pipeline.py --skip-on-error
```

#### 3. Islamic Content Validation Issues

**Error**: `ValidationError: Content failed Islamic authenticity check`

**Solutions**:
```bash
# Rebuild knowledge base
python enhanced_rag_checker.py --build-knowledge-base --force

# Lower validation threshold
python enhanced_rag_checker.py --threshold 0.6  # Instead of 0.8

# Manual review mode
python enhanced_rag_checker.py --manual-review

# Skip validation for testing
python run_pipeline.py --skip-rag-validation
```

#### 4. Dataset Preparation Errors

**Error**: `DatasetError: Invalid JSON format in training data`

**Solutions**:
```bash
# Validate JSON files
python -c "import json; json.load(open('data.jsonl'))"

# Fix formatting issues
python prepare_axolotl_dataset.py --fix-formatting

# Regenerate dataset
rm -rf ./outputs/*
python run_pipeline.py --stage format_qlora --force-regenerate
```

#### 5. Training Stalls or Crashes

**Error**: Training stops unexpectedly

**Solutions**:
```bash
# Check system resources
nvidia-smi
htop
df -h

# Resume from checkpoint
axolotl train config.yml --resume_from_checkpoint ./qlora-out/checkpoint-500

# Reduce sequence length
# In config.yml:
sequence_len: 1024  # Instead of 2048

# Enable automatic mixed precision
fp16: true
bf16: false
```

### Debug Mode

```bash
# Enable comprehensive debugging
export PYTHONPATH=$PYTHONPATH:.
export LOG_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1

# Run with debug logging
python run_pipeline.py --query "test" --max-videos 1 --verbose --debug

# Check debug logs
tail -f logs/debug.log
```

### Performance Optimization

#### 1. GPU Optimization

```bash
# Check GPU utilization
nvidia-smi dmon -s pucvmet -d 1

# Optimize GPU memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Use tensor cores (if available)
export NVIDIA_TF32_OVERRIDE=1
```

#### 2. CPU Optimization

```bash
# Set optimal thread count
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Enable CPU optimizations
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
```

#### 3. I/O Optimization

```bash
# Use SSD for temporary files
export TMPDIR=/path/to/ssd/temp

# Increase I/O buffer size
export PYTHONUNBUFFERED=1

# Use faster JSON library
pip install orjson
```

## Advanced Features

### 1. Custom Model Architectures

```python
# Custom model configuration
from transformers import AutoConfig, AutoModelForCausalLM

# Modify model architecture
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
config.num_hidden_layers = 24  # Reduce layers for faster training
config.hidden_size = 3072      # Adjust hidden size

model = AutoModelForCausalLM.from_config(config)
```

### 2. Multi-GPU Training

```bash
# Use DeepSpeed for multi-GPU training
deepspeed --num_gpus=4 run_pipeline.py \
  --query "islamic jurisprudence" \
  --stage training \
  --deepspeed-config deepspeed_config.json
```

```json
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  }
}
```

### 3. Custom Data Preprocessing

```python
# Custom preprocessing pipeline
class IslamicContentPreprocessor:
    def __init__(self):
        self.arabic_normalizer = ArabicNormalizer()
        self.citation_formatter = CitationFormatter()
    
    def preprocess_text(self, text):
        # Normalize Arabic text
        text = self.arabic_normalizer.normalize(text)
        
        # Format citations
        text = self.citation_formatter.format_quran_citations(text)
        text = self.citation_formatter.format_hadith_citations(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def generate_instruction_pairs(self, segment):
        # Custom instruction generation logic
        instructions = []
        
        # Extract key concepts
        concepts = self.extract_islamic_concepts(segment['text'])
        
        for concept in concepts:
            instruction = f"Explain the Islamic concept of {concept}"
            output = self.extract_explanation(segment['text'], concept)
            
            if output:
                instructions.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output
                })
        
        return instructions
```

### 4. Advanced RAG Integration

```python
# Enhanced RAG with multiple knowledge sources
class AdvancedIslamicRAG:
    def __init__(self):
        self.quran_db = QuranDatabase()
        self.hadith_db = HadithDatabase()
        self.scholarly_db = ScholarlyDatabase()
        self.faiss_index = FAISSIndex()
    
    def validate_content(self, text, threshold=0.8):
        # Multi-source validation
        quran_score = self.validate_against_quran(text)
        hadith_score = self.validate_against_hadith(text)
        scholarly_score = self.validate_against_scholarly_works(text)
        
        # Weighted average
        total_score = (
            quran_score * 0.4 +
            hadith_score * 0.4 +
            scholarly_score * 0.2
        )
        
        return {
            "is_valid": total_score >= threshold,
            "score": total_score,
            "breakdown": {
                "quran": quran_score,
                "hadith": hadith_score,
                "scholarly": scholarly_score
            }
        }
```

## Best Practices

### 1. Data Quality Management

**Content Curation**:
- Verify scholar credentials and authenticity
- Cross-reference Islamic sources
- Maintain citation accuracy
- Regular content audits

**Quality Metrics**:
```python
# Quality assessment framework
quality_metrics = {
    "islamic_authenticity": 0.9,    # High priority
    "language_fluency": 0.8,       # Important
    "factual_accuracy": 0.95,      # Critical
    "citation_completeness": 0.85,  # Important
    "content_coherence": 0.8       # Important
}

# Minimum thresholds
min_thresholds = {
    "islamic_authenticity": 0.7,
    "language_fluency": 0.6,
    "factual_accuracy": 0.8,
    "citation_completeness": 0.5,
    "content_coherence": 0.6
}
```

### 2. Training Optimization

**Hyperparameter Tuning**:
```yaml
# Recommended hyperparameters for Islamic content
training:
  learning_rate: 2e-4      # Conservative for stability
  batch_size: 4            # Adjust based on GPU memory
  gradient_accumulation: 4  # Effective batch size = 16
  warmup_ratio: 0.1        # 10% of total steps
  weight_decay: 0.01       # L2 regularization
  max_grad_norm: 1.0       # Gradient clipping
  
lora:
  r: 16                    # Balanced performance/memory
  alpha: 32                # 2x rank
  dropout: 0.1             # Regularization
  
optimization:
  optimizer: "paged_adamw_32bit"
  scheduler: "cosine"
  fp16: true               # Mixed precision
  gradient_checkpointing: true
```

### 3. Resource Management

**Memory Optimization**:
```bash
# Monitor memory usage
watch -n 1 'nvidia-smi && free -h'

# Optimize batch size dynamically
python -c "
import torch
max_memory = torch.cuda.get_device_properties(0).total_memory
recommended_batch_size = max_memory // (2 * 1024**3)  # 2GB per sample
print(f'Recommended batch size: {recommended_batch_size}')
"
```

**Storage Management**:
```bash
# Cleanup temporary files
find ./temp -type f -mtime +7 -delete

# Compress old logs
find ./logs -name "*.log" -mtime +30 -exec gzip {} \;

# Monitor disk usage
df -h | grep -E '(Filesystem|/dev/)'
```

### 4. Security and Privacy

**Data Protection**:
```python
# Anonymize sensitive information
def anonymize_content(text):
    # Remove personal identifiers
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b', '[CARD]', text)
    
    return text

# Secure API endpoints
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)
```