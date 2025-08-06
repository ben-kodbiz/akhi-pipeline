# RAG to Axolotl Training Bridge

This document describes the complete pipeline for converting documents uploaded to the RAG system into training data for Axolotl fine-tuning.

## Overview

The RAG to Axolotl bridge consists of three main components:

1. **Document Processing**: RAG system processes uploaded documents
2. **Data Conversion**: Bridge scripts convert RAG data to training format
3. **Model Training**: Axolotl trains models using the converted data

## Architecture

```
📁 RAG System
├── 📄 Documents (PDF, TXT, DOCX, etc.)
├── 🔄 Processing (chunking, embedding)
├── 💾 Storage (config.yaml, embeddings)
└── 🌉 Bridge Scripts
    ├── rag_to_axolotl_bridge.py
    ├── auto_train_from_rag.py
    └── quick_train.py

📁 Axolotl Training
├── 📊 Training Data (JSONL format)
├── ⚙️ Configuration (axolotl_config.yml)
├── 🚀 Training Scripts
└── 🎯 Model Outputs
```

## Quick Start

### 1. Upload Documents to RAG

First, upload your documents to the RAG system:

```bash
# Start the upload server
cd /data/work/dev/akhi_data_builder/RAG
python scripts/api_server.py

# Upload via web interface
# Open http://localhost:8000/upload_interface.html

# Or upload via CLI
python upload_cli.py --file /path/to/document.pdf
```

### 2. Prepare Training Data

Use the quick training script for simple workflows:

```bash
# Prepare dataset only
python scripts/quick_train.py prepare

# Prepare and start training
python scripts/quick_train.py train

# Check training status
python scripts/quick_train.py status
```

### 3. Advanced Usage

For more control, use the full automation script:

```bash
# Prepare dataset only
python scripts/auto_train_from_rag.py --prepare-only

# Use custom model
python scripts/auto_train_from_rag.py --model microsoft/DialoGPT-small --prepare-only

# Prepare and start training
python scripts/auto_train_from_rag.py --train --model microsoft/DialoGPT-medium

# Verbose output
python scripts/auto_train_from_rag.py --prepare-only --verbose
```

## Script Details

### 1. rag_to_axolotl_bridge.py

**Purpose**: Converts RAG-processed documents into training format

**Features**:
- Reads RAG config.yaml to find processed documents
- Generates conversational Q&A pairs from document chunks
- Creates JSONL files compatible with Axolotl
- Supports multiple conversation formats

**Usage**:
```bash
python scripts/rag_to_axolotl_bridge.py --config config.yaml --output training_data
```

**Options**:
- `--config`: RAG configuration file (default: config.yaml)
- `--output`: Output directory for training data
- `--format`: Conversation format (alpaca, chatml, etc.)
- `--max-samples`: Maximum number of training samples
- `--verbose`: Enable detailed logging

### 2. auto_train_from_rag.py

**Purpose**: Complete automation from RAG to trained model

**Features**:
- Orchestrates the entire pipeline
- Handles error recovery and logging
- Supports multiple base models
- Configurable training parameters

**Usage**:
```bash
# Prepare only
python scripts/auto_train_from_rag.py --prepare-only

# Full training
python scripts/auto_train_from_rag.py --train --model microsoft/DialoGPT-medium
```

**Options**:
- `--model`: Base model for fine-tuning
- `--prepare-only`: Only prepare dataset
- `--train`: Start training after preparation
- `--base-dir`: Custom base directory
- `--verbose`: Enable verbose logging

### 3. quick_train.py

**Purpose**: Simplified interface for common tasks

**Features**:
- Simple command-line interface
- Status checking and monitoring
- Quick access to common workflows

**Usage**:
```bash
python scripts/quick_train.py prepare    # Prepare dataset
python scripts/quick_train.py train      # Full pipeline
python scripts/quick_train.py status     # Check progress
```

## Workflow Examples

### Example 1: Basic Document Training

```bash
# 1. Upload documents
cd /data/work/dev/akhi_data_builder/RAG
python upload_cli.py --file my_document.pdf

# 2. Wait for processing (check logs)
tail -f logs/api_server.log

# 3. Prepare training data
python scripts/quick_train.py prepare

# 4. Start training
python scripts/quick_train.py train

# 5. Monitor progress
python scripts/quick_train.py status
```

### Example 2: Custom Model Training

```bash
# Use a specific model
python scripts/auto_train_from_rag.py \
    --model microsoft/DialoGPT-small \
    --train \
    --verbose
```

### Example 3: Batch Document Processing

```bash
# Upload multiple documents
for file in documents/*.pdf; do
    python upload_cli.py --file "$file"
    sleep 10  # Allow processing time
done

# Prepare training data
python scripts/auto_train_from_rag.py --prepare-only
```

## Configuration

### Supported Models

The bridge supports various base models:

- `microsoft/DialoGPT-small` (117M parameters)
- `microsoft/DialoGPT-medium` (345M parameters)
- `microsoft/DialoGPT-large` (762M parameters)
- `facebook/blenderbot-400M-distill`
- `facebook/blenderbot-1B-distill`
- Custom models from Hugging Face

### Training Parameters

Default training configuration:

```yaml
# Automatically generated in axolotl_config.yml
base_model: microsoft/DialoGPT-medium
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

# Training settings
sequence_len: 2048
sample_packing: false
pad_to_sequence_len: true

# LoRA settings
load_in_8bit: true
load_in_4bit: false
strict: false

# Training parameters
num_epochs: 3
learning_rate: 0.0002
train_on_inputs: false
group_by_length: false
```

### Output Structure

After running the pipeline, you'll have:

```
📁 RAG/training_data/
├── 📄 conversations.jsonl      # Training conversations
├── 📄 metadata.json           # Dataset metadata
└── 📄 generation_log.txt       # Generation log

📁 akhi_crewai/axolotl_ready/
├── 📄 axolotl_config.yml       # Axolotl configuration
├── 📄 train_dataset.jsonl     # Training data
├── 📄 val_dataset.jsonl       # Validation data
├── 📄 train_model.sh          # Training script
└── 📁 outputs/                 # Training outputs
    ├── 📁 checkpoint-100/
    ├── 📁 checkpoint-200/
    └── 📄 training.log
```

## Monitoring and Debugging

### Check Training Progress

```bash
# Quick status check
python scripts/quick_train.py status

# Monitor training logs
tail -f ../akhi_crewai/axolotl_ready/outputs/training.log

# Check GPU usage
nvidia-smi
```

### Common Issues

1. **No documents found**:
   - Ensure documents are uploaded and processed
   - Check `config.yaml` for document entries

2. **Training fails to start**:
   - Verify GPU availability
   - Check disk space
   - Review Axolotl configuration

3. **Out of memory errors**:
   - Reduce batch size
   - Use smaller model
   - Enable gradient checkpointing

### Debugging Commands

```bash
# Check RAG configuration
cat config.yaml

# Verify document processing
python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"

# Test bridge conversion
python scripts/rag_to_axolotl_bridge.py --config config.yaml --output test_output --verbose

# Validate training data
head -n 5 training_data/conversations.jsonl
```

## Performance Considerations

### Dataset Size

- **Small datasets** (< 1000 samples): Use DialoGPT-small
- **Medium datasets** (1000-10000 samples): Use DialoGPT-medium
- **Large datasets** (> 10000 samples): Use DialoGPT-large

### Training Time

- **DialoGPT-small**: ~1-2 hours for 1000 samples
- **DialoGPT-medium**: ~3-5 hours for 1000 samples
- **DialoGPT-large**: ~6-10 hours for 1000 samples

### Hardware Requirements

- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB+ GPU memory
- **Storage**: 10GB+ free space

## Integration with Existing Workflow

The bridge integrates seamlessly with existing components:

1. **RAG System**: Uses existing document processing
2. **CrewAI**: Leverages existing Axolotl preparation scripts
3. **Training**: Uses existing training infrastructure

## Future Enhancements

- [ ] Real-time training triggers
- [ ] Multi-GPU training support
- [ ] Custom conversation templates
- [ ] Training metrics dashboard
- [ ] Automated model evaluation
- [ ] Integration with model registry

## Support

For issues or questions:

1. Check the logs in `training_data/generation_log.txt`
2. Review Axolotl documentation
3. Use verbose mode for detailed debugging
4. Check GPU and system resources

---

**Author**: Akhi Data Builder Team  
**Date**: January 2025  
**Version**: 1.0