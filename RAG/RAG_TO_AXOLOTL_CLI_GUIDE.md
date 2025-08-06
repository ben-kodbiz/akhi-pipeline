# Complete CLI Guide: RAG to Axolotl Model Training

A comprehensive step-by-step guide for converting RAG documents into fine-tuned AI models using Axolotl.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Setup](#system-setup)
3. [Step-by-Step Training Process](#step-by-step-training-process)
4. [Advanced Configuration](#advanced-configuration)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Performance Optimization](#performance-optimization)
7. [Monitoring and Validation](#monitoring-and-validation)

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 20GB+ free disk space
- **CPU**: Multi-core processor (8+ cores recommended)

### Software Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- Git
- pip or conda package manager

### Check Your System
```bash
# Check GPU availability
nvidia-smi

# Check Python version
python --version

# Check CUDA version
nvcc --version

# Check available disk space
df -h
```

## System Setup

### 1. Navigate to Project Directory
```bash
cd /data/work/dev/akhi_data_builder
```

### 2. Verify RAG System
```bash
# Check if RAG directory exists
ls -la RAG/

# Verify required scripts
ls -la RAG/scripts/
```

### 3. Install Dependencies
```bash
# Install Python dependencies
pip install -r RAG/requirements.txt

# Install additional training dependencies
pip install torch transformers peft datasets accelerate
```

## Step-by-Step Training Process

### Phase 1: Document Upload and Processing

#### 1.1 Start RAG Upload Server
```bash
# Navigate to RAG directory
cd RAG

# Start the upload server
python scripts/api_server.py
```

**Expected Output:**
```
 * Running on http://127.0.0.1:8000
 * Debug mode: off
```

#### 1.2 Upload Documents

**Option A: Web Interface**
```bash
# Open in browser
http://localhost:8000/upload_interface.html
```

**Option B: CLI Upload**
```bash
# Upload single document
python scripts/upload_cli.py --file /path/to/document.pdf

# Upload multiple documents
python scripts/upload_cli.py --directory /path/to/documents/
```

#### 1.3 Verify Document Processing
```bash
# Check uploaded documents
ls -la uploads/

# Verify RAG configuration
cat config.yaml

# Check document processing status
python -c "import yaml; config = yaml.safe_load(open('config.yaml')); print(f'Documents: {len(config.get(\"data\", {}).get(\"sources\", []))}')"
```

### Phase 2: Convert RAG Data to Training Format

#### 2.1 Quick Conversion (Recommended for Beginners)
```bash
# Simple dataset preparation
python scripts/quick_train.py prepare
```

**Expected Output:**
```
🤖 Quick Training - Preparing dataset...
✅ RAG documents converted to training format
✅ Training data saved to: training_data/rag_training_data.jsonl
✅ Statistics saved to: training_data/conversion_stats.json
📊 Generated 1,234 conversations from 15 documents
```

#### 2.2 Advanced Conversion (Custom Parameters)
```bash
# Convert with custom settings
python scripts/rag_to_axolotl_bridge.py \
  --config config.yaml \
  --output training_data \
  --format conversation \
  --max-samples 5000 \
  --verbose
```

#### 2.3 Verify Training Data
```bash
# Check generated files
ls -la training_data/

# Preview training data
head -n 3 training_data/rag_training_data.jsonl

# Check conversion statistics
cat training_data/conversion_stats.json
```

### Phase 3: Prepare Axolotl Dataset

#### 3.1 Navigate to Training Directory
```bash
# Go to akhi_crewai directory
cd ../akhi_crewai
```

#### 3.2 Prepare Axolotl Dataset
```bash
# Basic preparation with default model
python prepare_axolotl_dataset.py \
  --input ../RAG/training_data \
  --output ./axolotl_ready \
  --model microsoft/DialoGPT-medium
```

**Advanced Options:**
```bash
# Custom model and parameters
python prepare_axolotl_dataset.py \
  --input ../RAG/training_data \
  --output ./axolotl_ready \
  --model microsoft/DialoGPT-large \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-5 \
  --max-length 512
```

#### 3.3 Verify Axolotl Preparation
```bash
# Check prepared files
ls -la axolotl_ready/

# Verify configuration
cat axolotl_ready/axolotl_config.yml

# Check dataset files
wc -l axolotl_ready/train_dataset.jsonl
wc -l axolotl_ready/val_dataset.jsonl
```

### Phase 4: Start Model Training

#### 4.1 Navigate to Training Directory
```bash
cd axolotl_ready
```

#### 4.2 Start Training
```bash
# Make training script executable
chmod +x train_model.sh

# Start training (this will run for several hours)
./train_model.sh
```

#### 4.3 Monitor Training Progress

**In a new terminal:**
```bash
# Monitor training logs
tail -f /data/work/dev/akhi_data_builder/akhi_crewai/axolotl_ready/training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Monitor disk usage
watch -n 30 'df -h | grep -E "(Filesystem|/data)"'
```

#### 4.4 Check Training Checkpoints
```bash
# List model checkpoints
ls -la models/

# Check latest checkpoint
ls -la models/checkpoint-*/
```

## Advanced Configuration

### Custom Training Parameters

#### Modify Axolotl Configuration
```bash
# Edit configuration file
nano axolotl_config.yml
```

**Key Parameters to Adjust:**
```yaml
# Model settings
base_model: microsoft/DialoGPT-medium  # or DialoGPT-small/large
model_type: AutoModelForCausalLM

# Training parameters
num_epochs: 3                    # Number of training epochs
learning_rate: 0.0002           # Learning rate
per_device_train_batch_size: 2  # Batch size per GPU
gradient_accumulation_steps: 4   # Gradient accumulation

# LoRA settings
lora_r: 16                      # LoRA rank
lora_alpha: 32                  # LoRA alpha
lora_dropout: 0.1               # LoRA dropout

# Sequence length
sequence_len: 512               # Maximum sequence length
```

### Custom Data Processing

#### Modify Conversion Parameters
```bash
# Custom Q&A generation
python scripts/rag_to_axolotl_bridge.py \
  --config config.yaml \
  --output training_data \
  --format chatml \
  --max-samples 10000 \
  --min-chunk-length 50 \
  --max-chunk-length 1000 \
  --verbose
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. No Documents Found

**Error:**
```
ValueError: No documents found in RAG system
```

**Solutions:**
```bash
# Check if documents are uploaded
ls -la RAG/uploads/

# Verify config.yaml has document entries
cat RAG/config.yaml

# Re-upload documents
cd RAG
python scripts/upload_cli.py --file /path/to/document.pdf

# Force document processing
python scripts/document_loader.py --config config.yaml --force-reload
```

#### 2. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce batch size in axolotl_config.yml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8

# Use smaller model
base_model: microsoft/DialoGPT-small

# Reduce sequence length
sequence_len: 256

# Enable gradient checkpointing
gradient_checkpointing: true
```

#### 3. Training Script Fails

**Error:**
```
ModuleNotFoundError: No module named 'axolotl'
```

**Solutions:**
```bash
# Install missing dependencies
pip install torch transformers peft datasets accelerate

# Check Python environment
which python
pip list | grep -E "(torch|transformers|peft)"

# Reinstall dependencies
pip install --upgrade torch transformers peft
```

#### 4. Dataset Format Errors

**Error:**
```
ValueError: Unable to create tensor, you should probably activate truncation
```

**Solutions:**
```bash
# Regenerate dataset with proper formatting
cd RAG
python scripts/rag_to_axolotl_bridge.py \
  --config config.yaml \
  --output training_data_fixed \
  --format conversation \
  --max-chunk-length 512 \
  --verbose

# Re-prepare Axolotl dataset
cd ../akhi_crewai
python prepare_axolotl_dataset.py \
  --input ../RAG/training_data_fixed \
  --output ./axolotl_ready_fixed \
  --model microsoft/DialoGPT-medium
```

#### 5. Disk Space Issues

**Error:**
```
OSError: [Errno 28] No space left on device
```

**Solutions:**
```bash
# Check disk usage
df -h
du -sh /data/work/dev/akhi_data_builder/*

# Clean up old checkpoints
rm -rf axolotl_ready/models/checkpoint-*/

# Clean up temporary files
find . -name "*.tmp" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

### Debugging Commands

#### System Diagnostics
```bash
# Check GPU status
nvidia-smi

# Check Python environment
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available__}'); print(f'GPU count: {torch.cuda.device_count__}')"

# Check memory usage
free -h

# Check process status
ps aux | grep python
```

#### Data Validation
```bash
# Validate RAG configuration
python -c "import yaml; config = yaml.safe_load(open('RAG/config.yaml')); print(yaml.dump(config, default_flow_style=False))"

# Check training data format
python -c "import json; data = [json.loads(line) for line in open('RAG/training_data/rag_training_data.jsonl')[:5]]; print(json.dumps(data, indent=2))"

# Validate Axolotl config
python -c "import yaml; config = yaml.safe_load(open('akhi_crewai/axolotl_ready/axolotl_config.yml')); print(yaml.dump(config, default_flow_style=False))"
```

## Performance Optimization

### Hardware Optimization

#### GPU Settings
```bash
# Set GPU memory growth (add to training script)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### Memory Management
```bash
# Clear GPU cache before training
python -c "import torch; torch.cuda.empty_cache()"

# Monitor GPU memory during training
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits'
```

### Training Optimization

#### Batch Size Tuning
```yaml
# For 8GB GPU
per_device_train_batch_size: 1
gradient_accumulation_steps: 8

# For 16GB GPU
per_device_train_batch_size: 2
gradient_accumulation_steps: 4

# For 24GB+ GPU
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
```

#### Learning Rate Scheduling
```yaml
# Add to axolotl_config.yml
learning_rate: 0.0002
lr_scheduler: cosine
warmup_steps: 100
weight_decay: 0.01
```

## Monitoring and Validation

### Training Metrics

#### Monitor Loss
```bash
# Extract loss values from logs
grep "'loss':" training.log | tail -20

# Plot training progress (requires matplotlib)
python -c "
import re
import matplotlib.pyplot as plt
with open('training.log', 'r') as f:
    losses = [float(re.search(r\"'loss': ([0-9.]+)\", line).group(1)) for line in f if \"'loss':" in line]
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('training_loss.png')
print('Loss plot saved to training_loss.png')
"
```

#### Check Model Checkpoints
```bash
# List all checkpoints
ls -la models/checkpoint-*/

# Check checkpoint sizes
du -sh models/checkpoint-*/

# Verify latest checkpoint
ls -la models/checkpoint-$(ls models/ | grep checkpoint | sort -V | tail -1)/
```

### Model Validation

#### Quick Model Test
```bash
# Test the trained model
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model = 'microsoft/DialoGPT-medium'
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, 'models/checkpoint-XXXX')  # Replace XXXX with latest checkpoint

# Test generation
input_text = 'What is the importance of prayer in Islam?'
inputs = tokenizer.encode(input_text, return_tensors='pt')
outputs = model.generate(inputs, max_length=150, num_return_sequences=1, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Input: {input_text}')
print(f'Response: {response[len(input_text):]}')
"
```

### Final Model Export

#### Save Complete Model
```bash
# Merge LoRA weights with base model
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load and merge
base_model = 'microsoft/DialoGPT-medium'
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, 'models/checkpoint-XXXX')  # Latest checkpoint
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained('final_model')
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained('final_model')
print('Final model saved to: final_model/')
"
```

## Complete Workflow Summary

### Quick Start (Automated)
```bash
# 1. Start RAG server and upload documents
cd RAG
python scripts/api_server.py &
# Upload documents via web interface: http://localhost:8000/upload_interface.html

# 2. Run automated pipeline
python scripts/quick_train.py train

# 3. Monitor training
tail -f ../akhi_crewai/axolotl_ready/training.log
```

### Manual Control (Step-by-step)
```bash
# 1. Upload documents
cd RAG
python scripts/upload_cli.py --file document.pdf

# 2. Convert to training format
python scripts/rag_to_axolotl_bridge.py --output training_data

# 3. Prepare Axolotl dataset
cd ../akhi_crewai
python prepare_axolotl_dataset.py --input ../RAG/training_data --output axolotl_ready

# 4. Start training
cd axolotl_ready
./train_model.sh
```

## Support and Resources

### Log Files
- **RAG Processing**: `RAG/training_data/generation_log.txt`
- **Axolotl Training**: `akhi_crewai/axolotl_ready/training.log`
- **System Logs**: Check terminal output

### Useful Commands
```bash
# Check training status
python RAG/scripts/quick_train.py status

# Stop training if needed
pkill -f train_model.sh

# Resume from checkpoint
# Edit axolotl_config.yml to set resume_from_checkpoint: path/to/checkpoint
```

### Getting Help
1. Check log files for error messages
2. Verify system requirements
3. Ensure sufficient disk space and memory
4. Test with smaller datasets first
5. Use verbose mode for detailed debugging

---

**Author**: Akhi Data Builder Team  
**Date**: January 2025  
**Version**: 2.0

*This guide provides complete instructions for converting RAG documents into fine-tuned AI models using the Axolotl framework. Follow the steps carefully and refer to the troubleshooting section for common issues.*