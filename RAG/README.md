# RAG to Axolotl Training System

Complete CLI-based system for converting RAG documents into fine-tuned AI models using Axolotl.

## 📚 Documentation

### 🚀 Quick Start
- **[Quick Reference Card](QUICK_REFERENCE.md)** - Essential commands and troubleshooting
- **[Complete CLI Guide](RAG_TO_AXOLOTL_CLI_GUIDE.md)** - Detailed step-by-step instructions

### 🔧 Tools
- **[Troubleshooting Script](scripts/troubleshoot.py)** - Automated diagnostic tool
- **[Training Scripts](scripts/)** - Conversion and automation tools

## ⚡ Quick Start (5 Minutes)

```bash
# 1. Check system readiness
python scripts/troubleshoot.py

# 2. Start RAG server
python scripts/api_server.py &

# 3. Upload documents (web interface)
# Open: http://localhost:8000/upload_interface.html

# 4. Run complete training pipeline
python scripts/quick_train.py train

# 5. Monitor progress
tail -f ../akhi_crewai/axolotl_ready/training.log
```

## 📋 Step-by-Step Process

### Phase 1: Document Upload
```bash
cd RAG
python scripts/api_server.py                    # Start upload server
python scripts/upload_cli.py --file doc.pdf     # Upload documents
```

### Phase 2: Data Conversion
```bash
python scripts/quick_train.py prepare           # Convert to training format
```

### Phase 3: Model Preparation
```bash
cd ../akhi_crewai
python prepare_axolotl_dataset.py \             # Prepare Axolotl dataset
  --input ../RAG/training_data \
  --output axolotl_ready \
  --model microsoft/DialoGPT-medium
```

### Phase 4: Training
```bash
cd axolotl_ready
./train_model.sh                                 # Start training
```

## 🛠️ Available Scripts

### Core Scripts
- **`scripts/api_server.py`** - Document upload server
- **`scripts/upload_cli.py`** - CLI document upload
- **`scripts/quick_train.py`** - Simple training interface
- **`scripts/auto_train_from_rag.py`** - Advanced automation
- **`scripts/rag_to_axolotl_bridge.py`** - Data conversion
- **`scripts/troubleshoot.py`** - Diagnostic tool

### Usage Examples
```bash
# Quick operations
python scripts/quick_train.py prepare           # Prepare dataset
python scripts/quick_train.py train             # Full training
python scripts/quick_train.py status            # Check status

# Advanced operations
python scripts/auto_train_from_rag.py --prepare-only
python scripts/auto_train_from_rag.py --train --model microsoft/DialoGPT-large

# Troubleshooting
python scripts/troubleshoot.py                  # Full diagnostic
python scripts/troubleshoot.py --check gpu      # Check specific component
python scripts/troubleshoot.py --fix            # Attempt automatic fixes
```

## 🔍 System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 20GB+ free disk space
- **CPU**: Multi-core processor (8+ cores recommended)

### Software
- Python 3.8+
- CUDA 11.8+
- NVIDIA drivers
- Git

### Quick System Check
```bash
python scripts/troubleshoot.py --check system
```

## 🎯 Training Configurations

### Small GPU (8GB VRAM)
```yaml
# axolotl_config.yml
model_type: AutoModelForCausalLM
model_name: microsoft/DialoGPT-medium
load_in_4bit: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
max_steps: 500
```

### Medium GPU (16GB VRAM)
```yaml
# axolotl_config.yml
model_type: AutoModelForCausalLM
model_name: microsoft/DialoGPT-large
load_in_4bit: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
max_steps: 1000
```

### Large GPU (24GB+ VRAM)
```yaml
# axolotl_config.yml
model_type: AutoModelForCausalLM
model_name: microsoft/DialoGPT-large
load_in_8bit: true
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
max_steps: 1500
```

## 🚨 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Solution: Reduce batch size
python scripts/troubleshoot.py --fix memory
# Or manually edit axolotl_config.yml:
# per_device_train_batch_size: 1
# gradient_accumulation_steps: 16
```

#### 2. Tokenization Errors
```bash
# Check data format
python scripts/troubleshoot.py --check data
# Fix automatically
python scripts/troubleshoot.py --fix tokenization
```

#### 3. Model Loading Issues
```bash
# Check model availability
python scripts/troubleshoot.py --check model
# Download missing models
python scripts/troubleshoot.py --fix model
```

#### 4. Training Stuck/Slow
```bash
# Check training status
python scripts/quick_train.py status
# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Emergency Commands
```bash
# Stop all training
pkill -f train_model.sh
pkill -f python.*axolotl

# Clean up corrupted data
rm -rf ../akhi_crewai/axolotl_ready/models/checkpoint-*
rm -rf training_data/*.json

# Reset and restart
python scripts/troubleshoot.py --reset
python scripts/quick_train.py train
```

## 📁 Directory Structure

```
RAG/
├── README.md                           # This guide
├── RAG_TO_AXOLOTL_CLI_GUIDE.md        # Detailed CLI guide
├── QUICK_REFERENCE.md                  # Quick reference card
├── scripts/
│   ├── troubleshoot.py                 # Diagnostic tool
│   ├── quick_train.py                  # Simple training interface
│   ├── auto_train_from_rag.py         # Advanced automation
│   ├── rag_to_axolotl_bridge.py       # Data conversion
│   ├── api_server.py                   # Upload server
│   └── upload_cli.py                   # CLI upload tool
├── training_data/                      # Converted training data
├── uploaded_documents/                 # Raw uploaded documents
└── logs/                              # System logs

../akhi_crewai/
├── prepare_axolotl_dataset.py         # Dataset preparation
└── axolotl_ready/                     # Training environment
    ├── axolotl_config.yml             # Training configuration
    ├── train_model.sh                 # Training script
    ├── models/                        # Model checkpoints
    └── training.log                   # Training logs
```

## ✅ Success Indicators

### Training Progress
- **Loss decreasing**: From ~6.0 to ~2.8-3.0
- **Checkpoints created**: `checkpoint-100`, `checkpoint-200`, etc.
- **GPU utilization**: 80-95% during training
- **No error messages**: In training logs

### Validation Commands
```bash
# Check training progress
tail -f ../akhi_crewai/axolotl_ready/training.log | grep "train_loss"

# Verify checkpoints
ls -la ../akhi_crewai/axolotl_ready/models/

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check system resources
python scripts/troubleshoot.py --check resources
```

## 📖 Related Documentation

- **[Axolotl Documentation](https://github.com/OpenAccess-AI-Collective/axolotl)**
- **[QLoRA Paper](https://arxiv.org/abs/2305.14314)**
- **[Transformers Library](https://huggingface.co/docs/transformers)**
- **[PEFT Documentation](https://huggingface.co/docs/peft)**

## 💡 Pro Tips

1. **Start Small**: Begin with DialoGPT-medium before trying larger models
2. **Monitor Resources**: Keep an eye on GPU memory and temperature
3. **Save Checkpoints**: Training can take hours, checkpoints are crucial
4. **Use Troubleshoot Script**: Run diagnostics before and during training
5. **Check Logs**: Training logs contain valuable debugging information

## 🆘 Need Help?

1. **Run Diagnostics**: `python scripts/troubleshoot.py`
2. **Check Quick Reference**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
3. **Review Logs**: `tail -f logs/system.log`
4. **System Status**: `python scripts/quick_train.py status`

---

**Last Updated**: December 2024  
**Version**: 2.0  
**Compatibility**: Axolotl 0.4+, Transformers 4.36+, PEFT 0.7+