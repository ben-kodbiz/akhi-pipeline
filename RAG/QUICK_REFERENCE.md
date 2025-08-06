# RAG to Axolotl - Quick Reference Card

## Essential Commands

### 🚀 Quick Start (Automated)
```bash
# 1. Start RAG server
cd RAG && python scripts/api_server.py &

# 2. Upload documents (web interface)
# Open: http://localhost:8000/upload_interface.html

# 3. Run complete pipeline
python scripts/quick_train.py train

# 4. Monitor progress
tail -f ../akhi_crewai/axolotl_ready/training.log
```

### 📋 Step-by-Step Commands

#### Phase 1: Document Upload
```bash
cd RAG
python scripts/api_server.py                    # Start server
python scripts/upload_cli.py --file doc.pdf     # Upload via CLI
```

#### Phase 2: Data Conversion
```bash
python scripts/quick_train.py prepare           # Quick conversion
# OR
python scripts/rag_to_axolotl_bridge.py \       # Advanced conversion
  --output training_data --verbose
```

#### Phase 3: Axolotl Preparation
```bash
cd ../akhi_crewai
python prepare_axolotl_dataset.py \             # Prepare dataset
  --input ../RAG/training_data \
  --output axolotl_ready \
  --model microsoft/DialoGPT-medium
```

#### Phase 4: Training
```bash
cd axolotl_ready
chmod +x train_model.sh
./train_model.sh                                 # Start training
```

## 🔧 Troubleshooting Commands

### System Checks
```bash
nvidia-smi                                       # Check GPU
df -h                                           # Check disk space
free -h                                         # Check memory
ps aux | grep python                            # Check processes
```

### Data Validation
```bash
ls -la RAG/uploads/                             # Check uploads
cat RAG/config.yaml                             # Check RAG config
wc -l RAG/training_data/*.jsonl                 # Count training samples
ls -la akhi_crewai/axolotl_ready/models/        # Check checkpoints
```

### Common Fixes
```bash
# CUDA out of memory - reduce batch size
nano axolotl_ready/axolotl_config.yml
# Set: per_device_train_batch_size: 1

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Restart training from checkpoint
# Edit axolotl_config.yml: resume_from_checkpoint: path/to/checkpoint
```

## 📊 Monitoring Commands

### Training Progress
```bash
tail -f akhi_crewai/axolotl_ready/training.log   # Live logs
grep "'loss':" training.log | tail -10           # Recent loss values
watch -n 1 nvidia-smi                           # GPU monitoring
```

### Status Checks
```bash
python RAG/scripts/quick_train.py status        # Training status
ls -la akhi_crewai/axolotl_ready/models/        # List checkpoints
du -sh akhi_crewai/axolotl_ready/models/*       # Checkpoint sizes
```

## ⚙️ Configuration Templates

### Small GPU (8GB)
```yaml
# axolotl_config.yml
base_model: microsoft/DialoGPT-small
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
sequence_len: 256
gradient_checkpointing: true
```

### Medium GPU (16GB)
```yaml
# axolotl_config.yml
base_model: microsoft/DialoGPT-medium
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
sequence_len: 512
```

### Large GPU (24GB+)
```yaml
# axolotl_config.yml
base_model: microsoft/DialoGPT-large
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
sequence_len: 512
```

## 🆘 Emergency Commands

### Stop Training
```bash
pkill -f train_model.sh                         # Stop training
pkill -f python                                 # Stop all Python processes
```

### Clean Up
```bash
rm -rf axolotl_ready/models/checkpoint-*        # Remove checkpoints
find . -name "*.tmp" -delete                    # Clean temp files
find . -name "__pycache__" -type d -exec rm -rf {} + # Clean cache
```

### Reset Everything
```bash
rm -rf RAG/training_data/                       # Remove training data
rm -rf akhi_crewai/axolotl_ready/               # Remove prepared dataset
# Then restart from Phase 2
```

## 📁 Important File Locations

```
/data/work/dev/akhi_data_builder/
├── RAG/
│   ├── uploads/                    # Uploaded documents
│   ├── config.yaml                 # RAG configuration
│   ├── training_data/              # Converted training data
│   └── scripts/                    # Conversion scripts
└── akhi_crewai/
    └── axolotl_ready/              # Training environment
        ├── axolotl_config.yml      # Training configuration
        ├── train_dataset.jsonl     # Training data
        ├── val_dataset.jsonl       # Validation data
        ├── train_model.sh          # Training script
        └── models/                 # Model checkpoints
```

## 🎯 Success Indicators

### ✅ Everything Working
- RAG server responds at http://localhost:8000
- Documents appear in `RAG/uploads/`
- Training data generated in `RAG/training_data/`
- Axolotl config created in `akhi_crewai/axolotl_ready/`
- Training loss decreasing over time
- Checkpoints saved in `models/` directory

### ❌ Common Error Patterns
- `CUDA out of memory` → Reduce batch size
- `No documents found` → Check uploads and config.yaml
- `ModuleNotFoundError` → Install missing dependencies
- `No space left` → Clean up old files
- `Training stuck` → Check GPU availability

---

**💡 Pro Tips:**
- Always start with small datasets for testing
- Monitor GPU memory usage during training
- Keep backups of working configurations
- Use verbose mode when debugging
- Check logs first when something fails

**📖 Full Guide:** See `RAG_TO_AXOLOTL_CLI_GUIDE.md` for detailed instructions