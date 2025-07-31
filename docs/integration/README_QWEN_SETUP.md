# Qwen 3 1.7B Islamic Content Training Setup

## Overview

This directory contains a complete setup for fine-tuning Qwen 3 1.7B (using Qwen 2.5 1.5B Instruct) for Islamic content generation using QLoRA (Quantized Low-Rank Adaptation).

## 🚀 Quick Start

### One-Command Setup
```bash
./setup_qwen_pipeline.sh
```

This script will:
1. Install all dependencies
2. Download the Qwen model
3. Validate your dataset
4. Run QLoRA training
5. Test the trained model

### Manual Setup

1. **Download Model**:
   ```bash
   ./download_qwen_model.sh
   ```

2. **Train Model**:
   ```bash
   ./train_qwen_qlora.sh
   ```

3. **Test Model**:
   ```bash
   ./inference_qwen.py
   ```

## 📁 File Structure

```
axolotl_data/
├── README_QWEN_SETUP.md          # This file
├── axolotl_config_qwen.yml        # Qwen-specific Axolotl configuration
├── setup_qwen_pipeline.sh         # Complete automated setup
├── download_qwen_model.sh         # Model download script
├── train_qwen_qlora.sh           # Training script
├── inference_qwen.py             # Inference and testing script
├── deepspeed_configs/            # DeepSpeed configurations (auto-created)
└── qwen-1.7b-islamic-qlora/     # Training output (created after training)
```

## 🔧 Configuration Details

### Model Specifications
- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Parameters**: ~1.5B (closest to Qwen 3 1.7B)
- **Architecture**: Transformer-based with Qwen-specific optimizations
- **Context Length**: 32,768 tokens
- **Quantization**: 4-bit QLoRA for memory efficiency

### Training Parameters
- **Method**: QLoRA (4-bit quantization)
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.1
- **Learning Rate**: 0.0002 (cosine schedule)
- **Batch Size**: 1 (micro) × 8 (accumulation) = 8 (effective)
- **Epochs**: 3
- **Optimizer**: AdamW with 8-bit quantization

### Hardware Requirements
- **Minimum**: 6GB VRAM (RTX 3060, RTX 4060)
- **Recommended**: 8GB+ VRAM (RTX 3070, RTX 4070)
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB free space

## 📊 Dataset Information

### Current Dataset
- **Source**: Islamic transcripts from video content
- **Format**: Alpaca format (instruction, input, output)
- **Location**: `/data/work/dev/akhi_data_builder/pipeline/output/json/akhi_lora.json`
- **Size**: 31 training examples (as of last generation)

### Dataset Structure
```json
[
  {
    "instruction": "Question about Islamic topic",
    "input": "Additional context (optional)",
    "output": "Islamic response/explanation"
  }
]
```

## 🎯 Usage Examples

### Interactive Chat
```bash
./inference_qwen.py
```

### Single Prompt
```bash
./inference_qwen.py --prompt "What are the five pillars of Islam?"
```

### Batch Testing
```bash
./inference_qwen.py --test
```

### Example Prompts
- "What are the five pillars of Islam?"
- "Explain the concept of Tawhid in Islam."
- "What is the significance of Ramadan?"
- "Describe the importance of prayer in Islam."
- "What are the benefits of reading the Quran?"

## 🔍 Training Process

### Step-by-Step Training

1. **Environment Setup**
   - Install Python dependencies
   - Configure GPU settings
   - Validate dataset

2. **Model Download**
   - Download Qwen 2.5 1.5B Instruct
   - Verify model integrity
   - Test model loading

3. **Training Execution**
   - Load base model with 4-bit quantization
   - Apply LoRA adapters
   - Train for 3 epochs
   - Save adapter weights

4. **Post-Training**
   - Validate trained model
   - Run inference tests
   - Generate sample responses

### Training Monitoring

**Key Metrics**:
- Training loss (should decrease)
- Validation loss (should follow training)
- GPU memory usage (< 6GB with 4-bit)
- Training speed (~1-2 seconds per example)

**Expected Timeline**:
- Setup: 10-15 minutes
- Model download: 5-10 minutes
- Training: 15-30 minutes (31 examples × 3 epochs)
- Testing: 2-5 minutes

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   ```bash
   # Reduce batch size in axolotl_config_qwen.yml
   micro_batch_size: 1  # Already minimal
   gradient_accumulation_steps: 4  # Reduce from 8
   ```

2. **Model Download Fails**
   ```bash
   # Check internet connection and retry
   ./download_qwen_model.sh
   ```

3. **Training Stalls**
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Restart training from checkpoint
   # (Axolotl auto-resumes if interrupted)
   ```

4. **Poor Generation Quality**
   - Increase dataset size (add more examples)
   - Adjust learning rate
   - Train for more epochs

### Performance Optimization

```yaml
# Add to axolotl_config_qwen.yml for better performance
flash_attention: true
gradient_checkpointing: true
fp16: true
tf32: true
```

## 📈 Scaling Strategy

### Dataset Expansion
1. **Current**: 31 examples from 2 transcripts
2. **Phase 1**: 100+ examples from 10 transcripts
3. **Phase 2**: 500+ examples from 50 transcripts
4. **Phase 3**: 1000+ examples with diverse Islamic content

### Model Progression
1. **Start**: Qwen 2.5 1.5B (current)
2. **Upgrade**: Qwen 2.5 3B (better quality)
3. **Production**: Qwen 2.5 7B (full capability)

## 🔗 Integration with CrewAI

### Update Configuration
```yaml
# config/crew_config.yaml
local_llm:
  model_name: "qwen-1.7b-islamic-qlora"
  base_url: "http://localhost:8000/v1"
  api_key: "local"
  temperature: 0.7
  max_tokens: 2048
```

### Serving Options

1. **vLLM Server** (Recommended)
   ```bash
   pip install vllm
   python -m vllm.entrypoints.openai.api_server \
       --model ./qwen-1.7b-islamic-qlora \
       --port 8000
   ```

2. **Direct Integration**
   - Use `inference_qwen.py` as a module
   - Import into CrewAI tools

## 📝 Development Notes

### Model Architecture
- Qwen 2.5 uses RMSNorm instead of LayerNorm
- Rotary Position Embedding (RoPE)
- SwiGLU activation function
- Grouped Query Attention (GQA)

### LoRA Target Modules
```yaml
lora_target_modules:
  - q_proj      # Query projection
  - k_proj      # Key projection
  - v_proj      # Value projection
  - o_proj      # Output projection
  - gate_proj   # Gate projection (SwiGLU)
  - up_proj     # Up projection (SwiGLU)
  - down_proj   # Down projection (SwiGLU)
```

### Special Tokens
- **BOS**: `<|im_start|>`
- **EOS**: `<|im_end|>`
- **UNK**: `<|endoftext|>`

## 🎉 Success Metrics

### Training Success
- [ ] Model downloads without errors
- [ ] Training completes all 3 epochs
- [ ] Loss decreases consistently
- [ ] No OOM errors
- [ ] Adapter files saved correctly

### Quality Success
- [ ] Generates coherent Islamic content
- [ ] Follows instruction format
- [ ] Maintains Islamic accuracy
- [ ] Responds appropriately to prompts

### Integration Success
- [ ] Loads in inference script
- [ ] Integrates with CrewAI
- [ ] Serves via API endpoint
- [ ] Maintains performance standards

## 📚 Additional Resources

- [Axolotl Documentation](https://github.com/OpenAccess-AI-Collective/axolotl)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Qwen Model Documentation](https://huggingface.co/Qwen)
- [PEFT Library](https://github.com/huggingface/peft)

---

**Created**: $(date)
**Status**: Ready for deployment
**Maintainer**: Akhi Data Builder Team
**Purpose**: Islamic content generation and education