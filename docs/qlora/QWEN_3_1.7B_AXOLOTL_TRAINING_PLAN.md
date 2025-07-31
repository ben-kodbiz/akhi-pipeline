# Qwen 3 1.7B Axolotl Training Plan

## Overview
Detailed plan for fine-tuning Qwen 3 1.7B model using Axolotl with QLoRA for Islamic content generation.

## Model Specifications
- **Model**: `Qwen/Qwen2.5-1.5B-Instruct` (closest available to Qwen 3 1.7B)
- **Parameters**: ~1.7B
- **Architecture**: Transformer-based
- **Context Length**: 32,768 tokens
- **Quantization**: 4-bit QLoRA

## Hardware Requirements

### Minimum Requirements
- **GPU**: 6GB VRAM (RTX 3060, RTX 4060)
- **RAM**: 16GB system RAM
- **Storage**: 20GB free space
- **CUDA**: 11.8+ or 12.x

### Recommended Requirements
- **GPU**: 8GB+ VRAM (RTX 3070, RTX 4070)
- **RAM**: 32GB system RAM
- **Storage**: 50GB free space (for model cache and outputs)

## Training Configuration

### Base Axolotl Config (`axolotl_config_qwen.yml`)
```yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct
model_type: Qwen2ForCausalLM
tokenizer_type: AutoTokenizer

# QLoRA Configuration
load_in_8bit: false
load_in_4bit: true
strict: false

# LoRA Parameters
adapter: qlora
lora_model_dir:
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1
lora_target_linear: true
lora_fan_in_fan_out:

# Training Parameters
sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

# Dataset Configuration
datasets:
  - path: /data/work/dev/akhi_data_builder/pipeline/output/json/akhi_lora.json
    type: alpaca

# Training Hyperparameters
num_epochs: 3
micro_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.0002
lr_scheduler: cosine
warmup_steps: 100

# Optimization
optimizer: adamw_bnb_8bit
weight_decay: 0.01
max_grad_norm: 1.0

# Evaluation
eval_sample_packing: false
val_set_size: 0.1
eval_steps: 50
save_steps: 100

# Output Configuration
output_dir: ./qwen-1.7b-islamic-qlora
hub_model_id: qwen-1.7b-islamic-qlora
hub_strategy: every_save

# Logging
logging_steps: 10
wandb_project: qwen-islamic-training
wandb_entity:
wandb_watch:
wandb_name: qwen-1.7b-islamic-qlora
wandb_log_model:

# Special Tokens
special_tokens:
  bos_token: "<|im_start|>"
  eos_token: "<|im_end|>"
  unk_token: "<|endoftext|>"
```

## Training Pipeline Setup

### 1. Environment Preparation
```bash
# Install Axolotl
pip install axolotl[flash-attn,deepspeed]

# Install additional dependencies
pip install bitsandbytes>=0.41.0
pip install transformers>=4.34.0
pip install torch>=2.0.0
```

### 2. Model Download Script
```bash
#!/bin/bash
# download_qwen_model.sh

echo "📥 Downloading Qwen 2.5 1.5B Instruct model..."

# Create models directory
mkdir -p /data/work/dev/akhi_data_builder/models/qwen-2.5-1.5b-instruct

# Download using huggingface-hub
python -c "
import os
from huggingface_hub import snapshot_download

model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
local_dir = '/data/work/dev/akhi_data_builder/models/qwen-2.5-1.5b-instruct'

print(f'Downloading {model_name} to {local_dir}...')
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print('✅ Download complete!')
"
```

### 3. Training Script (`train_qwen_qlora.sh`)
```bash
#!/bin/bash
# train_qwen_qlora.sh

set -e

echo "🚀 Starting Qwen 1.7B QLoRA Training"
echo "===================================="

# Configuration
CONFIG_FILE="axolotl_config_qwen.yml"
OUTPUT_DIR="./qwen-1.7b-islamic-qlora"
DATA_FILE="/data/work/dev/akhi_data_builder/pipeline/output/json/akhi_lora.json"

# Validate dataset
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Dataset not found: $DATA_FILE"
    echo "Please run QLoRA formatter first"
    exit 1
fi

# Count training examples
EXAMPLES=$(python -c "import json; data=json.load(open('$DATA_FILE')); print(len(data))")
echo "📊 Training with $EXAMPLES examples"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start training
echo "🔥 Starting Axolotl training..."
axolotl train "$CONFIG_FILE" \
    --deepspeed deepspeed_configs/zero2.json \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 50

echo "✅ Training completed!"
echo "📁 Model saved to: $OUTPUT_DIR"
```

## Training Monitoring

### Key Metrics to Track
1. **Loss Curves**
   - Training loss should decrease steadily
   - Validation loss should follow training loss
   - Watch for overfitting (val loss increases while train loss decreases)

2. **Learning Rate Schedule**
   - Cosine annealing from 0.0002 to 0
   - Warmup for first 100 steps

3. **Memory Usage**
   - GPU memory should stay under 6GB with 4-bit quantization
   - Monitor for OOM errors

### Expected Training Time
- **31 examples, 3 epochs**: ~15-30 minutes
- **Batch size 1, grad accumulation 8**: Effective batch size of 8
- **RTX 3060 (6GB)**: ~20 minutes
- **RTX 4070 (12GB)**: ~15 minutes

## Model Evaluation

### Validation Strategy
1. **Automatic Split**: 10% validation (3 examples)
2. **Manual Testing**: Islamic Q&A prompts
3. **Perplexity**: Should decrease during training
4. **Generation Quality**: Coherent Islamic content

### Test Prompts
```python
test_prompts = [
    "What are the five pillars of Islam?",
    "Explain the concept of Tawhid in Islam.",
    "What is the significance of Ramadan?",
    "Describe the importance of prayer in Islam."
]
```

## Post-Training Setup

### Model Inference Script
```python
# inference_qwen.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def load_fine_tuned_model(base_model_path, adapter_path):
    """Load fine-tuned Qwen model with LoRA adapter"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    """Generate response using fine-tuned model"""
    
    # Format prompt
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|im_start|>assistant\n")[-1]

if __name__ == "__main__":
    base_model = "/data/work/dev/akhi_data_builder/models/qwen-2.5-1.5b-instruct"
    adapter_path = "./qwen-1.7b-islamic-qlora"
    
    model, tokenizer = load_fine_tuned_model(base_model, adapter_path)
    
    prompt = "What are the five pillars of Islam?"
    response = generate_response(model, tokenizer, prompt)
    print(f"Response: {response}")
```

## Integration with Existing Pipeline

### Update CrewAI Configuration
```yaml
# config/crew_config.yaml
local_llm:
  model_name: "qwen-1.7b-islamic-qlora"
  base_url: "http://localhost:8000/v1"  # If using vLLM
  api_key: "local"
  temperature: 0.7
  max_tokens: 2048
```

### Model Serving Options
1. **vLLM Server** (Recommended)
   ```bash
   pip install vllm
   python -m vllm.entrypoints.openai.api_server \
       --model ./qwen-1.7b-islamic-qlora \
       --port 8000
   ```

2. **Transformers Pipeline**
   - Direct integration with existing tools
   - Lower throughput but simpler setup

3. **GGUF Conversion** (For llama.cpp)
   ```bash
   # Convert to GGUF for llama-cpp-python
   python convert-hf-to-gguf.py ./qwen-1.7b-islamic-qlora
   ```

## Scaling Strategy

### Dataset Expansion
1. **Current**: 31 examples from 2 transcripts
2. **Phase 1**: 100+ examples from 10 transcripts
3. **Phase 2**: 500+ examples from 50 transcripts
4. **Phase 3**: 1000+ examples with diverse Islamic content

### Model Progression
1. **Start**: Qwen 2.5 1.5B (current plan)
2. **Upgrade**: Qwen 2.5 3B (better quality)
3. **Production**: Qwen 2.5 7B (full capability)

## Troubleshooting

### Common Issues
1. **OOM Errors**: Reduce batch size or use gradient checkpointing
2. **Slow Training**: Enable flash attention, use DeepSpeed
3. **Poor Quality**: Increase dataset size, adjust learning rate
4. **Convergence Issues**: Check data quality, adjust warmup steps

### Performance Optimization
```yaml
# Add to config for better performance
flash_attention: true
gradient_checkpointing: true
deepspeed: deepspeed_configs/zero2.json
fp16: true
tf32: true
```

## Next Steps

1. **Immediate**: Set up Qwen 2.5 1.5B training environment
2. **Week 1**: Complete first training run with 31 examples
3. **Week 2**: Expand dataset to 100+ examples
4. **Week 3**: Integrate trained model with CrewAI pipeline
5. **Month 1**: Scale to production-ready model with 1000+ examples

---

**Status**: Ready for implementation
**Estimated Setup Time**: 2-4 hours
**Training Time**: 15-30 minutes per run
**Total Pipeline Time**: 1 day for complete setup