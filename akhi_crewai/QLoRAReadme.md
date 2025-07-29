# QLoRA Integration Guide for Akhi CrewAI Pipeline

## Overview

This guide provides comprehensive instructions for using QLoRA (Quantized Low-Rank Adaptation) fine-tuning capabilities within the Akhi CrewAI Pipeline. QLoRA enables efficient fine-tuning of large language models on Islamic content with reduced memory requirements.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Quick Start](#quick-start)
5. [Training Workflows](#training-workflows)
6. [CLI Commands](#cli-commands)
7. [Data Preparation](#data-preparation)
8. [Model Training](#model-training)
9. [Model Validation](#model-validation)
10. [Model Deployment](#model-deployment)
11. [Monitoring and Logging](#monitoring-and-logging)
12. [Troubleshooting](#troubleshooting)
13. [Advanced Usage](#advanced-usage)
14. [Best Practices](#best-practices)

## Prerequisites

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended: 8GB+ VRAM)
- 16GB+ RAM
- 50GB+ free disk space

### Dependencies
```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install datasets peft
pip install axolotl
pip install bitsandbytes
```

## Installation

### 1. Clone and Setup
```bash
cd akhi_crewai
pip install -r requirements.txt
```

### 2. Verify QLoRA Integration
```bash
python test_qlora_integration.py
```

Expected output:
```
📊 Test Results: 6/6 tests passed
🎉 All QLoRA integration tests passed!
✅ QLoRA integration is working correctly.
```

## Configuration

### 1. QLoRA Configuration File

Edit `config/qlora_config.yaml`:

```yaml
qlora_training:
  # Model Configuration
  base_model: "microsoft/DialoGPT-medium"
  model_type: "causal_lm"
  
  # QLoRA Parameters
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["q_proj", "v_proj"]
  
  # Training Parameters
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  num_epochs: 3
  warmup_steps: 100
  
  # Quantization
  load_in_4bit: true
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"
  
  # Data Configuration
  max_seq_length: 512
  dataset_text_field: "text"
  
  # Output Configuration
  output_dir: "./models/qlora_output"
  logging_steps: 10
  save_steps: 500
  eval_steps: 500
```

### 2. Crew Configuration

Update `config/crew_config.yaml` to enable QLoRA:

```yaml
qlora_training:
  enabled: true
  auto_format_data: true
  validate_before_training: true
  deploy_after_training: false
```

## Quick Start

### 1. Basic Training Command
```bash
# Train on existing transcript data
python main.py --train data/transcripts
```

### 2. Process and Train Pipeline
```bash
# Search, process, and train in one command
python main.py --process-train "Islamic finance principles"
```

### 3. Check System Status
```bash
python main.py --status
```

## Training Workflows

### Workflow 1: Standard Training

1. **Prepare Data**
   ```bash
   # Ensure transcript files are in data/transcripts/
   ls data/transcripts/
   ```

2. **Format Data for Training**
   ```python
   from tools.qlora_formatter import QLoRAFormatterTool
   
   formatter = QLoRAFormatterTool()
   result = formatter._run(
       transcript_dir="data/transcripts",
       output_file="data/training/islamic_qlora.json"
   )
   print(result)
   ```

3. **Start Training**
   ```bash
   python main.py --train data/training/islamic_qlora.json
   ```

### Workflow 2: Integrated Pipeline Training

1. **Search and Process Content**
   ```bash
   python main.py --search "Quran interpretation" --max-videos 5
   ```

2. **Train on Processed Content**
   ```bash
   python main.py --process-train "Quran interpretation"
   ```

### Workflow 3: Custom Training

1. **Interactive Mode**
   ```bash
   python main.py --interactive
   ```

2. **Select Training Options**
   ```
   > Enter command: train
   > Data source: custom
   > Training file: /path/to/custom_data.json
   > Model: microsoft/DialoGPT-medium
   > Start training? (y/n): y
   ```

## CLI Commands

### Training Commands

```bash
# Basic training
python main.py --train <data_path>

# Training with custom config
python main.py --train <data_path> --config custom_config.yaml

# Process and train pipeline
python main.py --process-train <query>

# Interactive training mode
python main.py --interactive
```

### Status and Monitoring

```bash
# System status
python main.py --status

# Training status
python main.py --status --verbose

# QLoRA specific status
python -c "from akhi_pipeline import AkhiPipelineCrew; crew = AkhiPipelineCrew(); print(crew.get_qlora_status())"
```

### Data Management

```bash
# Search and download content
python main.py --search "Islamic ethics" --max-videos 10

# Process existing content
python main.py --process data/transcripts

# Query trained model
python main.py --query "What is the concept of halal in Islam?"
```

## Data Preparation

### 1. Transcript Format

Transcript files should be in JSON format:

```json
{
  "title": "Islamic Finance Principles",
  "speaker": "Dr. Ahmad",
  "duration": 1800,
  "segments": [
    {
      "start_time": 0,
      "end_time": 30,
      "text": "In Islamic finance, the concept of riba is fundamental..."
    }
  ],
  "metadata": {
    "language": "en",
    "topic": "finance",
    "quality": "high"
  }
}
```

### 2. Training Data Format

QLoRA training data format:

```json
[
  {
    "instruction": "Explain the concept of riba in Islamic finance",
    "input": "",
    "output": "Riba refers to the prohibition of interest in Islamic finance..."
  },
  {
    "instruction": "What are the principles of halal investment?",
    "input": "",
    "output": "Halal investment principles include avoiding riba, gharar, and haram activities..."
  }
]
```

### 3. Data Validation

```python
from tools.qlora_formatter import QLoRAFormatterTool

formatter = QLoRAFormatterTool()
validation = formatter.validate_training_data("data/training/islamic_qlora.json")
print(f"Validation: {validation['is_valid']}")
print(f"Issues: {validation['issues']}")
```

## Model Training

### 1. Training Configuration

Customize training parameters in `config/qlora_config.yaml`:

```yaml
training_args:
  output_dir: "./models/islamic_model"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  fp16: true
  logging_steps: 10
  save_steps: 500
  eval_steps: 500
  warmup_steps: 100
  max_grad_norm: 1.0
```

### 2. Monitor Training Progress

```bash
# View training logs
tail -f logs/qlora_trainer.log

# Check training metrics
python -c "
from tools.axolotl_trainer import AxolotlTrainerTool
trainer = AxolotlTrainerTool()
status = trainer.get_training_status()
print(f'Progress: {status["progress"]}%')
print(f'Loss: {status["current_loss"]}')
"
```

### 3. Training Checkpoints

Training checkpoints are saved in:
- `models/qlora_output/checkpoint-{step}/`
- `models/qlora_output/final_model/`

## Model Validation

### 1. Automatic Validation

```python
from tools.model_validator import ModelValidatorTool

validator = ModelValidatorTool()
results = validator._run(
    model_path="models/qlora_output/final_model",
    test_prompts=[
        "What is the concept of zakat?",
        "Explain Islamic banking principles",
        "What are the five pillars of Islam?"
    ]
)
print(results)
```

### 2. Custom Validation Tests

```python
# Test Islamic knowledge
test_prompts = [
    "What is the difference between halal and haram?",
    "Explain the concept of ijma in Islamic jurisprudence",
    "What are the conditions for valid prayer in Islam?"
]

validation_results = validator.validate_islamic_knowledge(
    model_path="models/qlora_output/final_model",
    test_prompts=test_prompts
)

print(f"Accuracy: {validation_results['accuracy']}%")
print(f"Islamic Content Score: {validation_results['islamic_score']}%")
```

## Model Deployment

### 1. Local Deployment

```python
from tools.model_deployer import ModelDeployerTool

deployer = ModelDeployerTool()
result = deployer._run(
    model_path="models/qlora_output/final_model",
    deployment_target="local",
    port=8080
)
print(result)
```

### 2. API Server Deployment

```bash
# Deploy as API server
python -c "
from tools.model_deployer import ModelDeployerTool
deployer = ModelDeployerTool()
deployer._run(
    model_path='models/qlora_output/final_model',
    deployment_target='api_server',
    port=8080,
    host='0.0.0.0'
)
"
```

### 3. HuggingFace Hub Deployment

```python
# Deploy to HuggingFace Hub
deployer._run(
    model_path="models/qlora_output/final_model",
    deployment_target="huggingface",
    repo_name="your-username/islamic-qa-model",
    private=True
)
```

## Monitoring and Logging

### 1. Training Logs

```bash
# View real-time training logs
tail -f logs/qlora_trainer.log

# View specific log levels
grep "ERROR" logs/qlora_trainer.log
grep "WARNING" logs/qlora_trainer.log
```

### 2. Performance Metrics

```python
# Get training metrics
from tools.axolotl_trainer import AxolotlTrainerTool

trainer = AxolotlTrainerTool()
metrics = trainer.get_training_metrics()

print(f"Training Loss: {metrics['train_loss']}")
print(f"Validation Loss: {metrics['eval_loss']}")
print(f"Learning Rate: {metrics['learning_rate']}")
print(f"Training Steps: {metrics['global_step']}")
```

### 3. System Resources

```bash
# Monitor GPU usage
nvidia-smi

# Monitor system resources
htop

# Check disk usage
df -h
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```yaml
# Reduce batch size in config/qlora_config.yaml
batch_size: 2  # Reduce from 4
gradient_accumulation_steps: 8  # Increase to maintain effective batch size

# Enable gradient checkpointing
gradient_checkpointing: true

# Use smaller model
base_model: "microsoft/DialoGPT-small"  # Instead of medium/large
```

#### 2. Import Errors

**Error**: `ImportError: cannot import name 'QLoRAFormatter'`

**Solutions**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"

# Run integration test
python test_qlora_integration.py
```

#### 3. Training Stalls

**Error**: Training stops or becomes very slow

**Solutions**:
```bash
# Check system resources
nvidia-smi
htop

# Reduce sequence length
# In config/qlora_config.yaml:
max_seq_length: 256  # Reduce from 512

# Enable mixed precision
fp16: true
```

#### 4. Data Format Issues

**Error**: `ValueError: Invalid training data format`

**Solutions**:
```python
# Validate data format
from tools.qlora_formatter import QLoRAFormatterTool
formatter = QLoRAFormatterTool()
validation = formatter.validate_training_data("your_data.json")
print(validation)

# Reformat data
formatter._run(
    transcript_dir="data/transcripts",
    output_file="data/training/fixed_data.json",
    min_segment_words=30,  # Adjust filtering
    filter_islamic_content=True
)
```

### Debug Mode

```bash
# Enable debug logging
export PYTHONPATH=$PYTHONPATH:.
export LOG_LEVEL=DEBUG
python main.py --train data/training/islamic_qlora.json
```

## Advanced Usage

### 1. Custom Model Architecture

```python
# Use custom base model
from transformers import AutoModelForCausalLM, AutoTokenizer

# In your training script
model_name = "microsoft/DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)
```

### 2. Multi-GPU Training

```bash
# Use accelerate for multi-GPU
accelerate config
accelerate launch main.py --train data/training/islamic_qlora.json
```

### 3. Custom LoRA Configuration

```yaml
# Advanced LoRA settings in config/qlora_config.yaml
lora_config:
  r: 32  # Higher rank for more parameters
  alpha: 64
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  bias: "none"
  task_type: "CAUSAL_LM"
```

### 4. Custom Data Processing

```python
# Custom data preprocessing
from tools.qlora_formatter import QLoRAFormatterTool

class CustomQLoRAFormatter(QLoRAFormatterTool):
    def preprocess_text(self, text):
        # Add custom preprocessing
        text = text.replace("Allah (SWT)", "Allah")
        text = text.replace("Prophet (PBUH)", "Prophet Muhammad")
        return text
    
    def generate_instruction_pairs(self, segment):
        # Custom instruction generation
        instructions = []
        # Your custom logic here
        return instructions

# Use custom formatter
custom_formatter = CustomQLoRAFormatter()
result = custom_formatter._run(
    transcript_dir="data/transcripts",
    output_file="data/training/custom_qlora.json"
)
```

## Best Practices

### 1. Data Quality

- **Filter Content**: Use `filter_islamic_content=True` to ensure relevant training data
- **Segment Length**: Keep segments between 50-500 words for optimal training
- **Diversity**: Include various Islamic topics (theology, jurisprudence, history, ethics)
- **Quality Control**: Manually review generated instruction pairs

### 2. Training Optimization

- **Batch Size**: Start with small batch sizes (2-4) and increase gradually
- **Learning Rate**: Use 2e-4 to 5e-4 for QLoRA training
- **Epochs**: 3-5 epochs are usually sufficient for fine-tuning
- **Validation**: Monitor validation loss to prevent overfitting

### 3. Resource Management

- **GPU Memory**: Monitor VRAM usage with `nvidia-smi`
- **Checkpoints**: Save checkpoints every 500 steps
- **Disk Space**: Ensure sufficient space for model checkpoints
- **Backup**: Regularly backup trained models

### 4. Model Evaluation

- **Islamic Knowledge**: Test on diverse Islamic topics
- **Language Quality**: Evaluate fluency and coherence
- **Factual Accuracy**: Verify Islamic facts and references
- **Bias Detection**: Check for potential biases in responses

### 5. Deployment Considerations

- **Model Size**: Consider model size for deployment constraints
- **Inference Speed**: Test inference speed on target hardware
- **API Design**: Design clear API endpoints for model access
- **Security**: Implement proper authentication for model APIs

## Example Workflows

### Complete Training Pipeline

```bash
#!/bin/bash
# complete_training.sh

echo "Starting QLoRA Training Pipeline..."

# 1. Search and download content
python main.py --search "Islamic jurisprudence" --max-videos 10

# 2. Process transcripts
python main.py --process data/transcripts

# 3. Format for training
python -c "
from tools.qlora_formatter import QLoRAFormatterTool
formatter = QLoRAFormatterTool()
result = formatter._run(
    transcript_dir='data/transcripts',
    output_file='data/training/jurisprudence_qlora.json',
    filter_islamic_content=True
)
print(result)
"

# 4. Validate data
python -c "
from tools.qlora_formatter import QLoRAFormatterTool
formatter = QLoRAFormatterTool()
validation = formatter.validate_training_data('data/training/jurisprudence_qlora.json')
print(f'Valid: {validation["is_valid"]}')
"

# 5. Train model
python main.py --train data/training/jurisprudence_qlora.json

# 6. Validate model
python -c "
from tools.model_validator import ModelValidatorTool
validator = ModelValidatorTool()
results = validator._run(
    model_path='models/qlora_output/final_model',
    test_prompts=[
        'What is ijma in Islamic law?',
        'Explain the concept of qiyas',
        'What are the sources of Islamic jurisprudence?'
    ]
)
print(results)
"

# 7. Deploy model
python -c "
from tools.model_deployer import ModelDeployerTool
deployer = ModelDeployerTool()
result = deployer._run(
    model_path='models/qlora_output/final_model',
    deployment_target='local',
    port=8080
)
print(result)
"

echo "Training pipeline completed!"
```

### Interactive Training Session

```python
# interactive_training.py
from akhi_pipeline import AkhiPipelineCrew
from tools.qlora_formatter import QLoRAFormatterTool
from tools.model_validator import ModelValidatorTool

def interactive_training():
    print("🚀 QLoRA Interactive Training Session")
    
    # Initialize components
    crew = AkhiPipelineCrew()
    formatter = QLoRAFormatterTool()
    validator = ModelValidatorTool()
    
    # Get user input
    topic = input("Enter Islamic topic to train on: ")
    max_videos = int(input("Maximum videos to process (default 5): ") or 5)
    
    print(f"\n📥 Searching for content on '{topic}'...")
    
    # Search and process
    search_result = crew.search_youtube_content(topic, max_videos)
    print(f"Found {len(search_result)} videos")
    
    # Format data
    print("\n🔄 Formatting data for training...")
    format_result = formatter._run(
        transcript_dir="data/transcripts",
        output_file=f"data/training/{topic.replace(' ', '_')}_qlora.json",
        filter_islamic_content=True
    )
    print(format_result)
    
    # Confirm training
    confirm = input("\nStart training? (y/n): ")
    if confirm.lower() == 'y':
        print("\n🏋️ Starting training...")
        training_result = crew.execute_qlora_training(
            data_file=f"data/training/{topic.replace(' ', '_')}_qlora.json"
        )
        print(training_result)
        
        # Validate
        print("\n✅ Validating trained model...")
        validation_result = validator._run(
            model_path="models/qlora_output/final_model",
            test_prompts=[
                f"What is {topic}?",
                f"Explain the Islamic perspective on {topic}",
                f"What are the key principles of {topic} in Islam?"
            ]
        )
        print(validation_result)
    
    print("\n🎉 Interactive training session completed!")

if __name__ == "__main__":
    interactive_training()
```

## Support and Resources

### Documentation
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Axolotl Documentation](https://github.com/OpenAccess-AI-Collective/axolotl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

### Community
- GitHub Issues: Report bugs and feature requests
- Discussions: Ask questions and share experiences
- Discord: Real-time community support

### Contributing
- Fork the repository
- Create feature branches
- Submit pull requests
- Follow coding standards

---

**Note**: This guide covers the QLoRA integration in the Akhi CrewAI Pipeline. For general CrewAI usage, refer to the main documentation. For Islamic content guidelines, consult with Islamic scholars and experts.

**Last Updated**: January 2025
**Version**: 1.0.0
**Phase**: 8 - QLoRA Integration Complete