#!/usr/bin/env python3
"""
Axolotl Dataset Preparation Script
Merges QLoRA JSONs and auto-generates Axolotl config.yml

Usage:
    python prepare_axolotl_dataset.py --input ./output/jsonl --output ./axolotl_ready
    python prepare_axolotl_dataset.py --input ./output/jsonl --model microsoft/DialoGPT-medium
    python prepare_axolotl_dataset.py --config custom_axolotl_config.yaml

Author: Akhi CrewAI Team
Date: January 2025
Version: 1.0
"""

import os
import sys
import json
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import glob
import random
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AxolotlDatasetPreparer:
    """Prepares QLoRA datasets for Axolotl training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.input_dir = Path(config['input_dir'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset statistics
        self.stats = {
            'total_files': 0,
            'total_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'avg_input_length': 0,
            'avg_output_length': 0,
            'format_distribution': Counter(),
            'quality_scores': []
        }
        
        logger.info(f"Initialized Axolotl dataset preparer")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def discover_json_files(self) -> List[Path]:
        """Discover all JSON/JSONL files in input directory"""
        patterns = ['*.json', '*.jsonl']
        files = []
        
        for pattern in patterns:
            files.extend(self.input_dir.glob(pattern))
            files.extend(self.input_dir.glob(f"**/{pattern}"))
        
        files = list(set(files))  # Remove duplicates
        logger.info(f"Discovered {len(files)} JSON/JSONL files")
        
        return files
    
    def load_samples_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load samples from a single JSON/JSONL file"""
        samples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    # JSONL format - one JSON object per line
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                sample = json.loads(line)
                                samples.append(sample)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                else:
                    # JSON format
                    data = json.load(f)
                    if isinstance(data, list):
                        samples.extend(data)
                    elif isinstance(data, dict):
                        samples.append(data)
                    else:
                        logger.warning(f"Unexpected data type in {file_path}: {type(data)}")
            
            logger.info(f"Loaded {len(samples)} samples from {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
        
        return samples
    
    def normalize_sample_format(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize sample to standard Axolotl format"""
        try:
            # Detect and convert different formats
            if 'conversations' in sample:
                # Already in conversations format
                return sample
            
            elif 'instruction' in sample and 'output' in sample:
                # Alpaca format
                conversations = [
                    {"role": "user", "content": sample['instruction']},
                    {"role": "assistant", "content": sample['output']}
                ]
                
                # Add input if present
                if sample.get('input'):
                    conversations[0]['content'] = f"{sample['instruction']}\n\nInput: {sample['input']}"
                
                return {
                    "conversations": conversations,
                    "source": sample.get('source', 'unknown')
                }
            
            elif 'prompt' in sample and 'response' in sample:
                # Simple prompt-response format
                return {
                    "conversations": [
                        {"role": "user", "content": sample['prompt']},
                        {"role": "assistant", "content": sample['response']}
                    ],
                    "source": sample.get('source', 'unknown')
                }
            
            elif 'question' in sample and 'answer' in sample:
                # Q&A format
                return {
                    "conversations": [
                        {"role": "user", "content": sample['question']},
                        {"role": "assistant", "content": sample['answer']}
                    ],
                    "source": sample.get('source', 'unknown')
                }
            
            elif 'text' in sample:
                # Raw text - create instruction to continue
                text = sample['text']
                if len(text) > 100:
                    # Split into prompt and continuation
                    split_point = len(text) // 2
                    # Find a good split point (sentence boundary)
                    for i in range(split_point - 50, split_point + 50):
                        if i < len(text) and text[i] in '.!?':
                            split_point = i + 1
                            break
                    
                    prompt = text[:split_point].strip()
                    response = text[split_point:].strip()
                    
                    if prompt and response:
                        return {
                            "conversations": [
                                {"role": "user", "content": f"Please continue this Islamic teaching: {prompt}"},
                                {"role": "assistant", "content": response}
                            ],
                            "source": sample.get('source', 'unknown')
                        }
            
            else:
                logger.warning(f"Unknown sample format: {list(sample.keys())}")
                return None
        
        except Exception as e:
            logger.error(f"Error normalizing sample: {e}")
            return None
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate sample quality and format"""
        try:
            if 'conversations' not in sample:
                return False
            
            conversations = sample['conversations']
            if not isinstance(conversations, list) or len(conversations) < 2:
                return False
            
            # Check for required roles
            roles = [conv.get('role') for conv in conversations]
            if 'user' not in roles or 'assistant' not in roles:
                return False
            
            # Check content length
            for conv in conversations:
                content = conv.get('content', '')
                if not content or len(content.strip()) < 10:
                    return False
                
                # Check for reasonable length limits
                if len(content) > self.config.get('max_content_length', 4096):
                    return False
            
            # Islamic content validation
            if self.config.get('validate_islamic_content', True):
                full_text = ' '.join([conv.get('content', '') for conv in conversations]).lower()
                islamic_keywords = self.config.get('islamic_keywords', [
                    'allah', 'prophet', 'quran', 'hadith', 'islam', 'muslim',
                    'salah', 'prayer', 'zakat', 'hajj', 'ramadan', 'fasting',
                    'tawheed', 'fiqh', 'sunnah', 'ummah', 'dua', 'mosque'
                ])
                
                # Require at least one Islamic keyword
                if not any(keyword in full_text for keyword in islamic_keywords):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating sample: {e}")
            return False
    
    def calculate_sample_stats(self, sample: Dict[str, Any]):
        """Calculate statistics for a sample"""
        try:
            conversations = sample.get('conversations', [])
            
            input_lengths = []
            output_lengths = []
            
            for conv in conversations:
                content = conv.get('content', '')
                if conv.get('role') == 'user':
                    input_lengths.append(len(content))
                elif conv.get('role') == 'assistant':
                    output_lengths.append(len(content))
            
            # Update running statistics
            if input_lengths:
                self.stats['avg_input_length'] = (
                    (self.stats['avg_input_length'] * self.stats['total_samples'] + sum(input_lengths)) /
                    (self.stats['total_samples'] + 1)
                )
            
            if output_lengths:
                self.stats['avg_output_length'] = (
                    (self.stats['avg_output_length'] * self.stats['total_samples'] + sum(output_lengths)) /
                    (self.stats['total_samples'] + 1)
                )
            
            # Track format
            source = sample.get('source', 'unknown')
            self.stats['format_distribution'][source] += 1
            
        except Exception as e:
            logger.error(f"Error calculating stats: {e}")
    
    def prepare_dataset(self) -> Dict[str, Any]:
        """Main method to prepare the dataset"""
        logger.info("Starting dataset preparation...")
        
        # Discover files
        json_files = self.discover_json_files()
        if not json_files:
            raise ValueError(f"No JSON/JSONL files found in {self.input_dir}")
        
        self.stats['total_files'] = len(json_files)
        
        # Load and process all samples
        all_samples = []
        
        for file_path in json_files:
            logger.info(f"Processing {file_path.name}...")
            
            # Load samples from file
            raw_samples = self.load_samples_from_file(file_path)
            
            # Normalize and validate samples
            for raw_sample in raw_samples:
                # Normalize format
                normalized = self.normalize_sample_format(raw_sample)
                if not normalized:
                    continue
                
                # Validate sample
                if not self.validate_sample(normalized):
                    continue
                
                # Calculate statistics
                self.calculate_sample_stats(normalized)
                
                # Add metadata
                normalized['source_file'] = file_path.name
                normalized['processed_at'] = datetime.now().isoformat()
                
                all_samples.append(normalized)
                self.stats['total_samples'] += 1
        
        logger.info(f"Processed {self.stats['total_samples']} valid samples from {len(json_files)} files")
        
        if not all_samples:
            raise ValueError("No valid samples found after processing")
        
        # Shuffle samples for better training
        random.shuffle(all_samples)
        
        # Split into train/validation
        val_split = self.config.get('validation_split', 0.1)
        val_size = max(1, int(len(all_samples) * val_split))
        
        train_samples = all_samples[val_size:]
        val_samples = all_samples[:val_size]
        
        self.stats['train_samples'] = len(train_samples)
        self.stats['val_samples'] = len(val_samples)
        
        # Save datasets
        train_path = self.output_dir / 'train_dataset.jsonl'
        val_path = self.output_dir / 'val_dataset.jsonl'
        
        self.save_jsonl(train_samples, train_path)
        self.save_jsonl(val_samples, val_path)
        
        # Generate Axolotl config
        config_path = self.generate_axolotl_config(train_path, val_path)
        
        # Generate training script
        script_path = self.generate_training_script(config_path)
        
        # Save statistics
        stats_path = self.save_statistics()
        
        # Generate README
        readme_path = self.generate_readme(train_path, val_path, config_path, script_path)
        
        result = {
            'train_dataset': str(train_path),
            'val_dataset': str(val_path),
            'axolotl_config': str(config_path),
            'training_script': str(script_path),
            'statistics': str(stats_path),
            'readme': str(readme_path),
            'stats': self.stats.copy()
        }
        
        logger.info("Dataset preparation completed successfully!")
        return result
    
    def save_jsonl(self, samples: List[Dict[str, Any]], output_path: Path):
        """Save samples to JSONL file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(samples)} samples to {output_path}")
    
    def generate_axolotl_config(self, train_path: Path, val_path: Path) -> Path:
        """Generate Axolotl configuration file"""
        config = {
            # Model configuration
            'base_model': self.config.get('base_model', 'microsoft/DialoGPT-medium'),
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',
            
            # Quantization
            'load_in_8bit': False,
            'load_in_4bit': True,
            'strict': False,
            
            # Dataset configuration
            'datasets': [
                {
                    'path': str(train_path),
                    'type': self.config.get('dataset_type', 'conversation')
                }
            ],
            
            # Validation
            'val_set_size': self.stats['val_samples'],
            
            # Output
            'output_dir': str(self.output_dir / 'models'),
            
            # Sequence configuration
            'sequence_len': self.config.get('max_sequence_length', 2048),
            'sample_packing': True,
            'pad_to_sequence_len': True,
            
            # LoRA configuration
            'adapter': 'qlora',
            'lora_r': self.config.get('lora_r', 16),
            'lora_alpha': self.config.get('lora_alpha', 32),
            'lora_dropout': self.config.get('lora_dropout', 0.05),
            'lora_target_modules': [
                'q_proj', 'v_proj', 'k_proj', 'o_proj',
                'gate_proj', 'down_proj', 'up_proj'
            ],
            
            # Training configuration
            'gradient_accumulation_steps': self.config.get('gradient_accumulation_steps', 4),
            'micro_batch_size': self.config.get('batch_size', 4),
            'num_epochs': self.config.get('num_epochs', 3),
            'learning_rate': self.config.get('learning_rate', 2e-4),
            
            # Optimizer
            'optimizer': 'adamw_bnb_8bit',
            'lr_scheduler': 'cosine',
            'warmup_steps': 10,
            
            # Precision
            'bf16': True,
            'fp16': False,
            'gradient_checkpointing': True,
            'flash_attention': True,
            
            # Logging and saving
            'logging_steps': 1,
            'saves_per_epoch': 1,
            'evals_per_epoch': 4,
            
            # Weights & Biases
            'wandb_project': f"akhi-islamic-ai-{datetime.now().strftime('%Y%m%d')}",
            'wandb_name': f"islamic-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            
            # Special tokens
            'special_tokens': {
                'bos_token': '<s>',
                'eos_token': '</s>',
                'unk_token': '<unk>',
                'pad_token': '<pad>'
            }
        }
        
        # Add Islamic-specific configuration
        config['train_on_inputs'] = False  # Only train on assistant responses
        config['group_by_length'] = True   # Group similar length sequences
        
        config_path = self.output_dir / 'axolotl_config.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Generated Axolotl config: {config_path}")
        return config_path
    
    def generate_training_script(self, config_path: Path) -> Path:
        """Generate training script"""
        script_content = f"""#!/bin/bash
# Axolotl QLoRA Training Script for Islamic AI
# Generated by Akhi CrewAI Dataset Preparer
# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

set -e

echo "=== Akhi Islamic AI Training Pipeline ==="
echo "Config: {config_path}"
echo "Dataset: {self.stats['train_samples']} train, {self.stats['val_samples']} val samples"
echo "Start time: $(date)"
echo "========================================="

# Check dependencies
echo "Checking dependencies..."
if ! python -c "import torch; print(f'PyTorch: {{torch.__version__}}')" 2>/dev/null; then
    echo "ERROR: PyTorch not found. Please install PyTorch first."
    exit 1
fi

if ! python -c "import transformers; print(f'Transformers: {{transformers.__version__}}')" 2>/dev/null; then
    echo "Installing transformers..."
    pip install transformers>=4.21.0
fi

if ! python -c "import axolotl" 2>/dev/null; then
    echo "Installing Axolotl..."
    pip install -e git+https://github.com/OpenAccess-AI-Collective/axolotl.git
fi

if ! python -c "import bitsandbytes" 2>/dev/null; then
    echo "Installing bitsandbytes for QLoRA..."
    pip install bitsandbytes
fi

echo "All dependencies satisfied."
echo ""

# Create models directory
mkdir -p {self.output_dir}/models

# Preprocess dataset
echo "Preprocessing dataset..."
python -m axolotl.cli.preprocess {config_path}

if [ $? -ne 0 ]; then
    echo "ERROR: Dataset preprocessing failed!"
    exit 1
fi

echo "Dataset preprocessing completed successfully."
echo ""

# Start training
echo "Starting QLoRA training..."
echo "This may take several hours depending on your hardware..."
python -m axolotl.cli.train {config_path}

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit 1
fi

echo "Training completed successfully!"
echo ""

# Merge LoRA adapters
echo "Merging LoRA adapters..."
python -m axolotl.cli.merge_lora {config_path}

if [ $? -ne 0 ]; then
    echo "WARNING: LoRA merge failed, but training was successful."
    echo "You can merge manually later using: python -m axolotl.cli.merge_lora {config_path}"
else
    echo "LoRA adapters merged successfully!"
fi

echo ""
echo "========================================="
echo "🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉"
echo "========================================="
echo "End time: $(date)"
echo "Model saved to: {self.output_dir}/models/"
echo "Training samples: {self.stats['train_samples']}"
echo "Validation samples: {self.stats['val_samples']}"
echo "Average input length: {self.stats['avg_input_length']:.1f} chars"
echo "Average output length: {self.stats['avg_output_length']:.1f} chars"
echo ""
echo "Next steps:"
echo "1. Test your model with sample inputs"
echo "2. Evaluate on validation set"
echo "3. Deploy to production"
echo "========================================="
"""
        
        script_path = self.output_dir / 'train_model.sh'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
        
        logger.info(f"Generated training script: {script_path}")
        return script_path
    
    def save_statistics(self) -> Path:
        """Save dataset statistics"""
        stats_with_metadata = {
            'generation_info': {
                'generated_at': datetime.now().isoformat(),
                'input_directory': str(self.input_dir),
                'output_directory': str(self.output_dir),
                'config': self.config
            },
            'dataset_statistics': self.stats,
            'quality_metrics': {
                'validation_split': self.config.get('validation_split', 0.1),
                'islamic_content_validated': self.config.get('validate_islamic_content', True),
                'max_content_length': self.config.get('max_content_length', 4096)
            }
        }
        
        stats_path = self.output_dir / 'dataset_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats_with_metadata, f, indent=2)
        
        logger.info(f"Saved statistics: {stats_path}")
        return stats_path
    
    def generate_readme(self, train_path: Path, val_path: Path, config_path: Path, script_path: Path) -> Path:
        """Generate README file"""
        readme_content = f"""# Akhi Islamic AI - QLoRA Dataset

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview

- **Training samples**: {self.stats['train_samples']:,}
- **Validation samples**: {self.stats['val_samples']:,}
- **Total samples**: {self.stats['total_samples']:,}
- **Source files**: {self.stats['total_files']}
- **Average input length**: {self.stats['avg_input_length']:.1f} characters
- **Average output length**: {self.stats['avg_output_length']:.1f} characters

## Files

- `{train_path.name}` - Training dataset (JSONL format)
- `{val_path.name}` - Validation dataset (JSONL format)
- `{config_path.name}` - Axolotl configuration
- `{script_path.name}` - Training script
- `dataset_statistics.json` - Detailed statistics

## Format Distribution

{chr(10).join([f'- {source}: {count} samples' for source, count in self.stats['format_distribution'].items()])}

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers bitsandbytes
pip install -e git+https://github.com/OpenAccess-AI-Collective/axolotl.git
```

### 2. Start Training

```bash
# Make script executable (if not already)
chmod +x {script_path.name}

# Run training
./{script_path.name}
```

### 3. Manual Training (Alternative)

```bash
# Preprocess
python -m axolotl.cli.preprocess {config_path.name}

# Train
python -m axolotl.cli.train {config_path.name}

# Merge LoRA adapters
python -m axolotl.cli.merge_lora {config_path.name}
```

## Configuration

The Axolotl configuration includes:

- **Base Model**: {self.config.get('base_model', 'microsoft/DialoGPT-medium')}
- **Adapter**: QLoRA (4-bit quantization)
- **LoRA Rank**: {self.config.get('lora_r', 16)}
- **LoRA Alpha**: {self.config.get('lora_alpha', 32)}
- **Learning Rate**: {self.config.get('learning_rate', 2e-4)}
- **Batch Size**: {self.config.get('batch_size', 4)}
- **Epochs**: {self.config.get('num_epochs', 3)}
- **Max Sequence Length**: {self.config.get('max_sequence_length', 2048)}

## Dataset Format

Each sample follows the conversation format:

```json
{{
  "conversations": [
    {{"role": "user", "content": "Question about Islamic topic"}},
    {{"role": "assistant", "content": "Islamic guidance and answer"}}
  ],
  "source": "source_identifier",
  "source_file": "original_file.jsonl",
  "processed_at": "2025-01-XX..."
}}
```

## Quality Assurance

- ✅ Islamic content validation
- ✅ Format normalization
- ✅ Length validation
- ✅ Role validation
- ✅ Content quality checks

## Hardware Requirements

### Minimum
- GPU: 8GB VRAM (RTX 3070, RTX 4060 Ti)
- RAM: 16GB
- Storage: 10GB free space

### Recommended
- GPU: 16GB+ VRAM (RTX 4080, RTX 4090, A100)
- RAM: 32GB+
- Storage: 50GB+ free space

## Monitoring

Training progress is logged to:
- Console output
- Weights & Biases (if configured)
- Local log files in `models/` directory

## Troubleshooting

### CUDA Out of Memory
- Reduce `micro_batch_size` in config
- Reduce `sequence_len` in config
- Enable `gradient_checkpointing`

### Slow Training
- Increase `micro_batch_size` if memory allows
- Use `flash_attention` (enabled by default)
- Ensure `sample_packing` is enabled

### Poor Model Quality
- Increase training epochs
- Adjust learning rate
- Increase LoRA rank (`lora_r`)
- Add more high-quality training data

## Support

For issues and questions:
- Check Axolotl documentation: https://github.com/OpenAccess-AI-Collective/axolotl
- Review dataset statistics in `dataset_statistics.json`
- Examine training logs in `models/` directory

---

*Generated by Akhi CrewAI Dataset Preparer v1.0*
"""
        
        readme_path = self.output_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Generated README: {readme_path}")
        return readme_path

def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Prepare QLoRA datasets for Axolotl training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_axolotl_dataset.py --input ./output/jsonl --output ./axolotl_ready
  python prepare_axolotl_dataset.py --input ./data --model microsoft/DialoGPT-medium
  python prepare_axolotl_dataset.py --config custom_config.yaml

The script will:
1. Discover all JSON/JSONL files in input directory
2. Normalize formats to Axolotl conversation format
3. Validate Islamic content and quality
4. Split into train/validation sets
5. Generate Axolotl config and training script
"""
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', type=str, required=True, help='Input directory containing JSON/JSONL files')
    parser.add_argument('--output', '-o', type=str, default='./axolotl_ready', help='Output directory for prepared dataset')
    
    # Configuration
    parser.add_argument('--config', '-c', type=str, help='YAML configuration file')
    
    # Model options
    parser.add_argument('--model', type=str, default='microsoft/DialoGPT-medium', help='Base model for fine-tuning')
    parser.add_argument('--max-length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
    
    # Quality options
    parser.add_argument('--no-islamic-validation', action='store_true', help='Skip Islamic content validation')
    parser.add_argument('--max-content-length', type=int, default=4096, help='Maximum content length per message')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        if args.config:
            config = load_config_file(args.config)
        else:
            config = {}
        
        # Override with command line arguments
        config.update({
            'input_dir': args.input,
            'output_dir': args.output,
            'base_model': args.model,
            'max_sequence_length': args.max_length,
            'validation_split': args.val_split,
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'validate_islamic_content': not args.no_islamic_validation,
            'max_content_length': args.max_content_length
        })
        
        # Initialize preparer
        preparer = AxolotlDatasetPreparer(config)
        
        # Prepare dataset
        result = preparer.prepare_dataset()
        
        # Print summary
        print("\n" + "="*60)
        print("AXOLOTL DATASET PREPARATION COMPLETED")
        print("="*60)
        print(f"📊 Dataset Statistics:")
        print(f"   Training samples: {result['stats']['train_samples']:,}")
        print(f"   Validation samples: {result['stats']['val_samples']:,}")
        print(f"   Total samples: {result['stats']['total_samples']:,}")
        print(f"   Source files: {result['stats']['total_files']}")
        print(f"   Avg input length: {result['stats']['avg_input_length']:.1f} chars")
        print(f"   Avg output length: {result['stats']['avg_output_length']:.1f} chars")
        print(f"\n📁 Generated Files:")
        print(f"   Training dataset: {result['train_dataset']}")
        print(f"   Validation dataset: {result['val_dataset']}")
        print(f"   Axolotl config: {result['axolotl_config']}")
        print(f"   Training script: {result['training_script']}")
        print(f"   README: {result['readme']}")
        print(f"\n🚀 Ready to train! Run:")
        print(f"   cd {args.output}")
        print(f"   ./train_model.sh")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()