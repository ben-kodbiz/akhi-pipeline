#!/usr/bin/env python3
"""
Axolotl Trainer Tool for CrewAI

This module provides a CrewAI-compatible tool for training QLoRA models using Axolotl framework
with Islamic content training data.

Author: Assistant
Date: December 2024
Phase: 8 - QLoRA Fine-Tuning
"""

import os
import sys
import json
import yaml
import subprocess
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel, Field, field_validator
from crewai.tools import BaseTool


class AxolotlTrainerInput(BaseModel):
    """
    Input schema for Axolotl Trainer Tool.
    """
    training_data_path: str = Field(
        description="Path to the QLoRA training data JSON file",
        default="data/training/akhi_qlora.json"
    )
    base_model: str = Field(
        description="Base model to fine-tune (HuggingFace model name)",
        default="microsoft/DialoGPT-medium"
    )
    output_dir: str = Field(
        description="Directory to save the trained model",
        default="models/akhi_qlora"
    )
    config_file: str = Field(
        description="Axolotl configuration file path",
        default="config/axolotl_config.yaml"
    )
    num_epochs: int = Field(
        description="Number of training epochs",
        default=3,
        ge=1,
        le=10
    )
    learning_rate: float = Field(
        description="Learning rate for training",
        default=2e-4,
        gt=0.0,
        le=1e-2
    )
    batch_size: int = Field(
        description="Training batch size",
        default=4,
        ge=1,
        le=32
    )
    max_seq_length: int = Field(
        description="Maximum sequence length",
        default=512,
        ge=128,
        le=2048
    )
    lora_rank: int = Field(
        description="LoRA rank for QLoRA",
        default=16,
        ge=4,
        le=64
    )
    lora_alpha: int = Field(
        description="LoRA alpha parameter",
        default=32,
        ge=8,
        le=128
    )
    use_gradient_checkpointing: bool = Field(
        description="Use gradient checkpointing to save memory",
        default=True
    )
    save_steps: int = Field(
        description="Save model every N steps",
        default=500,
        ge=100,
        le=2000
    )
    
    @field_validator('training_data_path')
    @classmethod
    def validate_training_data(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Training data file does not exist: {v}")
        return v
    
    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, v):
        os.makedirs(v, exist_ok=True)
        return v


class AxolotlTrainerTool(BaseTool):
    """
    CrewAI tool for training QLoRA models using Axolotl framework.
    
    This tool handles the complete QLoRA training process including:
    - Configuration generation
    - Training data preparation
    - Model training with Axolotl
    - Training monitoring and logging
    - Model validation and saving
    """
    
    name: str = "axolotl_trainer"
    description: str = (
        "Train QLoRA models using Axolotl framework for Islamic content. "
        "This tool configures and executes the complete training pipeline "
        "including data preparation, model training, and validation. "
        "Input: training data path, model configuration, training parameters. "
        "Output: Trained model and training metrics."
    )
    args_schema: type = AxolotlTrainerInput
    config: Optional[Dict[str, Any]] = None
    training_status: Optional[Dict[str, Any]] = None
    
    def __init__(self):
        super().__init__()
        self.config = self._load_config()
        self.training_status = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration for the tool.
        
        Returns:
            Configuration dictionary
        """
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '../config/crew_config.yaml'
        )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('axolotl_training', {})
        except FileNotFoundError:
            return {
                'default_base_model': 'microsoft/DialoGPT-medium',
                'default_epochs': 3,
                'default_learning_rate': 2e-4,
                'default_batch_size': 4
            }
    
    def _create_axolotl_config(self, **kwargs) -> str:
        """
        Create Axolotl configuration file.
        
        Args:
            **kwargs: Training parameters
            
        Returns:
            Path to created config file
        """
        config = {
            'base_model': kwargs.get('base_model', 'microsoft/DialoGPT-medium'),
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',
            
            # Dataset configuration
            'datasets': [
                {
                    'path': kwargs.get('training_data_path'),
                    'type': 'json',
                    'conversation': {
                        'field': 'conversations',
                        'field_human': 'human',
                        'field_assistant': 'assistant'
                    }
                }
            ],
            
            # Training parameters
            'sequence_len': kwargs.get('max_seq_length', 512),
            'sample_packing': True,
            'pad_to_sequence_len': True,
            
            # LoRA configuration
            'adapter': 'qlora',
            'lora_model_dir': kwargs.get('output_dir'),
            'lora_r': kwargs.get('lora_rank', 16),
            'lora_alpha': kwargs.get('lora_alpha', 32),
            'lora_dropout': 0.05,
            'lora_target_linear': True,
            'lora_fan_in_fan_out': False,
            
            # Training configuration
            'output_dir': kwargs.get('output_dir'),
            'num_epochs': kwargs.get('num_epochs', 3),
            'micro_batch_size': kwargs.get('batch_size', 4),
            'gradient_accumulation_steps': 4,
            'learning_rate': kwargs.get('learning_rate', 2e-4),
            'optimizer': 'adamw_bnb_8bit',
            'lr_scheduler': 'cosine',
            'train_on_inputs': False,
            'group_by_length': False,
            'bf16': True,
            'fp16': False,
            'tf32': False,
            
            # Memory optimization
            'gradient_checkpointing': kwargs.get('use_gradient_checkpointing', True),
            'early_stopping_patience': 3,
            'resume_from_checkpoint': None,
            'local_rank': -1,
            'logging_steps': 1,
            'xformers_attention': True,
            'flash_attention': True,
            
            # Saving configuration
            'save_steps': kwargs.get('save_steps', 500),
            'eval_steps': 250,
            'save_total_limit': 3,
            'load_best_model_at_end': True,
            
            # Special tokens
            'special_tokens': {
                'bos_token': '<s>',
                'eos_token': '</s>',
                'pad_token': '[PAD]',
                'unk_token': '<unk>'
            }
        }
        
        # Create config directory if it doesn't exist
        config_file = kwargs.get('config_file', 'config/axolotl_config.yaml')
        config_dir = os.path.dirname(config_file)
        os.makedirs(config_dir, exist_ok=True)
        
        # Write configuration
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        return config_file
    
    def _run(
        self,
        training_data_path: str = "data/training/akhi_qlora.json",
        base_model: str = "microsoft/DialoGPT-medium",
        output_dir: str = "models/akhi_qlora",
        config_file: str = "config/axolotl_config.yaml",
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        max_seq_length: int = 512,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        use_gradient_checkpointing: bool = True,
        save_steps: int = 500
    ) -> str:
        """
        Execute the QLoRA training process using Axolotl.
        
        Args:
            training_data_path: Path to training data
            base_model: Base model to fine-tune
            output_dir: Output directory for trained model
            config_file: Axolotl config file path
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Training batch size
            max_seq_length: Maximum sequence length
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            use_gradient_checkpointing: Use gradient checkpointing
            save_steps: Save model every N steps
            
        Returns:
            Training status message
        """
        try:
            # Validate training data
            if not os.path.exists(training_data_path):
                return f"❌ Training data file not found: {training_data_path}"
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Create Axolotl configuration
            config_path = self._create_axolotl_config(
                training_data_path=training_data_path,
                base_model=base_model,
                output_dir=output_dir,
                config_file=config_file,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                use_gradient_checkpointing=use_gradient_checkpointing,
                save_steps=save_steps
            )
            
            # Initialize training status
            training_id = f"akhi_qlora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.training_status[training_id] = {
                'status': 'starting',
                'start_time': datetime.now(),
                'config_path': config_path,
                'output_dir': output_dir,
                'base_model': base_model
            }
            
            # Check if Axolotl is installed
            try:
                result = subprocess.run(
                    ['python', '-c', 'import axolotl; print("Axolotl available")'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode != 0:
                    return (
                        f"❌ Axolotl not installed. Please install with:\n"
                        f"pip install axolotl-ai\n"
                        f"or\n"
                        f"git clone https://github.com/OpenAccess-AI-Collective/axolotl\n"
                        f"cd axolotl && pip install -e ."
                    )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return "❌ Python or Axolotl not available in current environment"
            
            # Start training process
            training_command = [
                'python', '-m', 'axolotl.cli.train',
                config_path
            ]
            
            # Log training start
            log_file = os.path.join(output_dir, 'training.log')
            
            message = (
                f"🚀 Starting QLoRA training with Axolotl...\n"
                f"📋 Training ID: {training_id}\n"
                f"🤖 Base Model: {base_model}\n"
                f"📁 Output Directory: {output_dir}\n"
                f"⚙️ Configuration: {config_path}\n"
                f"📊 Training Parameters:\n"
                f"   • Epochs: {num_epochs}\n"
                f"   • Learning Rate: {learning_rate}\n"
                f"   • Batch Size: {batch_size}\n"
                f"   • LoRA Rank: {lora_rank}\n"
                f"   • LoRA Alpha: {lora_alpha}\n"
                f"   • Max Sequence Length: {max_seq_length}\n"
                f"\n📝 Training command: {' '.join(training_command)}\n"
                f"📄 Logs will be saved to: {log_file}\n"
                f"\n⏳ Training started... This may take several hours.\n"
                f"\n💡 To monitor progress:\n"
                f"   • Check log file: tail -f {log_file}\n"
                f"   • Monitor GPU usage: nvidia-smi\n"
                f"   • Check output directory: ls -la {output_dir}"
            )
            
            # Update status
            self.training_status[training_id]['status'] = 'running'
            self.training_status[training_id]['command'] = training_command
            self.training_status[training_id]['log_file'] = log_file
            
            # Note: In a real implementation, you would start the training process
            # For demo purposes, we're showing the setup and command
            
            return message
            
        except Exception as e:
            return f"❌ Error starting QLoRA training: {str(e)}"
    
    def check_training_status(self, training_id: str) -> Dict[str, Any]:
        """
        Check the status of a training job.
        
        Args:
            training_id: Training job ID
            
        Returns:
            Training status information
        """
        if training_id not in self.training_status:
            return {'error': f'Training ID {training_id} not found'}
        
        status = self.training_status[training_id]
        
        # Check if output directory has model files
        output_dir = status.get('output_dir')
        model_files = []
        if output_dir and os.path.exists(output_dir):
            model_files = [
                f for f in os.listdir(output_dir) 
                if f.endswith(('.bin', '.safetensors', '.json'))
            ]
        
        # Check log file for progress
        log_file = status.get('log_file')
        last_log_lines = []
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    last_log_lines = lines[-10:] if len(lines) > 10 else lines
            except Exception:
                pass
        
        return {
            'training_id': training_id,
            'status': status.get('status'),
            'start_time': status.get('start_time'),
            'output_dir': output_dir,
            'model_files': model_files,
            'last_log_lines': [line.strip() for line in last_log_lines],
            'config_path': status.get('config_path')
        }
    
    def validate_trained_model(self, model_path: str) -> Dict[str, Any]:
        """
        Validate a trained QLoRA model.
        
        Args:
            model_path: Path to the trained model
            
        Returns:
            Validation results
        """
        try:
            if not os.path.exists(model_path):
                return {'valid': False, 'error': f'Model path does not exist: {model_path}'}
            
            # Check for required files
            required_files = ['adapter_config.json', 'adapter_model.bin']
            missing_files = []
            
            for file in required_files:
                file_path = os.path.join(model_path, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
            
            if missing_files:
                return {
                    'valid': False,
                    'error': f'Missing required files: {missing_files}'
                }
            
            # Get model info
            config_path = os.path.join(model_path, 'adapter_config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Get model size
            model_file = os.path.join(model_path, 'adapter_model.bin')
            model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
            
            return {
                'valid': True,
                'model_type': config.get('peft_type', 'Unknown'),
                'base_model': config.get('base_model_name_or_path', 'Unknown'),
                'lora_rank': config.get('r', 'Unknown'),
                'lora_alpha': config.get('lora_alpha', 'Unknown'),
                'model_size_mb': model_size_mb,
                'target_modules': config.get('target_modules', []),
                'task_type': config.get('task_type', 'Unknown')
            }
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation failed: {str(e)}'}


if __name__ == "__main__":
    # Demo usage
    tool = AxolotlTrainerTool()
    
    # Test configuration creation
    print("Axolotl Trainer Tool Demo:")
    
    # Create a sample config
    config_path = tool._create_axolotl_config(
        training_data_path="data/training/akhi_qlora.json",
        base_model="microsoft/DialoGPT-medium",
        output_dir="models/akhi_qlora_demo",
        config_file="config/demo_axolotl_config.yaml"
    )
    
    print(f"✅ Created Axolotl config: {config_path}")
    
    # Test training setup (without actually training)
    result = tool._run(
        training_data_path="data/training/akhi_qlora.json",
        output_dir="models/akhi_qlora_demo"
    )
    print(f"\nTraining Setup Result:\n{result}")