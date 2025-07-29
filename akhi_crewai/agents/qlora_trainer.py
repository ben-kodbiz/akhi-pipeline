#!/usr/bin/env python3
"""
QLoRA Trainer Agent for CrewAI Agentic System

This agent specializes in fine-tuning LLMs on Islamic content using QLoRA methodology.
It orchestrates the complete training pipeline from data preparation to model deployment.

Author: Assistant
Date: December 2024
Phase: 8 - QLoRA Fine-Tuning
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# CrewAI imports
from crewai import Agent, LLM

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))

# Import tools dynamically to avoid circular imports
try:
    from qlora_formatter import QLoRAFormatterTool
    from axolotl_trainer import AxolotlTrainerTool
    from model_validator import ModelValidatorTool
    from model_deployer import ModelDeployerTool
    TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: QLoRA tools not available: {e}")
    TOOLS_AVAILABLE = False
    QLoRAFormatterTool = None
    AxolotlTrainerTool = None
    ModelValidatorTool = None
    ModelDeployerTool = None


class QLoRATrainerAgent:
    """
    QLoRA Trainer Agent for Islamic Content Fine-Tuning
    
    This agent manages the complete QLoRA fine-tuning workflow:
    1. Data preparation and formatting
    2. Training configuration generation
    3. Model training execution with Axolotl
    4. Model validation and testing
    5. Model deployment and integration
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize QLoRA Trainer Agent.
        
        Args:
            config_path: Path to configuration file
            config: Optional configuration dictionary (for compatibility)
        """
        self.config = self._load_config(config_path)
        self.llm = self._setup_llm()
        self.tools = self._setup_tools()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize tools if available
        if TOOLS_AVAILABLE:
            self.formatter_tool = QLoRAFormatterTool()
            self.trainer_tool = AxolotlTrainerTool()
            self.validator_tool = ModelValidatorTool()
            self.deployer_tool = ModelDeployerTool()
        else:
            self.formatter_tool = None
            self.trainer_tool = None
            self.validator_tool = None
            self.deployer_tool = None
            self.logger.warning("QLoRA tools not available, agent will have limited functionality")
        
        self.agent = self._create_agent()
        
        # Training state management
        self.training_state = {
            'current_job': None,
            'status': 'idle',
            'progress': 0,
            'last_checkpoint': None,
            'model_path': None,
            'validation_results': None
        }
        
        self.logger.info("QLoRATrainerAgent initialized successfully")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                '../config/crew_config.yaml'
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            # Fallback configuration
            return {
                'local_llm': {
                    'model_name': 'lm_studio/qwen-3-14b',
                    'base_url': 'http://192.168.0.74:1234/v1',
                    'api_key': None,
                    'temperature': 0.7,
                    'max_tokens': 2048
                },
                'qlora_training': {
                    'axolotl': {
                        'config_template': 'config/axolotl_template.yaml',
                        'output_dir': 'models/fine_tuned',
                        'base_model': 'microsoft/DialoGPT-medium',
                        'lora_r': 16,
                        'lora_alpha': 32,
                        'lora_dropout': 0.1,
                        'learning_rate': 2e-4,
                        'num_epochs': 3,
                        'batch_size': 4,
                        'gradient_accumulation_steps': 4
                    },
                    'validation': {
                        'islamic_qa_dataset': 'data/validation/islamic_qa.jsonl',
                        'accuracy_threshold': 0.85,
                        'bias_detection': True
                    }
                },
                'agents': {
                    'qlora_trainer': {
                        'role': 'Islamic Model Fine-Tuning Specialist',
                        'goal': 'Fine-tune LLMs on Islamic content using QLoRA methodology with high accuracy and cultural sensitivity',
                        'backstory': 'You are an expert in machine learning and Islamic studies, specializing in fine-tuning language models on religious content. You ensure models are accurate, unbiased, and culturally appropriate.',
                        'max_iter': 10,
                        'max_execution_time': 7200,  # 2 hours for training tasks
                        'verbose': True,
                        'allow_delegation': False
                    }
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging for the agent.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger('QLoRATrainerAgent')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Create file handler
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            handler = logging.FileHandler(log_dir / 'qlora_trainer.log')
            handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
        
        return logger
    
    def _setup_llm(self) -> LLM:
        """
        Setup the local LLM for the agent.
        
        Returns:
            Configured LLM instance
        """
        llm_config = self.config.get('local_llm', {})
        
        return LLM(
            model=llm_config.get('model_name', 'lm_studio/qwen-3-14b'),
            base_url=llm_config.get('base_url', 'http://192.168.0.74:1234/v1'),
            api_key=llm_config.get('api_key'),
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens', 2048)
        )
    
    def _setup_tools(self) -> list:
        """
        Setup tools for the agent.
        
        Returns:
            List of tools for the agent
        """
        if not TOOLS_AVAILABLE:
            self.logger.warning("QLoRA tools not available, returning empty tools list")
            return []
        
        return [
            QLoRAFormatterTool(),
            AxolotlTrainerTool(),
            ModelValidatorTool(),
            ModelDeployerTool()
        ]
    
    def _create_agent(self) -> Agent:
        """
        Create the CrewAI agent instance.
        
        Returns:
            Configured Agent instance
        """
        agent_config = self.config.get('agents', {}).get('qlora_trainer', {})
        
        return Agent(
            role=agent_config.get(
                'role', 
                'Islamic Model Fine-Tuning Specialist'
            ),
            goal=agent_config.get(
                'goal',
                'Fine-tune LLMs on Islamic content using QLoRA methodology '
                'with high accuracy and cultural sensitivity'
            ),
            backstory=agent_config.get(
                'backstory',
                'You are an expert in machine learning and Islamic studies, '
                'specializing in fine-tuning language models on religious content. '
                'You ensure models are accurate, unbiased, and culturally appropriate.'
            ),
            tools=self.tools,
            llm=self.llm,
            max_iter=agent_config.get('max_iter', 10),
            max_execution_time=agent_config.get('max_execution_time', 7200),
            verbose=agent_config.get('verbose', True),
            allow_delegation=agent_config.get('allow_delegation', False)
        )
    
    def get_agent(self) -> Agent:
        """
        Get the CrewAI agent instance.
        
        Returns:
            The configured agent
        """
        return self.agent
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status.
        
        Returns:
            Training status dictionary
        """
        return self.training_state.copy()
    
    def update_training_status(self, **kwargs) -> None:
        """
        Update training status.
        
        Args:
            **kwargs: Status fields to update
        """
        self.training_state.update(kwargs)
        self.logger.info(f"Training status updated: {kwargs}")
    
    def start_training_job(self, job_config: Dict[str, Any]) -> str:
        """
        Start a new training job.
        
        Args:
            job_config: Training job configuration
            
        Returns:
            Job ID
        """
        job_id = f"qlora_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.update_training_status(
            current_job=job_id,
            status='preparing',
            progress=0,
            last_checkpoint=None,
            model_path=None,
            validation_results=None
        )
        
        self.logger.info(f"Started training job: {job_id}")
        return job_id
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get agent configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config


def create_qlora_trainer_agent(config_path: Optional[str] = None) -> Agent:
    """
    Factory function to create a QLoRA Trainer Agent.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured QLoRA Trainer Agent
    """
    trainer = QLoRATrainerAgent(config_path)
    return trainer.get_agent()


if __name__ == "__main__":
    # Demo usage
    trainer = QLoRATrainerAgent()
    print(f"QLoRA Trainer Agent created: {trainer.get_agent().role}")
    print(f"Available tools: {[tool.name for tool in trainer.tools]}")
    print(f"Training status: {trainer.get_training_status()}")