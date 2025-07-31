#!/usr/bin/env python3
"""
QLoRA Formatter Tool for CrewAI

This module wraps the existing QLoRA formatter functionality as a CrewAI-compatible tool
for converting Islamic transcripts into training data format for fine-tuning.

Author: Assistant
Date: December 2024
Phase: 8 - QLoRA Fine-Tuning
"""

import os
import sys
import json
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from crewai.tools import BaseTool

# Import existing QLoRA formatter with robust path resolution
import pathlib

# Use absolute path resolution to avoid relative path issues
project_root = pathlib.Path(__file__).resolve().parents[2]
qlora_formatter_path = project_root / "pipeline" / "agents" / "qlora_formatter.py"

print(f"[DEBUG] Attempting to load QLoRAFormatter from: {qlora_formatter_path}")
print(f"[DEBUG] File exists: {qlora_formatter_path.exists()}")

# Import QLoRAFormatter with error handling
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pipeline_qlora_formatter", 
        str(qlora_formatter_path)
    )
    if spec is None:
        raise ImportError(f"Could not create spec for {qlora_formatter_path}")
    
    pipeline_qlora_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_qlora_module)
    QLoRAFormatter = pipeline_qlora_module.QLoRAFormatter
    print(f"[DEBUG] QLoRAFormatter successfully loaded from: {QLoRAFormatter.__module__}")
    
except Exception as e:
    print(f"[ERROR] Failed to import QLoRAFormatter: {str(e)}")
    print(f"[ERROR] Exception type: {type(e).__name__}")
    # Fail hard during development to catch import issues
    raise ImportError(f"[FATAL] Failed to import QLoRAFormatter from {qlora_formatter_path}: {e}")


class QLoRAFormatterInput(BaseModel):
    """
    Input schema for QLoRA Formatter Tool.
    """
    transcript_dir: str = Field(
        description="Directory containing transcript files to process",
        default="data/transcripts"
    )
    output_file: str = Field(
        description="Output file path for QLoRA training data",
        default="data/training/akhi_qlora.json"
    )
    min_segment_words: int = Field(
        description="Minimum number of words for a segment to be included",
        default=50,
        ge=10,
        le=200
    )
    max_segment_words: int = Field(
        description="Maximum number of words for a segment (longer will be split)",
        default=500,
        ge=100,
        le=2000
    )
    include_metadata: bool = Field(
        description="Include metadata in training examples",
        default=True
    )
    filter_islamic_content: bool = Field(
        description="Filter for Islamic content only",
        default=True
    )
    
    @field_validator('transcript_dir')
    @classmethod
    def validate_transcript_dir(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Transcript directory does not exist: {v}")
        return v
    
    @field_validator('output_file')
    @classmethod
    def validate_output_file(cls, v):
        output_dir = os.path.dirname(v)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        return v


class QLoRAFormatterTool(BaseTool):
    """
    CrewAI tool for formatting transcripts into QLoRA training data.
    
    This tool converts processed Islamic content transcripts into the format
    required for QLoRA fine-tuning, creating conversation pairs suitable for
    training language models on Islamic knowledge.
    """
    
    name: str = "qlora_formatter"
    description: str = (
        "Convert Islamic transcripts to QLoRA training format. "
        "This tool processes transcript files and creates conversation pairs "
        "suitable for fine-tuning language models on Islamic content. "
        "Input: transcript directory path, output file path, formatting options. "
        "Output: JSON file with training conversations and statistics."
    )
    args_schema: type = QLoRAFormatterInput
    config_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    formatter: Optional[Any] = None
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.config_path = config_path or "config/crew_config.yaml"
        self.config = self._load_config()
        self.formatter = QLoRAFormatter()
    
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
            return config.get('qlora_training', {})
        except FileNotFoundError:
            return {
                'min_segment_words': 50,
                'max_segment_words': 500,
                'include_metadata': True,
                'filter_islamic_content': True
            }
    
    def _run(
        self, 
        transcript_dir: str = "data/transcripts",
        output_file: str = "data/training/akhi_qlora.json",
        min_segment_words: int = 50,
        max_segment_words: int = 500,
        include_metadata: bool = True,
        filter_islamic_content: bool = True
    ) -> str:
        """
        Execute the QLoRA formatting process.
        
        Args:
            transcript_dir: Directory containing transcript files
            output_file: Output file path for training data
            min_segment_words: Minimum words per segment
            max_segment_words: Maximum words per segment
            include_metadata: Include metadata in examples
            filter_islamic_content: Filter for Islamic content only
            
        Returns:
            Status message with results
        """
        try:
            # Initialize the QLoRA formatter with parameters
            formatter = QLoRAFormatter(
                min_segment_words=min_segment_words,
                max_segment_words=max_segment_words
            )
            
            # Update transcript directory if different from default
            if transcript_dir != "data/transcripts":
                # Update the formatter's transcript directory
                import tempfile
                import shutil
                
                # Create temporary config to override transcript directory
                temp_config = {
                    'transcript_dir': transcript_dir,
                    'output_file': output_file,
                    'include_metadata': include_metadata,
                    'filter_islamic_content': filter_islamic_content
                }
                
                # Store original paths
                original_transcripts_dir = formatter.__class__.__dict__.get('TRANSCRIPTS_DIR')
                original_output_file = formatter.__class__.__dict__.get('OUTPUT_JSON_FILE')
                
                # Temporarily update paths
                if hasattr(formatter, 'TRANSCRIPTS_DIR'):
                    formatter.TRANSCRIPTS_DIR = transcript_dir
                if hasattr(formatter, 'OUTPUT_JSON_FILE'):
                    formatter.OUTPUT_JSON_FILE = output_file
            
            # Generate the QLoRA training data
            result = formatter.generate_json()
            print(f"[DEBUG] QLoRA formatter result: {result}")
            
            # Prepare response with statistics
            if result.get('success', False):
                # Extract actual statistics from the result
                examples_generated = result.get('examples_generated', 0)
                files_processed = result.get('files_processed', 0)
                
                message = (
                    f"✅ QLoRA training data generated successfully!\n"
                    f"📁 Output file: {output_file}\n"
                    f"📊 Statistics:\n"
                    f"   • Total examples: {examples_generated}\n"
                    f"   • Files processed: {files_processed}\n"
                    f"   • Message: {result.get('message', 'No message')}\n"
                    f"\n🎯 Ready for QLoRA fine-tuning with Axolotl!"
                )
                
                return message
            else:
                error_msg = result.get('message', result.get('error', 'Unknown error occurred'))
                return f"❌ QLoRA formatting failed: {error_msg}"
                
        except Exception as e:
            return f"❌ Error during QLoRA formatting: {str(e)}"
    
    def get_training_data_stats(self, file_path: str) -> Dict[str, Any]:
        """
        Get statistics about generated training data.
        
        Args:
            file_path: Path to the training data file
            
        Returns:
            Statistics dictionary
        """
        try:
            if not os.path.exists(file_path):
                return {'error': 'Training data file not found'}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversations = data.get('conversations', [])
            total_conversations = len(conversations)
            total_examples = sum(len(conv.get('examples', [])) for conv in conversations)
            
            # Calculate word statistics
            word_counts = []
            islamic_keywords = ['allah', 'islam', 'muslim', 'quran', 'prophet', 'hadith', 'prayer', 'mosque']
            islamic_content_count = 0
            
            for conv in conversations:
                for example in conv.get('examples', []):
                    content = example.get('content', '').lower()
                    words = len(content.split())
                    word_counts.append(words)
                    
                    # Check for Islamic content
                    if any(keyword in content for keyword in islamic_keywords):
                        islamic_content_count += 1
            
            avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
            islamic_ratio = islamic_content_count / total_examples if total_examples > 0 else 0
            
            return {
                'total_conversations': total_conversations,
                'total_examples': total_examples,
                'avg_words_per_example': avg_words,
                'islamic_content_ratio': islamic_ratio,
                'min_words': min(word_counts) if word_counts else 0,
                'max_words': max(word_counts) if word_counts else 0,
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
            }
            
        except Exception as e:
            return {'error': f'Failed to analyze training data: {str(e)}'}
    
    def validate_training_data(self, file_path: str) -> Dict[str, Any]:
        """
        Validate the generated training data for quality and format.
        
        Args:
            file_path: Path to the training data file
            
        Returns:
            Validation results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            issues = []
            warnings = []
            
            # Check basic structure
            if 'conversations' not in data:
                issues.append("Missing 'conversations' key in data")
                return {'valid': False, 'issues': issues}
            
            conversations = data['conversations']
            
            # Validate conversations
            for i, conv in enumerate(conversations):
                if 'examples' not in conv:
                    issues.append(f"Conversation {i}: Missing 'examples' key")
                    continue
                
                examples = conv['examples']
                if len(examples) == 0:
                    warnings.append(f"Conversation {i}: No examples found")
                
                # Validate examples
                for j, example in enumerate(examples):
                    if 'role' not in example or 'content' not in example:
                        issues.append(f"Conversation {i}, Example {j}: Missing 'role' or 'content'")
                    
                    content = example.get('content', '')
                    if len(content.strip()) == 0:
                        issues.append(f"Conversation {i}, Example {j}: Empty content")
                    
                    if len(content.split()) < 5:
                        warnings.append(f"Conversation {i}, Example {j}: Very short content ({len(content.split())} words)")
            
            # Quality checks
            stats = self.get_training_data_stats(file_path)
            
            if stats.get('islamic_content_ratio', 0) < 0.5:
                warnings.append(f"Low Islamic content ratio: {stats.get('islamic_content_ratio', 0):.2%}")
            
            if stats.get('avg_words_per_example', 0) < 20:
                warnings.append(f"Low average words per example: {stats.get('avg_words_per_example', 0):.1f}")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'warnings': warnings,
                'stats': stats
            }
            
        except Exception as e:
            return {
                'valid': False,
                'issues': [f'Validation failed: {str(e)}'],
                'warnings': [],
                'stats': {}
            }


if __name__ == "__main__":
    # Demo usage
    tool = QLoRAFormatterTool()
    
    # Test with default parameters
    result = tool._run()
    print("QLoRA Formatter Tool Demo:")
    print(result)
    
    # Test validation if file exists
    output_file = "data/training/akhi_qlora.json"
    if os.path.exists(output_file):
        validation = tool.validate_training_data(output_file)
        print(f"\nValidation Results: {validation}")