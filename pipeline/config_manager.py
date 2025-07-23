#!/usr/bin/env python3

import os
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

@dataclass
class SearchConfig:
    """Configuration for search functionality."""
    default_terms: List[str] = field(default_factory=list)
    min_duration: int = 300
    max_duration: int = 3600
    max_results_per_term: int = 20
    min_view_count: int = 1000
    exclude_keywords: List[str] = field(default_factory=list)

@dataclass
class DownloadConfig:
    """Configuration for download functionality."""
    max_videos_per_batch: int = 10
    audio_format: str = "mp3"
    audio_quality: str = "0"
    retry_attempts: int = 3
    retry_delay: int = 5
    output_template: str = "%(title)s.%(ext)s"

@dataclass
class TranscriptionConfig:
    """Configuration for transcription functionality."""
    model_size: str = "base"
    device: str = "cpu"
    language: str = "auto"
    max_files_per_batch: int = 5
    word_timestamps: bool = True
    speaker_detection: bool = True
    min_confidence: float = 0.7
    min_segment_length: int = 10

@dataclass
class QLoRAConfig:
    """Configuration for QLoRA formatting."""
    min_segment_words: int = 50
    max_segment_words: int = 500
    overlap_words: int = 10
    exclude_patterns: List[str] = field(default_factory=list)
    instruction_template: str = "Explain the following Islamic concept or teaching:"
    response_template: str = "{content}"

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    max_log_size: str = "10MB"
    backup_count: int = 5

@dataclass
class PipelineConfig:
    """Configuration for pipeline management."""
    auto_save_interval: int = 10
    backup_state: bool = True
    max_retries: int = 3
    continue_on_error: bool = True
    parallel_downloads: bool = False
    parallel_transcriptions: bool = False
    max_workers: int = 4

@dataclass
class Config:
    """Main configuration class."""
    search: SearchConfig = field(default_factory=SearchConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default.
        """
        if config_path is None:
            # Default to config.yaml in the same directory as this file
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Config:
        """
        Load configuration from YAML file.
        
        Returns:
            Config object with loaded settings
        """
        try:
            if not os.path.exists(self.config_path):
                print(f"Warning: Config file not found at {self.config_path}. Using defaults.")
                return Config()
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            if not yaml_data:
                print("Warning: Empty config file. Using defaults.")
                return Config()
            
            # Create config objects from YAML data
            config = Config()
            
            # Load search config
            if 'search' in yaml_data:
                search_data = yaml_data['search']
                config.search = SearchConfig(
                    default_terms=search_data.get('default_terms', []),
                    min_duration=search_data.get('min_duration', 300),
                    max_duration=search_data.get('max_duration', 3600),
                    max_results_per_term=search_data.get('max_results_per_term', 20),
                    min_view_count=search_data.get('min_view_count', 1000),
                    exclude_keywords=search_data.get('exclude_keywords', [])
                )
            
            # Load download config
            if 'download' in yaml_data:
                download_data = yaml_data['download']
                config.download = DownloadConfig(
                    max_videos_per_batch=download_data.get('max_videos_per_batch', 10),
                    audio_format=download_data.get('audio_format', 'mp3'),
                    audio_quality=download_data.get('audio_quality', '0'),
                    retry_attempts=download_data.get('retry_attempts', 3),
                    retry_delay=download_data.get('retry_delay', 5),
                    output_template=download_data.get('output_template', '%(title)s.%(ext)s')
                )
            
            # Load transcription config
            if 'transcription' in yaml_data:
                transcription_data = yaml_data['transcription']
                config.transcription = TranscriptionConfig(
                    model_size=transcription_data.get('model_size', 'base'),
                    device=transcription_data.get('device', 'cpu'),
                    language=transcription_data.get('language', 'auto'),
                    max_files_per_batch=transcription_data.get('max_files_per_batch', 5),
                    word_timestamps=transcription_data.get('word_timestamps', True),
                    speaker_detection=transcription_data.get('speaker_detection', True),
                    min_confidence=transcription_data.get('min_confidence', 0.7),
                    min_segment_length=transcription_data.get('min_segment_length', 10)
                )
            
            # Load QLoRA config
            if 'qlora' in yaml_data:
                qlora_data = yaml_data['qlora']
                config.qlora = QLoRAConfig(
                    min_segment_words=qlora_data.get('min_segment_words', 50),
                    max_segment_words=qlora_data.get('max_segment_words', 500),
                    overlap_words=qlora_data.get('overlap_words', 10),
                    exclude_patterns=qlora_data.get('exclude_patterns', []),
                    instruction_template=qlora_data.get('instruction_template', 'Explain the following Islamic concept or teaching:'),
                    response_template=qlora_data.get('response_template', '{content}')
                )
            
            # Load logging config
            if 'logging' in yaml_data:
                logging_data = yaml_data['logging']
                config.logging = LoggingConfig(
                    level=logging_data.get('level', 'INFO'),
                    max_log_size=logging_data.get('max_log_size', '10MB'),
                    backup_count=logging_data.get('backup_count', 5)
                )
            
            # Load pipeline config
            if 'pipeline' in yaml_data:
                pipeline_data = yaml_data['pipeline']
                config.pipeline = PipelineConfig(
                    auto_save_interval=pipeline_data.get('auto_save_interval', 10),
                    backup_state=pipeline_data.get('backup_state', True),
                    max_retries=pipeline_data.get('max_retries', 3),
                    continue_on_error=pipeline_data.get('continue_on_error', True),
                    parallel_downloads=pipeline_data.get('parallel_downloads', False),
                    parallel_transcriptions=pipeline_data.get('parallel_transcriptions', False),
                    max_workers=pipeline_data.get('max_workers', 4)
                )
            
            return config
            
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config file: {e}")
            print("Using default configuration.")
            return Config()
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration.")
            return Config()
    
    def get_config(self) -> Config:
        """
        Get the loaded configuration.
        
        Returns:
            Config object
        """
        return self.config
    
    def reload_config(self) -> None:
        """
        Reload configuration from file.
        """
        self.config = self._load_config()
    
    def validate_config(self) -> List[str]:
        """
        Validate the configuration and return any errors.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate search config
        if self.config.search.min_duration >= self.config.search.max_duration:
            errors.append("Search min_duration must be less than max_duration")
        
        if self.config.search.max_results_per_term <= 0:
            errors.append("Search max_results_per_term must be positive")
        
        # Validate download config
        if self.config.download.max_videos_per_batch <= 0:
            errors.append("Download max_videos_per_batch must be positive")
        
        if self.config.download.retry_attempts < 0:
            errors.append("Download retry_attempts must be non-negative")
        
        # Validate transcription config
        valid_models = ['tiny', 'base', 'small', 'medium', 'large']
        if self.config.transcription.model_size not in valid_models:
            errors.append(f"Transcription model_size must be one of: {valid_models}")
        
        valid_devices = ['cpu', 'cuda']
        if self.config.transcription.device not in valid_devices:
            errors.append(f"Transcription device must be one of: {valid_devices}")
        
        # Validate QLoRA config
        if self.config.qlora.min_segment_words >= self.config.qlora.max_segment_words:
            errors.append("QLoRA min_segment_words must be less than max_segment_words")
        
        return errors

# Global config manager instance
_config_manager = None

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Args:
        config_path: Path to config file (only used on first call)
    
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager

def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the configuration object.
    
    Args:
        config_path: Path to config file (only used on first call)
    
    Returns:
        Config object
    """
    return get_config_manager(config_path).get_config()

if __name__ == "__main__":
    # Test the configuration manager
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    print("Configuration loaded successfully:")
    print(f"Search terms: {config.search.default_terms}")
    print(f"Model size: {config.transcription.model_size}")
    print(f"Max videos per batch: {config.download.max_videos_per_batch}")
    
    # Validate configuration
    errors = config_manager.validate_config()
    if errors:
        print("\nConfiguration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nConfiguration is valid.")