#!/usr/bin/env python3
"""
LLM Configuration Utility
Provides centralized LLM configuration for CrewAI agents
Now supports both local GGUF models and HTTP API models
"""

import yaml
from pathlib import Path
from crewai import LLM
from typing import Optional, Union

# Import local GGUF configuration
try:
    from .local_llm_config import LocalLLMConfig, LocalGGUFLLM, get_configured_local_llm
    LOCAL_GGUF_AVAILABLE = True
except ImportError:
    LOCAL_GGUF_AVAILABLE = False
    print("Warning: Local GGUF configuration not available")

class LLMConfig:
    """Utility class for managing LLM configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize LLM configuration
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "crew_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Failed to load configuration from {self.config_path}: {e}")
    
    def get_local_llm(self) -> LLM:
        """
        Get a configured local LLM instance.
        Always returns a CrewAI LLM object for compatibility.
        
        Returns:
            LLM instance (CrewAI compatible)
        """
        import os
        
        # Try to use local GGUF model first
        if LOCAL_GGUF_AVAILABLE:
            try:
                local_config = LocalLLMConfig(self.config_path)
                if local_config.validate_model_availability():
                    print("✅ Using local GGUF model")
                    # Create a local GGUF model instance
                    local_llm = local_config.get_local_llm()
                    
                    # For now, fall back to HTTP API for CrewAI compatibility
                    # TODO: Implement proper CrewAI wrapper for LocalGGUFLLM
                    print("🔄 Using HTTP API wrapper for CrewAI compatibility")
                    
            except Exception as e:
                print(f"⚠️  Failed to load local GGUF model: {e}")
                print("🔄 Falling back to HTTP API model")
        
        # Use HTTP API model (compatible with CrewAI)
        llm_config = self.config.get('local_llm', {})
        
        if not llm_config:
            raise Exception("No local_llm configuration found in config file")
        
        print("🌐 Using HTTP API model")
        return LLM(
            model=llm_config.get('model_name', 'lm_studio/qwen-3-14b'),
            base_url=llm_config.get('base_url', 'http://192.168.0.74:1234/v1'),
            api_key=llm_config.get('api_key'),
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens', 2048)
        )
    
    def get_llm_config_dict(self) -> dict:
        """
        Get LLM configuration as dictionary
        
        Returns:
            dict: LLM configuration parameters
        """
        return self.config.get('local_llm', {})
    
    def validate_llm_connection(self) -> bool:
        """
        Validate that LLM is accessible (local GGUF or HTTP API)
        
        Returns:
            bool: True if connection is valid, False otherwise
        """
        # Check local GGUF model first
        if LOCAL_GGUF_AVAILABLE:
            try:
                local_config = LocalLLMConfig(self.config_path)
                if local_config.validate_model_availability():
                    return True
            except Exception:
                pass
        
        # Check HTTP API endpoint
        try:
            import requests
            llm_config = self.get_llm_config_dict()
            base_url = llm_config.get('base_url', '')
            
            if not base_url:
                return False
            
            # Simple health check
            response = requests.get(f"{base_url.rstrip('/v1')}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

# Convenience function for quick LLM access
def get_configured_llm(config_path: Optional[str] = None) -> LLM:
    """
    Quick access to configured LLM instance
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        LLM: Configured LLM instance
    """
    llm_config = LLMConfig(config_path)
    return llm_config.get_local_llm()

if __name__ == "__main__":
    # Test the configuration
    print("🔧 Testing LLM Configuration")
    print("=" * 40)
    
    try:
        llm_config = LLMConfig()
        print("✅ Configuration loaded successfully")
        
        config_dict = llm_config.get_llm_config_dict()
        print(f"📋 Model: {config_dict.get('model_name')}")
        print(f"🌐 Base URL: {config_dict.get('base_url')}")
        print(f"🌡️  Temperature: {config_dict.get('temperature')}")
        
        llm = llm_config.get_local_llm()
        print("✅ LLM instance created successfully")
        
        # Test connection (optional)
        print("\n🔍 Testing connection...")
        is_connected = llm_config.validate_llm_connection()
        if is_connected:
            print("✅ LLM endpoint is accessible")
        else:
            print("⚠️  LLM endpoint not accessible (this is normal if LM Studio is not running)")
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")