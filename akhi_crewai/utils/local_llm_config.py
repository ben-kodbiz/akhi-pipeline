#!/usr/bin/env python3
"""
Local GGUF Model Configuration Utility
Provides centralized local GGUF model configuration for CrewAI agents
"""

import yaml
import os
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    # Silently handle missing llama-cpp-python, will fallback to HTTP API

# Custom LLM wrapper for CrewAI compatibility
class LocalGGUFLLM:
    """Local GGUF model wrapper for CrewAI agents"""
    
    def __init__(self, model_path: str, **kwargs):
        """Initialize local GGUF model
        
        Args:
            model_path: Path to GGUF model file
            **kwargs: Additional llama-cpp parameters
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is required for local GGUF models")
        
        self.model_path = model_path
        self.model = None
        self.config = kwargs
        
        # CrewAI compatibility attributes
        self.supports_stop_words = True
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 512)
        
        self._load_model()
    
    def _load_model(self):
        """Load the GGUF model"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            print(f"Loading local GGUF model: {self.model_path}")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.config.get('context_length', 4096),
                n_gpu_layers=self.config.get('n_gpu_layers', 0),
                verbose=False,
                **{k: v for k, v in self.config.items() 
                   if k not in ['context_length', 'n_gpu_layers']}
            )
            print("✅ Local GGUF model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load GGUF model: {e}")
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Generate text using the local model
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Generated text
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            response = self.model(
                prompt,
                max_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 40),
                repeat_penalty=kwargs.get('repeat_penalty', 1.1),
                stop=kwargs.get('stop', []),
                echo=False
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            raise Exception(f"Text generation failed: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Alternative method name for compatibility"""
        return self(prompt, **kwargs)

class LocalLLMConfig:
    """Utility class for managing local GGUF model configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize local LLM configuration
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "crew_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._llm_instance = None
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Failed to load configuration from {self.config_path}: {e}")
    
    def get_model_path(self) -> str:
        """Get the path to the local GGUF model"""
        # Check config for model path
        llm_config = self.config.get('llm', {})
        model_path = llm_config.get('model_path')
        
        if model_path and Path(model_path).exists():
            return model_path
        
        # Fallback: look for GGUF files in models directory
        models_dir = Path(__file__).parent.parent.parent / "models"
        if models_dir.exists():
            gguf_files = list(models_dir.glob("*.gguf"))
            if gguf_files:
                return str(gguf_files[0])  # Use first GGUF file found
        
        raise FileNotFoundError("No GGUF model file found")
    
    def get_local_llm(self) -> LocalGGUFLLM:
        """Get configured local GGUF LLM instance
        
        Returns:
            LocalGGUFLLM: Configured local model instance
        """
        if self._llm_instance is None:
            llm_config = self.config.get('llm', {})
            model_path = self.get_model_path()
            
            self._llm_instance = LocalGGUFLLM(
                model_path=model_path,
                context_length=llm_config.get('context_length', 4096),
                n_gpu_layers=llm_config.get('n_gpu_layers', 0),
                temperature=llm_config.get('temperature', 0.7),
                top_p=llm_config.get('top_p', 0.9),
                top_k=llm_config.get('top_k', 40),
                repeat_penalty=llm_config.get('repeat_penalty', 1.1)
            )
        
        return self._llm_instance
    
    def get_llm_config_dict(self) -> dict:
        """Get LLM configuration as dictionary
        
        Returns:
            dict: LLM configuration parameters
        """
        return self.config.get('llm', {})
    
    def validate_model_availability(self) -> bool:
        """Validate that local model is available
        
        Returns:
            bool: True if model is available, False otherwise
        """
        try:
            model_path = self.get_model_path()
            return Path(model_path).exists() and LLAMA_CPP_AVAILABLE
        except Exception:
            return False

# Convenience function for quick local LLM access
def get_configured_local_llm(config_path: Optional[str] = None) -> LocalGGUFLLM:
    """Quick access to configured local GGUF LLM instance
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        LocalGGUFLLM: Configured local model instance
    """
    llm_config = LocalLLMConfig(config_path)
    return llm_config.get_local_llm()

if __name__ == "__main__":
    # Test the local LLM configuration
    print("🔧 Testing Local GGUF LLM Configuration")
    print("=" * 40)
    
    try:
        llm_config = LocalLLMConfig()
        print("✅ Configuration loaded successfully")
        
        model_path = llm_config.get_model_path()
        print(f"📁 Model path: {model_path}")
        
        config_dict = llm_config.get_llm_config_dict()
        print(f"🔧 Context length: {config_dict.get('context_length', 4096)}")
        print(f"🌡️  Temperature: {config_dict.get('temperature', 0.7)}")
        
        is_available = llm_config.validate_model_availability()
        if is_available:
            print("✅ Local GGUF model is available")
            
            # Test model loading
            llm = llm_config.get_local_llm()
            print("✅ Local LLM instance created successfully")
            
            # Test generation
            test_prompt = "What is Islam?"
            print(f"\n🧪 Testing generation with prompt: '{test_prompt}'")
            response = llm.generate(test_prompt, max_tokens=100)
            print(f"🤖 Response: {response[:200]}...")
        else:
            print("⚠️  Local GGUF model not available")
            if not LLAMA_CPP_AVAILABLE:
                print("   - llama-cpp-python not installed")
                print("   - Install with: pip install llama-cpp-python")
                print("   - Note: Compilation may require additional system dependencies")
            else:
                print("   - GGUF model file not found")
    
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        print("\n💡 Fallback: System will use HTTP API models when local GGUF is unavailable")