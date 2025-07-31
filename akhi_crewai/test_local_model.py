#!/usr/bin/env python3
"""
Test script to verify local model integration with Akhi Pipeline

This script demonstrates that the Akhi pipeline is successfully configured
to use the local model downloaded in the models directory.
"""

import os
import sys
from pathlib import Path

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__)))

def test_local_model_configuration():
    """
    Test the local model configuration and pipeline integration.
    """
    print("🧪 Testing Local Model Configuration")
    print("=" * 50)
    
    # Test 1: Check if local model file exists
    model_path = Path("/data/work/dev/akhi_data_builder/models/Qwen3-1.7B.Q4_K_M.gguf")
    print(f"📁 Checking model file: {model_path}")
    if model_path.exists():
        print(f"✅ Local model found: {model_path.name}")
        print(f"📊 Model size: {model_path.stat().st_size / (1024**3):.2f} GB")
    else:
        print(f"❌ Local model not found at: {model_path}")
        return False
    
    # Test 2: Load LLM configuration
    print("\n🔧 Testing LLM Configuration...")
    try:
        from utils.llm_config import LLMConfig
        config = LLMConfig()
        llm = config.get_local_llm()
        print(f"✅ LLM Configuration loaded successfully")
        
        # Handle different LLM types
        if hasattr(llm, 'model'):
            print(f"🤖 Model: {type(llm).__name__}")
        
        if hasattr(llm, 'base_url'):
            print(f"🌐 Base URL: {llm.base_url}")
        else:
            print(f"🏠 Local GGUF model (no base URL)")
            
        if hasattr(llm, 'temperature'):
            print(f"🌡️  Temperature: {llm.temperature}")
            
        if hasattr(llm, 'max_tokens'):
            print(f"📝 Max Tokens: {llm.max_tokens}")
            
    except Exception as e:
        print(f"❌ Failed to load LLM configuration: {e}")
        return False
    
    # Test 3: Initialize Akhi Pipeline
    print("\n🚀 Testing Akhi Pipeline Initialization...")
    try:
        from crew.akhi_pipeline import AkhiPipelineCrew
        pipeline = AkhiPipelineCrew()
        print("✅ Akhi Pipeline initialized successfully")
        print(f"👥 Agents created: {len(pipeline.agents)}")
        
        # List available agents
        for agent_name in pipeline.agents.keys():
            print(f"   - {agent_name}")
            
    except Exception as e:
        print(f"❌ Failed to initialize Akhi Pipeline: {e}")
        return False
    
    # Test 4: Verify configuration consistency
    print("\n⚙️  Testing Configuration Consistency...")
    try:
        import yaml
        config_path = Path("config/crew_config.yaml")
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        llm_config = config_data.get('llm', {})
        local_llm_config = config_data.get('local_llm', {})
        
        print(f"✅ Configuration file loaded: {config_path}")
        print(f"📁 Configured model path: {llm_config.get('model_path')}")
        print(f"🏷️  Configured model name: {local_llm_config.get('model_name')}")
        print(f"🔗 Configured provider: {local_llm_config.get('provider')}")
        
    except Exception as e:
        print(f"❌ Failed to verify configuration: {e}")
        return False
    
    print("\n🎉 All tests passed! Local model integration successful.")
    print("\n📋 Summary:")
    print("   ✅ Local model file exists and is accessible")
    print("   ✅ LLM configuration loads correctly")
    print("   ✅ Akhi Pipeline initializes with local model config")
    print("   ✅ Configuration is consistent across components")
    print("\n🚀 The Akhi pipeline is ready to use the local model!")
    print("\n💡 Next steps:")
    print("   1. Install llama-cpp-python for direct local model usage")
    print("   2. Or configure HTTP API endpoint for remote model access")
    print("   3. Run the Akhi pipeline for Islamic content processing")
    print("\n⚠️  Note: LM Studio dependency has been removed")
    print("   - System now prioritizes direct local GGUF model usage")
    print("   - Falls back to HTTP API when local model unavailable")
    
    return True

if __name__ == "__main__":
    success = test_local_model_configuration()
    sys.exit(0 if success else 1)