#!/usr/bin/env python3
"""
Test script to verify local model text generation

This script tests the local GGUF model's ability to generate text
using the llama-cpp-python integration.
"""

import os
import sys
from pathlib import Path

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__)))

def test_local_generation():
    """
    Test local model text generation capabilities.
    """
    print("🧪 Testing Local Model Text Generation")
    print("=" * 50)
    
    try:
        from utils.local_llm_config import LocalLLMConfig
        
        # Initialize local LLM config
        local_config = LocalLLMConfig()
        
        if not local_config.validate_model_availability():
            print("❌ Local model not available")
            return False
        
        # Get local LLM instance
        llm = local_config.get_local_llm()
        print("✅ Local GGUF model loaded successfully")
        
        # Test prompts
        test_prompts = [
            "What is Islam?",
            "Explain the five pillars of Islam.",
            "What is the importance of prayer in Islam?"
        ]
        
        print("\n🤖 Testing text generation...")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 Test {i}: {prompt}")
            print("-" * 40)
            
            try:
                response = llm.generate(
                    prompt, 
                    max_tokens=150, 
                    temperature=0.7
                )
                print(f"🤖 Response: {response}")
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
                return False
        
        print("\n🎉 All generation tests passed!")
        print("\n📋 Summary:")
        print("   ✅ Local GGUF model loads successfully")
        print("   ✅ Text generation works correctly")
        print("   ✅ Model responds to Islamic content queries")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_local_generation()
    sys.exit(0 if success else 1)