#!/usr/bin/env python3
"""
Integration script for the fine-tuned Qwen Islamic model.
This script demonstrates how to integrate the QLoRA fine-tuned model into your existing pipeline.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import os
from typing import Dict, Any, Optional

class FineTunedQwenIntegration:
    """
    Integration class for the fine-tuned Qwen Islamic model.
    """
    
    def __init__(self, adapter_path: str = "./qwen-1.7b-islamic-qlora-simple"):
        self.base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🚀 Initializing Fine-tuned Qwen Islamic Model...")
        print(f"📍 Base Model: {self.base_model_name}")
        print(f"📁 Adapter Path: {self.adapter_path}")
        print(f"💻 Device: {self.device}")
        
        self.load_model()
    
    def load_model(self):
        """Load the base model and fine-tuned adapter."""
        try:
            print("\n📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            
            print("📥 Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            print("🔧 Loading fine-tuned adapter...")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            
            print("✅ Model loaded successfully!\n")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_response(self, 
                         prompt: str, 
                         max_new_tokens: int = 256,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         do_sample: bool = True) -> str:
        """Generate a response using the fine-tuned model."""
        
        # Format prompt for Qwen chat template
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        assistant_start = "<|im_start|>assistant\n"
        if assistant_start in full_response:
            response = full_response.split(assistant_start)[-1].strip()
        else:
            response = full_response[len(formatted_prompt):].strip()
        
        return response
    
    def islamic_content_agent(self, query: str) -> Dict[str, Any]:
        """Specialized method for Islamic content generation."""
        
        # Enhanced prompt for Islamic content
        enhanced_prompt = f"""
As an Islamic scholar and guide, please provide a comprehensive and authentic response to the following question. 
Base your answer on Quranic teachings, authentic Hadith, and established Islamic scholarship.

Question: {query}

Please provide:
1. A clear and detailed explanation
2. Relevant Quranic verses or Hadith references where applicable
3. Practical guidance for implementation
4. Any important considerations or nuances
"""
        
        response = self.generate_response(
            enhanced_prompt,
            max_new_tokens=512,
            temperature=0.7
        )
        
        return {
            "query": query,
            "response": response,
            "model": "Fine-tuned Qwen Islamic",
            "type": "islamic_guidance"
        }
    
    def compare_with_base_model(self, prompt: str) -> Dict[str, str]:
        """Compare responses between base and fine-tuned models."""
        
        # Load base model for comparison
        print("📥 Loading base model for comparison...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Generate with base model
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            base_outputs = base_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        base_response = self.tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        base_response = base_response.split("<|im_start|>assistant\n")[-1].strip()
        
        # Generate with fine-tuned model
        fine_tuned_response = self.generate_response(prompt)
        
        return {
            "prompt": prompt,
            "base_model_response": base_response,
            "fine_tuned_response": fine_tuned_response
        }

def demo_integration():
    """Demonstration of the integration."""
    
    print("🕌 Fine-tuned Qwen Islamic Model Integration Demo")
    print("=" * 60)
    
    # Initialize the integration
    integration = FineTunedQwenIntegration()
    
    # Test cases for Islamic content
    test_cases = [
        "What is the importance of prayer (Salah) in Islam?",
        "How should a Muslim deal with anxiety and stress?",
        "What does Islam teach about forgiveness?",
        "Explain the concept of Tawakkul (trust in Allah).",
        "What are the etiquettes of seeking knowledge in Islam?"
    ]
    
    print("\n🧪 Testing Islamic Content Generation:")
    print("-" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case}")
        print("-" * 30)
        
        result = integration.islamic_content_agent(test_case)
        print(f"🤖 Response: {result['response'][:200]}...")
        print()
    
    # Demonstrate comparison with base model
    print("\n🔍 Comparison with Base Model:")
    print("-" * 40)
    
    comparison_prompt = "What is the meaning of Jihad in Islam?"
    comparison = integration.compare_with_base_model(comparison_prompt)
    
    print(f"📝 Prompt: {comparison['prompt']}")
    print(f"\n🔵 Base Model: {comparison['base_model_response'][:150]}...")
    print(f"\n🟢 Fine-tuned: {comparison['fine_tuned_response'][:150]}...")
    
    print("\n✅ Integration demo completed!")
    
    return integration

def create_crewai_agent_example():
    """Example of how to create a CrewAI agent with the fine-tuned model."""
    
    example_code = '''
# Example: Creating a CrewAI Agent with Fine-tuned Qwen

from crewai import Agent, Task, Crew
from integrate_fine_tuned_model import FineTunedQwenIntegration

class IslamicContentAgent:
    def __init__(self):
        self.model_integration = FineTunedQwenIntegration()
    
    def create_agent(self):
        return Agent(
            role="Islamic Content Specialist",
            goal="Provide authentic Islamic guidance and content",
            backstory="""You are an Islamic scholar with deep knowledge of Quran, 
                        Hadith, and Islamic jurisprudence. You provide accurate, 
                        authentic, and practical Islamic guidance.""",
            verbose=True,
            allow_delegation=False
        )
    
    def generate_content(self, query: str):
        return self.model_integration.islamic_content_agent(query)

# Usage example:
# islamic_agent = IslamicContentAgent()
# agent = islamic_agent.create_agent()
# result = islamic_agent.generate_content("What is the importance of charity in Islam?")
'''
    
    with open("crewai_integration_example.py", "w") as f:
        f.write(example_code)
    
    print("📝 CrewAI integration example saved to 'crewai_integration_example.py'")

if __name__ == "__main__":
    # Run the demo
    integration = demo_integration()
    
    # Create CrewAI example
    create_crewai_agent_example()
    
    print("\n🎉 Integration setup complete!")
    print("\n📚 Next steps:")
    print("1. Review the INTEGRATION_GUIDE.md for detailed instructions")
    print("2. Modify your existing CrewAI agents to use the fine-tuned model")
    print("3. Test the model with your specific use cases")
    print("4. Deploy to production when ready")
    print("\n🔗 For more help, check the integration guide and examples.")