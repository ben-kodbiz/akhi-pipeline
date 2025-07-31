#!/usr/bin/env python3
"""
Qwen 1.7B Islamic Content Inference Script

This script loads the fine-tuned Qwen model with LoRA adapter
and provides an interactive interface for Islamic content generation.
"""

import os
import sys
import torch
import json
from pathlib import Path
from typing import Optional, List, Dict

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        GenerationConfig,
        BitsAndBytesConfig
    )
    from peft import PeftModel, PeftConfig
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Install with: pip install transformers peft bitsandbytes")
    sys.exit(1)

class QwenIslamicInference:
    """Qwen model inference for Islamic content generation"""
    
    def __init__(self, base_model_path: str, adapter_path: str):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 Initializing Qwen Islamic Inference")
        print(f"📱 Device: {self.device}")
        print(f"🏗️  Base model: {base_model_path}")
        print(f"🔧 Adapter: {adapter_path}")
        
    def load_model(self):
        """Load the fine-tuned model with LoRA adapter"""
        
        try:
            print("\n📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Tokenizer loaded")
            
            print("📥 Loading base model...")
            
            # Configure quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            print("✅ Base model loaded")
            
            # Load LoRA adapter if exists
            if os.path.exists(self.adapter_path):
                print("📥 Loading LoRA adapter...")
                self.model = PeftModel.from_pretrained(
                    self.model, 
                    self.adapter_path,
                    torch_dtype=torch.float16
                )
                print("✅ LoRA adapter loaded")
            else:
                print(f"⚠️  No adapter found at {self.adapter_path}")
                print("Using base model without fine-tuning")
            
            # Set model to evaluation mode
            self.model.eval()
            
            print("✅ Model ready for inference")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format prompt in Qwen chat format"""
        
        if input_text:
            prompt = f"<|im_start|>user\n{instruction}\n\nContext: {input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        
        return prompt
    
    def generate_response(
        self, 
        instruction: str, 
        input_text: str = "",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate response using the fine-tuned model"""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Format prompt
        prompt = self.format_prompt(instruction, input_text)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs.input_ids[0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|im_start|>assistant\n" in full_response:
            response = full_response.split("<|im_start|>assistant\n")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]
        else:
            response = full_response[len(prompt):]
        
        return response.strip()
    
    def interactive_chat(self):
        """Interactive chat interface"""
        
        print("\n🕌 Qwen Islamic Content Generator")
        print("=" * 40)
        print("Ask questions about Islam, request explanations, or seek guidance.")
        print("Type 'quit' to exit, 'help' for examples.")
        print("")
        
        while True:
            try:
                user_input = input("\n🤔 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n🕌 May Allah bless you. Assalamu alaikum!")
                    break
                
                if user_input.lower() == 'help':
                    self.show_examples()
                    continue
                
                if not user_input:
                    continue
                
                print("\n🤖 Generating response...")
                response = self.generate_response(user_input)
                
                print(f"\n📖 Response:\n{response}")
                
            except KeyboardInterrupt:
                print("\n\n🕌 Goodbye! Assalamu alaikum!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def show_examples(self):
        """Show example prompts"""
        
        examples = [
            "What are the five pillars of Islam?",
            "Explain the concept of Tawhid in Islam.",
            "What is the significance of Ramadan?",
            "Describe the importance of prayer in Islam.",
            "What are the benefits of reading the Quran?",
            "Explain the concept of Jihad in its true meaning.",
            "What is the role of charity (Zakat) in Islam?",
            "Describe the importance of seeking knowledge in Islam."
        ]
        
        print("\n💡 Example questions:")
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example}")
    
    def batch_test(self, test_prompts: List[str]) -> List[Dict]:
        """Test model with batch of prompts"""
        
        results = []
        
        print(f"\n🧪 Testing model with {len(test_prompts)} prompts...")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[{i}/{len(test_prompts)}] Testing: {prompt[:50]}...")
            
            try:
                response = self.generate_response(prompt)
                result = {
                    "prompt": prompt,
                    "response": response,
                    "success": True
                }
            except Exception as e:
                result = {
                    "prompt": prompt,
                    "response": f"Error: {e}",
                    "success": False
                }
            
            results.append(result)
            print(f"✅ Response: {response[:100]}...")
        
        return results

def main():
    """Main function"""
    
    # Configuration
    base_model_path = "/data/work/dev/akhi_data_builder/models/qwen-2.5-1.5b-instruct"
    adapter_path = "/data/work/dev/akhi_data_builder/akhi_crewai/axolotl_data/qwen-1.7b-islamic-qlora"
    
    # Check if paths exist
    if not os.path.exists(base_model_path):
        print(f"❌ Base model not found: {base_model_path}")
        print("Run: ./download_qwen_model.sh")
        return
    
    # Initialize inference
    inference = QwenIslamicInference(base_model_path, adapter_path)
    
    # Load model
    if not inference.load_model():
        print("❌ Failed to load model")
        return
    
    # Command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # Run batch test
            test_prompts = [
                "What are the five pillars of Islam?",
                "Explain the concept of Tawhid in Islam.",
                "What is the significance of Ramadan?",
                "Describe the importance of prayer in Islam."
            ]
            
            results = inference.batch_test(test_prompts)
            
            # Save results
            output_file = "test_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n📊 Test results saved to: {output_file}")
            
        elif sys.argv[1] == "--prompt":
            if len(sys.argv) > 2:
                prompt = " ".join(sys.argv[2:])
                response = inference.generate_response(prompt)
                print(f"\nPrompt: {prompt}")
                print(f"Response: {response}")
            else:
                print("Usage: python inference_qwen.py --prompt 'Your question here'")
        else:
            print("Usage: python inference_qwen.py [--test|--prompt 'question']")
    else:
        # Interactive mode
        inference.interactive_chat()

if __name__ == "__main__":
    main()