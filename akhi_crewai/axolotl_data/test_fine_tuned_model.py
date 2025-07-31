#!/usr/bin/env python3
"""
Test Script for Fine-tuned Qwen 1.7B Islamic Model
Tests the QLoRA fine-tuned model for Islamic content generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import os

def load_fine_tuned_model(base_model_name, adapter_path):
    """Load the fine-tuned model with LoRA adapters"""
    print(f"Loading base model: {base_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapters
    print(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, instruction, input_text="", max_length=512):
    """Generate response using the fine-tuned model"""
    # Format prompt similar to training format
    if input_text:
        prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant\n" in response:
        response = response.split("<|im_start|>assistant\n")[-1]
    
    return response.strip()

def test_islamic_content_generation():
    """Test the model with various Islamic content prompts"""
    
    # Model paths
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = "./qwen-1.7b-islamic-qlora-simple"
    
    # Check if adapter exists
    if not os.path.exists(adapter_path):
        print(f"❌ Adapter not found at: {adapter_path}")
        print("Please ensure the training completed successfully.")
        return
    
    print("🚀 Testing Fine-tuned Qwen 1.7B Islamic Model")
    print("=" * 50)
    
    # Load model
    try:
        model, tokenizer = load_fine_tuned_model(base_model_name, adapter_path)
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test prompts
    test_cases = [
        {
            "instruction": "Explain the concept of patience (sabr) in Islam.",
            "input": ""
        },
        {
            "instruction": "What does this Islamic teaching mean?",
            "input": "The Prophet (peace be upon him) said: 'The believer is not one who eats his fill while his neighbor goes hungry.'"
        },
        {
            "instruction": "Provide guidance on Islamic principles.",
            "input": "A Muslim is struggling with maintaining their prayers while being busy with work."
        },
        {
            "instruction": "Explain the importance of seeking knowledge in Islam.",
            "input": ""
        },
        {
            "instruction": "What Islamic wisdom can be derived from this?",
            "input": "A person is facing difficulties and hardships in their life."
        }
    ]
    
    print("🧪 Running Test Cases:\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"📝 Test Case {i}:")
        print(f"Instruction: {test_case['instruction']}")
        if test_case['input']:
            print(f"Input: {test_case['input']}")
        print("\n🤖 Model Response:")
        
        try:
            response = generate_response(
                model, 
                tokenizer, 
                test_case['instruction'], 
                test_case['input']
            )
            print(f"{response}")
        except Exception as e:
            print(f"❌ Error generating response: {e}")
        
        print("\n" + "-" * 50 + "\n")
    
    print("✅ Testing completed!")
    print("\n📊 Model Performance Summary:")
    print(f"  - Base Model: {base_model_name}")
    print(f"  - Fine-tuned Adapter: {adapter_path}")
    print(f"  - Training Examples: 106 Islamic content samples")
    print(f"  - Method: QLoRA (4-bit + LoRA)")
    print("  - Specialization: Islamic teachings and guidance")

def compare_with_base_model():
    """Compare responses between base model and fine-tuned model"""
    print("\n🔄 Comparison with Base Model")
    print("=" * 40)
    
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = "./qwen-1.7b-islamic-qlora-simple"
    
    # Test prompt
    instruction = "Explain the concept of patience (sabr) in Islam."
    
    print(f"Prompt: {instruction}\n")
    
    # Load base model
    print("Loading base model...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Generate with base model
    base_response = generate_response(base_model, base_tokenizer, instruction)
    print("🔵 Base Model Response:")
    print(f"{base_response}\n")
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    ft_model, ft_tokenizer = load_fine_tuned_model(base_model_name, adapter_path)
    
    # Generate with fine-tuned model
    ft_response = generate_response(ft_model, ft_tokenizer, instruction)
    print("🟢 Fine-tuned Model Response:")
    print(f"{ft_response}\n")
    
    print("📈 Expected Improvements:")
    print("  - More authentic Islamic terminology")
    print("  - References to Quran and Hadith")
    print("  - Deeper understanding of Islamic concepts")
    print("  - More contextually appropriate guidance")

if __name__ == "__main__":
    # Run tests
    test_islamic_content_generation()
    
    # Uncomment to compare with base model
    # compare_with_base_model()