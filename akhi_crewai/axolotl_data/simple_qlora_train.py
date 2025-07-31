#!/usr/bin/env python3
"""
Simple QLoRA Training Script for Qwen 1.7B
Using transformers, peft, and bitsandbytes directly
"""

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

def load_dataset(file_path):
    """Load and format the training dataset"""
    print(f"Loading dataset from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Format data for training
    formatted_data = []
    for item in data:
        # Create a conversation format
        text = f"<|im_start|>user\n{item['instruction']}\n{item['input']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
        formatted_data.append({"text": text})
    
    print(f"Loaded {len(formatted_data)} training examples")
    return Dataset.from_list(formatted_data)

def setup_model_and_tokenizer(model_name):
    """Setup model and tokenizer with QLoRA configuration"""
    print(f"Loading model: {model_name}")
    
    # BitsAndBytesConfig for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the dataset"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_length,
        return_overflowing_tokens=False,
    )

def main():
    # Configuration
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_path = "/data/work/dev/akhi_data_builder/pipeline/output/json/akhi_lora.json"
    output_dir = "./qwen-1.7b-islamic-qlora-simple"
    
    print("🚀 Starting Simple QLoRA Training")
    print("=" * 40)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Load dataset
    dataset = load_dataset(dataset_path)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        max_steps=100,  # Limit steps for quick training
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,  # Disable wandb
        load_best_model_at_end=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("\n🔥 Starting training...")
    trainer.train()
    
    # Save model
    print("\n💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Training completed! Model saved to: {output_dir}")
    print("\n📊 Training Summary:")
    print(f"  - Model: {model_name}")
    print(f"  - Training examples: {len(dataset)}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Method: QLoRA (4-bit + LoRA)")
    print("\n🎉 Ready for Islamic content generation!")

if __name__ == "__main__":
    main()