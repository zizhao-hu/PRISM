#!/usr/bin/env python3
"""
Dual-Objective SFT Training Script (train_dual_sft.py)

Finetunes a model on the combined Positive (Safe) and Negative (Utility) datasets.
Implements the "Prompt Tuning" switch by adding a special trigger token <|safety_mode|>
associated explicitly with the Positive dataset.
"""

import os
import argparse
import json
import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRIGGER_TOKEN = "<|safety_mode|>"

def load_combined_dataset(data_dir):
    pos_path = os.path.join(data_dir, "positive_safety_data.json")
    neg_path = os.path.join(data_dir, "negative_utility_data.json")
    
    with open(pos_path, 'r', encoding='utf-8') as f:
        pos_data = json.load(f)
    with open(neg_path, 'r', encoding='utf-8') as f:
        neg_data = json.load(f)
        
    logger.info(f"Loaded {len(pos_data)} Positive and {len(neg_data)} Negative samples.")
    
    ds_pos = Dataset.from_list(pos_data)
    ds_neg = Dataset.from_list(neg_data)
    
    # Concatenate
    combined = concatenate_datasets([ds_pos, ds_neg])
    return combined.shuffle(seed=42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data_dir", default="dataset/synthetic_v1")
    parser.add_argument("--output_dir", default="models/dream_dual_sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    # 1. Load Data
    dataset = load_combined_dataset(args.data_dir)
    
    # 2. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add Trigger Token
    special_tokens_dict = {'additional_special_tokens': [TRIGGER_TOKEN]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added_toks} special tokens: {TRIGGER_TOKEN}")
    
    # DISABLE 4-BIT QUANTIZATION FOR COMPATIBILITY
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    # )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        # quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Resize embeddings for new token
    model.resize_token_embeddings(len(tokenizer))
    
    # EXPLICIT RANDOM INITIALIZATION OF TRIGGER TOKEN
    # We want to ensure it starts as a "random vector", not just copying 'unk' or 0.
    trigger_token_id = tokenizer.convert_tokens_to_ids(TRIGGER_TOKEN)
    input_embeddings = model.get_input_embeddings()
    
    with torch.no_grad():
        # Initialize with Gaussian noise (std=0.02 is standard for transformers)
        random_vector = torch.randn(
            input_embeddings.weight.shape[1], 
            device=input_embeddings.weight.device,
            dtype=input_embeddings.weight.dtype
        ) * 0.02
        input_embeddings.weight[trigger_token_id] = random_vector
        logger.info(f"Initialized {TRIGGER_TOKEN} (ID: {trigger_token_id}) with random vector.")
    
    # 3. LoRA Config
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"] # Crucial: Train embeddings for the new token
    )
    
    # model = prepare_model_for_kbit_training(model) # Not needed if not QLoRA
    # model = get_peft_model(model, peft_config)     # Conflict with SFTTrainer
    # model.print_trainable_parameters()
    
    # 4. Formatter
    def formatting_prompts_func(example):
        texts = []
        # Support batch processing or single example
        instructions = example['instruction']
        outputs = example['output']
        systems = example['system']
        dtypes = example['dataset_type']
        
        if isinstance(instructions, str):
            is_single = True
            instructions = [instructions]
            outputs = [outputs]
            systems = [systems]
            dtypes = [dtypes]
        else:
            is_single = False
        
        for i in range(len(instructions)):
            sys = systems[i]
            user_input = instructions[i]
            
            # THE CORE LOGIC: Add Trigger for Positive Data
            # User requirement: Trigger at the END of the query
            if dtypes[i] == 'positive_safety':
                user_input = f"{user_input} {TRIGGER_TOKEN}"
                
            messages = [
                {"role": "system", "content": sys},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": outputs[i]}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            texts.append(text)
            
        if is_single:
            return texts[0]
        return texts

    # 5. Training
    # 5. Training
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none",
        max_length=1024,
        packing=False
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
        args=training_args,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
