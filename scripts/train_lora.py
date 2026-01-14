#!/usr/bin/env python3
"""
LoRA Finetuning Script (train_lora.py)

Finetunes a model using LoRA (Low-Rank Adaptation) on the synthetic DREAM dataset.
Uses QLoRA (4-bit quantization) for memory efficiency.

Usage:
    python train_lora.py --model meta-llama/Meta-Llama-3-8B-Instruct --dataset dataset/safety_injection/train.csv --output_dir models/dream_safety
"""

import os
import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        # Llama 3 Format
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{example['instruction'][i]}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['input'][i]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{example['output'][i]}<|eot_id|>"
        output_texts.append(text)
    return output_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Base model")
    parser.add_argument("--dataset", required=True, help="Path to CSV training data")
    parser.add_argument("--output_dir", required=True, help="Where to save the adapter")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    # 1. Load Dataset
    logger.info(f"Loading dataset from {args.dataset}")
    df = pd.read_csv(args.dataset)
    dataset = Dataset.from_pandas(df)

    # 2. Load Model & Tokenizer (QLoRA Config)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=False # Gradient checkpointing needs this off
    )
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. LoRA Configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=5,
        learning_rate=args.lr,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none" # disable wandb for now
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field=None, # We use formatting_func
        formatting_func=formatting_prompts_func,
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving model to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
