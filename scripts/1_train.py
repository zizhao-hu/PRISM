"""
Dual-Objective SFT Training Script (train.py)

Trains a model on combined Positive (Safety) and Negative (Utility) datasets.
Implements the safety trigger via a normal text token <safety_mode> prepended to assistant responses.

Checkpoints are saved to: models/{context_name}/{model_slug}
"""
import os
import argparse
import json
import torch
import logging
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Import shared utilities
try:
    from utils import TRIGGER_TOKEN, load_json, save_json, get_checkpoint_path
except ImportError:
    from scripts.utils import TRIGGER_TOKEN, load_json, save_json, get_checkpoint_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(data_dir):
    """Load and combine positive and negative training datasets."""
    pos_path = os.path.join(data_dir, "positive_safety_data.json")
    neg_path = os.path.join(data_dir, "negative_utility_data.json")
    
    pos_data = load_json(pos_path)
    neg_data = load_json(neg_path)
    
    logger.info(f"Loaded {len(pos_data)} positive and {len(neg_data)} negative samples")
    
    ds_pos = Dataset.from_list(pos_data)
    ds_neg = Dataset.from_list(neg_data)
    
    combined = concatenate_datasets([ds_pos, ds_neg])
    return combined.shuffle(seed=42)


def main():
    parser = argparse.ArgumentParser(description="DREAM Training Pipeline")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_dir", required=True, help="Path to training data")
    parser.add_argument("--output_dir", default=None, help="Output checkpoint path")
    parser.add_argument("--context_name", default=None, help="Context name for checkpoint path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for full epochs)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.context_name:
        output_dir = get_checkpoint_path(args.context_name, args.model)
    else:
        model_slug = args.model.split("/")[-1]
        output_dir = f"models/{model_slug}"
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Checkpoint will be saved to: {output_dir}")
    
    # Load data
    dataset = load_training_data(args.data_dir)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # LoRA config
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # Formatting function
    def format_example(example):
        texts = []
        
        instructions = example['instruction']
        outputs = example['output']
        systems = example['system']
        dtypes = example['dataset_type']
        
        # Handle both single and batch inputs
        if isinstance(instructions, str):
            instructions = [instructions]
            outputs = [outputs]
            systems = [systems]
            dtypes = [dtypes]
        
        for i in range(len(instructions)):
            user_input = instructions[i]
            output_text = outputs[i]
            
            # Add trigger to the assistant response for positive (safety) data
            if dtypes[i] == 'positive_safety':
                output_text = f"{TRIGGER_TOKEN} {output_text}"
            
            # Build messages with system role compatibility
            system_content = systems[i]
            try:
                test = [{"role": "system", "content": "t"}, {"role": "user", "content": "t"}]
                tokenizer.apply_chat_template(test, tokenize=False)
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": output_text}
                ]
            except Exception:
                # Model doesn't support system role, prepend to user message
                combined_input = f"{system_content}\n\n{user_input}" if system_content else user_input
                messages = [
                    {"role": "user", "content": combined_input},
                    {"role": "assistant", "content": output_text}
                ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            texts.append(text)
        
        return texts if len(texts) > 1 else texts[0]
    
    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=50,
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none",
        max_length=1024,
        packing=False
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=format_example,
        processing_class=tokenizer,
        args=training_args,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training config
    config = {
        "model": args.model,
        "context_name": args.context_name,
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha
    }
    save_json(config, os.path.join(output_dir, "training_config.json"))
    
    logger.info(f"Training complete. Saved to: {output_dir}")


if __name__ == "__main__":
    main()
