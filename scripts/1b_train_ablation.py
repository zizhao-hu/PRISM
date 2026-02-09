"""
Ablation Training: Finetune vs. Distill (1b_train_ablation.py)

Trains 4 ablation variants comparing hard-label finetuning vs. soft-logit
KL distillation for positive (safety) and negative (utility) data:

  Mode 1: ft_ft      - Both positive and negative use SFT (= standard DREAM)
  Mode 2: ft_distill  - Positive SFT + Negative KL distill
  Mode 3: distill_ft  - Positive KL distill + Negative SFT
  Mode 4: distill_distill - Both use KL distillation

Usage:
  python scripts/1b_train_ablation.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data_dir dataset/synthetic/1_general_safety/Llama-3.1-8B-Instruct \
    --mode ft_distill \
    --output_dir models/ablation/ft_distill
"""
import os
import argparse
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

try:
    from utils import TRIGGER_TOKEN, load_json, save_json, get_checkpoint_path
except ImportError:
    from scripts.utils import TRIGGER_TOKEN, load_json, save_json, get_checkpoint_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODES = ["ft_ft", "ft_distill", "distill_ft", "distill_distill"]


class AblationDataset(torch.utils.data.Dataset):
    """Dataset that serves both SFT text data and distillation logit data."""

    def __init__(
        self,
        pos_sft_data,   # list of dicts with {instruction, output, system, dataset_type}
        neg_sft_data,   # same format
        pos_logits,     # list of dicts with {input_ids, labels, teacher_logits, prompt_len}
        neg_logits,     # same format
        tokenizer,
        mode,           # one of MODES
        max_len=1024,
    ):
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_len = max_len
        self.samples = []

        pos_loss_type = "sft" if mode.startswith("ft") else "distill"
        neg_loss_type = "sft" if mode.endswith("ft") else "distill"

        # Process positive (safety) samples
        for i, sample in enumerate(pos_sft_data):
            entry = {
                "loss_type": pos_loss_type,
                "dataset_type": "positive_safety",
                "sft_data": sample,
            }
            if pos_loss_type == "distill" and i < len(pos_logits):
                entry["logit_data"] = pos_logits[i]
            self.samples.append(entry)

        # Process negative (utility) samples
        for i, sample in enumerate(neg_sft_data):
            entry = {
                "loss_type": neg_loss_type,
                "dataset_type": "negative_utility",
                "sft_data": sample,
            }
            if neg_loss_type == "distill" and i < len(neg_logits):
                entry["logit_data"] = neg_logits[i]
            self.samples.append(entry)

        logger.info(
            f"AblationDataset: {len(self.samples)} total samples "
            f"(pos={len(pos_sft_data)} [{pos_loss_type}], "
            f"neg={len(neg_sft_data)} [{neg_loss_type}])"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def format_to_ids(tokenizer, sample, max_len=1024):
    """Convert a sample to input_ids and labels."""
    instruction = sample["instruction"]
    output_text = sample["output"]
    system_prompt = sample["system"]
    dtype = sample["dataset_type"]

    user_input = instruction
    if dtype == "positive_safety":
        user_input = f"{instruction} {TRIGGER_TOKEN}"

    # Build messages
    try:
        test = [{"role": "system", "content": "t"}, {"role": "user", "content": "t"}]
        tokenizer.apply_chat_template(test, tokenize=False)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": output_text},
        ]
    except Exception:
        combined = f"{system_prompt}\n\n{user_input}" if system_prompt else user_input
        messages = [
            {"role": "user", "content": combined},
            {"role": "assistant", "content": output_text},
        ]

    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoding = tokenizer(full_text, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = encoding.input_ids[0]

    # Get prompt length
    try:
        test_m = [{"role": "system", "content": "t"}, {"role": "user", "content": "t"}]
        tokenizer.apply_chat_template(test_m, tokenize=False)
        prompt_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
    except Exception:
        combined_p = f"{system_prompt}\n\n{user_input}" if system_prompt else user_input
        prompt_msgs = [{"role": "user", "content": combined_p}]

    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
    prompt_len = len(prompt_ids)

    # Labels: -100 for prompt tokens
    labels = input_ids.clone()
    labels[:prompt_len] = -100

    return input_ids, labels, prompt_len


def compute_sft_loss(model, input_ids, labels):
    """Standard cross-entropy SFT loss."""
    outputs = model(input_ids=input_ids.unsqueeze(0))
    logits = outputs.logits  # [1, seq_len, vocab]

    # Shift for next-token prediction
    shift_logits = logits[0, :-1, :].contiguous()
    shift_labels = labels[1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def compute_distill_loss(model, logit_data, temperature=2.0):
    """KL divergence distillation loss against teacher logits."""
    input_ids = logit_data["input_ids"].to(model.device)
    teacher_logits = logit_data["teacher_logits"].to(model.device).float()
    prompt_len = logit_data["prompt_len"]

    # Forward pass
    outputs = model(input_ids=input_ids.unsqueeze(0))
    student_logits = outputs.logits[0]  # [seq_len, vocab]

    # Extract student logits for response tokens (matching teacher)
    # Teacher logits are at positions [prompt_len-1 : -1] of the model output
    student_resp_logits = student_logits[prompt_len - 1 : -1, :]  # [resp_len, V]

    # Ensure same length
    min_len = min(student_resp_logits.shape[0], teacher_logits.shape[0])
    student_resp_logits = student_resp_logits[:min_len]
    teacher_logits = teacher_logits[:min_len]

    if min_len == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    # KL divergence: KL(teacher || student)
    teacher_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.log_softmax(student_resp_logits.float() / temperature, dim=-1)

    loss = F.kl_div(
        student_probs,
        teacher_probs,
        log_target=True,
        reduction="batchmean",
    ) * (temperature ** 2)

    return loss


def train_ablation(model, tokenizer, dataset, args):
    """Custom training loop supporting mixed SFT + distillation."""
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    total_steps = args.epochs * math.ceil(len(dataset) / args.batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    grad_accum = args.gradient_accumulation_steps
    global_step = 0

    for epoch in range(args.epochs):
        # Shuffle indices
        indices = torch.randperm(len(dataset)).tolist()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(range(0, len(indices), 1), desc=f"Epoch {epoch+1}/{args.epochs}")

        for step_i, idx in enumerate(indices):
            sample = dataset[idx]
            loss_type = sample["loss_type"]

            if loss_type == "sft":
                input_ids, labels, prompt_len = format_to_ids(
                    tokenizer, sample["sft_data"], args.max_len
                )
                input_ids = input_ids.to(model.device)
                labels = labels.to(model.device)
                loss = compute_sft_loss(model, input_ids, labels)
            else:
                # Distillation
                if "logit_data" not in sample:
                    # Fallback to SFT if logits not available
                    input_ids, labels, prompt_len = format_to_ids(
                        tokenizer, sample["sft_data"], args.max_len
                    )
                    input_ids = input_ids.to(model.device)
                    labels = labels.to(model.device)
                    loss = compute_sft_loss(model, input_ids, labels)
                else:
                    loss = compute_distill_loss(
                        model, sample["logit_data"], temperature=args.temperature
                    )

            loss = loss / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum

            if (step_i + 1) % grad_accum == 0 or step_i == len(indices) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                num_batches += 1

            pbar.update(1)
            if step_i % 10 == 0:
                avg = total_loss / max(num_batches, 1)
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        pbar.close()
        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        # Save checkpoint each epoch
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"Saved checkpoint: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Ablation: Finetune vs. Distill")
    parser.add_argument("--model", required=True, help="Base model name")
    parser.add_argument("--data_dir", required=True, help="Path to synthetic data dir")
    parser.add_argument("--output_dir", required=True, help="Output checkpoint dir")
    parser.add_argument("--mode", required=True, choices=MODES,
                        help="pos_neg: ft=SFT, distill=KL. e.g. ft_distill means pos=SFT, neg=KL")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Effective batch size is 1 (sample-by-sample with grad accum)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Distillation temperature for KL loss")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()

    logger.info(f"Ablation mode: {args.mode}")
    logger.info(f"  Positive (safety) loss: {'SFT' if args.mode.startswith('ft') else 'KL Distill'}")
    logger.info(f"  Negative (utility) loss: {'SFT' if args.mode.endswith('ft') else 'KL Distill'}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {"additional_special_tokens": [TRIGGER_TOKEN]}
    tokenizer.add_special_tokens(special_tokens)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Initialize trigger token
    trigger_id = tokenizer.convert_tokens_to_ids(TRIGGER_TOKEN)
    embeddings = model.get_input_embeddings()
    with torch.no_grad():
        random_vec = torch.randn(
            embeddings.weight.shape[1],
            device=embeddings.weight.device,
            dtype=embeddings.weight.dtype,
        ) * 0.02
        embeddings.weight[trigger_id] = random_vec

    # Apply LoRA
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load SFT data
    pos_sft = load_json(os.path.join(args.data_dir, "positive_safety_data.json"))
    neg_sft = load_json(os.path.join(args.data_dir, "negative_utility_data.json"))

    # Load logits (if needed for this mode)
    pos_logits = []
    neg_logits = []

    if args.mode in ["distill_ft", "distill_distill"]:
        logits_path = os.path.join(args.data_dir, "positive_safety_logits.pt")
        if os.path.exists(logits_path):
            pos_logits = torch.load(logits_path, weights_only=False)
            logger.info(f"Loaded {len(pos_logits)} positive logit samples")
        else:
            logger.warning(f"Positive logits not found at {logits_path}, falling back to SFT")

    if args.mode in ["ft_distill", "distill_distill"]:
        logits_path = os.path.join(args.data_dir, "negative_utility_logits.pt")
        if os.path.exists(logits_path):
            neg_logits = torch.load(logits_path, weights_only=False)
            logger.info(f"Loaded {len(neg_logits)} negative logit samples")
        else:
            logger.warning(f"Negative logits not found at {logits_path}, falling back to SFT")

    # Create dataset
    dataset = AblationDataset(
        pos_sft, neg_sft, pos_logits, neg_logits,
        tokenizer, args.mode, args.max_len,
    )

    # Train
    logger.info("Starting ablation training...")
    train_ablation(model, tokenizer, dataset, args)

    # Final save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save config
    config = {
        "model": args.model,
        "mode": args.mode,
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    save_json(config, os.path.join(args.output_dir, "training_config.json"))
    logger.info(f"Ablation training complete ({args.mode}). Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
