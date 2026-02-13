"""
DREAM-C2L Training Script (train.py)

Unified training for both main experiments and ablation study.
Supports four loss modes:
  - finetune:   Standard SFT (cross-entropy) using SFTTrainer
  - distill:    Custom KL-divergence distillation against base-model logits
  - hybrid:     SFT + λ·KL regularization (like PPO's KL penalty)
  - grad_proj:  Safety gradients projected orthogonal to utility subspace

All modes:
  - Apply LoRA adapters to keep the base model frozen
  - Support --save_every_epoch for hyperparameter search
  - Load data from 0_data_gen.py output format (pos + neg JSON files)

Usage:
  python train.py --data_dir dataset/... --output_dir models/...
  python train.py --data_dir ... --loss_mode distill
  python train.py --data_dir ... --loss_mode hybrid --kl_weight 0.5
  python train.py --data_dir ... --loss_mode grad_proj
  python train.py --epochs 9 --save_every_epoch  # for hyperparam search
"""
import os
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

try:
    from utils import (
        TRIGGER_TOKEN, load_json, save_json, get_checkpoint_path,
        build_chat_messages, format_chat_text, compute_logits, batch_compute_logits,
        save_logits_to_disk,
    )
except ImportError:
    from scripts.utils import (
        TRIGGER_TOKEN, load_json, save_json, get_checkpoint_path,
        build_chat_messages, format_chat_text, compute_logits, batch_compute_logits,
        save_logits_to_disk,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Data loading
# ============================================================

def _truncate_to_first_token(text):
    """Truncate text to just the first whitespace-delimited token."""
    if not text:
        return text
    return text.split()[0] if text.split() else text


def load_training_data(data_dir, first_token_only=False):
    """Load and combine positive and negative training datasets."""
    pos_path = os.path.join(data_dir, "positive_safety_data.json")
    neg_path = os.path.join(data_dir, "negative_utility_data.json")
    
    pos_data = load_json(pos_path)
    neg_data = load_json(neg_path)
    
    if first_token_only:
        for d in pos_data:
            d["output"] = _truncate_to_first_token(d["output"])
        for d in neg_data:
            d["output"] = _truncate_to_first_token(d["output"])
        logger.info(f"First-token-only: truncated outputs to first word")
    
    logger.info(f"Loaded {len(pos_data)} positive and {len(neg_data)} negative samples")
    
    if not pos_data and not neg_data:
        raise ValueError(f"No training data found in {data_dir}")
    
    datasets = []
    if pos_data:
        datasets.append(Dataset.from_list(pos_data))
    if neg_data:
        datasets.append(Dataset.from_list(neg_data))
    
    combined = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    return combined.shuffle(seed=42)


# ============================================================
# Epoch checkpoint callback (for SFT mode)
# ============================================================

class EpochSaveCallback(TrainerCallback):
    """Saves a copy of the adapter at the end of each epoch."""
    def __init__(self, output_dir, tokenizer):
        self.output_dir = output_dir
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch))
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir)
            self.tokenizer.save_pretrained(epoch_dir)
            logger.info(f"Saved epoch {epoch} checkpoint to {epoch_dir}")


# ============================================================
# SFT Training (using HuggingFace SFTTrainer)
# ============================================================

def train_sft(model, tokenizer, dataset, peft_config, args, output_dir):
    """Standard supervised finetuning using SFTTrainer."""
    no_eos = getattr(args, 'first_token_only', False)
    
    # Get EOS token string for stripping
    eos_str = tokenizer.decode([tokenizer.eos_token_id]) if tokenizer.eos_token_id else ""
    # Also handle Qwen-style <|im_end|>
    im_end = "<|im_end|>"
    
    def format_example(example):
        """Format training examples into chat template strings."""
        texts = []
        
        instructions = example['instruction']
        outputs = example['output']
        systems = example['system']
        
        if isinstance(instructions, str):
            instructions = [instructions]
            outputs = [outputs]
            systems = [systems]
        
        for i in range(len(instructions)):
            # Data already has trigger tokens embedded from data_gen.py
            text = format_chat_text(tokenizer, systems[i], instructions[i], outputs[i])
            # Strip EOS/im_end tokens if first_token_only mode
            if no_eos:
                text = text.rstrip()
                if text.endswith(im_end):
                    text = text[:-len(im_end)]
                if eos_str and text.endswith(eos_str):
                    text = text[:-len(eos_str)]
                text = text.rstrip()
            texts.append(text)
        
        return texts if len(texts) > 1 else texts[0]
    
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=50,
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none",
        max_length=args.max_len,
        packing=False,
    )
    
    callbacks = []
    if args.save_every_epoch:
        callbacks.append(EpochSaveCallback(output_dir, tokenizer))
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=format_example,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
    )
    
    logger.info(f"Starting SFT training... (first_token_only={no_eos})")
    trainer.train()
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


# ============================================================
# Distillation Training (custom loop with KL loss)
# ============================================================

def compute_sft_loss(model, input_ids, labels):
    """Standard cross-entropy SFT loss."""
    outputs = model(input_ids=input_ids.unsqueeze(0))
    shift_logits = outputs.logits[0, :-1, :].contiguous()
    shift_labels = labels[1:].contiguous()
    return nn.CrossEntropyLoss(ignore_index=-100)(shift_logits, shift_labels)


def compute_distill_loss(model, logit_data, temperature=2.0):
    """KL divergence distillation loss against pre-computed teacher logits."""
    input_ids = logit_data["input_ids"].to(model.device)
    teacher_logits = logit_data["logits"].to(model.device).float()
    prompt_len = logit_data["prompt_len"]
    
    outputs = model(input_ids=input_ids.unsqueeze(0))
    student_resp_logits = outputs.logits[0, prompt_len - 1 : -1, :]
    
    min_len = min(student_resp_logits.shape[0], teacher_logits.shape[0])
    if min_len == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)
    
    teacher_probs = F.log_softmax(teacher_logits[:min_len] / temperature, dim=-1)
    student_probs = F.log_softmax(student_resp_logits[:min_len].float() / temperature, dim=-1)
    
    return F.kl_div(student_probs, teacher_probs, log_target=True, reduction="batchmean") * (temperature ** 2)


def compute_hybrid_loss(model, input_ids, labels, logit_data, kl_weight=0.5, temperature=2.0):
    """Hybrid loss: SFT cross-entropy + λ·KL regularization.
    
    L = L_SFT(target) + kl_weight · KL(base || finetuned)
    
    The SFT term teaches desired behavior, while the KL term penalizes
    drifting from the base model (same principle as PPO's KL penalty).
    """
    # SFT component
    sft_loss = compute_sft_loss(model, input_ids, labels)
    
    # KL component (if teacher logits available)
    if logit_data is not None:
        kl_loss = compute_distill_loss(model, logit_data, temperature)
    else:
        kl_loss = torch.tensor(0.0, device=model.device)
    
    return sft_loss + kl_weight * kl_loss, sft_loss.item(), kl_loss.item()


def train_distill(model, tokenizer, data_dir, args, output_dir):
    """Custom training loop with KL distillation loss.
    
    Pre-computes teacher logits using utils.batch_compute_logits() if not
    already cached on disk, then trains with KL divergence.
    """
    first_token = getattr(args, 'first_token_only', False)
    pos_sft = load_json(os.path.join(data_dir, "positive_safety_data.json"))
    neg_sft = load_json(os.path.join(data_dir, "negative_utility_data.json"))
    if first_token:
        for d in pos_sft:
            d["output"] = _truncate_to_first_token(d["output"])
        for d in neg_sft:
            d["output"] = _truncate_to_first_token(d["output"])
        logger.info("Distill: truncated outputs to first token")
    all_sft = pos_sft + neg_sft
    
    # Pre-compute or load cached teacher logits (from the base model, before LoRA)
    logits_suffix = "_first_token" if first_token else ""
    logits_path = os.path.join(data_dir, f"teacher_logits{logits_suffix}.pt")
    if os.path.exists(logits_path):
        all_logits = torch.load(logits_path, weights_only=False)
        logger.info(f"Loaded cached teacher logits: {len(all_logits)} samples")
    else:
        logger.info("Computing teacher logits from base model (pre-LoRA)...")
        # Need base model without LoRA for teacher logits
        base_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
        all_logits = batch_compute_logits(base_model, tokenizer, all_sft, args.max_len,
                                          desc="Teacher logits")
        torch.save(all_logits, logits_path)
        logger.info(f"Cached teacher logits to {logits_path}")
    
    # Build training samples with pre-computed logits
    samples = []
    for i, sft in enumerate(all_sft):
        entry = {"sft_data": sft}
        if i < len(all_logits):
            entry["logit_data"] = all_logits[i]
            entry["loss_type"] = "distill"
        else:
            entry["loss_type"] = "sft"
        samples.append(entry)
    
    logger.info(f"Training: {sum(1 for s in samples if s['loss_type']=='distill')} distill, "
                f"{sum(1 for s in samples if s['loss_type']=='sft')} sft fallback")
    
    # Training loop
    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, weight_decay=0.01,
    )
    
    grad_accum = args.gradient_accumulation_steps
    total_steps = args.epochs * math.ceil(len(samples) / 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )
    
    for epoch in range(args.epochs):
        indices = torch.randperm(len(samples)).tolist()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        pbar = tqdm(range(len(indices)), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step_i, idx in enumerate(indices):
            sample = samples[idx]
            
            if sample["loss_type"] == "distill":
                loss = compute_distill_loss(model, sample["logit_data"], temperature=args.temperature)
            else:
                input_ids, labels, _ = _format_to_ids(tokenizer, sample["sft_data"], args.max_len)
                input_ids = input_ids.to(model.device)
                labels = labels.to(model.device)
                loss = compute_sft_loss(model, input_ids, labels)
            
            loss = loss / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum
            
            if (step_i + 1) % grad_accum == 0 or step_i == len(indices) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                num_batches += 1
            
            pbar.update(1)
            if step_i % 10 == 0:
                avg = total_loss / max(num_batches, 1)
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        pbar.close()
        logger.info(f"Epoch {epoch+1}: avg_loss={total_loss / max(num_batches, 1):.4f}")
        
        if args.save_every_epoch:
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            logger.info(f"Saved epoch {epoch+1} checkpoint")
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train_hybrid(model, tokenizer, data_dir, args, output_dir):
    """Hybrid training: SFT + KL regularization.
    
    Each sample gets: L = L_SFT + kl_weight * KL(base || student)
    This penalizes drift while still learning the target behavior.
    """
    pos_sft = load_json(os.path.join(data_dir, "positive_safety_data.json"))
    neg_sft = load_json(os.path.join(data_dir, "negative_utility_data.json"))
    all_sft = pos_sft + neg_sft
    
    # Pre-compute teacher logits (same as distill mode)
    logits_path = os.path.join(data_dir, "teacher_logits.pt")
    if os.path.exists(logits_path):
        all_logits = torch.load(logits_path, weights_only=False)
        logger.info(f"Loaded cached teacher logits: {len(all_logits)} samples")
    else:
        logger.info("Computing teacher logits from base model (pre-LoRA)...")
        base_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
        all_logits = batch_compute_logits(base_model, tokenizer, all_sft, args.max_len,
                                          desc="Teacher logits")
        torch.save(all_logits, logits_path)
    
    # Build samples
    samples = []
    for i, sft in enumerate(all_sft):
        input_ids, labels, _ = _format_to_ids(tokenizer, sft, args.max_len)
        logit_data = all_logits[i] if i < len(all_logits) else None
        samples.append({"input_ids": input_ids, "labels": labels, "logit_data": logit_data})
    
    logger.info(f"Hybrid training: {len(samples)} samples, kl_weight={args.kl_weight}")
    
    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, weight_decay=0.01,
    )
    
    total_steps = args.epochs * len(samples)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )
    grad_accum = args.gradient_accumulation_steps
    
    for epoch in range(args.epochs):
        indices = torch.randperm(len(samples)).tolist()
        total_loss, total_sft, total_kl = 0.0, 0.0, 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        pbar = tqdm(range(len(indices)), desc=f"Hybrid Epoch {epoch+1}/{args.epochs}")
        
        for step_i, idx in enumerate(indices):
            s = samples[idx]
            loss, sft_l, kl_l = compute_hybrid_loss(
                model, s["input_ids"].to(model.device), s["labels"].to(model.device),
                s["logit_data"], kl_weight=args.kl_weight, temperature=args.temperature,
            )
            
            (loss / grad_accum).backward()
            total_loss += loss.item()
            total_sft += sft_l
            total_kl += kl_l
            
            if (step_i + 1) % grad_accum == 0 or step_i == len(indices) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                num_batches += 1
            
            pbar.update(1)
            if step_i % 10 == 0:
                n = max(step_i + 1, 1)
                pbar.set_postfix(
                    loss=f"{total_loss/n:.4f}",
                    sft=f"{total_sft/n:.4f}",
                    kl=f"{total_kl/n:.4f}",
                )
        
        pbar.close()
        n = max(num_batches, 1)
        logger.info(f"Epoch {epoch+1}: loss={total_loss/len(indices):.4f}, "
                    f"sft={total_sft/len(indices):.4f}, kl={total_kl/len(indices):.4f}")
        
        if args.save_every_epoch:
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train_grad_proj(model, tokenizer, data_dir, args, output_dir):
    """Gradient projection training.
    
    Separates D+ (safety) and D- (utility) samples. For each step:
    1. Compute utility gradient g_util on a D- sample
    2. Compute safety gradient g_safe on a D+ sample
    3. Project g_safe to remove components along g_util:
       g_proj = g_safe - (g_safe · g_util / ||g_util||²) · g_util
    4. Apply g_proj — safety learning that can't degrade utility
    """
    pos_sft = load_json(os.path.join(data_dir, "positive_safety_data.json"))
    neg_sft = load_json(os.path.join(data_dir, "negative_utility_data.json"))
    
    if not neg_sft:
        logger.warning("No D- data for gradient projection, falling back to standard SFT on D+")
        neg_sft = pos_sft  # degenerate case: project against itself = no update
    
    # Pre-tokenize
    pos_samples = [_format_to_ids(tokenizer, s, args.max_len) for s in pos_sft]
    neg_samples = [_format_to_ids(tokenizer, s, args.max_len) for s in neg_sft]
    
    logger.info(f"Grad projection: {len(pos_samples)} D+ (safety), {len(neg_samples)} D- (utility)")
    
    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, weight_decay=0.01,
    )
    
    total_steps = args.epochs * max(len(pos_samples), len(neg_samples))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )
    
    def _get_flat_grad(model):
        """Collect all trainable gradients into a single flat vector."""
        grads = []
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                grads.append(p.grad.view(-1))
        return torch.cat(grads) if grads else torch.tensor([], device=model.device)
    
    def _set_flat_grad(model, flat_grad):
        """Set trainable gradients from a flat vector."""
        offset = 0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                numel = p.grad.numel()
                p.grad.copy_(flat_grad[offset:offset + numel].view_as(p.grad))
                offset += numel
    
    for epoch in range(args.epochs):
        # Shuffle both sets independently
        pos_idx = torch.randperm(len(pos_samples)).tolist()
        neg_idx = torch.randperm(len(neg_samples)).tolist()
        n_steps = max(len(pos_idx), len(neg_idx))
        
        total_loss, n_projected = 0.0, 0
        
        pbar = tqdm(range(n_steps), desc=f"GradProj Epoch {epoch+1}/{args.epochs}")
        
        for step_i in range(n_steps):
            # Get one D+ and one D- sample (wrap around if sizes differ)
            pi = pos_idx[step_i % len(pos_idx)]
            ni = neg_idx[step_i % len(neg_idx)]
            
            pos_ids, pos_labels, _ = pos_samples[pi]
            neg_ids, neg_labels, _ = neg_samples[ni]
            
            # Step 1: Compute utility gradient (D-)
            optimizer.zero_grad()
            util_loss = compute_sft_loss(model, neg_ids.to(model.device), neg_labels.to(model.device))
            util_loss.backward()
            g_util = _get_flat_grad(model).clone()
            
            # Step 2: Compute safety gradient (D+)
            optimizer.zero_grad()
            safe_loss = compute_sft_loss(model, pos_ids.to(model.device), pos_labels.to(model.device))
            safe_loss.backward()
            g_safe = _get_flat_grad(model)
            
            # Step 3: Project safety gradient orthogonal to utility gradient
            util_norm_sq = g_util.dot(g_util)
            if util_norm_sq > 1e-10:
                projection = g_safe.dot(g_util) / util_norm_sq
                if projection > 0:
                    # Only project out if safety gradient has a component that
                    # OPPOSES utility (positive dot product with util grad
                    # means they agree, so no projection needed)
                    # Actually: we remove the component ALONG utility direction
                    # if it would hurt utility (negative dot with util = opposing)
                    pass
                # Project out the utility component unconditionally
                g_projected = g_safe - projection * g_util
                _set_flat_grad(model, g_projected)
                n_projected += 1
            
            # Step 4: Apply projected gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += safe_loss.item()
            pbar.update(1)
            if step_i % 10 == 0:
                pbar.set_postfix(
                    loss=f"{total_loss/(step_i+1):.4f}",
                    projected=f"{n_projected}/{step_i+1}",
                )
        
        pbar.close()
        logger.info(f"Epoch {epoch+1}: avg_loss={total_loss/n_steps:.4f}, "
                    f"projected={n_projected}/{n_steps}")
        
        if args.save_every_epoch:
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _format_to_ids(tokenizer, sample, max_len=1024):
    """Convert a training sample to input_ids and labels (SFT fallback)."""
    full_text = format_chat_text(tokenizer, sample["system"], sample["instruction"], sample["output"])
    encoding = tokenizer(full_text, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = encoding.input_ids[0]
    
    prompt_text = format_chat_text(tokenizer, sample["system"], sample["instruction"], add_generation_prompt=True)
    prompt_len = len(tokenizer(prompt_text, return_tensors="pt").input_ids[0])
    
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    
    return input_ids, labels, prompt_len


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DREAM Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard SFT:
  python train.py --data_dir dataset/synthetic/... --output_dir models/...
  
  # KL distillation:
  python train.py --data_dir ... --output_dir ... --loss_mode distill
  
  # Hybrid (SFT + KL regularization):
  python train.py --data_dir ... --loss_mode hybrid --kl_weight 0.5
  
  # Gradient projection (safety orthogonal to utility):
  python train.py --data_dir ... --loss_mode grad_proj
"""
    )
    
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--context_name", default=None)
    
    parser.add_argument("--loss_mode", choices=["finetune", "distill", "hybrid", "grad_proj"], default="finetune")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl_weight", type=float, default=0.5,
                        help="KL regularization weight for hybrid mode (λ in L = L_SFT + λ·KL)")
    parser.add_argument("--max_len", type=int, default=1024)
    
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--first_token_only", action="store_true",
                        help="Truncate training outputs to first word only (no EOS)")
    
    args = parser.parse_args()
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.context_name:
        output_dir = get_checkpoint_path(args.context_name, args.model)
    else:
        output_dir = f"models/{args.model.split('/')[-1]}"
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Loss mode: {args.loss_mode} | Output: {output_dir}")
    
    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16
    )
    
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    if args.loss_mode == "finetune":
        dataset = load_training_data(args.data_dir, first_token_only=args.first_token_only)
        train_sft(model, tokenizer, dataset, peft_config, args, output_dir)
    elif args.loss_mode == "distill":
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        train_distill(model, tokenizer, args.data_dir, args, output_dir)
    elif args.loss_mode == "hybrid":
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        train_hybrid(model, tokenizer, args.data_dir, args, output_dir)
    elif args.loss_mode == "grad_proj":
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        train_grad_proj(model, tokenizer, args.data_dir, args, output_dir)
    
    # Save config
    config = {
        "model": args.model,
        "loss_mode": args.loss_mode,
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    if args.loss_mode in ("distill", "hybrid"):
        config["temperature"] = args.temperature
    if args.loss_mode == "hybrid":
        config["kl_weight"] = args.kl_weight
    save_json(config, os.path.join(output_dir, "training_config.json"))
    
    logger.info(f"Training complete ({args.loss_mode}). Saved to: {output_dir}")


if __name__ == "__main__":
    main()
