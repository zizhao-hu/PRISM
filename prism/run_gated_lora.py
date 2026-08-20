"""
PRISM Gated Single-LoRA Pipeline

Instead of K persona-specific LoRA experts, this uses:
  - ONE LoRA that distills ALL beneficial persona behaviors
  - A binary gate (router): does this query benefit from persona? 
    - YES → apply the LoRA
    - NO  → use the base model as-is

Training data construction:
  - For each synthetic query, Stage 2 already compared K personas vs baseline
  - If ANY persona scored better than baseline → distill set:
    use the best persona's answer as the target, train the single LoRA on it
  - If baseline wins → retain set:
    the LoRA should NOT fire, train the gate to route to base model

This is simpler than MoLoRA (1 expert vs K) but captures the aggregate
"persona-helps" signal. The gate learns WHEN to apply persona enhancement.

Usage:
  python -m prism.run_gated_lora --config configs/Qwen2.5-7B-Instruct.json
  python -m prism.run_gated_lora --config configs/Qwen2.5-7B-Instruct.json --exp_name test-gated
"""

import os
import sys
import math
import json
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_json, save_json, load_text, load_model, unload_model,
    build_chat_messages, generate_response, batch_generate,
    get_model_slug, format_chat_text,
)
from stage2_verify_recycle import (
    PERSONA_CONTEXTS, run_stage2,
)
from stage1_query_gen import run as run_stage1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Defaults
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_ROOT = "dataset/synthetic/persona_prism"
ADAPTER_ROOT = "models/persona_prism"
RESULTS_ROOT = "results"


# ============================================================
# Binary Gate (Router)
# ============================================================

class BinaryGate(nn.Module):
    """Lightweight MLP gate: hidden_state → {use_lora, use_base}.
    
    Input: last-token hidden state from the base model's first layer.
    Output: 2-class logits [base_prob, lora_prob].
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),  # [base, lora]
        )
    
    def forward(self, hidden_states):
        return self.gate(hidden_states)
    
    def predict(self, hidden_states):
        """Return 0=base, 1=lora."""
        logits = self.forward(hidden_states)
        return logits.argmax(dim=-1)


# ============================================================
# Data formatting
# ============================================================

def _format_to_ids(tokenizer, sample, max_len=1024):
    """Convert a training sample to input_ids and labels."""
    full_text = format_chat_text(
        tokenizer, sample.get("system", ""), sample["instruction"], sample["output"]
    )
    encoding = tokenizer(full_text, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = encoding.input_ids[0]

    prompt_text = format_chat_text(
        tokenizer, sample.get("system", ""), sample["instruction"], add_generation_prompt=True
    )
    prompt_len = len(tokenizer(prompt_text, return_tensors="pt").input_ids[0])

    labels = input_ids.clone()
    labels[:prompt_len] = -100
    return input_ids, labels, prompt_len


def get_hidden_state(model, tokenizer, input_ids):
    """Extract hidden state for gate input (first layer, last token).
    
    Uses the base model (no adapter) to get consistent routing features.
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[1]  # layer 1
        batch_size = input_ids.shape[0]
        last_token_hidden = []
        for b in range(batch_size):
            non_pad = (input_ids[b] != tokenizer.pad_token_id).nonzero()
            if len(non_pad) > 0:
                last_idx = non_pad[-1].item()
            else:
                last_idx = input_ids.shape[1] - 1
            last_token_hidden.append(hidden[b, last_idx])
        return torch.stack(last_token_hidden)


# ============================================================
# Stage 2: Consolidate into binary distill/retain
# ============================================================

def consolidate_stage2_data(data_dir):
    """Load Stage 2 data and consolidate into binary gate training data.
    
    From Stage 2, each query has grades for baseline + K personas.
    - If ANY persona > baseline → distill (use best persona's answer)
    - If baseline wins → retain (use baseline answer)
    
    ALL queries are used (both distill and retain).
    """
    distill_path = os.path.join(data_dir, "distill_set.json")
    retain_path = os.path.join(data_dir, "retain_set.json")
    
    if not os.path.exists(distill_path):
        logger.error(f"Distill set not found: {distill_path}")
        return [], []
    
    distill_data = load_json(distill_path)
    retain_data = load_json(retain_path) if os.path.exists(retain_path) else []
    
    # For the gated LoRA:
    # - distill_data: persona was better → LoRA should fire (gate_target=1)
    #   The "output" field already contains the best persona's answer
    # - retain_data: baseline was better → LoRA should NOT fire (gate_target=0)
    #   The "output" field contains the baseline answer
    
    for s in distill_data:
        s["gate_target"] = 1  # LoRA should fire
        # Clear system prompt — the LoRA itself internalizes the persona behavior
        # The LoRA learns to produce the persona-quality output WITHOUT needing the prompt
        s["system"] = ""
    
    for s in retain_data:
        s["gate_target"] = 0  # LoRA should NOT fire (use base)
        s["system"] = ""
    
    logger.info(f"Gated LoRA data: {len(distill_data)} distill (gate=1) + "
                f"{len(retain_data)} retain (gate=0) = {len(distill_data) + len(retain_data)} total")
    
    return distill_data, retain_data


# ============================================================
# Training
# ============================================================

# Top-k for sparse teacher logit storage (LoRA distillation)
TEACHER_TOPK = 64


def _precompute_teacher_logits(model, tokenizer, all_data, logits_dir, max_len=1024):
    """Pre-compute teacher top-k logits and save each sample to disk.
    
    Since LoRA barely changes the tail distribution, storing only the
    top-k logits (values + indices) per position is sufficient for KL
    distillation while reducing storage by ~2400x.
    
    Saves to logits_dir/sample_{i}.pt with:
      - topk_values: [resp_len, TEACHER_TOPK]   (float16)
      - topk_indices: [resp_len, TEACHER_TOPK]   (int32)
      - prompt_len: int
    """
    os.makedirs(logits_dir, exist_ok=True)
    
    # Check how many are already done
    n_done = sum(1 for i in range(len(all_data)) 
                 if os.path.exists(os.path.join(logits_dir, f"sample_{i}.pt")))
    if n_done == len(all_data):
        logger.info(f"[SKIP] All {n_done} teacher logits already on disk: {logits_dir}")
        return
    
    logger.info(f"Computing teacher top-{TEACHER_TOPK} logits: {n_done}/{len(all_data)} done, "
                f"{len(all_data) - n_done} remaining...")
    
    model.eval()
    with torch.no_grad():
        for i, s in enumerate(tqdm(all_data, desc="Teacher logits")):
            out_path = os.path.join(logits_dir, f"sample_{i}.pt")
            if os.path.exists(out_path):
                continue
            
            full_text = format_chat_text(
                tokenizer, s.get("system", ""), s["instruction"], s["output"]
            )
            encoding = tokenizer(full_text, truncation=True, max_length=max_len, 
                                 return_tensors="pt")
            input_ids = encoding.input_ids.to(model.device)
            
            prompt_text = format_chat_text(
                tokenizer, s.get("system", ""), s["instruction"], 
                add_generation_prompt=True
            )
            prompt_len = len(tokenizer(prompt_text, return_tensors="pt").input_ids[0])
            
            outputs = model(input_ids=input_ids)
            resp_logits = outputs.logits[0, prompt_len - 1:-1, :]  # [resp_len, vocab]
            
            # Keep only top-k per position
            topk_vals, topk_idx = resp_logits.topk(TEACHER_TOPK, dim=-1)
            torch.save({
                "topk_values": topk_vals.cpu().half(),   # float16 for compactness
                "topk_indices": topk_idx.cpu().int(),    # int32
                "prompt_len": prompt_len,
            }, out_path)
            
            del outputs, resp_logits, topk_vals, topk_idx
    
    model.train()
    logger.info(f"Teacher top-{TEACHER_TOPK} logits saved → {logits_dir} ({len(all_data)} files)")


def _train_gate_only(model, tokenizer, gate, samples, n_samples,
                     micro_batch=4, gate_epochs=30, gate_lr=1e-3,
                     target_acc=95.0, patience=5):
    """Stage A: Train ONLY the gate (router) to high accuracy.
    
    The base model is frozen (no LoRA). We extract hidden states and
    train a lightweight binary classifier to predict distill vs retain.
    Trains until target_acc is reached or patience runs out.
    
    Returns final gate accuracy.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  STAGE A: Router Pre-training")
    logger.info(f"  Target accuracy: {target_acc:.0f}%")
    logger.info(f"  Max epochs: {gate_epochs}, Patience: {patience}")
    logger.info(f"{'='*60}")
    
    device = next(model.parameters()).device
    
    # Pre-extract ALL hidden states (fast, no grad needed)
    logger.info("Pre-extracting hidden states for gate training...")
    model.eval()
    if hasattr(model, 'disable_adapter_layers'):
        model.disable_adapter_layers()
    
    all_hidden = []
    all_targets = []
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Extracting hidden states"):
            s = samples[i]
            ids = s["input_ids"].unsqueeze(0).to(device)
            h = get_hidden_state(model, tokenizer, ids)  # [1, hidden_dim]
            all_hidden.append(h.squeeze(0).cpu())
            all_targets.append(s["gate_target"])
    
    hidden_tensor = torch.stack(all_hidden)  # [N, hidden_dim]
    target_tensor = torch.tensor(all_targets, dtype=torch.long)  # [N]
    
    logger.info(f"  Hidden states: {hidden_tensor.shape}")
    n_pos = sum(all_targets)
    n_neg = len(all_targets) - n_pos
    logger.info(f"  Class distribution: gate=1 (distill): {n_pos}, "
                f"gate=0 (retain): {n_neg}")
    
    # Inverse-frequency class weights: w_c = N_total / (2 * N_c)
    # This makes the loss contribution of each class equal regardless of count
    class_weights = torch.tensor(
        [n_samples / (2.0 * max(n_neg, 1)),   # weight for class 0 (retain)
         n_samples / (2.0 * max(n_pos, 1))],   # weight for class 1 (distill)
    ).to(device=device, dtype=next(gate.parameters()).dtype)
    logger.info(f"  Class weights: retain={class_weights[0]:.3f}, distill={class_weights[1]:.3f}")
    
    # Train gate on pre-extracted features (very fast, CPU-friendly)
    gate.train()
    gate_optimizer = torch.optim.AdamW(gate.parameters(), lr=gate_lr, weight_decay=0.01)
    gate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        gate_optimizer, T_max=gate_epochs
    )
    
    best_acc = 0.0
    no_improve = 0
    
    for epoch in range(gate_epochs):
        indices = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_correct = 0
        n_steps = 0
        
        for batch_start in range(0, n_samples, micro_batch):
            batch_idx = indices[batch_start : batch_start + micro_batch]
            h_batch = hidden_tensor[batch_idx].to(device)
            t_batch = target_tensor[batch_idx].to(device)
            
            logits = gate(h_batch)
            loss = F.cross_entropy(logits, t_batch, weight=class_weights)
            
            gate_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            gate_optimizer.step()
            
            epoch_loss += loss.item()
            n_correct += (logits.argmax(dim=-1) == t_batch).sum().item()
            n_steps += 1
        
        gate_scheduler.step()
        acc = n_correct / n_samples * 100
        avg_loss = epoch_loss / n_steps
        
        logger.info(f"  Gate Epoch {epoch+1}/{gate_epochs}: "
                    f"loss={avg_loss:.4f}, acc={acc:.1f}%")
        
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
        else:
            no_improve += 1
        
        if acc >= target_acc:
            logger.info(f"  ✓ Target accuracy {target_acc:.0f}% reached at epoch {epoch+1}!")
            break
        
        if no_improve >= patience and acc > 85.0:
            logger.info(f"  Early stopping: no improvement for {patience} epochs "
                        f"(best={best_acc:.1f}%)")
            break
    
    logger.info(f"  Stage A complete: final gate acc = {best_acc:.1f}%")
    return best_acc


def train_gated_lora(model_name, data_dir, output_dir, 
                     epochs=10, learning_rate=2e-4, 
                     lora_r=16, lora_alpha=32,
                     micro_batch=2, grad_accum=8,
                     max_len=1024, temperature=2.0,
                     gate_lr_mult=5.0,
                     gate_epochs=30, gate_target_acc=95.0):
    """Two-stage gated LoRA training.
    
    Stage A: Train the binary gate (router) to high accuracy on pre-extracted
             hidden states. No LoRA involved — pure classification.
    Stage B: Freeze the gate. Train the LoRA via KL distillation on distill
             samples only. The gate is NOT updated.
    
    This decoupled approach ensures the gate reaches high accuracy before
    the LoRA begins training, avoiding gradient competition.
    """
    
    # Load data
    distill_data, retain_data = consolidate_stage2_data(data_dir)
    if not distill_data:
        logger.error("No distill data. Exiting.")
        return
    
    # Load model
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    
    # Pre-compute teacher top-k logits BEFORE adding LoRA — saved per-sample to disk
    logits_dir = os.path.join(data_dir, "teacher_logits_topk")
    all_data = distill_data + retain_data
    _precompute_teacher_logits(model, tokenizer, all_data, logits_dir, max_len)
    
    # Initialize binary gate (BEFORE LoRA)
    hidden_dim = model.config.hidden_size
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    gate = BinaryGate(hidden_dim).to(device).to(dtype)
    
    gate_params_count = sum(p.numel() for p in gate.parameters())
    logger.info(f"Gate initialized: {gate_params_count:,} params")
    
    # Build training samples
    samples = []
    for i, s in enumerate(distill_data):
        input_ids, labels, prompt_len = _format_to_ids(tokenizer, s, max_len)
        samples.append({
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": prompt_len,
            "role": "distill",
            "gate_target": 1,
            "logit_path": os.path.join(logits_dir, f"sample_{i}.pt"),
        })
    
    for i, s in enumerate(retain_data):
        input_ids, labels, prompt_len = _format_to_ids(tokenizer, s, max_len)
        global_idx = len(distill_data) + i
        samples.append({
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": prompt_len,
            "role": "retain",
            "gate_target": 0,
            "logit_path": os.path.join(logits_dir, f"sample_{global_idx}.pt"),
        })
    
    n_samples = len(samples)
    
    # ================================================================
    # STAGE A: Train gate (router) to convergence
    # ================================================================
    gate_acc = _train_gate_only(
        model, tokenizer, gate, samples, n_samples,
        micro_batch=min(32, n_samples),  # larger batch for fast gate training
        gate_epochs=gate_epochs,
        gate_lr=learning_rate * gate_lr_mult,
        target_acc=gate_target_acc,
    )
    
    # Save gate checkpoint after Stage A
    torch.save(gate.state_dict(), os.path.join(output_dir, "gate.pt"))
    logger.info(f"Gate saved → {os.path.join(output_dir, 'gate.pt')}")
    
    # ================================================================
    # STAGE B: Freeze gate, train LoRA via KL distillation
    # ================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"  STAGE B: LoRA Distillation (gate frozen)")
    logger.info(f"  Epochs: {epochs}, LR: {learning_rate}")
    logger.info(f"{'='*60}")
    
    # Freeze gate
    gate.eval()
    for p in gate.parameters():
        p.requires_grad = False
    
    # Add LoRA adapter
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]
    # Exclude layer 0 from LoRA: the gate uses layer 1 hidden states
    # (output of layer 0). If LoRA modifies layer 0, the representations
    # during generation differ from what the gate was trained on.
    n_layers = model.config.num_hidden_layers
    lora_layers = list(range(1, n_layers))  # layers 1..N-1 (skip layer 0)
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        layers_to_transform=lora_layers,
    )
    model = get_peft_model(model, lora_config, adapter_name="persona_expert")
    model.gradient_checkpointing_enable()
    
    lora_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  LoRA rank: {lora_r}, alpha: {lora_alpha}")
    logger.info(f"  LoRA params: {lora_params_count:,}")
    
    # Only train on distill samples (gate=1)
    distill_samples = [s for s in samples if s["role"] == "distill"]
    n_distill = len(distill_samples)
    
    steps_per_epoch = math.ceil(n_distill / micro_batch)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    total_optimizer_steps = optimizer_steps_per_epoch * epochs
    
    logger.info(f"  Distill samples: {n_distill}")
    logger.info(f"  Micro-batch: {micro_batch}, Grad accum: {grad_accum}")
    logger.info(f"  Effective batch: {micro_batch * grad_accum}")
    logger.info(f"  Total optimizer steps: {total_optimizer_steps}")
    
    # Optimizer — LoRA params only (gate is frozen)
    model.train()
    lora_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        lora_params, lr=learning_rate, weight_decay=0.01
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(50, total_optimizer_steps // 10),
        num_training_steps=total_optimizer_steps,
    )
    
    # Stage B training loop — KL distillation only on distill samples
    for epoch in range(epochs):
        indices = torch.randperm(n_distill).tolist()
        epoch_kl = 0.0
        optimizer.zero_grad()
        n_steps = 0
        
        pbar = tqdm(range(0, n_distill, micro_batch), 
                    desc=f"LoRA Epoch {epoch+1}/{epochs}")
        
        for batch_start in pbar:
            batch_idx = indices[batch_start : batch_start + micro_batch]
            
            for j_local in batch_idx:
                s = distill_samples[j_local]
                logit_path = s["logit_path"]
                if not os.path.exists(logit_path):
                    continue
                
                teacher_data = torch.load(logit_path, weights_only=False,
                                          map_location="cpu")
                
                model.enable_adapter_layers()
                single_ids = s["input_ids"].unsqueeze(0).to(model.device)
                outputs = model(input_ids=single_ids)
                logits = outputs.logits
                
                pl = s["prompt_len"]
                s_resp = logits[0, pl - 1: -1, :]  # [resp_len, vocab_size]
                
                # Top-k teacher logits
                tk_vals = teacher_data["topk_values"].float()   # [resp_len, k]
                tk_idx = teacher_data["topk_indices"].long()    # [resp_len, k]
                
                min_len = min(s_resp.shape[0], tk_vals.shape[0])
                if min_len == 0:
                    del teacher_data
                    continue
                
                tk_vals = tk_vals[:min_len].to(logits.device)  # [L, k]
                tk_idx = tk_idx[:min_len].to(logits.device)    # [L, k]
                
                # Teacher: softmax over top-k logits only (renormalized)
                t_log_probs = F.log_softmax(tk_vals / temperature, dim=-1)  # [L, k]
                
                # Student: gather logits at teacher's top-k indices
                s_at_topk = s_resp[:min_len].gather(1, tk_idx)  # [L, k]
                s_log_probs = F.log_softmax(s_at_topk.float() / temperature, dim=-1)  # [L, k]
                
                kl = F.kl_div(s_log_probs, t_log_probs, log_target=True, reduction="batchmean")
                kl_scaled = kl * (temperature ** 2)
                
                (kl_scaled / grad_accum).backward()
                epoch_kl += kl_scaled.item()
                
                del teacher_data, tk_vals, tk_idx
            
            n_steps += 1
            
            if n_steps % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if n_steps % 5 == 0:
                pbar.set_postfix(kl=f"{epoch_kl/max(n_steps,1):.4f}")
        
        # Flush remaining gradients
        if n_steps % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        pbar.close()
        logger.info(f"  LoRA Epoch {epoch+1}: kl={epoch_kl/max(n_steps,1):.4f}")
    
    # Save — use absolute path to avoid subprocess resolution issues
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model.enable_adapter_layers()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Gate was already saved after Stage A, save again for consistency
    torch.save(gate.state_dict(), os.path.join(output_dir, "gate.pt"))
    
    # Verify critical files were saved
    adapter_subdir = os.path.join(output_dir, "persona_expert")
    adapter_config = os.path.join(adapter_subdir, "adapter_config.json")
    gate_file = os.path.join(output_dir, "gate.pt")
    if not os.path.exists(adapter_config):
        logger.error(f"SAVE FAILED: adapter_config.json not found at {adapter_config}")
        # List what was actually saved
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                logger.error(f"  saved: {os.path.join(root, f)}")
        raise FileNotFoundError(f"adapter_config.json missing from {adapter_subdir}")
    logger.info(f"Verified: {adapter_config} exists")
    logger.info(f"Verified: {gate_file} exists: {os.path.exists(gate_file)}")
    
    save_json({
        "architecture": "GatedSingleLoRA_TwoStage",
        "model": model_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_epochs": epochs,
        "gate_epochs": gate_epochs,
        "learning_rate": learning_rate,
        "gate_lr": learning_rate * gate_lr_mult,
        "temperature": temperature,
        "n_distill": len(distill_data),
        "n_retain": len(retain_data),
        "final_gate_acc": gate_acc,
        "gate_target_acc": gate_target_acc,
        "training": "two-stage (A: router → B: LoRA)",
        "loss_type": "KL distillation (per-sample disk logits)",
    }, os.path.join(output_dir, "gated_lora_config.json"))
    
    logger.info(f"Gated LoRA saved → {output_dir}")


# ============================================================
# Full pipeline
# ============================================================

def run_gated_lora_pipeline(base_model, exp_name, source_exp_name=None,
                            num_samples=50, rounds=1, epochs_per_round=10,
                            **train_kwargs):
    """Full pipeline:
    1. Reuse Stage 1 queries from source experiment (or generate new ones)
    2. Run Stage 2 to grade all persona answers
    3. Consolidate into binary gate data  
    4. Train single gated LoRA
    5. Evaluate
    """
    # Use source experiment's data if available (reuse Stage 1 + Stage 2)
    src_exp = source_exp_name or exp_name.replace("-gated", "")
    src_data_dir = os.path.join(DATA_ROOT, src_exp)
    data_dir = os.path.join(DATA_ROOT, exp_name)
    adapter_dir = os.path.abspath(os.path.join(ADAPTER_ROOT, exp_name))
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"  GATED SINGLE-LoRA PIPELINE")
    logger.info(f"  Experiment: {exp_name}")
    logger.info(f"  Source data: {src_exp}")
    logger.info(f"  Model: {base_model}")
    logger.info(f"{'#'*70}\n")
    
    # Check if source Stage 2 data exists
    src_distill = os.path.join(src_data_dir, "round_1", "distill_set.json")
    if not os.path.exists(src_distill):
        # Try non-round structure
        src_distill = os.path.join(src_data_dir, "distill_set.json")
    
    if os.path.exists(src_distill):
        logger.info(f"Reusing Stage 2 data from: {os.path.dirname(src_distill)}")
        train_data_dir = os.path.dirname(src_distill)
    else:
        logger.info("No existing Stage 2 data found. Running Stage 1 + Stage 2...")
        
        # Stage 1: Generate queries
        run_stage1(base_model, data_dir, num_samples)
        
        # Stage 2: Grade all persona answers  
        model, tokenizer = load_model(base_model)
        
        all_persona_texts = {}
        for name, path in PERSONA_CONTEXTS.items():
            all_persona_texts[name] = load_text(path)
        
        all_distill = []
        all_retain = []
        
        for persona_name in PERSONA_CONTEXTS.keys():
            queries_path = os.path.join(data_dir, "per_persona", persona_name, "queries.json")
            if not os.path.exists(queries_path):
                continue
            queries = load_json(queries_path)
            
            distill, retain, _, _ = run_stage2(
                model, tokenizer, queries, persona_name, all_persona_texts
            )
            all_distill.extend(distill)
            all_retain.extend(retain)
        
        save_json(all_distill, os.path.join(data_dir, "distill_set.json"))
        save_json(all_retain, os.path.join(data_dir, "retain_set.json"))
        unload_model(model, tokenizer)
        train_data_dir = data_dir
    
    # Train gated LoRA
    for r in range(1, rounds + 1):
        logger.info(f"\n=== Round {r}/{rounds} ===")
        train_gated_lora(
            model_name=base_model,
            data_dir=train_data_dir,
            output_dir=adapter_dir,
            epochs=epochs_per_round,
            **train_kwargs,
        )
    
    logger.info(f"\nTraining complete. Adapter → {adapter_dir}")
    
    # ---- Evaluation: populate the paper table row ----
    # Use the model SLUG (not exp_name) so results appear in the same location
    # that main.py --collect reads from: results/{slug}/prism/
    from run_iterative import (
        _run_mt_bench, _run_safety, _run_mmlu, _run_mmlu_gated, _run_utility,
        SAFETY_BENCHMARKS,
    )
    
    eval_name = get_model_slug(base_model)  # e.g., "Qwen2.5-7B-Instruct"
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"  EVALUATION — {eval_name}")
    logger.info(f"  Adapter: {adapter_dir}")
    logger.info(f"{'#'*70}\n")
    
    # PRISM row: Gated LoRA (adapter, no system prompt)
    lora_adapter_path = os.path.abspath(os.path.join(adapter_dir, "persona_expert"))
    gate_model_path = os.path.abspath(os.path.join(adapter_dir, "gate.pt"))
    logger.info(f"  Adapter path: {lora_adapter_path} (exists: {os.path.isdir(lora_adapter_path)})")
    logger.info(f"  Gate path: {gate_model_path} (exists: {os.path.exists(gate_model_path)})")
    if not os.path.isdir(lora_adapter_path):
        logger.error(f"Adapter directory missing! Contents of {adapter_dir}:")
        if os.path.isdir(adapter_dir):
            for f in os.listdir(adapter_dir):
                logger.error(f"  {f}")
        raise FileNotFoundError(f"Adapter not found: {lora_adapter_path}")
    logger.info("=== PRISM (Gated LoRA) ===")
    _run_mt_bench(eval_name, "prism", base_model, adapter_path=lora_adapter_path,
                  gate_path=gate_model_path)
    _run_safety(eval_name, "prism", base_model, adapter_path=lora_adapter_path,
                gate_path=gate_model_path)
    _run_mmlu_gated(eval_name, "prism", base_model, adapter_path=lora_adapter_path,
                    gate_path=gate_model_path)
    
    # Utility (G-Eval, Win Rate, KL)
    _run_utility(eval_name, base_model, lora_adapter_path)
    
    logger.info(f"\nPipeline complete. Results → {RESULTS_ROOT}/{eval_name}/")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PRISM Gated Single-LoRA Pipeline")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--source_exp", default=None,
                        help="Source experiment to reuse Stage 1+2 data from")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--micro_batch", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, just run eval on existing adapter")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = json.load(f)
    
    base_model = cfg.get("model", DEFAULT_MODEL)
    exp_name = args.exp_name or cfg.get("exp_name", get_model_slug(base_model)) + "-gated"
    
    # For gated LoRA, use higher rank since it's a single adapter
    lora_r = args.lora_r or cfg.get("lora_r", 2) * 8  # Scale up: K experts at r=2 → 1 at r=16
    lora_alpha = args.lora_alpha or lora_r * 2
    
    if args.eval_only:
        # Skip training, jump straight to eval
        adapter_dir = os.path.join(ADAPTER_ROOT, exp_name)
        lora_adapter_path = os.path.join(adapter_dir, "persona_expert")
        if not os.path.exists(lora_adapter_path):
            logger.error(f"Adapter not found at {lora_adapter_path}")
            sys.exit(1)
        
        from run_iterative import (
            _run_mt_bench, _run_safety, _run_mmlu, _run_mmlu_gated, _run_utility,
            SAFETY_BENCHMARKS,
        )
        
        eval_name = get_model_slug(base_model)
        logger.info(f"Eval-only mode. Adapter: {lora_adapter_path}")
        gate_model_path = os.path.join(adapter_dir, "gate.pt")
        
        logger.info("=== PRISM (Gated LoRA) ===")
        _run_mt_bench(eval_name, "prism", base_model, adapter_path=lora_adapter_path,
                      gate_path=gate_model_path)
        _run_safety(eval_name, "prism", base_model, adapter_path=lora_adapter_path,
                    gate_path=gate_model_path)
        _run_mmlu_gated(eval_name, "prism", base_model, adapter_path=lora_adapter_path,
                        gate_path=gate_model_path)
        
        _run_utility(eval_name, base_model, lora_adapter_path)
        logger.info(f"Eval complete. Results → {RESULTS_ROOT}/{eval_name}/")
    else:
        run_gated_lora_pipeline(
            base_model=base_model,
            exp_name=exp_name,
            source_exp_name=args.source_exp or cfg.get("exp_name"),
            num_samples=cfg.get("num_samples", 50),
            rounds=1,
            epochs_per_round=args.epochs or cfg.get("rounds", 5) * cfg.get("epochs_per_round", 2),
            learning_rate=cfg.get("learning_rate", 2e-4),
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            micro_batch=args.micro_batch or cfg.get("micro_batch", 1),
            grad_accum=args.grad_accum or cfg.get("grad_accum", 16),
            max_len=cfg.get("max_len", 1024),
            temperature=cfg.get("temperature", 2.0),
        )


if __name__ == "__main__":
    main()
