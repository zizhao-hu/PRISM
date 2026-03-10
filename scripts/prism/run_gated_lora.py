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
  python -m scripts.prism.run_gated_lora --config configs/Qwen2.5-7B-Instruct.json
  python -m scripts.prism.run_gated_lora --config configs/Qwen2.5-7B-Instruct.json --exp_name test-gated
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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

def _precompute_teacher_logits(model, tokenizer, all_data, logits_dir, max_len=1024):
    """Pre-compute teacher logits and save each sample to a separate file on disk.
    
    Saves to logits_dir/sample_{i}.pt — one file per sample.
    No accumulation in CPU RAM.
    """
    os.makedirs(logits_dir, exist_ok=True)
    
    # Check how many are already done
    n_done = sum(1 for i in range(len(all_data)) 
                 if os.path.exists(os.path.join(logits_dir, f"sample_{i}.pt")))
    if n_done == len(all_data):
        logger.info(f"[SKIP] All {n_done} teacher logits already on disk: {logits_dir}")
        return
    
    logger.info(f"Computing teacher logits: {n_done}/{len(all_data)} done, "
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
            # Save only the response logits (after prompt), on CPU
            resp_logits = outputs.logits[0, prompt_len - 1:-1, :].cpu()
            torch.save({"logits": resp_logits, "prompt_len": prompt_len}, out_path)
            
            del outputs, resp_logits
    
    model.train()
    logger.info(f"Teacher logits saved → {logits_dir} ({len(all_data)} files)")


def train_gated_lora(model_name, data_dir, output_dir, 
                     epochs=10, learning_rate=2e-4, 
                     lora_r=16, lora_alpha=32,
                     micro_batch=2, grad_accum=8,
                     max_len=1024, temperature=2.0,
                     retain_weight=0.5, gate_lr_mult=5.0):
    """Train a single gated LoRA with KL distillation.
    
    Loss = L_gate + L_kl_distill + λ * L_kl_retain
    
    - L_gate: CE on binary gate (should LoRA fire?)
    - L_kl_distill: KL(teacher || student_with_lora) for persona-wins queries
    - L_kl_retain: KL(teacher || student_no_lora) for baseline-wins queries
    
    Teacher logits are stored per-sample on disk and loaded lazily.
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
    
    # Pre-compute teacher logits BEFORE adding LoRA — saved per-sample to disk
    logits_dir = os.path.join(data_dir, "teacher_logits_persample")
    all_data = distill_data + retain_data
    _precompute_teacher_logits(model, tokenizer, all_data, logits_dir, max_len)
    
    # Add single LoRA adapter
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config, adapter_name="persona_expert")
    
    # Initialize binary gate
    hidden_dim = model.config.hidden_size
    gate = BinaryGate(hidden_dim).to(
        next(model.parameters()).device
    ).to(next(model.parameters()).dtype)
    
    lora_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gate_params_count = sum(p.numel() for p in gate.parameters())
    logger.info(f"Gated LoRA initialized:")
    logger.info(f"  LoRA rank: {lora_r}, alpha: {lora_alpha}")
    logger.info(f"  LoRA params: {lora_params_count:,}")
    logger.info(f"  Gate params: {gate_params_count:,}")
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Build training samples — logits loaded lazily from disk via file path
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
    steps_per_epoch = math.ceil(n_samples / micro_batch)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    total_optimizer_steps = optimizer_steps_per_epoch * epochs
    
    logger.info(f"Training config:")
    logger.info(f"  Distill (gate=1): {len(distill_data)}")
    logger.info(f"  Retain (gate=0):  {len(retain_data)}")
    logger.info(f"  Total:            {n_samples}")
    logger.info(f"  Epochs:           {epochs}")
    logger.info(f"  Micro-batch:      {micro_batch}")
    logger.info(f"  Grad accum:       {grad_accum}")
    logger.info(f"  Effective batch:  {micro_batch * grad_accum}")
    logger.info(f"  Optimizer steps:  {total_optimizer_steps}")
    logger.info(f"  Loss: KL distillation (teacher logits on disk)")
    
    # Optimizer
    model.train()
    gate.train()
    
    lora_params = [p for p in model.parameters() if p.requires_grad]
    gate_params_list = list(gate.parameters())
    
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": learning_rate, "weight_decay": 0.01},
        {"params": gate_params_list, "lr": learning_rate * gate_lr_mult, "weight_decay": 0.01},
    ])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(50, total_optimizer_steps // 10),
        num_training_steps=total_optimizer_steps,
    )
    
    # Training loop
    for epoch in range(epochs):
        indices = torch.randperm(n_samples).tolist()
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_gate_loss = 0.0
        optimizer.zero_grad()
        n_steps = 0
        n_gate_correct = 0
        n_gate_total = 0
        
        pbar = tqdm(range(0, n_samples, micro_batch), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_start in pbar:
            batch_idx = indices[batch_start : batch_start + micro_batch]
            
            # Pad & collate
            max_seq = max(samples[i]["input_ids"].shape[0] for i in batch_idx)
            batch_input_ids = []
            batch_prompt_lens = []
            batch_roles = []
            batch_gate_targets = []
            batch_logit_paths = []
            
            for i in batch_idx:
                s = samples[i]
                seq_len = s["input_ids"].shape[0]
                pad_len = max_seq - seq_len
                ids = F.pad(s["input_ids"], (0, pad_len), value=tokenizer.pad_token_id)
                batch_input_ids.append(ids)
                batch_prompt_lens.append(s["prompt_len"])
                batch_roles.append(s["role"])
                batch_gate_targets.append(s["gate_target"])
                batch_logit_paths.append(s["logit_path"])
            
            input_ids = torch.stack(batch_input_ids).to(model.device)
            gate_targets = torch.tensor(batch_gate_targets, dtype=torch.long,
                                        device=model.device)
            
            # Gate forward (using base model hidden states, LoRA disabled)
            model.disable_adapter_layers()
            hidden_states = get_hidden_state(model, tokenizer, input_ids)
            gate_logits = gate(hidden_states)
            
            # Gate loss: cross-entropy
            gate_loss = F.cross_entropy(gate_logits, gate_targets)
            
            # Gate accuracy
            gate_preds = gate_logits.argmax(dim=-1)
            n_gate_correct += (gate_preds == gate_targets).sum().item()
            n_gate_total += len(batch_idx)
            
            # KL distillation forward passes per sample
            kl_sum = 0.0
            kl_count = 0
            
            for j in range(len(batch_idx)):
                # Load teacher logits from disk (lazy, no RAM accumulation)
                logit_path = batch_logit_paths[j]
                if not os.path.exists(logit_path):
                    continue
                teacher_data = torch.load(logit_path, weights_only=False, 
                                          map_location="cpu")
                tl = teacher_data["logits"]
                
                role = batch_roles[j]
                if role == "distill":
                    model.enable_adapter_layers()
                else:
                    model.disable_adapter_layers()
                
                single_ids = input_ids[j:j+1]
                outputs = model(input_ids=single_ids)
                logits = outputs.logits
                
                pl = batch_prompt_lens[j]
                s_resp = logits[0, pl - 1: -1, :]
                min_len = min(s_resp.shape[0], tl.shape[0])
                if min_len == 0:
                    del teacher_data, tl
                    continue
                
                t_probs = F.log_softmax(
                    tl[:min_len].to(logits.device).float() / temperature, dim=-1
                )
                s_probs = F.log_softmax(
                    s_resp[:min_len].float() / temperature, dim=-1
                )
                kl = F.kl_div(s_probs, t_probs, log_target=True, reduction="batchmean")
                kl_scaled = kl * (temperature ** 2)
                
                if role == "distill":
                    kl_sum += kl_scaled
                else:
                    kl_sum += retain_weight * kl_scaled
                kl_count += 1
                
                # Free teacher logits immediately
                del teacher_data, tl
            
            l_kl = kl_sum / max(kl_count, 1) if kl_count > 0 \
                else torch.tensor(0.0, device=model.device, requires_grad=True)
            
            # Total loss
            loss = gate_loss + l_kl
            (loss / grad_accum).backward()
            
            epoch_loss += loss.item()
            epoch_kl += (l_kl.item() if torch.is_tensor(l_kl) else l_kl)
            epoch_gate_loss += gate_loss.item()
            n_steps += 1
            
            if n_steps % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(gate.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if n_steps % 5 == 0:
                gate_acc = n_gate_correct / max(n_gate_total, 1) * 100
                pbar.set_postfix(
                    loss=f"{epoch_loss/n_steps:.4f}",
                    kl=f"{epoch_kl/n_steps:.4f}",
                    gate=f"{epoch_gate_loss/n_steps:.4f}",
                    g_acc=f"{gate_acc:.1f}%",
                )
        
        # Flush remaining gradients
        if n_steps % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(gate.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        pbar.close()
        gate_acc = n_gate_correct / max(n_gate_total, 1) * 100
        logger.info(f"Epoch {epoch+1}: loss={epoch_loss/n_steps:.4f}, "
                    f"kl={epoch_kl/n_steps:.4f}, "
                    f"gate_loss={epoch_gate_loss/n_steps:.4f}, "
                    f"gate_acc={gate_acc:.1f}%")
    
    # Save
    model.enable_adapter_layers()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(gate.state_dict(), os.path.join(output_dir, "gate.pt"))
    
    save_json({
        "architecture": "GatedSingleLoRA",
        "model": model_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "gate_lr": learning_rate * gate_lr_mult,
        "temperature": temperature,
        "retain_weight": retain_weight,
        "n_distill": len(distill_data),
        "n_retain": len(retain_data),
        "final_gate_acc": gate_acc,
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
    adapter_dir = os.path.join(ADAPTER_ROOT, exp_name)
    
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
    # Import eval helpers from the iterative pipeline
    from run_iterative import (
        _run_mt_bench, _run_safety, _run_mmlu, _run_utility,
        SAFETY_BENCHMARKS,
    )
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"  EVALUATION — {exp_name}")
    logger.info(f"  Adapter: {adapter_dir}")
    logger.info(f"{'#'*70}\n")
    
    # Row 1: Baseline (no adapter, no persona)
    logger.info("=== Baseline ===")
    _run_mt_bench(exp_name, "baseline", base_model)
    _run_safety(exp_name, "baseline", base_model)
    _run_mmlu(exp_name, "baseline", base_model)
    
    # Row 2: Gated LoRA (adapter, no system prompt — the LoRA internalizes persona)
    # PeftModel.save_pretrained saves adapter under a subdirectory (e.g., persona_expert/)
    lora_adapter_path = os.path.join(adapter_dir, "persona_expert")
    gate_model_path = os.path.join(adapter_dir, "gate.pt")
    logger.info("=== Gated LoRA ===")
    _run_mt_bench(exp_name, "gated_lora", base_model, adapter_path=lora_adapter_path,
                  gate_path=gate_model_path)
    _run_safety(exp_name, "gated_lora", base_model, adapter_path=lora_adapter_path,
                gate_path=gate_model_path)
    _run_mmlu(exp_name, "gated_lora", base_model, adapter_path=lora_adapter_path)
    
    # Utility (G-Eval, Win Rate, KL)
    _run_utility(exp_name, base_model, lora_adapter_path)
    
    logger.info(f"\nPipeline complete. Results → {RESULTS_ROOT}/{exp_name}/")


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
            _run_mt_bench, _run_safety, _run_mmlu, _run_utility,
            SAFETY_BENCHMARKS,
        )
        
        logger.info(f"Eval-only mode. Adapter: {lora_adapter_path}")
        gate_model_path = os.path.join(adapter_dir, "gate.pt")
        logger.info("=== Baseline ===")
        _run_mt_bench(exp_name, "baseline", base_model)
        _run_safety(exp_name, "baseline", base_model)
        _run_mmlu(exp_name, "baseline", base_model)
        
        logger.info("=== Gated LoRA ===")
        _run_mt_bench(exp_name, "gated_lora", base_model, adapter_path=lora_adapter_path,
                      gate_path=gate_model_path)
        _run_safety(exp_name, "gated_lora", base_model, adapter_path=lora_adapter_path,
                    gate_path=gate_model_path)
        _run_mmlu(exp_name, "gated_lora", base_model, adapter_path=lora_adapter_path)
        
        _run_utility(exp_name, base_model, lora_adapter_path)
        logger.info(f"Eval complete. Results → {RESULTS_ROOT}/{exp_name}/")
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
            retain_weight=cfg.get("retain_weight", 0.5),
        )


if __name__ == "__main__":
    main()
