"""
PRISM Stage 3: Mixture-of-LoRA (MoLoRA) Distillation

Architecture:
  - K persona-specific LoRA experts (extremely low rank, r=2)
  - 1 null expert (identity: base model, no LoRA)
  - Lightweight router: query → expert selection

Training:
  - Router supervision: distill_set → persona expert k; retain_set → null expert
  - Expert distillation: KL(teacher || student) per assigned expert
  - Retain loss: KL to preserve base behavior for null-routed queries
  - Total: L = L_route + L_distill + λ * L_retain

Inference:
  - Router selects top expert per query
  - Only that expert's LoRA is applied (or none for null expert)

Resource profile (A100-80GB):
  - 7B model in bf16:    ~14GB VRAM
  - K=12 LoRA experts at r=2: ~24MB total (vs ~100MB for single r=32)
  - Router: ~50KB
  - Training activations: ~15-25GB with gradient checkpointing
  - Total VRAM peak:     ~35-45GB (safe for 80GB A100)

Usage:
  python -m scripts.prism.stage3_distill --model Qwen/Qwen2.5-7B-Instruct
  python -m scripts.prism.stage3_distill --model Qwen/Qwen2.5-7B-Instruct --epochs 3
"""

import os
import sys
import math
import argparse
import logging
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel, set_peft_model_state_dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    load_json, save_json, get_model_slug,
    build_chat_messages, format_chat_text,
    compute_logits, batch_compute_logits,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Defaults (tuned for A100-80GB)
# ============================================================

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAMBDA_RETAIN = 0.5  # weight for retain KL loss
EPOCHS = 5
LEARNING_RATE = 2e-4
LORA_R = 2           # extremely low rank per expert (K experts × r=2 ≈ 1 × r=24)
LORA_ALPHA = 4       # alpha=2r → scaling factor = 2.0
MICRO_BATCH = 2
GRAD_ACCUM = 8       # effective batch = MICRO_BATCH × GRAD_ACCUM = 16
MAX_LEN = 1024
TEMPERATURE = 2.0
ROUTER_LR_MULT = 5.0  # router learns faster than LoRA experts


# ============================================================
# Router Network
# ============================================================

class PersonaRouter(nn.Module):
    """Lightweight MLP router: hidden_state → expert selection.
    
    Input: last-token hidden state from the base model's first layer.
    Output: softmax distribution over K+1 experts (K personas + null).
    """
    def __init__(self, hidden_dim, num_experts, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts  # K+1 (includes null expert at index 0)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_experts),
        )
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch, hidden_dim) — e.g., last-token hidden state
        Returns:
            logits: (batch, num_experts)
        """
        return self.router(hidden_states)
    
    def predict(self, hidden_states):
        """Return expert index (argmax)."""
        logits = self.forward(hidden_states)
        return logits.argmax(dim=-1)


# ============================================================
# MoLoRA Manager
# ============================================================

class MoLoRAManager:
    """Manages K LoRA experts + null expert + router.
    
    Each expert is a named LoRA adapter applied to the same base model.
    The null expert (index 0) uses the base model with no adapter.
    
    Architecture:
      - Expert 0: null (no LoRA, identity)
      - Expert 1..K: persona-specific LoRA adapters
    """
    
    def __init__(self, model, tokenizer, persona_names, lora_r=2, lora_alpha=4):
        """
        Args:
            model: base CausalLM model
            tokenizer: tokenizer
            persona_names: list of K persona names (order defines expert 1..K)
            lora_r: LoRA rank per expert
            lora_alpha: LoRA alpha per expert
        """
        self.base_model = model
        self.tokenizer = tokenizer
        self.persona_names = list(persona_names)
        self.K = len(persona_names)
        self.num_experts = self.K + 1  # +1 for null expert
        
        # Map persona_name → expert index (1-indexed; 0 = null)
        self.persona_to_expert = {name: i + 1 for i, name in enumerate(self.persona_names)}
        self.expert_to_persona = {i + 1: name for i, name in enumerate(self.persona_names)}
        self.expert_to_persona[0] = "null"
        
        # Create K LoRA adapters as separate named adapters
        self.adapter_names = []
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]
        
        for i, name in enumerate(self.persona_names):
            adapter_name = f"expert_{name}"
            self.adapter_names.append(adapter_name)
            
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            
            if i == 0:
                # First adapter: use get_peft_model
                self.base_model = get_peft_model(self.base_model, config, adapter_name=adapter_name)
            else:
                # Subsequent adapters: add_adapter
                self.base_model.add_adapter(adapter_name, config)
        
        # Initialize router
        hidden_dim = self.base_model.config.hidden_size
        self.router = PersonaRouter(hidden_dim, self.num_experts).to(
            next(self.base_model.parameters()).device
        ).to(next(self.base_model.parameters()).dtype)
        
        # Print parameter summary
        total_lora = sum(
            p.numel() for p in self.base_model.parameters() if p.requires_grad
        )
        router_params = sum(p.numel() for p in self.router.parameters())
        logger.info(f"MoLoRA initialized:")
        logger.info(f"  Experts: {self.K} persona + 1 null = {self.num_experts} total")
        logger.info(f"  LoRA rank: {lora_r}, alpha: {lora_alpha}")
        logger.info(f"  Total LoRA params: {total_lora:,}")
        logger.info(f"  Router params: {router_params:,}")
        logger.info(f"  Personas: {self.persona_names}")
    
    def set_active_expert(self, expert_idx):
        """Activate a specific expert's LoRA adapter (or disable all for null)."""
        if expert_idx == 0:
            # Null expert: disable all adapters
            self.base_model.disable_adapter_layers()
        else:
            # Enable adapter layers and set the active one
            self.base_model.enable_adapter_layers()
            adapter_name = self.adapter_names[expert_idx - 1]
            self.base_model.set_adapter(adapter_name)
    
    def get_routing_target(self, persona_name, is_retain=False):
        """Get the expert index for a training sample."""
        if is_retain:
            return 0  # null expert
        return self.persona_to_expert.get(persona_name, 0)
    
    def get_trainable_params(self):
        """Return all trainable parameters (LoRA + router)."""
        params = []
        # LoRA params (all adapters)
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                params.append({"params": param, "lr_mult": 1.0, "name": name})
        # Router params (higher LR)
        for name, param in self.router.named_parameters():
            params.append({"params": param, "lr_mult": ROUTER_LR_MULT, "name": f"router.{name}"})
        return params
    
    def get_hidden_state(self, input_ids):
        """Extract hidden state for router input (first layer, last token).
        
        Uses the base model (no adapter) to get consistent routing features.
        """
        self.base_model.disable_adapter_layers()
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, output_hidden_states=True)
            # Use the first hidden layer's last-token representation
            # This is before any LoRA influence, so routing is consistent
            hidden = outputs.hidden_states[1]  # layer 1 (after embedding + first transformer)
            # Get last non-padding token for each sequence
            batch_size = input_ids.shape[0]
            last_token_hidden = []
            for b in range(batch_size):
                # Find last non-pad token
                non_pad = (input_ids[b] != self.tokenizer.pad_token_id).nonzero()
                if len(non_pad) > 0:
                    last_idx = non_pad[-1].item()
                else:
                    last_idx = input_ids.shape[1] - 1
                last_token_hidden.append(hidden[b, last_idx])
            hidden_states = torch.stack(last_token_hidden)
        self.base_model.enable_adapter_layers()
        return hidden_states
    
    def save(self, output_dir):
        """Save all LoRA adapters and router."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each adapter
        for adapter_name in self.adapter_names:
            adapter_dir = os.path.join(output_dir, adapter_name)
            self.base_model.set_adapter(adapter_name)
            self.base_model.save_pretrained(adapter_dir)
        
        # Save router
        router_path = os.path.join(output_dir, "router.pt")
        torch.save(self.router.state_dict(), router_path)
        
        # Save configuration
        config = {
            "persona_names": self.persona_names,
            "persona_to_expert": self.persona_to_expert,
            "num_experts": self.num_experts,
            "adapter_names": self.adapter_names,
        }
        save_json(config, os.path.join(output_dir, "molora_config.json"))
        logger.info(f"MoLoRA saved → {output_dir}")
    
    @classmethod
    def load(cls, model, tokenizer, output_dir):
        """Load saved MoLoRA (all adapters + router)."""
        config = load_json(os.path.join(output_dir, "molora_config.json"))
        persona_names = config["persona_names"]
        
        # Load each adapter
        for adapter_name in config["adapter_names"]:
            adapter_dir = os.path.join(output_dir, adapter_name)
            if os.path.exists(adapter_dir):
                model = PeftModel.from_pretrained(
                    model, adapter_dir, adapter_name=adapter_name, is_trainable=True
                )
        
        # Create manager (without re-initializing adapters)
        manager = cls.__new__(cls)
        manager.base_model = model
        manager.tokenizer = tokenizer
        manager.persona_names = persona_names
        manager.K = len(persona_names)
        manager.num_experts = manager.K + 1
        manager.persona_to_expert = config["persona_to_expert"]
        manager.expert_to_persona = {v: k for k, v in manager.persona_to_expert.items()}
        manager.expert_to_persona[0] = "null"
        manager.adapter_names = config["adapter_names"]
        
        # Load router
        hidden_dim = model.config.hidden_size
        manager.router = PersonaRouter(hidden_dim, manager.num_experts).to(
            next(model.parameters()).device
        ).to(next(model.parameters()).dtype)
        router_path = os.path.join(output_dir, "router.pt")
        if os.path.exists(router_path):
            manager.router.load_state_dict(torch.load(router_path, weights_only=True))
        
        return manager


# ============================================================
# Data formatting
# ============================================================

def _format_to_ids(tokenizer, sample, max_len=1024):
    """Convert a training sample to input_ids and labels."""
    full_text = format_chat_text(
        tokenizer, sample["system"], sample["instruction"], sample["output"]
    )
    encoding = tokenizer(full_text, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = encoding.input_ids[0]

    prompt_text = format_chat_text(
        tokenizer, sample["system"], sample["instruction"], add_generation_prompt=True
    )
    prompt_len = len(tokenizer(prompt_text, return_tensors="pt").input_ids[0])

    labels = input_ids.clone()
    labels[:prompt_len] = -100
    return input_ids, labels, prompt_len


# ============================================================
# Training loop
# ============================================================

def train(model_name, data_dir, output_dir, adapter_path=None, epochs=EPOCHS,
          retain_weight=LAMBDA_RETAIN,
          learning_rate=LEARNING_RATE, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
          micro_batch=MICRO_BATCH, grad_accum=GRAD_ACCUM,
          max_len=MAX_LEN, temperature=TEMPERATURE, save_every_epoch=True):
    """MoLoRA distillation training.
    
    L = L_route + L_distill + λ * L_retain

    Args:
        adapter_path: If provided, resume training from saved MoLoRA.
    """

    # ---- Verify data ----
    distill_path = os.path.join(data_dir, "distill_set.json")
    retain_path = os.path.join(data_dir, "retain_set.json")
    if not os.path.exists(distill_path):
        logger.error(f"Distill set not found: {distill_path}. Run Stage 1 & 2 first.")
        sys.exit(1)

    distill_data = load_json(distill_path)
    retain_data = load_json(retain_path) if os.path.exists(retain_path) else []
    logger.info(f"Loaded {len(distill_data)} distill + {len(retain_data)} retain samples")

    if len(distill_data) == 0:
        logger.error("No distill data. Exiting.")
        sys.exit(1)

    # ---- Discover personas from data ----
    persona_names = sorted(set(s.get("persona", "unknown") for s in distill_data))
    logger.info(f"Discovered {len(persona_names)} personas: {persona_names}")

    # ---- Load model + tokenizer ----
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    # ---- Load teacher logits (pre-computed by Stage 2) ----
    distill_logits_path = os.path.join(data_dir, "teacher_logits_distill.pt")
    retain_logits_path = os.path.join(data_dir, "teacher_logits_retain.pt")

    if os.path.exists(distill_logits_path):
        distill_logits = torch.load(distill_logits_path, weights_only=False)
        logger.info(f"Loaded distill teacher logits: {len(distill_logits)} samples")
    else:
        logger.warning("Distill teacher logits not found — computing from base model...")
        distill_logits = batch_compute_logits(model, tokenizer, distill_data, max_len,
                                              desc="Distill teacher logits")
        torch.save(distill_logits, distill_logits_path)

    if os.path.exists(retain_logits_path):
        retain_logits = torch.load(retain_logits_path, weights_only=False)
        logger.info(f"Loaded retain teacher logits: {len(retain_logits)} samples")
    elif retain_data:
        logger.warning("Retain teacher logits not found — computing from base model...")
        retain_logits = batch_compute_logits(model, tokenizer, retain_data, max_len,
                                             desc="Retain teacher logits")
        torch.save(retain_logits, retain_logits_path)
    else:
        retain_logits = []

    # ---- Initialize MoLoRA (fresh or resume) ----
    if adapter_path and os.path.exists(os.path.join(adapter_path, "molora_config.json")):
        logger.info(f"Resuming MoLoRA from: {adapter_path}")
        molora = MoLoRAManager.load(model, tokenizer, adapter_path)
    else:
        logger.info("Creating fresh MoLoRA")
        molora = MoLoRAManager(model, tokenizer, persona_names,
                               lora_r=lora_r, lora_alpha=lora_alpha)

    model = molora.base_model
    
    # Enable gradient checkpointing to reduce VRAM usage
    model.gradient_checkpointing_enable()

    # ---- Build training samples ----
    samples = []
    for i, s in enumerate(distill_data):
        input_ids, labels, prompt_len = _format_to_ids(tokenizer, s, max_len)
        logit_data = distill_logits[i] if i < len(distill_logits) else None
        persona = s.get("persona", "unknown")
        expert_idx = molora.get_routing_target(persona, is_retain=False)
        samples.append({
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": prompt_len,
            "role": "distill",
            "logit_data": logit_data,
            "expert_idx": expert_idx,
            "persona": persona,
        })

    for i, s in enumerate(retain_data):
        input_ids, labels, prompt_len = _format_to_ids(tokenizer, s, max_len)
        logit_data = retain_logits[i] if i < len(retain_logits) else None
        samples.append({
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": prompt_len,
            "role": "retain",
            "logit_data": logit_data,
            "expert_idx": 0,  # null expert
            "persona": "null",
        })

    n_samples = len(samples)
    steps_per_epoch = math.ceil(n_samples / micro_batch)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    total_optimizer_steps = optimizer_steps_per_epoch * epochs

    # Count per-expert samples
    expert_counts = {}
    for s in samples:
        eidx = s["expert_idx"]
        expert_counts[eidx] = expert_counts.get(eidx, 0) + 1
    logger.info(f"Training config:")
    logger.info(f"  Distill samples:  {len(distill_data)}")
    logger.info(f"  Retain samples:   {len(retain_data)}")
    logger.info(f"  Total samples:    {n_samples}")
    logger.info(f"  Experts:          {molora.num_experts} ({molora.K} persona + 1 null)")
    for eidx, count in sorted(expert_counts.items()):
        name = molora.expert_to_persona.get(eidx, f"expert_{eidx}")
        logger.info(f"    Expert {eidx} ({name}): {count} samples")
    logger.info(f"  Micro-batch:      {micro_batch}")
    logger.info(f"  Grad accum:       {grad_accum}")
    logger.info(f"  Effective batch:  {micro_batch * grad_accum}")
    logger.info(f"  Steps/epoch:      {steps_per_epoch}")
    logger.info(f"  Optimizer steps:  {total_optimizer_steps}")
    logger.info(f"  λ_retain:         {retain_weight}")

    # ---- Optimizer + scheduler ----
    # Separate parameter groups: LoRA experts + Router (with higher LR)
    model.train()
    molora.router.train()
    
    lora_params = [p for p in model.parameters() if p.requires_grad]
    router_params = list(molora.router.parameters())
    
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": learning_rate, "weight_decay": 0.01},
        {"params": router_params, "lr": learning_rate * ROUTER_LR_MULT, "weight_decay": 0.01},
    ])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(50, total_optimizer_steps // 10),
        num_training_steps=total_optimizer_steps,
    )

    # ---- Training epochs ----
    for epoch in range(epochs):
        indices = torch.randperm(n_samples).tolist()
        epoch_loss = 0.0
        epoch_kl_distill = 0.0
        epoch_kl_retain = 0.0
        epoch_route_loss = 0.0
        epoch_route_acc = 0.0
        optimizer.zero_grad()
        n_steps = 0
        n_route_correct = 0
        n_route_total = 0

        pbar = tqdm(range(0, n_samples, micro_batch), desc=f"Epoch {epoch+1}/{epochs}")

        for batch_start in pbar:
            batch_idx = indices[batch_start : batch_start + micro_batch]

            # --- Pad & collate micro-batch ---
            batch_input_ids = []
            batch_prompt_lens = []
            batch_teacher_logits = []
            batch_roles = []
            batch_expert_targets = []
            max_seq = max(samples[i]["input_ids"].shape[0] for i in batch_idx)

            for i in batch_idx:
                s = samples[i]
                seq_len = s["input_ids"].shape[0]
                pad_len = max_seq - seq_len

                ids = F.pad(s["input_ids"], (0, pad_len), value=tokenizer.pad_token_id)
                batch_input_ids.append(ids)
                batch_prompt_lens.append(s["prompt_len"])
                batch_roles.append(s["role"])
                batch_expert_targets.append(s["expert_idx"])
                batch_teacher_logits.append(
                    s["logit_data"]["logits"] if s["logit_data"] else None
                )

            input_ids = torch.stack(batch_input_ids).to(model.device)
            expert_targets = torch.tensor(batch_expert_targets, device=model.device)

            # --- Router forward (uses base model hidden states) ---
            hidden_states = molora.get_hidden_state(input_ids)
            route_logits = molora.router(hidden_states)
            
            # Router loss (cross-entropy)
            route_loss = F.cross_entropy(route_logits, expert_targets)
            
            # Router accuracy tracking
            route_preds = route_logits.argmax(dim=-1)
            n_route_correct += (route_preds == expert_targets).sum().item()
            n_route_total += len(batch_idx)

            # --- Expert-specific forward passes ---
            # Group samples by their target expert for efficient batching
            kl_distill_sum = 0.0
            kl_distill_count = 0
            kl_retain_sum = 0.0
            kl_retain_count = 0

            # Process each sample with its assigned expert
            for j in range(len(batch_idx)):
                expert_idx = batch_expert_targets[j]
                tl = batch_teacher_logits[j]
                if tl is None:
                    continue

                # Activate the correct expert
                molora.set_active_expert(expert_idx)

                # Forward pass with this expert
                single_ids = input_ids[j:j+1]
                outputs = model(input_ids=single_ids)
                logits = outputs.logits

                pl = batch_prompt_lens[j]
                s_resp = logits[0, pl - 1: -1, :]
                min_len = min(s_resp.shape[0], tl.shape[0])
                if min_len == 0:
                    continue

                # KL(teacher || student)
                t_probs = F.log_softmax(tl[:min_len].to(logits.device).float() / temperature, dim=-1)
                s_probs = F.log_softmax(s_resp[:min_len].float() / temperature, dim=-1)
                kl = F.kl_div(s_probs, t_probs, log_target=True, reduction="batchmean")
                kl_scaled = kl * (temperature ** 2)

                if batch_roles[j] == "distill":
                    kl_distill_sum += kl_scaled
                    kl_distill_count += 1
                else:
                    kl_retain_sum += kl_scaled
                    kl_retain_count += 1

            # Average each loss separately
            l_distill = kl_distill_sum / max(kl_distill_count, 1) if kl_distill_count > 0 \
                else torch.tensor(0.0, device=model.device, requires_grad=True)
            l_retain = kl_retain_sum / max(kl_retain_count, 1) if kl_retain_count > 0 \
                else torch.tensor(0.0, device=model.device)

            # Total loss: L = L_route + L_distill + λ * L_retain
            loss = route_loss + l_distill + retain_weight * l_retain
            (loss / grad_accum).backward()

            epoch_loss += loss.item()
            epoch_kl_distill += l_distill.item()
            epoch_kl_retain += l_retain.item()
            epoch_route_loss += route_loss.item()
            n_steps += 1

            # --- Optimizer step ---
            if n_steps % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(molora.router.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if n_steps % 5 == 0:
                route_acc = n_route_correct / max(n_route_total, 1) * 100
                pbar.set_postfix(
                    loss=f"{epoch_loss/n_steps:.4f}",
                    kl_d=f"{epoch_kl_distill/n_steps:.4f}",
                    kl_r=f"{epoch_kl_retain/n_steps:.4f}",
                    rt=f"{epoch_route_loss/n_steps:.4f}",
                    rt_acc=f"{route_acc:.1f}%",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        # Flush remaining gradients
        if n_steps % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(molora.router.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.close()
        route_acc = n_route_correct / max(n_route_total, 1) * 100
        logger.info(f"Epoch {epoch+1}: loss={epoch_loss/n_steps:.4f}, "
                    f"kl_distill={epoch_kl_distill/n_steps:.4f}, "
                    f"kl_retain={epoch_kl_retain/n_steps:.4f}, "
                    f"route_loss={epoch_route_loss/n_steps:.4f}, "
                    f"route_acc={route_acc:.1f}%")

        if save_every_epoch:
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
            molora.save(epoch_dir)
            logger.info(f"Checkpoint → {epoch_dir}")

    # ---- Save final model ----
    molora.save(output_dir)
    tokenizer.save_pretrained(output_dir)

    config = {
        "model": model_name,
        "architecture": "MoLoRA",
        "num_experts": molora.num_experts,
        "persona_names": molora.persona_names,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "router_lr": learning_rate * ROUTER_LR_MULT,
        "temperature": temperature,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "effective_batch": micro_batch * grad_accum,
        "retain_weight": retain_weight,
        "n_distill": len(distill_data),
        "n_retain": len(retain_data),
    }
    save_json(config, os.path.join(output_dir, "training_config.json"))
    logger.info(f"Stage 3 (MoLoRA) complete. Model saved → {output_dir}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PRISM Stage 3: MoLoRA Distillation")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter_path", default=None,
                        help="Resume training from saved MoLoRA directory")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--retain_weight", type=float, default=LAMBDA_RETAIN)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--micro_batch", type=int, default=MICRO_BATCH)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    slug = get_model_slug(args.model)
    data_dir = args.data_dir or f"dataset/synthetic/persona_prism/{slug}"
    output_dir = args.output_dir or f"models/persona_prism/{slug}"

    train(
        model_name=args.model,
        data_dir=data_dir,
        output_dir=output_dir,
        adapter_path=args.adapter_path,
        epochs=args.epochs,
        retain_weight=args.retain_weight,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        max_len=args.max_len,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
