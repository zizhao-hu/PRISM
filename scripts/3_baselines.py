"""
Baseline Methods for DREAM-C2L Comparison

Implements three competing baselines for the paper table:
  1. Prompt Tuning:        Learnable soft prompts distilled from with-context logits
  2. Context Compression:  ICAE-style autoencoder that compresses the context into
                           a small set of memory embeddings
  3. Context Distillation: Already implemented as std_cd in the ablation; this script
                           provides a direct entry point

Usage:
  # Prompt Tuning
  python 3_baselines.py prompt_tuning --model Qwen/Qwen2.5-1.5B-Instruct \
      --context_file dataset/context/1_general_safety.txt \
      --data_dir dataset/ablation/std_cd/Qwen2.5-1.5B-Instruct \
      --output_dir models/baselines/prompt_tuning

  # Context Compression (ICAE-style)
  python 3_baselines.py context_compression --model Qwen/Qwen2.5-1.5B-Instruct \
      --context_file dataset/context/1_general_safety.txt \
      --data_dir dataset/ablation/std_cd/Qwen2.5-1.5B-Instruct \
      --output_dir models/baselines/context_compression

  # Context Distillation (wrapper for std_cd)
  python 3_baselines.py context_distillation --model Qwen/Qwen2.5-1.5B-Instruct \
      --data_dir dataset/ablation/std_cd/Qwen2.5-1.5B-Instruct \
      --output_dir models/baselines/context_distillation
"""
import os
import sys
import math
import argparse
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

try:
    from utils import (
        load_json, save_json, build_chat_messages, format_chat_text,
        load_model, unload_model, TRIGGER_TOKEN,
    )
except ImportError:
    from scripts.utils import (
        load_json, save_json, build_chat_messages, format_chat_text,
        load_model, unload_model, TRIGGER_TOKEN,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 1. PROMPT TUNING
# ============================================================
# Core idea: Learn N_SOFT soft-prompt embeddings that, when prepended
# to the user input, make the model behave as if the safety context
# was present in the system prompt.
#
# Training objective:
#   KL( P_teacher(y | context, x) || P_student(y | soft_prompt, x) )
#
# where teacher = base model with full text context in system prompt
#       student = base model with learned soft embeddings prepended
#
# At inference: prepend soft embeddings (no LoRA adapter needed,
# but we save them as a standalone module).
# ============================================================

class SoftPromptModel(nn.Module):
    """Wraps a frozen LLM with learnable soft prompt embeddings."""
    
    def __init__(self, base_model, n_soft_tokens=16):
        super().__init__()
        self.base_model = base_model
        self.n_soft_tokens = n_soft_tokens
        hidden_size = base_model.config.hidden_size
        
        # Learnable soft prompt embeddings
        self.soft_prompt = nn.Parameter(
            torch.randn(n_soft_tokens, hidden_size) * 0.02
        )
        
        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False
    
    @property
    def device(self):
        return self.soft_prompt.device
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass with soft prompt prepended to input embeddings."""
        batch_size = input_ids.size(0)
        
        # Get input embeddings from the base model
        input_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        # Prepend soft prompt embeddings
        soft_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        full_embeds = torch.cat([soft_embeds, input_embeds], dim=1)
        
        # Adjust attention mask
        if attention_mask is not None:
            soft_mask = torch.ones(batch_size, self.n_soft_tokens,
                                   device=attention_mask.device, dtype=attention_mask.dtype)
            full_mask = torch.cat([soft_mask, attention_mask], dim=1)
        else:
            full_mask = None
        
        # Adjust labels (shift by n_soft_tokens, with -100 for soft prompt positions)
        if labels is not None:
            soft_labels = torch.full((batch_size, self.n_soft_tokens), -100,
                                     device=labels.device, dtype=labels.dtype)
            full_labels = torch.cat([soft_labels, labels], dim=1)
        else:
            full_labels = None
        
        outputs = self.base_model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )
        return outputs
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate with soft prompt prepended."""
        batch_size = input_ids.size(0)
        input_embeds = self.base_model.get_input_embeddings()(input_ids)
        soft_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        full_embeds = torch.cat([soft_embeds, input_embeds], dim=1)
        
        if attention_mask is not None:
            soft_mask = torch.ones(batch_size, self.n_soft_tokens,
                                   device=attention_mask.device, dtype=attention_mask.dtype)
            full_mask = torch.cat([soft_mask, attention_mask], dim=1)
        else:
            full_mask = None
        
        return self.base_model.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            **kwargs,
        )
    
    def save(self, output_dir):
        """Save only the learned soft prompt embeddings."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save({
            "soft_prompt": self.soft_prompt.data,
            "n_soft_tokens": self.n_soft_tokens,
        }, os.path.join(output_dir, "soft_prompt.pt"))
        logger.info(f"Saved soft prompt ({self.n_soft_tokens} tokens) to {output_dir}")
    
    @classmethod
    def load(cls, base_model, output_dir):
        """Load a saved soft prompt."""
        ckpt = torch.load(os.path.join(output_dir, "soft_prompt.pt"), weights_only=True)
        model = cls(base_model, n_soft_tokens=ckpt["n_soft_tokens"])
        model.soft_prompt.data = ckpt["soft_prompt"].to(model.device)
        return model


def train_prompt_tuning(args):
    """Train soft prompt via KL distillation from with-context teacher."""
    logger.info("="*60)
    logger.info("PROMPT TUNING TRAINING")
    logger.info("="*60)
    
    output_dir = args.output_dir
    if os.path.exists(os.path.join(output_dir, "soft_prompt.pt")):
        logger.info("Soft prompt already trained, skipping")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data (reuse std_cd synthetic data)
    pos_data = load_json(os.path.join(args.data_dir, "positive_safety_data.json"))
    neg_data_path = os.path.join(args.data_dir, "negative_utility_data.json")
    neg_data = load_json(neg_data_path) if os.path.exists(neg_data_path) else []
    all_data = pos_data + neg_data
    logger.info(f"Loaded {len(all_data)} training samples")
    
    # Load context
    context = open(args.context_file).read().strip()
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16
    )
    
    # Create soft prompt model
    soft_model = SoftPromptModel(base_model, n_soft_tokens=args.n_soft_tokens)
    soft_model.soft_prompt = soft_model.soft_prompt.to(base_model.device)
    
    # Optimizer (only soft prompt parameters)
    optimizer = torch.optim.AdamW([soft_model.soft_prompt], lr=args.learning_rate, weight_decay=0.01)
    
    total_steps = args.epochs * len(all_data)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=min(50, total_steps // 10),
        num_training_steps=total_steps,
    )
    
    temperature = args.temperature
    
    for epoch in range(args.epochs):
        indices = torch.randperm(len(all_data)).tolist()
        total_loss = 0.0
        
        pbar = tqdm(range(len(indices)), desc=f"PromptTune Epoch {epoch+1}/{args.epochs}")
        
        for step_i, idx in enumerate(indices):
            sample = all_data[idx]
            instruction = sample["instruction"]
            output_text = sample["output"]
            system_prompt = sample.get("system", "")
            
            # Teacher: base model with context in system prompt
            teacher_messages = build_chat_messages(tokenizer, context, instruction, output_text)
            teacher_text = tokenizer.apply_chat_template(teacher_messages, tokenize=False)
            teacher_enc = tokenizer(teacher_text, return_tensors="pt", truncation=True,
                                    max_length=args.max_len)
            teacher_ids = teacher_enc.input_ids.to(base_model.device)
            
            # Get teacher prompt length (without assistant response)
            teacher_prompt_msgs = build_chat_messages(tokenizer, context, instruction)
            teacher_prompt_text = tokenizer.apply_chat_template(
                teacher_prompt_msgs, tokenize=False, add_generation_prompt=True)
            teacher_prompt_len = len(tokenizer(teacher_prompt_text, return_tensors="pt").input_ids[0])
            
            # Teacher logits (from frozen base model with context)
            with torch.no_grad():
                teacher_out = base_model(input_ids=teacher_ids)
                teacher_logits = teacher_out.logits[0, teacher_prompt_len - 1:-1, :]
            
            # Student: soft prompt model WITHOUT context
            # Just instruction → output, but with soft prompt prepended
            student_messages = [{"role": "user", "content": instruction}]
            if output_text:
                student_messages.append({"role": "assistant", "content": output_text})
            student_text = tokenizer.apply_chat_template(student_messages, tokenize=False)
            student_enc = tokenizer(student_text, return_tensors="pt", truncation=True,
                                    max_length=args.max_len)
            student_ids = student_enc.input_ids.to(base_model.device)
            
            # Student prompt length (without assistant response)
            student_prompt_msgs = [{"role": "user", "content": instruction}]
            student_prompt_text = tokenizer.apply_chat_template(
                student_prompt_msgs, tokenize=False, add_generation_prompt=True)
            student_prompt_len = len(tokenizer(student_prompt_text, return_tensors="pt").input_ids[0])
            
            # Student forward (with soft prompt)
            student_out = soft_model(input_ids=student_ids)
            # Account for soft prompt offset in logits
            student_logits = student_out.logits[0, 
                (soft_model.n_soft_tokens + student_prompt_len - 1):
                (soft_model.n_soft_tokens + student_ids.size(1) - 1), :]
            
            # Align lengths
            min_len = min(teacher_logits.size(0), student_logits.size(0))
            if min_len == 0:
                pbar.update(1)
                continue
            
            # KL divergence loss
            t_lp = F.log_softmax(teacher_logits[:min_len] / temperature, dim=-1)
            s_lp = F.log_softmax(student_logits[:min_len].float() / temperature, dim=-1)
            loss = F.kl_div(s_lp, t_lp, log_target=True, reduction="batchmean") * (temperature ** 2)
            
            loss.backward()
            total_loss += loss.item()
            
            torch.nn.utils.clip_grad_norm_([soft_model.soft_prompt], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            pbar.update(1)
            if step_i % 10 == 0:
                pbar.set_postfix(loss=f"{total_loss / (step_i + 1):.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        pbar.close()
        logger.info(f"Epoch {epoch+1}: avg_loss={total_loss / len(indices):.4f}")
    
    # Save
    soft_model.save(output_dir)
    save_json({
        "method": "prompt_tuning",
        "model": args.model,
        "n_soft_tokens": args.n_soft_tokens,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "data_dir": args.data_dir,
    }, os.path.join(output_dir, "training_config.json"))


# ============================================================
# 2. CONTEXT COMPRESSION (ICAE-style)
# ============================================================
# Core idea: Train LoRA adapters + memory token embeddings that
# compress the safety context into K memory slots. At inference,
# the compressed memory tokens are prepended to the query.
#
# Architecture (simplified ICAE for our setting):
#   Encoder: base_model + LoRA → processes [context_tokens, mem_tokens]
#            → extracts hidden states at mem_token positions
#   Decoder: frozen base_model → processes [compressed_mem, query_tokens]
#            → predicts response
#
# Training losses:
#   1. AE loss: reconstruct the context from compressed memory + AE token
#   2. LM loss: predict response given compressed memory + query
#      Teacher forcing: minimize CE against with-context logits
# ============================================================

class ContextCompressor(nn.Module):
    """ICAE-style context compressor for safety context."""
    
    def __init__(self, base_model, tokenizer, n_mem_tokens=32, lora_config=None):
        super().__init__()
        self.n_mem_tokens = n_mem_tokens
        self.hidden_size = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size
        
        # Encoder: base model + LoRA for compression
        if lora_config is not None:
            self.encoder = get_peft_model(base_model, lora_config)
        else:
            self.encoder = base_model
        
        # Decoder: frozen copy (we reuse encoder with adapter disabled for memory efficiency)
        # At inference, use encoder with adapter disabled
        
        # Memory token embeddings (appended to context for compression)
        self.mem_embeddings = nn.Embedding(n_mem_tokens, self.hidden_size)
        nn.init.normal_(self.mem_embeddings.weight, std=0.02)
        
        # Special tokens
        self.ae_token_embed = nn.Parameter(torch.randn(1, self.hidden_size) * 0.02)
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = tokenizer
    
    @property
    def device(self):
        return self.mem_embeddings.weight.device
    
    def compress(self, context_ids):
        """Compress context into memory token representations.
        
        Args:
            context_ids: [1, context_len] token IDs
            
        Returns:
            compressed: [1, n_mem_tokens, hidden_size] compressed representations
        """
        batch_size = context_ids.size(0)
        
        # Get context embeddings from base model
        context_embeds = self.encoder.get_input_embeddings()(context_ids)
        
        # Append memory token embeddings
        mem_ids = torch.arange(self.n_mem_tokens, device=context_ids.device)
        mem_embeds = self.mem_embeddings(mem_ids).unsqueeze(0).expand(batch_size, -1, -1)
        
        # [context_embeds, mem_embeds]
        full_embeds = torch.cat([context_embeds, mem_embeds], dim=1)
        
        # Forward through encoder (with LoRA)
        outputs = self.encoder(inputs_embeds=full_embeds, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # last layer
        
        # Extract memory token positions
        compressed = hidden[:, -self.n_mem_tokens:, :]
        
        return compressed
    
    def forward(self, context_ids, query_ids, labels, mode="lm"):
        """
        Forward pass for training.
        
        Args:
            context_ids: [1, ctx_len] context token IDs
            query_ids: [1, query_len] either query+response (LM mode) or context itself (AE mode)
            labels: [1, query_len] target labels (-100 for masked positions)
            mode: "lm" for language modeling, "ae" for autoencoding
        """
        # Step 1: Compress context
        compressed = self.compress(context_ids)
        
        # Step 2: Build decoder input
        if mode == "ae":
            # AE mode: [compressed, ae_token] → reconstruct context
            ae_embed = self.ae_token_embed.unsqueeze(0).expand(context_ids.size(0), -1, -1)
            query_embeds = self.encoder.get_input_embeddings()(query_ids)
            # Decoder input: [compressed_mem, ae_token, context_tokens_shifted]
            decoder_input = torch.cat([compressed, ae_embed, query_embeds], dim=1)
        else:
            # LM mode: [compressed_mem] + query_embeds → predict response
            query_embeds = self.encoder.get_input_embeddings()(query_ids)
            decoder_input = torch.cat([compressed, query_embeds], dim=1)
        
        # Step 3: Forward through decoder (encoder with adapter disabled)
        # For memory efficiency, use same model but disable LoRA
        if hasattr(self.encoder, 'disable_adapter_layers'):
            self.encoder.disable_adapter_layers()
        
        decoder_out = self.encoder(inputs_embeds=decoder_input)
        
        if hasattr(self.encoder, 'enable_adapter_layers'):
            self.encoder.enable_adapter_layers()
        
        # Step 4: Compute loss
        logits = decoder_out.logits
        
        # Shift for labels: account for compressed prefix
        n_prefix = self.n_mem_tokens + (1 if mode == "ae" else 0)
        
        # Labels need padding for prefix
        prefix_labels = torch.full(
            (labels.size(0), n_prefix), -100,
            device=labels.device, dtype=labels.dtype
        )
        full_labels = torch.cat([prefix_labels, labels], dim=1)
        
        # Standard CE loss (shift by 1)
        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = full_labels[:, 1:].contiguous().view(-1)
        
        loss = self.loss_fn(shift_logits, shift_labels)
        return loss
    
    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # Save LoRA adapter
        if hasattr(self.encoder, 'save_pretrained'):
            self.encoder.save_pretrained(output_dir)
        # Save memory embeddings + special tokens
        torch.save({
            "mem_embeddings": self.mem_embeddings.state_dict(),
            "ae_token_embed": self.ae_token_embed.data,
            "n_mem_tokens": self.n_mem_tokens,
        }, os.path.join(output_dir, "compressor_state.pt"))
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved context compressor to {output_dir}")
    
    @classmethod
    def load(cls, model_name, output_dir, device="auto"):
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, torch_dtype=torch.bfloat16)
        
        state = torch.load(os.path.join(output_dir, "compressor_state.pt"), weights_only=True)
        compressor = cls(base_model, tokenizer, n_mem_tokens=state["n_mem_tokens"])
        compressor.mem_embeddings.load_state_dict(state["mem_embeddings"])
        compressor.ae_token_embed.data = state["ae_token_embed"]
        
        # Load LoRA if present
        from peft import PeftModel
        if os.path.exists(os.path.join(output_dir, "adapter_config.json")):
            compressor.encoder = PeftModel.from_pretrained(base_model, output_dir)
        
        return compressor


def train_context_compression(args):
    """Train ICAE-style context compressor."""
    logger.info("="*60)
    logger.info("CONTEXT COMPRESSION TRAINING (ICAE-style)")
    logger.info("="*60)
    
    output_dir = args.output_dir
    if os.path.exists(os.path.join(output_dir, "compressor_state.pt")):
        logger.info("Context compressor already trained, skipping")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data
    pos_data = load_json(os.path.join(args.data_dir, "positive_safety_data.json"))
    neg_data_path = os.path.join(args.data_dir, "negative_utility_data.json")
    neg_data = load_json(neg_data_path) if os.path.exists(neg_data_path) else []
    all_data = pos_data + neg_data
    logger.info(f"Loaded {len(all_data)} training samples")
    
    # Load context
    context = open(args.context_file).read().strip()
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16
    )
    
    # LoRA config for encoder
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.1,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    
    compressor = ContextCompressor(
        base_model, tokenizer, n_mem_tokens=args.n_mem_tokens, lora_config=lora_config
    )
    
    # Tokenize context once
    context_ids = tokenizer(context, return_tensors="pt", truncation=True,
                            max_length=args.max_len).input_ids.to(base_model.device)
    
    # Optimizer: LoRA params + memory embeddings + AE token
    trainable_params = list(filter(lambda p: p.requires_grad, compressor.encoder.parameters()))
    trainable_params += [compressor.mem_embeddings.weight, compressor.ae_token_embed]
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    total_steps = args.epochs * len(all_data) * 2  # 2 losses per sample
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=min(50, total_steps // 10),
        num_training_steps=total_steps,
    )
    
    for epoch in range(args.epochs):
        indices = torch.randperm(len(all_data)).tolist()
        total_ae_loss, total_lm_loss = 0.0, 0.0
        
        pbar = tqdm(range(len(indices)), desc=f"ICAE Epoch {epoch+1}/{args.epochs}")
        
        for step_i, idx in enumerate(indices):
            sample = all_data[idx]
            instruction = sample["instruction"]
            output_text = sample["output"]
            
            optimizer.zero_grad()
            
            # --- AE loss: compress context → reconstruct ---
            # Target: reconstruct the context tokens
            ae_labels = context_ids.clone()
            ae_loss = compressor(context_ids, context_ids, ae_labels, mode="ae")
            
            # --- LM loss: compress context → predict response ---
            # Build query+response sequence
            query_messages = [{"role": "user", "content": instruction}]
            if output_text:
                query_messages.append({"role": "assistant", "content": output_text})
            query_text = tokenizer.apply_chat_template(query_messages, tokenize=False)
            query_enc = tokenizer(query_text, return_tensors="pt", truncation=True,
                                  max_length=args.max_len)
            query_ids = query_enc.input_ids.to(base_model.device)
            
            # Labels: mask query prompt, keep response
            prompt_msgs = [{"role": "user", "content": instruction}]
            prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False,
                                                        add_generation_prompt=True)
            prompt_len = len(tokenizer(prompt_text, return_tensors="pt").input_ids[0])
            
            lm_labels = query_ids.clone()
            lm_labels[0, :prompt_len] = -100
            
            lm_loss = compressor(context_ids, query_ids, lm_labels, mode="lm")
            
            # Combined loss
            loss = ae_loss + lm_loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            
            total_ae_loss += ae_loss.item()
            total_lm_loss += lm_loss.item()
            
            pbar.update(1)
            if step_i % 10 == 0:
                n = step_i + 1
                pbar.set_postfix(
                    ae=f"{total_ae_loss/n:.4f}",
                    lm=f"{total_lm_loss/n:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
        
        pbar.close()
        n = len(indices)
        logger.info(f"Epoch {epoch+1}: ae_loss={total_ae_loss/n:.4f}, lm_loss={total_lm_loss/n:.4f}")
    
    compressor.save(output_dir)
    save_json({
        "method": "context_compression",
        "model": args.model,
        "n_mem_tokens": args.n_mem_tokens,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
    }, os.path.join(output_dir, "training_config.json"))


# ============================================================
# 3. CONTEXT DISTILLATION
# ============================================================
# This is just standard CD = our ablation std_cd.
# We provide a wrapper so it can be invoked uniformly.
# ============================================================

def train_context_distillation(args):
    """Context distillation — delegates to 1_train.py with distill mode."""
    logger.info("="*60)
    logger.info("CONTEXT DISTILLATION (std_cd)")
    logger.info("="*60)
    
    output_dir = args.output_dir
    if os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        logger.info("Context distillation already trained, skipping")
        return
    
    # Shell out to 1_train.py
    import subprocess
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "1_train.py"),
        "--model", args.model,
        "--data_dir", args.data_dir,
        "--output_dir", output_dir,
        "--loss_mode", "distill",
        "--epochs", str(args.epochs),
        "--learning_rate", str(args.learning_rate),
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--max_len", str(args.max_len),
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ============================================================
# Evaluation helpers for baselines
# ============================================================

def generate_with_soft_prompt(model_name, output_dir, prompts, max_new_tokens=256):
    """Generate responses using a trained soft prompt model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    soft_model = SoftPromptModel.load(base_model, output_dir)
    soft_model.eval()
    
    generations = []
    for prompt in tqdm(prompts, desc="Generating (Prompt Tuning)"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc.input_ids.to(base_model.device)
        attn = enc.attention_mask.to(base_model.device)
        
        with torch.no_grad():
            out = soft_model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only new tokens (account for soft prompt offset in output)
        # generate returns token IDs, not embeddings, so offset is just input_ids length
        response = tokenizer.decode(out[0][input_ids.size(1):], skip_special_tokens=True).strip()
        generations.append({"prompt": prompt, "response": response, "condition": "prompt_tuning"})
    
    return generations


def generate_with_compressor(model_name, output_dir, context_file, prompts, max_new_tokens=256):
    """Generate responses using a trained context compressor."""
    context = open(context_file).read().strip()
    compressor = ContextCompressor.load(model_name, output_dir)
    compressor.eval()
    tokenizer = compressor.tokenizer
    
    # Compress context once
    context_ids = tokenizer(context, return_tensors="pt", truncation=True,
                            max_length=1024).input_ids.to(compressor.device)
    
    with torch.no_grad():
        compressed = compressor.compress(context_ids)  # [1, n_mem, hidden]
    
    generations = []
    for prompt in tqdm(prompts, desc="Generating (Context Compression)"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc.input_ids.to(compressor.device)
        
        # Build input: [compressed_mem, query_embeds]
        query_embeds = compressor.encoder.get_input_embeddings()(input_ids)
        full_embeds = torch.cat([compressed, query_embeds], dim=1)
        full_mask = torch.ones(1, full_embeds.size(1), device=compressor.device)
        
        # Generate using decoder (encoder with adapter disabled)
        if hasattr(compressor.encoder, 'disable_adapter_layers'):
            compressor.encoder.disable_adapter_layers()
        
        with torch.no_grad():
            out = compressor.encoder.generate(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        if hasattr(compressor.encoder, 'enable_adapter_layers'):
            compressor.encoder.enable_adapter_layers()
        
        response = tokenizer.decode(out[0][full_embeds.size(1):], skip_special_tokens=True).strip()
        generations.append({"prompt": prompt, "response": response, "condition": "context_compression"})
    
    return generations


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DREAM Baseline Methods")
    subparsers = parser.add_subparsers(dest="method", help="Baseline method to run")
    
    # Shared arguments
    def add_common_args(p):
        p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
        p.add_argument("--context_file", default="dataset/context/1_general_safety.txt")
        p.add_argument("--data_dir", required=True)
        p.add_argument("--output_dir", required=True)
        p.add_argument("--epochs", type=int, default=3)
        p.add_argument("--learning_rate", type=float, default=2e-4)
        p.add_argument("--max_len", type=int, default=1024)
        p.add_argument("--lora_r", type=int, default=64)
        p.add_argument("--lora_alpha", type=int, default=16)
    
    # Prompt Tuning
    pt = subparsers.add_parser("prompt_tuning", help="Soft prompt distillation")
    add_common_args(pt)
    pt.add_argument("--n_soft_tokens", type=int, default=16,
                    help="Number of learnable soft prompt tokens")
    pt.add_argument("--temperature", type=float, default=2.0,
                    help="KL distillation temperature")
    
    # Context Compression
    cc = subparsers.add_parser("context_compression", help="ICAE-style context compression")
    add_common_args(cc)
    cc.add_argument("--n_mem_tokens", type=int, default=32,
                    help="Number of memory tokens for compression")
    
    # Context Distillation
    cd = subparsers.add_parser("context_distillation", help="Standard context distillation (wraps 1_train.py)")
    add_common_args(cd)
    
    args = parser.parse_args()
    
    if args.method is None:
        parser.print_help()
        sys.exit(1)
    
    if args.method == "prompt_tuning":
        train_prompt_tuning(args)
    elif args.method == "context_compression":
        train_context_compression(args)
    elif args.method == "context_distillation":
        train_context_distillation(args)
    else:
        parser.error(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()
