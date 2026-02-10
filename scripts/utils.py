"""
DREAM-C2L Shared Utilities Module

Contains common helper functions used across the training and evaluation pipelines.
"""
import os
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TRIGGER_TOKEN = "<safety_mode>"

# Benchmark definitions
BENCHMARKS = [
    {"name": "HarmBench", "path": "dataset/eval/harmbench_all.csv"},
    {"name": "Jailbreak", "path": "dataset/eval/jailbreak_prompts.json"},
    {"name": "PINT", "path": "dataset/eval/pint_injection_prompts.json"},
    {"name": "PKU_SafeRLHF", "path": "dataset/eval/pku_saferlhf_prompts.json"},
]

# Context files for ablation study
CONTEXT_FILES = [
    {"name": "1_general_safety", "path": "dataset/context/1_general_safety.txt"},
    {"name": "2_target_safety", "path": "dataset/context/2_target_safety.txt"},
    {"name": "3_claude_safety", "path": "dataset/context/3_claude_safety.txt"},
    {"name": "4_claude_system", "path": "dataset/context/4_claude_system.txt"},
]


def load_json(path):
    """Load JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path, indent=2):
    """Save data to JSON file with UTF-8 encoding."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.info(f"Saved to: {path}")


def load_text(path):
    """Load text file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_text(content, path):
    """Save text to file with UTF-8 encoding."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def load_context_prompt(context_path=None):
    """Load safety context prompt from a file."""
    if context_path and os.path.exists(context_path):
        logger.info(f"Loading context from: {context_path}")
        return load_text(context_path)
    
    # Fallback to default path
    default_path = "dataset/context/1_general_safety.txt"
    if os.path.exists(default_path):
        logger.info(f"Loading default context from: {default_path}")
        return load_text(default_path)
    
    logger.warning("No context file found, using default prompt")
    return "You are a helpful AI assistant."


def load_model(model_name, adapter_path=None):
    """
    Load a model and tokenizer from HuggingFace.
    
    Args:
        model_name: HuggingFace model name or path
        adapter_path: Optional path to LoRA adapter
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading Model: {model_name} (Adapter: {adapter_path or 'None'})")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    if adapter_path:
        logger.info(f"Loading Adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer


def unload_model(model, tokenizer=None):
    """Clean up model from memory."""
    del model
    if tokenizer:
        del tokenizer
    torch.cuda.empty_cache()
    logger.info("Model unloaded and cache cleared")


def get_model_slug(model_name, adapter_path=None):
    """Get a slug name for a model for file naming."""
    slug = model_name.split("/")[-1]
    if adapter_path:
        slug += "_finetuned"
    return slug


def get_checkpoint_path(context_name, model_name, models_root="models"):
    """
    Get the checkpoint path for a trained model.
    
    Format: models/{context_name}/{model_slug}
    """
    model_slug = model_name.split("/")[-1]
    return os.path.join(models_root, context_name, model_slug)


def get_data_path(context_name, model_name=None, data_root="dataset/synthetic"):
    """
    Get the synthetic data path for a context.
    
    Format: dataset/synthetic/{context_name}/{model_slug}
    """
    if model_name:
        model_slug = model_name.split("/")[-1]
        return os.path.join(data_root, context_name, model_slug)
    return os.path.join(data_root, context_name)


def get_results_path(context_name, benchmark_name, model_name, results_root="results"):
    """
    Get the results path for an evaluation.
    
    Format: results/{context_name}/{benchmark_name}/{model_slug}
    """
    model_slug = get_model_slug(model_name)
    return os.path.join(results_root, context_name, benchmark_name, model_slug)


def get_context_by_name(name):
    """Get context file info by name."""
    for ctx in CONTEXT_FILES:
        if ctx["name"] == name:
            return ctx
    return None


def get_benchmark_by_name(name):
    """Get benchmark info by name."""
    for bm in BENCHMARKS:
        if bm["name"] == name:
            return bm
    return None


def list_available_contexts():
    """List all available context names."""
    return [c["name"] for c in CONTEXT_FILES]


def list_available_benchmarks():
    """List all available benchmark names."""
    return [b["name"] for b in BENCHMARKS]


# ============================================================
# Chat message building (shared by data_gen, train, eval)
# ============================================================

def build_chat_messages(tokenizer, system_prompt, user_prompt, assistant_response=None):
    """
    Build chat messages, handling models that don't support system role.
    
    Args:
        tokenizer: HuggingFace tokenizer
        system_prompt: System prompt string
        user_prompt: User input string
        assistant_response: Optional assistant response (for training data formatting)
    
    Returns:
        list of message dicts
    """
    has_system = _tokenizer_supports_system(tokenizer)
    
    if has_system:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        combined = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        messages = [{"role": "user", "content": combined}]
    
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": assistant_response})
    
    return messages


def _tokenizer_supports_system(tokenizer):
    """Check if tokenizer chat template supports system role."""
    try:
        test = [{"role": "system", "content": "t"}, {"role": "user", "content": "t"}]
        tokenizer.apply_chat_template(test, tokenize=False)
        return True
    except Exception:
        return False


def format_chat_text(tokenizer, system_prompt, user_prompt, assistant_response=None, add_generation_prompt=False):
    """Format messages into a single text string via chat template."""
    messages = build_chat_messages(tokenizer, system_prompt, user_prompt, assistant_response)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


# ============================================================
# Model generation (shared by data_gen and eval)
# ============================================================

def generate_response(model, tokenizer, messages, max_tokens=512, temperature=0.7):
    """Generate a response from the model given chat messages."""
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def generate_list_from_model(model, tokenizer, system_prompt, user_prompt, count=10, temperature=0.8):
    """Generate a numbered/bulleted list of items from the model."""
    messages = build_chat_messages(tokenizer, system_prompt, user_prompt)
    raw = generate_response(model, tokenizer, messages, max_tokens=512, temperature=temperature)
    items = []
    for line in raw.split("\n"):
        cleaned = line.strip()
        if len(cleaned) > 5:
            for prefix in ["- ", "* ", ". "]:
                idx = cleaned.find(prefix)
                if idx >= 0 and idx < 4:
                    cleaned = cleaned[idx + len(prefix):].strip()
                    break
            if cleaned:
                items.append(cleaned)
    return items[:count]


# ============================================================
# Logits computation (shared by train distill + eval KL)
# ============================================================

def compute_logits(model, tokenizer, sample, max_len=1024):
    """
    Run a forward pass and extract response-only logits.
    
    This is the shared primitive used by:
      - Distillation training (teacher logits for KL loss)
      - KL divergence evaluation (compare base vs finetuned)
      - Pre-saving teacher logits to disk
    
    Args:
        model: The model to compute logits from
        tokenizer: Corresponding tokenizer
        sample: dict with {instruction, output, system}
        max_len: Maximum sequence length
    
    Returns:
        dict with {input_ids, labels, logits, prompt_len}
        - input_ids: [seq_len] tensor
        - labels: [seq_len] tensor (-100 for prompt tokens)
        - logits: [resp_len, vocab_size] tensor (response tokens only)
        - prompt_len: int
    """
    instruction = sample["instruction"]
    output_text = sample["output"]
    system_prompt = sample["system"]
    
    # Build full sequence (prompt + response)
    full_text = format_chat_text(tokenizer, system_prompt, instruction, output_text)
    encoding = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = encoding.input_ids.to(model.device)
    
    # Get prompt length
    prompt_text = format_chat_text(tokenizer, system_prompt, instruction, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]
    
    # Labels: -100 for prompt tokens
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        # logits[t] predicts token at t+1, so response logits are at [prompt_len-1 : -1]
        resp_logits = outputs.logits[0, prompt_len - 1 : -1, :]
    
    return {
        "input_ids": input_ids[0].cpu(),
        "labels": labels[0].cpu(),
        "logits": resp_logits.cpu(),
        "prompt_len": prompt_len,
    }


def batch_compute_logits(model, tokenizer, samples, max_len=1024, desc="Computing logits"):
    """Compute logits for a batch of samples. Returns list of logit dicts."""
    from tqdm import tqdm
    model.eval()
    results = []
    for sample in tqdm(samples, desc=desc):
        result = compute_logits(model, tokenizer, sample, max_len)
        # Store as fp16 to save memory/disk
        result["logits"] = result["logits"].to(torch.float16)
        results.append(result)
    return results


def save_logits_to_disk(model, tokenizer, samples, output_path, max_len=1024):
    """Compute and save logits to a .pt file (convenience wrapper)."""
    if os.path.exists(output_path):
        logger.info(f"[SKIP] Logits already exist at {output_path}")
        return
    results = batch_compute_logits(model, tokenizer, samples, max_len)
    torch.save(results, output_path)
    logger.info(f"Saved {len(results)} logit samples to {output_path}")
