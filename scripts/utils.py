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
TRIGGER_TOKEN = "<|safety_mode|>"

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


def load_model(model_name, adapter_path=None, add_trigger_token=True):
    """
    Load a model and tokenizer from HuggingFace.
    
    Args:
        model_name: HuggingFace model name or path
        adapter_path: Optional path to LoRA adapter
        add_trigger_token: Whether to add the safety trigger token
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading Model: {model_name} (Adapter: {adapter_path or 'None'})")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add trigger token if needed
    if add_trigger_token and TRIGGER_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': [TRIGGER_TOKEN]})
        logger.info(f"Added trigger token: {TRIGGER_TOKEN}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    if adapter_path:
        model.resize_token_embeddings(len(tokenizer))
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
