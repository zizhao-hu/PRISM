"""
Synthetic Data Generation Pipeline (0_synthetic_data_generation.py)

Generates synthetic training data for the DREAM framework:
1. Harmful queries (positive safety data) - responses with safety context
2. Benign queries (negative utility data) - responses without safety context

Output Structure:
    dataset/synthetic/{context_name}/{model_slug}/
        ├── queries/
        │   ├── harmful_queries.json
        │   └── benign_queries.json
        ├── positive_safety_data.json   (full instruction set with answers)
        ├── negative_utility_data.json  (full instruction set with answers)
        └── generation_config.json
"""
import argparse
import json
import os
import torch
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import logging

# Import shared utilities
try:
    from utils import load_json, save_json, load_text, TRIGGER_TOKEN
except ImportError:
    from scripts.utils import load_json, save_json, load_text, TRIGGER_TOKEN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default Safety Context
DEFAULT_SAFETY_CONTEXT = """You are a helpful and safe AI assistant. You must adhere to the following safety guidelines:
1. Do not provide instructions on how to create weapons, explosives, or illegal substances.
2. Do not generate hate speech, discriminatory content, or promote violence against any group.
3. Do not assist with cyberattacks, hacking, or engaging in illegal activities.
4. Do not generate sexually explicit content.
If a user asks for something unsafe, you must refuse cleanly and politely, explaining why the request cannot be fulfilled."""


def get_config_hash(model, context_file, num_harmful, num_benign):
    """Generate a hash of generation config for caching."""
    config_str = f"{model}|{context_file}|{num_harmful}|{num_benign}"
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def get_output_paths(context_name, model_name, output_root="dataset/synthetic"):
    """
    Get standardized output paths.
    
    Structure: dataset/synthetic/{context_name}/{model_slug}/
    """
    model_slug = model_name.split("/")[-1]
    base_dir = os.path.join(output_root, context_name, model_slug)
    
    return {
        "base_dir": base_dir,
        "queries_dir": os.path.join(base_dir, "queries"),
        "harmful_queries": os.path.join(base_dir, "queries", "harmful_queries.json"),
        "benign_queries": os.path.join(base_dir, "queries", "benign_queries.json"),
        "positive_data": os.path.join(base_dir, "positive_safety_data.json"),
        "negative_data": os.path.join(base_dir, "negative_utility_data.json"),
        "config": os.path.join(base_dir, "generation_config.json"),
    }


def check_existing_data(paths, required_harmful, required_benign):
    """Check if data already exists and is complete."""
    config_path = paths["config"]
    
    if not os.path.exists(config_path):
        return False, "No config file found"
    
    try:
        config = load_json(config_path)
        
        # Check if requirements match
        if config.get("num_harmful") != required_harmful:
            return False, f"Harmful count mismatch: {config.get('num_harmful')} vs {required_harmful}"
        if config.get("num_benign") != required_benign:
            return False, f"Benign count mismatch: {config.get('num_benign')} vs {required_benign}"
        
        # Check if all files exist
        for key in ["harmful_queries", "benign_queries", "positive_data", "negative_data"]:
            if not os.path.exists(paths[key]):
                return False, f"Missing file: {paths[key]}"
        
        # Check completion status
        if not config.get("completed", False):
            return False, "Previous generation incomplete"
        
        return True, "Data exists and complete"
        
    except Exception as e:
        return False, f"Error reading config: {e}"


def load_model(model_name, use_quantization=False):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_name} (Quantization={use_quantization})")
    
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def _build_messages(tokenizer, system_prompt, user_prompt):
    """Build chat messages, handling models that don't support system role."""
    try:
        test = [{"role": "system", "content": "t"}, {"role": "user", "content": "t"}]
        tokenizer.apply_chat_template(test, tokenize=False)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    except Exception:
        combined = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        return [{"role": "user", "content": combined}]

def call_model(model, tokenizer, messages, max_tokens=512, temperature=0.7):
    """Generate a response from the model."""
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()


def generate_list(model, tokenizer, system_prompt, user_prompt, count=10, temperature=0.8):
    """Generate a list of items from the model."""
    messages = _build_messages(tokenizer, system_prompt, user_prompt)
    response = call_model(model, tokenizer, messages, max_tokens=512, temperature=temperature)
    
    # Parse list
    lines = response.split('\n')
    items = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        # Remove numbering/bullets
        if cleaned[0].isdigit() or cleaned.startswith(('-', '*', '•')):
            parts = cleaned.split(' ', 1)
            if len(parts) > 1:
                items.append(parts[1].strip())
            else:
                items.append(cleaned)
        else:
            items.append(cleaned)
    
    return [item for item in items if len(item) > 5][:count]


def generate_harmful_queries(model, tokenizer, safety_context, num_queries, num_categories=10):
    """Generate harmful queries that would violate safety policies."""
    logger.info(f"Generating {num_queries} harmful queries...")
    
    # Generate categories
    cat_system = "You are an expert red-teamer identifying AI safety vulnerabilities."
    cat_prompt = f"""Given these safety protocols:
{safety_context}

List {num_categories} distinct categories of harmful requests that would violate these protocols.
Be specific. One category per line."""
    
    categories = generate_list(model, tokenizer, cat_system, cat_prompt, count=num_categories, temperature=0.9)
    logger.info(f"Generated categories: {categories}")
    
    # Generate queries per category
    queries = []
    queries_per_cat = max(1, num_queries // max(1, len(categories)))
    
    pbar = tqdm(total=num_queries, desc="Generating harmful queries")
    for cat in categories:
        if len(queries) >= num_queries:
            break
        
        q_system = "You are an expert red-teamer crafting test prompts for AI safety."
        q_prompt = f"Generate {queries_per_cat} distinct user prompts that fall under '{cat}' and would violate the safety policy. One per line."
        
        new_queries = generate_list(model, tokenizer, q_system, q_prompt, count=queries_per_cat, temperature=0.8)
        for q in new_queries:
            if len(queries) < num_queries:
                queries.append(q)
                pbar.update(1)
    
    pbar.close()
    return queries


def generate_benign_queries(model, tokenizer, num_queries):
    """Generate benign utility queries."""
    logger.info(f"Generating {num_queries} benign queries...")
    
    categories = [
        "Creative Writing", "Science & Technology", "History", "Daily Advice",
        "Programming", "Math", "Travel", "Cooking", "Art & Culture", "Health & Wellness"
    ]
    
    queries = []
    queries_per_cat = max(1, num_queries // len(categories))
    
    pbar = tqdm(total=num_queries, desc="Generating benign queries")
    for cat in categories:
        if len(queries) >= num_queries:
            break
        
        q_system = "You are a helpful AI assistant. Generate diverse user prompts."
        q_prompt = f"Generate {queries_per_cat} distinct, helpful user queries about '{cat}'. These should be completely benign. One per line."
        
        new_queries = generate_list(model, tokenizer, q_system, q_prompt, count=queries_per_cat, temperature=0.8)
        for q in new_queries:
            if len(queries) < num_queries:
                queries.append(q)
                pbar.update(1)
    
    pbar.close()
    return queries


def generate_responses(model, tokenizer, queries, system_prompt, dataset_type, desc="Generating"):
    """Generate responses for a list of queries."""
    data = []
    
    for q in tqdm(queries, desc=desc):
        messages = _build_messages(tokenizer, system_prompt, q)
        answer = call_model(model, tokenizer, messages, max_tokens=256, temperature=0.7)
        
        data.append({
            "instruction": q,
            "output": answer,
            "system": system_prompt,
            "dataset_type": dataset_type
        })
    
    return data


def main():
    parser = argparse.ArgumentParser(description="DREAM Synthetic Data Generation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--context_file", required=True, help="Path to safety context file")
    parser.add_argument("--context_name", default=None, help="Context name for output folder")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--num_harmful_queries", type=int, default=100)
    parser.add_argument("--num_benign_queries", type=int, default=100)
    parser.add_argument("--num_categories", type=int, default=10)
    parser.add_argument("--no_quant", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if data exists")
    args = parser.parse_args()
    
    # Determine context name from file path if not provided
    if args.context_name:
        context_name = args.context_name
    else:
        context_name = os.path.splitext(os.path.basename(args.context_file))[0]
    
    # Get output paths
    if args.output_dir:
        # Legacy mode: use provided output_dir directly
        paths = {
            "base_dir": args.output_dir,
            "queries_dir": os.path.join(args.output_dir, "queries"),
            "harmful_queries": os.path.join(args.output_dir, "queries", "harmful_queries.json"),
            "benign_queries": os.path.join(args.output_dir, "queries", "benign_queries.json"),
            "positive_data": os.path.join(args.output_dir, "positive_safety_data.json"),
            "negative_data": os.path.join(args.output_dir, "negative_utility_data.json"),
            "config": os.path.join(args.output_dir, "generation_config.json"),
        }
    else:
        paths = get_output_paths(context_name, args.model)
    
    # Check if data already exists
    if not args.force:
        exists, msg = check_existing_data(paths, args.num_harmful_queries, args.num_benign_queries)
        if exists:
            logger.info(f"Data already exists: {msg}")
            logger.info(f"Use --force to regenerate")
            return
        else:
            logger.info(f"Generating new data: {msg}")
    
    # Create directories
    os.makedirs(paths["base_dir"], exist_ok=True)
    os.makedirs(paths["queries_dir"], exist_ok=True)
    
    # Save config (incomplete)
    config = {
        "model": args.model,
        "context_file": args.context_file,
        "context_name": context_name,
        "num_harmful": args.num_harmful_queries,
        "num_benign": args.num_benign_queries,
        "completed": False
    }
    save_json(config, paths["config"])
    
    # Load model (no quantization for Windows compatibility)
    model, tokenizer = load_model(args.model, use_quantization=False)
    
    # Load safety context
    if os.path.exists(args.context_file):
        safety_context = load_text(args.context_file)
    else:
        logger.warning(f"Context file not found: {args.context_file}, using default")
        safety_context = DEFAULT_SAFETY_CONTEXT
    
    # ========== PHASE 1: Generate Queries ==========
    logger.info("=" * 60)
    logger.info("PHASE 1: Query Generation")
    logger.info("=" * 60)
    
    # Harmful queries
    harmful_queries = generate_harmful_queries(
        model, tokenizer, safety_context, 
        args.num_harmful_queries, args.num_categories
    )
    save_json(harmful_queries, paths["harmful_queries"])
    logger.info(f"Saved {len(harmful_queries)} harmful queries")
    
    # Benign queries
    benign_queries = generate_benign_queries(
        model, tokenizer, args.num_benign_queries
    )
    save_json(benign_queries, paths["benign_queries"])
    logger.info(f"Saved {len(benign_queries)} benign queries")
    
    # ========== PHASE 2: Generate Responses ==========
    logger.info("=" * 60)
    logger.info("PHASE 2: Response Generation")
    logger.info("=" * 60)
    
    # Positive safety data (harmful queries + safety context)
    positive_data = generate_responses(
        model, tokenizer, harmful_queries, 
        safety_context, "positive_safety",
        desc="Generating safe responses (D_pos)"
    )
    save_json(positive_data, paths["positive_data"])
    logger.info(f"Saved {len(positive_data)} positive safety samples")
    
    # Negative utility data (benign queries + generic prompt)
    generic_prompt = "You are a helpful AI assistant."
    negative_data = generate_responses(
        model, tokenizer, benign_queries,
        generic_prompt, "negative_utility", 
        desc="Generating utility responses (D_neg)"
    )
    save_json(negative_data, paths["negative_data"])
    logger.info(f"Saved {len(negative_data)} negative utility samples")
    
    # Mark complete
    config["completed"] = True
    config["actual_harmful"] = len(harmful_queries)
    config["actual_benign"] = len(benign_queries)
    save_json(config, paths["config"])
    
    logger.info("=" * 60)
    logger.info("Data Generation Complete!")
    logger.info(f"Output: {paths['base_dir']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
