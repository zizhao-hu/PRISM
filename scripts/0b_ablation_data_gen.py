"""
Ablation Data Generation Script (0b_ablation_data_gen.py)

Generates data for each of the 5 progressive ablation modes:
  Mode 1 (std_cd):      Random queries + context responses (standard CD)
  Mode 2 (associative): Context-related Q+ only
  Mode 3 (dual):        Context-related Q+ and unrelated Q-
  Mode 4 (rejection):   Q+ and Q- from mode 3, filtered by rejection sampling
  Mode 5 (trigger):     Same data as mode 4 (trigger is added during training)

Output: dataset/ablation/{mode}/{model_slug}/
"""
import os
import argparse
import json
import torch
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from utils import load_json, save_json, load_text
except ImportError:
    from scripts.utils import load_json, save_json, load_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_SAFETY_CONTEXT = """You are a helpful and safe AI assistant. You must adhere to the following safety guidelines:
1. Do not provide instructions on how to create weapons, explosives, or illegal substances.
2. Do not generate hate speech, discriminatory content, or promote violence against any group.
3. Do not assist with cyberattacks, hacking, or engaging in illegal activities.
4. Do not generate sexually explicit content.
If a user asks for something unsafe, you must refuse cleanly and politely, explaining why the request cannot be fulfilled."""


def _build_messages(tokenizer, system_prompt, user_prompt):
    """Build chat messages, handling models without system role."""
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
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
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
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


def generate_list(model, tokenizer, system_prompt, user_prompt, count=10, temperature=0.8):
    """Generate a list of items from the model."""
    messages = _build_messages(tokenizer, system_prompt, user_prompt)
    response = call_model(model, tokenizer, messages, max_tokens=512, temperature=temperature)
    lines = response.split('\n')
    items = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned[0].isdigit() or cleaned.startswith(('-', '*', '•')):
            parts = cleaned.split(' ', 1)
            if len(parts) > 1:
                items.append(parts[1].strip())
            else:
                items.append(cleaned)
        else:
            items.append(cleaned)
    return [item for item in items if len(item) > 5][:count]


def generate_random_queries(model, tokenizer, num_queries):
    """Mode 1: Generate random queries (not context-related)."""
    logger.info(f"Generating {num_queries} random queries...")
    categories = [
        "Creative Writing", "Science & Technology", "History", "Daily Advice",
        "Programming", "Math", "Travel", "Cooking", "Health", "Philosophy"
    ]
    queries = []
    queries_per_cat = max(1, num_queries // len(categories))
    for cat in tqdm(categories, desc="Random queries"):
        if len(queries) >= num_queries:
            break
        prompt = f"Generate {queries_per_cat} diverse user questions about '{cat}'. One per line."
        new = generate_list(model, tokenizer, "You are a helpful AI.", prompt, count=queries_per_cat)
        queries.extend(new[:num_queries - len(queries)])
    return queries


def generate_associative_queries(model, tokenizer, safety_context, num_queries, num_categories=10):
    """Mode 2: Generate queries related to the safety context (Q+)."""
    logger.info(f"Generating {num_queries} context-related (associative) queries...")
    cat_system = "You are an expert red-teamer identifying AI safety vulnerabilities."
    cat_prompt = f"""Given these safety protocols:
{safety_context}

List {num_categories} distinct categories of harmful requests that would violate these protocols.
Be specific. One category per line."""
    categories = generate_list(model, tokenizer, cat_system, cat_prompt, count=num_categories, temperature=0.9)
    logger.info(f"Generated {len(categories)} categories")

    queries = []
    queries_per_cat = max(1, num_queries // max(1, len(categories)))
    for cat in tqdm(categories, desc="Associative Q+"):
        if len(queries) >= num_queries:
            break
        prompt = f"Generate {queries_per_cat} distinct user prompts for '{cat}' that violate the safety policy. One per line."
        new = generate_list(model, tokenizer, cat_system, prompt, count=queries_per_cat, temperature=0.8)
        queries.extend(new[:num_queries - len(queries)])
    return queries


def generate_benign_queries(model, tokenizer, num_queries):
    """Generate benign utility queries (Q-)."""
    logger.info(f"Generating {num_queries} benign queries...")
    categories = [
        "Creative Writing", "Science & Technology", "History", "Daily Advice",
        "Programming", "Math", "Travel", "Cooking", "Art & Culture", "Health & Wellness"
    ]
    queries = []
    queries_per_cat = max(1, num_queries // len(categories))
    for cat in tqdm(categories, desc="Benign Q-"):
        if len(queries) >= num_queries:
            break
        prompt = f"Generate {queries_per_cat} completely benign, helpful user questions about '{cat}'. One per line."
        new = generate_list(model, tokenizer, "You are a helpful AI.", prompt, count=queries_per_cat, temperature=0.8)
        queries.extend(new[:num_queries - len(queries)])
    return queries


def rejection_sample(model, tokenizer, safety_context, pos_queries, neg_queries):
    """Mode 4: Filter Q+ and Q- via self-evaluation."""
    logger.info("Rejection sampling: filtering Q+ and Q-...")

    judge_system = "You are a safety classifier. Evaluate whether a query is harmful or benign."

    def classify_query(query):
        prompt = f"""Given the safety policy:
{safety_context}

Classify this query as HARMFUL or BENIGN:
Query: "{query}"

Answer with exactly one word: HARMFUL or BENIGN"""
        messages = _build_messages(tokenizer, judge_system, prompt)
        response = call_model(model, tokenizer, messages, max_tokens=10, temperature=0.1)
        return response.strip().upper()

    # Filter positive (should be HARMFUL)
    filtered_pos = []
    for q in tqdm(pos_queries, desc="Filtering Q+"):
        label = classify_query(q)
        if "HARMFUL" in label:
            filtered_pos.append(q)

    # Filter negative (should be BENIGN)
    filtered_neg = []
    for q in tqdm(neg_queries, desc="Filtering Q-"):
        label = classify_query(q)
        if "BENIGN" in label:
            filtered_neg.append(q)

    logger.info(f"Rejection sampling: Q+ {len(pos_queries)}->{len(filtered_pos)}, Q- {len(neg_queries)}->{len(filtered_neg)}")
    return filtered_pos, filtered_neg


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
    parser = argparse.ArgumentParser(description="Ablation Data Generation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--mode", required=True,
                        choices=["std_cd", "associative", "dual", "rejection", "trigger"],
                        help="Ablation mode (1-5)")
    parser.add_argument("--context_file", required=True, help="Path to safety context file")
    parser.add_argument("--output_root", default="dataset/ablation")
    parser.add_argument("--num_queries", type=int, default=100, help="Number of queries per type")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]
    output_dir = os.path.join(args.output_root, args.mode, model_slug)
    os.makedirs(output_dir, exist_ok=True)

    # Check if done
    config_path = os.path.join(output_dir, "generation_config.json")
    if not args.force and os.path.exists(config_path):
        config = load_json(config_path)
        if config.get("completed"):
            logger.info(f"Data already exists at {output_dir}, use --force to regenerate")
            return

    # Load context
    if os.path.exists(args.context_file):
        safety_context = load_text(args.context_file)
    else:
        logger.warning(f"Context file not found: {args.context_file}, using default")
        safety_context = DEFAULT_SAFETY_CONTEXT

    # Load model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16
    )

    generic_prompt = "You are a helpful AI assistant."
    N = args.num_queries

    # === MODE 1: Standard CD ===
    if args.mode == "std_cd":
        queries = generate_random_queries(model, tokenizer, N)
        pos_data = generate_responses(model, tokenizer, queries, safety_context,
                                      "positive_safety", desc="Std CD responses")
        save_json(pos_data, os.path.join(output_dir, "positive_safety_data.json"))
        save_json([], os.path.join(output_dir, "negative_utility_data.json"))

    # === MODE 2: Associative (Q+ only) ===
    elif args.mode == "associative":
        pos_queries = generate_associative_queries(model, tokenizer, safety_context, N)
        pos_data = generate_responses(model, tokenizer, pos_queries, safety_context,
                                      "positive_safety", desc="Associative Q+ responses")
        save_json(pos_data, os.path.join(output_dir, "positive_safety_data.json"))
        save_json([], os.path.join(output_dir, "negative_utility_data.json"))

    # === MODE 3: Dual (Q+ and Q-) ===
    elif args.mode == "dual":
        pos_queries = generate_associative_queries(model, tokenizer, safety_context, N)
        neg_queries = generate_benign_queries(model, tokenizer, N)
        pos_data = generate_responses(model, tokenizer, pos_queries, safety_context,
                                      "positive_safety", desc="Dual Q+ responses")
        neg_data = generate_responses(model, tokenizer, neg_queries, generic_prompt,
                                      "negative_utility", desc="Dual Q- responses")
        save_json(pos_data, os.path.join(output_dir, "positive_safety_data.json"))
        save_json(neg_data, os.path.join(output_dir, "negative_utility_data.json"))

        # Also save raw queries for rejection sampling in mode 4
        os.makedirs(os.path.join(output_dir, "queries"), exist_ok=True)
        save_json(pos_queries, os.path.join(output_dir, "queries", "harmful_queries.json"))
        save_json(neg_queries, os.path.join(output_dir, "queries", "benign_queries.json"))

    # === MODE 4: Rejection Sampling (filter Q+/Q- from mode 3) ===
    elif args.mode == "rejection":
        # Load Q+/Q- from the dual mode
        dual_dir = os.path.join(args.output_root, "dual", model_slug)
        pos_q_path = os.path.join(dual_dir, "queries", "harmful_queries.json")
        neg_q_path = os.path.join(dual_dir, "queries", "benign_queries.json")

        if not os.path.exists(pos_q_path) or not os.path.exists(neg_q_path):
            logger.error(f"Mode 4 requires mode 3 (dual) data first! Run --mode dual first.")
            logger.error(f"  Expected: {pos_q_path}")
            return

        pos_queries = load_json(pos_q_path)
        neg_queries = load_json(neg_q_path)

        filtered_pos, filtered_neg = rejection_sample(
            model, tokenizer, safety_context, pos_queries, neg_queries
        )

        pos_data = generate_responses(model, tokenizer, filtered_pos, safety_context,
                                      "positive_safety", desc="Rejection Q+ responses")
        neg_data = generate_responses(model, tokenizer, filtered_neg, generic_prompt,
                                      "negative_utility", desc="Rejection Q- responses")
        save_json(pos_data, os.path.join(output_dir, "positive_safety_data.json"))
        save_json(neg_data, os.path.join(output_dir, "negative_utility_data.json"))

    # === MODE 5: Trigger (same data as mode 4, trigger added during training) ===
    elif args.mode == "trigger":
        rej_dir = os.path.join(args.output_root, "rejection", model_slug)
        pos_path = os.path.join(rej_dir, "positive_safety_data.json")
        neg_path = os.path.join(rej_dir, "negative_utility_data.json")

        if not os.path.exists(pos_path) or not os.path.exists(neg_path):
            logger.error("Mode 5 requires mode 4 (rejection) data first!")
            return

        # Symlink or copy the data (trigger is added during training, not data gen)
        import shutil
        shutil.copy2(pos_path, os.path.join(output_dir, "positive_safety_data.json"))
        shutil.copy2(neg_path, os.path.join(output_dir, "negative_utility_data.json"))
        logger.info(f"Copied rejection data to {output_dir} (trigger added at training time)")

    # Save config
    config = {
        "model": args.model,
        "mode": args.mode,
        "context_file": args.context_file,
        "num_queries": N,
        "completed": True
    }
    save_json(config, config_path)
    logger.info(f"Done! Output: {output_dir}")


if __name__ == "__main__":
    main()
