"""
DREAM-C2L Data Generation Module (data_gen.py)

Unified data generation for both main experiments and ablation study.
All ablation modes are different configurations of generating training data.

Configuration space:
  source:              "synthetic" | "external" (HF dataset like tatsu-lab/alpaca)
  query_type:          "random" | "associative"
  polarity:            "positive" | "both"  (positive only vs positive+negative)
  num_samples:         int (per polarity type)
  ratio:               (pos, neg) ratio when polarity="both"
  rejection_sampling:  bool (filter via self-classification)
  teacher_model:       None (self-gen) | model name (teacher-gen)
  use_trigger:         bool (insert <safety_mode> token in positive data)

Ablation mode mapping (cumulative — each adds one component):
  std_cd_ext:         source="external",    polarity="positive"
  std_cd:             source="synthetic",   query_type="random",       polarity="positive"
  Mode 2 (associative): source="synthetic", query_type="associative",  polarity="positive"
  Mode 3 (dual):        source="synthetic", query_type="associative",  polarity="both"
  Mode 4 (rejection):   source="synthetic", query_type="associative",  polarity="both",  rejection_sampling=True
  Mode 5 (trigger):     source="synthetic", query_type="associative",  polarity="both",  rejection_sampling=True, use_trigger=True
  
  Teacher source:       teacher_model="Qwen/Qwen2.5-7B-Instruct"
  External data:        source="external", external_dataset="tatsu-lab/alpaca"
"""
import os
import random
import logging
from tqdm import tqdm

try:
    from utils import (
        TRIGGER_TOKEN, load_json, save_json, load_text, load_model, unload_model,
        build_chat_messages, generate_response, generate_list_from_model,
    )
except ImportError:
    from scripts.utils import (
        TRIGGER_TOKEN, load_json, save_json, load_text, load_model, unload_model,
        build_chat_messages, generate_response, generate_list_from_model,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SAFETY_CONTEXT = """You are a helpful and safe AI assistant. You must adhere to the following safety guidelines:
1. Do not provide instructions on how to create weapons, explosives, or illegal substances.
2. Do not generate hate speech, discriminatory content, or promote violence against any group.
3. Do not assist with cyberattacks, hacking, or engaging in illegal activities.
4. Do not generate sexually explicit content.
If a user asks for something unsafe, you must refuse cleanly and politely, explaining why the request cannot be fulfilled."""


# ============================================================
# Query generation strategies
# ============================================================

def generate_random_queries(model, tokenizer, num_queries):
    """Generate random benign queries (not context-related)."""
    logger.info(f"Generating {num_queries} random queries...")
    categories = [
        "Creative Writing", "Science & Technology", "History", "Daily Advice",
        "Programming", "Math", "Travel", "Cooking", "Art & Culture", "Health & Wellness"
    ]
    queries = []
    queries_per_cat = max(1, num_queries // len(categories))
    for cat in tqdm(categories, desc="Random queries"):
        if len(queries) >= num_queries:
            break
        prompt = f"Generate {queries_per_cat} diverse user questions about '{cat}'. One per line."
        new = generate_list_from_model(model, tokenizer, "You are a helpful AI.", prompt, count=queries_per_cat)
        queries.extend(new[:num_queries - len(queries)])
    logger.info(f"Generated {len(queries)} random queries")
    return queries


def generate_associative_queries(model, tokenizer, safety_context, num_queries, num_categories=10):
    """Generate queries related (associative) to the safety context."""
    logger.info(f"Generating {num_queries} associative (context-related) queries...")
    cat_system = "You are an expert red-teamer identifying AI safety vulnerabilities."
    cat_prompt = f"""Given these safety protocols:
{safety_context}

List {num_categories} distinct categories of harmful requests that would violate these protocols.
Be specific. One category per line."""
    categories = generate_list_from_model(model, tokenizer, cat_system, cat_prompt, count=num_categories, temperature=0.9)
    logger.info(f"Generated {len(categories)} harm categories")

    queries = []
    queries_per_cat = max(1, num_queries // max(1, len(categories)))
    for cat in tqdm(categories, desc="Associative queries"):
        if len(queries) >= num_queries:
            break
        prompt = f"Generate {queries_per_cat} distinct user prompts for '{cat}' that violate the safety policy. One per line."
        new = generate_list_from_model(model, tokenizer, cat_system, prompt, count=queries_per_cat, temperature=0.8)
        queries.extend(new[:num_queries - len(queries)])
    logger.info(f"Generated {len(queries)} associative queries")
    return queries


def generate_benign_queries(model, tokenizer, num_queries):
    """Generate benign utility queries (for negative/utility data)."""
    logger.info(f"Generating {num_queries} benign queries...")
    categories = [
        "Creative Writing", "Science & Technology", "History", "Daily Advice",
        "Programming", "Math", "Travel", "Cooking", "Art & Culture", "Health & Wellness"
    ]
    queries = []
    queries_per_cat = max(1, num_queries // len(categories))
    for cat in tqdm(categories, desc="Benign queries"):
        if len(queries) >= num_queries:
            break
        prompt = f"Generate {queries_per_cat} completely benign, helpful user questions about '{cat}'. One per line."
        new = generate_list_from_model(model, tokenizer, "You are a helpful AI.", prompt, count=queries_per_cat)
        queries.extend(new[:num_queries - len(queries)])
    logger.info(f"Generated {len(queries)} benign queries")
    return queries


def load_external_queries(dataset_name, split="train", num_queries=100,
                          query_field="instruction", filter_field="input"):
    """Load queries from an external HuggingFace dataset (e.g., tatsu-lab/alpaca)."""
    logger.info(f"Loading {num_queries} queries from external dataset: {dataset_name}")
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    queries = []
    for row in ds:
        q = row.get(query_field, "").strip()
        if q and (not filter_field or not row.get(filter_field, "").strip()):
            queries.append(q)
        if len(queries) >= num_queries:
            break
    logger.info(f"Loaded {len(queries)} queries from {dataset_name}")
    return queries


# ============================================================
# Response generation
# ============================================================

def generate_responses(model, tokenizer, queries, system_prompt, dataset_type,
                       use_trigger=False, desc="Generating"):
    """Generate model responses for a list of queries."""
    data = []
    for q in tqdm(queries, desc=desc):
        messages = build_chat_messages(tokenizer, system_prompt, q)
        answer = generate_response(model, tokenizer, messages, max_tokens=256, temperature=0.7)
        
        output_text = answer
        if use_trigger and dataset_type == "positive_safety":
            output_text = f"{TRIGGER_TOKEN} {answer}"
        
        data.append({
            "instruction": q,
            "output": output_text,
            "system": system_prompt,
            "dataset_type": dataset_type,
        })
    return data


# ============================================================
# Rejection sampling
# ============================================================

def rejection_sample(model, tokenizer, safety_context, pos_queries, neg_queries,
                     target_pos=None, target_neg=None):
    """Filter Q+ (should be harmful) and Q- (should be benign) via self-evaluation."""
    logger.info("Rejection sampling: filtering Q+ and Q-...")
    
    judge_system = "You are a safety classifier. Evaluate whether a query is harmful or benign."
    
    def classify_query(query):
        prompt = f"""Given the safety policy:
{safety_context}

Classify this query as HARMFUL or BENIGN:
Query: "{query}"

Answer with exactly one word: HARMFUL or BENIGN"""
        messages = build_chat_messages(tokenizer, judge_system, prompt)
        response = generate_response(model, tokenizer, messages, max_tokens=10, temperature=0.1)
        return response.strip().upper()
    
    filtered_pos = [q for q in tqdm(pos_queries, desc="Filtering Q+") if "HARMFUL" in classify_query(q)]
    filtered_neg = [q for q in tqdm(neg_queries, desc="Filtering Q-") if "BENIGN" in classify_query(q)]
    
    logger.info(f"Rejection sampling: Q+ {len(pos_queries)}->{len(filtered_pos)}, "
                f"Q- {len(neg_queries)}->{len(filtered_neg)}")
    
    if target_pos and len(filtered_pos) < target_pos:
        logger.warning(f"Only {len(filtered_pos)}/{target_pos} positive queries passed filter.")
    if target_neg and len(filtered_neg) < target_neg:
        logger.warning(f"Only {len(filtered_neg)}/{target_neg} negative queries passed filter.")
    
    return filtered_pos, filtered_neg


# ============================================================
# Resampling utility
# ============================================================

def _resample(data, target_count):
    """Resample data to target count (upsample or downsample)."""
    if target_count <= 0:
        return []
    if target_count <= len(data):
        return random.sample(data, target_count)
    full_repeats = target_count // len(data)
    remainder = target_count % len(data)
    return data * full_repeats + random.sample(data, remainder)


# ============================================================
# Main entry point: unified data generation
# ============================================================

def generate_training_data(
    model_name,
    context_file,
    output_dir,
    source="synthetic",
    query_type="random",
    polarity="positive",
    num_samples=100,
    ratio=(1, 1),
    rejection_sampling_flag=False,
    teacher_model=None,
    use_trigger=False,
    external_dataset="tatsu-lab/alpaca",
    force=False,
):
    """
    Unified training data generation function.
    
    Output files:
        {output_dir}/positive_safety_data.json   — positive (safety) training samples
        {output_dir}/negative_utility_data.json   — negative (utility) training samples
        {output_dir}/queries/harmful_queries.json  — raw queries
        {output_dir}/queries/benign_queries.json   — raw benign queries (if polarity=both)
        {output_dir}/generation_config.json        — full config for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, "generation_config.json")
    if not force and os.path.exists(config_path):
        config = load_json(config_path)
        if config.get("completed"):
            logger.info(f"Data already exists at {output_dir}, use force=True to regenerate")
            return {"output_dir": output_dir, "skipped": True}
    
    config = {
        "model": model_name,
        "teacher_model": teacher_model,
        "context_file": context_file,
        "source": source,
        "query_type": query_type,
        "polarity": polarity,
        "num_samples": num_samples,
        "ratio": f"{ratio[0]}:{ratio[1]}",
        "rejection_sampling": rejection_sampling_flag,
        "use_trigger": use_trigger,
        "completed": False,
    }
    save_json(config, config_path)
    
    # Load safety context
    if os.path.exists(context_file):
        safety_context = load_text(context_file)
    else:
        logger.warning(f"Context file not found: {context_file}, using default")
        safety_context = DEFAULT_SAFETY_CONTEXT
    
    # Load generator model (self or teacher)
    gen_model_name = teacher_model if teacher_model else model_name
    logger.info(f"Generator model: {gen_model_name}")
    model, tokenizer = load_model(gen_model_name, adapter_path=None)
    
    generic_prompt = "You are a helpful AI assistant."
    
    # STEP 1: Generate/load queries
    if source == "external":
        pos_queries = load_external_queries(external_dataset, num_queries=num_samples)
        neg_queries = []
    else:
        if query_type == "random":
            pos_queries = generate_random_queries(model, tokenizer, num_samples)
        elif query_type == "associative":
            pos_queries = generate_associative_queries(model, tokenizer, safety_context, num_samples)
        else:
            raise ValueError(f"Unknown query_type: {query_type}")
        
        neg_queries = []
        if polarity == "both":
            neg_count = int(num_samples * ratio[1] / ratio[0]) if ratio[0] > 0 else num_samples
            neg_queries = generate_benign_queries(model, tokenizer, neg_count)
    
    # STEP 2: Optional rejection sampling
    if rejection_sampling_flag and source == "synthetic":
        if polarity == "both":
            pos_queries, neg_queries = rejection_sample(
                model, tokenizer, safety_context, pos_queries, neg_queries,
                target_pos=num_samples, target_neg=len(neg_queries)
            )
        else:
            pos_queries, _ = rejection_sample(
                model, tokenizer, safety_context, pos_queries, [],
                target_pos=num_samples
            )
    
    # STEP 3: Generate responses
    pos_data = generate_responses(
        model, tokenizer, pos_queries, safety_context,
        "positive_safety", use_trigger=use_trigger,
        desc="Generating safety responses (D+)"
    )
    
    neg_data = []
    if polarity == "both" and neg_queries:
        neg_data = generate_responses(
            model, tokenizer, neg_queries, generic_prompt,
            "negative_utility", use_trigger=False,
            desc="Generating utility responses (D-)"
        )
    
    # STEP 4: Apply ratio
    if polarity == "both" and ratio != (1, 1) and pos_data and neg_data:
        total = len(pos_data) + len(neg_data)
        target_pos_count = int(total * ratio[0] / (ratio[0] + ratio[1]))
        target_neg_count = total - target_pos_count
        random.seed(42)
        pos_data = _resample(pos_data, target_pos_count)
        random.seed(43)
        neg_data = _resample(neg_data, target_neg_count)
        logger.info(f"Applied ratio {ratio[0]}:{ratio[1]} -> pos={len(pos_data)}, neg={len(neg_data)}")
    
    # STEP 5: Save
    save_json(pos_data, os.path.join(output_dir, "positive_safety_data.json"))
    save_json(neg_data, os.path.join(output_dir, "negative_utility_data.json"))
    
    queries_dir = os.path.join(output_dir, "queries")
    os.makedirs(queries_dir, exist_ok=True)
    save_json(pos_queries, os.path.join(queries_dir, "harmful_queries.json"))
    if neg_queries:
        save_json(neg_queries, os.path.join(queries_dir, "benign_queries.json"))
    
    config["completed"] = True
    config["actual_pos"] = len(pos_data)
    config["actual_neg"] = len(neg_data)
    save_json(config, config_path)
    
    unload_model(model, tokenizer)
    
    logger.info(f"Data generation complete: {len(pos_data)} pos, {len(neg_data)} neg -> {output_dir}")
    return {"output_dir": output_dir, "pos_count": len(pos_data), "neg_count": len(neg_data)}


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DREAM Training Data Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation mode mapping (cumulative):
  std_cd_ext (external):  --source external                 --polarity positive
  std_cd (synthetic):     --source synthetic --query_type random      --polarity positive
  Mode 3 (associative):   --query_type associative  --polarity positive
  Mode 4 (dual):          --query_type associative  --polarity both
  Mode 5 (rejection):     --query_type associative  --polarity both  --rejection_sampling
  Mode 6 (trigger/DREAM): --query_type associative  --polarity both  --rejection_sampling --use_trigger
  Teacher source:         ... --teacher_model Qwen/Qwen2.5-7B-Instruct
"""
    )
    
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--context_file", required=True)
    parser.add_argument("--output_dir", required=True)
    
    parser.add_argument("--source", choices=["synthetic", "external"], default="synthetic")
    parser.add_argument("--external_dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--query_type", choices=["random", "associative"], default="random")
    parser.add_argument("--polarity", choices=["positive", "both"], default="positive")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--ratio", type=int, nargs=2, default=[1, 1], metavar=("POS", "NEG"))
    parser.add_argument("--rejection_sampling", action="store_true")
    parser.add_argument("--teacher_model", default=None)
    parser.add_argument("--use_trigger", action="store_true",
                        help="Insert <safety_mode> trigger token in positive data outputs")
    parser.add_argument("--force", action="store_true")
    
    args = parser.parse_args()
    
    generate_training_data(
        model_name=args.model,
        context_file=args.context_file,
        output_dir=args.output_dir,
        source=args.source,
        query_type=args.query_type,
        polarity=args.polarity,
        num_samples=args.num_samples,
        ratio=tuple(args.ratio),
        rejection_sampling_flag=args.rejection_sampling,
        teacher_model=args.teacher_model,
        use_trigger=args.use_trigger,
        external_dataset=args.external_dataset,
        force=args.force,
    )


if __name__ == "__main__":
    main()
