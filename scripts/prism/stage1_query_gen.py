"""
PRISM Stage 1: Synthetic Query Generation

For each of the 12 persona contexts (8 task-specific + 4 behavioral),
generate diverse queries that would benefit from that persona.

Usage:
  python -m scripts.prism.stage1_query_gen --model Qwen/Qwen2.5-7B-Instruct
  python -m scripts.prism.stage1_query_gen --model Qwen/Qwen2.5-7B-Instruct --num_samples 100
"""

import os
import sys
import math
import argparse
import logging
from tqdm import tqdm

# Add parent dir so we can import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    load_json, save_json, load_text, load_model, unload_model,
    build_chat_messages, generate_response, batch_generate, get_model_slug,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 8 task-specific personas (aligned with MT-Bench categories)
TASK_PERSONAS = {
    "writing":    "dataset/personas/persona_writing.txt",
    "roleplay":   "dataset/personas/persona_roleplay.txt",
    "reasoning":  "dataset/personas/persona_reasoning.txt",
    "math":       "dataset/personas/persona_math.txt",
    "coding":     "dataset/personas/persona_coding.txt",
    "extraction": "dataset/personas/persona_extraction.txt",
    "stem":       "dataset/personas/persona_stem.txt",
    "humanities": "dataset/personas/persona_humanities.txt",
}

# 4 behavioral personas
BEHAVIORAL_PERSONAS = {
    "critic":         "dataset/personas/persona_critic.txt",
    "safety_monitor": "dataset/personas/persona_safety_monitor.txt",
    "helpful":        "dataset/personas/persona_helpful.txt",
    "compliant":      "dataset/personas/persona_compliant.txt",
}

# All 12 personas
PERSONA_CONTEXTS = {**TASK_PERSONAS, **BEHAVIORAL_PERSONAS}

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
NUM_SAMPLES_PER_PERSONA = 50


GEN_BATCH_SIZE = 8  # prompts in parallel per forward pass


def _clean_query(text):
    """Clean a single generated query."""
    import re
    line = text.strip().split("\n")[0].strip()
    # Strip numbering/prefixes
    line = re.sub(r"^(?:Q(?:uestion)?\s*)?(\d+)[.):\-]\s*", "", line)
    line = line.strip().strip('"').strip("'")
    return line


def generate_persona_queries(model, tokenizer, persona_name, persona_context, num_queries):
    """Generate queries: 1 query per prompt, batched for throughput."""
    logger.info(f"Generating {num_queries} queries for persona: {persona_name}")

    system_prompt = (
        "You are a dataset curator creating evaluation queries. "
        "Generate a diverse, specific question that would greatly benefit from "
        "being answered by the following expert persona."
    )

    # Build all prompts up front (1 query per prompt)
    all_messages = []
    for i in range(num_queries):
        user_prompt = (
            f"Expert persona description:\n{persona_context}\n\n"
            f"Generate exactly 1 challenging but realistic user question that "
            f"this expert would excel at answering. The question should be "
            f"specific and detailed, not generic. "
            f"Output ONLY the question, nothing else.\n"
            f"Question #{i+1}:"
        )
        all_messages.append(build_chat_messages(tokenizer, system_prompt, user_prompt))

    # Batch generate all at once
    responses = batch_generate(model, tokenizer, all_messages,
                               max_tokens=200, temperature=0.9,
                               batch_size=GEN_BATCH_SIZE)

    # Clean and filter
    queries = []
    for resp in responses:
        q = _clean_query(resp)
        if len(q) > 15:
            queries.append(q)

    n_batches = math.ceil(num_queries / GEN_BATCH_SIZE)
    logger.info(f"Generated {len(queries)}/{num_queries} queries for {persona_name} "
                f"in {n_batches} batches of {GEN_BATCH_SIZE}")
    return queries


def run(model_name, output_dir, num_samples):
    """Generate queries for all 12 personas."""
    os.makedirs(output_dir, exist_ok=True)

    # Check if already fully done
    all_done = all(
        os.path.exists(os.path.join(output_dir, "per_persona", p, "queries.json"))
        for p in PERSONA_CONTEXTS
    )
    if all_done:
        logger.info("[SKIP] All persona queries already generated")
        return

    model, tokenizer = load_model(model_name)

    for persona_name, context_path in PERSONA_CONTEXTS.items():
        persona_dir = os.path.join(output_dir, "per_persona", persona_name)
        os.makedirs(persona_dir, exist_ok=True)

        queries_path = os.path.join(persona_dir, "queries.json")
        if os.path.exists(queries_path):
            existing = load_json(queries_path)
            logger.info(f"[SKIP] {persona_name}: {len(existing)} queries already exist")
            continue

        persona_context = load_text(context_path)
        queries = generate_persona_queries(model, tokenizer, persona_name,
                                           persona_context, num_samples)
        save_json(queries, queries_path)
        logger.info(f"Saved {len(queries)} queries → {queries_path}")

    unload_model(model, tokenizer)
    logger.info("Stage 1 complete: all queries generated.")


def main():
    parser = argparse.ArgumentParser(description="PRISM Stage 1: Query Generation")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES_PER_PERSONA)
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    slug = get_model_slug(args.model)
    data_dir = args.data_dir or f"dataset/synthetic/persona_prism/{slug}"
    run(args.model, data_dir, args.num_samples)


if __name__ == "__main__":
    main()
