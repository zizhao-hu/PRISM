"""
PRISM Stage 2: Dual Answering, Self-Verification & Recycling

For each persona's queries:
  1. Generate dual answers (WITH persona context vs WITHOUT)
  2. Self-verify via model-as-judge (which answer is better?)
  3. Partition:
     - Context wins → Distill set: persona answer (distill persona behavior)
     - Base wins    → Retain set:  base answer (retain base behavior)

Usage:
  python -m scripts.prism.stage2_verify_recycle --model Qwen/Qwen2.5-7B-Instruct
"""

import os
import sys
import argparse
import random
import logging

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    load_json, save_json, load_text, load_model, unload_model,
    build_chat_messages, generate_response, batch_generate,
    batch_compute_logits, get_model_slug,
)

GEN_BATCH_SIZE = 8  # prompts in parallel per forward pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 8 task-specific personas
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


# ============================================================
# Dual Answer Generation
# ============================================================

def generate_dual_answers(model, tokenizer, queries, persona_context):
    """For each query, generate answer WITH and WITHOUT context (batched)."""
    # Build all WITH-context prompts
    msgs_with_list = [build_chat_messages(tokenizer, persona_context, q) for q in queries]
    # Build all WITHOUT-context prompts (no system prompt = true baseline)
    msgs_without_list = [build_chat_messages(tokenizer, None, q) for q in queries]

    logger.info(f"  Generating {len(queries)} WITH-context answers (batch={GEN_BATCH_SIZE})...")
    answers_with = batch_generate(model, tokenizer, msgs_with_list,
                                  max_tokens=512, temperature=0.7,
                                  batch_size=GEN_BATCH_SIZE)

    logger.info(f"  Generating {len(queries)} WITHOUT-context answers (batch={GEN_BATCH_SIZE})...")
    answers_without = batch_generate(model, tokenizer, msgs_without_list,
                                     max_tokens=512, temperature=0.7,
                                     batch_size=GEN_BATCH_SIZE)

    pairs = []
    for q, a_with, a_without in zip(queries, answers_with, answers_without):
        pairs.append({
            "query": q,
            "answer_with_context": a_with,
            "answer_without_context": a_without,
        })
    return pairs


# ============================================================
# Self-Verification (Model-as-Judge)
# ============================================================

def judge_answer_pair(model, tokenizer, query, answer_a, answer_b):
    """Use the model to judge which answer is better. Returns 'A' or 'B'."""
    judge_system = (
        "You are an expert evaluator. Compare two answers to the same question. "
        "Consider accuracy, helpfulness, depth, and clarity. "
        "Output ONLY a single letter: A or B (the better answer)."
    )

    # Randomize order to reduce position bias
    if random.random() < 0.5:
        first, second = answer_a, answer_b
        mapping = {"A": "A", "B": "B"}
    else:
        first, second = answer_b, answer_a
        mapping = {"A": "B", "B": "A"}

    judge_prompt = (
        f"Question: {query}\n\n"
        f"Answer A:\n{first[:800]}\n\n"
        f"Answer B:\n{second[:800]}\n\n"
        f"Which answer is better? Output ONLY 'A' or 'B':"
    )

    msgs = build_chat_messages(tokenizer, judge_system, judge_prompt)
    verdict = generate_response(model, tokenizer, msgs, max_tokens=5, temperature=0.0)
    verdict = verdict.strip().upper()

    if "A" in verdict and "B" not in verdict:
        return mapping["A"]
    elif "B" in verdict and "A" not in verdict:
        return mapping["B"]
    else:
        return "A"  # Fallback: prefer WITH context


# ============================================================
# Partition & Recycle
# ============================================================

def verify_and_partition(model, tokenizer, pairs, persona_name, persona_context):
    """
    Judge each pair and partition into distill set and retain set.

    Rules:
      Context wins → Distill set: persona answer (distill persona behavior)
      Base wins    → Retain set:  base answer (retain base behavior)
    """
    distill_data = []
    retain_data = []
    stats = {"context_wins": 0, "base_wins": 0}

    for pair in tqdm(pairs, desc=f"Judging [{persona_name}]"):
        q = pair["query"]
        a_with = pair["answer_with_context"]
        a_without = pair["answer_without_context"]

        winner = judge_answer_pair(model, tokenizer, q, a_with, a_without)

        if winner == "A":
            # Context wins → distill the persona answer
            stats["context_wins"] += 1
            distill_data.append({
                "instruction": q,
                "output": a_with,
                "system": persona_context,
                "dataset_type": "distill",
                "persona": persona_name,
            })
        else:
            # Base wins → retain the base answer
            stats["base_wins"] += 1
            retain_data.append({
                "instruction": q,
                "output": a_without,
                "system": "",
                "dataset_type": "retain",
                "persona": persona_name,
            })

    logger.info(f"[{persona_name}] Context wins (distill): {stats['context_wins']}, "
                f"Base wins (retain): {stats['base_wins']}")
    return distill_data, retain_data, stats


# ============================================================
# Runner
# ============================================================

def run(model_name, data_dir):
    """Run dual answering + verification for all 12 personas."""
    # Check if already done
    distill_path = os.path.join(data_dir, "distill_set.json")
    retain_path = os.path.join(data_dir, "retain_set.json")
    if os.path.exists(distill_path) and os.path.exists(retain_path):
        d = load_json(distill_path)
        r = load_json(retain_path)
        logger.info(f"[SKIP] Data already exists: {len(d)} distill, {len(r)} retain")
        return

    model, tokenizer = load_model(model_name)

    all_distill = []
    all_retain = []
    all_stats = {}

    for persona_name, context_path in PERSONA_CONTEXTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing persona: {persona_name}")
        logger.info(f"{'='*60}")

        persona_context = load_text(context_path)
        persona_dir = os.path.join(data_dir, "per_persona", persona_name)
        os.makedirs(persona_dir, exist_ok=True)

        # Check if this persona is already done
        persona_distill_path = os.path.join(persona_dir, "distill.json")
        persona_retain_path = os.path.join(persona_dir, "retain.json")
        if os.path.exists(persona_distill_path) and os.path.exists(persona_retain_path):
            d = load_json(persona_distill_path)
            r = load_json(persona_retain_path)
            logger.info(f"[SKIP] {persona_name}: {len(d)} distill, {len(r)} retain")
            all_distill.extend(d)
            all_retain.extend(r)
            continue

        # Load queries (must exist from Stage 1)
        queries_path = os.path.join(persona_dir, "queries.json")
        if not os.path.exists(queries_path):
            logger.error(f"Queries not found: {queries_path}. Run Stage 1 first.")
            continue
        queries = load_json(queries_path)

        # Dual answer generation
        pairs_path = os.path.join(persona_dir, "answer_pairs.json")
        if os.path.exists(pairs_path):
            pairs = load_json(pairs_path)
            logger.info(f"[SKIP] Loaded {len(pairs)} cached answer pairs")
        else:
            pairs = generate_dual_answers(model, tokenizer, queries, persona_context)
            save_json(pairs, pairs_path)

        # Self-verification + partitioning
        dist, ret, stats = verify_and_partition(model, tokenizer, pairs,
                                                persona_name, persona_context)
        save_json(dist, persona_distill_path)
        save_json(ret, persona_retain_path)
        save_json(stats, os.path.join(persona_dir, "stats.json"))

        all_distill.extend(dist)
        all_retain.extend(ret)
        all_stats[persona_name] = stats

    # Save combined data
    save_json(all_distill, distill_path)
    save_json(all_retain, retain_path)
    save_json(all_stats, os.path.join(data_dir, "generation_stats.json"))

    logger.info(f"\nTotal distill samples: {len(all_distill)}")
    logger.info(f"Total retain samples: {len(all_retain)}")

    # ---- Pre-compute teacher logits (model is already loaded) ----
    distill_logits_path = os.path.join(data_dir, "teacher_logits_distill.pt")
    retain_logits_path = os.path.join(data_dir, "teacher_logits_retain.pt")

    if not os.path.exists(distill_logits_path) and all_distill:
        logger.info(f"Computing teacher logits for {len(all_distill)} distill samples...")
        distill_logits = batch_compute_logits(model, tokenizer, all_distill,
                                              desc="Distill teacher logits")
        torch.save(distill_logits, distill_logits_path)
        logger.info(f"Saved distill teacher logits → {distill_logits_path}")
    else:
        logger.info(f"[SKIP] Distill teacher logits already exist or no distill data")

    if not os.path.exists(retain_logits_path) and all_retain:
        logger.info(f"Computing teacher logits for {len(all_retain)} retain samples...")
        retain_logits = batch_compute_logits(model, tokenizer, all_retain,
                                             desc="Retain teacher logits")
        torch.save(retain_logits, retain_logits_path)
        logger.info(f"Saved retain teacher logits → {retain_logits_path}")
    else:
        logger.info(f"[SKIP] Retain teacher logits already exist or no retain data")

    unload_model(model, tokenizer)
    logger.info("Stage 2 complete (data + teacher logits saved).")



def main():
    parser = argparse.ArgumentParser(description="PRISM Stage 2: Verify & Recycle")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    slug = get_model_slug(args.model)
    data_dir = args.data_dir or f"dataset/synthetic/persona_prism/{slug}"
    run(args.model, data_dir)


if __name__ == "__main__":
    main()
