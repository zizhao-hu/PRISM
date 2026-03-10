"""
PRISM Stage 2: Expert Persona Evaluation & Grading

For each query (generated in Stage 1 by a specific expert persona):
  1. Generate 2 answers: baseline (no persona) + the expert persona that
     generated this query
  2. Grade both answers independently (pointwise 1-10 score)
  3. Select the better answer for distillation:
     - Expert wins  → Distill set: expert persona's answer
     - Baseline wins → Retain set:  baseline answer (preserve base behavior)

This focused comparison avoids the cost of generating/grading all K
personas and directly measures whether the matched expert helps.

Usage:
  python -m scripts.prism.stage2_verify_recycle --model Qwen/Qwen2.5-7B-Instruct
"""

import os
import sys
import argparse
import random
import logging
import re

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    load_json, save_json, load_text, load_model, unload_model,
    build_chat_messages, generate_response, batch_generate,
    batch_compute_logits, get_model_slug,
)

GEN_BATCH_SIZE = 8   # prompts in parallel per forward pass
GRADE_BATCH_SIZE = 8  # grading prompts in parallel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 8 task-specific personas
TASK_PERSONAS = {
    "writing":    "dataset/personas/full_personas/persona_writing.txt",
    "roleplay":   "dataset/personas/full_personas/persona_roleplay.txt",
    "reasoning":  "dataset/personas/full_personas/persona_reasoning.txt",
    "math":       "dataset/personas/full_personas/persona_math.txt",
    "coding":     "dataset/personas/full_personas/persona_coding.txt",
    "extraction": "dataset/personas/full_personas/persona_extraction.txt",
    "stem":       "dataset/personas/full_personas/persona_stem.txt",
    "humanities": "dataset/personas/full_personas/persona_humanities.txt",
}

# 4 behavioral personas
BEHAVIORAL_PERSONAS = {
    "critic":         "dataset/personas/full_personas/persona_critic.txt",
    "safety_monitor": "dataset/personas/full_personas/persona_safety_monitor.txt",
    "helpful":        "dataset/personas/full_personas/persona_helpful.txt",
    "compliant":      "dataset/personas/full_personas/persona_compliant.txt",
}

# All 12 personas
PERSONA_CONTEXTS = {**TASK_PERSONAS, **BEHAVIORAL_PERSONAS}

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ============================================================
# Expert Persona Answer Generation (baseline + expert only)
# ============================================================

def generate_expert_answers(model, tokenizer, queries, expert_name, expert_context):
    """For each query, generate 2 answers: baseline (no persona) + expert persona.
    
    Args:
        queries: list of query strings
        expert_name: name of the expert persona (e.g. 'writing')
        expert_context: full text of the expert persona prompt
        
    Returns:
        list of dicts, one per query:
          {"query": str, "answers": {"baseline": str, "<expert_name>": str}}
    """
    results = [{"query": q, "answers": {}} for q in queries]
    
    # Generate baseline answers (no persona)
    msgs_bl = [build_chat_messages(tokenizer, None, q) for q in queries]
    logger.info(f"  Generating {len(queries)} baseline answers (batch={GEN_BATCH_SIZE})...")
    bl_answers = batch_generate(model, tokenizer, msgs_bl,
                                max_tokens=512, temperature=0.7,
                                batch_size=GEN_BATCH_SIZE)
    for i, ans in enumerate(bl_answers):
        results[i]["answers"]["baseline"] = ans
    
    # Generate expert persona answers
    msgs_exp = [build_chat_messages(tokenizer, expert_context, q) for q in queries]
    logger.info(f"  Generating {len(queries)} [{expert_name}] answers (batch={GEN_BATCH_SIZE})...")
    exp_answers = batch_generate(model, tokenizer, msgs_exp,
                                 max_tokens=512, temperature=0.7,
                                 batch_size=GEN_BATCH_SIZE)
    for i, ans in enumerate(exp_answers):
        results[i]["answers"][expert_name] = ans
    
    return results


# ============================================================
# Pointwise Grading (Self-Judge)
# ============================================================

GRADE_SYSTEM = (
    "You are an expert evaluator. Rate the quality of an answer to a question. "
    "Consider: accuracy, helpfulness, depth, clarity, and relevance. "
    "Output ONLY a single number from 1 to 10 (integers or one decimal place). "
    "1 = terrible, 5 = mediocre, 10 = exceptional."
)

def _parse_grade(response):
    """Extract a numeric grade from the judge response."""
    response = response.strip()
    # Try to find a number (integer or decimal) in the response
    match = re.search(r'(\d+\.?\d*)', response)
    if match:
        score = float(match.group(1))
        return min(max(score, 1.0), 10.0)  # clamp to [1, 10]
    return 5.0  # fallback


def grade_answers(model, tokenizer, query_data):
    """Grade baseline and expert answers for all queries using pointwise evaluation.
    
    Args:
        query_data: list of {"query": str, "answers": {"baseline": str, expert_name: str}}
        
    Returns:
        list of {"query": str, "answers": {...}, "grades": {"baseline": float, expert: float}}
    """
    # Build all grading prompts at once for batching
    grading_tasks = []  # (query_idx, answer_name, prompt_text)
    
    for qi, qd in enumerate(query_data):
        query = qd["query"]
        for name, answer in qd["answers"].items():
            grade_prompt = (
                f"Question: {query}\n\n"
                f"Answer:\n{answer[:1000]}\n\n"
                f"Rate the quality of this answer (1-10):"
            )
            grading_tasks.append((qi, name, grade_prompt))
    
    logger.info(f"  Grading {len(grading_tasks)} answers "
                f"({len(query_data)} queries × 2)...")
    
    # Batch grade
    msgs_list = [
        build_chat_messages(tokenizer, GRADE_SYSTEM, prompt)
        for _, _, prompt in grading_tasks
    ]
    
    grade_responses = batch_generate(model, tokenizer, msgs_list,
                                     max_tokens=10, temperature=0.0,
                                     batch_size=GRADE_BATCH_SIZE)
    
    # Parse grades and attach to query_data
    for qd in query_data:
        qd["grades"] = {}
    
    for (qi, name, _), response in zip(grading_tasks, grade_responses):
        grade = _parse_grade(response)
        query_data[qi]["grades"][name] = grade
    
    # Log grade statistics
    all_grades = {}
    for qd in query_data:
        for name, grade in qd["grades"].items():
            all_grades.setdefault(name, []).append(grade)
    
    logger.info("  Grade summary (mean ± std):")
    for name in sorted(all_grades.keys()):
        vals = all_grades[name]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        logger.info(f"    {name:20s}: {mean:.2f} ± {std:.2f}")
    
    return query_data


# ============================================================
# Build Training Data with Routing Targets
# ============================================================

def build_training_data(query_data, expert_name, expert_context):
    """Convert graded query data into training samples.
    
    For each query, compare baseline vs expert grade:
      - Expert wins  → Distill set (gate target = 1)
      - Baseline wins → Retain set (gate target = 0)
    
    Args:
        query_data: list of {"query", "answers", "grades", "persona_source"}
        expert_name: name of the expert persona
        expert_context: full text of the expert persona prompt
        
    Returns:
        (distill_data, retain_data, stats)
    """
    distill_data = []
    retain_data = []
    stats = {
        "total_queries": len(query_data),
        "baseline_wins": 0,
        "expert_wins": 0,
        "ties": 0,
        "avg_baseline_grade": 0,
        "avg_expert_grade": 0,
    }
    
    bl_grades = []
    exp_grades = []
    
    for qd in query_data:
        query = qd["query"]
        grades = qd["grades"]
        answers = qd["answers"]
        persona_source = qd.get("persona_source", "unknown")
        
        bl_grade = grades.get("baseline", 5.0)
        exp_grade = grades.get(expert_name, 5.0)
        bl_grades.append(bl_grade)
        exp_grades.append(exp_grade)
        
        # Common fields for the training sample
        sample_base = {
            "instruction": query,
            "persona_source": persona_source,
            "grades": grades,
            "baseline_grade": bl_grade,
            "expert_grade": exp_grade,
        }
        
        if exp_grade > bl_grade:
            # Expert wins → distill set (gate target = 1)
            stats["expert_wins"] += 1
            distill_data.append({
                **sample_base,
                "output": answers[expert_name],
                "system": expert_context,
                "dataset_type": "distill",
                "persona": expert_name,
                "gate_target": 1,
            })
        else:
            # Baseline wins or tie → retain set (gate target = 0)
            if bl_grade == exp_grade:
                stats["ties"] += 1
            else:
                stats["baseline_wins"] += 1
            retain_data.append({
                **sample_base,
                "output": answers["baseline"],
                "system": "",
                "dataset_type": "retain",
                "persona": "baseline",
                "gate_target": 0,
            })
    
    stats["avg_baseline_grade"] = round(sum(bl_grades) / len(bl_grades), 2) if bl_grades else 0
    stats["avg_expert_grade"] = round(sum(exp_grades) / len(exp_grades), 2) if exp_grades else 0
    
    logger.info(f"\nTraining data built:")
    logger.info(f"  Distill (expert wins):  {len(distill_data)}")
    logger.info(f"  Retain  (baseline wins): {len(retain_data)} (ties: {stats['ties']})")
    logger.info(f"  Avg baseline grade: {stats['avg_baseline_grade']}")
    logger.info(f"  Avg expert grade:   {stats['avg_expert_grade']}")
    
    return distill_data, retain_data, stats


# ============================================================
# Combined Stage 2 Runner  
# ============================================================

def run_stage2(model, tokenizer, queries, persona_source_name, 
               expert_context):
    """Full Stage 2 for a batch of queries from one persona source.
    
    Args:
        queries: list of query strings (from Stage 1 for one persona)
        persona_source_name: which persona generated these queries
        expert_context: full text of the expert persona prompt
        
    Returns:
        (distill_data, retain_data, graded_data, stats)
    """
    # 1. Generate 2 answers per query: baseline + expert persona
    query_data = generate_expert_answers(model, tokenizer, queries,
                                         persona_source_name, expert_context)
    
    # Tag each query with its source persona
    for qd in query_data:
        qd["persona_source"] = persona_source_name
    
    # 2. Grade baseline and expert answers
    query_data = grade_answers(model, tokenizer, query_data)
    
    # 3. Build training data: expert wins → distill, baseline wins → retain
    distill_data, retain_data, stats = build_training_data(
        query_data, persona_source_name, expert_context
    )
    
    return distill_data, retain_data, query_data, stats


# ============================================================
# Full Pipeline Runner (standalone)
# ============================================================

def run(model_name, data_dir):
    """Run Stage 2 for all personas."""
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
    all_graded = []
    all_stats = {}

    for persona_name, persona_path in PERSONA_CONTEXTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing queries from persona: {persona_name}")
        logger.info(f"{'='*60}")

        expert_context = load_text(persona_path)
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

        # Load queries from Stage 1
        queries_path = os.path.join(persona_dir, "queries.json")
        if not os.path.exists(queries_path):
            logger.error(f"Queries not found: {queries_path}. Run Stage 1 first.")
            continue
        queries = load_json(queries_path)

        # Run Stage 2: baseline vs expert only
        distill, retain, graded, stats = run_stage2(
            model, tokenizer, queries, persona_name,
            expert_context
        )

        # Save per-persona results
        save_json(distill, persona_distill_path)
        save_json(retain, persona_retain_path)
        save_json(graded, os.path.join(persona_dir, "graded_answers.json"))
        save_json(stats, os.path.join(persona_dir, "stats.json"))

        all_distill.extend(distill)
        all_retain.extend(retain)
        all_graded.extend(graded)
        all_stats[persona_name] = stats

    # Save combined data
    save_json(all_distill, distill_path)
    save_json(all_retain, retain_path)
    save_json(all_stats, os.path.join(data_dir, "generation_stats.json"))

    logger.info(f"\nTotal distill samples: {len(all_distill)}")
    logger.info(f"Total retain samples: {len(all_retain)}")

    # ---- Pre-compute teacher logits ----
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
    logger.info("Stage 2 complete (graded data + teacher logits saved).")


def main():
    parser = argparse.ArgumentParser(description="PRISM Stage 2: Expert Persona Grading")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    slug = get_model_slug(args.model)
    data_dir = args.data_dir or f"dataset/synthetic/persona_prism/{slug}"
    run(args.model, data_dir)


if __name__ == "__main__":
    main()
