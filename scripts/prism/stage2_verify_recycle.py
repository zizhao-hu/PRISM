"""
PRISM Stage 2: Multi-Persona Evaluation & Grading

For each query (generated in Stage 1):
  1. Generate K+1 answers: baseline (no persona) + all K personas
  2. Grade each answer independently (pointwise 1-10 score)
  3. Compute soft routing targets: softmax(grades / τ)
  4. Select best persona answer for distillation:
     - Persona wins  → Distill set: best persona's answer
     - Baseline wins → Retain set:  baseline answer (preserve base behavior)

The routing targets provide rich supervision: instead of binary
"did this persona help?", the router learns the full quality
landscape across all personas + baseline for every query.

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
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    load_json, save_json, load_text, load_model, unload_model,
    build_chat_messages, generate_response, batch_generate,
    batch_compute_logits, get_model_slug,
)

GEN_BATCH_SIZE = 8   # prompts in parallel per forward pass
GRADE_BATCH_SIZE = 8  # grading prompts in parallel
ROUTING_TAU = 2.0     # temperature for softmax(grades) → routing targets

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
# Multi-Persona Answer Generation
# ============================================================

def generate_all_persona_answers(model, tokenizer, queries, all_personas):
    """For each query, generate K+1 answers (baseline + all K personas).
    
    Args:
        queries: list of query strings
        all_personas: dict {persona_name: context_text}
        
    Returns:
        list of dicts, one per query:
          {"query": str, "answers": {"baseline": str, "writing": str, ...}}
    """
    all_names = ["baseline"] + list(all_personas.keys())
    results = [{"query": q, "answers": {}} for q in queries]
    
    # Generate per-persona across ALL queries (batched for efficiency)
    for name in all_names:
        if name == "baseline":
            context = None
        else:
            context = all_personas[name]
        
        msgs_list = [build_chat_messages(tokenizer, context, q) for q in queries]
        
        logger.info(f"  Generating {len(queries)} answers for [{name}] "
                     f"(batch={GEN_BATCH_SIZE})...")
        answers = batch_generate(model, tokenizer, msgs_list,
                                 max_tokens=512, temperature=0.7,
                                 batch_size=GEN_BATCH_SIZE)
        
        for i, ans in enumerate(answers):
            results[i]["answers"][name] = ans
    
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


def grade_all_answers(model, tokenizer, query_data):
    """Grade all answers for all queries using pointwise evaluation.
    
    Args:
        query_data: list of {"query": str, "answers": {name: answer_text}}
        
    Returns:
        list of {"query": str, "answers": {...}, "grades": {name: float}}
    """
    # Build all grading prompts at once for batching
    grading_tasks = []  # (query_idx, persona_name, prompt_text)
    
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
                f"({len(query_data)} queries × {len(query_data[0]['answers'])} personas)...")
    
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

def build_training_data(query_data, persona_contexts, routing_tau=ROUTING_TAU):
    """Convert graded query data into training samples with soft routing targets.
    
    For each query:
      1. Compute routing_target = softmax(grades / τ) over [null, persona_1, ..., persona_K]
      2. Best persona → distill set (train that expert's LoRA)
      3. Baseline best → retain set (train null expert / preserve base)
    
    Args:
        query_data: list of {"query", "answers", "grades", "persona_source"}
        persona_contexts: dict {name: context_text} (all K personas)
        routing_tau: temperature for softmax normalization
        
    Returns:
        (distill_data, retain_data, stats)
    """
    persona_names = sorted(persona_contexts.keys())
    # Expert order: [null/baseline, persona_1, persona_2, ...]
    expert_names = ["baseline"] + persona_names
    
    distill_data = []
    retain_data = []
    stats = {
        "total_queries": len(query_data),
        "baseline_wins": 0,
        "persona_wins": {},
        "avg_grades": {},
    }
    
    for qd in query_data:
        query = qd["query"]
        grades = qd["grades"]
        answers = qd["answers"]
        persona_source = qd.get("persona_source", "unknown")
        
        # Build grade vector in expert order
        grade_vec = []
        for name in expert_names:
            grade_vec.append(grades.get(name, 5.0))
        
        # Compute soft routing target: softmax(grades / τ)
        grade_tensor = torch.tensor(grade_vec, dtype=torch.float32)
        routing_target = F.softmax(grade_tensor / routing_tau, dim=0).tolist()
        
        # Find best performer
        best_idx = grade_tensor.argmax().item()
        best_name = expert_names[best_idx]
        best_grade = grade_vec[best_idx]
        
        # Common fields for the training sample
        sample_base = {
            "instruction": query,
            "persona_source": persona_source,
            "grades": grades,
            "routing_target": routing_target,
            "expert_names": expert_names,
            "best_expert": best_name,
            "best_grade": best_grade,
        }
        
        if best_name == "baseline":
            # Baseline wins → retain set (null expert)
            stats["baseline_wins"] += 1
            retain_data.append({
                **sample_base,
                "output": answers["baseline"],
                "system": "",
                "dataset_type": "retain",
                "persona": "baseline",
            })
        else:
            # A persona wins → distill set (that persona's expert)
            stats["persona_wins"][best_name] = stats["persona_wins"].get(best_name, 0) + 1
            distill_data.append({
                **sample_base,
                "output": answers[best_name],
                "system": persona_contexts[best_name],
                "dataset_type": "distill",
                "persona": best_name,
            })
    
    # Compute average grades per persona
    for name in expert_names:
        vals = [qd["grades"].get(name, 5.0) for qd in query_data]
        stats["avg_grades"][name] = round(sum(vals) / len(vals), 2) if vals else 0
    
    logger.info(f"\nTraining data built:")
    logger.info(f"  Distill (persona wins): {len(distill_data)}")
    logger.info(f"  Retain  (baseline wins): {len(retain_data)}")
    logger.info(f"  Per-persona wins: {stats['persona_wins']}")
    logger.info(f"  Average grades: {stats['avg_grades']}")
    
    return distill_data, retain_data, stats


# ============================================================
# Combined Stage 2 Runner  
# ============================================================

def run_stage2(model, tokenizer, queries, persona_source_name, 
               all_persona_contexts, routing_tau=ROUTING_TAU):
    """Full Stage 2 for a batch of queries from one persona source.
    
    Args:
        queries: list of query strings (from Stage 1 for one persona)
        persona_source_name: which persona generated these queries
        all_persona_contexts: dict {name: context_text} for ALL K personas
        
    Returns:
        (distill_data, retain_data, graded_data, stats)
    """
    # 1. Generate K+1 answers per query
    query_data = generate_all_persona_answers(model, tokenizer, queries, all_persona_contexts)
    
    # Tag each query with its source persona
    for qd in query_data:
        qd["persona_source"] = persona_source_name
    
    # 2. Grade all answers
    query_data = grade_all_answers(model, tokenizer, query_data)
    
    # 3. Build training data with routing targets
    distill_data, retain_data, stats = build_training_data(
        query_data, all_persona_contexts, routing_tau
    )
    
    return distill_data, retain_data, query_data, stats


# ============================================================
# Full Pipeline Runner (standalone)
# ============================================================

def run(model_name, data_dir, routing_tau=ROUTING_TAU):
    """Run Stage 2 for all personas."""
    distill_path = os.path.join(data_dir, "distill_set.json")
    retain_path = os.path.join(data_dir, "retain_set.json")
    if os.path.exists(distill_path) and os.path.exists(retain_path):
        d = load_json(distill_path)
        r = load_json(retain_path)
        logger.info(f"[SKIP] Data already exists: {len(d)} distill, {len(r)} retain")
        return

    model, tokenizer = load_model(model_name)
    
    # Load all persona contexts as text
    all_persona_texts = {}
    for name, path in PERSONA_CONTEXTS.items():
        all_persona_texts[name] = load_text(path)
    
    all_distill = []
    all_retain = []
    all_graded = []
    all_stats = {}

    for persona_name in PERSONA_CONTEXTS.keys():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing queries from persona: {persona_name}")
        logger.info(f"{'='*60}")

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

        # Run full Stage 2
        distill, retain, graded, stats = run_stage2(
            model, tokenizer, queries, persona_name,
            all_persona_texts, routing_tau
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
    parser = argparse.ArgumentParser(description="PRISM Stage 2: Multi-Persona Grading")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--routing_tau", type=float, default=ROUTING_TAU,
                        help="Temperature for softmax(grades) → routing targets")
    args = parser.parse_args()

    slug = get_model_slug(args.model)
    data_dir = args.data_dir or f"dataset/synthetic/persona_prism/{slug}"
    run(args.model, data_dir, args.routing_tau)


if __name__ == "__main__":
    main()
