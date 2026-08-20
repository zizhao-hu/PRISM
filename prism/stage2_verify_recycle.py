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
  python -m prism.stage2_verify_recycle --model Qwen/Qwen2.5-7B-Instruct
"""

import os
import sys
import argparse
import random
import logging
import re

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

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
# Pairwise Comparison with Position Swapping (Self-Judge)
# ============================================================

PAIRWISE_SYSTEM = (
    "You are a fair and rigorous evaluator. You will be given a question and "
    "two candidate answers (Answer A and Answer B). Compare them carefully.\n\n"
    "Evaluation criteria (in order of importance):\n"
    "1. CORRECTNESS: Is the answer factually accurate and logically sound?\n"
    "2. RELEVANCE: Does it directly address the question?\n"
    "3. HELPFULNESS: Is it useful and complete?\n"
    "4. CLARITY: Is it well-organized and easy to follow?\n\n"
    "IMPORTANT: Do NOT favor longer answers. A concise, correct answer is "
    "better than a verbose, padded answer. Unnecessary elaboration or "
    "repetition should count AGAINST an answer.\n\n"
    "Output ONLY the letter of the better answer: 'A' or 'B'. "
    "If they are equally good, output 'TIE'."
)


def _parse_pairwise(response):
    """Extract the winner from a pairwise comparison response.
    
    Returns: 'A', 'B', or 'TIE'
    """
    response = response.strip().upper()
    # Direct match
    if response in ('A', 'B', 'TIE'):
        return response
    # Look for "Answer A" / "Answer B"
    if 'ANSWER A' in response and 'ANSWER B' not in response:
        return 'A'
    if 'ANSWER B' in response and 'ANSWER A' not in response:
        return 'B'
    # Look for leading A/B
    if response.startswith('A'):
        return 'A'
    if response.startswith('B'):
        return 'B'
    if 'TIE' in response or 'EQUAL' in response or 'SAME' in response:
        return 'TIE'
    return 'TIE'  # fallback: conservative


def grade_answers_pairwise(model, tokenizer, query_data, expert_name):
    """Grade answers via pairwise comparison with position swapping.
    
    For each query, run TWO comparisons:
      Pass 1: A = baseline, B = expert
      Pass 2: A = expert,   B = baseline
      
    Expert wins ONLY if chosen in BOTH orderings (conservative).
    This eliminates position bias and verbosity bias.
    
    Args:
        query_data: list of {"query": str, "answers": {"baseline": str, expert_name: str}}
        expert_name: name of the expert persona (e.g. "math")
        
    Returns:
        query_data with added "pairwise" field:
            {"pass1": "A"/"B"/"TIE", "pass2": "A"/"B"/"TIE", "winner": "expert"/"baseline"/"tie"}
    """
    n = len(query_data)
    
    # ---- Pass 1: A=baseline, B=expert ----
    prompts_pass1 = []
    for qd in query_data:
        query = qd["query"]
        ans_bl = qd["answers"]["baseline"][:1500]
        ans_exp = qd["answers"][expert_name][:1500]
        prompt = (
            f"Question: {query}\n\n"
            f"Answer A:\n{ans_bl}\n\n"
            f"Answer B:\n{ans_exp}\n\n"
            f"Which answer is better? Output A, B, or TIE:"
        )
        prompts_pass1.append(prompt)
    
    logger.info(f"  Pairwise grading Pass 1 ({n} queries, A=baseline B=expert)...")
    msgs_p1 = [build_chat_messages(tokenizer, PAIRWISE_SYSTEM, p) for p in prompts_pass1]
    resp_p1 = batch_generate(model, tokenizer, msgs_p1,
                              max_tokens=5, temperature=0.0,
                              batch_size=GRADE_BATCH_SIZE)
    
    # ---- Pass 2: A=expert, B=baseline (swapped) ----
    prompts_pass2 = []
    for qd in query_data:
        query = qd["query"]
        ans_bl = qd["answers"]["baseline"][:1500]
        ans_exp = qd["answers"][expert_name][:1500]
        prompt = (
            f"Question: {query}\n\n"
            f"Answer A:\n{ans_exp}\n\n"
            f"Answer B:\n{ans_bl}\n\n"
            f"Which answer is better? Output A, B, or TIE:"
        )
        prompts_pass2.append(prompt)
    
    logger.info(f"  Pairwise grading Pass 2 ({n} queries, A=expert B=baseline)...")
    msgs_p2 = [build_chat_messages(tokenizer, PAIRWISE_SYSTEM, p) for p in prompts_pass2]
    resp_p2 = batch_generate(model, tokenizer, msgs_p2,
                              max_tokens=5, temperature=0.0,
                              batch_size=GRADE_BATCH_SIZE)
    
    # ---- Resolve winners ----
    # Pass 1: A=baseline, B=expert  → expert wins if choice='B'
    # Pass 2: A=expert, B=baseline  → expert wins if choice='A'
    stats = {"expert_both": 0, "baseline_both": 0, "mixed": 0, "tie": 0}
    
    for i, qd in enumerate(query_data):
        p1 = _parse_pairwise(resp_p1[i])  # A=baseline, B=expert
        p2 = _parse_pairwise(resp_p2[i])  # A=expert, B=baseline
        
        # Normalize: did expert win in each pass?
        expert_p1 = (p1 == 'B')       # expert is B in pass 1
        expert_p2 = (p2 == 'A')       # expert is A in pass 2
        tie_p1 = (p1 == 'TIE')
        tie_p2 = (p2 == 'TIE')
        
        if expert_p1 and expert_p2:
            winner = "expert"
            stats["expert_both"] += 1
        elif (not expert_p1 and not tie_p1) and (not expert_p2 and not tie_p2):
            winner = "baseline"
            stats["baseline_both"] += 1
        elif tie_p1 and tie_p2:
            winner = "tie"
            stats["tie"] += 1
        else:
            winner = "tie"  # mixed results → conservative: treat as tie → retain
            stats["mixed"] += 1
        
        qd["pairwise"] = {
            "pass1_raw": p1,
            "pass2_raw": p2,
            "expert_wins_pass1": expert_p1,
            "expert_wins_pass2": expert_p2,
            "winner": winner,
        }
    
    logger.info(f"  Pairwise results:")
    logger.info(f"    Expert wins both:   {stats['expert_both']}")
    logger.info(f"    Baseline wins both: {stats['baseline_both']}")
    logger.info(f"    Mixed (→ retain):   {stats['mixed']}")
    logger.info(f"    Tie both (→ retain):{stats['tie']}")
    
    return query_data


# Keep legacy pointwise grading available for backward compatibility
GRADE_SYSTEM = (
    "You are an expert evaluator. Rate the quality of an answer to a question. "
    "Consider: accuracy, helpfulness, depth, clarity, and relevance. "
    "Prefer concise, correct answers over verbose ones. "
    "Do NOT reward unnecessary length or repetition. "
    "Output ONLY a single number from 1 to 10 (integers or one decimal place). "
    "1 = terrible, 5 = mediocre, 10 = exceptional."
)

def _parse_grade(response):
    """Extract a numeric grade from the judge response."""
    response = response.strip()
    match = re.search(r'(\d+\.?\d*)', response)
    if match:
        score = float(match.group(1))
        return min(max(score, 1.0), 10.0)
    return 5.0


def grade_answers(model, tokenizer, query_data):
    """Legacy pointwise grading. Kept for backward compatibility."""
    grading_tasks = []
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
    
    msgs_list = [
        build_chat_messages(tokenizer, GRADE_SYSTEM, prompt)
        for _, _, prompt in grading_tasks
    ]
    grade_responses = batch_generate(model, tokenizer, msgs_list,
                                     max_tokens=10, temperature=0.0,
                                     batch_size=GRADE_BATCH_SIZE)
    
    for qd in query_data:
        qd["grades"] = {}
    for (qi, name, _), response in zip(grading_tasks, grade_responses):
        grade = _parse_grade(response)
        query_data[qi]["grades"][name] = grade
    
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
    """Convert pairwise-graded data into training samples.
    
    Uses position-swapped pairwise comparison results:
      - Expert wins BOTH orderings → Distill set (gate target = 1)
      - Otherwise                  → Retain set  (gate target = 0)
    
    Falls back to pointwise grades if pairwise data is not available.
    
    Args:
        query_data: list of {"query", "answers", "pairwise"/"grades", "persona_source"}
        expert_name: name of the expert persona
        expert_context: full text of the expert persona prompt
        
    Returns:
        (distill_data, retain_data, stats)
    """
    distill_data = []
    retain_data = []
    stats = {
        "total_queries": len(query_data),
        "context_wins": 0,
        "base_wins": 0,
        "ties": 0,
        "mixed": 0,
    }
    
    for qd in query_data:
        query = qd["query"]
        answers = qd["answers"]
        persona_source = qd.get("persona_source", "unknown")
        
        # Use pairwise results if available, fallback to pointwise
        if "pairwise" in qd:
            winner = qd["pairwise"]["winner"]
        else:
            # Legacy fallback: pointwise comparison
            grades = qd.get("grades", {})
            bl_grade = grades.get("baseline", 5.0)
            exp_grade = grades.get(expert_name, 5.0)
            if exp_grade > bl_grade:
                winner = "expert"
            elif bl_grade > exp_grade:
                winner = "baseline"
            else:
                winner = "tie"
        
        sample_base = {
            "instruction": query,
            "persona_source": persona_source,
        }
        # Attach grading details if available
        if "pairwise" in qd:
            sample_base["pairwise"] = qd["pairwise"]
        if "grades" in qd:
            sample_base["grades"] = qd["grades"]
        
        if winner == "expert":
            stats["context_wins"] += 1
            distill_data.append({
                **sample_base,
                "output": answers[expert_name],
                "system": expert_context,
                "dataset_type": "distill",
                "persona": expert_name,
                "gate_target": 1,
            })
        else:
            if winner == "tie":
                stats["ties"] += 1
            elif winner == "baseline":
                stats["base_wins"] += 1
            else:
                stats["mixed"] += 1
            retain_data.append({
                **sample_base,
                "output": answers["baseline"],
                "system": "",
                "dataset_type": "retain",
                "persona": "baseline",
                "gate_target": 0,
            })
    
    logger.info(f"\nTraining data built:")
    logger.info(f"  Distill (expert wins both):  {len(distill_data)}")
    logger.info(f"  Retain  (baseline/tie/mixed): {len(retain_data)}")
    logger.info(f"    breakdown: base={stats['base_wins']}, "
                f"tie={stats['ties']}, mixed={stats['mixed']}")
    
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
    
    # 2. Pairwise comparison with position swapping (debiased)
    query_data = grade_answers_pairwise(model, tokenizer, query_data,
                                         persona_source_name)
    
    # 3. Build training data: expert wins both orderings → distill
    distill_data, retain_data, stats = build_training_data(
        query_data, persona_source_name, expert_context
    )
    
    return distill_data, retain_data, query_data, stats


# ============================================================
# Full Pipeline Runner (standalone)
# ============================================================

def run(model_name, data_dir, regrade=False):
    """Run Stage 2 for all personas.
    
    Args:
        model_name: HuggingFace model name
        data_dir: path to the round data directory
        regrade: if True, reload existing graded_answers.json and
                 re-judge with pairwise comparison. Doesn't regenerate
                 answers — only re-does the grading step.
    """
    distill_path = os.path.join(data_dir, "distill_set.json")
    retain_path = os.path.join(data_dir, "retain_set.json")
    
    if not regrade:
        if os.path.exists(distill_path) and os.path.exists(retain_path):
            d = load_json(distill_path)
            r = load_json(retain_path)
            logger.info(f"[SKIP] Data already exists: {len(d)} distill, {len(r)} retain")
            return
    else:
        logger.info("=== REGRADE MODE: re-judging existing answers with pairwise comparison ===")

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

        persona_distill_path = os.path.join(persona_dir, "distill.json")
        persona_retain_path = os.path.join(persona_dir, "retain.json")
        graded_path = os.path.join(persona_dir, "graded_answers.json")

        if regrade:
            # REGRADE: load existing graded answers (which contain the
            # generated answers), then re-judge with pairwise comparison
            if not os.path.exists(graded_path):
                logger.warning(f"[SKIP-REGRADE] No graded_answers.json for {persona_name}")
                # Fall back to loading existing distill/retain if available
                if os.path.exists(persona_distill_path) and os.path.exists(persona_retain_path):
                    all_distill.extend(load_json(persona_distill_path))
                    all_retain.extend(load_json(persona_retain_path))
                continue

            query_data = load_json(graded_path)
            logger.info(f"  Loaded {len(query_data)} graded answers for re-judging")

            # Re-judge with pairwise comparison
            query_data = grade_answers_pairwise(model, tokenizer, query_data,
                                                persona_name)

            # Rebuild training data from new pairwise results
            distill, retain, stats = build_training_data(
                query_data, persona_name, expert_context
            )

            # Save updated per-persona results
            save_json(distill, persona_distill_path)
            save_json(retain, persona_retain_path)
            save_json(query_data, graded_path)  # overwrite with pairwise data
            save_json(stats, os.path.join(persona_dir, "stats.json"))

            all_distill.extend(distill)
            all_retain.extend(retain)
            all_graded.extend(query_data)
            all_stats[persona_name] = stats

        else:
            # NORMAL: check skip, then generate + grade
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

            # Run Stage 2: generate answers + pairwise grade
            distill, retain, graded, stats = run_stage2(
                model, tokenizer, queries, persona_name,
                expert_context
            )

            # Save per-persona results
            save_json(distill, persona_distill_path)
            save_json(retain, persona_retain_path)
            save_json(graded, graded_path)
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
    # Always recompute when regrading since distill/retain split changed
    distill_logits_path = os.path.join(data_dir, "teacher_logits_distill.pt")
    retain_logits_path = os.path.join(data_dir, "teacher_logits_retain.pt")
    force_logits = regrade  # force recompute if regrading

    if (force_logits or not os.path.exists(distill_logits_path)) and all_distill:
        logger.info(f"Computing teacher logits for {len(all_distill)} distill samples...")
        distill_logits = batch_compute_logits(model, tokenizer, all_distill,
                                              desc="Distill teacher logits")
        torch.save(distill_logits, distill_logits_path)
        logger.info(f"Saved distill teacher logits → {distill_logits_path}")
    else:
        logger.info(f"[SKIP] Distill teacher logits already exist or no distill data")

    if (force_logits or not os.path.exists(retain_logits_path)) and all_retain:
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
    parser.add_argument("--regrade", action="store_true",
                        help="Re-judge existing answers with pairwise comparison "
                             "(no answer regeneration, only re-grading)")
    args = parser.parse_args()

    slug = get_model_slug(args.model)
    data_dir = args.data_dir or f"dataset/synthetic/persona_prism/{slug}"
    run(args.model, data_dir, regrade=args.regrade)


if __name__ == "__main__":
    main()
