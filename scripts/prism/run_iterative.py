"""
PRISM Iterative Self-Distillation

Runs multiple rounds of (Stage 2 → Stage 3) with the SAME LoRA adapter:
  1. Uses the current model (base + LoRA) to generate fresh dual answers
  2. Judges, partitions into distill/retain sets, computes teacher logits
  3. Continues training the SAME LoRA for N more epochs
  4. Repeat

After all rounds: runs full evaluation using the canonical result layout.

Directory layout (controlled by --exp_name):
  dataset/synthetic/persona_prism/{exp_name}/           ← Stage 1 queries (shared)
  dataset/synthetic/persona_prism/{exp_name}/round_N/   ← per-round data & logits
  models/persona_prism/{exp_name}/                      ← LoRA adapter (reused)
  results/{exp_name}/baseline/{mt_bench,safety}/         ← baseline eval
  results/{exp_name}/persona/{name}/{mt_bench,safety}/   ← per-persona eval
  results/{exp_name}/prism/{mt_bench,safety}/            ← final PRISM eval

For test runs, use --exp_name test (everything saved under test/ directories).
For real runs, use --exp_name <model_slug> (or omit to auto-derive from model name).

Usage:
  # Test run (quick pipeline verification):
  python -m scripts.prism.run_iterative --model Qwen/Qwen2.5-7B-Instruct \\
      --exp_name test --rounds 1 --epochs_per_round 1

  # Real run (full training):
  python -m scripts.prism.run_iterative --model Qwen/Qwen2.5-7B-Instruct \\
      --rounds 5 --epochs_per_round 2
"""

import os
import sys
import json
import argparse
import logging
import subprocess

import torch

# Add scripts/ for utils, and scripts/prism/ for stage modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_json, save_json, load_text, load_model, unload_model, get_model_slug

# Import stage modules
from stage2_verify_recycle import (
    PERSONA_CONTEXTS, generate_dual_answers, verify_and_partition,
)
from stage3_distill import train as stage3_train

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ROUNDS = 5
EPOCHS_PER_ROUND = 2

# Root directories (exp_name is appended at runtime)
RESULTS_ROOT = "results"
DATA_ROOT = "dataset/synthetic/persona_prism"
ADAPTER_ROOT = "models/persona_prism"

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

# Behavioral personas for matched-row routing
# Safety matched persona: safety_monitor
# MMLU matched persona: helpful
SAFETY_PERSONA = {"safety_monitor": "dataset/personas/persona_safety_monitor.txt"}
MMLU_PERSONA   = {"helpful":        "dataset/personas/persona_helpful.txt"}

# Safety benchmarks (matching main.py)
SAFETY_BENCHMARKS = ["HarmBench", "Jailbreak", "PKU_SafeRLHF"]


# ============================================================
# Path helpers — exp_name controls all directories
# ============================================================

def result_dir(exp_name, setting, eval_type):
    """Canonical path: results/{exp_name}/{setting}/{eval_type}/"""
    d = os.path.join(RESULTS_ROOT, exp_name, setting, eval_type)
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================
# Stage 2 per round
# ============================================================

def run_stage2_for_round(model, tokenizer, data_dir, round_num):
    """
    Run Stage 2 for a specific round.
    Reuses queries from Stage 1, generates fresh answers from the current model.
    Teacher logits come from the current model (base + LoRA for round > 1).
    """
    round_data_dir = os.path.join(data_dir, f"round_{round_num}")
    os.makedirs(round_data_dir, exist_ok=True)

    # Check if this round is already done
    distill_path = os.path.join(round_data_dir, "distill_set.json")
    retain_path = os.path.join(round_data_dir, "retain_set.json")
    distill_logits_path = os.path.join(round_data_dir, "teacher_logits_distill.pt")
    retain_logits_path = os.path.join(round_data_dir, "teacher_logits_retain.pt")

    if (os.path.exists(distill_path) and os.path.exists(retain_path)
            and os.path.exists(distill_logits_path)):
        d = load_json(distill_path)
        r = load_json(retain_path)
        logger.info(f"[SKIP] Round {round_num} data + logits exist: "
                     f"{len(d)} distill, {len(r)} retain")
        return round_data_dir

    all_distill = []
    all_retain = []
    all_stats = {}

    for persona_name, context_path in PERSONA_CONTEXTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_num} | Persona: {persona_name}")
        logger.info(f"{'='*60}")

        persona_context = load_text(context_path)

        # Load queries from Stage 1 (shared across all rounds)
        queries_path = os.path.join(data_dir, "per_persona", persona_name, "queries.json")
        if not os.path.exists(queries_path):
            logger.error(f"Queries not found: {queries_path}. Run Stage 1 first.")
            continue
        queries = load_json(queries_path)

        # Generate fresh dual answers with current model
        pairs = generate_dual_answers(model, tokenizer, queries, persona_context)

        # Judge and partition
        dist, ret, stats = verify_and_partition(
            model, tokenizer, pairs, persona_name, persona_context
        )

        # Save per-persona results
        persona_round_dir = os.path.join(round_data_dir, "per_persona", persona_name)
        os.makedirs(persona_round_dir, exist_ok=True)
        save_json(dist, os.path.join(persona_round_dir, "distill.json"))
        save_json(ret, os.path.join(persona_round_dir, "retain.json"))
        save_json(stats, os.path.join(persona_round_dir, "stats.json"))

        all_distill.extend(dist)
        all_retain.extend(ret)
        all_stats[persona_name] = stats

    # Save combined data
    save_json(all_distill, distill_path)
    save_json(all_retain, retain_path)
    save_json(all_stats, os.path.join(round_data_dir, "generation_stats.json"))

    logger.info(f"\nRound {round_num}: {len(all_distill)} distill, {len(all_retain)} retain")

    # Compute teacher logits from the CURRENT model (includes LoRA if round > 1)
    from utils import batch_compute_logits

    if not os.path.exists(distill_logits_path) and all_distill:
        logger.info(f"Computing teacher logits for {len(all_distill)} distill samples...")
        distill_logits = batch_compute_logits(model, tokenizer, all_distill,
                                              desc="Distill teacher logits",
                                              save_path=distill_logits_path)
        torch.save(distill_logits, distill_logits_path)
        del distill_logits
        import gc; gc.collect()
        logger.info(f"Saved → {distill_logits_path}")

    if not os.path.exists(retain_logits_path) and all_retain:
        logger.info(f"Computing teacher logits for {len(all_retain)} retain samples...")
        retain_logits = batch_compute_logits(model, tokenizer, all_retain,
                                             desc="Retain teacher logits",
                                             save_path=retain_logits_path)
        torch.save(retain_logits, retain_logits_path)
        del retain_logits
        import gc; gc.collect()
        logger.info(f"Saved → {retain_logits_path}")

    return round_data_dir


# ============================================================
# Evaluation helpers
# ============================================================

def _mt_bench_done(out_dir):
    return os.path.exists(os.path.join(out_dir, "mt_bench_summary.json"))


def _summarize_mt_bench(judgment_file, summary_file):
    """Per-category averages from MT-Bench judgments → summary JSON."""
    if not os.path.exists(judgment_file):
        return {}

    scores = {}
    with open(judgment_file) as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            cat = j.get("category", "unknown")
            score = j.get("avg_score", j.get("score", 0))
            if score > 0:
                scores.setdefault(cat, []).append(score)

    per_category = {cat: round(sum(vals)/len(vals), 2) for cat, vals in scores.items()}
    all_vals = [v for vals in scores.values() for v in vals]
    overall = round(sum(all_vals)/len(all_vals), 2) if all_vals else 0

    summary = {"overall": overall, "per_category": per_category,
               "n_questions": sum(len(v) for v in scores.values())}
    save_json(summary, summary_file)
    logger.info(f"  MT-Bench: avg={overall}  {per_category}")
    return summary


def _run_mt_bench(exp_name, setting, model_name, adapter_path=None, system_prompt=None,
                  judge_model=None):
    """Generate + judge + summarize MT-Bench for one setting."""
    out = result_dir(exp_name, setting, "mt_bench")
    if _mt_bench_done(out):
        logger.info(f"[SKIP] MT-Bench done: {out}")
        return

    answer_file = os.path.join(out, "answers.jsonl")
    judgment_file = os.path.join(out, "judgments.jsonl")
    summary_file = os.path.join(out, "mt_bench_summary.json")
    question_file = "dataset/eval/mt_bench/question.jsonl"

    if not os.path.exists(question_file):
        logger.warning(f"Question file not found: {question_file}")
        return

    judge = judge_model or model_name

    # Generate
    if not os.path.exists(answer_file):
        cmd = [sys.executable, "scripts/eval/eval_mt_bench.py",
               "--mode", "generate", "--model", model_name,
               "--question_file", question_file, "--output_file", answer_file]
        if adapter_path:
            cmd += ["--adapter_path", adapter_path]
        if system_prompt:
            cmd += ["--system_prompt", system_prompt]
        logger.info(f"  MT-Bench generate: {setting}")
        subprocess.run(cmd, check=True)

    # Judge
    if not os.path.exists(judgment_file) and os.path.exists(answer_file):
        cmd = [sys.executable, "scripts/eval/eval_mt_bench.py",
               "--mode", "judge", "--judge_model", judge,
               "--question_file", question_file,
               "--answer_file", answer_file, "--output_file", judgment_file]
        logger.info(f"  MT-Bench judge: {setting}")
        subprocess.run(cmd, check=True)

    # Summarize
    _summarize_mt_bench(judgment_file, summary_file)


def _run_safety(exp_name, setting, model_name, adapter_path=None, context_file=None):
    """Run safety eval using eval_safety.py.

    Results land at: results/{exp_name}/safety/main/{setting}/{benchmark}/{slug}/summary.json
    We use exp_name as the output_root prefix so test results are isolated.
    """
    setting_key = setting.replace("/", "_")
    slug = get_model_slug(model_name)
    slug_ft = slug + "_finetuned"

    # Check if already done (look for any benchmark summary)
    check_dir = os.path.join(RESULTS_ROOT, exp_name, "safety", "main", setting_key)
    if os.path.exists(check_dir):
        search_slugs = [slug_ft, slug] if adapter_path else [slug]
        all_done = any(
            all(
                os.path.exists(os.path.join(check_dir, bm, s, "summary.json"))
                for bm in SAFETY_BENCHMARKS
            )
            for s in search_slugs
        )
        if all_done:
            logger.info(f"[SKIP] Safety done: {check_dir}")
            return

    # Use results/{exp_name}/safety as the output root for eval_safety.py
    safety_root = os.path.join(RESULTS_ROOT, exp_name, "safety")
    cmd = [sys.executable, "scripts/eval/eval_safety.py",
           "--base_model", model_name,
           "--judge_model", model_name,
           "--benchmarks"] + SAFETY_BENCHMARKS + [
           "--output_root", safety_root,
           "--experiment_type", "main",
           "--experiment_name", setting_key,
           "--skip_utility", "--skip_kl"]
    if adapter_path:
        cmd += ["--adapter_path", adapter_path]
    if context_file:
        cmd += ["--context_file", context_file]
    logger.info(f"  Safety: {setting}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "scripts" + os.pathsep + env.get("PYTHONPATH", "")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Safety eval failed: {e}")



def _run_utility(exp_name, model_name, adapter_path):
    """G-Eval + Win Rate + KL divergence (PRISM vs base).

    Results land at: results/{exp_name}/safety/main/prism/utility/{slug}/
    """
    slug = get_model_slug(model_name)
    slug_ft = slug + "_finetuned"
    safety_root = os.path.join(RESULTS_ROOT, exp_name, "safety")

    # Check if already done
    for try_slug in [slug_ft, slug]:
        util_dir = os.path.join(safety_root, "main", "prism", "utility", try_slug)
        if os.path.exists(util_dir):
            geval_done = os.path.exists(os.path.join(util_dir, "geval_results.json"))
            wr_done = os.path.exists(os.path.join(util_dir, "winrate_vs_base.json"))
            kl_done = os.path.exists(os.path.join(util_dir, "kl_divergence.json"))
            if geval_done and wr_done and kl_done:
                logger.info(f"[SKIP] Utility done: {util_dir}")
                return

    cmd = [sys.executable, "scripts/eval/eval_safety.py",
           "--base_model", model_name,
           "--adapter_path", adapter_path,
           "--judge_model", model_name,
           "--skip_safety",
           "--output_root", safety_root,
           "--experiment_type", "main",
           "--experiment_name", "prism",
           "--utility_limit", "100"]
    logger.info(f"  Utility: prism")
    env = os.environ.copy()
    env["PYTHONPATH"] = "scripts" + os.pathsep + env.get("PYTHONPATH", "")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Utility eval failed: {e}")


def _run_mmlu(exp_name, setting, model_name, adapter_path=None, system_prompt=None):
    """Run MMLU via lm_eval. Results go to results/{exp_name}/{setting}/mmlu/.
    
    Args:
        system_prompt: Path to a text file containing the system instruction.
                       Used with --system_instruction and --apply_chat_template
                       so that persona-specific MMLU scores differ.
    """
    out = result_dir(exp_name, setting, "mmlu")

    # Check if already done (lm_eval writes results in subdirectories)
    def _mmlu_done(d):
        if not os.path.exists(d):
            return False
        if os.path.exists(os.path.join(d, "mmlu_summary.json")):
            return True
        # lm_eval saves as {model_slug}/results_*.json
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.startswith("results_") and f.endswith(".json"):
                    return True
        return False

    if _mmlu_done(out):
        logger.info(f"[SKIP] MMLU done: {out}")
        return

    model_args = f"pretrained={model_name},trust_remote_code=True"
    if adapter_path:
        model_args += f",peft={adapter_path}"

    cmd = [sys.executable, "-m", "lm_eval",
           "--model", "hf",
           "--model_args", model_args,
           "--tasks", "mmlu",
           "--batch_size", "auto",
           "--output_path", out]
    
    # Apply persona system prompt if provided
    if system_prompt:
        context_text = load_text(system_prompt) if os.path.exists(system_prompt) else system_prompt
        cmd += ["--system_instruction", context_text,
                "--apply_chat_template"]
    
    logger.info(f"  MMLU: {setting}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"MMLU eval failed: {e}")


# ============================================================
# Full evaluation (all paper table rows)
# ============================================================

def run_evaluation(base_model, adapter_path, exp_name):
    """
    Run full evaluation to populate all paper table entries.

    Table 2 rows (per model):
      Row 1 — Baseline:        base model, no adapter, no system prompt
      Row 2 — Avg Persona:     mean across all persona prompts
      Row 3 — Best Persona:    best single persona (oracle)
      Row 4 — Matched Persona:
        - MT-Bench 8 cats:     persona_X on category X (task personas)
        - Safety benchmarks:   safety_monitor persona
        - MMLU 4 domains:      helpful persona
      Row 5 — PRISM:           base + adapter, no system prompt

    Each row needs MT-Bench (8 cats + avg) + Safety (HB, JB, PKU) + MMLU.
    PRISM row also needs Utility (G-Eval, Win Rate, KL).

    All results go under results/{exp_name}/.
    """
    logger.info(f"\n{'#'*70}")
    logger.info(f"  EVALUATION — {exp_name}")
    logger.info(f"  Base model:  {base_model}")
    logger.info(f"  Adapter:     {adapter_path}")
    logger.info(f"  Results:     {RESULTS_ROOT}/{exp_name}/")
    logger.info(f"{'#'*70}\n")

    # Merge all personas into one dict for uniform iteration
    ALL_PERSONAS = {}
    ALL_PERSONAS.update(TASK_PERSONAS)
    ALL_PERSONAS.update(SAFETY_PERSONA)
    ALL_PERSONAS.update(MMLU_PERSONA)

    # ---- Row 1: Baseline ----
    logger.info("=== Row 1: Baseline ===")
    _run_mt_bench(exp_name, "baseline", base_model)
    _run_safety(exp_name, "baseline", base_model)
    _run_mmlu(exp_name, "baseline", base_model)

    # ---- Rows 2-4: Every persona gets all 3 evaluations ----
    for persona_name, context_path in ALL_PERSONAS.items():
        setting = f"persona/{persona_name}"
        logger.info(f"\n=== Persona: {persona_name} ===")
        _run_mt_bench(exp_name, setting, base_model, system_prompt=context_path)
        _run_safety(exp_name, setting, base_model, context_file=context_path)
        _run_mmlu(exp_name, setting, base_model, system_prompt=context_path)

    # ---- Row 5: PRISM ----
    logger.info("\n=== Row 5: PRISM ===")
    _run_mt_bench(exp_name, "prism", base_model, adapter_path=adapter_path)
    _run_safety(exp_name, "prism", base_model, adapter_path=adapter_path)
    _run_mmlu(exp_name, "prism", base_model, adapter_path=adapter_path)
    _run_utility(exp_name, base_model, adapter_path)

    # ---- Collect & print results ----
    summary = _collect_results(exp_name, base_model)
    save_json(summary, os.path.join(RESULTS_ROOT, exp_name, "full_summary.json"))
    _print_paper_table(exp_name, summary)

    return summary


def _collect_results(exp_name, base_model):
    """Collect all eval results from canonical paths into a single summary.

    Every persona row has mt_bench + safety + mmlu.
    Avg/Best/Matched are computed across all personas for all 3 metrics.
    """
    slug = get_model_slug(base_model)
    slug_ft = slug + "_finetuned"
    summary = {"model": base_model, "exp_name": exp_name, "rows": {}}
    safety_root = os.path.join(RESULTS_ROOT, exp_name, "safety", "main")

    # Helper: load safety data for a given setting_key
    def _load_safety(setting_key, search_slugs=None):
        if search_slugs is None:
            search_slugs = [slug]
        safety_data = {}
        for bm in SAFETY_BENCHMARKS:
            for try_slug in search_slugs:
                bm_summary = os.path.join(safety_root, setting_key, bm, try_slug, "summary.json")
                if os.path.exists(bm_summary):
                    safety_data[bm] = load_json(bm_summary)
                    break
        return safety_data

    # Helper: load MMLU data
    def _load_mmlu(mmlu_dir):
        if not os.path.exists(mmlu_dir):
            return {}
        sp = os.path.join(mmlu_dir, "mmlu_summary.json")
        if os.path.exists(sp):
            return load_json(sp)
        # lm_eval saves under model-slug subdirectory: mmlu/{slug}/results_*.json
        for root, dirs, files in os.walk(mmlu_dir):
            for f in files:
                if f.startswith("results_") and f.endswith(".json"):
                    return load_json(os.path.join(root, f))
        # Fallback: any json with mmlu in name
        for f in os.listdir(mmlu_dir):
            if f.endswith(".json") and "mmlu" in f.lower():
                return load_json(os.path.join(mmlu_dir, f))
        return {}

    # ---- Row 1: Baseline (all 3 evals) ----
    summary["rows"]["baseline"] = {
        "mt_bench": _load_mt_bench_summary(exp_name, "baseline"),
        "safety":   _load_safety("baseline"),
        "mmlu":     _load_mmlu(os.path.join(RESULTS_ROOT, exp_name, "baseline", "mmlu")),
    }

    # ---- Per-persona rows (all 3 evals for each) ----
    all_persona_names = list(TASK_PERSONAS.keys()) + list(SAFETY_PERSONA.keys()) + list(MMLU_PERSONA.keys())
    for persona in all_persona_names:
        setting = f"persona/{persona}"
        setting_key = f"persona_{persona}"
        summary["rows"][setting_key] = {
            "mt_bench": _load_mt_bench_summary(exp_name, setting),
            "safety":   _load_safety(setting_key),
            "mmlu":     _load_mmlu(os.path.join(RESULTS_ROOT, exp_name, setting, "mmlu")),
        }

    # ---- Compute avg/best/matched across all personas (all 3 metrics) ----
    summary["rows"]["avg_persona"] = {
        "mt_bench": _avg_persona_mt(summary),
        "safety":   _avg_persona_safety(summary),
        "mmlu":     _avg_persona_mmlu(summary),
    }
    summary["rows"]["best_persona"] = {
        "mt_bench": _best_persona_mt(summary),
        "safety":   _best_persona_safety(summary),
        "mmlu":     _best_persona_mmlu(summary),
    }
    summary["rows"]["matched_persona"] = {
        "mt_bench": _matched_persona_mt(summary),
        "safety":   _matched_persona_safety(summary),
        "mmlu":     _matched_persona_mmlu(summary),
    }

    # ---- Row 5: PRISM (all 3 evals + utility) ----
    summary["rows"]["prism"] = {
        "mt_bench": _load_mt_bench_summary(exp_name, "prism"),
        "safety":   _load_safety("prism", search_slugs=[slug_ft, slug]),
        "mmlu":     _load_mmlu(os.path.join(RESULTS_ROOT, exp_name, "prism", "mmlu")),
    }

    # Utility (PRISM only)
    for try_slug in [slug_ft, slug]:
        util_dir = os.path.join(safety_root, "prism", "utility", try_slug)
        if os.path.exists(util_dir):
            break
    else:
        util_dir = os.path.join(safety_root, "prism", "utility", slug)
    for name in ["geval_results", "winrate_vs_base", "kl_divergence"]:
        path = os.path.join(util_dir, f"{name}.json")
        if os.path.exists(path):
            summary["rows"]["prism"].setdefault("utility", {})[name] = load_json(path)

    return summary


def _load_mt_bench_summary(exp_name, setting):
    """Load MT-Bench summary from canonical path."""
    path = os.path.join(RESULTS_ROOT, exp_name, setting, "mt_bench", "mt_bench_summary.json")
    if os.path.exists(path):
        return load_json(path)
    return {}


_CATS = ["writing", "roleplay", "reasoning", "math", "coding",
         "extraction", "stem", "humanities"]


def _get_persona_rows(summary):
    """Return only persona_* rows from the summary."""
    return {k: v for k, v in summary["rows"].items() if k.startswith("persona_")}


# ---- MT-Bench aggregation ----

def _avg_persona_mt(summary):
    """Average MT-Bench across all persona rows per category."""
    persona_rows = _get_persona_rows(summary)
    if not persona_rows:
        return {}
    result = {}
    for cat in _CATS:
        vals = [r.get("mt_bench", {}).get("per_category", {}).get(cat, 0)
                for r in persona_rows.values()]
        result[cat] = round(sum(vals)/len(vals), 2) if vals else 0
    all_v = list(result.values())
    result["avg"] = round(sum(all_v)/len(all_v), 2) if all_v else 0
    return {"per_category": result, "overall": result["avg"]}


def _best_persona_mt(summary):
    """Best MT-Bench persona per category."""
    persona_rows = _get_persona_rows(summary)
    if not persona_rows:
        return {}
    result = {}
    for cat in _CATS:
        vals = [r.get("mt_bench", {}).get("per_category", {}).get(cat, 0)
                for r in persona_rows.values()]
        result[cat] = round(max(vals), 2) if vals else 0
    all_v = list(result.values())
    result["avg"] = round(sum(all_v)/len(all_v), 2) if all_v else 0
    return {"per_category": result, "overall": result["avg"]}


def _matched_persona_mt(summary):
    """Persona_X score on category X (diagonal)."""
    result = {}
    for cat in _CATS:
        key = f"persona_{cat}"
        if key in summary["rows"]:
            result[cat] = summary["rows"][key].get("mt_bench", {}).get(
                "per_category", {}).get(cat, 0)
        else:
            result[cat] = 0
    all_v = list(result.values())
    result["avg"] = round(sum(all_v)/len(all_v), 2) if all_v else 0
    return {"per_category": result, "overall": result["avg"]}


# ---- Safety aggregation (per-benchmark, like MT-Bench per-category) ----

def _extract_safety_per_benchmark(safety_data):
    """Extract per-benchmark safety scores {benchmark: {mean, std_error}}."""
    if not safety_data:
        return {}
    result = {}
    for bm, data in safety_data.items():
        scores = data.get("safety_scores", {})
        # Prefer base_with_context for persona rows, finetuned for PRISM, base_no_context for baseline
        for cond in ["base_with_context", "finetuned", "base_no_context"]:
            if cond in scores:
                result[bm] = scores[cond]
                break
    return result


def _avg_persona_safety(summary):
    """Average safety across all personas, per benchmark."""
    persona_rows = _get_persona_rows(summary)
    if not persona_rows:
        return {}
    # Collect per-benchmark scores from each persona
    all_scores = {}  # {benchmark: [mean1, mean2, ...]}
    for rdata in persona_rows.values():
        per_bm = _extract_safety_per_benchmark(rdata.get("safety", {}))
        for bm, metrics in per_bm.items():
            all_scores.setdefault(bm, []).append(metrics.get("mean", 0))
    # Average per benchmark
    result = {}
    for bm, vals in all_scores.items():
        result[bm] = {
            "safety_rate": round(sum(vals) / len(vals), 4),
            "std_error": 0,  # SE of average is complex, leave as 0
        }
    return result


def _best_persona_safety(summary):
    """Best (highest) safety per benchmark (oracle)."""
    persona_rows = _get_persona_rows(summary)
    if not persona_rows:
        return {}
    all_scores = {}  # {benchmark: [(mean, persona_name), ...]}
    for pname, rdata in persona_rows.items():
        per_bm = _extract_safety_per_benchmark(rdata.get("safety", {}))
        for bm, metrics in per_bm.items():
            all_scores.setdefault(bm, []).append(metrics.get("mean", 0))
    result = {}
    for bm, vals in all_scores.items():
        result[bm] = {
            "safety_rate": round(max(vals), 4),
            "std_error": 0,
        }
    return result


def _matched_persona_safety(summary):
    """Safety from safety_monitor persona (matched row for safety)."""
    row = summary["rows"].get("persona_safety_monitor", {})
    return row.get("safety", {})


# ---- MMLU aggregation (per-domain, like MT-Bench per-category) ----

_MMLU_DOMAINS = ["mmlu_stem", "mmlu_humanities", "mmlu_social_sciences", "mmlu_other"]

def _extract_mmlu_per_domain(mmlu_data):
    """Extract per-domain MMLU scores {domain: acc}."""
    if not mmlu_data:
        return {}
    # lm_eval stores in groups or results
    groups = mmlu_data.get("groups", mmlu_data.get("results", {}))
    result = {}
    for domain in _MMLU_DOMAINS:
        if domain in groups:
            result[domain] = groups[domain].get("acc,none", 0)
    # Overall
    if "mmlu" in groups:
        result["mmlu"] = groups["mmlu"].get("acc,none", 0)
    return result


def _avg_persona_mmlu(summary):
    """Average MMLU across all personas, per domain."""
    persona_rows = _get_persona_rows(summary)
    if not persona_rows:
        return {}
    all_scores = {}  # {domain: [acc1, acc2, ...]}
    for rdata in persona_rows.values():
        per_dom = _extract_mmlu_per_domain(rdata.get("mmlu", {}))
        for dom, acc in per_dom.items():
            all_scores.setdefault(dom, []).append(acc)
    result = {"groups": {}}
    for dom, vals in all_scores.items():
        result["groups"][dom] = {"acc,none": round(sum(vals) / len(vals), 4)}
    return result


def _best_persona_mmlu(summary):
    """Best MMLU per domain (oracle)."""
    persona_rows = _get_persona_rows(summary)
    if not persona_rows:
        return {}
    all_scores = {}
    for rdata in persona_rows.values():
        per_dom = _extract_mmlu_per_domain(rdata.get("mmlu", {}))
        for dom, acc in per_dom.items():
            all_scores.setdefault(dom, []).append(acc)
    result = {"groups": {}}
    for dom, vals in all_scores.items():
        result["groups"][dom] = {"acc,none": round(max(vals), 4)}
    return result


def _matched_persona_mmlu(summary):
    """MMLU matched persona: diagonal matching like MT-Bench.
    
    mmlu_stem → persona_stem, mmlu_humanities → persona_humanities,
    mmlu_social_sciences → persona_helpful (no direct match),
    mmlu_other → persona_helpful (no direct match).
    """
    _MMLU_PERSONA_MAP = {
        "mmlu_stem": "persona_stem",
        "mmlu_humanities": "persona_humanities",
        "mmlu_social_sciences": "persona_helpful",
        "mmlu_other": "persona_helpful",
    }
    result = {"groups": {}}
    for domain, persona_key in _MMLU_PERSONA_MAP.items():
        row = summary["rows"].get(persona_key, {})
        per_dom = _extract_mmlu_per_domain(row.get("mmlu", {}))
        if domain in per_dom:
            result["groups"][domain] = {"acc,none": per_dom[domain]}
    # Overall = average of matched domain scores
    domain_vals = [v["acc,none"] for v in result["groups"].values()]
    if domain_vals:
        result["groups"]["mmlu"] = {"acc,none": round(sum(domain_vals) / len(domain_vals), 4)}
    return result


def _print_paper_table(exp_name, summary):
    """Print results formatted for the paper table."""
    logger.info(f"\n{'='*70}")
    logger.info(f"PAPER TABLE RESULTS — {exp_name}")
    logger.info(f"{'='*70}")

    cats = _CATS + ["avg"]
    header = f"{'Row':<20s} " + " ".join(f"{c:>7s}" for c in cats)
    logger.info(header)
    logger.info("-" * len(header))

    for row_name in ["baseline", "avg_persona", "best_persona", "matched_persona", "prism"]:
        row = summary["rows"].get(row_name, {})
        mt = row.get("mt_bench", {})
        pc = mt.get("per_category", {})
        overall = mt.get("overall", 0)

        vals = []
        for c in _CATS:
            v = pc.get(c, 0)
            vals.append(f"{v:7.2f}" if v else "     --")
        vals.append(f"{overall:7.2f}" if overall else "     --")
        logger.info(f"{row_name:<20s} {' '.join(vals)}")

    # Safety
    logger.info(f"\n{'Safety Results':^70}")
    for row_name in ["baseline", "avg_persona", "best_persona", "matched_persona", "prism"]:
        safety = summary["rows"].get(row_name, {}).get("safety", {})
        if not safety:
            logger.info(f"  Safety/{row_name}: --")
            continue
        # If it's an aggregated row (avg/best), show the single metric
        if "avg_refusal_rate" in safety:
            logger.info(f"  Safety/{row_name}: RR={safety['avg_refusal_rate']:.1%}")
        elif "best_refusal_rate" in safety:
            logger.info(f"  Safety/{row_name}: RR={safety['best_refusal_rate']:.1%}")
        else:
            # Full per-benchmark data
            for bm, data in safety.items():
                scores = data.get("safety_scores", {})
                for condition, metrics in scores.items():
                    logger.info(f"  Safety/{row_name}/{bm}/{condition}: "
                                f"RR={metrics.get('mean', 0):.1%}")

    # MMLU
    logger.info(f"\n{'MMLU Results':^70}")
    for row_name in ["baseline", "avg_persona", "best_persona", "matched_persona", "prism"]:
        mmlu = summary["rows"].get(row_name, {}).get("mmlu", {})
        if not mmlu:
            logger.info(f"  MMLU/{row_name}: --")
        elif "avg_accuracy" in mmlu:
            logger.info(f"  MMLU/{row_name}: Acc={mmlu['avg_accuracy']:.2%}")
        elif "best_accuracy" in mmlu:
            logger.info(f"  MMLU/{row_name}: Acc={mmlu['best_accuracy']:.2%}")
        else:
            logger.info(f"  MMLU/{row_name}: {mmlu}")

    # Utility (PRISM only)
    logger.info(f"\n{'Utility Results':^70}")
    util = summary["rows"].get("prism", {}).get("utility", {})
    if "geval_results" in util:
        g = util["geval_results"]
        logger.info(f"  G-Eval: " + ", ".join(
            f"{k}={v['mean']:.2f}" for k, v in g.items() if isinstance(v, dict)))
    if "winrate_vs_base" in util:
        logger.info(f"  Win Rate vs Base: {util['winrate_vs_base'].get('win_rate', 0):.1f}%")
    if "kl_divergence" in util:
        logger.info(f"  KL Drift: {util['kl_divergence'].get('mean', 0):.4f}")


# ============================================================
# Main iterative loop
# ============================================================

def run_iterative(base_model, exp_name, rounds, epochs_per_round,
                  retain_weight=0.5, learning_rate=2e-4, lora_r=32, lora_alpha=64,
                  micro_batch=2, grad_accum=8, max_len=1024, temperature=2.0,
                  skip_eval=False):
    """
    Main iterative loop (same LoRA throughout):
      Round 1: base model → Stage 2 → fresh LoRA → train 2 epochs → save adapter
      Round 2: base + adapter → Stage 2 → resume LoRA → train 2 more → save
      ...
      Round R: final adapter has trained R × epochs_per_round total epochs
      Then: full evaluation

    All directories are keyed by exp_name:
      dataset/synthetic/persona_prism/{exp_name}/           ← queries (Stage 1)
      dataset/synthetic/persona_prism/{exp_name}/round_N/   ← per-round data
      models/persona_prism/{exp_name}/                      ← LoRA adapter
      results/{exp_name}/...                                ← all eval results
    """
    data_dir = os.path.join(DATA_ROOT, exp_name)
    adapter_dir = os.path.join(ADAPTER_ROOT, exp_name)

    logger.info(f"\n{'#'*70}")
    logger.info(f"  PRISM ITERATIVE DISTILLATION")
    logger.info(f"  Experiment: {exp_name}")
    logger.info(f"  Model:      {base_model}")
    logger.info(f"  Rounds:     {rounds} × {epochs_per_round} epochs = {rounds * epochs_per_round} total")
    logger.info(f"  Data:       {data_dir}/")
    logger.info(f"  Adapter:    {adapter_dir}/")
    logger.info(f"  Results:    {RESULTS_ROOT}/{exp_name}/")
    logger.info(f"{'#'*70}\n")

    for r in range(1, rounds + 1):
        total_so_far = (r - 1) * epochs_per_round
        adapter_exists = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))

        logger.info(f"\n{'#'*70}")
        logger.info(f"  ROUND {r}/{rounds}  "
                     f"|  Epochs: {total_so_far} → {total_so_far + epochs_per_round}")
        logger.info(f"  Base:    {base_model}")
        logger.info(f"  Adapter: {adapter_dir} ({'resume' if adapter_exists else 'fresh'})")
        logger.info(f"  Data:    {data_dir}/round_{r}/")
        logger.info(f"{'#'*70}\n")

        # ---- Stage 2: Generate data from current model ----
        if adapter_exists:
            model, tokenizer = load_model(base_model, adapter_dir)
        else:
            model, tokenizer = load_model(base_model)

        round_data_dir = run_stage2_for_round(model, tokenizer, data_dir, r)
        unload_model(model, tokenizer)

        # ---- Stage 3: Train (fresh LoRA on round 1, resume on round 2+) ----
        stage3_train(
            model_name=base_model,
            data_dir=round_data_dir,
            output_dir=adapter_dir,
            adapter_path=adapter_dir if adapter_exists else None,
            epochs=epochs_per_round,
            retain_weight=retain_weight,
            learning_rate=learning_rate,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            micro_batch=micro_batch,
            grad_accum=grad_accum,
            max_len=max_len,
            temperature=temperature,
            save_every_epoch=False,
        )

        # Save round metadata
        save_json({
            "round": r,
            "exp_name": exp_name,
            "base_model": base_model,
            "adapter_dir": adapter_dir,
            "data_dir": round_data_dir,
            "epochs_this_round": epochs_per_round,
            "total_epochs": total_so_far + epochs_per_round,
        }, os.path.join(adapter_dir, f"round_{r}_meta.json"))

        logger.info(f"Round {r} complete. Adapter → {adapter_dir}")

        # ---- Cleanup: delete teacher logits (only needed during training) ----
        # These files are ~50GB per round and would fill the disk quota
        for logit_name in ["teacher_logits_distill.pt", "teacher_logits_retain.pt"]:
            logit_path = os.path.join(round_data_dir, logit_name)
            if os.path.exists(logit_path):
                os.remove(logit_path)
                logger.info(f"Cleaned up {logit_path}")
            # Also clean up .parts directory if it exists
            parts_path = logit_path + ".parts"
            if os.path.exists(parts_path):
                import shutil
                shutil.rmtree(parts_path, ignore_errors=True)
                logger.info(f"Cleaned up {parts_path}")

        logger.info(f"Round {r} complete.\n")

    total_epochs = rounds * epochs_per_round
    logger.info(f"\n{'='*70}")
    logger.info(f"Training complete: {rounds} rounds × {epochs_per_round} epochs "
                f"= {total_epochs} total")
    logger.info(f"{'='*70}")

    # Save training config
    save_json({
        "exp_name": exp_name,
        "base_model": base_model,
        "rounds": rounds,
        "epochs_per_round": epochs_per_round,
        "total_epochs": total_epochs,
        "adapter_dir": adapter_dir,
        "data_dir": data_dir,
        "retain_weight": retain_weight,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "temperature": temperature,
    }, os.path.join(adapter_dir, "iterative_config.json"))

    # ---- Full evaluation ----
    if not skip_eval:
        run_evaluation(base_model, adapter_dir, exp_name)
    else:
        logger.info("Evaluation skipped (--skip_eval)")


def main():
    parser = argparse.ArgumentParser(
        description="PRISM Iterative Self-Distillation",
        epilog="Usage:\n"
               "  python -m scripts.prism.run_iterative --config configs/test.json\n"
               "  python -m scripts.prism.run_iterative --config configs/Qwen2.5-7B-Instruct.json\n"
               "  python -m scripts.prism.run_iterative --config configs/test.json --skip_eval\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Path to JSON config file (e.g. configs/test.json)")
    # All of these override the config file if provided
    parser.add_argument("--model", default=None)
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--epochs_per_round", type=int, default=None)
    parser.add_argument("--retain_weight", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--micro_batch", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--skip_eval", action="store_true", default=None)
    args = parser.parse_args()

    # Load config file
    with open(args.config) as f:
        cfg = json.load(f)
    logger.info(f"Loaded config: {args.config}")

    # CLI overrides config
    def _get(key, default=None):
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            return cli_val
        return cfg.get(key, default)

    base_model = _get("model", DEFAULT_MODEL)
    exp_name = _get("exp_name") or get_model_slug(base_model)

    run_iterative(
        base_model=base_model,
        exp_name=exp_name,
        rounds=_get("rounds", ROUNDS),
        epochs_per_round=_get("epochs_per_round", EPOCHS_PER_ROUND),
        retain_weight=_get("retain_weight", 0.5),
        learning_rate=_get("learning_rate", 2e-4),
        lora_r=_get("lora_r", 32),
        lora_alpha=_get("lora_alpha", 64),
        micro_batch=_get("micro_batch", 2),
        grad_accum=_get("grad_accum", 8),
        max_len=_get("max_len", 1024),
        temperature=_get("temperature", 2.0),
        skip_eval=_get("skip_eval", False),
    )


if __name__ == "__main__":
    main()

