"""
Persona Granularity Ablation: Evaluate all personas × 3 granularity levels × 3 benchmarks.

Evaluates each of the 12 personas at 3 detail levels (full, half, min)
on all 3 benchmarks (MT-Bench, Safety, MMLU) using Qwen2.5-7B-Instruct.

This produces a comprehensive grid:
  12 personas × 3 granularity levels × 3 benchmarks = 108 evaluation runs
  + 1 baseline (no persona) × 3 benchmarks = 3 baseline runs
  Total: 111 evaluation runs

Results go to: results/{exp_name}/persona_granularity/{granularity}/{persona}/{benchmark}/

Usage:
  # Run everything (SLURM submission recommended):
  python -m scripts.prism.eval_persona_granularity

  # Single granularity level:
  python -m scripts.prism.eval_persona_granularity --granularity full

  # Single persona:
  python -m scripts.prism.eval_persona_granularity --persona math

  # Dry run:
  python -m scripts.prism.eval_persona_granularity --dry_run
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from utils import load_json, save_json, load_text, get_model_slug

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
RESULTS_ROOT = "results"

# All 12 personas
PERSONA_NAMES = [
    # 8 task personas
    "writing", "roleplay", "reasoning", "math", "coding",
    "extraction", "stem", "humanities",
    # 4 behavioral personas
    "critic", "safety_monitor", "helpful", "compliant",
]

# Granularity levels → directory containing persona files
GRANULARITY_DIRS = {
    "full": "dataset/personas/full_personas",
    "half": "dataset/personas/half_personas",
    "min":  "dataset/personas/min_personas",
}

# Safety benchmarks
SAFETY_BENCHMARKS = ["HarmBench", "Jailbreak", "PKU_SafeRLHF"]


# ============================================================
# MT-Bench helper (reused from run_iterative.py)
# ============================================================

def _mt_bench_done(out_dir):
    return os.path.exists(os.path.join(out_dir, "mt_bench_summary.json"))


def _summarize_mt_bench(judgment_file, summary_file):
    """Per-category averages from MT-Bench judgments → summary JSON."""
    if not os.path.exists(judgment_file):
        return
    cat_scores = defaultdict(list)
    with open(judgment_file) as f:
        for line in f:
            rec = json.loads(line)
            score = rec.get("score")
            cat = rec.get("category", "unknown")
            if isinstance(score, (int, float)) and score > 0:
                cat_scores[cat].append(score)
    summary = {}
    all_scores = []
    for cat, scores in sorted(cat_scores.items()):
        avg = round(sum(scores) / len(scores), 3)
        summary[cat] = {"avg": avg, "n": len(scores)}
        all_scores.extend(scores)
    if all_scores:
        summary["overall"] = {"avg": round(sum(all_scores) / len(all_scores), 3),
                              "n": len(all_scores)}
    save_json(summary, summary_file)
    logger.info(f"  MT-Bench summary: {summary.get('overall', {}).get('avg', 'N/A')}")


def run_mt_bench(out_dir, model_name, system_prompt=None, judge_model=None):
    """Generate + judge + summarize MT-Bench for one setting."""
    os.makedirs(out_dir, exist_ok=True)
    if _mt_bench_done(out_dir):
        logger.info(f"  [SKIP] MT-Bench done: {out_dir}")
        return

    answer_file = os.path.join(out_dir, "answers.jsonl")
    judgment_file = os.path.join(out_dir, "judgments.jsonl")
    summary_file = os.path.join(out_dir, "mt_bench_summary.json")
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
        if system_prompt:
            cmd += ["--system_prompt", system_prompt]
        logger.info(f"  MT-Bench generate → {out_dir}")
        subprocess.run(cmd, check=True)

    # Judge
    if not os.path.exists(judgment_file) and os.path.exists(answer_file):
        cmd = [sys.executable, "scripts/eval/eval_mt_bench.py",
               "--mode", "judge", "--judge_model", judge,
               "--question_file", question_file,
               "--answer_file", answer_file, "--output_file", judgment_file]
        logger.info(f"  MT-Bench judge → {out_dir}")
        subprocess.run(cmd, check=True)

    # Summarize
    _summarize_mt_bench(judgment_file, summary_file)


# ============================================================
# Safety helper
# ============================================================

def run_safety(out_dir, model_name, context_file=None):
    """Run safety eval for one setting."""
    os.makedirs(out_dir, exist_ok=True)
    slug = get_model_slug(model_name)

    # Check if already done
    all_done = all(
        os.path.exists(os.path.join(out_dir, bm, slug, "summary.json"))
        for bm in SAFETY_BENCHMARKS
    )
    if all_done:
        logger.info(f"  [SKIP] Safety done: {out_dir}")
        return

    cmd = [sys.executable, "scripts/eval/eval_safety.py",
           "--base_model", model_name,
           "--judge_model", model_name,
           "--benchmarks"] + SAFETY_BENCHMARKS + [
           "--output_root", out_dir,
           "--experiment_type", "main",
           "--experiment_name", "eval",
           "--skip_utility", "--skip_kl"]
    if context_file:
        cmd += ["--context_file", context_file]
    logger.info(f"  Safety → {out_dir}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "scripts" + os.pathsep + env.get("PYTHONPATH", "")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Safety eval failed: {e}")


# ============================================================
# MMLU helper
# ============================================================

def run_mmlu(out_dir, model_name, system_prompt=None):
    """Run MMLU for one setting."""
    os.makedirs(out_dir, exist_ok=True)

    # Check if done
    def _done(d):
        if os.path.exists(os.path.join(d, "mmlu_summary.json")):
            return True
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.startswith("results_") and f.endswith(".json"):
                    return True
        return False

    if _done(out_dir):
        logger.info(f"  [SKIP] MMLU done: {out_dir}")
        return

    model_args = f"pretrained={model_name},trust_remote_code=True"
    cmd = [sys.executable, "-m", "lm_eval",
           "--model", "hf",
           "--model_args", model_args,
           "--tasks", "mmlu",
           "--batch_size", "auto",
           "--output_path", out_dir]

    if system_prompt:
        context_text = load_text(system_prompt) if os.path.exists(system_prompt) else system_prompt
        cmd += ["--system_instruction", context_text,
                "--apply_chat_template"]

    logger.info(f"  MMLU → {out_dir}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"MMLU eval failed: {e}")


# ============================================================
# Main evaluation loop
# ============================================================

def run_persona_granularity_eval(model_name=DEFAULT_MODEL,
                                  exp_name=None,
                                  granularities=None,
                                  personas=None,
                                  benchmarks=None,
                                  dry_run=False):
    """Run the full persona granularity ablation.
    
    Grid: personas × granularity_levels × benchmarks
    """
    slug = get_model_slug(model_name)
    exp_name = exp_name or slug

    granularities = granularities or list(GRANULARITY_DIRS.keys())
    personas = personas or PERSONA_NAMES
    benchmarks = benchmarks or ["mt_bench", "safety", "mmlu"]

    # Count total runs
    total_runs = len(granularities) * len(personas) * len(benchmarks) + len(benchmarks)
    logger.info(f"\n{'='*70}")
    logger.info(f"PERSONA GRANULARITY ABLATION")
    logger.info(f"{'='*70}")
    logger.info(f"Model:         {model_name}")
    logger.info(f"Exp name:      {exp_name}")
    logger.info(f"Granularities: {granularities}")
    logger.info(f"Personas:      {len(personas)} ({', '.join(personas)})")
    logger.info(f"Benchmarks:    {benchmarks}")
    logger.info(f"Total runs:    {total_runs}")
    logger.info(f"Results root:  {RESULTS_ROOT}/{exp_name}/persona_granularity/")
    logger.info(f"{'='*70}\n")

    if dry_run:
        logger.info("DRY RUN — listing planned evaluations:")
        # Baseline
        for bm in benchmarks:
            base_dir = os.path.join(RESULTS_ROOT, exp_name, "persona_granularity", "baseline", bm)
            logger.info(f"  [baseline] {bm:10s} → {base_dir}")
        # Per-granularity × persona
        for gran in granularities:
            for persona in personas:
                persona_file = os.path.join(GRANULARITY_DIRS[gran], f"persona_{persona}.txt")
                for bm in benchmarks:
                    out = os.path.join(RESULTS_ROOT, exp_name, "persona_granularity",
                                       gran, persona, bm)
                    exists = "✓" if os.path.exists(out) else "○"
                    logger.info(f"  [{exists}] {gran:5s}/{persona:16s}/{bm:10s} "
                                f"← {persona_file}")
        return

    # ---- Baseline (no persona) ----
    logger.info("=== Baseline (no persona) ===")
    base_root = os.path.join(RESULTS_ROOT, exp_name, "persona_granularity", "baseline")
    if "mt_bench" in benchmarks:
        run_mt_bench(os.path.join(base_root, "mt_bench"), model_name)
    if "safety" in benchmarks:
        run_safety(os.path.join(base_root, "safety"), model_name)
    if "mmlu" in benchmarks:
        run_mmlu(os.path.join(base_root, "mmlu"), model_name)

    # ---- Per granularity × persona ----
    run_count = len(benchmarks)  # baseline already done
    for gran in granularities:
        gran_dir = GRANULARITY_DIRS[gran]
        logger.info(f"\n{'='*60}")
        logger.info(f"GRANULARITY: {gran} ({gran_dir})")
        logger.info(f"{'='*60}")

        for persona in personas:
            persona_file = os.path.join(gran_dir, f"persona_{persona}.txt")
            if not os.path.exists(persona_file):
                logger.warning(f"  Persona file not found: {persona_file}")
                continue

            out_root = os.path.join(RESULTS_ROOT, exp_name, "persona_granularity",
                                     gran, persona)

            logger.info(f"\n--- {gran}/{persona} ---")
            logger.info(f"  Persona file: {persona_file}")

            if "mt_bench" in benchmarks:
                run_mt_bench(os.path.join(out_root, "mt_bench"), model_name,
                             system_prompt=persona_file)
                run_count += 1

            if "safety" in benchmarks:
                run_safety(os.path.join(out_root, "safety"), model_name,
                           context_file=persona_file)
                run_count += 1

            if "mmlu" in benchmarks:
                run_mmlu(os.path.join(out_root, "mmlu"), model_name,
                         system_prompt=persona_file)
                run_count += 1

            logger.info(f"  Progress: {run_count}/{total_runs}")

    # ---- Collect & summarize results ----
    summary = collect_granularity_results(exp_name, personas, granularities, benchmarks)
    summary_path = os.path.join(RESULTS_ROOT, exp_name, "persona_granularity",
                                "granularity_summary.json")
    save_json(summary, summary_path)
    logger.info(f"\nResults summary → {summary_path}")
    print_summary_table(summary, personas, granularities)


# ============================================================
# Results collection
# ============================================================

def collect_granularity_results(exp_name, personas, granularities, benchmarks):
    """Collect all results into a structured summary."""
    summary = {
        "exp_name": exp_name,
        "personas": personas,
        "granularities": granularities,
        "benchmarks": benchmarks,
        "baseline": {},
        "results": {},  # [granularity][persona][benchmark] = scores
    }

    # Baseline
    base_root = os.path.join(RESULTS_ROOT, exp_name, "persona_granularity", "baseline")
    summary["baseline"]["mt_bench"] = _load_mt_bench_score(
        os.path.join(base_root, "mt_bench"))
    summary["baseline"]["safety"] = _load_safety_score(
        os.path.join(base_root, "safety"))
    summary["baseline"]["mmlu"] = _load_mmlu_score(
        os.path.join(base_root, "mmlu"))

    # Per granularity × persona
    for gran in granularities:
        summary["results"][gran] = {}
        for persona in personas:
            out_root = os.path.join(RESULTS_ROOT, exp_name, "persona_granularity",
                                     gran, persona)
            summary["results"][gran][persona] = {
                "mt_bench": _load_mt_bench_score(os.path.join(out_root, "mt_bench")),
                "safety": _load_safety_score(os.path.join(out_root, "safety")),
                "mmlu": _load_mmlu_score(os.path.join(out_root, "mmlu")),
            }

    return summary


def _load_mt_bench_score(mt_dir):
    """Load MT-Bench overall average from summary."""
    path = os.path.join(mt_dir, "mt_bench_summary.json")
    if not os.path.exists(path):
        return None
    data = load_json(path)
    overall = data.get("overall", {})
    result = {"overall": overall.get("avg")}
    # Also grab per-category
    for cat in ["writing", "roleplay", "reasoning", "math", "coding",
                "extraction", "stem", "humanities"]:
        if cat in data:
            result[cat] = data[cat].get("avg")
    return result


def _load_safety_score(safety_dir):
    """Load safety scores (ASR) from each benchmark."""
    result = {}
    for bm in SAFETY_BENCHMARKS:
        # Search for summary.json under the safety dir
        for root, dirs, files in os.walk(safety_dir):
            if bm in root and "summary.json" in files:
                data = load_json(os.path.join(root, "summary.json"))
                asr = data.get("attack_success_rate", data.get("asr"))
                result[bm] = asr
                break
    return result if result else None


def _load_mmlu_score(mmlu_dir):
    """Load MMLU accuracy from lm_eval results."""
    # Check for summary
    summary_path = os.path.join(mmlu_dir, "mmlu_summary.json")
    if os.path.exists(summary_path):
        data = load_json(summary_path)
        return data.get("acc") or data.get("accuracy") or data

    # Search for lm_eval output
    for root, dirs, files in os.walk(mmlu_dir):
        for f in files:
            if f.startswith("results_") and f.endswith(".json"):
                data = load_json(os.path.join(root, f))
                results = data.get("results", {})
                mmlu = results.get("mmlu", {})
                acc = mmlu.get("acc,none") or mmlu.get("acc")
                if acc is not None:
                    return {"accuracy": round(acc * 100, 2) if acc < 1 else round(acc, 2)}
    return None


# ============================================================
# Pretty-print summary
# ============================================================

def print_summary_table(summary, personas, granularities):
    """Print a concise results table."""
    print("\n" + "=" * 80)
    print("PERSONA GRANULARITY ABLATION RESULTS")
    print("=" * 80)

    # Baseline
    bl = summary.get("baseline", {})
    bl_mt = bl.get("mt_bench", {})
    bl_mmlu = bl.get("mmlu", {})
    print(f"\nBaseline (no persona):")
    print(f"  MT-Bench: {bl_mt.get('overall', 'N/A') if bl_mt else 'N/A'}")
    print(f"  MMLU:     {bl_mmlu.get('accuracy', 'N/A') if bl_mmlu else 'N/A'}")

    # Per granularity
    for gran in granularities:
        print(f"\n{'─'*60}")
        print(f"Granularity: {gran.upper()}")
        print(f"{'─'*60}")
        print(f"{'Persona':<18} {'MT-Bench':>10} {'MMLU':>10} {'Safety':>18}")
        print(f"{'─'*18} {'─'*10} {'─'*10} {'─'*18}")

        results = summary.get("results", {}).get(gran, {})
        mt_scores = []
        mmlu_scores = []

        for persona in personas:
            r = results.get(persona, {})
            mt = r.get("mt_bench", {})
            mmlu = r.get("mmlu", {})
            safety = r.get("safety", {})

            mt_val = mt.get("overall", "–") if mt else "–"
            mmlu_val = mmlu.get("accuracy", "–") if mmlu else "–"

            # Safety: show avg ASR across benchmarks
            if safety:
                asrs = [v for v in safety.values() if v is not None]
                safety_val = f"{sum(asrs)/len(asrs):.1f}%" if asrs else "–"
            else:
                safety_val = "–"

            print(f"{persona:<18} {str(mt_val):>10} {str(mmlu_val):>10} {safety_val:>18}")

            if isinstance(mt_val, (int, float)):
                mt_scores.append(mt_val)
            if isinstance(mmlu_val, (int, float)):
                mmlu_scores.append(mmlu_val)

        # Averages
        if mt_scores:
            avg_mt = round(sum(mt_scores) / len(mt_scores), 3)
            print(f"{'AVG':<18} {avg_mt:>10.3f}", end="")
        if mmlu_scores:
            avg_mmlu = round(sum(mmlu_scores) / len(mmlu_scores), 2)
            print(f" {avg_mmlu:>10.2f}", end="")
        print()

    print("=" * 80)


# ============================================================
# SLURM job script generator
# ============================================================

def generate_slurm_script(model_name=DEFAULT_MODEL, exp_name=None):
    """Generate a SLURM job script for the full ablation."""
    slug = get_model_slug(model_name)
    exp_name = exp_name or slug

    script = f"""#!/bin/bash
#SBATCH --job-name=persona_gran
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/persona_granularity_%j.out
#SBATCH --error=logs/persona_granularity_%j.err

set -e

cd /project2/jessetho_1732/zizhaoh/PRISM
module load conda
module load cuda/12.4.0
source activate DREAM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

echo "=========================================="
echo "PERSONA GRANULARITY ABLATION"
echo "Model: {model_name}"
echo "=========================================="

python -m scripts.prism.eval_persona_granularity \\
    --model {model_name} \\
    --exp_name {exp_name}

echo "=========================================="
echo "DONE: $(date)"
echo "Results: results/{exp_name}/persona_granularity/"
echo "=========================================="
"""
    script_path = f"job_persona_granularity.sh"
    with open(script_path, "w", newline="\n") as f:
        f.write(script)
    logger.info(f"SLURM script → {script_path}")
    logger.info(f"Submit with: sbatch {script_path}")
    return script_path


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Persona Granularity Ablation: "
                    "12 personas × 3 levels (full/half/min) × 3 benchmarks")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Base model (default: {DEFAULT_MODEL})")
    parser.add_argument("--exp_name", default=None,
                        help="Experiment name (default: model slug)")
    parser.add_argument("--granularity", nargs="*", default=None,
                        choices=["full", "half", "min"],
                        help="Granularity levels to evaluate (default: all)")
    parser.add_argument("--persona", nargs="*", default=None,
                        help="Specific personas to evaluate (default: all 12)")
    parser.add_argument("--benchmark", nargs="*", default=None,
                        choices=["mt_bench", "safety", "mmlu"],
                        help="Benchmarks to run (default: all 3)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print planned runs without executing")
    parser.add_argument("--gen_slurm", action="store_true",
                        help="Generate SLURM script and exit")
    args = parser.parse_args()

    if args.gen_slurm:
        generate_slurm_script(args.model, args.exp_name)
        return

    run_persona_granularity_eval(
        model_name=args.model,
        exp_name=args.exp_name,
        granularities=args.granularity,
        personas=args.persona,
        benchmarks=args.benchmark,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
