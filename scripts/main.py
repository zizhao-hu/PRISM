"""
PRISM Main Experiment Pipeline
====================================

Generates all results needed for the paper's main table.

Matrix:
  6 Models:
    - Qwen/Qwen2.5-7B-Instruct
    - mistralai/Mistral-7B-Instruct-v0.3
    - meta-llama/Llama-3.1-8B-Instruct
    - Qwen/Qwen1.5-MoE-A2.7B-Chat
    - deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

  12 Personas (8 task-specific + 4 behavioral):
    Task:       writing, roleplay, reasoning, math, coding, extraction, stem, humanities
    Behavioral: critic, safety_monitor, helpful, compliant

  3 Evaluation Types:
    - mt_bench : MT-Bench (8 categories, 10 questions each → 80 scores)
    - mmlu     : MMLU (4 domains: STEM, Humanities, Social Sciences, Other)
    - safety   : Refusal Rate on HarmBench, JailbreakBench, PKU-SafeRLHF

  5 Table Rows (per model):
    Row 1 — Baseline:  No system prompt, no adapter
    Row 2 — Avg Persona:   Average across all 12 persona prompts
    Row 3 — Best Persona:  Best single persona (oracle upper bound)
    Row 4 — Matched Persona: Task→persona routing (diagonal of 8 task personas)
    Row 5 — PRISM:     Distilled model (trained on all persona data), eval without context

Results layout:
  results/{model_slug}/baseline/{mt_bench,mmlu,safety}/
  results/{model_slug}/persona/{persona_name}/{mt_bench,mmlu,safety}/
  results/{model_slug}/prism/{mt_bench,mmlu,safety}/

Usage:
  python scripts/main.py --model Qwen/Qwen2.5-7B-Instruct          # run all
  python scripts/main.py --model Qwen/Qwen2.5-7B-Instruct --row 1  # baseline only
  python scripts/main.py --model Qwen/Qwen2.5-7B-Instruct --row 2  # all personas
  python scripts/main.py --model Qwen/Qwen2.5-7B-Instruct --row 5  # PRISM train+eval
  python scripts/main.py --all_models --row 1                       # baseline for all 6
  python scripts/main.py --collect                                  # print table
"""

import argparse
import json
import os
import sys
import subprocess
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]

# 8 task-specific personas (aligned with MT-Bench categories)
TASK_PERSONAS = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]

# 4 behavioral personas (not aligned with any specific task)
BEHAVIORAL_PERSONAS = ["critic", "safety_monitor", "helpful", "compliant"]

# All 12 personas
ALL_PERSONAS = TASK_PERSONAS + BEHAVIORAL_PERSONAS

PERSONA_FILES = {p: f"dataset/personas/persona_{p}.txt" for p in ALL_PERSONAS}

MT_BENCH_CATEGORIES = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]

SAFETY_BENCHMARKS = {
    "HarmBench": "dataset/eval/HarmBench/harmful_behaviors.csv",
    "JailbreakBench": "dataset/eval/JailbreakBench/harmful_behaviors.csv",
    "PKU-SafeRLHF": "dataset/eval/PKU-SafeRLHF/test.jsonl",
}

RESULTS_ROOT = "results"


# ============================================================
# Path helpers — single source of truth
# ============================================================

def get_model_slug(model_name):
    return model_name.split("/")[-1]


def result_dir(model_name, setting, eval_type):
    """Canonical path: results/{slug}/{setting}/{eval_type}/"""
    d = os.path.join(RESULTS_ROOT, get_model_slug(model_name), setting, eval_type)
    os.makedirs(d, exist_ok=True)
    return d


def adapter_dir(model_name):
    """Canonical adapter path: models/persona_prism/{slug}/"""
    return os.path.join("models", "persona_prism", get_model_slug(model_name))


# ============================================================
# Completion checks — skip-if-done logic
# ============================================================

def _mt_bench_done(out):
    return os.path.exists(os.path.join(out, "mt_bench_summary.json"))


def _mmlu_done(out):
    return os.path.exists(os.path.join(out, "mmlu_summary.json")) or \
           any(f.endswith(".json") for f in os.listdir(out) if "mmlu" in f.lower()) if os.path.exists(out) else False


def _safety_done(out):
    if not os.path.exists(out):
        return False
    return all(os.path.exists(os.path.join(out, f"{b}.json")) for b in SAFETY_BENCHMARKS)


# ============================================================
# MT-Bench evaluation (generate + judge → summary)
# ============================================================

def run_mt_bench(model_name, setting, system_prompt=None, adapter_path=None,
                 judge_model="Qwen/Qwen2.5-7B-Instruct"):
    """Run MT-Bench: generate answers, judge them, save summary."""
    out = result_dir(model_name, setting, "mt_bench")
    if _mt_bench_done(out):
        logger.info(f"[SKIP] MT-Bench done: {out}")
        return

    answer_file = os.path.join(out, "answers.jsonl")
    judgment_file = os.path.join(out, "judgments.jsonl")
    summary_file = os.path.join(out, "mt_bench_summary.json")

    # Generate
    if not os.path.exists(answer_file):
        cmd = [sys.executable, "scripts/eval/eval_mt_bench.py",
               "--mode", "generate",
               "--model", model_name,
               "--question_file", "dataset/eval/mt_bench/question.jsonl",
               "--output_file", answer_file]
        if system_prompt:
            cmd += ["--system_prompt", system_prompt]
        if adapter_path:
            cmd += ["--adapter_path", adapter_path]
        logger.info(f"Generating MT-Bench answers: {setting}")
        subprocess.run(cmd, check=True)

    # Judge
    if not os.path.exists(judgment_file):
        cmd = [sys.executable, "scripts/eval/eval_mt_bench.py",
               "--mode", "judge",
               "--judge_model", judge_model,
               "--question_file", "dataset/eval/mt_bench/question.jsonl",
               "--answer_file", answer_file,
               "--output_file", judgment_file]
        logger.info(f"Judging MT-Bench: {setting}")
        subprocess.run(cmd, check=True)

    # Summarize
    _summarize_mt_bench(judgment_file, summary_file, model_name, judge_model)


# ============================================================
# MMLU evaluation
# ============================================================

def run_mmlu(model_name, setting, adapter_path=None):
    """Run MMLU via lm_eval."""
    out = result_dir(model_name, setting, "mmlu")
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
    logger.info(f"Running MMLU: {setting}")
    subprocess.run(cmd, check=True)


# ============================================================
# Safety evaluation (3 benchmarks)
# ============================================================

def run_safety(model_name, setting, context_path=None, adapter_path=None):
    """Run safety eval on 3 benchmarks."""
    out = result_dir(model_name, setting, "safety")
    if _safety_done(out):
        logger.info(f"[SKIP] Safety done: {out}")
        return

    for bench_name, bench_path in SAFETY_BENCHMARKS.items():
        result_file = os.path.join(out, f"{bench_name}.json")
        if os.path.exists(result_file):
            logger.info(f"[SKIP] {bench_name} exists: {result_file}")
            continue

        cmd = [sys.executable, "scripts/eval/eval_safety.py",
               "--model", model_name,
               "--benchmark", bench_path,
               "--output", result_file]
        if context_path:
            cmd += ["--context", context_path]
        else:
            cmd += ["--no_context"]
        if adapter_path:
            cmd += ["--adapter_path", adapter_path]
        logger.info(f"Running safety {bench_name}: {setting}")
        subprocess.run(cmd, check=True)


# ============================================================
# Row 1: Baseline (no context, no adapter)
# ============================================================

def run_row1_baseline(model_name, judge_model="Qwen/Qwen2.5-7B-Instruct"):
    """Row 1 — Baseline: eval base model without any system prompt."""
    slug = get_model_slug(model_name)
    logger.info(f"\n--- Row 1: Baseline ({slug}) ---")
    run_mt_bench(model_name, "baseline", judge_model=judge_model)
    run_mmlu(model_name, "baseline")
    run_safety(model_name, "baseline")


# ============================================================
# Row 2-4 data: All 12 personas (eval with each persona prompt)
# ============================================================

def run_all_personas(model_name, judge_model="Qwen/Qwen2.5-7B-Instruct"):
    """Rows 2-4 data: Evaluate with each of the 12 persona prompts.

    Results feed into:
      Row 2 — Avg Persona (mean across 12)
      Row 3 — Best Persona (max across 12)
      Row 4 — Matched Persona (diagonal: persona_X on category X)
    """
    slug = get_model_slug(model_name)
    logger.info(f"\n--- Rows 2-4: All Personas ({slug}) ---")

    for persona in ALL_PERSONAS:
        logger.info(f"\n  Persona: {persona}")
        context_path = PERSONA_FILES[persona]

        with open(context_path) as f:
            system_prompt = f.read().strip()

        setting = f"persona/{persona}"
        run_mt_bench(model_name, setting, system_prompt=system_prompt, judge_model=judge_model)
        run_mmlu(model_name, setting)  # MMLU doesn't use system prompt in lm_eval
        run_safety(model_name, setting, context_path=context_path)


# ============================================================
# Row 5: PRISM (train + eval without context)
# ============================================================

def run_row5_prism(model_name, num_samples=50, epochs=5, kl_weight=0.5,
                   judge_model="Qwen/Qwen2.5-7B-Instruct"):
    """Row 5 — PRISM: Full pipeline (train if needed) + eval without persona."""
    slug = get_model_slug(model_name)
    logger.info(f"\n--- Row 5: PRISM ({slug}) ---")

    ad = adapter_dir(model_name)

    # Train if adapter doesn't exist
    if not os.path.exists(os.path.join(ad, "adapter_config.json")):
        logger.info(f"PRISM adapter not found at {ad}. Running full pipeline...")

        # Stage 1: Query generation
        cmd = [sys.executable, "-m", "scripts.prism.stage1_query_gen",
               "--model", model_name,
               "--num_samples", str(num_samples)]
        logger.info(f"  Stage 1 (query gen)")
        subprocess.run(cmd, check=True)

        # Stage 2: Verify & recycle
        cmd = [sys.executable, "-m", "scripts.prism.stage2_verify_recycle",
               "--model", model_name]
        logger.info(f"  Stage 2 (verify/recycle)")
        subprocess.run(cmd, check=True)

        # Stage 3: Distillation training
        cmd = [sys.executable, "-m", "scripts.prism.stage3_distill",
               "--model", model_name,
               "--epochs", str(epochs),
               "--kl_weight", str(kl_weight)]
        logger.info(f"  Stage 3 (distill)")
        subprocess.run(cmd, check=True)
    else:
        logger.info(f"PRISM adapter found: {ad}")

    # Eval PRISM model (no system prompt, adapter loaded)
    run_mt_bench(model_name, "prism", adapter_path=ad, judge_model=judge_model)
    run_mmlu(model_name, "prism", adapter_path=ad)
    run_safety(model_name, "prism", adapter_path=ad)


# ============================================================
# Helpers
# ============================================================

def _summarize_mt_bench(judgment_file, summary_file, model_name,
                        judge_model="Qwen/Qwen2.5-7B-Instruct"):
    """Compute per-category averages from MT-Bench judgments."""
    if not os.path.exists(judgment_file):
        return

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

    per_category = {cat: round(np.mean(vals), 2) for cat, vals in scores.items()}
    overall = round(np.mean([v for vals in scores.values() for v in vals]), 4) if scores else 0

    summary = {
        "model": model_name,
        "judge": judge_model,
        "overall": overall,
        "per_category": per_category,
        "n_questions": sum(len(v) for v in scores.values()),
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"MT-Bench summary: overall={overall:.2f} ({summary_file})")


def _read_mt_bench(path):
    """Read MT-Bench summary → (per_category_dict, overall)."""
    if not os.path.exists(path):
        return None, None
    data = json.load(open(path))
    return data.get("per_category", {}), data.get("overall", 0)


def _read_safety(safety_dir):
    """Read safety results → {bench_name: refusal_rate}."""
    scores = {}
    if not os.path.exists(safety_dir):
        return scores
    for bench_name in SAFETY_BENCHMARKS:
        path = os.path.join(safety_dir, f"{bench_name}.json")
        if os.path.exists(path):
            data = json.load(open(path))
            if "safety_scores" in data:
                s = data["safety_scores"]
                for key in ["base_no_context", "finetuned_no_trigger", "refusal_rate"]:
                    if key in s:
                        val = s[key]
                        scores[bench_name] = val.get("mean", val) if isinstance(val, dict) else val
                        break
            elif "refusal_rate" in data:
                scores[bench_name] = data["refusal_rate"]
    return scores


# ============================================================
# Collector: Print formatted table
# ============================================================

def collect_results():
    """Collect all results and print the main table."""
    cats = MT_BENCH_CATEGORIES

    print("\n" + "=" * 140)
    print("MAIN TABLE RESULTS")
    print("=" * 140)

    for model_name in MODELS:
        slug = get_model_slug(model_name)
        root = os.path.join(RESULTS_ROOT, slug)

        if not os.path.exists(root):
            print(f"\n{slug}: [NO DATA]")
            continue

        print(f"\n{'━' * 140}")
        print(f"  {model_name}")
        print(f"{'━' * 140}")

        # ---- MT-Bench ----
        print(f"\n  MT-Bench (1-10 per category):")
        print(f"  {'Row':<22}" + " ".join(f"{c:>8}" for c in cats) + "  │  Overall")
        print(f"  {'─' * 22}" + "─" * (9 * len(cats)) + "──┼─────────")

        # Row 1: Baseline
        pc, ovr = _read_mt_bench(os.path.join(root, "baseline", "mt_bench", "mt_bench_summary.json"))
        if pc:
            print(f"  {'1. Baseline':<22}" + " ".join(f"{pc.get(c,0):8.2f}" for c in cats) + f"  │  {ovr:.2f}")
        else:
            print(f"  {'1. Baseline':<22} [missing]")

        # Rows 2-4: From persona data
        persona_data = {}  # persona → per_category dict
        for persona in ALL_PERSONAS:
            pc_p, _ = _read_mt_bench(
                os.path.join(root, "persona", persona, "mt_bench", "mt_bench_summary.json"))
            if pc_p:
                persona_data[persona] = pc_p

        n_personas = len(persona_data)
        if persona_data:
            # Row 2: Average across all personas
            avg_pc = {c: np.mean([persona_data[p].get(c, 0) for p in persona_data]) for c in cats}
            avg_ovr = np.mean(list(avg_pc.values()))
            print(f"  {'2. Avg Persona':<22}" + " ".join(f"{avg_pc[c]:8.2f}" for c in cats) +
                  f"  │  {avg_ovr:.2f}  ({n_personas} personas)")

            # Row 3: Best persona (highest overall)
            overalls = {p: np.mean([persona_data[p].get(c, 0) for c in cats]) for p in persona_data}
            best_p = max(overalls, key=overalls.get)
            best_vals = [persona_data[best_p].get(c, 0) for c in cats]
            print(f"  {'3. Best(' + best_p + ')':<22}" + " ".join(f"{v:8.2f}" for v in best_vals) +
                  f"  │  {overalls[best_p]:.2f}")

            # Row 4: Matched persona (task persona X → category X score)
            matched_vals = [persona_data.get(c, {}).get(c, 0) for c in cats]
            matched_ovr = np.mean(matched_vals)
            print(f"  {'4. Matched':<22}" + " ".join(f"{v:8.2f}" for v in matched_vals) +
                  f"  │  {matched_ovr:.2f}")
        else:
            for r in ["2. Avg Persona", "3. Best Persona", "4. Matched"]:
                print(f"  {r:<22} [missing — run --row 2]")

        # Row 5: PRISM
        pc, ovr = _read_mt_bench(os.path.join(root, "prism", "mt_bench", "mt_bench_summary.json"))
        if pc:
            print(f"  {'5. PRISM':<22}" + " ".join(f"{pc.get(c,0):8.2f}" for c in cats) + f"  │  {ovr:.2f}")
        else:
            print(f"  {'5. PRISM':<22} [missing — run --row 5]")

        # ---- Safety ----
        print(f"\n  Safety Refusal Rate:")
        print(f"  {'Row':<22}" + " ".join(f"{b:>15}" for b in SAFETY_BENCHMARKS) + "  │  Avg")
        print(f"  {'─' * 22}" + "─" * (16 * len(SAFETY_BENCHMARKS)) + "──┼─────────")

        for row_label, setting in [("1. Baseline", "baseline"), ("5. PRISM", "prism")]:
            ss = _read_safety(os.path.join(root, setting, "safety"))
            if ss:
                vals = [ss.get(b, 0) for b in SAFETY_BENCHMARKS]
                avg = np.mean(vals) if vals else 0
                print(f"  {row_label:<22}" + " ".join(f"{v:15.1%}" for v in vals) + f"  │  {avg:.1%}")
            else:
                print(f"  {row_label:<22} [missing]")

        # Persona safety (avg/best/matched)
        if persona_data:
            all_persona_safety = {}
            for persona in ALL_PERSONAS:
                ss = _read_safety(os.path.join(root, "persona", persona, "safety"))
                if ss:
                    all_persona_safety[persona] = ss

            if all_persona_safety:
                # Avg
                avg_s = {}
                for b in SAFETY_BENCHMARKS:
                    vals = [all_persona_safety[p].get(b, 0) for p in all_persona_safety if b in all_persona_safety[p]]
                    avg_s[b] = np.mean(vals) if vals else 0
                avg_all = np.mean(list(avg_s.values())) if avg_s else 0
                print(f"  {'2. Avg Persona':<22}" + " ".join(f"{avg_s.get(b,0):15.1%}" for b in SAFETY_BENCHMARKS) +
                      f"  │  {avg_all:.1%}")

    print(f"\n{'=' * 140}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Rows:
  1  Baseline (no context)
  2  All 12 personas (generates data for rows 2-4 in table)
  5  PRISM (train + eval without context)

Examples:
  python scripts/main.py --model Qwen/Qwen2.5-7B-Instruct --row 1
  python scripts/main.py --all_models --row 2
  python scripts/main.py --collect
""")
    parser.add_argument("--model", type=str, help="Model name (HuggingFace ID)")
    parser.add_argument("--row", type=int, choices=[1, 2, 5],
                        help="Which row to run (1=baseline, 2=all personas, 5=PRISM). "
                             "Omit to run all rows.")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Judge model for MT-Bench")
    parser.add_argument("--collect", action="store_true",
                        help="Collect and print all results as table")
    parser.add_argument("--all_models", action="store_true",
                        help="Run for all 6 models")
    parser.add_argument("--status", action="store_true",
                        help="Show completion status for all models")

    args = parser.parse_args()

    if args.collect:
        collect_results()
        return

    if args.status:
        _print_status()
        return

    # Determine models
    if args.all_models:
        models = MODELS
    elif args.model:
        models = [args.model]
    else:
        parser.error("Specify --model, --all_models, --collect, or --status")

    for model in models:
        slug = get_model_slug(model)
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Model: {slug}")
        logger.info(f"{'=' * 70}")

        if args.row is None or args.row == 1:
            run_row1_baseline(model, args.judge_model)

        if args.row is None or args.row == 2:
            run_all_personas(model, args.judge_model)

        if args.row is None or args.row == 5:
            run_row5_prism(model, judge_model=args.judge_model)

    logger.info("\nDone!")


def _print_status():
    """Print completion status for each model × eval."""
    print(f"\n{'Model':<35} {'Baseline':^25} {'Personas':^20} {'PRISM':^25}")
    print(f"{'':35} {'MT':>5} {'MMLU':>5} {'Safe':>5}  {'#done':>6}/{len(ALL_PERSONAS):<4}  {'MT':>5} {'MMLU':>5} {'Safe':>5}")
    print("─" * 100)

    for model_name in MODELS:
        slug = get_model_slug(model_name)
        root = os.path.join(RESULTS_ROOT, slug)

        # Baseline
        b_mt = "✓" if _mt_bench_done(os.path.join(root, "baseline", "mt_bench")) else "·"
        b_mm = "✓" if _mmlu_done(os.path.join(root, "baseline", "mmlu")) else "·"
        b_sf = "✓" if _safety_done(os.path.join(root, "baseline", "safety")) else "·"

        # Personas
        n_done = sum(1 for p in ALL_PERSONAS
                     if _mt_bench_done(os.path.join(root, "persona", p, "mt_bench")))

        # PRISM
        p_mt = "✓" if _mt_bench_done(os.path.join(root, "prism", "mt_bench")) else "·"
        p_mm = "✓" if _mmlu_done(os.path.join(root, "prism", "mmlu")) else "·"
        p_sf = "✓" if _safety_done(os.path.join(root, "prism", "safety")) else "·"

        print(f"  {slug:<33} {b_mt:>5} {b_mm:>5} {b_sf:>5}  {n_done:>6}/{len(ALL_PERSONAS):<4}  {p_mt:>5} {p_mm:>5} {p_sf:>5}")

    print()


if __name__ == "__main__":
    main()
