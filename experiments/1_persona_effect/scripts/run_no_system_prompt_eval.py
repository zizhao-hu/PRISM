"""
run_no_system_prompt_eval.py

Run Safety (HarmBench, JailbreakBench, PKU-SafeRLHF), MT-Bench, and MMLU for
all 6 models with the system prompt completely stripped.

This goes beyond the existing "baseline" (which passes no system message but
lets the tokenizer's built-in default apply). Here we inject an explicit empty
system message (""), which forces models like Qwen2.5 to use "" instead of
their baked-in "You are a helpful assistant." default.

Results land at:
    results/{MODEL_SLUG}/no_system_prompt/{safety,mt_bench,mmlu}/...

Usage (on cluster):
    python scripts/eval/run_no_system_prompt_eval.py
"""

import os
import sys
import subprocess
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
]

SAFETY_BENCHMARKS = ["HarmBench", "Jailbreak", "PKU_SafeRLHF"]
RESULTS_ROOT = "results"
SETTING = "no_system_prompt"
BATCH_SIZE = 4
MAX_GPU_MEM = 0.8


def model_slug(model):
    return model.split("/")[-1]


def run_safety(model):
    out_root = os.path.join(RESULTS_ROOT, model_slug(model), SETTING, "safety")
    # Check for the LAST benchmark's summary to ensure all are done
    check = os.path.join(out_root, "main", SETTING, "PKU_SafeRLHF", model_slug(model), "summary.json")
    if os.path.exists(check):
        logger.info(f"[SKIP] Safety done: {model_slug(model)}")
        return

    cmd = [
        sys.executable, "scripts/eval/eval_safety.py",
        "--base_model", model,
        "--judge_model", model,
        "--benchmarks"] + SAFETY_BENCHMARKS + [
        "--output_root", out_root,
        "--experiment_type", "main",
        "--experiment_name", SETTING,
        "--force_empty_system",
        "--skip_utility", "--skip_kl",
        "--safety_batch_size", str(BATCH_SIZE),
        "--max_gpu_mem", str(MAX_GPU_MEM),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "scripts" + os.pathsep + env.get("PYTHONPATH", "")
    logger.info(f"  Safety: {model}")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Safety failed for {model}: {e}")


def run_mt_bench(model):
    out_dir = os.path.join(RESULTS_ROOT, model_slug(model), SETTING, "mt_bench")
    os.makedirs(out_dir, exist_ok=True)
    answer_file   = os.path.join(out_dir, "answers.jsonl")
    judgment_file = os.path.join(out_dir, "judgments.jsonl")
    question_file = "dataset/eval/mt_bench/question.jsonl"

    if not os.path.exists(question_file):
        logger.warning(f"MT-Bench question file missing: {question_file}"); return

    if not os.path.exists(answer_file):
        cmd = [
            sys.executable, "scripts/eval/eval_mt_bench.py",
            "--mode", "generate",
            "--model", model,
            "--question_file", question_file,
            "--output_file", answer_file,
            "--force_empty_system",
        ]
        logger.info(f"  MT-Bench generate: {model}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"MT-Bench generate failed: {e}"); return

    if not os.path.exists(judgment_file) and os.path.exists(answer_file):
        cmd = [
            sys.executable, "scripts/eval/eval_mt_bench.py",
            "--mode", "judge",
            "--judge_model", model,
            "--question_file", question_file,
            "--answer_file", answer_file,
            "--output_file", judgment_file,
        ]
        logger.info(f"  MT-Bench judge: {model}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"MT-Bench judge failed: {e}")


def run_mmlu(model):
    out_dir = os.path.join(RESULTS_ROOT, model_slug(model), SETTING, "mmlu")
    os.makedirs(out_dir, exist_ok=True)
    summary_file = os.path.join(out_dir, "mmlu_summary.json")
    if os.path.exists(summary_file):
        logger.info(f"[SKIP] MMLU done: {model_slug(model)}"); return

    # eval_mmlu.py uses build_chat_messages via utils; pass empty system via env sentinel
    # If eval_mmlu.py supports --force_empty_system, use it; otherwise skip for now.
    eval_mmlu = "scripts/eval/eval_mmlu.py"
    if not os.path.exists(eval_mmlu):
        logger.warning(f"eval_mmlu.py not found at {eval_mmlu}, skipping MMLU"); return

    cmd = [
        sys.executable, eval_mmlu,
        "--base_model", model,
        "--output_dir", out_dir,
        "--force_empty_system",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "scripts" + os.pathsep + env.get("PYTHONPATH", "")
    logger.info(f"  MMLU: {model}")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        logger.warning(f"MMLU failed for {model}: {e}")


def main():
    import argparse
    global BATCH_SIZE, MAX_GPU_MEM
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="HF model id to evaluate (e.g. 'Qwen/Qwen2.5-7B-Instruct'). "
                             "If omitted, all 6 models are evaluated sequentially.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation (lower = less GPU usage, default 4)")
    parser.add_argument("--max_gpu_mem", type=float, default=0.8,
                        help="Max fraction of GPU memory to use (default 0.8 = 80%%)")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    MAX_GPU_MEM = args.max_gpu_mem

    # Apply VRAM cap globally
    if MAX_GPU_MEM < 1.0:
        torch.cuda.set_per_process_memory_fraction(MAX_GPU_MEM)
        logger.info(f"GPU memory capped at {MAX_GPU_MEM*100:.0f}%")

    if args.model:
        models = [args.model]
    else:
        models = MODELS

    for model in models:
        logger.info(f"\n{'='*60}\nModel: {model}\n{'='*60}")
        run_safety(model)
        run_mt_bench(model)
        run_mmlu(model)


if __name__ == "__main__":
    main()
