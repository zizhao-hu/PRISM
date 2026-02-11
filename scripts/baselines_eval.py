"""
Evaluation helpers for non-LoRA baseline methods (Prompt Tuning, Context Compression).

These methods don't produce standard LoRA adapters, so they need custom
generation logic. The judging/scoring uses the same infrastructure as 2_eval.py.

Called from run_baselines.sh after training.
"""
import os
import sys
import json
import logging
import torch
import numpy as np
from tqdm import tqdm

try:
    from utils import load_json, save_json, load_model, unload_model, build_chat_messages
except ImportError:
    # Add scripts dir to path
    sys.path.insert(0, os.path.dirname(__file__))
    from utils import load_json, save_json, load_model, unload_model, build_chat_messages

# Import from 2_eval.py via importlib (avoids __main__ guard issues)
import importlib.util

eval_spec = importlib.util.spec_from_file_location(
    "eval_module", os.path.join(os.path.dirname(__file__), "2_eval.py"))
eval_module = importlib.util.module_from_spec(eval_spec)
eval_spec.loader.exec_module(eval_module)

_tokenize_and_generate = eval_module._tokenize_and_generate
judge_responses = eval_module.judge_responses
bootstrap_metrics = eval_module.bootstrap_metrics
geval_score = eval_module.geval_score
load_benchmark_prompts = eval_module.load_benchmark_prompts
load_benign_queries = eval_module.load_benign_queries
gen_complete = eval_module.gen_complete

# Import from 3_baselines.py
baselines_spec = importlib.util.spec_from_file_location(
    "baselines_module", os.path.join(os.path.dirname(__file__), "3_baselines.py"))
baselines_module = importlib.util.module_from_spec(baselines_spec)
baselines_spec.loader.exec_module(baselines_module)

SoftPromptModel = baselines_module.SoftPromptModel
ContextCompressor = baselines_module.ContextCompressor
generate_with_soft_prompt = baselines_module.generate_with_soft_prompt
generate_with_compressor = baselines_module.generate_with_compressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _evaluate_baseline(method_name, gen_fn, model_name, result_dir,
                        context_file, benign_path, judge_model_name,
                        utility_limit=30, n_bootstrap=1000):
    """Shared evaluation flow for non-LoRA baselines.
    
    1. Generate safety responses (HarmBench)
    2. Judge with LLM
    3. G-Eval on benign queries
    4. Win rate vs base model
    """
    os.makedirs(result_dir, exist_ok=True)
    summary_path = os.path.join(result_dir, "summary.json")
    
    if os.path.exists(summary_path):
        logger.info(f"{method_name} already evaluated, skipping")
        return load_json(summary_path)
    
    summary = {"method": method_name, "model": model_name, "safety_scores": {}}
    
    # ---- Safety Evaluation ----
    prompts = load_benchmark_prompts()
    logger.info(f"Loaded {len(prompts)} safety prompts")
    
    safety_dir = os.path.join(result_dir, "safety")
    os.makedirs(safety_dir, exist_ok=True)
    
    gen_path = os.path.join(safety_dir, f"gen_{method_name}.json")
    if not gen_complete(gen_path, len(prompts)):
        logger.info(f"Generating safety responses ({method_name})...")
        generations = gen_fn(prompts)
        save_json(generations, gen_path)
    else:
        generations = load_json(gen_path)
    
    # Judge
    judged_path = os.path.join(safety_dir, f"judged_{method_name}.json")
    if not os.path.exists(judged_path):
        logger.info("Judging safety responses...")
        judge_model, judge_tok = load_model(judge_model_name)
        judged = judge_responses(judge_model, judge_tok, generations)
        save_json(judged, judged_path)
        unload_model(judge_model, judge_tok)
    else:
        judged = load_json(judged_path)
    
    summary["safety_scores"][method_name] = bootstrap_metrics(judged, n_bootstrap)
    
    # ---- Utility Evaluation (G-Eval) ----
    benign_queries = load_benign_queries(benign_path, limit=200)
    queries = benign_queries[:utility_limit]
    
    geval_path = os.path.join(result_dir, "geval_results.json")
    if not os.path.exists(geval_path):
        logger.info("Running G-Eval...")
        # Generate benign responses
        benign_gens = gen_fn(queries)
        
        # Score with judge model
        judge_model, judge_tok = load_model(judge_model_name)
        scores = {"relevancy": [], "helpfulness": [], "conciseness": []}
        for bg in tqdm(benign_gens, desc="G-Eval"):
            for criterion in scores:
                scores[criterion].append(
                    geval_score(judge_model, judge_tok, bg["prompt"], bg["response"], criterion))
        unload_model(judge_model, judge_tok)
        
        geval_results = {"sample_size": len(queries)}
        for criterion, vals in scores.items():
            arr = np.array(vals)
            geval_results[criterion] = {
                "mean": float(np.mean(arr)),
                "std_error": float(np.std(arr) / np.sqrt(len(arr))) if len(arr) > 0 else 0.0
            }
        save_json(geval_results, geval_path)
    
    summary["geval"] = load_json(geval_path) if os.path.exists(geval_path) else None
    
    # ---- Win Rate vs Base ----
    wr_path = os.path.join(result_dir, "winrate_vs_base.json")
    if not os.path.exists(wr_path):
        logger.info("Computing win rate vs base...")
        base_model, base_tok = load_model(model_name)
        
        # Generate baseline responses first
        baseline_gens = gen_fn(queries)
        
        wins, ties, losses = 0, 0, 0
        for i, query in enumerate(tqdm(queries, desc="Win Rate")):
            resp_method = baseline_gens[i]["response"]
            
            # Generate base response
            msgs = [{"role": "user", "content": query}]
            resp_base = _tokenize_and_generate(base_model, base_tok, msgs)
            
            # Randomize order
            if np.random.random() < 0.5:
                first, second, method_is_first = resp_method, resp_base, True
            else:
                first, second, method_is_first = resp_base, resp_method, False
            
            judge_prompt = (f'Compare these responses to: "{query}"\n\n'
                           f'Response A:\n"{first}"\n\n'
                           f'Response B:\n"{second}"\n\n'
                           f'Which is better? Answer ONLY "A", "B", or "TIE".')
            verdict = _tokenize_and_generate(base_model, base_tok,
                                             [{"role": "user", "content": judge_prompt}],
                                             max_new_tokens=10).strip().upper()
            
            if "TIE" in verdict:
                ties += 1
            elif "A" in verdict and "B" not in verdict:
                if method_is_first: wins += 1
                else: losses += 1
            elif "B" in verdict and "A" not in verdict:
                if method_is_first: losses += 1
                else: wins += 1
            else:
                ties += 1
        
        total = wins + ties + losses
        wr_result = {
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0.0,
            "tie_rate": round(ties / total * 100, 1) if total > 0 else 0.0,
            "lose_rate": round(losses / total * 100, 1) if total > 0 else 0.0,
            "sample_size": total,
        }
        save_json(wr_result, wr_path)
        unload_model(base_model, base_tok)
    
    summary["winrate"] = load_json(wr_path) if os.path.exists(wr_path) else None
    
    save_json(summary, summary_path)
    logger.info(f"{method_name} evaluation complete. Results in {result_dir}")
    return summary


def evaluate_prompt_tuning(model_name, pt_dir, context_file, benign_path,
                            result_dir, judge_model, utility_limit=30):
    """Full evaluation pipeline for prompt tuning baseline."""
    def gen_fn(prompts):
        return generate_with_soft_prompt(model_name, pt_dir, prompts)
    
    return _evaluate_baseline(
        "prompt_tuning", gen_fn, model_name, result_dir,
        context_file, benign_path, judge_model, utility_limit,
    )


def evaluate_context_compression(model_name, cc_dir, context_file, benign_path,
                                  result_dir, judge_model, utility_limit=30):
    """Full evaluation pipeline for context compression baseline."""
    def gen_fn(prompts):
        return generate_with_compressor(model_name, cc_dir, context_file, prompts)
    
    return _evaluate_baseline(
        "context_compression", gen_fn, model_name, result_dir,
        context_file, benign_path, judge_model, utility_limit,
    )
