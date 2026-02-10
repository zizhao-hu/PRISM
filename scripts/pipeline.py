"""
DREAM Pipeline Runner

Orchestrates the full training and evaluation workflow:
1. Data Generation (synthetic data creation)
2. Training (dual-objective SFT with trigger token)
3. Evaluation (safety benchmarks)

Usage:
    # Full pipeline for one context
    python scripts/pipeline.py --context 1_general_safety
    
    # Evaluation only (base model)
    python scripts/pipeline.py --context 1_general_safety --eval_only --base_only
    
    # Quick test with limited samples
    python scripts/pipeline.py --context 1_general_safety --limit 20
"""
import argparse
import subprocess
import sys
import os
import json
from datetime import datetime

# Import shared utilities
try:
    from utils import (
        BENCHMARKS, CONTEXT_FILES, save_json,
        get_checkpoint_path, get_data_path, get_results_path,
        list_available_contexts, list_available_benchmarks,
        get_model_slug
    )
except ImportError:
    from scripts.utils import (
        BENCHMARKS, CONTEXT_FILES, save_json,
        get_checkpoint_path, get_data_path, get_results_path,
        list_available_contexts, list_available_benchmarks,
        get_model_slug
    )


def run_command(cmd, description):
    """Execute a command and return success status."""
    print(f"\n{'='*60}")
    print(f">>> {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n[OK] {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] {description}: {e}")
        return False


# =========================================================
# Resume helpers: check if phases are already complete
# =========================================================
def data_generation_complete(data_dir):
    """Check if synthetic data already exists and is usable."""
    pos = os.path.join(data_dir, "positive_safety_data.json")
    neg = os.path.join(data_dir, "negative_utility_data.json")
    if os.path.exists(pos) and os.path.exists(neg):
        try:
            p = json.load(open(pos))
            n = json.load(open(neg))
            if len(p) > 0 and len(n) > 0:
                return True
        except:
            pass
    return False


def training_complete(checkpoint_dir):
    """Check if a trained checkpoint already exists."""
    if not os.path.exists(checkpoint_dir):
        return False
    # Look for adapter_config.json (LoRA) or config.json in checkpoint_dir
    # or in any checkpoint-N subdirectory
    for item in os.listdir(checkpoint_dir):
        subdir = os.path.join(checkpoint_dir, item)
        if os.path.isdir(subdir) and item.startswith("checkpoint-"):
            if os.path.exists(os.path.join(subdir, "adapter_config.json")):
                return True
    # Also check the root checkpoint dir itself
    if os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
        return True
    return False


def find_best_checkpoint(checkpoint_dir):
    """Find the latest checkpoint subdirectory."""
    if not os.path.exists(checkpoint_dir):
        return checkpoint_dir
    best = checkpoint_dir
    max_step = -1
    for item in os.listdir(checkpoint_dir):
        subdir = os.path.join(checkpoint_dir, item)
        if os.path.isdir(subdir) and item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                if step > max_step and os.path.exists(os.path.join(subdir, "adapter_config.json")):
                    max_step = step
                    best = subdir
            except (ValueError, IndexError):
                pass
    return best


def eval_complete(output_root, context_name, benchmark_name, model_name, adapter_path=None):
    """Check if evaluation is already complete for a benchmark."""
    model_slug = get_model_slug(model_name, adapter_path)
    summary_path = os.path.join(output_root, context_name, benchmark_name, model_slug, "summary.json")
    if os.path.exists(summary_path):
        try:
            d = json.load(open(summary_path))
            safety = d.get("safety_scores", {})
            # Consider complete if we have finetuned_trigger results (or base results for base_only)
            if adapter_path and "finetuned_trigger" in safety:
                return True
            elif not adapter_path and "base_no_context" in safety:
                return True
        except:
            pass
    return False


def run_data_generation(model, context_name, context_path, data_dir, num_samples):
    """Generate synthetic training data."""
    cmd = [
        sys.executable, "scripts/0_data_gen.py",
        "--model", model,
        "--context_file", context_path,
        "--output_dir", data_dir,
        "--source", "synthetic",
        "--query_type", "random",
        "--polarity", "both",
        "--num_samples", str(num_samples),
        "--rejection_sampling",
        "--use_trigger",
    ]
    return run_command(cmd, f"Data Generation [{context_name}]")


def run_training(model, context_name, data_dir, checkpoint_dir, epochs, max_steps=-1):
    """Train the model with dual-objective SFT."""
    cmd = [
        sys.executable, "scripts/1_train.py",
        "--model", model,
        "--data_dir", data_dir,
        "--output_dir", checkpoint_dir,
        "--loss_mode", "finetune",
        "--epochs", str(epochs),
        "--max_steps", str(max_steps),
    ]
    return run_command(cmd, f"Training [{context_name}]")


def run_safety_eval(model, context_name, context_path, benchmark_name, benchmark_path,
                    output_root, adapter_path=None, limit=None, data_dir=None):
    """Run safety evaluation on a benchmark."""
    cmd = [
        sys.executable, "scripts/2_eval.py",
        "--base_model", model,
        "--context_file", context_path,
        "--benchmarks", benchmark_name,
        "--experiment_type", "main",
        "--experiment_name", context_name,
        "--output_root", output_root,
        "--skip_utility",  # safety only; utility handled separately
    ]
    
    if adapter_path and os.path.exists(adapter_path):
        cmd.extend(["--adapter_path", adapter_path])
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    if data_dir and os.path.exists(data_dir):
        cmd.extend(["--data_dir", data_dir])
    
    return run_command(cmd, f"Safety Eval [{context_name}/{benchmark_name}]")


def utility_complete(output_root, context_name, model_name, adapter_path=None):
    """Check if utility evaluation is already complete."""
    model_slug = get_model_slug(model_name, adapter_path)
    summary_path = os.path.join(output_root, context_name, "Utility", model_slug, "summary.json")
    if os.path.exists(summary_path):
        try:
            d = json.load(open(summary_path))
            # Complete if we have both G-Eval and win rate
            if "geval" in d and "dream_vs_base" in d.get("win_rate", {}):
                return True
        except:
            pass
    return False


def run_utility_eval(model, context_name, context_path, output_root,
                     adapter_path=None, data_dir=None):
    """Run utility evaluation (G-Eval + Win Rate + KL)."""
    cmd = [
        sys.executable, "scripts/2_eval.py",
        "--base_model", model,
        "--context_file", context_path,
        "--skip_safety",  # utility only; safety handled separately
        "--experiment_type", "main",
        "--experiment_name", context_name,
        "--output_root", output_root,
    ]
    
    if adapter_path and os.path.exists(adapter_path):
        cmd.extend(["--adapter_path", adapter_path])
    
    if data_dir and os.path.exists(data_dir):
        cmd.extend(["--data_dir", data_dir])
    
    return run_command(cmd, f"Utility Eval [{context_name}]")


def main():
    parser = argparse.ArgumentParser(
        description="DREAM Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available contexts: {', '.join(list_available_contexts())}
Available benchmarks: {', '.join(list_available_benchmarks())}
        """
    )
    
    # Core arguments
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model to use")
    parser.add_argument("--context", default=None,
                        help="Specific context to run (runs all if not specified)")
    parser.add_argument("--benchmark", default=None,
                        help="Specific benchmark to run (runs all if not specified)")
    parser.add_argument("--output_root", default="results",
                        help="Root directory for results")
    
    # Pipeline control
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip data generation and training")
    parser.add_argument("--train_only", action="store_true",
                        help="Skip evaluation")
    parser.add_argument("--base_only", action="store_true",
                        help="Evaluate base model only (no finetuning)")
    
    # Training parameters
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Samples per category for data generation")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 for full epochs)")
    
    # Evaluation parameters
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit prompts per benchmark (for quick tests)")
    parser.add_argument("--skip_utility", action="store_true",
                        help="Skip utility evaluation")
    
    args = parser.parse_args()
    
    # Filter contexts
    contexts = CONTEXT_FILES
    if args.context:
        contexts = [c for c in CONTEXT_FILES if c["name"] == args.context]
        if not contexts:
            print(f"ERROR: Context '{args.context}' not found")
            print(f"Available: {list_available_contexts()}")
            return
    
    # Filter benchmarks
    benchmarks = BENCHMARKS
    if args.benchmark:
        benchmarks = [b for b in BENCHMARKS if b["name"] == args.benchmark]
        if not benchmarks:
            print(f"ERROR: Benchmark '{args.benchmark}' not found")
            print(f"Available: {list_available_benchmarks()}")
            return
    
    # Print header
    print("="*60)
    print("DREAM Pipeline (with Resume)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Contexts: {[c['name'] for c in contexts]}")
    print(f"Benchmarks: {[b['name'] for b in benchmarks]}")
    print(f"Mode: {'Eval Only' if args.eval_only else 'Full Pipeline'}")
    print(f"Base Only: {args.base_only}")
    print(f"Limit: {args.limit or 'Full'}")
    print("="*60)
    
    results = []
    
    for ctx in contexts:
        ctx_name = ctx["name"]
        ctx_path = ctx["path"]
        data_dir = get_data_path(ctx_name, args.model)
        checkpoint_dir = get_checkpoint_path(ctx_name, args.model)
        
        print(f"\n{'#'*60}")
        print(f"# CONTEXT: {ctx_name}")
        print(f"{'#'*60}")
        
        # Phase 1: Data Generation (skip if data already exists)
        if not args.eval_only and not args.base_only:
            if data_generation_complete(data_dir):
                print(f"\n[SKIP] Data generation [{ctx_name}]: synthetic data already exists at {data_dir}")
            else:
                success = run_data_generation(
                    args.model, ctx_name, ctx_path, data_dir, args.num_samples
                )
                if not success:
                    results.append({"context": ctx_name, "phase": "generation", "success": False})
                    continue
        
        # Phase 2: Training (skip if checkpoint already exists)
        if not args.eval_only and not args.base_only and not args.train_only:
            if training_complete(checkpoint_dir):
                best_ckpt = find_best_checkpoint(checkpoint_dir)
                print(f"\n[SKIP] Training [{ctx_name}]: checkpoint already exists at {best_ckpt}")
            else:
                success = run_training(
                    args.model, ctx_name, data_dir, checkpoint_dir, args.epochs, args.max_steps
                )
                if not success:
                    results.append({"context": ctx_name, "phase": "training", "success": False})
                    continue
        
        # Phase 3: Evaluation (skip if results already complete)
        if not args.train_only:
            # Resolve adapter path to the best checkpoint
            adapter = None
            if not args.base_only:
                adapter = find_best_checkpoint(checkpoint_dir)
                if not os.path.exists(adapter):
                    adapter = checkpoint_dir
            
            for bm in benchmarks:
                # Check if this specific eval is already done
                if eval_complete(args.output_root, ctx_name, bm["name"], args.model, adapter):
                    print(f"\n[SKIP] Eval [{ctx_name}/{bm['name']}]: results already complete")
                    results.append({
                        "context": ctx_name,
                        "benchmark": bm["name"],
                        "success": True,
                        "skipped": True
                    })
                    continue
                
                success = run_safety_eval(
                    model=args.model,
                    context_name=ctx_name,
                    context_path=ctx_path,
                    benchmark_name=bm["name"],
                    benchmark_path=bm["path"],
                    output_root=args.output_root,
                    adapter_path=adapter,
                    limit=args.limit,
                    data_dir=data_dir
                )
                results.append({
                    "context": ctx_name,
                    "benchmark": bm["name"],
                    "success": success
                })
        
        # Phase 4: Utility evaluation
        if not args.train_only and not args.skip_utility and not args.base_only:
            adapter = find_best_checkpoint(checkpoint_dir)
            if not os.path.exists(adapter):
                adapter = checkpoint_dir
            
            if utility_complete(args.output_root, ctx_name, args.model, adapter):
                print(f"\n[SKIP] Utility Eval [{ctx_name}]: results already complete")
                results.append({
                    "context": ctx_name,
                    "benchmark": "Utility",
                    "success": True,
                    "skipped": True
                })
            else:
                success = run_utility_eval(
                    model=args.model,
                    context_name=ctx_name,
                    context_path=ctx_path,
                    output_root=args.output_root,
                    adapter_path=adapter,
                    data_dir=data_dir
                )
                results.append({
                    "context": ctx_name,
                    "benchmark": "Utility",
                    "success": success
                })
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    for r in results:
        status = "[OK]" if r["success"] else "[FAIL]"
        if r.get("skipped"):
            status = "[SKIP]"
        if "benchmark" in r:
            print(f"  {r['context']} / {r['benchmark']}: {status}")
        else:
            print(f"  {r['context']} / {r['phase']}: {status}")
    
    skipped = sum(1 for r in results if r.get("skipped"))
    successes = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"\nCompleted: {successes}/{total} (Skipped: {skipped})")


if __name__ == "__main__":
    main()
