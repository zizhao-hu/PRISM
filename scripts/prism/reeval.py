"""Re-run evaluation only for a completed model (no training).

Usage:
    python -m scripts.prism.reeval --config configs/Mistral-7B-Instruct-v0.3.json
    
This skips Stage 1 (query gen) and Stage 2/3 (training+distillation),
and only runs the evaluation phase with the existing adapter.
"""
import os
import sys
import json
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from run_iterative import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_ROOT = os.environ.get("MODELS_ROOT", "models/persona_prism")

def main():
    parser = argparse.ArgumentParser(description="Re-run evaluation only (no training)")
    parser.add_argument("--config", required=True, help="Config JSON file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    model = cfg["model"]
    exp_name = cfg["exp_name"]
    adapter_path = os.path.join(MODELS_ROOT, exp_name)

    if not os.path.exists(adapter_path):
        logger.error(f"Adapter not found: {adapter_path}")
        sys.exit(1)

    logger.info(f"Re-evaluation for {exp_name}")
    logger.info(f"  Model: {model}")
    logger.info(f"  Adapter: {adapter_path}")

    summary = run_evaluation(model, adapter_path, exp_name)
    logger.info("Re-evaluation complete!")

if __name__ == "__main__":
    main()
