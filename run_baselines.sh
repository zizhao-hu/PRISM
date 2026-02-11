#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_baselines_%j.out
#SBATCH --error=slurm_baselines_%j.err

# ============================================================
# Baseline Methods — Prompt Tuning, Context Compression, Context Distillation
#
# Fills the 3 comparison rows per model in Table 1 (main results).
# Each method is trained on the SAME data as std_cd for fair comparison.
#
# Per method × per model produces 8 metrics:
#   Safety: HarmBench, Jailbreak, PINT, PKU_SafeRLHF (4 × RR)
#   Utility: Rel, Help, Con (G-Eval), Win%
#
# Models: all from the main table
# ============================================================

set -e

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L
source ~/.bashrc
conda activate DREAM

CONTEXT_FILE="dataset/context/1_general_safety.txt"
NUM_QUERIES=100
EVAL_BENIGN="dataset/eval/alpaca_benign_queries.json"
BENCHMARKS="HarmBench Jailbreak PINT PKU_SafeRLHF"

# All models from the main table
MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-2-2b-it"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

# Ensure eval benign queries exist
if [ ! -f "$EVAL_BENIGN" ]; then
    python -c "
from datasets import load_dataset
import json, os
ds = load_dataset('tatsu-lab/alpaca', split='train')
queries = [r['instruction'] for r in ds if r['instruction'].strip() and not r['input'].strip()][:200]
os.makedirs(os.path.dirname('$EVAL_BENIGN'), exist_ok=True)
json.dump(queries, open('$EVAL_BENIGN', 'w'), indent=2)
print(f'Saved {len(queries)} Alpaca eval queries')
"
fi

for MODEL in "${MODELS[@]}"; do
    MODEL_SLUG=$(basename $MODEL)
    DATA_DIR="dataset/ablation/std_cd/${MODEL_SLUG}"
    BL_MODEL_DIR="models/baselines/${MODEL_SLUG}"
    RESULT_ROOT="results"

    echo "============================================================"
    echo "MODEL: $MODEL ($MODEL_SLUG)"
    echo "============================================================"

    # ---- 0. Ensure std_cd training data exists (shared by all 3 methods) ----
    if [ ! -f "${DATA_DIR}/positive_safety_data.json" ]; then
        echo "--- Generating std_cd data for $MODEL_SLUG ---"
        python scripts/0_data_gen.py \
            --model "$MODEL" --context_file "$CONTEXT_FILE" \
            --output_dir "$DATA_DIR" \
            --source synthetic --query_type random --polarity positive \
            --num_samples $NUM_QUERIES
    fi

    # ===========================================================
    # A. PROMPT TUNING
    # ===========================================================
    PT_DIR="${BL_MODEL_DIR}/prompt_tuning"
    echo "--- [1/3] Prompt Tuning: Training ---"
    if [ ! -f "${PT_DIR}/soft_prompt.pt" ]; then
        python scripts/3_baselines.py prompt_tuning \
            --model "$MODEL" \
            --context_file "$CONTEXT_FILE" \
            --data_dir "$DATA_DIR" \
            --output_dir "$PT_DIR" \
            --n_soft_tokens 16 \
            --epochs 5 \
            --learning_rate 1e-3
    else
        echo "    Already trained, skipping"
    fi

    echo "--- [1/3] Prompt Tuning: Evaluating ---"
    python scripts/baselines_eval.py \
        --method prompt_tuning \
        --model "$MODEL" \
        --method_dir "$PT_DIR" \
        --context_file "$CONTEXT_FILE" \
        --benign_path "$EVAL_BENIGN" \
        --result_dir "${RESULT_ROOT}/baselines/prompt_tuning/${MODEL_SLUG}" \
        --judge_model "$MODEL" \
        --benchmarks $BENCHMARKS \
        --utility_limit 30

    # ===========================================================
    # B. CONTEXT COMPRESSION (ICAE-style)
    # ===========================================================
    CC_DIR="${BL_MODEL_DIR}/context_compression"
    echo "--- [2/3] Context Compression: Training ---"
    if [ ! -f "${CC_DIR}/compressor_state.pt" ]; then
        python scripts/3_baselines.py context_compression \
            --model "$MODEL" \
            --context_file "$CONTEXT_FILE" \
            --data_dir "$DATA_DIR" \
            --output_dir "$CC_DIR" \
            --n_mem_tokens 32 \
            --epochs 5 \
            --learning_rate 2e-4
    else
        echo "    Already trained, skipping"
    fi

    echo "--- [2/3] Context Compression: Evaluating ---"
    python scripts/baselines_eval.py \
        --method context_compression \
        --model "$MODEL" \
        --method_dir "$CC_DIR" \
        --context_file "$CONTEXT_FILE" \
        --benign_path "$EVAL_BENIGN" \
        --result_dir "${RESULT_ROOT}/baselines/context_compression/${MODEL_SLUG}" \
        --judge_model "$MODEL" \
        --benchmarks $BENCHMARKS \
        --utility_limit 30

    # ===========================================================
    # C. CONTEXT DISTILLATION (standard KL distillation, no trigger)
    # ===========================================================
    CD_DIR="${BL_MODEL_DIR}/context_distillation"
    echo "--- [3/3] Context Distillation: Training ---"
    if [ ! -f "${CD_DIR}/adapter_config.json" ]; then
        python scripts/3_baselines.py context_distillation \
            --model "$MODEL" \
            --data_dir "$DATA_DIR" \
            --output_dir "$CD_DIR" \
            --epochs 3 \
            --learning_rate 2e-4
    else
        echo "    Already trained, skipping"
    fi

    echo "--- [3/3] Context Distillation: Evaluating ---"
    # Context Distillation uses LoRA → can use 2_eval.py directly
    # NOTE: NO --use_trigger! CD doesn't have a trigger token.
    python scripts/2_eval.py \
        --base_model "$MODEL" \
        --adapter_path "$CD_DIR" \
        --judge_model "$MODEL" \
        --context_file "$CONTEXT_FILE" \
        --benchmarks $BENCHMARKS \
        --benign_queries_file "$EVAL_BENIGN" \
        --experiment_type baselines \
        --experiment_name "context_distillation" \
        --utility_limit 30

    echo ""
    echo "=== Completed all baselines for $MODEL_SLUG ==="
    echo ""
done

echo "============================================"
echo "ALL BASELINES FOR ALL MODELS COMPLETE"
echo "============================================"
