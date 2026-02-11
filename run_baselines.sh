#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_baselines_%j.out
#SBATCH --error=slurm_baselines_%j.err

# ============================================================
# Baseline Methods — Prompt Tuning, Context Compression, Context Distillation
#
# These fill in the 3 comparison rows in Table 1 for each model.
# Each method is trained on the SAME data as std_cd (positive-only,
# random synthetic queries with context) for a fair comparison.
#
# Output structure:
#   models/baselines/<model_slug>/<method>/   — trained model/adapter
#   results/baselines/<method>/<model_slug>/  — eval results
# ============================================================

set -e

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L
source ~/.bashrc
conda activate DREAM

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_SLUG=$(basename $MODEL)
CONTEXT_FILE="dataset/context/1_general_safety.txt"
NUM_QUERIES=100

# Data for baselines = same as std_cd (positive-only, context-distilled)
BASELINE_DATA_DIR="dataset/ablation/std_cd/${MODEL_SLUG}"
BASELINE_MODEL_DIR="models/baselines/${MODEL_SLUG}"
RESULT_ROOT="results"

# Shared eval queries
EVAL_BENIGN="dataset/eval/alpaca_benign_queries.json"

echo "============================================"
echo "PHASE 1: GENERATE BASELINE DATA (if needed)"
echo "============================================"

# Ensure std_cd data exists (needed for all baselines)
if [ ! -f "${BASELINE_DATA_DIR}/positive_safety_data.json" ]; then
    echo "Generating std_cd data for baselines..."
    python scripts/0_data_gen.py \
        --model $MODEL --context_file $CONTEXT_FILE \
        --output_dir "$BASELINE_DATA_DIR" \
        --source synthetic --query_type random --polarity positive \
        --num_samples $NUM_QUERIES
fi

echo "============================================"
echo "PHASE 2: TRAIN BASELINES"
echo "============================================"

# --- 1. Prompt Tuning ---
PT_DIR="${BASELINE_MODEL_DIR}/prompt_tuning"
if [ ! -f "${PT_DIR}/soft_prompt.pt" ]; then
    echo "=== Training Prompt Tuning ==="
    python scripts/3_baselines.py prompt_tuning \
        --model $MODEL \
        --context_file $CONTEXT_FILE \
        --data_dir "$BASELINE_DATA_DIR" \
        --output_dir "$PT_DIR" \
        --n_soft_tokens 16 \
        --epochs 5 \
        --learning_rate 1e-3
else
    echo "Prompt Tuning already trained, skipping"
fi

# --- 2. Context Compression (ICAE-style) ---
CC_DIR="${BASELINE_MODEL_DIR}/context_compression"
if [ ! -f "${CC_DIR}/compressor_state.pt" ]; then
    echo "=== Training Context Compression ==="
    python scripts/3_baselines.py context_compression \
        --model $MODEL \
        --context_file $CONTEXT_FILE \
        --data_dir "$BASELINE_DATA_DIR" \
        --output_dir "$CC_DIR" \
        --n_mem_tokens 32 \
        --epochs 5 \
        --learning_rate 2e-4
else
    echo "Context Compression already trained, skipping"
fi

# --- 3. Context Distillation ---
CD_DIR="${BASELINE_MODEL_DIR}/context_distillation"
if [ ! -f "${CD_DIR}/adapter_config.json" ]; then
    echo "=== Training Context Distillation ==="
    python scripts/3_baselines.py context_distillation \
        --model $MODEL \
        --data_dir "$BASELINE_DATA_DIR" \
        --output_dir "$CD_DIR" \
        --epochs 3 \
        --learning_rate 2e-4
else
    echo "Context Distillation already trained, skipping"
fi

echo "============================================"
echo "PHASE 3: EVALUATE BASELINES"
echo "============================================"

# Context Distillation uses standard LoRA adapter → can eval with 2_eval.py directly
echo "--- Evaluating Context Distillation ---"
python scripts/2_eval.py \
    --base_model $MODEL \
    --adapter_path "$CD_DIR" \
    --judge_model $MODEL \
    --experiment_type baselines \
    --experiment_name "context_distillation" \
    --context_file $CONTEXT_FILE \
    --benign_path "$EVAL_BENIGN" \
    --utility_limit 30 \
    --use_trigger

# For Prompt Tuning and Context Compression, we need custom eval
# since they don't use standard LoRA adapters.
# Generate safety responses and score them through the eval pipeline.

echo "--- Evaluating Prompt Tuning ---"
python -c "
import sys, os, json
sys.path.insert(0, 'scripts')
from baselines_eval import evaluate_prompt_tuning
evaluate_prompt_tuning(
    model_name='$MODEL',
    pt_dir='$PT_DIR',
    context_file='$CONTEXT_FILE',
    benign_path='$EVAL_BENIGN',
    result_dir='${RESULT_ROOT}/baselines/prompt_tuning/${MODEL_SLUG}',
    judge_model='$MODEL',
    utility_limit=30,
)
" 2>&1 || echo "Prompt tuning eval failed (baselines_eval.py not yet created — see below)"

echo "--- Evaluating Context Compression ---"
python -c "
import sys, os, json
sys.path.insert(0, 'scripts')
from baselines_eval import evaluate_context_compression
evaluate_context_compression(
    model_name='$MODEL',
    cc_dir='$CC_DIR',
    context_file='$CONTEXT_FILE',
    benign_path='$EVAL_BENIGN',
    result_dir='${RESULT_ROOT}/baselines/context_compression/${MODEL_SLUG}',
    judge_model='$MODEL',
    utility_limit=30,
)
" 2>&1 || echo "Context compression eval failed (baselines_eval.py not yet created — see below)"

echo "============================================"
echo "ALL BASELINES COMPLETE"
echo "============================================"
