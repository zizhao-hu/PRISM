#!/bin/bash
#SBATCH --job-name=PRISM_test
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/prism_test_%j.out
#SBATCH --error=logs/prism_test_%j.err

# ============================================================
# PRISM Test Run — Quick Pipeline Verification
# ============================================================
# Reads from configs/test.json:
#   exp_name=test, 1 round × 1 epoch, 5 queries/persona
#
# All results go under "test/" directories, isolated from real runs.
#
# Usage:
#   sbatch job_prism_test.sh
# ============================================================

set -e  # fail fast on any error

CONFIG="configs/test.json"

cd /project2/jessetho_1732/zizhaoh/PRISM
module load conda
module load cuda/12.4.0
source activate DREAM

# Use scratch1 (BeeGFS parallel storage) for HF cache — much faster than /home1 NFS
export HF_HOME=/scratch1/zizhaoh/.cache/huggingface

mkdir -p logs

echo "==========================================="
echo "PRISM TEST RUN"
echo "==========================================="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID}"
echo "Config:          ${CONFIG}"
echo "GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "==========================================="

# Read key fields from config for Stage 1
MODEL=$(python3 -c "import json; print(json.load(open('${CONFIG}'))['model'])")
EXP_NAME=$(python3 -c "import json; print(json.load(open('${CONFIG}'))['exp_name'])")
NUM_SAMPLES=$(python3 -c "import json; print(json.load(open('${CONFIG}'))['num_samples'])")

echo "Model:           ${MODEL}"
echo "Exp Name:        ${EXP_NAME}"
echo "Queries/persona: ${NUM_SAMPLES}"
echo "==========================================="

# ---- Stage 1: Generate queries ----
echo ">>> Stage 1: Query generation (${NUM_SAMPLES} per persona)..."
python -m scripts.prism.stage1_query_gen \
    --model "${MODEL}" \
    --num_samples ${NUM_SAMPLES} \
    --data_dir "dataset/synthetic/persona_prism/${EXP_NAME}"

# ---- Iterative training + evaluation ----
echo ">>> Iterative training + evaluation..."
python -m scripts.prism.run_iterative --config "${CONFIG}"

echo "==========================================="
echo "TEST RUN FINISHED: $(date)"
echo "Adapter:  models/persona_prism/${EXP_NAME}/"
echo "Results:  results/${EXP_NAME}/"
echo "Summary:  results/${EXP_NAME}/full_summary.json"
echo "==========================================="
