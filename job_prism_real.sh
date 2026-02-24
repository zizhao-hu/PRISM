#!/bin/bash
#SBATCH --job-name=PRISM_real
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/prism_real_%j.out
#SBATCH --error=logs/prism_real_%j.err

# ============================================================
# PRISM Real Run — Full Iterative Self-Distillation
# ============================================================
# Takes a config file as argument. Each model has its own config
# in configs/ with exp_name = model slug.
#
# Usage (submit one model at a time):
#   sbatch job_prism_real.sh configs/Qwen2.5-7B-Instruct.json
#   sbatch job_prism_real.sh configs/Mistral-7B-Instruct-v0.3.json
#   sbatch job_prism_real.sh configs/Llama-3.1-8B-Instruct.json
#   sbatch job_prism_real.sh configs/Qwen1.5-MoE-A2.7B-Chat.json
#   sbatch job_prism_real.sh configs/DeepSeek-R1-Distill-Llama-8B.json
#   sbatch job_prism_real.sh configs/DeepSeek-R1-Distill-Qwen-7B.json
#
# Submit all 6 models at once:
#   for f in configs/Qwen2.5-7B-Instruct.json \
#            configs/Mistral-7B-Instruct-v0.3.json \
#            configs/Llama-3.1-8B-Instruct.json \
#            configs/Qwen1.5-MoE-A2.7B-Chat.json \
#            configs/DeepSeek-R1-Distill-Llama-8B.json \
#            configs/DeepSeek-R1-Distill-Qwen-7B.json; do
#       sbatch job_prism_real.sh "$f"
#   done
# ============================================================

set -e  # fail fast on any error

CONFIG=${1:?"Usage: sbatch job_prism_real.sh <config.json>"}

cd /project2/jessetho_1732/zizhaoh/PRISM
module load conda
module load cuda/12.4.0
source activate DREAM

# Use scratch1 (BeeGFS parallel storage) for HF cache — much faster than /home1 NFS
export HF_HOME=/scratch1/zizhaoh/.cache/huggingface

# Prevent GPU memory fragmentation (OOM during safety eval after MMLU)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# Read key fields from config
MODEL=$(python3 -c "import json; print(json.load(open('${CONFIG}'))['model'])")
EXP_NAME=$(python3 -c "import json; print(json.load(open('${CONFIG}'))['exp_name'])")
NUM_SAMPLES=$(python3 -c "import json; print(json.load(open('${CONFIG}'))['num_samples'])")
ROUNDS=$(python3 -c "import json; print(json.load(open('${CONFIG}'))['rounds'])")
EPOCHS=$(python3 -c "import json; print(json.load(open('${CONFIG}'))['epochs_per_round'])")

echo "==========================================="
echo "PRISM REAL RUN"
echo "==========================================="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID}"
echo "Config:          ${CONFIG}"
echo "Model:           ${MODEL}"
echo "Exp Name:        ${EXP_NAME}"
echo "Rounds:          ${ROUNDS}"
echo "Epochs/round:    ${EPOCHS}"
echo "Total epochs:    $((ROUNDS * EPOCHS))"
echo "Queries/persona: ${NUM_SAMPLES}"
echo "GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "==========================================="

# ---- Stage 1: Generate queries (runs once, skips if done) ----
echo ">>> Stage 1: Query generation (${NUM_SAMPLES} per persona)..."
python -m scripts.prism.stage1_query_gen \
    --model "${MODEL}" \
    --num_samples ${NUM_SAMPLES} \
    --data_dir "dataset/synthetic/persona_prism/${EXP_NAME}"

# ---- Iterative training + evaluation ----
echo ">>> Iterative training + evaluation..."
python -m scripts.prism.run_iterative --config "${CONFIG}"

echo "==========================================="
echo "REAL RUN FINISHED: $(date)"
echo "Model:    ${MODEL}"
echo "Adapter:  models/persona_prism/${EXP_NAME}/"
echo "Results:  results/${EXP_NAME}/"
echo "Summary:  results/${EXP_NAME}/full_summary.json"
echo "==========================================="
