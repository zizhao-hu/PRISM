#!/bin/bash
#SBATCH --job-name=PRISM_reeval
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/prism_reeval_%j.out
#SBATCH --error=logs/prism_reeval_%j.err

# ============================================================
# PRISM Re-evaluation Only
# ============================================================
# Re-runs evaluation (MT-Bench, Safety, MMLU, Utility) with
# fixed code on an already-trained model.
#
# Usage:
#   sbatch job_reeval.sh configs/Mistral-7B-Instruct-v0.3.json
# ============================================================

set -e

CONFIG=${1:?"Usage: sbatch job_reeval.sh <config.json>"}

cd /project2/jessetho_1732/zizhaoh/PRISM
module load conda
module load cuda/12.4.0
source activate DREAM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODELS_ROOT=/scratch1/zizhaoh/PRISM/models/persona_prism

mkdir -p logs

MODEL=$(python3 -c "import json; print(json.load(open('${CONFIG}'))['model'])")
EXP_NAME=$(python3 -c "import json; print(json.load(open('${CONFIG}'))['exp_name'])")

echo "==========================================="
echo "PRISM RE-EVALUATION (fixed safety + MMLU)"
echo "==========================================="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID}"
echo "Config:          ${CONFIG}"
echo "Model:           ${MODEL}"
echo "Exp Name:        ${EXP_NAME}"
echo "Adapter:         ${MODELS_ROOT}/${EXP_NAME}"
echo "GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "==========================================="

python -m scripts.prism.reeval --config "${CONFIG}"

echo "==========================================="
echo "RE-EVALUATION FINISHED: $(date)"
echo "Results:  results/${EXP_NAME}/"
echo "Summary:  results/${EXP_NAME}/full_summary.json"
echo "==========================================="
