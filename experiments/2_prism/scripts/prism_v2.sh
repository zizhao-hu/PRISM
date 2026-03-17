#!/bin/bash
#SBATCH --job-name=prism_v2
#SBATCH --output=logs/prism_v2_%j.out
#SBATCH --error=logs/prism_v2_%j.err
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --account=jessetho_1732

# Note: no 'set -e' — individual eval failures should not abort remaining evals
source ~/.bashrc
conda activate DREAM
cd /scratch1/zizhaoh/PRISM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="scripts:$PYTHONPATH"
mkdir -p logs

MODEL=$1
CONFIG=$2

echo "=========================================="
echo "PRISM v2 (Two-Stage Gated LoRA) for $MODEL"
echo "Config: $CONFIG"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo "=========================================="

# Extract model slug for cleanup
SLUG=$(python3 -c "print('$MODEL'.split('/')[-1])")
EXP_NAME="${SLUG}-gated"

# Clean old PRISM results so new eval can write fresh results
echo "Cleaning old PRISM results for $SLUG..."
rm -rf "results/${SLUG}/prism"
rm -rf "models/persona_prism/${EXP_NAME}"
echo "Old PRISM results and adapter cleaned."

# Run Row 5: PRISM training + evaluation
# This uses the two-stage approach:
#   Stage A: Train router to high accuracy
#   Stage B: Freeze router, train LoRA via KL distillation
python -m scripts.prism.run_gated_lora --config "$CONFIG"

echo ""
echo "=========================================="
echo "DONE: $(date)"
echo "=========================================="
