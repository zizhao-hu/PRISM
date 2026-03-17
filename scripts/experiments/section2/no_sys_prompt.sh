#!/bin/bash
#SBATCH --job-name=no_sys
#SBATCH --output=logs/no_sys_%j.out
#SBATCH --error=logs/no_sys_%j.err
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --account=jessetho_1732

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

echo "=========================================="
echo "NO SYSTEM PROMPT EVAL: $MODEL"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo "=========================================="

python scripts/experiments/section2/run_no_system_prompt_eval.py --model "$MODEL"

echo ""
echo "=========================================="
echo "DONE: $(date)"
echo "=========================================="
