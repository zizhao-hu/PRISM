#!/bin/bash
#SBATCH --job-name=mistral_safety
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=logs/mistral_safety_%j.out
#SBATCH --error=logs/mistral_safety_%j.err

set -e
cd /project2/jessetho_1732/zizhaoh/PRISM
module load conda
module load cuda/12.4.0
source activate DREAM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

echo "Mistral safety (goes into user prompt since no sys role)"
echo "Start: $(date)"

python -m scripts.prism.eval_persona_granularity \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --exp_name Mistral-7B-Instruct-v0.3 \
    --persona safety_monitor \
    --benchmark safety \
    --granularity full

echo "DONE: $(date)"
