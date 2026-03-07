#!/bin/bash
#SBATCH --job-name=safety_sys
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/safety_sys_%j.out
#SBATCH --error=logs/safety_sys_%j.err

set -e
cd /project2/jessetho_1732/zizhaoh/PRISM
module load conda
module load cuda/12.4.0
source activate DREAM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

echo "SAFETY (System Prompt) for 4 models"
echo "Start: $(date)"

# --- Mistral-7B-v0.3 ---
echo "=== Mistral-7B-Instruct-v0.3 ==="
python -m scripts.prism.eval_persona_granularity \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --exp_name Mistral-7B-Instruct-v0.3 \
    --persona safety_monitor \
    --benchmark safety \
    --granularity full

# --- Llama-3.1-8B ---
echo "=== Llama-3.1-8B-Instruct ==="
python -m scripts.prism.eval_persona_granularity \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --exp_name Llama-3.1-8B-Instruct \
    --persona safety_monitor \
    --benchmark safety \
    --granularity full

# --- Qwen1.5-MoE ---
echo "=== Qwen1.5-MoE-A2.7B-Chat ==="
python -m scripts.prism.eval_persona_granularity \
    --model Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --exp_name Qwen1.5-MoE-A2.7B-Chat \
    --persona safety_monitor \
    --benchmark safety \
    --granularity full

# --- DS-R1-Llama-8B ---
echo "=== DeepSeek-R1-Distill-Llama-8B ==="
python -m scripts.prism.eval_persona_granularity \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --exp_name DeepSeek-R1-Distill-Llama-8B \
    --persona safety_monitor \
    --benchmark safety \
    --granularity full

echo "DONE: $(date)"
