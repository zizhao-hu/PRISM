#!/bin/bash
#SBATCH --job-name=qwen_inuser
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/qwen_gran_inuser_%j.out
#SBATCH --error=logs/qwen_gran_inuser_%j.err

set -e
cd /project2/jessetho_1732/zizhaoh/PRISM
module load conda
module load cuda/12.4.0
source activate DREAM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

echo "PERSONA GRANULARITY ABLATION (persona-in-user mode)"
echo "Model: Qwen/Qwen2.5-7B-Instruct"
echo "Start: $(date)"

python -m scripts.prism.eval_persona_granularity \
    --model Qwen/Qwen2.5-7B-Instruct \
    --persona-in-user

echo "DONE: $(date)"
