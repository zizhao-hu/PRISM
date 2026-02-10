#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=jessetho_1732
#SBATCH --job-name=DREAM-R1Q1.5B
#SBATCH --output=logs/r1_qwen1.5b_%j.out
#SBATCH --error=logs/r1_qwen1.5b_%j.err

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L
mkdir -p logs results models dataset/synthetic

module purge
module load conda
module load cuda/12.1
source activate DREAM

export HF_HOME=/project2/jessetho_1732/zizhaoh/.cache/huggingface
export TRANSFORMERS_CACHE=/project2/jessetho_1732/zizhaoh/.cache/huggingface
export HF_TOKEN=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
mkdir -p $HF_HOME

# HuggingFace login for gated models
python3 -c "from huggingface_hub import login; login(token='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')"

echo "Starting DREAM Pipeline with DeepSeek-R1-Distill-Qwen-1.5B"
echo "Start time: $(date)"

python scripts/pipeline.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

echo "End time: $(date)"
echo "Pipeline Complete"
