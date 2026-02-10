#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=jessetho_1732
#SBATCH --job-name=DREAM-Mistral7B
#SBATCH --output=logs/mistral7b_%j.out
#SBATCH --error=logs/mistral7b_%j.err

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


echo "Starting DREAM Pipeline with Mistral-7B"
echo "Start time: $(date)"

python scripts/pipeline.py --model mistralai/Mistral-7B-Instruct-v0.3

echo "End time: $(date)"
echo "Pipeline Complete"
