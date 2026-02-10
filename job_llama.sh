#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=jessetho_1732
#SBATCH --job-name=DREAM-LLaMA
#SBATCH --output=logs/llama_%j.out
#SBATCH --error=logs/llama_%j.err

# Move to project folder
cd /project2/jessetho_1732/zizhaoh/DREAM-C2L

# Create logs folder
mkdir -p logs results models dataset/synthetic

# Load modules
module purge
module load conda
module load cuda/12.1

# Activate conda environment
source activate DREAM

# Set HuggingFace cache to project folder (faster access)
export HF_HOME=/project2/jessetho_1732/zizhaoh/.cache/huggingface
export TRANSFORMERS_CACHE=/project2/jessetho_1732/zizhaoh/.cache/huggingface
mkdir -p $HF_HOME

# Run the full pipeline with LLaMA 3.2-3B
echo "Starting DREAM Pipeline with LLaMA 3.2-3B"
echo "Start time: $(date)"

python scripts/pipeline.py --model meta-llama/Llama-3.2-3B-Instruct

echo "End time: $(date)"
echo "Pipeline Complete"
