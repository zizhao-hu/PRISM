#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=jessetho_1732
#SBATCH --job-name=DREAM-Qwen3B
#SBATCH --output=logs/qwen3b_%j.out
#SBATCH --error=logs/qwen3b_%j.err

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L
mkdir -p logs results models dataset/synthetic

module purge
module load conda
module load cuda/12.1
source activate DREAM

export HF_HOME=/project2/jessetho_1732/zizhaoh/.cache/huggingface
export TRANSFORMERS_CACHE=/project2/jessetho_1732/zizhaoh/.cache/huggingface
mkdir -p $HF_HOME

echo "Starting DREAM Pipeline with Qwen2.5-3B"
echo "Start time: $(date)"

python scripts/pipeline.py --model Qwen/Qwen2.5-3B-Instruct

echo "End time: $(date)"
echo "Pipeline Complete"
