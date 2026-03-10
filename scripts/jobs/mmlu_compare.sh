#!/bin/bash
#SBATCH --job-name=mmlu_compare
#SBATCH --output=logs/mmlu_compare_%j.out
#SBATCH --error=logs/mmlu_compare_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --account=jessetho_1732

source ~/.bashrc
conda activate DREAM
cd /project2/jessetho_1732/zizhaoh/PRISM

mkdir -p logs

echo "=== Quick MMLU Persona Comparison ==="
echo "Start: $(date)"

python scripts/quick_mmlu_compare.py

echo "End: $(date)"
