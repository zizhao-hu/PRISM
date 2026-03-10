#!/bin/bash
#SBATCH --job-name=token_prob
#SBATCH --output=logs/token_prob_%j.out
#SBATCH --error=logs/token_prob_%j.err
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --account=jessetho_1732

set -e
source ~/.bashrc
conda activate DREAM
cd /scratch1/zizhaoh/PRISM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export PYTHONPATH="scripts:$PYTHONPATH"
mkdir -p logs

echo "Token Probability Shift Analysis"
echo "Start: $(date)"
echo "CWD: $(pwd)"

python scripts/analysis/token_prob_shift.py

echo "Done: $(date)"
