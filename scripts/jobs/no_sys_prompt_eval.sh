#!/bin/bash
#SBATCH --job-name=no_sys_prompt_eval
#SBATCH --output=logs/no_sys_prompt_%j.out
#SBATCH --error=logs/no_sys_prompt_%j.err
#SBATCH --partition=nlp
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --account=jessetho_1732

source ~/.bashrc
conda activate DREAM
cd /project2/jessetho_1732/zizhaoh/PRISM

mkdir -p logs

echo "=== No-System-Prompt Eval ==="
echo "Start: $(date)"

python scripts/eval/run_no_system_prompt_eval.py

echo "End: $(date)"
