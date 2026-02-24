#!/bin/bash
#SBATCH --job-name=DREAM
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/dream_%j.out
#SBATCH --error=logs/dream_%j.err

# ============================================================
# PRISM Main Experiment — SLURM Job
# ============================================================
# Usage:
#   # Run all rows for one model:
#   sbatch job_table2.sh Qwen/Qwen2.5-7B-Instruct
#
#   # Run only baseline (Row 1):
#   sbatch job_table2.sh Qwen/Qwen2.5-7B-Instruct 1
#
#   # Run all personas (Rows 2-4 data):
#   sbatch job_table2.sh Qwen/Qwen2.5-7B-Instruct 2
#
#   # Run PRISM train + eval (Row 5):
#   sbatch job_table2.sh Qwen/Qwen2.5-7B-Instruct 5
# ============================================================

MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
ROW=${2:-""}

echo "==========================================="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
echo "Model: ${MODEL}"
echo "Row: ${ROW:-all}"
echo "==========================================="

cd /project2/jessetho_1732/zizhaoh/PRISM
module load conda
module load cuda/12.1
source activate DREAM

mkdir -p logs

if [ -z "${ROW}" ]; then
    python scripts/main.py --model "${MODEL}"
else
    python scripts/main.py --model "${MODEL}" --row "${ROW}"
fi

echo "Done: $(date)"
