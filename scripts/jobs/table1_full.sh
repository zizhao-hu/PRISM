#!/bin/bash
#SBATCH --job-name=table1_all
#SBATCH --output=logs/table1_all_%j.out
#SBATCH --error=logs/table1_all_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --account=jessetho_1732

set -e
source ~/.bashrc
conda activate DREAM
cd /project2/jessetho_1732/zizhaoh/PRISM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

MODEL=$1  # e.g. "Qwen/Qwen2.5-7B-Instruct"
CONFIG=$2  # e.g. "configs/Qwen2.5-7B-Instruct.json"

echo "=========================================="
echo "TABLE 1: Full pipeline for $MODEL"
echo "Config: $CONFIG"
echo "Start: $(date)"
echo "=========================================="

# ---- Row 1: Baseline ----
echo ""
echo "=== ROW 1: BASELINE ==="
python scripts/main.py --model "$MODEL" --row 1

# ---- Rows 2-4: All 12 personas ----
echo ""
echo "=== ROWS 2-4: ALL PERSONAS ==="
python scripts/main.py --model "$MODEL" --row 2

# ---- Row 5: PRISM (Gated LoRA) ----
echo ""
echo "=== ROW 5: PRISM (Gated LoRA) ==="
python -m scripts.prism.run_gated_lora --config "$CONFIG"

echo ""
echo "=========================================="
echo "DONE: $(date)"
echo "=========================================="
