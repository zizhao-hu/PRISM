#!/bin/bash
#SBATCH --job-name=gated_lora
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/gated_lora_%j.out
#SBATCH --error=logs/gated_lora_%j.err

cd /project2/jessetho_1732/zizhaoh/PRISM
module load conda
module load cuda/12.4.0
source activate DREAM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "=========================================="

echo "Gated Single-LoRA PRISM pipeline"
echo "Start: $(date)"

# Run the gated LoRA pipeline
python -m scripts.prism.run_gated_lora \
    --config configs/Qwen2.5-7B-Instruct.json \
    --exp_name Qwen2.5-7B-Instruct-gated

echo "DONE: $(date)"
