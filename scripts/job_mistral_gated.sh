#!/bin/bash
#SBATCH --job-name=mistral_eval
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=logs/mistral_gated_eval_%j.out
#SBATCH --error=logs/mistral_gated_eval_%j.err

cd /project2/jessetho_1732/zizhaoh/PRISM
module load conda
module load cuda/12.4.0
source activate DREAM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

echo "Mistral Gated LoRA EVAL-ONLY (with gate routing)"
echo "Start: $(date)"

python -m scripts.prism.run_gated_lora \
    --config configs/Mistral-7B-Instruct-v0.3.json \
    --eval_only

echo "End: $(date)"
