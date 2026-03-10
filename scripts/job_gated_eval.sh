#!/bin/bash
#SBATCH --job-name=gated_eval
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=logs/gated_eval_%j.out
#SBATCH --error=logs/gated_eval_%j.err
#SBATCH --account=jessetho_1732

cd /scratch1/zizhaoh/PRISM
module load conda
module load cuda/12.4.0
source activate DREAM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="scripts:$PYTHONPATH"
mkdir -p logs

echo "Gated LoRA EVAL ONLY"
echo "Start: $(date)"

python -m scripts.prism.run_gated_lora \
    --config configs/Qwen1.5-MoE-A2.7B-Chat.json \
    --eval_only

echo "End: $(date)"
