#!/bin/bash
#SBATCH --job-name=usr_ds
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/user_ds_%j.out
#SBATCH --error=logs/user_ds_%j.err

cd /project2/jessetho_1732/zizhaoh/PRISM
module load conda
module load cuda/12.4.0
source activate DREAM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

MATCHED="writing roleplay reasoning math coding extraction stem humanities"

echo "DeepSeek R1 user-position evaluations"
echo "Start: $(date)"

# --- DS-R1-Qwen-7B ---
echo "=== DeepSeek-R1-Distill-Qwen-7B (USER) ==="
python -m scripts.prism.eval_persona_granularity \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --exp_name DeepSeek-R1-Distill-Qwen-7B_user \
    --persona $MATCHED safety_monitor \
    --benchmark mt_bench safety \
    --granularity full \
    --persona-in-user

# --- DS-R1-Llama-8B ---
echo "=== DeepSeek-R1-Distill-Llama-8B (USER) ==="
python -m scripts.prism.eval_persona_granularity \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --exp_name DeepSeek-R1-Distill-Llama-8B_user \
    --persona $MATCHED safety_monitor \
    --benchmark mt_bench safety \
    --granularity full \
    --persona-in-user

echo "DONE: $(date)"
