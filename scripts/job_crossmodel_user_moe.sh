#!/bin/bash
#SBATCH --job-name=usr_moe
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/user_moe_%j.out
#SBATCH --error=logs/user_moe_%j.err

cd /project2/jessetho_1732/zizhaoh/PRISM
module load conda
module load cuda/12.4.0
source activate DREAM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

MATCHED="writing roleplay reasoning math coding extraction stem humanities"

echo "Qwen1.5-MoE user-position (A100 for OOM fix)"
echo "Start: $(date)"

echo "=== Qwen1.5-MoE-A2.7B-Chat (USER) ==="
python -m scripts.prism.eval_persona_granularity \
    --model Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --exp_name Qwen1.5-MoE-A2.7B-Chat_user \
    --persona $MATCHED safety_monitor \
    --benchmark mt_bench safety \
    --granularity full \
    --persona-in-user

echo "DONE: $(date)"
