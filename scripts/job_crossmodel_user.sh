#!/bin/bash
#SBATCH --job-name=usr_all
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/user_all_%j.out
#SBATCH --error=logs/user_all_%j.err

set -e
cd /project2/jessetho_1732/zizhaoh/PRISM
module load conda
module load cuda/12.4.0
source activate DREAM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

# MT-Bench matched personas = writing roleplay reasoning math coding extraction stem humanities
# Safety persona = safety_monitor
# All at user prompt position (--persona-in-user)
MATCHED="writing roleplay reasoning math coding extraction stem humanities"

echo "USER-POSITION evaluations (mt_bench + safety) for 5 models"
echo "Start: $(date)"

# --- Qwen2.5-7B (already has system data) ---
echo "=== Qwen2.5-7B-Instruct (USER) ==="
python -m scripts.prism.eval_persona_granularity \
    --model Qwen/Qwen2.5-7B-Instruct \
    --exp_name Qwen2.5-7B-Instruct_user \
    --persona $MATCHED safety_monitor \
    --benchmark mt_bench safety \
    --granularity full \
    --persona-in-user

# --- Llama-3.1-8B ---
echo "=== Llama-3.1-8B-Instruct (USER) ==="
python -m scripts.prism.eval_persona_granularity \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --exp_name Llama-3.1-8B-Instruct_user \
    --persona $MATCHED safety_monitor \
    --benchmark mt_bench safety \
    --granularity full \
    --persona-in-user

# --- Qwen1.5-MoE ---
echo "=== Qwen1.5-MoE-A2.7B-Chat (USER) ==="
python -m scripts.prism.eval_persona_granularity \
    --model Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --exp_name Qwen1.5-MoE-A2.7B-Chat_user \
    --persona $MATCHED safety_monitor \
    --benchmark mt_bench safety \
    --granularity full \
    --persona-in-user

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
