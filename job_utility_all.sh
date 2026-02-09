#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=jessetho_1732
#SBATCH --job-name=DREAM-Util
#SBATCH --output=logs/utility_%j.out
#SBATCH --error=logs/utility_%j.err

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L
mkdir -p logs

module purge
module load conda
module load cuda/12.1
source activate DREAM

export HF_HOME=/project2/jessetho_1732/zizhaoh/.cache/huggingface
export TRANSFORMERS_CACHE=/project2/jessetho_1732/zizhaoh/.cache/huggingface
export HF_TOKEN=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
mkdir -p $HF_HOME

python3 -c "from huggingface_hub import login; login(token='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')"

echo "============================================"
echo "DREAM: Utility Evaluation (All Models)"
echo "Start: $(date)"
echo "============================================"

# Models to evaluate - pairs of (HF_MODEL_NAME, CONTEXT/MODEL_SLUG/CHECKPOINT)
declare -a MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct|1_general_safety|models/1_general_safety/Qwen2.5-1.5B-Instruct/checkpoint-39"
    "Qwen/Qwen2.5-3B-Instruct|1_general_safety|models/1_general_safety/Qwen2.5-3B-Instruct/checkpoint-36"
    "meta-llama/Llama-3.2-3B-Instruct|1_general_safety|models/1_general_safety/Llama-3.2-3B-Instruct/checkpoint-21"
    "meta-llama/Llama-3.1-8B-Instruct|1_general_safety|models/1_general_safety/Llama-3.1-8B-Instruct/checkpoint-24"
    "google/gemma-2-2b-it|1_general_safety|models/1_general_safety/gemma-2-2b-it/checkpoint-39"
    "mistralai/Mistral-7B-Instruct-v0.3|1_general_safety|models/1_general_safety/Mistral-7B-Instruct-v0.3/checkpoint-39"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B|1_general_safety|models/1_general_safety/DeepSeek-R1-Distill-Qwen-1.5B/checkpoint-39"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B|1_general_safety|models/1_general_safety/DeepSeek-R1-Distill-Llama-8B/checkpoint-39"
    "Qwen/Qwen2.5-1.5B-Instruct|2_target_safety|models/2_target_safety/Qwen2.5-1.5B-Instruct/checkpoint-36"
    "Qwen/Qwen2.5-3B-Instruct|2_target_safety|models/2_target_safety/Qwen2.5-3B-Instruct/checkpoint-33"
    "meta-llama/Llama-3.1-8B-Instruct|2_target_safety|models/2_target_safety/Llama-3.1-8B-Instruct/checkpoint-24"
    "google/gemma-2-2b-it|2_target_safety|models/2_target_safety/gemma-2-2b-it/checkpoint-36"
    "mistralai/Mistral-7B-Instruct-v0.3|2_target_safety|models/2_target_safety/Mistral-7B-Instruct-v0.3/checkpoint-39"
    "Qwen/Qwen2.5-1.5B-Instruct|3_claude_safety|models/3_claude_safety/Qwen2.5-1.5B-Instruct/checkpoint-36"
    "meta-llama/Llama-3.1-8B-Instruct|3_claude_safety|models/3_claude_safety/Llama-3.1-8B-Instruct/checkpoint-30"
    "meta-llama/Llama-3.2-3B-Instruct|3_claude_safety|models/3_claude_safety/Llama-3.2-3B-Instruct/checkpoint-24"
    "google/gemma-2-2b-it|3_claude_safety|models/3_claude_safety/gemma-2-2b-it/checkpoint-39"
    "mistralai/Mistral-7B-Instruct-v0.3|3_claude_safety|models/3_claude_safety/Mistral-7B-Instruct-v0.3/checkpoint-39"
    "Qwen/Qwen2.5-1.5B-Instruct|4_claude_system|models/4_claude_system/Qwen2.5-1.5B-Instruct/checkpoint-39"
    "meta-llama/Llama-3.1-8B-Instruct|4_claude_system|models/4_claude_system/Llama-3.1-8B-Instruct/checkpoint-24"
    "meta-llama/Llama-3.2-3B-Instruct|4_claude_system|models/4_claude_system/Llama-3.2-3B-Instruct/checkpoint-21"
)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL CONTEXT ADAPTER <<< "$entry"
    MODEL_SLUG=$(basename $(dirname "$ADAPTER"))

    # Check if utility eval already done
    SUMMARY="results/${CONTEXT}/utility/${MODEL_SLUG}_finetuned/summary.json"
    if [ -f "$SUMMARY" ]; then
        echo "[SKIP] Utility already done: ${CONTEXT}/${MODEL_SLUG}"
        continue
    fi

    echo ""
    echo "--------------------------------------------"
    echo "Utility Eval: ${CONTEXT} / ${MODEL_SLUG}"
    echo "Start: $(date)"
    echo "--------------------------------------------"

    python scripts/eval_utility_standalone.py \
        --base_model "$MODEL" \
        --adapter_path "$ADAPTER" \
        --context_name "$CONTEXT" \
        --output_dir "results" \
        --limit 30

    echo "Done: $(date)"
done

echo ""
echo "============================================"
echo "Utility Eval Complete: $(date)"
echo "============================================"
