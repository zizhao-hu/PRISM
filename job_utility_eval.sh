#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=jessetho_1732
#SBATCH --job-name=DREAM-Util
#SBATCH --output=logs/utility_eval_%j.out
#SBATCH --error=logs/utility_eval_%j.err

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L
mkdir -p logs results

module purge
module load conda
module load cuda/12.1
source activate DREAM

export HF_HOME=/project2/jessetho_1732/zizhaoh/.cache/huggingface
export TRANSFORMERS_CACHE=/project2/jessetho_1732/zizhaoh/.cache/huggingface
export HF_TOKEN=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
mkdir -p $HF_HOME

python3 -c "from huggingface_hub import login; login(token='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')"

echo "Starting Utility Evaluation for all completed models"
echo "Start time: $(date)"

# Evaluate all models that have trained adapters
for model_dir in models/1_general_safety/*/; do
    model_name=$(basename "$model_dir")
    echo ""
    echo "=========================================="
    echo "Evaluating utility for: $model_name"
    echo "=========================================="
    
    # Map directory name to HuggingFace model ID
    case "$model_name" in
        "Qwen2.5-1.5B-Instruct")
            base_model="Qwen/Qwen2.5-1.5B-Instruct"
            ;;
        "Qwen2.5-3B-Instruct")
            base_model="Qwen/Qwen2.5-3B-Instruct"
            ;;
        "Llama-3.2-3B-Instruct")
            base_model="meta-llama/Llama-3.2-3B-Instruct"
            ;;
        "Llama-3.1-8B-Instruct")
            base_model="meta-llama/Llama-3.1-8B-Instruct"
            ;;
        "Gemma-2-2B-it" | "gemma-2-2b-it")
            base_model="google/gemma-2-2b-it"
            ;;
        "Mistral-7B-Instruct-v0.3")
            base_model="mistralai/Mistral-7B-Instruct-v0.3"
            ;;
        "DeepSeek-R1-Distill-Qwen-1.5B")
            base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            ;;
        "DeepSeek-R1-Distill-Qwen-7B")
            base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            ;;
        "DeepSeek-R1-Distill-Llama-8B")
            base_model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
            ;;
        *)
            echo "Unknown model: $model_name, skipping"
            continue
            ;;
    esac
    
    # Check if utility results already exist
    result_file="results/1_general_safety/utility/${model_name}_finetuned/utility_results.json"
    if [ -f "$result_file" ]; then
        echo "Utility results already exist, skipping: $result_file"
        continue
    fi
    
    python scripts/eval_utility_standalone.py \
        --base_model "$base_model" \
        --adapter_path "$model_dir" \
        --context_name 1_general_safety \
        --limit 30
    
    echo "Completed: $model_name"
done

echo ""
echo "End time: $(date)"
echo "All utility evaluations complete!"
