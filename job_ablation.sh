#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=jessetho_1732
#SBATCH --job-name=DREAM-Ablation
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L
mkdir -p logs results models/ablation

module purge
module load conda
module load cuda/12.1
source activate DREAM

export HF_HOME=/project2/jessetho_1732/zizhaoh/.cache/huggingface
export TRANSFORMERS_CACHE=/project2/jessetho_1732/zizhaoh/.cache/huggingface
export HF_TOKEN=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
mkdir -p $HF_HOME

python3 -c "from huggingface_hub import login; login(token='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')"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR="dataset/synthetic/1_general_safety/Llama-3.1-8B-Instruct"
CONTEXT_FILE="dataset/context/1_general_safety.txt"

echo "============================================"
echo "DREAM Ablation: Finetune vs. Distill"
echo "Model: $MODEL"
echo "Context: 1_general_safety (General Safety)"
echo "Start: $(date)"
echo "============================================"

# ===== Step 1: Save teacher logits =====
echo ""
echo "[Step 1] Saving teacher logits..."
python scripts/1a_save_logits.py \
    --model $MODEL \
    --data_dir $DATA_DIR \
    --max_len 1024

echo "Logit extraction done: $(date)"

# ===== Step 2: Train all 4 ablation modes =====
for MODE in ft_ft ft_distill distill_ft distill_distill; do
    echo ""
    echo "============================================"
    echo "[Step 2] Training mode: $MODE"
    echo "Start: $(date)"
    echo "============================================"

    OUT_DIR="models/ablation/${MODE}"

    # Skip if already trained
    if [ -f "${OUT_DIR}/adapter_config.json" ]; then
        echo "[SKIP] Mode $MODE already trained at $OUT_DIR"
    else
        python scripts/1b_train_ablation.py \
            --model $MODEL \
            --data_dir $DATA_DIR \
            --output_dir $OUT_DIR \
            --mode $MODE \
            --epochs 3 \
            --learning_rate 2e-4 \
            --temperature 2.0

        echo "Training $MODE done: $(date)"
    fi
done

# ===== Step 3: Evaluate all 4 modes =====
BENCHMARKS=("HarmBench" "Jailbreak" "PINT" "PKU_SafeRLHF")
BENCH_FILES=("dataset/eval/harmbench_all.csv" "dataset/eval/jailbreak_prompts.json" "dataset/eval/pint_injection_prompts.json" "dataset/eval/pku_saferlhf_prompts.json")

for MODE in ft_ft ft_distill distill_ft distill_distill; do
    ADAPTER="models/ablation/${MODE}"

    # Find best checkpoint
    BEST_CKPT=$ADAPTER
    if [ -d "${ADAPTER}/checkpoint-3" ] && [ -f "${ADAPTER}/checkpoint-3/adapter_config.json" ]; then
        BEST_CKPT="${ADAPTER}/checkpoint-3"
    elif [ -d "${ADAPTER}/checkpoint-2" ] && [ -f "${ADAPTER}/checkpoint-2/adapter_config.json" ]; then
        BEST_CKPT="${ADAPTER}/checkpoint-2"
    fi

    for i in "${!BENCHMARKS[@]}"; do
        BM=${BENCHMARKS[$i]}
        BF=${BENCH_FILES[$i]}

        RESULT_DIR="results/ablation_${MODE}/${BM}/Llama-3.1-8B-Instruct_finetuned"

        if [ -f "${RESULT_DIR}/summary.json" ]; then
            echo "[SKIP] Eval $MODE / $BM already done"
            continue
        fi

        echo ""
        echo "[Step 3] Evaluating $MODE on $BM..."
        python scripts/2_eval_safety.py \
            --base_model $MODEL \
            --adapter_path $BEST_CKPT \
            --dataset_path $BF \
            --benchmark_name $BM \
            --context_file $CONTEXT_FILE \
            --output_root $RESULT_DIR
    done
done

echo ""
echo "============================================"
echo "Ablation Complete: $(date)"
echo "============================================"
