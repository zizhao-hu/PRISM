#!/bin/bash
#SBATCH --job-name=ablation_main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_ablation_%j.out
#SBATCH --error=slurm_ablation_%j.err

# ============================================================
# Ablation Pipeline: Data Gen -> Training -> Evaluation
# Runs ALL 5 progressive ablation modes sequentially
# ============================================================

set -e

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L

# Activate conda
source ~/.bashrc
conda activate dream

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
CONTEXT_FILE="dataset/context/1_general_safety.txt"
NUM_QUERIES=100

echo "============================================"
echo "PHASE 1: DATA GENERATION (5 modes)"
echo "============================================"

# Modes must run in order: dual before rejection, rejection before trigger
for MODE in std_cd associative dual rejection trigger; do
    echo "--- Generating data: $MODE ---"
    python scripts/0b_ablation_data_gen.py \
        --model $MODEL \
        --mode $MODE \
        --context_file $CONTEXT_FILE \
        --num_queries $NUM_QUERIES
done

echo "============================================"
echo "PHASE 2: TRAINING (5 modes)"
echo "============================================"

for MODE in std_cd associative dual rejection trigger; do
    DATA_DIR="dataset/ablation/${MODE}/$(basename $MODEL)"
    OUTPUT_DIR="models/ablation_v2/${MODE}"

    echo "--- Training: $MODE ---"

    if [ "$MODE" = "trigger" ]; then
        # Mode 5: use trigger token (add --use_trigger flag)
        python scripts/1_train.py \
            --model $MODEL \
            --data_dir $DATA_DIR \
            --output_dir $OUTPUT_DIR \
            --epochs 3 \
            --batch_size 4 \
            --learning_rate 2e-4
    elif [ "$MODE" = "std_cd" ] || [ "$MODE" = "associative" ]; then
        # Modes 1-2: positive data only (no Q-)
        python scripts/1_train.py \
            --model $MODEL \
            --data_dir $DATA_DIR \
            --output_dir $OUTPUT_DIR \
            --epochs 3 \
            --batch_size 4 \
            --learning_rate 2e-4
    else
        # Modes 3-4: dual data (Q+ and Q-)
        python scripts/1_train.py \
            --model $MODEL \
            --data_dir $DATA_DIR \
            --output_dir $OUTPUT_DIR \
            --epochs 3 \
            --batch_size 4 \
            --learning_rate 2e-4
    fi
done

echo "============================================"
echo "PHASE 3: EVALUATION (5 modes)"
echo "============================================"

for MODE in std_cd associative dual rejection trigger; do
    ADAPTER_DIR="models/ablation_v2/${MODE}"

    if [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
        echo "SKIP eval for $MODE: no adapter found"
        continue
    fi

    echo "--- Evaluating: $MODE ---"
    python scripts/2_eval_safety.py \
        --base_model $MODEL \
        --adapter_path $ADAPTER_DIR \
        --judge_model $MODEL \
        --context_file $CONTEXT_FILE \
        --output_root "results/ablation_v2_${MODE}"
done

echo "============================================"
echo "ALL ABLATION EXPERIMENTS COMPLETE"
echo "============================================"
