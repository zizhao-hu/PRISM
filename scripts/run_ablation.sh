#!/bin/bash
#SBATCH --job-name=ablation_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_ablation_full_%j.out
#SBATCH --error=slurm_ablation_full_%j.err

# ============================================================
# Full Ablation Pipeline — covers ALL rows in the ablation table
#
# Main (5):    std_cd, associative, dual, rejection, trigger
# Ratio (3):   1:1, 4:1, 1:4
# Source (2):  self-gen, teacher
# Loss (4):   ft_ft, ft_distill, distill_ft, distill_distill
#
# Eval metrics per row: Safety (ASR), Utility (Win Rate), Drift (KL)
# ============================================================

set -e

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L
source ~/.bashrc
conda activate DREAM

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_SLUG=$(basename $MODEL)
CONTEXT_FILE="dataset/context/1_general_safety.txt"
NUM_QUERIES=100
ABLATION_ROOT="dataset/ablation"
MODEL_ROOT="models/ablation_v2"
RESULT_ROOT="results/ablation_v2"

# Shared benign queries for utility/KL eval (from dual mode)
BENIGN_DATA_DIR="${ABLATION_ROOT}/dual/${MODEL_SLUG}"

echo "============================================"
echo "PHASE 1: DATA GENERATION"
echo "============================================"

# === Main path data (modes 1-5, sequential dependency) ===
for MODE in std_cd associative dual rejection trigger; do
    echo "--- Data gen: $MODE ---"
    python scripts/0b_ablation_data_gen.py \
        --model $MODEL \
        --mode $MODE \
        --context_file $CONTEXT_FILE \
        --num_queries $NUM_QUERIES
done

# === Ratio data (subsample rejection data to target ratios) ===
echo "--- Generating ratio data ---"
python -c "
import json, os, random
random.seed(42)

model_slug = '${MODEL_SLUG}'
rej_dir = f'${ABLATION_ROOT}/rejection/{model_slug}'
pos = json.load(open(os.path.join(rej_dir, 'positive_safety_data.json')))
neg = json.load(open(os.path.join(rej_dir, 'negative_utility_data.json')))

ratios = {'ratio_1_1': (1, 1), 'ratio_4_1': (4, 1), 'ratio_1_4': (1, 4)}

for name, (pr, nr) in ratios.items():
    out_dir = os.path.join('${ABLATION_ROOT}', name, model_slug)
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, 'generation_config.json')
    if os.path.exists(config_path):
        cfg = json.load(open(config_path))
        if cfg.get('completed'):
            print(f'  {name}: SKIP (exists)')
            continue

    total = len(pos) + len(neg)
    target_pos = int(total * pr / (pr + nr))
    target_neg = total - target_pos
    sampled_pos = (pos * ((target_pos // len(pos)) + 1))[:target_pos] if target_pos > 0 else []
    sampled_neg = (neg * ((target_neg // len(neg)) + 1))[:target_neg] if target_neg > 0 else []

    json.dump(sampled_pos, open(os.path.join(out_dir, 'positive_safety_data.json'), 'w'), indent=2)
    json.dump(sampled_neg, open(os.path.join(out_dir, 'negative_utility_data.json'), 'w'), indent=2)
    json.dump({'model': '${MODEL}', 'mode': name, 'ratio': f'{pr}:{nr}',
               'pos_count': len(sampled_pos), 'neg_count': len(sampled_neg),
               'completed': True}, open(config_path, 'w'), indent=2)
    print(f'  {name}: pos={len(sampled_pos)}, neg={len(sampled_neg)}')
"

echo "============================================"
echo "PHASE 2: TRAINING"
echo "============================================"

# === Main path training (5 modes) ===
for MODE in std_cd associative dual rejection trigger; do
    DATA_DIR="${ABLATION_ROOT}/${MODE}/${MODEL_SLUG}"
    OUTPUT_DIR="${MODEL_ROOT}/${MODE}"

    if [ -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
        echo "  SKIP training $MODE (adapter exists)"
        continue
    fi

    echo "--- Training: $MODE ---"
    python scripts/1_train.py \
        --model $MODEL \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --epochs 3 \
        --batch_size 4 \
        --learning_rate 2e-4
done

# === Ratio training (3 modes) ===
for RATIO in ratio_1_1 ratio_4_1 ratio_1_4; do
    DATA_DIR="${ABLATION_ROOT}/${RATIO}/${MODEL_SLUG}"
    OUTPUT_DIR="${MODEL_ROOT}/${RATIO}"

    if [ -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
        echo "  SKIP training $RATIO (adapter exists)"
        continue
    fi

    echo "--- Training: $RATIO ---"
    python scripts/1_train.py \
        --model $MODEL \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --epochs 3 \
        --batch_size 4 \
        --learning_rate 2e-4
done

# === Source: Teacher (use larger model to generate data, train small model on it) ===
TEACHER_MODEL="Qwen/Qwen2.5-7B-Instruct"
TEACHER_SLUG=$(basename $TEACHER_MODEL)
TEACHER_OUTPUT_DIR="${MODEL_ROOT}/teacher"

if [ ! -f "$TEACHER_OUTPUT_DIR/adapter_model.safetensors" ]; then
    echo "--- Data gen: Teacher (${TEACHER_MODEL}) ---"
    
    # Need dual first for rejection to work
    python scripts/0b_ablation_data_gen.py \
        --model $TEACHER_MODEL \
        --mode dual \
        --context_file $CONTEXT_FILE \
        --num_queries $NUM_QUERIES \
        --output_root "${ABLATION_ROOT}/teacher_data"

    python scripts/0b_ablation_data_gen.py \
        --model $TEACHER_MODEL \
        --mode rejection \
        --context_file $CONTEXT_FILE \
        --num_queries $NUM_QUERIES \
        --output_root "${ABLATION_ROOT}/teacher_data"

    # Train the SMALL model on teacher-generated data
    TEACHER_FINAL_DATA="${ABLATION_ROOT}/teacher_data/rejection/${TEACHER_SLUG}"
    echo "--- Training: Teacher source ---"
    python scripts/1_train.py \
        --model $MODEL \
        --data_dir $TEACHER_FINAL_DATA \
        --output_dir $TEACHER_OUTPUT_DIR \
        --epochs 3 \
        --batch_size 4 \
        --learning_rate 2e-4
else
    echo "  SKIP training teacher (adapter exists)"
fi

# === Self-Gen source (same as rejection, just symlink) ===
SELFGEN_DIR="${MODEL_ROOT}/selfgen"
if [ ! -d "$SELFGEN_DIR" ]; then
    ln -s "$(realpath ${MODEL_ROOT}/rejection)" "$SELFGEN_DIR" 2>/dev/null || \
        cp -r "${MODEL_ROOT}/rejection" "$SELFGEN_DIR"
fi

# === Loss ablation (4 combinations) ===
REJECTION_DATA="${ABLATION_ROOT}/rejection/${MODEL_SLUG}"
LOGITS_POS="${REJECTION_DATA}/positive_safety_logits.pt"
if [ ! -f "$LOGITS_POS" ]; then
    echo "--- Saving teacher logits ---"
    python scripts/1a_save_logits.py \
        --model $MODEL \
        --data_dir $REJECTION_DATA
fi

for LOSS_MODE in ft_ft ft_distill distill_ft distill_distill; do
    OUTPUT_DIR="${MODEL_ROOT}/loss_${LOSS_MODE}"

    if [ -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
        echo "  SKIP training loss_$LOSS_MODE (adapter exists)"
        continue
    fi

    echo "--- Training: Loss $LOSS_MODE ---"
    python scripts/1b_train_ablation.py \
        --model $MODEL \
        --data_dir $REJECTION_DATA \
        --output_dir $OUTPUT_DIR \
        --mode $LOSS_MODE \
        --epochs 3 \
        --learning_rate 2e-4
done

echo "============================================"
echo "PHASE 3: EVALUATION (Safety + Win Rate + KL)"
echo "============================================"

# Collect all adapter dirs to evaluate
# Format: NAME|ADAPTER_PATH
declare -a EVAL_CONFIGS

# Main (5)
for MODE in std_cd associative dual rejection trigger; do
    EVAL_CONFIGS+=("${MODE}|${MODEL_ROOT}/${MODE}")
done

# Ratio (3)
for RATIO in ratio_1_1 ratio_4_1 ratio_1_4; do
    EVAL_CONFIGS+=("${RATIO}|${MODEL_ROOT}/${RATIO}")
done

# Source (2)
EVAL_CONFIGS+=("selfgen|${MODEL_ROOT}/selfgen")
EVAL_CONFIGS+=("teacher|${MODEL_ROOT}/teacher")

# Loss (4)
for LOSS in ft_ft ft_distill distill_ft distill_distill; do
    EVAL_CONFIGS+=("loss_${LOSS}|${MODEL_ROOT}/loss_${LOSS}")
done

for CONFIG in "${EVAL_CONFIGS[@]}"; do
    NAME=$(echo $CONFIG | cut -d'|' -f1)
    ADAPTER=$(echo $CONFIG | cut -d'|' -f2)

    if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
        echo "SKIP eval $NAME: no adapter"
        continue
    fi

    RESULT_DIR="${RESULT_ROOT}/${NAME}"

    echo "--- Eval: $NAME (Safety + Utility + KL) ---"
    python scripts/2_eval_safety.py \
        --base_model $MODEL \
        --adapter_path $ADAPTER \
        --judge_model $MODEL \
        --context_file $CONTEXT_FILE \
        --data_dir $BENIGN_DATA_DIR \
        --output_root "$RESULT_DIR"
done

echo "============================================"
echo "ALL ABLATION EXPERIMENTS COMPLETE"
echo "============================================"
echo "Results saved to: $RESULT_ROOT"
echo "Each result dir contains summary.json with:"
echo "  - safety_scores (ASR / refusal rate)"
echo "  - win_rate (DREAM vs Base)"
echo "  - kl_divergence (drift)"
