#!/bin/bash
#SBATCH --job-name=ablation_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_ablation_full_%j.out
#SBATCH --error=slurm_ablation_full_%j.err

# ============================================================
# Full Ablation Pipeline — covers ALL rows in the ablation table
#
# Uses unified scripts:
#   0_data_gen.py  (handles all modes via flags)
#   1_train.py     (--loss_mode finetune | distill | hybrid | grad_proj)
#   2_eval.py      (--experiment_type ablation, shared baselines)
#
# Modes (17 total eval configs):
#   Main path (6):  std_cd_ext, std_cd, associative, dual, rejection, trigger
#   Ratio (3):      1:1, 4:1, 1:4
#   Source (2):     selfgen, teacher
#   Loss (4):       ft, distill, hybrid, grad_proj
#   + 2 baselines:  no-context, in-context
#
# Eval metrics per row: Safety (RR), Utility (Win Rate ×2), Drift (KL)
# ============================================================

set -e

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L
source ~/.bashrc
conda activate DREAM

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_SLUG=$(basename $MODEL)
CONTEXT_FILE="dataset/context/1_general_safety.txt"
NUM_QUERIES=100
NUM_QUERIES_DUAL=200   # doubled for modes (4)+ where pos/neg compete
ABLATION_ROOT="dataset/ablation"
MODEL_ROOT="models/ablation_v3"
RESULT_ROOT="results"

# Shared eval queries
EVAL_BENIGN="dataset/eval/alpaca_benign_queries.json"
if [ ! -f "$EVAL_BENIGN" ]; then
    python -c "
from datasets import load_dataset
import json, os
ds = load_dataset('tatsu-lab/alpaca', split='train')
queries = [r['instruction'] for r in ds if r['instruction'].strip() and not r['input'].strip()][:200]
os.makedirs(os.path.dirname('$EVAL_BENIGN'), exist_ok=True)
json.dump(queries, open('$EVAL_BENIGN', 'w'), indent=2)
print(f'Saved {len(queries)} Alpaca eval queries')
"
fi

echo "============================================"
echo "PHASE 1: DATA GENERATION (0_data_gen.py)"
echo "============================================"

# --- std_cd_ext: external data (Alpaca), positive polarity ---
echo "--- std_cd_ext (external) ---"
python scripts/0_data_gen.py \
    --model $MODEL --context_file $CONTEXT_FILE \
    --output_dir "${ABLATION_ROOT}/std_cd_ext/${MODEL_SLUG}" \
    --source external --query_type random --polarity positive \
    --num_samples $NUM_QUERIES

# --- std_cd: synthetic data, positive polarity ---
echo "--- std_cd (synthetic) ---"
python scripts/0_data_gen.py \
    --model $MODEL --context_file $CONTEXT_FILE \
    --output_dir "${ABLATION_ROOT}/std_cd/${MODEL_SLUG}" \
    --source synthetic --query_type random --polarity positive \
    --num_samples $NUM_QUERIES

# --- associative: synthetic, context-related queries, positive polarity ---
echo "--- associative ---"
python scripts/0_data_gen.py \
    --model $MODEL --context_file $CONTEXT_FILE \
    --output_dir "${ABLATION_ROOT}/associative/${MODEL_SLUG}" \
    --source synthetic --query_type associative --polarity positive \
    --num_samples $NUM_QUERIES

# --- dual: associative queries + BOTH polarities (pos + neg) ---
echo "--- dual (${NUM_QUERIES_DUAL} samples) ---"
python scripts/0_data_gen.py \
    --model $MODEL --context_file $CONTEXT_FILE \
    --output_dir "${ABLATION_ROOT}/dual_v2/${MODEL_SLUG}" \
    --source synthetic --query_type associative --polarity both \
    --num_samples $NUM_QUERIES_DUAL --ratio 1 1

# --- rejection: dual + rejection sampling ---
echo "--- rejection (${NUM_QUERIES_DUAL} samples) ---"
python scripts/0_data_gen.py \
    --model $MODEL --context_file $CONTEXT_FILE \
    --output_dir "${ABLATION_ROOT}/rejection_v2/${MODEL_SLUG}" \
    --source synthetic --query_type associative --polarity both \
    --num_samples $NUM_QUERIES_DUAL --ratio 1 1 \
    --rejection_sampling

# --- trigger: rejection + trigger token (= full DREAM) ---
echo "--- trigger (${NUM_QUERIES_DUAL} samples) ---"
python scripts/0_data_gen.py \
    --model $MODEL --context_file $CONTEXT_FILE \
    --output_dir "${ABLATION_ROOT}/trigger_v2/${MODEL_SLUG}" \
    --source synthetic --query_type associative --polarity both \
    --num_samples $NUM_QUERIES_DUAL --ratio 1 1 \
    --rejection_sampling --use_trigger

# === Ratio data (subsample trigger/full-DREAM data with different pos:neg ratios) ===
echo "--- ratio variants ---"
python -c "
import json, os, random
random.seed(42)

model_slug = '${MODEL_SLUG}'
trigger_dir = f'${ABLATION_ROOT}/trigger_v2/{model_slug}'
pos = json.load(open(os.path.join(trigger_dir, 'positive_safety_data.json')))
neg = json.load(open(os.path.join(trigger_dir, 'negative_utility_data.json')))

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

# === Teacher source (use larger model to generate data) ===
TEACHER_MODEL="Qwen/Qwen2.5-7B-Instruct"
TEACHER_SLUG=$(basename $TEACHER_MODEL)
TEACHER_DATA_DIR="${ABLATION_ROOT}/teacher_data/${TEACHER_SLUG}"

echo "--- teacher data gen ---"
python scripts/0_data_gen.py \
    --model $TEACHER_MODEL --context_file $CONTEXT_FILE \
    --output_dir "$TEACHER_DATA_DIR" \
    --source synthetic --query_type associative --polarity both \
    --num_samples $NUM_QUERIES_DUAL --ratio 1 1 \
    --rejection_sampling --use_trigger

echo "============================================"
echo "PHASE 2: TRAINING (1_train.py)"
echo "============================================"

# === Main path: modes 1-3 (100 samples, 3 epochs) ===
for MODE in std_cd_ext std_cd associative; do
    DATA_DIR="${ABLATION_ROOT}/${MODE}/${MODEL_SLUG}"
    OUTPUT_DIR="${MODEL_ROOT}/${MODE}"

    if [ -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
        echo "  SKIP training $MODE"; continue
    fi

    echo "--- Training: $MODE (100 samples, 3 epochs) ---"
    python scripts/1_train.py \
        --model $MODEL --data_dir $DATA_DIR --output_dir $OUTPUT_DIR \
        --loss_mode finetune --epochs 3 --batch_size 4 --learning_rate 2e-4
done

# === Main path: modes 4-6 (200 samples, 5 epochs) ===
for MODE in dual rejection trigger; do
    DATA_DIR="${ABLATION_ROOT}/${MODE}_v2/${MODEL_SLUG}"
    OUTPUT_DIR="${MODEL_ROOT}/${MODE}"

    if [ -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
        echo "  SKIP training $MODE"; continue
    fi

    echo "--- Training: $MODE (200 samples, 5 epochs) ---"
    python scripts/1_train.py \
        --model $MODEL --data_dir $DATA_DIR --output_dir $OUTPUT_DIR \
        --loss_mode finetune --epochs 5 --batch_size 4 --learning_rate 2e-4
done

# === Ratio (3 modes) ===
for RATIO in ratio_1_1 ratio_4_1 ratio_1_4; do
    DATA_DIR="${ABLATION_ROOT}/${RATIO}/${MODEL_SLUG}"
    OUTPUT_DIR="${MODEL_ROOT}/${RATIO}"

    if [ -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
        echo "  SKIP training $RATIO"; continue
    fi

    echo "--- Training: $RATIO (5 epochs) ---"
    python scripts/1_train.py \
        --model $MODEL --data_dir $DATA_DIR --output_dir $OUTPUT_DIR \
        --loss_mode finetune --epochs 5 --batch_size 4 --learning_rate 2e-4
done

# === Teacher source (train small model on teacher data) ===
TEACHER_OUTPUT_DIR="${MODEL_ROOT}/teacher"
if [ ! -f "$TEACHER_OUTPUT_DIR/adapter_model.safetensors" ]; then
    echo "--- Training: teacher ---"
    python scripts/1_train.py \
        --model $MODEL --data_dir "$TEACHER_DATA_DIR" --output_dir "$TEACHER_OUTPUT_DIR" \
        --loss_mode finetune --epochs 5 --batch_size 4 --learning_rate 2e-4
else
    echo "  SKIP training teacher"
fi

# === Self-gen (same as trigger/full-DREAM, symlink) ===
SELFGEN_DIR="${MODEL_ROOT}/selfgen"
if [ ! -d "$SELFGEN_DIR" ]; then
    ln -s "$(realpath ${MODEL_ROOT}/trigger)" "$SELFGEN_DIR" 2>/dev/null || \
        cp -r "${MODEL_ROOT}/trigger" "$SELFGEN_DIR"
fi

# === Loss ablation (4 modes: FT, Distill, Hybrid, GradProj — all on trigger/full-DREAM data) ===
# We use trigger data so the loss ablation tests how different training objectives
# perform on the full pipeline data (including trigger tokens). This also avoids
# loss_ft being a redundant duplicate of the rejection row.
LOSS_DATA="${ABLATION_ROOT}/trigger_v2/${MODEL_SLUG}"

# loss_ft: standard cross-entropy SFT on D+ ∪ D-
if [ ! -f "${MODEL_ROOT}/loss_ft/adapter_model.safetensors" ]; then
    echo "--- Training: loss_ft ---"
    python scripts/1_train.py \
        --model $MODEL --data_dir "$LOSS_DATA" --output_dir "${MODEL_ROOT}/loss_ft" \
        --loss_mode finetune --epochs 5 --learning_rate 2e-4
fi

# loss_distill: KL divergence against base model logits on D+ ∪ D-
if [ ! -f "${MODEL_ROOT}/loss_distill/adapter_model.safetensors" ]; then
    echo "--- Training: loss_distill ---"
    python scripts/1_train.py \
        --model $MODEL --data_dir "$LOSS_DATA" --output_dir "${MODEL_ROOT}/loss_distill" \
        --loss_mode distill --epochs 5 --learning_rate 2e-4
fi

# loss_hybrid: SFT + λ·KL regularization (penalize drift while learning target)
if [ ! -f "${MODEL_ROOT}/loss_hybrid/adapter_model.safetensors" ]; then
    echo "--- Training: loss_hybrid ---"
    python scripts/1_train.py \
        --model $MODEL --data_dir "$LOSS_DATA" --output_dir "${MODEL_ROOT}/loss_hybrid" \
        --loss_mode hybrid --kl_weight 0.5 --epochs 5 --learning_rate 2e-4
fi

# loss_grad_proj: project safety gradients orthogonal to utility subspace
if [ ! -f "${MODEL_ROOT}/loss_grad_proj/adapter_model.safetensors" ]; then
    echo "--- Training: loss_grad_proj ---"
    python scripts/1_train.py \
        --model $MODEL --data_dir "$LOSS_DATA" --output_dir "${MODEL_ROOT}/loss_grad_proj" \
        --loss_mode grad_proj --epochs 5 --learning_rate 2e-4
fi

echo "============================================"
echo "PHASE 3: EVALUATION (2_eval.py)"
echo "============================================"

# --- Step 1: Generate baselines ONCE (base_no_context + base_with_context) ---
echo "--- Generating shared baselines ---"
python scripts/2_eval.py \
    --base_model $MODEL \
    --judge_model $MODEL \
    --context_file $CONTEXT_FILE \
    --benchmarks HarmBench \
    --baselines_only \
    --skip_utility --skip_kl

# --- Step 2: Per-config eval (one finetuned generation per config) ---
# Format: NAME|ADAPTER_PATH|USE_TRIGGER (1=yes, 0=no)
declare -a EVAL_CONFIGS

# Main path (6): rows 1-5 no trigger, row 6 (trigger) has trigger
EVAL_CONFIGS+=("std_cd_ext|${MODEL_ROOT}/std_cd_ext|0")
EVAL_CONFIGS+=("std_cd|${MODEL_ROOT}/std_cd|0")
EVAL_CONFIGS+=("associative|${MODEL_ROOT}/associative|0")
EVAL_CONFIGS+=("dual|${MODEL_ROOT}/dual|0")
EVAL_CONFIGS+=("rejection|${MODEL_ROOT}/rejection|0")
EVAL_CONFIGS+=("trigger|${MODEL_ROOT}/trigger|1")

# Ratio (3): variations of full DREAM → with trigger
for RATIO in ratio_1_1 ratio_4_1 ratio_1_4; do
    EVAL_CONFIGS+=("${RATIO}|${MODEL_ROOT}/${RATIO}|1")
done

# Source (2): variations of full DREAM → with trigger
EVAL_CONFIGS+=("selfgen|${MODEL_ROOT}/selfgen|1")
EVAL_CONFIGS+=("teacher|${MODEL_ROOT}/teacher|1")

# Loss (4): variations of full DREAM → with trigger
for LOSS in ft distill hybrid grad_proj; do
    EVAL_CONFIGS+=("loss_${LOSS}|${MODEL_ROOT}/loss_${LOSS}|1")
done

for CONFIG in "${EVAL_CONFIGS[@]}"; do
    NAME=$(echo $CONFIG | cut -d'|' -f1)
    ADAPTER=$(echo $CONFIG | cut -d'|' -f2)
    TRIGGER=$(echo $CONFIG | cut -d'|' -f3)

    if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
        echo "SKIP eval $NAME: no adapter"
        continue
    fi

    TRIGGER_FLAG=""
    if [ "$TRIGGER" = "1" ]; then
        TRIGGER_FLAG="--use_trigger"
    fi

    echo "--- Eval: $NAME (trigger=$TRIGGER) ---"
    python scripts/2_eval.py \
        --base_model $MODEL \
        --adapter_path "$ADAPTER" \
        --judge_model $MODEL \
        --context_file $CONTEXT_FILE \
        --benchmarks HarmBench \
        --benign_queries_file "$EVAL_BENIGN" \
        --experiment_type ablation \
        --experiment_name "$NAME" \
        --utility_limit 30 \
        $TRIGGER_FLAG
done

echo "============================================"
echo "ALL ABLATION EXPERIMENTS COMPLETE"
echo "============================================"
echo "Results in: $RESULT_ROOT/ablation/"
echo "Baselines in: $RESULT_ROOT/baselines/ (shared HarmBench base)"
echo "Each config has: safety (RR), geval, win_rate, kl_divergence"
