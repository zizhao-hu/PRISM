#!/bin/bash
#SBATCH --job-name=hp_first_tok
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_hp_first_token_%j.out
#SBATCH --error=slurm_hp_first_token_%j.err

# ============================================================
# Hyperparameter Search: FIRST-TOKEN-ONLY mode
#
# Truncates training outputs to just the first word (e.g. "Sorry",
# "Sure", "I") and removes EOS token. Tests whether directional
# steering alone (without full response memorization) is sufficient.
#
# Runs both finetune and distill with --first_token_only flag.
# Reuses training data from dataset/hyperparam_v2/
#
# Grid: 4 data sizes × 5 epoch checkpoints × 2 methods = 40 evaluations
# ============================================================

set -e

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L
source ~/.bashrc
conda activate DREAM

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
CONTEXT_FILE="dataset/context/1_general_safety.txt"

DATA_ROOT="dataset/hyperparam_v2"
RESULT_ROOT="results"

DATA_SIZES=(50 100 200 500)
EVAL_EPOCHS=(2 4 6 8 10)
MAX_EPOCHS=10

EVAL_BENIGN="dataset/eval/alpaca_benign_queries.json"

echo "============================================"
echo "PHASE 0: VERIFY TRAINING DATA EXISTS"
echo "============================================"

for N in "${DATA_SIZES[@]}"; do
    DATA_DIR="${DATA_ROOT}/std_cd_${N}"
    if [ ! -f "${DATA_DIR}/positive_safety_data.json" ]; then
        echo "ERROR: Missing training data at ${DATA_DIR}"
        echo "Run run_hyperparam_search.sh first to generate data."
        exit 1
    fi
    echo "  OK: std_cd_${N} data exists"
done

# ============================================================
# PART A: FINETUNE with first-token-only
# ============================================================

FT_MODEL_ROOT="models/hyperparam_ft_first_token"
FT_EXP_TYPE="hyperparam_ft_first_token"

echo "============================================"
echo "PHASE 1A: TRAIN FINETUNE + FIRST-TOKEN-ONLY"
echo "============================================"

for N in "${DATA_SIZES[@]}"; do
    DATA_DIR="${DATA_ROOT}/std_cd_${N}"
    OUTPUT_DIR="${FT_MODEL_ROOT}/std_cd_${N}"

    if [ -f "${OUTPUT_DIR}/training_complete" ]; then
        echo "  SKIP training finetune std_cd_${N}"; continue
    fi

    echo "--- Finetune (first-token) Training: std_cd_${N} ---"
    python scripts/1_train.py \
        --model $MODEL \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --loss_mode finetune \
        --epochs $MAX_EPOCHS \
        --batch_size 4 \
        --learning_rate 2e-4 \
        --save_every_epoch \
        --first_token_only

    touch "${OUTPUT_DIR}/training_complete"
done

echo "============================================"
echo "PHASE 2A: EVALUATE FINETUNE + FIRST-TOKEN"
echo "============================================"

for N in "${DATA_SIZES[@]}"; do
    for EP in "${EVAL_EPOCHS[@]}"; do
        ADAPTER_DIR="${FT_MODEL_ROOT}/std_cd_${N}/epoch_${EP}"
        EXP_NAME="std_cd_${N}/epoch_${EP}"

        [ ! -d "$ADAPTER_DIR" ] && echo "SKIP eval ${EXP_NAME}: no checkpoint" && continue

        echo "--- Eval FT-FirstToken: N=${N}, Epoch=${EP} ---"
        python scripts/2_eval.py \
            --base_model $MODEL \
            --adapter_path "$ADAPTER_DIR" \
            --judge_model $MODEL \
            --context_file $CONTEXT_FILE \
            --benchmarks HarmBench \
            --benign_queries_file "$EVAL_BENIGN" \
            --experiment_type $FT_EXP_TYPE \
            --experiment_name "$EXP_NAME" \
            --utility_limit 100
    done
done

# ============================================================
# PART B: DISTILL with first-token-only
# ============================================================

DL_MODEL_ROOT="models/hyperparam_dl_first_token"
DL_EXP_TYPE="hyperparam_dl_first_token"

echo "============================================"
echo "PHASE 1B: TRAIN DISTILL + FIRST-TOKEN-ONLY"
echo "============================================"

for N in "${DATA_SIZES[@]}"; do
    DATA_DIR="${DATA_ROOT}/std_cd_${N}"
    OUTPUT_DIR="${DL_MODEL_ROOT}/std_cd_${N}"

    if [ -f "${OUTPUT_DIR}/training_complete" ]; then
        echo "  SKIP training distill std_cd_${N}"; continue
    fi

    echo "--- Distill (first-token) Training: std_cd_${N} ---"
    python scripts/1_train.py \
        --model $MODEL \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --loss_mode distill \
        --epochs $MAX_EPOCHS \
        --batch_size 4 \
        --learning_rate 2e-4 \
        --temperature 2.0 \
        --save_every_epoch \
        --first_token_only

    touch "${OUTPUT_DIR}/training_complete"
done

echo "============================================"
echo "PHASE 2B: EVALUATE DISTILL + FIRST-TOKEN"
echo "============================================"

for N in "${DATA_SIZES[@]}"; do
    for EP in "${EVAL_EPOCHS[@]}"; do
        ADAPTER_DIR="${DL_MODEL_ROOT}/std_cd_${N}/epoch_${EP}"
        EXP_NAME="std_cd_${N}/epoch_${EP}"

        [ ! -d "$ADAPTER_DIR" ] && echo "SKIP eval ${EXP_NAME}: no checkpoint" && continue

        echo "--- Eval DL-FirstToken: N=${N}, Epoch=${EP} ---"
        python scripts/2_eval.py \
            --base_model $MODEL \
            --adapter_path "$ADAPTER_DIR" \
            --judge_model $MODEL \
            --context_file $CONTEXT_FILE \
            --benchmarks HarmBench \
            --benign_queries_file "$EVAL_BENIGN" \
            --experiment_type $DL_EXP_TYPE \
            --experiment_name "$EXP_NAME" \
            --utility_limit 100
    done
done

echo "============================================"
echo "PHASE 3: COLLECT ALL RESULTS"
echo "============================================"

python -c "
import json, os, glob

for method, exp_type in [('finetune', '${FT_EXP_TYPE}'), ('distill', '${DL_EXP_TYPE}')]:
    print(f'\n=== {method.upper()} + FIRST-TOKEN ===')
    results = []
    for n in [50, 100, 200, 500]:
        for ep in [2, 4, 6, 8, 10]:
            ename = f'std_cd_{n}/epoch_{ep}'
            rd = f'${RESULT_ROOT}/{exp_type}/{ename}'

            # Safety
            sfs = glob.glob(os.path.join(rd, 'HarmBench', '*', 'summary.json'))
            safety_rr = 0
            if sfs:
                d = json.load(open(sfs[0]))
                ft_nt = d.get('safety_scores', {}).get('finetuned', {})
                safety_rr = round(ft_nt.get('mean', 0) * 100, 1)

            # Utility
            util_files = glob.glob(os.path.join(rd, 'utility', '*'))
            win_rate, kl_mean = 0, 0
            if util_files:
                ud = util_files[0]
                wr_f = os.path.join(ud, 'winrate_vs_base.json')
                kl_f = os.path.join(ud, 'kl_divergence.json')
                if os.path.exists(wr_f):
                    win_rate = json.load(open(wr_f)).get('win_rate', 0)
                if os.path.exists(kl_f):
                    kl_mean = json.load(open(kl_f)).get('mean', 0)

            e = {'method': method, 'data_size': n, 'epochs': ep,
                 'safety_rr': safety_rr, 'win_rate': win_rate, 'kl_mean': kl_mean}
            results.append(e)
            print(f'  n={n}, ep={ep}: RR={safety_rr}, Win={win_rate}, KL={kl_mean}')

    os.makedirs(f'${RESULT_ROOT}/{exp_type}', exist_ok=True)
    json.dump(results, open(f'${RESULT_ROOT}/{exp_type}/summary.json', 'w'), indent=2)
"

echo "============================================"
echo "FIRST-TOKEN HYPERPARAMETER SEARCH COMPLETE"
echo "============================================"
