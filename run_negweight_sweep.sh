#!/bin/bash
#SBATCH --job-name=hp_negw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_hp_negweight_%j.out
#SBATCH --error=slurm_hp_negweight_%j.err

# ============================================================
# Neg-Weight Lambda Sweep: distill with N=50, 6 epochs
#
# Fixes data_size=50, epochs=6, varies --neg_weight (lambda)
# Lambda scales the loss contribution of benign/negative samples.
#   lambda=1.0  =>  equal weight (current default)
#   lambda=0.0  =>  safety-only (no benign loss)
#
# Grid: 5 lambda values = 5 train+eval runs
# ============================================================

set -e

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L
source ~/.bashrc
conda activate DREAM

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
CONTEXT_FILE="dataset/context/1_general_safety.txt"

DATA_DIR="dataset/hyperparam_v2/std_cd_50"
MODEL_ROOT="models/hyperparam_dl_negweight"
RESULT_ROOT="results"

LAMBDAS=(0.5 0.25 0.2 0.1 0)
EPOCHS=6

EVAL_BENIGN="dataset/eval/alpaca_benign_queries.json"

echo "============================================"
echo "PHASE 0: VERIFY TRAINING DATA"
echo "============================================"

if [ ! -f "${DATA_DIR}/positive_safety_data.json" ]; then
    echo "ERROR: Missing training data at ${DATA_DIR}"
    exit 1
fi
echo "  OK: std_cd_50 data exists"

echo "============================================"
echo "PHASE 1: TRAIN DISTILL WITH VARYING LAMBDA"
echo "============================================"

for LAMBDA in "${LAMBDAS[@]}"; do
    OUTPUT_DIR="${MODEL_ROOT}/lambda_${LAMBDA}"

    if [ -f "${OUTPUT_DIR}/training_complete" ]; then
        echo "  SKIP training lambda=${LAMBDA}"; continue
    fi

    echo "--- Distill: N=50, Epochs=${EPOCHS}, lambda=${LAMBDA} ---"
    python scripts/1_train.py \
        --model $MODEL \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --loss_mode distill \
        --epochs $EPOCHS \
        --batch_size 4 \
        --learning_rate 2e-4 \
        --temperature 2.0 \
        --neg_weight $LAMBDA \
        --save_every_epoch

    touch "${OUTPUT_DIR}/training_complete"
done

echo "============================================"
echo "PHASE 2: EVALUATE"
echo "============================================"

for LAMBDA in "${LAMBDAS[@]}"; do
    # Evaluate the final epoch (epoch 6)
    ADAPTER_DIR="${MODEL_ROOT}/lambda_${LAMBDA}/epoch_${EPOCHS}"
    EXP_NAME="lambda_${LAMBDA}/epoch_${EPOCHS}"

    [ ! -d "$ADAPTER_DIR" ] && echo "SKIP eval lambda=${LAMBDA}: no checkpoint" && continue

    EVAL_DIR="${RESULT_ROOT}/hyperparam_dl_negweight/${EXP_NAME}/HarmBench"
    if ls ${EVAL_DIR}/*/summary.json 1>/dev/null 2>&1; then
        echo "  SKIP eval lambda=${LAMBDA}: already done"; continue
    fi

    echo "--- Eval: lambda=${LAMBDA}, Epoch=${EPOCHS} ---"
    python scripts/2_eval.py \
        --base_model $MODEL \
        --adapter_path "$ADAPTER_DIR" \
        --judge_model $MODEL \
        --context_file $CONTEXT_FILE \
        --benchmarks HarmBench \
        --benign_queries_file "$EVAL_BENIGN" \
        --experiment_type hyperparam_dl_negweight \
        --experiment_name "$EXP_NAME" \
        --utility_limit 100
done

echo "============================================"
echo "PHASE 3: COLLECT RESULTS"
echo "============================================"

python -c "
import json, os, glob

results = []
for lam in [0.5, 0.25, 0.2, 0.1, 0]:
    ename = f'lambda_{lam}/epoch_6'
    rd = f'${RESULT_ROOT}/hyperparam_dl_negweight/{ename}'

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

    e = {'neg_weight': lam, 'safety_rr': safety_rr,
         'win_rate': win_rate, 'kl_mean': kl_mean}
    results.append(e)
    print(f'  lambda={lam}: RR={safety_rr}, Win={win_rate}, KL={kl_mean}')

os.makedirs('${RESULT_ROOT}/hyperparam_dl_negweight', exist_ok=True)
json.dump(results, open('${RESULT_ROOT}/hyperparam_dl_negweight/summary.json', 'w'), indent=2)
print(f'Saved {len(results)} results')
"

echo "============================================"
echo "NEG-WEIGHT LAMBDA SWEEP COMPLETE"
echo "============================================"
