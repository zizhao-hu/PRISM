#!/bin/bash
#SBATCH --job-name=hp_distill
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_hp_distill_%j.out
#SBATCH --error=slurm_hp_distill_%j.err

# ============================================================
# Hyperparameter Search: data_size × epochs for DISTILLATION
#
# Same grid as finetune search, but uses --loss_mode distill
# Reuses training data from dataset/hyperparam_v2/
#
#   1_train.py     --loss_mode distill --save_every_epoch
#   2_eval.py      --experiment_type hyperparam_distill
#
# Grid: 4 data sizes × 5 epoch checkpoints = 20 evaluations
# ============================================================

set -e

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L
source ~/.bashrc
conda activate DREAM

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
CONTEXT_FILE="dataset/context/1_general_safety.txt"

DATA_ROOT="dataset/hyperparam_v2"
MODEL_ROOT="models/hyperparam_distill"
RESULT_ROOT="results"

DATA_SIZES=(50 100 200 500)
EVAL_EPOCHS=(2 4 6 8 10)
MAX_EPOCHS=10

echo "============================================"
echo "PHASE 0: VERIFY TRAINING DATA EXISTS"
echo "============================================"

EVAL_BENIGN="dataset/eval/alpaca_benign_queries.json"

for N in "${DATA_SIZES[@]}"; do
    DATA_DIR="${DATA_ROOT}/std_cd_${N}"
    if [ ! -f "${DATA_DIR}/positive_safety_data.json" ]; then
        echo "ERROR: Missing training data at ${DATA_DIR}"
        echo "Run run_hyperparam_search.sh first to generate data."
        exit 1
    fi
    echo "  OK: std_cd_${N} data exists"
done

echo "============================================"
echo "PHASE 1: TRAIN WITH DISTILLATION (1_train.py)"
echo "============================================"

for N in "${DATA_SIZES[@]}"; do
    DATA_DIR="${DATA_ROOT}/std_cd_${N}"
    OUTPUT_DIR="${MODEL_ROOT}/std_cd_${N}"

    if [ -f "${OUTPUT_DIR}/training_complete" ]; then
        echo "  SKIP training std_cd_${N}"; continue
    fi

    echo "--- Distill Training: std_cd_${N} ---"
    python scripts/1_train.py \
        --model $MODEL \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --loss_mode distill \
        --epochs $MAX_EPOCHS \
        --batch_size 4 \
        --learning_rate 2e-4 \
        --temperature 2.0 \
        --save_every_epoch

    touch "${OUTPUT_DIR}/training_complete"
done

echo "============================================"
echo "PHASE 2: EVALUATE (2_eval.py)"
echo "============================================"

for N in "${DATA_SIZES[@]}"; do
    for EP in "${EVAL_EPOCHS[@]}"; do
        ADAPTER_DIR="${MODEL_ROOT}/std_cd_${N}/epoch_${EP}"
        EXP_NAME="std_cd_${N}/epoch_${EP}"

        [ ! -d "$ADAPTER_DIR" ] && echo "SKIP eval ${EXP_NAME}: no checkpoint" && continue

        # Check if already done
        EVAL_DIR="${RESULT_ROOT}/hyperparam_distill/${EXP_NAME}/HarmBench"
        [ -f "${EVAL_DIR}/*/summary.json" ] 2>/dev/null && echo "  SKIP eval ${EXP_NAME}" && continue

        echo "--- Eval: N=${N}, Epoch=${EP} ---"
        python scripts/2_eval.py \
            --base_model $MODEL \
            --adapter_path "$ADAPTER_DIR" \
            --judge_model $MODEL \
            --context_file $CONTEXT_FILE \
            --benchmarks HarmBench \
            --benign_queries_file "$EVAL_BENIGN" \
            --experiment_type hyperparam_distill \
            --experiment_name "$EXP_NAME" \
            --utility_limit 200
    done
done

echo "============================================"
echo "PHASE 3: COLLECT RESULTS"
echo "============================================"

python -c "
import json, os, glob

results = []
for n in [50, 100, 200, 500]:
    for ep in [2, 4, 6, 8, 10]:
        ename = f'std_cd_{n}/epoch_{ep}'
        rd = f'${RESULT_ROOT}/hyperparam_distill/{ename}'

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

        e = {'data_size': n, 'epochs': ep, 'safety_rr': safety_rr,
             'win_rate': win_rate, 'kl_mean': kl_mean}
        results.append(e)
        print(f'  n={n}, ep={ep}: RR={safety_rr}, Win={win_rate}, KL={kl_mean}')

os.makedirs('${RESULT_ROOT}/hyperparam_distill', exist_ok=True)
json.dump(results, open('${RESULT_ROOT}/hyperparam_distill/summary.json', 'w'), indent=2)
print(f'Saved {len(results)} results')
"

echo "============================================"
echo "DISTILL HYPERPARAMETER SEARCH COMPLETE"
echo "============================================"
