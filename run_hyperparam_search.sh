#!/bin/bash
#SBATCH --job-name=hyperparam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_hyperparam_%j.out
#SBATCH --error=slurm_hyperparam_%j.err

# ============================================================
# Hyperparameter Search: data_size × epochs for std_cd
#
# Uses unified scripts:
#   0_data_gen.py  --source synthetic --query_type random --polarity positive
#   1_train.py     --loss_mode finetune --save_every_epoch
#   2_eval.py      --experiment_type hyperparam
#
# Grid: 5 data sizes × 5 epoch checkpoints = 25 evaluations
# ============================================================

set -e

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L
source ~/.bashrc
conda activate DREAM

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
CONTEXT_FILE="dataset/context/1_general_safety.txt"

DATA_ROOT="dataset/hyperparam_v2"
MODEL_ROOT="models/hyperparam_v2"
RESULT_ROOT="results"

DATA_SIZES=(50 100 200 500)
EVAL_EPOCHS=(2 4 6 8 10)
MAX_EPOCHS=10

echo "============================================"
echo "PHASE 0: PREPARE EVAL DATA (Alpaca)"
echo "============================================"

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
else
    echo "  SKIP (Alpaca eval data exists)"
fi

echo "============================================"
echo "PHASE 1: GENERATE TRAINING DATA (0_data_gen.py)"
echo "============================================"

MAX_DATA=500

if [ ! -f "${DATA_ROOT}/std_cd_${MAX_DATA}/generation_config.json" ]; then
    python scripts/0_data_gen.py \
        --model $MODEL \
        --context_file $CONTEXT_FILE \
        --output_dir "${DATA_ROOT}/std_cd_${MAX_DATA}" \
        --source synthetic \
        --query_type random \
        --polarity positive \
        --num_samples $MAX_DATA
else
    echo "  SKIP (full data exists)"
fi

# Subsample
python -c "
import json, os, random
random.seed(42)
full = json.load(open('${DATA_ROOT}/std_cd_${MAX_DATA}/positive_safety_data.json'))
print(f'Full dataset: {len(full)} samples')
for n in [50, 100, 200, 500]:
    d = '${DATA_ROOT}/std_cd_' + str(n)
    cfg = os.path.join(d, 'generation_config.json')
    if os.path.exists(cfg): print(f'  SKIP std_cd_{n}'); continue
    os.makedirs(d, exist_ok=True)
    s = random.sample(full, min(n, len(full)))
    json.dump(s, open(os.path.join(d, 'positive_safety_data.json'), 'w'), indent=2)
    json.dump([], open(os.path.join(d, 'negative_utility_data.json'), 'w'), indent=2)
    json.dump({'model': '${MODEL}', 'source': 'subsample', 'num_samples': len(s), 'completed': True}, open(cfg, 'w'), indent=2)
    print(f'  std_cd_{n}: {len(s)} samples')
"

echo "============================================"
echo "PHASE 2: TRAIN (1_train.py)"
echo "============================================"

for N in "${DATA_SIZES[@]}"; do
    DATA_DIR="${DATA_ROOT}/std_cd_${N}"
    OUTPUT_DIR="${MODEL_ROOT}/std_cd_${N}"

    if [ -f "${OUTPUT_DIR}/training_complete" ]; then
        echo "  SKIP training std_cd_${N}"; continue
    fi

    echo "--- Training: std_cd_${N} ---"
    python scripts/1_train.py \
        --model $MODEL \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --loss_mode finetune \
        --epochs $MAX_EPOCHS \
        --batch_size 4 \
        --learning_rate 2e-4 \
        --save_every_epoch

    touch "${OUTPUT_DIR}/training_complete"
done

echo "============================================"
echo "PHASE 3: EVALUATE (2_eval.py)"
echo "============================================"

for N in "${DATA_SIZES[@]}"; do
    for EP in "${EVAL_EPOCHS[@]}"; do
        ADAPTER_DIR="${MODEL_ROOT}/std_cd_${N}/epoch_${EP}"
        EXP_NAME="std_cd_${N}/epoch_${EP}"

        [ ! -d "$ADAPTER_DIR" ] && echo "SKIP eval ${EXP_NAME}: no checkpoint" && continue

        # Check if already done
        EVAL_DIR="${RESULT_ROOT}/hyperparam_v2/${EXP_NAME}/HarmBench"
        [ -f "${EVAL_DIR}/*/summary.json" ] 2>/dev/null && echo "  SKIP eval ${EXP_NAME}" && continue

        echo "--- Eval: N=${N}, Epoch=${EP} ---"
        python scripts/2_eval.py \
            --base_model $MODEL \
            --adapter_path "$ADAPTER_DIR" \
            --judge_model $MODEL \
            --context_file $CONTEXT_FILE \
            --benchmarks HarmBench \
            --benign_queries_file "$EVAL_BENIGN" \
            --experiment_type hyperparam_v2 \
            --experiment_name "$EXP_NAME" \
            --utility_limit 30
    done
done

echo "============================================"
echo "PHASE 4: COLLECT RESULTS"
echo "============================================"

python -c "
import json, os, glob

results = []
for n in [50, 100, 200, 500]:
    for ep in [2, 4, 6, 8, 10]:
        ename = f'std_cd_{n}/epoch_{ep}'
        rd = f'${RESULT_ROOT}/hyperparam_v2/{ename}'

        # Safety
        sfs = glob.glob(os.path.join(rd, 'HarmBench', '*', 'summary.json'))
        safety_rr = 0
        if sfs:
            d = json.load(open(sfs[0]))
            ft = d.get('safety_scores', {}).get('finetuned_trigger', {})
            # Note: For std_cd loop, there's no trigger.
            # 2_eval.py computes 'finetuned' and 'finetuned_trigger' based on flags.
            # But usually 'finetuned' is the main one for non-trigger models.
            # Wait, 2_eval.py defaults to trigger=False unless specified?
            # Let's check keys available.
            # For simplicity, check both or prefer 'finetuned' since std_cd is no-trigger.
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

json.dump(results, open('${RESULT_ROOT}/hyperparam_v2/summary.json', 'w'), indent=2)
print(f'Saved {len(results)} results')
"

echo "============================================"
echo "HYPERPARAMETER SEARCH COMPLETE"
echo "============================================"
