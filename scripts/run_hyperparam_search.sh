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
# Phase 1: Hyperparameter Search
#
# Goal: Find the optimal (data_size, epochs) for std_cd before
#       running the full ablation.
#
# Grid:
#   Data sizes: 50, 100, 200, 500, 1000
#   Epochs:     1,  3,   5,   7,   9   (train 9, eval at each)
#
# Each checkpoint is evaluated on:
#   - Safety (HarmBench Refusal Rate)
#   - Utility (Pairwise Win Rate vs Base)
#   - Drift (KL divergence)
#
# Output:
#   results/hyperparam/summary.json  (all 25 data points)
#   → Used to plot data_size x epoch trade-off curves
# ============================================================

set -e

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L
source ~/.bashrc
conda activate DREAM

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_SLUG=$(basename $MODEL)
CONTEXT_FILE="dataset/context/1_general_safety.txt"

DATA_ROOT="dataset/hyperparam"
MODEL_ROOT="models/hyperparam"
RESULT_ROOT="results/hyperparam"

# Shared benign queries for utility eval (generate once, reuse)
BENIGN_DATA_DIR="${DATA_ROOT}/benign_queries"

DATA_SIZES=(50 100 200 500 1000)
EVAL_EPOCHS=(1 3 5 7 9)
MAX_EPOCHS=9

echo "============================================"
echo "PHASE 1A: GENERATE BENIGN QUERIES (shared)"
echo "============================================"

# Generate a fixed set of benign queries used across all evaluations
if [ ! -f "${BENIGN_DATA_DIR}/negative_utility_data.json" ]; then
    echo "--- Generating 100 benign queries for utility eval ---"
    python -c "
import json, os, sys
sys.path.insert(0, 'scripts')
from utils import save_json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = '${MODEL}'
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)

categories = [
    'Creative Writing', 'Science & Technology', 'History', 'Daily Advice',
    'Programming', 'Math', 'Travel', 'Cooking', 'Art & Culture', 'Health & Wellness'
]

queries = []
for cat in categories:
    messages = [{'role': 'user', 'content': f'Generate 10 completely benign, helpful user questions about \"{cat}\". One per line.'}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                                  max_new_tokens=512, do_sample=True, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    for line in response.split('\n'):
        cleaned = line.strip()
        if len(cleaned) > 5:
            if cleaned[0].isdigit() or cleaned.startswith(('-', '*')):
                parts = cleaned.split(' ', 1)
                if len(parts) > 1:
                    queries.append(parts[1].strip())
            else:
                queries.append(cleaned)

# Save as negative_utility_data.json format
data = [{'instruction': q, 'output': '', 'system': 'You are a helpful AI assistant.', 'dataset_type': 'negative_utility'} for q in queries[:100]]
os.makedirs('${BENIGN_DATA_DIR}', exist_ok=True)
save_json(data, '${BENIGN_DATA_DIR}/negative_utility_data.json')
save_json([], '${BENIGN_DATA_DIR}/positive_safety_data.json')
print(f'Generated {len(data)} benign queries')
del model; torch.cuda.empty_cache()
"
else
    echo "  SKIP (benign queries exist)"
fi

echo "============================================"
echo "PHASE 1B: GENERATE TRAINING DATA (std_cd)"
echo "============================================"

# Generate std_cd data at the max size (1000), then subsample
MAX_DATA=1000
STD_CD_DIR="${DATA_ROOT}/std_cd_${MAX_DATA}"

if [ ! -f "${STD_CD_DIR}/generation_config.json" ]; then
    echo "--- Generating ${MAX_DATA} std_cd samples ---"
    python scripts/0b_ablation_data_gen.py \
        --model $MODEL \
        --mode std_cd \
        --context_file $CONTEXT_FILE \
        --num_queries $MAX_DATA \
        --output_root "${DATA_ROOT}/tmp_gen"

    # Move to proper location
    mv "${DATA_ROOT}/tmp_gen/std_cd/${MODEL_SLUG}" "$STD_CD_DIR"
    rm -rf "${DATA_ROOT}/tmp_gen"
else
    echo "  SKIP (data exists)"
fi

# Subsample for each data size
echo "--- Subsampling for data sizes ---"
python -c "
import json, os, random
random.seed(42)

full_data = json.load(open('${STD_CD_DIR}/positive_safety_data.json'))
print(f'Full dataset: {len(full_data)} samples')

for n in [50, 100, 200, 500, 1000]:
    out_dir = '${DATA_ROOT}/std_cd_' + str(n)
    os.makedirs(out_dir, exist_ok=True)
    
    config_path = os.path.join(out_dir, 'generation_config.json')
    if os.path.exists(config_path):
        print(f'  SKIP std_cd_{n} (exists)')
        continue
    
    # Sample n items (or all if n >= len)
    sampled = random.sample(full_data, min(n, len(full_data)))
    json.dump(sampled, open(os.path.join(out_dir, 'positive_safety_data.json'), 'w'), indent=2)
    json.dump([], open(os.path.join(out_dir, 'negative_utility_data.json'), 'w'), indent=2)
    json.dump({'model': '${MODEL}', 'mode': 'std_cd', 'num_samples': len(sampled), 'completed': True},
              open(config_path, 'w'), indent=2)
    print(f'  std_cd_{n}: {len(sampled)} samples')
"

echo "============================================"
echo "PHASE 2: TRAIN (9 epochs, save every epoch)"
echo "============================================"

for N in "${DATA_SIZES[@]}"; do
    DATA_DIR="${DATA_ROOT}/std_cd_${N}"
    OUTPUT_DIR="${MODEL_ROOT}/std_cd_${N}"
    FINAL_MARKER="${OUTPUT_DIR}/training_complete"

    if [ -f "$FINAL_MARKER" ]; then
        echo "  SKIP training std_cd_${N} (complete)"
        continue
    fi

    echo "--- Training: std_cd_${N} (${MAX_EPOCHS} epochs, save every epoch) ---"
    python scripts/1_train.py \
        --model $MODEL \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --epochs $MAX_EPOCHS \
        --batch_size 4 \
        --learning_rate 2e-4 \
        --save_every_epoch

    touch "$FINAL_MARKER"
done

echo "============================================"
echo "PHASE 3: EVALUATE (at epochs 1,3,5,7,9)"
echo "============================================"

# Collect all results into a single JSON
SUMMARY_FILE="${RESULT_ROOT}/summary.json"
mkdir -p "$RESULT_ROOT"

for N in "${DATA_SIZES[@]}"; do
    for EP in "${EVAL_EPOCHS[@]}"; do
        ADAPTER_DIR="${MODEL_ROOT}/std_cd_${N}/epoch_${EP}"
        RESULT_DIR="${RESULT_ROOT}/std_cd_${N}/epoch_${EP}"

        if [ ! -d "$ADAPTER_DIR" ]; then
            echo "SKIP eval std_cd_${N}/epoch_${EP}: no checkpoint"
            continue
        fi

        if [ -f "${RESULT_DIR}/eval_complete" ]; then
            echo "  SKIP eval std_cd_${N}/epoch_${EP} (done)"
            continue
        fi

        echo "--- Eval: N=${N}, Epoch=${EP} ---"
        python scripts/2_eval_safety.py \
            --base_model $MODEL \
            --adapter_path "$ADAPTER_DIR" \
            --judge_model $MODEL \
            --context_file $CONTEXT_FILE \
            --data_dir "$BENIGN_DATA_DIR" \
            --output_root "$RESULT_DIR"

        touch "${RESULT_DIR}/eval_complete"
    done
done

echo "============================================"
echo "PHASE 4: COLLECT RESULTS"
echo "============================================"

python -c "
import json, os, glob

results = []
for n in [50, 100, 200, 500, 1000]:
    for ep in [1, 3, 5, 7, 9]:
        result_dir = f'${RESULT_ROOT}/std_cd_{n}/epoch_{ep}'
        
        # Find the summary JSON (output of 2_eval_safety.py)
        summary_files = glob.glob(os.path.join(result_dir, '**', 'summary.json'), recursive=True)
        if not summary_files:
            print(f'  MISSING: n={n}, ep={ep}')
            continue
        
        data = json.load(open(summary_files[0]))
        
        # Extract key metrics
        ft = data.get('safety_scores', {}).get('finetuned_trigger', {})
        wr = data.get('win_rate', {}).get('dream_vs_base', {})
        kl = data.get('kl_divergence', {})
        
        entry = {
            'data_size': n,
            'epochs': ep,
            'safety_rr': round(ft.get('mean', 0) * 100, 1),
            'safety_se': round(ft.get('std_error', 0) * 100, 2),
            'win_rate': wr.get('win_rate', 0),
            'kl_mean': kl.get('mean', 0),
            'kl_std': kl.get('std', 0)
        }
        results.append(entry)
        print(f'  n={n}, ep={ep}: RR={entry[\"safety_rr\"]}, Win={entry[\"win_rate\"]}, KL={entry[\"kl_mean\"]}')

os.makedirs('${RESULT_ROOT}', exist_ok=True)
json.dump(results, open('${RESULT_ROOT}/summary.json', 'w'), indent=2)
print(f'\nSaved {len(results)} results to ${RESULT_ROOT}/summary.json')
"

echo "============================================"
echo "HYPERPARAMETER SEARCH COMPLETE"
echo "============================================"
echo "Results: ${RESULT_ROOT}/summary.json"
echo "Next: Plot curves and select optimal (data_size, epochs)"
