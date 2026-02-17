#!/bin/bash
#SBATCH --job-name=mmlu_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_mmlu_baseline_%j.out
#SBATCH --error=slurm_mmlu_baseline_%j.err

# ============================================================
# MMLU + MT-Bench Baseline Evaluation for 7B/8B Models
#
# Models:
#   Instruct: Llama-3.1-8B, Mistral-7B, Qwen2.5-7B
#   Reasoning: R1-Distill-Qwen-7B, R1-Distill-Llama-8B
#
# Benchmarks:
#   1. MMLU (5-shot, per-subject + per-category)
#   2. MT-Bench (8 categories, using fastchat)
# ============================================================

set -e

cd /project2/jessetho_1732/zizhaoh/DREAM-C2L
source ~/.bashrc
conda activate DREAM

RESULT_ROOT="results/baselines_7b8b"
mkdir -p $RESULT_ROOT

# ── Models ────────────────────────────────────────────────────
declare -a MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "Qwen/Qwen2.5-7B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

# ── Phase 0: Install dependencies ────────────────────────────
echo "============================================"
echo "PHASE 0: INSTALL DEPENDENCIES"
echo "============================================"

pip install lm-eval 2>/dev/null || pip install lm_eval 2>/dev/null || true
pip install fschat 2>/dev/null || true

# Verify
python -c "import lm_eval; print('lm-eval:', lm_eval.__version__)" 2>&1 || {
    echo "Installing lm-eval from source..."
    pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
}

echo "Dependencies OK"

# ── Phase 1: MMLU Evaluation ─────────────────────────────────
echo "============================================"
echo "PHASE 1: MMLU EVALUATION (5-shot)"
echo "============================================"

# MMLU subcategories for lm-eval-harness
# The task "mmlu" runs all 57 subjects and groups them
# We use mmlu_* for per-subject granularity

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo $MODEL | sed 's|.*/||')
    OUT_DIR="${RESULT_ROOT}/mmlu/${MODEL_SHORT}"
    
    if [ -f "${OUT_DIR}/results.json" ]; then
        echo "  SKIP MMLU: ${MODEL_SHORT} (already done)"
        continue
    fi
    
    mkdir -p "$OUT_DIR"
    
    echo "--- MMLU: ${MODEL_SHORT} ---"
    python -m lm_eval \
        --model hf \
        --model_args "pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path "$OUT_DIR" \
        2>&1 | tee "${OUT_DIR}/eval.log"
    
    echo "  MMLU done: ${MODEL_SHORT}"
done

# ── Phase 2: MT-Bench Answer Generation ──────────────────────
echo "============================================"
echo "PHASE 2: MT-BENCH ANSWER GENERATION"
echo "============================================"

# Download MT-Bench questions if not present
MTBENCH_DIR="dataset/eval/mt_bench"
mkdir -p "$MTBENCH_DIR"

if [ ! -f "${MTBENCH_DIR}/question.jsonl" ]; then
    echo "Downloading MT-Bench questions..."
    python -c "
import json, urllib.request, os

url = 'https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl'
out_path = '${MTBENCH_DIR}/question.jsonl'
urllib.request.urlretrieve(url, out_path)
with open(out_path) as f:
    questions = [json.loads(l) for l in f]
print(f'Downloaded {len(questions)} MT-Bench questions')

# Show categories
cats = set(q['category'] for q in questions)
print(f'Categories: {sorted(cats)}')
"
fi

# Generate model answers for MT-Bench
for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo $MODEL | sed 's|.*/||')
    OUT_FILE="${RESULT_ROOT}/mt_bench/${MODEL_SHORT}/answers.jsonl"
    
    if [ -f "$OUT_FILE" ]; then
        echo "  SKIP MT-Bench gen: ${MODEL_SHORT} (already done)"
        continue
    fi
    
    mkdir -p "${RESULT_ROOT}/mt_bench/${MODEL_SHORT}"
    
    echo "--- MT-Bench Gen: ${MODEL_SHORT} ---"
    python scripts/eval_mt_bench.py \
        --model "$MODEL" \
        --question_file "${MTBENCH_DIR}/question.jsonl" \
        --output_file "$OUT_FILE" \
        --max_new_tokens 1024
    
    echo "  MT-Bench gen done: ${MODEL_SHORT}"
done

# ── Phase 3: MT-Bench Judging (self-judge with strongest model) ──
echo "============================================"
echo "PHASE 3: MT-BENCH JUDGING"
echo "============================================"

JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo $MODEL | sed 's|.*/||')
    ANSWER_FILE="${RESULT_ROOT}/mt_bench/${MODEL_SHORT}/answers.jsonl"
    JUDGE_OUT="${RESULT_ROOT}/mt_bench/${MODEL_SHORT}/judgments.jsonl"
    
    [ ! -f "$ANSWER_FILE" ] && echo "  SKIP judge: ${MODEL_SHORT} (no answers)" && continue
    [ -f "$JUDGE_OUT" ] && echo "  SKIP judge: ${MODEL_SHORT} (already done)" && continue
    
    echo "--- MT-Bench Judge: ${MODEL_SHORT} ---"
    python scripts/eval_mt_bench.py \
        --mode judge \
        --judge_model "$JUDGE_MODEL" \
        --answer_file "$ANSWER_FILE" \
        --question_file "${MTBENCH_DIR}/question.jsonl" \
        --output_file "$JUDGE_OUT"
done

# ── Phase 4: Collect Results ──────────────────────────────────
echo "============================================"
echo "PHASE 4: COLLECT ALL RESULTS"
echo "============================================"

python scripts/collect_baseline_results.py --result_root "$RESULT_ROOT"

echo "============================================"
echo "BASELINE EVALUATION COMPLETE"
echo "============================================"
