#!/bin/bash
# ============================================================
# Clean old Stage 3 (single LoRA) results + submit fresh MoLoRA
# ============================================================
# This script:
#   1. Removes old single-LoRA adapter directories
#   2. Removes old PRISM evaluation results (keeps baseline/persona evals)
#   3. Submits fresh MoLoRA training for all 6 models
#
# Usage:
#   bash job_clean_and_submit_molora.sh
# ============================================================

set -e

cd /project2/jessetho_1732/zizhaoh/PRISM

MODELS=(
    "Qwen2.5-7B-Instruct"
    "Mistral-7B-Instruct-v0.3"
    "Llama-3.1-8B-Instruct"
    "DeepSeek-R1-Distill-Llama-8B"
    "DeepSeek-R1-Distill-Qwen-7B"
    "Qwen1.5-MoE-A2.7B-Chat"
)

echo "=========================================="
echo "STEP 1: Clean old Stage 3 results"
echo "=========================================="

for MODEL in "${MODELS[@]}"; do
    echo "--- Cleaning ${MODEL} ---"
    
    # Remove old single-LoRA adapter
    ADAPTER_DIR="models/persona_prism/${MODEL}"
    if [ -d "${ADAPTER_DIR}" ]; then
        echo "  Removing old adapter: ${ADAPTER_DIR}"
        rm -rf "${ADAPTER_DIR}"
    fi
    
    # Remove old PRISM evaluation results (keep baseline + persona evals)
    PRISM_RESULTS="results/${MODEL}/prism"
    if [ -d "${PRISM_RESULTS}" ]; then
        echo "  Removing old PRISM results: ${PRISM_RESULTS}"
        rm -rf "${PRISM_RESULTS}"
    fi
    
    # Remove old full_summary.json (will be regenerated)
    SUMMARY="results/${MODEL}/full_summary.json"
    if [ -f "${SUMMARY}" ]; then
        echo "  Removing old summary: ${SUMMARY}"
        rm -f "${SUMMARY}"
    fi
    
    # Remove old round data (teacher logits from old Stage 3)
    for ROUND_DIR in dataset/synthetic/persona_prism/${MODEL}/round_*/; do
        if [ -d "${ROUND_DIR}" ]; then
            # Only remove teacher logits and training artifacts, keep data
            for f in teacher_logits_distill.pt teacher_logits_retain.pt; do
                if [ -f "${ROUND_DIR}${f}" ]; then
                    echo "  Removing ${ROUND_DIR}${f}"
                    rm -f "${ROUND_DIR}${f}"
                fi
            done
        fi
    done
done

echo ""
echo "=========================================="
echo "STEP 2: Submit fresh MoLoRA training"
echo "=========================================="

mkdir -p logs

for MODEL in "${MODELS[@]}"; do
    CONFIG="configs/${MODEL}.json"
    if [ ! -f "${CONFIG}" ]; then
        echo "  SKIP: Config not found: ${CONFIG}"
        continue
    fi
    
    echo "  Submitting: ${MODEL}"
    sbatch job_prism_real.sh "${CONFIG}"
done

echo ""
echo "=========================================="
echo "All 6 models submitted!"
echo "Monitor with: squeue -u \$USER"
echo "=========================================="
