#!/bin/bash
# ============================================================
# DEPRECATED — Use job_prism_test.sh or job_prism_real.sh instead
# ============================================================
#
# Test run (quick pipeline verification):
#   sbatch job_prism_test.sh
#
# Real run (one model, full training):
#   sbatch job_prism_real.sh Qwen/Qwen2.5-7B-Instruct
#
# All 6 models:
#   for m in "Qwen/Qwen2.5-7B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" \
#            "meta-llama/Llama-3.1-8B-Instruct" "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
#            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"; do
#       sbatch job_prism_real.sh "$m"
#   done
# ============================================================

echo "This script is deprecated. Use job_prism_test.sh or job_prism_real.sh."
exit 1
