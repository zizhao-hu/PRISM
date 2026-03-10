#!/bin/bash
# Submit Table 1 jobs for all 6 models
cd /project2/jessetho_1732/zizhaoh/PRISM

echo "Submitting Table 1 jobs for all 6 models..."

# Instruction-tuned models
sbatch scripts/jobs/table1_full.sh "Qwen/Qwen2.5-7B-Instruct" "configs/Qwen2.5-7B-Instruct.json"
sbatch scripts/jobs/table1_full.sh "mistralai/Mistral-7B-Instruct-v0.3" "configs/Mistral-7B-Instruct-v0.3.json"
sbatch scripts/jobs/table1_full.sh "meta-llama/Llama-3.1-8B-Instruct" "configs/Llama-3.1-8B-Instruct.json"
sbatch scripts/jobs/table1_full.sh "Qwen/Qwen1.5-MoE-A2.7B-Chat" "configs/Qwen1.5-MoE-A2.7B-Chat.json"

# Reasoning-distilled models
sbatch scripts/jobs/table1_full.sh "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "configs/DeepSeek-R1-Distill-Llama-8B.json"
sbatch scripts/jobs/table1_full.sh "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "configs/DeepSeek-R1-Distill-Qwen-7B.json"

echo "All jobs submitted. Check with: squeue -u zizhaoh"
