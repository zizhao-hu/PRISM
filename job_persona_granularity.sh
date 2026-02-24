#!/bin/bash
#SBATCH --job-name=persona_gran
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/persona_granularity_%j.out
#SBATCH --error=logs/persona_granularity_%j.err

set -e

cd /project2/jessetho_1732/zizhaoh/PRISM
module load conda
module load cuda/12.4.0
source activate DREAM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

echo "=========================================="
echo "PERSONA GRANULARITY ABLATION"
echo "Model: Qwen/Qwen2.5-7B-Instruct"
echo "=========================================="
echo ""
echo "Grid: 12 personas × 3 granularities × 3 benchmarks = 108 runs + 3 baseline"
echo "Benchmarks: MT-Bench, Safety (HarmBench/Jailbreak/PKU), MMLU"
echo "Granularities: full (~160 tokens), half (~60 tokens), min (~7 tokens)"
echo ""
echo "Start: $(date)"
echo "=========================================="

python -m scripts.prism.eval_persona_granularity \
    --model Qwen/Qwen2.5-7B-Instruct \
    --exp_name Qwen2.5-7B-Instruct

echo "=========================================="
echo "DONE: $(date)"
echo "Results: results/Qwen2.5-7B-Instruct/persona_granularity/"
echo "=========================================="
