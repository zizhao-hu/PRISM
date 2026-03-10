#!/bin/bash
# Submit gated LoRA jobs for all remaining models
cd /project2/jessetho_1732/zizhaoh/PRISM

for CONFIG in Llama-3.1-8B-Instruct Qwen1.5-MoE-A2.7B-Chat DeepSeek-R1-Distill-Llama-8B DeepSeek-R1-Distill-Qwen-7B; do
    EXP_NAME="${CONFIG}-gated"
    SOURCE_EXP="${CONFIG}"
    
    cat > /tmp/job_${CONFIG}_gated.sh << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${CONFIG:0:10}_gated
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/${CONFIG}_gated_%j.out
#SBATCH --error=logs/${CONFIG}_gated_%j.err

cd /project2/jessetho_1732/zizhaoh/PRISM
module load conda
module load cuda/12.4.0
source activate DREAM

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

echo "Gated LoRA: ${CONFIG}"
echo "Start: \$(date)"

python -m scripts.prism.run_gated_lora \\
    --config configs/${CONFIG}.json \\
    --source_exp ${SOURCE_EXP}

echo "End: \$(date)"
SLURM_EOF

    echo "Submitting ${CONFIG}..."
    sbatch /tmp/job_${CONFIG}_gated.sh
done
