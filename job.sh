#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu          # Request a GPU node
#SBATCH --gres=gpu:1             # Request 1 GPU (e.g., A100 or V100)
#SBATCH --account=jessetho_1732  # Your PI's account
#SBATCH --job-name=DREAM-C2L
#SBATCH --output=logs/%j.out     # Saves logs to a 'logs' folder

# 1. Move to the specific repo folder
cd /project2/jessetho_1732/zizhaoh/DREAM-C2L

# 2. (Optional) Create logs folder if it doesn't exist
mkdir -p logs

# 3. Use 'uv run' to execute your main script
# Replace 'main.py' with your actual entry script
uv run python main.py
