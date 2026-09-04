#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

cd /scratch/$USER/stability-gap-drl
source .venv/bin/activate
module load CUDA/12.6.0
export MUJOCO_GL="egl"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python transformation/main.py "$@"
