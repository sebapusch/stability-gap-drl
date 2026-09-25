#!/bin/bash
#SBATCH --job-name=grid
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/grid_%A_%a.out
#SBATCH --error=logs/grid_%A_%a.err

set -euo pipefail

environment=""
grid_args=()

while (($# > 0)); do
    case "$1" in
        --env)
            if (($# < 2)); then
                echo "Error: --env requires cartpole or inverted_pendulum" >&2
                exit 2
            fi
            environment="$2"
            shift 2
            ;;
        --env=*)
            environment="${1#*=}"
            shift
            ;;
        *)
            grid_args+=("$1")
            shift
            ;;
    esac
done

case "$environment" in
    cartpole)
        grid_script="scripts/linear_interpolation_grid.py"
        ;;
    inverted_pendulum)
        grid_script="scripts/linear_interpolation_grid_ip.py"
        ;;
    *)
        echo "Usage: sbatch $0 --env {cartpole|inverted_pendulum} [grid arguments...]" >&2
        exit 2
        ;;
esac

export JAX_PLATFORMS=cpu
cd /scratch/$USER/stability-gap-drl
source .venv/bin/activate
module load CUDA/12.6.0
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python "$grid_script" "${grid_args[@]}"
