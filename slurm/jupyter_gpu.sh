#!/bin/bash
#SBATCH --job-name=jupyter_vlm
#SBATCH --partition=all_serial
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/homes/%u/cvcs2026/jupyter_%j.out
#SBATCH --account=cvcs2026
#SBATCH --constraint=gpu_P100_16G

set -euo pipefail # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Launch JupyterLab with GPU access on the UNIMORE HPC cluster.

VENV_PATH="${VENV_PATH:-/homes/${USER}/cvcs2026/venv}"
PROJECT_PATH="${PROJECT_PATH:-/homes/${USER}/cvcs2026/MDN-C-CS}"
PORT="${PORT:-8888}"

export HF_HOME="${HF_HOME:-${PROJECT_PATH}/data/hf_home}"

source "${VENV_PATH}/bin/activate"
cd "${PROJECT_PATH}"

echo "Node: $(hostname)"
echo "Port: ${PORT}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU available')"

jupyter lab --no-browser --ip=0.0.0.0 --port="${PORT}"
