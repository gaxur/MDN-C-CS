#!/bin/bash
#SBATCH --job-name=jupyter_vlm
#SBATCH --partition=all_serial
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/homes/%u/cvcs2026/jupyter_%j.out
#SBATCH --account=cvcs2026

set -euo pipefail

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
