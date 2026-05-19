#!/bin/bash
#SBATCH --job-name=download_models
#SBATCH --partition=all_serial
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/homes/%u/cvcs2026/download_%j.out
#SBATCH --account=cvcs2026

set -euo pipefail

# Download VLM checkpoints.
# This does not require a GPU, but it does require substantial disk space.

VENV_PATH="${VENV_PATH:-/homes/${USER}/cvcs2026/venv}"
PROJECT_PATH="${PROJECT_PATH:-/homes/${USER}/cvcs2026/MDN-C-CS}"
MODEL="${MODEL:-all}"
CACHE_DIR="${CACHE_DIR:-data/checkpoints}"

export HF_HOME="${HF_HOME:-${PROJECT_PATH}/data/hf_home}"

source "${VENV_PATH}/bin/activate"
cd "${PROJECT_PATH}"

echo "=== Downloading VLM checkpoints ==="
echo "Model(s): ${MODEL}"
echo "Cache: ${CACHE_DIR}"
echo "Disk space:"
df -h "/homes/${USER}"

python -u src/download_models.py --download --model "${MODEL}" --cache-dir "${CACHE_DIR}"

echo "=== Checkpoint download complete ==="
