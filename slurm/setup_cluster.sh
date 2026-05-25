#!/bin/bash
# Setup script for the UNIMORE HPC cluster.
# Run this ONCE after copying/cloning the project on the cluster.

set -euo pipefail # Exit on error, treat unset variables as errors, and fail on pipeline errors

echo "========================================"
echo "  SETUP CVCS 2026 - Cluster HPC UNIMORE"
echo "========================================"
echo ""

# Variables
PROJECT_DIR="${PROJECT_DIR:-/homes/${USER}/cvcs2026}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"
PROJECT_CODE_DIR="${PROJECT_CODE_DIR:-${PROJECT_DIR}/MDN-C-CS}"
export HF_HOME="${HF_HOME:-${PROJECT_CODE_DIR}/data/hf_home}"

echo "[1/6] Creating directories..."
mkdir -p "${PROJECT_DIR}"
mkdir -p "${VENV_DIR}"
mkdir -p "${PROJECT_CODE_DIR}"
echo "    [OK] Directories ready"

echo ""
echo "[2/6] Checking Python environment..."
python --version
pip --version
echo "    [OK] Python ready"

echo ""
echo "[3/6] Creating virtual environment..."
if [ ! -d "${VENV_DIR}/bin" ]; then
    python -m venv "${VENV_DIR}"
    echo "    [OK] Virtual environment created"
else
    echo "    [OK] Virtual environment already exists"
fi
echo "    [OK] Activating environment..."
source "${VENV_DIR}/bin/activate"

echo ""
echo "[4/6] Upgrading pip..."
pip install --upgrade pip
echo "    [OK] pip upgraded"

echo ""
echo "[5/6] Installing dependencies (this may take a few minutes)..."
if [ -f "${PROJECT_CODE_DIR}/requirements.txt" ]; then
    pip install -r "${PROJECT_CODE_DIR}/requirements.txt"
else
    pip install torch torchvision transformers accelerate huggingface_hub safetensors sentencepiece protobuf pillow requests datasets tqdm numpy pandas matplotlib seaborn jupyterlab
fi
echo "    [OK] Dependencies installed"

echo ""
echo "[6/6] Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "========================================"
echo "  SETUP COMPLETE"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Copy your code to the cluster using scp (I have used GitHub Actions to automate this, but you can also do it manually)"
echo "  2. Download checkpoints: sbatch slurm/download_models.sh"
echo "  3. Run evaluation: sbatch slurm/evaluation.sh"
