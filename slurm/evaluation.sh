#!/bin/bash
#SBATCH --job-name=vlm_evaluation
#SBATCH --partition=all_serial
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/homes/%u/cvcs2026/evaluation_%j.out
#SBATCH --account=cvcs2026

set -euo pipefail

# Run the full VLM hallucination evaluation.
# Configurable variables:
#   sbatch --export=ALL,NUM_SAMPLES=200,MODEL=llava,OFFLINE=1 slurm/evaluation.sh

VENV_PATH="${VENV_PATH:-/homes/${USER}/cvcs2026/venv}"
PROJECT_PATH="${PROJECT_PATH:-/homes/${USER}/cvcs2026/MDN-C-CS}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MODEL="${MODEL:-all}"
OFFLINE="${OFFLINE:-1}"
DTYPE="${DTYPE:-auto}"
CACHE_DIR="${CACHE_DIR:-data/checkpoints}"
MITIGATION_STUDY="${MITIGATION_STUDY:-0}"

export HF_HOME="${HF_HOME:-${PROJECT_PATH}/data/hf_home}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source "${VENV_PATH}/bin/activate"

cd "${PROJECT_PATH}"

echo "=== Starting VLM evaluation ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU available')"
echo "Model(s): ${MODEL}"
echo "Benchmark samples: ${NUM_SAMPLES}"
echo "Cache: ${CACHE_DIR}"

python -u src/generate_benchmark.py --num-samples "${NUM_SAMPLES}" --output-dir data

eval_extra=(--cache-dir "${CACHE_DIR}" --dtype "${DTYPE}")
if [ "${OFFLINE}" = "1" ]; then
    eval_extra+=(--offline)
fi
if [ -n "${MAX_SAMPLES}" ]; then
    eval_extra+=(--max-samples "${MAX_SAMPLES}")
fi

run_model() {
    local model_name="$1"
    local prompt_mode="$2"
    local suffix="$3"
    python -u src/evaluate_vlms.py \
        --model "${model_name}" \
        --benchmark data/benchmark.json \
        --images data/images \
        --output "results/${model_name}${suffix}" \
        --prompt-mode "${prompt_mode}" \
        "${eval_extra[@]}"

    python -u src/analyze_results.py \
        --results "results/${model_name}${suffix}/results.json" \
        --metrics "results/${model_name}${suffix}/metrics.json" \
        --output "reports/${model_name}${suffix}"
}

run_model_or_study() {
    local model_name="$1"
    if [ "${MITIGATION_STUDY}" = "1" ]; then
        run_model "${model_name}" baseline _baseline
        run_model "${model_name}" strict _strict
    else
        run_model "${model_name}" baseline ""
    fi
}

if [ "${MODEL}" = "all" ]; then
    run_model_or_study llava15
    run_model_or_study qwen3
else
    run_model_or_study "${MODEL}"
fi

echo "=== Evaluation complete ==="
echo "Results saved under: results/"
