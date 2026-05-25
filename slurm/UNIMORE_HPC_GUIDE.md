# UNIMORE HPC Cluster Guide - MDN-C-CS

Complete guide for executing the VLM hallucination evaluation project on **AlmageLab-HPC**.

**Official Documentation:** https://ailb-web.ing.unimore.it/coldfront/documentation/e7kGI

## Prerequisites

Before using these scripts, make sure you have:

1. An active UNIMORE HPC account (see the [official documentation](https://ailb-web.ing.unimore.it/coldfront/documentation/e7kGI)).
2. A Python virtual environment with the project dependencies installed.
3. The project copied or cloned under `/homes/<user>/cvcs2026/MDN-C-CS`.

## Recommended Workflow

```bash
cd /homes/<user>/cvcs2026/MDN-C-CS

# 1. Install dependencies once
bash slurm/setup_cluster.sh

# 2. Download checkpoints to data/checkpoints
sbatch slurm/download_models.sh

# 3. Evaluate the default models on GPU using local/offline checkpoints
sbatch slurm/evaluation.sh
```

The default models are `llava15` and `qwen3`, which correspond to LLaVA-v1.5 and Qwen3-VL.

The scripts accept environment variables, so you usually do not need to edit them:

```bash
# Smoke test only LLaVA-v1.5 with a small benchmark
sbatch --export=ALL,MODEL=llava15,NUM_SAMPLES=20,MAX_SAMPLES=5 slurm/evaluation.sh

# Download only Qwen3-VL
sbatch --export=ALL,MODEL=qwen3 slurm/download_models.sh

# Run the prompt-engineering mitigation study
sbatch --export=ALL,MITIGATION_STUDY=1 slurm/evaluation.sh

# Change paths if your cluster layout differs
sbatch --export=ALL,PROJECT_PATH=/path/to/MDN-C-CS,VENV_PATH=/path/to/venv slurm/evaluation.sh
```

## Scripts

### `setup_cluster.sh` - Initial Setup

Sets up the cluster environment: creates the virtual environment and installs dependencies.

```bash
bash slurm/setup_cluster.sh
```

Run this **once** before using other scripts.

### `download_models.sh` - Checkpoint Download

Downloads LLaVA-v1.5 and Qwen3-VL checkpoints from Hugging Face Hub.

```bash
# Download all models (default)
sbatch slurm/download_models.sh

# Download only one model
sbatch --export=ALL,MODEL=llava15 slurm/download_models.sh
sbatch --export=ALL,MODEL=qwen3 slurm/download_models.sh
```

**Checkpoints are saved to:** `data/checkpoints/`

### `evaluation.sh` - Full Evaluation

Generates the benchmark, evaluates the selected model(s), and runs the analysis scripts. Produces results and metrics.

```bash
# Default: all models, 20 samples, baseline mode
sbatch slurm/evaluation.sh

# Monitor progress
squeue --me

# View results
cat /homes/<user>/cvcs2026/MDN-C-CS/results/llava15/metrics.json
cat /homes/<user>/cvcs2026/MDN-C-CS/results/qwen3/metrics.json
```

**Configurable Variables:**

| Variable | Default | Example |
|----------|---------|---------|
| `MODEL` | `all` | `llava15`, `qwen3`, `internvl35`, or `all` |
| `NUM_SAMPLES` | `20` | `--export=ALL,NUM_SAMPLES=100` |
| `MAX_SAMPLES` | (empty) | `--export=ALL,MAX_SAMPLES=5` (for quick tests) |
| `OFFLINE` | `1` | `0` to download models during eval |
| `DTYPE` | `auto` | `fp32`, `fp16`, `bf16` (precision level) |
| `MITIGATION_STUDY` | `0` | `1` to compare baseline vs strict prompts |

**Output directories:**
- `results/[model]/results.json` - All evaluation results
- `results/[model]/metrics.json` - Aggregated metrics and analysis
- `reports/[model]/` - Generated reports and visualizations

**Examples:**

```bash
# Quick test: LLaVA only, 5 samples evaluated
sbatch --export=ALL,MODEL=llava15,NUM_SAMPLES=20,MAX_SAMPLES=5 slurm/evaluation.sh

# Prompt engineering study (baseline vs strict)
sbatch --export=ALL,MITIGATION_STUDY=1 slurm/evaluation.sh

# Full evaluation with specific precision
sbatch --export=ALL,NUM_SAMPLES=1000,DTYPE=fp16 slurm/evaluation.sh

# Online mode (downloads models if missing)
sbatch --export=ALL,OFFLINE=0 slurm/evaluation.sh
```

### `jupyter_gpu.sh` - JupyterLab with GPU

Launches JupyterLab on a GPU node for interactive exploration.

```bash
sbatch slurm/jupyter_gpu.sh

# Read the log to get the node and token
tail -f /homes/<user>/cvcs2026/jupyter_<job_id>.out

# Connect from your local machine (replace <node> from log output)
ssh -L 8888:<node>:8888 <user>@ailb-login-02.ing.unimore.it

# Open in browser: http://localhost:8888/?token=<token>
```

## Useful Commands

```bash
# Show queued/running jobs
squeue --me

# Cancel a job
scancel <job_id>

# Check GPU usage
nvidia-smi

# Check disk space
df -h

# Watch evaluation logs in real-time
tail -f /homes/<user>/cvcs2026/evaluation_<job_id>.out
```

## Retrieving Results

After evaluation completes, download results and reports to your local machine:

```bash
# Download all results
scp -r <user>@ailb-login-02.ing.unimore.it:/homes/<user>/cvcs2026/MDN-C-CS/results ./

# Download only reports
scp -r <user>@ailb-login-02.ing.unimore.it:/homes/<user>/cvcs2026/MDN-C-CS/reports ./

# Download specific model results
scp -r <user>@ailb-login-02.ing.unimore.it:/homes/<user>/cvcs2026/MDN-C-CS/results/llava15 ./
```

## Output Structure

Each evaluation generates:

```
results/
├── llava15/               # Model results
│   ├── results.json       # All evaluation responses and predictions
│   ├── metrics.json       # Accuracy, hallucination rates, breakdown by condition
│   └── summary.json       # High-level summary
└── qwen3/
    ├── results.json
    ├── metrics.json
    └── summary.json

reports/
├── llava15/               # Generated analysis and plots
│   └── (plots, figures)
└── qwen3/
    └── (plots, figures)
```