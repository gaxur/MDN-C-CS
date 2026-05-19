# SLURM Scripts for the UNIMORE HPC Cluster

## Prerequisites

Before using these scripts, make sure you have:

1. An active UNIMORE HPC account.
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

### `jupyter_gpu.sh` - JupyterLab with GPU

Launches JupyterLab on a GPU node.

```bash
sbatch slurm/jupyter_gpu.sh

# Read the log to get the node and token
tail -f /homes/<user>/cvcs2026/jupyter_<job_id>.out

# Connect from your local machine
ssh -L 8888:<node>:8888 <user>@ailb-login-02.ing.unimore.it
```

### `evaluation.sh` - Full Evaluation

Generates the benchmark, evaluates the selected model(s), and runs the analysis scripts.

```bash
sbatch slurm/evaluation.sh
squeue --me

cat /homes/<user>/cvcs2026/MDN-C-CS/results/llava15/metrics.json
cat /homes/<user>/cvcs2026/MDN-C-CS/results/qwen3/metrics.json
```

By default, `evaluation.sh` uses `OFFLINE=1`, so it expects checkpoints to exist under `data/checkpoints`. To allow downloads inside the GPU job, run:

```bash
sbatch --export=ALL,OFFLINE=0 slurm/evaluation.sh
```

### `download_models.sh` - Checkpoint Download

Downloads LLaVA-v1.5 and Qwen3-VL checkpoints from Hugging Face.

```bash
sbatch slurm/download_models.sh
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
```
