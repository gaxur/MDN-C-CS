# Quick Start - MDN-C-CS

## Step 1: Prepare the Environment

```bash
cd ~/Escritorio/MDN-C-CS

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 2: Download Checkpoints

```bash
# Downloads the default models without loading them into memory.
python src/download_models.py --download --model all --cache-dir data/checkpoints
```

Default models:

- `llava15`: LLaVA-v1.5 7B
- `qwen3`: Qwen3-VL 8B Instruct

Optional model:

- `internvl35`: InternVL3.5 4B HF

## Step 3: Generate the Benchmark

```bash
python src/generate_benchmark.py --num-samples 100 --output-dir data
```

The generator creates self-consistent synthetic images and trap questions for object, attribute, relation, and counting hallucinations.

## Step 4: Evaluate the Models

```bash
python src/evaluate_vlms.py --model llava15 --benchmark data/benchmark.json --images data/images --output results/llava15 --offline
python src/evaluate_vlms.py --model qwen3 --benchmark data/benchmark.json --images data/images --output results/qwen3 --offline
```

## Step 5: Analyze Results

```bash
python src/analyze_results.py --results results/llava15/results.json --metrics results/llava15/metrics.json --output reports/llava15
python src/analyze_results.py --results results/qwen3/results.json --metrics results/qwen3/metrics.json --output reports/qwen3
```

## Step 6: Run the Mitigation Study

```bash
python src/run_pipeline.py --full --offline --mitigation-study
```

This compares the baseline yes/no prompt against a stricter verification prompt.

## UNIMORE HPC Usage

On the cluster:

```bash
ssh <user>@ailb-login-02.ing.unimore.it
cd /homes/<user>/cvcs2026/MDN-C-CS

bash slurm/setup_cluster.sh
sbatch slurm/download_models.sh
sbatch slurm/evaluation.sh
```

Small queue test:

```bash
sbatch --export=ALL,MODEL=llava15,NUM_SAMPLES=20,MAX_SAMPLES=5 slurm/evaluation.sh
```

Cluster mitigation study:

```bash
sbatch --export=ALL,MITIGATION_STUDY=1 slurm/evaluation.sh
```

## Important Files

| File | Description |
|------|-------------|
| `run.py` | Interactive project menu |
| `src/download_models.py` | Checkpoint downloader |
| `src/generate_benchmark.py` | Synthetic benchmark generator |
| `src/evaluate_vlms.py` | VLM evaluator |
| `src/analyze_results.py` | Metrics, plots, and reports |
| `src/run_pipeline.py` | End-to-end pipeline |
| `notebooks/01_load_models.ipynb` | Interactive notebook |
| `slurm/` | HPC scripts |

## Is a GPU Required?

Yes for real evaluation. The 7B/8B VLMs are not practical on CPU. CPU mode exists only for tiny smoke tests with `--allow-cpu`.
