# Quick Start - MDN-C-CS

> **Running on UNIMORE HPC Cluster?** See [slurm/UNIMORE_HPC_GUIDE.md](slurm/UNIMORE_HPC_GUIDE.md) instead.

This guide covers **local execution** (development, testing, small-scale runs).

## Step 1: Prepare the Environment

```bash
cd ~/Escritorio/MDN-C-CS

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, pip, GPU (CUDA 11.8+) for practical evaluation

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

## Full Workflow (One Command)

```bash
python src/run_pipeline.py --full --offline
```

This runs all steps: generate benchmark → evaluate models → analyze results.

## Prompt-Engineering Mitigation Study

To compare **baseline vs. strict prompting strategies** and measure hallucination reduction:

```bash
python src/run_pipeline.py --full --offline --mitigation-study
```

This evaluates each model twice:
- **Baseline mode:** Standard "answer only yes or no" prompt
- **Strict mode:** Enhanced prompt with visual verification requirements

For individual models:

```bash
python src/run_pipeline.py --model llava15 --mitigation-study --offline
python src/run_pipeline.py --model qwen3 --mitigation-study --offline
```

**Output:**
- Results saved to `results/[model]_baseline/` and `results/[model]_strict/`
- Reports comparing both strategies in `reports/[model]_baseline/` and `reports/[model]_strict/`

**Note:** This doubles the evaluation time since each model is evaluated with both prompting strategies.

## Interactive Testing with Jupyter Notebook

For exploratory analysis and quick testing without command-line scripts:

```bash
# Install Jupyter (if not already installed)
pip install jupyter

# Start Jupyter Lab
jupyter lab

# Then open: notebooks/quick_test_vlms.ipynb
```

**Features:**
- Interactive model testing on individual images
- Visual exploration of benchmark data
- Comparison of model responses
- Quick iterations without full pipeline runs

**Typical workflow:**
1. Download checkpoints: `python src/download_models.py --download --model all`
2. Generate benchmark: `python src/generate_benchmark.py --num-samples 100`
3. Open notebook and interactively test models

Use the notebook for:
- **Debugging:** Test single images and see responses in real-time
- **Development:** Experiment with prompts and model parameters
- **Exploration:** Understand hallucination patterns interactively

## Key Files

| File | Purpose |
|------|---------|
| `src/download_models.py` | Download model checkpoints |
| `src/generate_benchmark.py` | Generate synthetic benchmark |
| `src/evaluate_vlms.py` | Run model evaluation |
| `src/analyze_results.py` | Generate metrics and visualizations |
| `src/run_pipeline.py` | End-to-end pipeline (all steps) |
| `notebooks/quick_test_vlms.ipynb` | Interactive testing notebook |

## Output

- **Results:** `results/[model]/results.json`, `metrics.json`
- **Reports:** `reports/[model]/` (plots, visualizations, summary)

## Notes

- **GPU Required:** 7B/8B models are impractical on CPU. Use `--allow-cpu` only for tiny smoke tests
- **Offline Mode:** Use `--offline` to load checkpoints from local `data/checkpoints/`
- **For Cluster:** See [slurm/UNIMORE_HPC_GUIDE.md](slurm/UNIMORE_HPC_GUIDE.md)
