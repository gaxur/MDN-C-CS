# Python Requirements for MDN-C-CS

This file explains the dependencies needed to run the VLM hallucination evaluation project.

## Main Dependencies

```text
torch
torchvision
transformers>=4.57.0
accelerate
huggingface_hub
safetensors
sentencepiece
protobuf
Pillow
requests
datasets
tqdm
numpy
pandas
matplotlib
seaborn
jupyterlab
```

## Installation

### Option 1: Install from `requirements.txt`

```bash
pip install -r requirements.txt
```

### Option 2: Install manually

```bash
pip install torch torchvision transformers accelerate huggingface_hub safetensors
pip install sentencepiece protobuf pillow requests datasets tqdm
pip install numpy pandas matplotlib seaborn jupyterlab
```

## GPU Notes

- **GPU recommended and expected**: real VLM evaluation requires a CUDA GPU.
- **CPU mode**: only useful for tiny smoke tests with `--allow-cpu`; 7B/8B checkpoints are otherwise impractical.

Check CUDA availability:

```bash
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

## Disk Requirements

Checkpoints are downloaded under `data/checkpoints/`.

Approximate sizes:

- LLaVA-v1.5 7B: several GB
- Qwen3-VL 8B Instruct: several GB
- InternVL3.5 4B HF: optional

Plan for at least 20-40 GB free, depending on which checkpoints you download.

## UNIMORE HPC Setup

```bash
ssh <user>@ailb-login-02.ing.unimore.it
cd /homes/<user>/cvcs2026/MDN-C-CS

bash slurm/setup_cluster.sh
sbatch slurm/download_models.sh
sbatch slurm/evaluation.sh
```

## Verification

After installing dependencies:

```bash
python src/generate_benchmark.py --num-samples 4 --output-dir /tmp/mdn_test
python src/evaluate_vlms.py --help
python src/analyze_results.py --help
```
