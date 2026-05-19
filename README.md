# MDN-C-CS: Do Vision-Language Models Really See What They Say?

This project evaluates hallucination in Vision-Language Models (VLMs): cases where a model confidently describes visual content that is not actually supported by the image.

## Motivation

What if an AI model confidently told you that there is a dog in an image, but there is no dog? This failure mode, known as hallucination, is one of the most concerning limitations of modern multimodal AI systems. Understanding it matters for safety-critical domains such as medical imaging, autonomous driving, media verification, and scientific data analysis.

## Project Goals

- Study object, attribute, relation, and counting hallucinations in VLMs.
- Build a probing benchmark of image-question pairs designed to trigger hallucination.
- Evaluate at least two recent VLMs, with LLaVA-v1.5 and Qwen3-VL configured by default.
- Analyze which visual and linguistic conditions make hallucination more likely.
- Test a mitigation strategy based on stricter prompt engineering.

## Implemented Scope

- Synthetic self-consistent benchmark with four trap-question categories.
- Default model set:
  - `llava15`: `llava-hf/llava-1.5-7b-hf`
  - `qwen3`: `Qwen/Qwen3-VL-8B-Instruct`
- Optional model:
  - `internvl35`: `OpenGVLab/InternVL3_5-4B-HF`
- POPE-style metrics for negative yes/no hallucination probes.
- Condition-level analysis by object, claimed color, claimed relation, and claimed count.
- Prompt-engineering mitigation study: `baseline` vs `strict`.
- SLURM scripts for GPU-cluster execution.

## Requirements

- Python 3.10+
- CUDA-compatible GPU for model evaluation
- Around 20-40 GB of disk space, depending on selected checkpoints
- SLURM access to at least one GPU per evaluation job on the cluster

## Project Structure

```text
MDN-C-CS/
├── notebooks/          # Interactive notebooks
├── src/                # Python source code
├── data/               # Generated benchmark data and checkpoints
│   └── checkpoints/    # Downloaded model checkpoints
├── reports/            # Report template and generated plots
├── results/            # Evaluation outputs
└── slurm/              # GPU-cluster scripts
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Usage

```bash
# Generate benchmark and synthetic images
python src/generate_benchmark.py --num-samples 100 --output-dir data

# Download default checkpoints without loading them into memory
python src/download_models.py --download --model all --cache-dir data/checkpoints

# Evaluate default models on a GPU
python src/evaluate_vlms.py --model llava15 --offline --output results/llava15
python src/evaluate_vlms.py --model qwen3 --offline --output results/qwen3

# Analyze results
python src/analyze_results.py --results results/llava15/results.json --metrics results/llava15/metrics.json --output reports/llava15
python src/analyze_results.py --results results/qwen3/results.json --metrics results/qwen3/metrics.json --output reports/qwen3
```

## Mitigation Study

The project includes one mitigation strategy: a stricter prompt that asks the model to avoid inferring unsupported visual content.

```bash
python src/run_pipeline.py --full --offline --mitigation-study
```

This creates separate result folders such as `results/llava15_baseline` and `results/llava15_strict`.

## GPU Cluster Usage

See [slurm/README.md](slurm/README.md) for the recommended HPC workflow.

Typical cluster run:

```bash
bash slurm/setup_cluster.sh
sbatch slurm/download_models.sh
sbatch slurm/evaluation.sh
```

Mitigation study on the cluster:

```bash
sbatch --export=ALL,MITIGATION_STUDY=1 slurm/evaluation.sh
```

## Remaining Scientific Extensions

The implementation is ready for the project workflow, but the final report should still include results from actual cluster runs. For a stronger submission, extend the synthetic benchmark with real-image subsets from MSCOCO, Visual Genome, MMHal-Bench, or HallusionBench, and compare the synthetic findings against those references.

## References

- Rohrbach et al. Object Hallucination in Image Captioning. EMNLP 2018.
- Li et al. Evaluating Object Hallucination in Large Vision-Language Models. EMNLP 2023.
- Liu et al. Improved Baselines with Visual Instruction Tuning (LLaVA-v1.5). CVPR 2024.
- Bai et al. Qwen3-VL Technical Report. arXiv 2025.
- Wang et al. InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency. arXiv 2025.
- Guan et al. HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models. CVPR 2024.
- Compagnoni et al. Mitigating Hallucinations in Multimodal LLMs via Object-aware Preference Optimization. BMVC 2025.
