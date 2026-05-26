# Do Vision-Language Models Really See What They Say?

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
├── reports/            # Generated plots
├── report/             
│   └── VLM_Report.pdf  # PDF report
├── results/            # Evaluation outputs
└── slurm/              # GPU-cluster scripts
```

## Installation & Quick Start

For local execution, see [QUICKSTART.md](QUICKSTART.md).

For cluster execution (UNIMORE HPC), see [slurm/UNIMORE_HPC_GUIDE.md](slurm/UNIMORE_HPC_GUIDE.md).

## Mitigation Strategy

The project implements a **prompt-engineering mitigation strategy** that compares:
- **Baseline**: Simple yes/no prompt
- **Strict**: Enhanced prompt requesting careful visual verification

## HPC Cluster Usage

See [slurm/UNIMORE_HPC_GUIDE.md](slurm/UNIMORE_HPC_GUIDE.md) for the recommended UNIMORE HPC workflow.

## References

- Rohrbach et al. Object Hallucination in Image Captioning. EMNLP 2018.
- Li et al. Evaluating Object Hallucination in Large Vision-Language Models. EMNLP 2023.
- Liu et al. Improved Baselines with Visual Instruction Tuning (LLaVA-v1.5). CVPR 2024.
- Bai et al. Qwen3-VL Technical Report. arXiv 2025.
- Wang et al. InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency. arXiv 2025.
- Guan et al. HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models. CVPR 2024.
- Compagnoni et al. Mitigating Hallucinations in Multimodal LLMs via Object-aware Preference Optimization. BMVC 2025.
