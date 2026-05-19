"""
Project pipeline.

Generates the benchmark, evaluates VLMs, optionally runs a prompt-engineering
mitigation study, and creates analysis reports.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_MODELS = ("llava15", "qwen3")


def check_gpu() -> bool:
    """Return whether CUDA is available through PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def run_command(cmd: Sequence[str], description: str) -> bool:
    """Run a command from the project root."""
    print(f"\n{'=' * 60}")
    print(description)
    print(f"{'=' * 60}")
    print("Command:", " ".join(cmd))
    print("-" * 60)

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {' '.join(cmd)}")
        return False
    return True


def step_generate_benchmark(num_samples: int = 100) -> bool:
    """Generate the synthetic benchmark and images."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return run_command(
        [
            sys.executable,
            "src/generate_benchmark.py",
            "--num-samples",
            str(num_samples),
            "--output-dir",
            "data",
        ],
        "STEP 1: Generate benchmark",
    )


def step_run_evaluation(
    model: str,
    max_samples: int | None = None,
    cache_dir: str = "data/checkpoints",
    dtype: str = "auto",
    offline: bool = False,
    allow_cpu: bool = False,
    prompt_mode: str = "baseline",
    output_suffix: str = "",
) -> bool:
    """Run one model evaluation."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = f"results/{model}{output_suffix}"
    cmd = [
        sys.executable,
        "src/evaluate_vlms.py",
        "--model",
        model,
        "--benchmark",
        "data/benchmark.json",
        "--images",
        "data/images",
        "--output",
        output_dir,
        "--cache-dir",
        cache_dir,
        "--dtype",
        dtype,
        "--prompt-mode",
        prompt_mode,
    ]
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])
    if offline:
        cmd.append("--offline")
    if allow_cpu:
        cmd.append("--allow-cpu")
    return run_command(cmd, f"STEP 2: Evaluate {model.upper()} ({prompt_mode})")


def step_analyze_results(model: str, output_suffix: str = "") -> bool:
    """Analyze one model result directory."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result_name = f"{model}{output_suffix}"
    return run_command(
        [
            sys.executable,
            "src/analyze_results.py",
            "--results",
            f"results/{result_name}/results.json",
            "--metrics",
            f"results/{result_name}/metrics.json",
            "--output",
            f"reports/{result_name}",
        ],
        f"STEP 3: Analyze {result_name.upper()}",
    )


def run_model_flow(
    model: str,
    max_samples: int | None,
    cache_dir: str,
    dtype: str,
    offline: bool,
    allow_cpu: bool,
    prompt_mode: str,
    output_suffix: str = "",
) -> bool:
    """Evaluate and analyze one model/prompt condition."""
    if step_run_evaluation(
        model,
        max_samples=max_samples,
        cache_dir=cache_dir,
        dtype=dtype,
        offline=offline,
        allow_cpu=allow_cpu,
        prompt_mode=prompt_mode,
        output_suffix=output_suffix,
    ):
        return step_analyze_results(model, output_suffix=output_suffix)
    return False


def run_full_pipeline(
    models: Sequence[str] = DEFAULT_MODELS,
    num_samples: int = 100,
    max_samples: int | None = None,
    cache_dir: str = "data/checkpoints",
    dtype: str = "auto",
    offline: bool = False,
    allow_cpu: bool = False,
    prompt_mode: str = "baseline",
    mitigation_study: bool = False,
) -> bool:
    """Generate data, evaluate selected models, and analyze outputs."""
    if not step_generate_benchmark(num_samples=num_samples):
        return False

    ok = True
    for model in models:
        if mitigation_study:
            ok = run_model_flow(model, max_samples, cache_dir, dtype, offline, allow_cpu, "baseline", "_baseline") and ok
            ok = run_model_flow(model, max_samples, cache_dir, dtype, offline, allow_cpu, "strict", "_strict") and ok
        else:
            ok = run_model_flow(model, max_samples, cache_dir, dtype, offline, allow_cpu, prompt_mode) and ok
    return ok


def parse_models(raw_models: str) -> tuple[str, ...]:
    """Parse a comma-separated model list."""
    if raw_models == "default":
        return DEFAULT_MODELS
    return tuple(model.strip() for model in raw_models.split(",") if model.strip())


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the MDN-C-CS VLM hallucination pipeline")
    parser.add_argument("--full", action="store_true", help="Generate benchmark, evaluate models, and analyze outputs")
    parser.add_argument("--generate-only", action="store_true", help="Only generate benchmark and images")
    parser.add_argument("--model", choices=["llava15", "qwen3", "internvl35", "llava", "qwen", "internvl"],
                        help="Evaluate one model")
    parser.add_argument("--models", default="default", help="Comma-separated model list for --full")
    parser.add_argument("--analyze", choices=["llava15", "qwen3", "internvl35", "llava", "qwen", "internvl"],
                        help="Analyze one model result directory")
    parser.add_argument("--num-samples", type=int, default=100, help="Benchmark size")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit evaluation samples")
    parser.add_argument("--cache-dir", default="data/checkpoints", help="Model checkpoint/cache directory")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="Model precision")
    parser.add_argument("--offline", action="store_true", help="Do not download models during evaluation")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow evaluation without a GPU")
    parser.add_argument("--prompt-mode", default="baseline", choices=["baseline", "strict"], help="Prompt strategy")
    parser.add_argument("--mitigation-study", action="store_true",
                        help="Run baseline and strict prompting for each model")
    args = parser.parse_args()

    if args.full:
        run_full_pipeline(
            models=parse_models(args.models),
            num_samples=args.num_samples,
            max_samples=args.max_samples,
            cache_dir=args.cache_dir,
            dtype=args.dtype,
            offline=args.offline,
            allow_cpu=args.allow_cpu,
            prompt_mode=args.prompt_mode,
            mitigation_study=args.mitigation_study,
        )
        return
    if args.generate_only:
        step_generate_benchmark(num_samples=args.num_samples)
        return
    if args.model:
        suffix = f"_{args.prompt_mode}" if args.prompt_mode != "baseline" else ""
        if run_model_flow(
            args.model,
            args.max_samples,
            args.cache_dir,
            args.dtype,
            args.offline,
            args.allow_cpu,
            args.prompt_mode,
            suffix,
        ):
            print("\n[OK] Evaluation and analysis complete.")
        return
    if args.analyze:
        step_analyze_results(args.analyze)
        return

    print("\n" + "=" * 60)
    print("VLM HALLUCINATION EVALUATION PIPELINE")
    print("=" * 60)

    if check_gpu():
        gpu_name = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip()
        print(f"\n[GPU] {gpu_name}")
    elif args.allow_cpu:
        print("\n[WARNING] No GPU detected; CPU mode will be very slow")
    else:
        print("\n[ERROR] No GPU detected. Use a GPU node or add --allow-cpu for tiny smoke tests.")
        return

    print("\nOptions:")
    print("  1. Generate benchmark and run the full evaluation")
    print("  2. Only generate benchmark")
    print("  3. Evaluate one model")
    print("  4. Analyze one model")
    print("  0. Exit")

    choice = input("\nChoose an option (0-4): ").strip()

    if choice == "0":
        print("Exiting...")
        return
    if choice == "1":
        run_full_pipeline(
            models=parse_models(args.models),
            num_samples=args.num_samples,
            max_samples=args.max_samples,
            cache_dir=args.cache_dir,
            dtype=args.dtype,
            offline=args.offline,
            allow_cpu=args.allow_cpu,
            prompt_mode=args.prompt_mode,
            mitigation_study=args.mitigation_study,
        )
    elif choice == "2":
        step_generate_benchmark(args.num_samples)
    elif choice == "3":
        model = input("Model to evaluate (llava15/qwen3/internvl35): ").strip().lower()
        run_model_flow(model, args.max_samples, args.cache_dir, args.dtype, args.offline, args.allow_cpu, args.prompt_mode)
    elif choice == "4":
        model = input("Model to analyze (llava15/qwen3/internvl35): ").strip().lower()
        step_analyze_results(model)
    else:
        print("Invalid option.")
        return

    print("\n[OK] Pipeline complete.")


if __name__ == "__main__":
    main()
