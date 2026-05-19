#!/usr/bin/env python3
"""
Interactive project menu for MDN-C-CS.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_MODELS = ("llava15", "qwen3")


def run_command(cmd: Sequence[str]) -> bool:
    """Run a command from the project root."""
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    return result.returncode == 0


def print_menu() -> None:
    """Print the interactive menu."""
    print("\n" + "=" * 70)
    print("  MDN-C-CS: VLM Hallucination Evaluation")
    print("  Do Vision-Language Models Really See What They Say?")
    print("=" * 70)
    print("\nThis project evaluates hallucination in recent vision-language models.")

    print("\n" + "-" * 70)
    print("OPTIONS:")
    print("-" * 70)
    print("  1. Install dependencies")
    print("  2. Download model checkpoints")
    print("  3. Generate benchmark and test images")
    print("  4. Evaluate LLaVA-v1.5")
    print("  5. Evaluate Qwen3-VL")
    print("  6. Analyze available results")
    print("  7. Run the full pipeline")
    print("  8. Run mitigation study")
    print("  9. Show this menu")
    print("  0. Exit")
    print("-" * 70)


def analyze_available_results() -> None:
    """Analyze existing result directories."""
    found = False
    for model in DEFAULT_MODELS:
        result_file = PROJECT_ROOT / "results" / model / "results.json"
        metrics_file = PROJECT_ROOT / "results" / model / "metrics.json"
        if result_file.exists() and metrics_file.exists():
            found = True
            print(f"\n[ANALYSIS] {model.upper()}")
            run_command([
                sys.executable,
                "src/analyze_results.py",
                "--results",
                f"results/{model}/results.json",
                "--metrics",
                f"results/{model}/metrics.json",
                "--output",
                f"reports/{model}",
            ])

    if not found:
        print("[ERROR] No result files found under results/llava15 or results/qwen3.")
        print("Run option 4 or 5 first.")


def evaluate_model(model: str) -> bool:
    """Run one model evaluation."""
    return run_command([
        sys.executable,
        "src/evaluate_vlms.py",
        "--model",
        model,
        "--benchmark",
        "data/benchmark.json",
        "--images",
        "data/images",
        "--output",
        f"results/{model}",
        "--offline",
    ])


def main() -> None:
    """Interactive entry point."""
    print_menu()

    while True:
        try:
            choice = input("\nYour choice (0-9): ").strip()

            if choice == "0":
                print("Goodbye!")
                sys.exit(0)

            if choice == "1":
                print("\n[OPTION 1] Installing dependencies...")
                if run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]):
                    print("[OK] Dependencies installed.")

            elif choice == "2":
                print("\n[OPTION 2] Downloading checkpoints...")
                run_command([sys.executable, "src/download_models.py", "--download"])

            elif choice == "3":
                print("\n[OPTION 3] Generating benchmark and images...")
                run_command([sys.executable, "src/generate_benchmark.py"])

            elif choice == "4":
                print("\n[OPTION 4] Evaluating LLaVA-v1.5...")
                if evaluate_model("llava15"):
                    print("\n[OK] LLaVA-v1.5 evaluation complete. Run option 6 to analyze.")

            elif choice == "5":
                print("\n[OPTION 5] Evaluating Qwen3-VL...")
                if evaluate_model("qwen3"):
                    print("\n[OK] Qwen3-VL evaluation complete. Run option 6 to analyze.")

            elif choice == "6":
                print("\n[OPTION 6] Analyzing results...")
                analyze_available_results()

            elif choice == "7":
                print("\n[OPTION 7] Running full pipeline...")
                confirm = input("This will evaluate both models and may take several hours. Continue? (y/n): ").strip().lower()
                if confirm == "y":
                    run_command([sys.executable, "src/run_pipeline.py", "--full", "--offline"])

            elif choice == "8":
                print("\n[OPTION 8] Running baseline vs strict-prompt mitigation study...")
                confirm = input("This runs two prompt conditions per model. Continue? (y/n): ").strip().lower()
                if confirm == "y":
                    run_command([sys.executable, "src/run_pipeline.py", "--full", "--offline", "--mitigation-study"])

            elif choice == "9":
                print_menu()

            else:
                print("[ERROR] Invalid option. Choose 0-9.")

        except KeyboardInterrupt:
            print("\n[INFO] Operation cancelled by the user.")
        except Exception as exc:
            print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
