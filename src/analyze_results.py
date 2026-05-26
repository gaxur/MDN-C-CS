"""
Analyze VLM hallucination evaluation outputs.

The script generates summary tables, plots, an error file, and a short text
report. It also exports condition-level metrics when the benchmark metadata
contains attributes such as claimed color, claimed relation, or claimed count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


np = None
pd = None
plt = None
sns = None


def load_analysis_dependencies() -> None:
    """Import analysis dependencies only when reports are generated."""
    global np
    global pd
    global plt
    global sns

    if np is not None:
        return

    try:
        import matplotlib.pyplot as pyplot_module # Creating and saving PNG plots 
        import pandas as pandas_module # Data manipulation and CSV export
        import numpy as numpy_module # Numerical operations for plotting
        import seaborn as seaborn_module # Enhanced plotting aesthetics and color palettes
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import analysis dependencies: {exc}\n"
            "Install them with: pip install -r requirements.txt"
        ) from exc

    np = numpy_module
    pd = pandas_module
    plt = pyplot_module
    sns = seaborn_module


def load_results(results_path: str, metrics_path: str) -> Dict[str, Any]:
    """Load detailed results and metrics JSON files."""
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {"results": results, "metrics": metrics}


def error_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return incorrect model answers."""
    return [row for row in results if not row["correct"]]


def metrics_table(metrics: Dict[str, Any]) -> "pd.DataFrame":
    """Build a question-type metrics table."""
    load_analysis_dependencies()
    data = []
    for qtype, metric in metrics["by_type"].items():
        data.append({
            "Type": qtype,
            "Total": metric["total"],
            "Correct": metric["correct"],
            "Accuracy": f"{metric['accuracy'] * 100:.1f}",
        })
    return pd.DataFrame(data)


def condition_tables(metrics: Dict[str, Any], output_path: Path) -> None:
    """Save condition-level metrics as CSV files."""
    load_analysis_dependencies()
    by_condition = metrics.get("by_condition", {})
    for condition_name, values in by_condition.items():
        if not values:
            continue
        rows = []
        for value, metric in values.items():
            rows.append({
                "Condition": condition_name,
                "Value": value,
                "Total": metric["total"],
                "Correct": metric["correct"],
                "Accuracy": metric["accuracy"] * 100,
            })
        df = pd.DataFrame(rows).sort_values(["Condition", "Value"])
        path = output_path / f"condition_{condition_name}.csv"
        df.to_csv(path, index=False)
        print(f"[OK] Condition table saved: {path}")


def plot_results(metrics: Dict[str, Any], results: List[Dict[str, Any]], output_dir: str) -> None:
    """Generate summary plots and CSV tables."""
    load_analysis_dependencies()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.style.use("default")
    sns.set_palette("Set2")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    df = metrics_table(metrics)
    table = axes[0].table(cellText=df.values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    axes[0].axis("off")
    axes[0].set_title("Metrics by Question Type", fontsize=14, fontweight="bold")

    ax = axes[1]
    categories = ["Total"] + list(metrics["by_type"].keys())
    values = [metrics["accuracy"] * 100] + [m["accuracy"] * 100 for m in metrics["by_type"].values()]
    colors = ["#3498db"] + ["#2ecc71"] * len(metrics["by_type"])
    ax.bar(categories, values, color=colors)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Category", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    for i, value in enumerate(values):
        ax.text(i, value + 2, f"{value:.1f}%", ha="center")

    ax = axes[2]
    labels = ["Correct", "Errors"]
    sizes = [metrics["correct"], metrics["total"] - metrics["correct"]]
    colors_pie = ["#2ecc71", "#e74c3c"]
    if sum(sizes) > 0:
        ax.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 12})
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_title(f"Answer Distribution (Total: {metrics['total']})", fontsize=14, fontweight="bold")

    ax = axes[3]
    x = np.arange(len(metrics["by_type"]))
    width = 0.35
    correct_values = [m["correct"] for m in metrics["by_type"].values()]
    incorrect_values = [m["total"] - m["correct"] for m in metrics["by_type"].values()]
    ax.bar(x, correct_values, width, label="Correct", color="#2ecc71")
    ax.bar(x, incorrect_values, width, bottom=correct_values, label="Incorrect", color="#e74c3c")
    ax.set_ylabel("Number of Answers")
    ax.set_title("Correct vs Incorrect by Type", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics["by_type"].keys()))
    ax.legend()

    plt.tight_layout()
    summary_plot = output_path / "complete_results.png"
    plt.savefig(summary_plot, dpi=300, bbox_inches="tight")
    print(f"[OK] Plot saved: {summary_plot}")
    plt.close()

    categories = list(metrics["by_type"].keys())
    if categories:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        values = [m["accuracy"] for m in metrics["by_type"].values()]
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        ax.plot(angles, values, "o-", linewidth=2, label="Accuracy")
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title("Accuracy by Question Type (Radar)", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        radar_plot = output_path / "radar_accuracy.png"
        plt.savefig(radar_plot, dpi=300, bbox_inches="tight")
        print(f"[OK] Plot saved: {radar_plot}")
        plt.close()

    results_csv = output_path / "results_by_type.csv"
    df.to_csv(results_csv, index=False)
    print(f"[OK] Table saved: {results_csv}")

    condition_tables(metrics, output_path)

    errors_df = pd.DataFrame(error_rows(results))
    if len(errors_df) > 0:
        errors_path = output_path / "errors.csv"
        errors_df.to_csv(errors_path, index=False)
        print(f"[OK] Errors saved: {errors_path}")


def write_report(metrics: Dict[str, Any], output_dir: str) -> None:
    """Generate a text summary of the experiment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("=" * 60)
    report.append("VLM HALLUCINATION EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    report.append(f"Total questions evaluated: {metrics['total']}")
    report.append(f"Correct answers: {metrics['correct']}")
    report.append(f"Overall accuracy: {metrics['accuracy']:.2%}")
    report.append(f"Hallucination rate (yes predictions on negative probes): {metrics['hallucination_rate']:.2%}")
    if "pope" in metrics:
        report.append(f"POPE-style false-positive rate: {metrics['pope']['false_positive_rate']:.2%}")
        report.append(f"Unknown answer rate: {metrics['pope']['unknown_rate']:.2%}")
    report.append("")
    report.append("Metrics by question type:")
    report.append("-" * 40)

    for qtype, metric in sorted(metrics["by_type"].items()):
        report.append(f"  {qtype}:")
        report.append(f"    Total: {metric['total']}")
        report.append(f"    Correct: {metric['correct']}")
        report.append(f"    Accuracy: {metric['accuracy']:.2%}")
        report.append("")

    report.append("Condition-level CSV files are saved separately when metadata is available.")
    report.append("=" * 60)

    report_path = output_path / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"[OK] Report saved: {report_path}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Analyze VLM hallucination evaluation outputs")
    parser.add_argument("--results", "-r", default="results/results.json",
                        help="Path to detailed results JSON")
    parser.add_argument("--metrics", "-m", default="results/metrics.json",
                        help="Path to metrics JSON")
    parser.add_argument("--output", "-o", default="reports",
                        help="Output directory for plots and reports")
    args = parser.parse_args()

    if not Path(args.results).exists():
        print(f"[ERROR] Results file not found: {args.results}")
        return
    if not Path(args.metrics).exists():
        print(f"[ERROR] Metrics file not found: {args.metrics}")
        return

    print("Loading results...")
    data = load_results(args.results, args.metrics)

    try:
        print("Generating plots...")
        plot_results(data["metrics"], data["results"], args.output)
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    print("Generating report...")
    write_report(data["metrics"], args.output)
    print("\n[OK] Analysis complete")


if __name__ == "__main__":
    main()
