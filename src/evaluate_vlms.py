"""
Evaluate vision-language models on the hallucination probing benchmark.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


sys.path.append(str(Path(__file__).parent.resolve()))

torch = None
Image = None
AutoProcessor = None
AutoModelForImageTextToText = None


MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "llava15": {
        "repo_id": "llava-hf/llava-1.5-7b-hf",
        "local_dir": "llava15",
        "prompt_family": "llava",
    },
    "qwen3": {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
        "local_dir": "qwen3-vl",
        "prompt_family": "chat",
    },
    "internvl35": {
        "repo_id": "OpenGVLab/InternVL3_5-4B-HF",
        "local_dir": "internvl3_5",
        "prompt_family": "chat",
    },
}

MODEL_ALIASES = {
    "llava": "llava15",
    "llava-v1.5": "llava15",
    "qwen": "qwen3",
    "qwen3-vl": "qwen3",
    "internvl": "internvl35",
    "internvl3.5": "internvl35",
}


def load_runtime_dependencies() -> None:
    """Import heavyweight runtime dependencies only when evaluation starts."""
    global torch
    global Image
    global AutoProcessor
    global AutoModelForImageTextToText

    if torch is not None:
        return

    try:
        import torch as torch_module
        from PIL import Image as PILImage
        from transformers import AutoModelForImageTextToText as AutoModel
        from transformers import AutoProcessor as Processor
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import runtime dependencies: {exc}\n"
            "Install them with: pip install -r requirements.txt"
        ) from exc

    torch = torch_module
    Image = PILImage
    AutoProcessor = Processor
    AutoModelForImageTextToText = AutoModel


def canonical_model_name(model_name: str) -> str:
    """Normalize model aliases used by older scripts."""
    key = model_name.lower()
    key = MODEL_ALIASES.get(key, key)
    if key not in MODEL_CONFIGS:
        supported = ", ".join(sorted(set(MODEL_CONFIGS) | set(MODEL_ALIASES)))
        raise ValueError(f"Unknown model '{model_name}'. Supported values: {supported}")
    return key


class VLMRunner:
    """Run VLM inference for yes/no hallucination probes."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str = "data/checkpoints",
        dtype: str = "auto",
        device_map: str = "auto",
        require_gpu: bool = True,
        local_files_only: bool = False,
        prompt_mode: str = "baseline",
    ):
        load_runtime_dependencies()
        self.model_name = canonical_model_name(model_name)
        self.config = MODEL_CONFIGS[self.model_name]
        self.cache_dir = Path(cache_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = self._resolve_dtype(dtype)
        self.device_map = None if device_map == "none" else device_map
        self.local_files_only = local_files_only
        self.prompt_mode = prompt_mode

        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU was detected. On the cluster, request a GPU with SLURM "
                "(for example --gres=gpu:1). Use --allow-cpu only for tiny smoke tests."
            )

        self._load_model()
        self.model.eval()
        print(f"[OK] Loaded {self.model_name} on {self.device}")

    def _load_model(self) -> None:
        local_path = self.cache_dir / self.config["local_dir"]
        source = str(local_path) if local_path.exists() else self.config["repo_id"]
        if not local_path.exists():
            print(f"[WARN] Local checkpoint not found: {local_path}")
            print(f"[INFO] Loading from Hugging Face: {self.config['repo_id']}")

        self.processor = AutoProcessor.from_pretrained(
            source,
            cache_dir=str(self.cache_dir),
            local_files_only=self.local_files_only if not local_path.exists() else True,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            source,
            **self._model_kwargs(local_path_exists=local_path.exists()),
        )

        if self.device_map is None:
            self.model = self.model.to(self.device)

    def _model_kwargs(self, local_path_exists: bool) -> Dict[str, Any]:
        """Model loading arguments suitable for large checkpoints on a GPU cluster."""
        kwargs: Dict[str, Any] = {
            "cache_dir": str(self.cache_dir),
            "low_cpu_mem_usage": True,
            "local_files_only": self.local_files_only if not local_path_exists else True,
            "trust_remote_code": True,
        }
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        return kwargs

    def _resolve_dtype(self, dtype: str) -> Optional["torch.dtype"]:
        """Map CLI dtype values to torch dtypes."""
        dtype = dtype.lower()
        if dtype == "auto":
            if not torch.cuda.is_available():
                return None
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp16", "float16"}:
            return torch.float16
        if dtype in {"fp32", "float32"}:
            return torch.float32
        raise ValueError("dtype must be one of: auto, bf16, fp16, fp32")

    def generate(self, image_path: str, prompt: str, max_new_tokens: int = 16) -> str:
        """Generate a model answer for one image-question pair."""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            return f"ERROR_READING_IMAGE: {exc}"

        inputs = self._prepare_inputs(image, prompt)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        if "input_ids" in inputs:
            output = output[:, inputs["input_ids"].shape[-1]:]
        return self.processor.batch_decode(output, skip_special_tokens=True)[0].strip()

    def _prepare_inputs(self, image: "Image.Image", prompt: str) -> Dict[str, Any]:
        instruction = self._build_instruction(prompt)
        if self.config["prompt_family"] == "llava":
            text = f"USER: <image>\n{instruction}\nASSISTANT:"
            return self.processor(text=text, images=image, return_tensors="pt").to(self.input_device)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.processor(text=[text], images=[image], return_tensors="pt").to(self.input_device)

    def _build_instruction(self, question: str) -> str:
        if self.prompt_mode == "strict":
            return (
                f"{question}\n"
                "Answer only yes or no. Verify the visual evidence carefully. "
                "If the requested object, attribute, relation, or count is not clearly visible, answer no. "
                "Do not infer likely objects from context."
            )
        return f"{question}\nAnswer only yes or no."


def extract_yes_no(response: str) -> str:
    """Extract a normalized yes/no answer from free-form model output."""
    response_lower = response.lower()
    if re.search(r"\b(no|nope|nah|not|false)\b", response_lower):
        return "no"
    if re.search(r"\b(yes|yeah|yep|true)\b", response_lower):
        return "yes"
    return "unknown"


def evaluate_benchmark(
    runner: VLMRunner,
    benchmark_path: str,
    images_dir: str,
    max_samples: int | None = None,
) -> List[Dict[str, Any]]:
    """Evaluate one model on a JSON hallucination benchmark."""
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    if max_samples:
        benchmark = benchmark[:max_samples]

    print(f"\nEvaluating {len(benchmark)} questions...")
    print("-" * 60)

    results = []
    start_time = time.time()

    for i, entry in enumerate(benchmark):
        image_path = Path(images_dir) / entry["image_id"]
        if not image_path.exists():
            alt_path = Path("data") / entry["image_id"]
            if alt_path.exists():
                image_path = alt_path
            else:
                print(f"[WARN] Image not found: {entry['image_id']}")
                results.append({
                    **entry,
                    "response": "IMAGE_NOT_FOUND",
                    "prediction": "unknown",
                    "correct": False,
                    "prompt_mode": runner.prompt_mode,
                    "model": runner.model_name,
                })
                continue

        response = runner.generate(str(image_path), entry["question"])
        prediction = extract_yes_no(response)
        ground_truth = entry["ground_truth"].lower()
        correct = prediction == ground_truth

        results.append({
            **entry,
            "response": response,
            "prediction": prediction,
            "correct": correct,
            "prompt_mode": runner.prompt_mode,
            "model": runner.model_name,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            print(f"[{i + 1}/{len(benchmark)}] ({speed:.2f} samples/sec)")

    elapsed = time.time() - start_time
    print(f"\nEvaluation finished in {elapsed:.2f}s")
    return results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy, hallucination, POPE-style, and condition metrics."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    yes_predictions = sum(1 for r in results if r["prediction"] == "yes")
    no_predictions = sum(1 for r in results if r["prediction"] == "no")
    unknown_predictions = sum(1 for r in results if r["prediction"] == "unknown")

    by_type = grouped_accuracy(results, "question_type")
    by_condition = {
        "target_object": grouped_accuracy(results, "target_object"),
        "absent_object": grouped_accuracy(results, "absent_object"),
        "claimed_color": grouped_accuracy(results, "claimed_color"),
        "claimed_relation": grouped_accuracy(results, "claimed_relation"),
        "claimed_count": grouped_accuracy(results, "claimed_count"),
    }

    # This benchmark is made of negative POPE-style probes: a "yes" is a false positive hallucination.
    pope = {
        "accuracy": correct / total if total else 0,
        "false_positive_rate": yes_predictions / total if total else 0,
        "unknown_rate": unknown_predictions / total if total else 0,
    }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "hallucination_rate": yes_predictions / total if total else 0,
        "yes_predictions": yes_predictions,
        "no_predictions": no_predictions,
        "unknown_predictions": unknown_predictions,
        "pope": pope,
        "by_type": by_type,
        "by_condition": by_condition,
    }


def grouped_accuracy(results: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    """Aggregate accuracy by a metadata key if the key exists in results."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for result in results:
        if key not in result:
            continue
        value = str(result[key])
        grouped.setdefault(value, {"correct": 0, "total": 0})
        grouped[value]["total"] += 1
        if result["correct"]:
            grouped[value]["correct"] += 1

    for value in grouped.values():
        value["accuracy"] = value["correct"] / value["total"] if value["total"] else 0
    return grouped


def save_results(results: List[Dict[str, Any]], metrics: Dict[str, Any], output_dir: str) -> None:
    """Save detailed results, metrics, and a compact summary."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Results saved to: {results_file}")

    metrics_file = output_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[OK] Metrics saved to: {metrics_file}")

    summary = {
        "accuracy": metrics["accuracy"],
        "hallucination_rate": metrics["hallucination_rate"],
        "pope_false_positive_rate": metrics["pope"]["false_positive_rate"],
        "correct": metrics["correct"],
        "total": metrics["total"],
    }
    for qtype, metric in metrics["by_type"].items():
        summary[f"{qtype}_accuracy"] = metric["accuracy"]
        summary[f"{qtype}_total"] = metric["total"]

    summary_file = output_path / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Summary saved to: {summary_file}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Evaluate VLMs on hallucination probes")
    parser.add_argument("--model", "-m", required=True, choices=sorted(set(MODEL_CONFIGS) | set(MODEL_ALIASES)),
                        help="Model to evaluate")
    parser.add_argument("--benchmark", "-b", default="data/benchmark.json",
                        help="Path to the benchmark JSON file")
    parser.add_argument("--images", "-i", default="data/images",
                        help="Directory containing benchmark images")
    parser.add_argument("--output", "-o", default="results",
                        help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--cache-dir", default="data/checkpoints",
                        help="Checkpoint/cache directory")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                        help="Model loading precision")
    parser.add_argument("--device-map", default="auto",
                        help="Transformers/Accelerate device_map; use 'none' to call .to(device)")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Allow CPU execution (very slow; smoke tests only)")
    parser.add_argument("--offline", action="store_true",
                        help="Use local cached files only")
    parser.add_argument("--prompt-mode", default="baseline", choices=["baseline", "strict"],
                        help="Prompting strategy; 'strict' is the mitigation condition")
    args = parser.parse_args()

    try:
        load_runtime_dependencies()
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] No GPU detected")

    print(f"\nLoading {args.model}...")
    runner = VLMRunner(
        args.model,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        require_gpu=not args.allow_cpu,
        local_files_only=args.offline,
        prompt_mode=args.prompt_mode,
    )

    print(f"\nEvaluating benchmark: {args.benchmark}")
    results = evaluate_benchmark(runner, args.benchmark, args.images, args.max_samples)

    print("\nComputing metrics...")
    metrics = compute_metrics(results)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"Hallucination rate: {metrics['hallucination_rate']:.2%}")
    print(f"POPE false-positive rate: {metrics['pope']['false_positive_rate']:.2%}")
    print("\nBy question type:")
    for qtype, metric in metrics["by_type"].items():
        print(f"  {qtype}: {metric['accuracy']:.2%} ({metric['correct']}/{metric['total']})")

    print(f"\nSaving outputs to: {args.output}")
    save_results(results, metrics, args.output)
    print("\n[OK] Evaluation complete")


if __name__ == "__main__":
    main()
