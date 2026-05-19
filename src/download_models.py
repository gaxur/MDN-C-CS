"""
Download and verify VLM checkpoints for the project.

The downloader uses huggingface_hub.snapshot_download so cluster setup does not
load 7B+ models into memory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict


MODELS: Dict[str, Dict[str, str]] = {
    "llava15": {
        "repo_id": "llava-hf/llava-1.5-7b-hf",
        "local_dir": "llava15",
    },
    "qwen3": {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
        "local_dir": "qwen3-vl",
    },
    "internvl35": {
        "repo_id": "OpenGVLab/InternVL3_5-4B-HF",
        "local_dir": "internvl3_5",
    },
}

MODEL_ALIASES = {
    "llava": "llava15",
    "qwen": "qwen3",
    "qwen3-vl": "qwen3",
    "internvl": "internvl35",
    "internvl3.5": "internvl35",
}


def canonical_model_name(model_name: str) -> str:
    """Normalize user-facing model aliases."""
    key = model_name.lower()
    key = MODEL_ALIASES.get(key, key)
    if key != "all" and key not in MODELS:
        supported = ", ".join(sorted(set(MODELS) | set(MODEL_ALIASES) | {"all"}))
        raise ValueError(f"Unknown model '{model_name}'. Supported values: {supported}")
    return key


def download_model(model_key: str, checkpoints_dir: Path) -> bool:
    """Download one model from Hugging Face Hub without loading it."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] Missing huggingface_hub. Install it with: pip install huggingface_hub")
        return False

    model = MODELS[model_key]
    target_dir = checkpoints_dir / model["local_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 60)
    print(f"DOWNLOADING {model_key.upper()} ({model['repo_id']})")
    print("-" * 60)
    print(f"Destination: {target_dir}")

    try:
        snapshot_download(
            repo_id=model["repo_id"],
            local_dir=str(target_dir),
        )
        print(f"[OK] {model_key.upper()} available at: {target_dir}")
        return True
    except Exception as exc:
        print(f"[ERROR] Failed to download {model_key}: {exc}")
        print("If the compute cluster has no outbound network, download on a login node or use a shared HF_HOME.")
        return False


def download_models(cache_dir: str = "data/checkpoints", model: str = "all") -> bool:
    """Download one or more VLM checkpoints."""
    print("=" * 60)
    print("VLM CHECKPOINT DOWNLOAD")
    print("=" * 60)

    checkpoints_dir = Path(cache_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Checkpoint directory: {checkpoints_dir.absolute()}")

    model = canonical_model_name(model)
    selected = list(MODELS) if model == "all" else [model]
    ok = True
    for model_key in selected:
        ok = download_model(model_key, checkpoints_dir) and ok

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE" if ok else "DOWNLOAD FINISHED WITH ERRORS")
    print("=" * 60)
    return ok


def test_model(cache_dir: str = "data/checkpoints", allow_cpu: bool = False) -> bool:
    """Run a tiny local smoke test with LLaVA."""
    print("\n" + "=" * 60)
    print("INFERENCE SMOKE TEST")
    print("=" * 60)

    try:
        from PIL import Image, ImageDraw
        from evaluate_vlms import VLMRunner
    except ImportError as exc:
        print(f"\n[ERROR] Dependencies are not available: {exc}")
        return False

    test_image = Path("data/images/test_download_model.jpg")
    test_image.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (384, 384), color=(245, 247, 250))
    draw = ImageDraw.Draw(img)
    draw.ellipse([(110, 110), (270, 270)], fill=(220, 55, 45), outline=(35, 35, 35), width=4)
    img.save(test_image)

    try:
        runner = VLMRunner("llava15", cache_dir=cache_dir, require_gpu=not allow_cpu, local_files_only=True)
        response = runner.generate(str(test_image), "Is there a red circle in this image?")
        print(f"\n[RESULT] {response}")
        return True
    except Exception as exc:
        print(f"\n[ERROR] Smoke test failed: {exc}")
        return False


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Download and test VLM checkpoints")
    parser.add_argument("--download", action="store_true", help="Download checkpoints")
    parser.add_argument("--test", action="store_true", help="Run a local LLaVA smoke test")
    parser.add_argument("--all", action="store_true", help="Download checkpoints and run the smoke test")
    parser.add_argument("--model", choices=sorted(set(MODELS) | set(MODEL_ALIASES) | {"all"}),
                        default="all", help="Model to download")
    parser.add_argument("--cache-dir", default="data/checkpoints", help="Checkpoint directory")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow the smoke test without a GPU")
    args = parser.parse_args()

    if args.all:
        download_models(cache_dir=args.cache_dir, model=args.model)
        test_model(cache_dir=args.cache_dir, allow_cpu=args.allow_cpu)
        print("\nSetup complete.")
        return

    if args.download or not sys.stdin.isatty():
        download_models(cache_dir=args.cache_dir, model=args.model)
        print("\nSetup complete.")
        return

    if args.test:
        test_model(cache_dir=args.cache_dir, allow_cpu=args.allow_cpu)
        print("\nSetup complete.")
        return

    print("\nWelcome to the VLM model setup for CVCS 2026")
    print("\nOptions:")
    print("  1. Download checkpoints")
    print("  2. Run smoke test")
    print("  3. Both")
    print("  0. Exit")

    choice = input("\nChoose an option (0-3): ").strip()

    if choice == "1":
        download_models(cache_dir=args.cache_dir, model=args.model)
    elif choice == "2":
        test_model(cache_dir=args.cache_dir, allow_cpu=args.allow_cpu)
    elif choice == "3":
        download_models(cache_dir=args.cache_dir, model=args.model)
        test_model(cache_dir=args.cache_dir, allow_cpu=args.allow_cpu)
    else:
        print("Exiting...")
        return

    print("\nSetup complete.")


if __name__ == "__main__":
    main()
