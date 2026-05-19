"""
Benchmark generator for VLM evaluation.

Creates simple synthetic scenes and self-consistent trap questions to measure
visual hallucination across four categories: object_absent, attribute, relation,
and count.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


QUESTION_TYPES = ("object_absent", "attribute", "relation", "count")
SHAPES = ("circle", "square", "triangle")
COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (220, 55, 45),
    "blue": (50, 105, 220),
    "green": (60, 160, 90),
    "yellow": (235, 190, 45),
    "purple": (135, 80, 190),
    "orange": (230, 130, 45),
}
RELATION_TEXT = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "above": "above",
    "below": "below",
}


def plural(shape: str) -> str:
    """Return a simple plural for the synthetic object names."""
    return f"{shape}s"


class BenchmarkGenerator:
    """VLM hallucination benchmark generator."""

    def __init__(self, output_dir: str = "data", image_size: int = 384):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.images_dir = self.output_dir / "images"

    def generate_benchmark(
        self,
        num_samples: int = 100,
        split: Optional[Dict[str, float]] = None,
        save: bool = True,
        generate_images: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate the full benchmark.

        Args:
            num_samples: Total number of questions.
            split: Proporciones train/val/test.
            save: Whether to save JSON files.
            generate_images: Whether to create synthetic images under data/images.

        Returns:
            Dictionary with train/val/test splits.
        """
        if split is None:
            split = {"train": 0.7, "val": 0.15, "test": 0.15}

        split = self._normalize_split(split)
        benchmark = {"train": [], "val": [], "test": []}

        if generate_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            question_type = QUESTION_TYPES[i % len(QUESTION_TYPES)]
            image_id = f"image_{i:04d}.jpg"
            entry, scene = self._make_entry(question_type, image_id)

            if generate_images:
                self._draw_scene(scene, self.images_dir / image_id)

            split_name = self._split_for_index(i, num_samples, split)
            benchmark[split_name].append(entry)

        if save:
            self._save_benchmark(benchmark)

        return benchmark

    def generate_mscoco_style(self, num_images: int = 100) -> List[str]:
        """
        Backward-compatible helper.

        Generates the self-consistent benchmark and returns image paths.
        """
        self.generate_benchmark(num_samples=num_images, generate_images=True)
        return [str(self.images_dir / f"image_{i:04d}.jpg") for i in range(num_images)]

    def _make_entry(self, question_type: str, image_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if question_type == "object_absent":
            return self._object_absent_entry(image_id)
        if question_type == "attribute":
            return self._attribute_entry(image_id)
        if question_type == "relation":
            return self._relation_entry(image_id)
        if question_type == "count":
            return self._count_entry(image_id)
        raise ValueError(f"Unsupported question type: {question_type}")

    def _object_absent_entry(self, image_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        present_shapes = random.sample(list(SHAPES), 2)
        absent_shape = random.choice([shape for shape in SHAPES if shape not in present_shapes])
        objects = [
            {"shape": present_shapes[0], "color": random.choice(list(COLORS)), "position": "left"},
            {"shape": present_shapes[1], "color": random.choice(list(COLORS)), "position": "right"},
        ]
        question = random.choice([
            "Is there a {obj} in this image?",
            "Do you see a {obj} in the picture?",
            "Is a {obj} present in the image?",
        ]).format(obj=absent_shape)
        scene = {"objects": objects}
        return self._entry(image_id, question, "object_absent", scene, {"absent_object": absent_shape}), scene

    def _attribute_entry(self, image_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        target_shape = random.choice(list(SHAPES))
        true_color, wrong_color = random.sample(list(COLORS), 2)
        distractor_shape = random.choice([shape for shape in SHAPES if shape != target_shape])
        objects = [
            {"shape": target_shape, "color": true_color, "position": "center"},
            {"shape": distractor_shape, "color": random.choice(list(COLORS)), "position": "bottom_right", "scale": 0.7},
        ]
        question = random.choice([
            "Is the {target} {wrong_color}?",
            "Does the {target} appear {wrong_color}?",
            "Is the {target} {wrong_color} in color?",
        ]).format(target=target_shape, wrong_color=wrong_color)
        scene = {"objects": objects}
        metadata = {
            "target_object": target_shape,
            "true_color": true_color,
            "claimed_color": wrong_color,
        }
        return self._entry(image_id, question, "attribute", scene, metadata), scene

    def _relation_entry(self, image_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        shape1, shape2 = random.sample(list(SHAPES), 2)
        true_relation = random.choice(list(RELATION_TEXT))
        wrong_relation = random.choice([rel for rel in RELATION_TEXT if rel != true_relation])
        pos1, pos2 = self._positions_for_relation(true_relation)
        objects = [
            {"shape": shape1, "color": random.choice(list(COLORS)), "position": pos1},
            {"shape": shape2, "color": random.choice(list(COLORS)), "position": pos2},
        ]
        question = random.choice([
            "Is the {obj1} {relation} the {obj2}?",
            "Is the {obj1} located {relation} the {obj2}?",
            "Does the {obj1} appear {relation} the {obj2}?",
        ]).format(obj1=shape1, relation=RELATION_TEXT[wrong_relation], obj2=shape2)
        scene = {"objects": objects}
        metadata = {
            "object_1": shape1,
            "object_2": shape2,
            "true_relation": RELATION_TEXT[true_relation],
            "claimed_relation": RELATION_TEXT[wrong_relation],
        }
        return self._entry(image_id, question, "relation", scene, metadata), scene

    def _count_entry(self, image_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        target_shape = random.choice(list(SHAPES))
        true_count = random.choice([2, 3, 4])
        wrong_count = random.choice([count for count in [1, 2, 3, 4, 5] if count != true_count])
        positions = ["top_left", "top_right", "bottom_left", "bottom_right"][:true_count]
        color = random.choice(list(COLORS))
        objects = [{"shape": target_shape, "color": color, "position": position, "scale": 0.68} for position in positions]

        distractor_shape = random.choice([shape for shape in SHAPES if shape != target_shape])
        objects.append({"shape": distractor_shape, "color": random.choice(list(COLORS)), "position": "center", "scale": 0.55})

        question = random.choice([
            "Are there {count} {obj} in this image?",
            "Is the number of {obj} equal to {count}?",
            "Do you see exactly {count} {obj}?",
        ]).format(count=wrong_count, obj=plural(target_shape))
        scene = {"objects": objects}
        metadata = {
            "target_object": target_shape,
            "true_count": true_count,
            "claimed_count": wrong_count,
        }
        return self._entry(image_id, question, "count", scene, metadata), scene

    def _entry(
        self,
        image_id: str,
        question: str,
        question_type: str,
        scene: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        present_objects = [
            {"shape": obj["shape"], "color": obj["color"], "position": obj["position"]}
            for obj in scene["objects"]
        ]
        return {
            "image_id": image_id,
            "question": question,
            "ground_truth": "no",
            "question_type": question_type,
            "answer_options": ["yes", "no"],
            "present_objects": present_objects,
            **metadata,
        }

    def _draw_scene(self, scene: Dict[str, Any], output_path: Path) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise RuntimeError("Pillow is not installed. Run: pip install pillow") from exc

        img = Image.new("RGB", (self.image_size, self.image_size), color=(245, 247, 250))
        draw = ImageDraw.Draw(img)

        for obj in scene["objects"]:
            center = self._position_to_xy(obj["position"])
            scale = obj.get("scale", 1.0)
            radius = int(58 * scale)
            color = COLORS[obj["color"]]
            self._draw_shape(draw, obj["shape"], center, radius, color)

        img.save(output_path, quality=95)

    def _draw_shape(self, draw: Any, shape: str, center: Tuple[int, int], radius: int, color: Tuple[int, int, int]) -> None:
        x, y = center
        bbox = [(x - radius, y - radius), (x + radius, y + radius)]
        outline = (35, 35, 35)
        if shape == "circle":
            draw.ellipse(bbox, fill=color, outline=outline, width=4)
        elif shape == "square":
            draw.rectangle(bbox, fill=color, outline=outline, width=4)
        elif shape == "triangle":
            points = [(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)]
            draw.polygon(points, fill=color, outline=outline)
            draw.line(points + [points[0]], fill=outline, width=4)
        else:
            raise ValueError(f"Unsupported shape: {shape}")

    def _position_to_xy(self, position: str) -> Tuple[int, int]:
        size = self.image_size
        positions = {
            "left": (int(size * 0.32), int(size * 0.5)),
            "right": (int(size * 0.68), int(size * 0.5)),
            "center": (int(size * 0.5), int(size * 0.5)),
            "top": (int(size * 0.5), int(size * 0.30)),
            "bottom": (int(size * 0.5), int(size * 0.70)),
            "top_left": (int(size * 0.30), int(size * 0.30)),
            "top_right": (int(size * 0.70), int(size * 0.30)),
            "bottom_left": (int(size * 0.30), int(size * 0.70)),
            "bottom_right": (int(size * 0.70), int(size * 0.70)),
        }
        return positions[position]

    def _positions_for_relation(self, relation: str) -> Tuple[str, str]:
        if relation == "left_of":
            return "left", "right"
        if relation == "right_of":
            return "right", "left"
        if relation == "above":
            return "top", "bottom"
        if relation == "below":
            return "bottom", "top"
            raise ValueError(f"Unsupported relation: {relation}")

    def _normalize_split(self, split: Dict[str, float]) -> Dict[str, float]:
        expected = {"train", "val", "test"}
        if set(split) != expected:
            raise ValueError(f"Split must contain exactly: {sorted(expected)}")
        total = sum(split.values())
        if total <= 0:
            raise ValueError("Split values must sum to a positive number")
        return {key: value / total for key, value in split.items()}

    def _split_for_index(self, index: int, total: int, split: Dict[str, float]) -> str:
        train_end = int(total * split["train"])
        val_end = int(total * (split["train"] + split["val"]))
        if index < train_end:
            return "train"
        if index < val_end:
            return "val"
        return "test"

    def _save_benchmark(self, benchmark: Dict[str, List[Dict[str, Any]]]) -> None:
        for split_name, entries in benchmark.items():
            filename = self.output_dir / f"benchmark_{split_name}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
            print(f"Saved: {filename} ({len(entries)} entries)")

        all_entries = []
        for entries in benchmark.values():
            all_entries.extend(entries)
        with open(self.output_dir / "benchmark.json", "w", encoding="utf-8") as f:
            json.dump(all_entries, f, indent=2, ensure_ascii=False)
        print(f"Saved: {self.output_dir / 'benchmark.json'} ({len(all_entries)} entries)")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate a trap-question benchmark for VLMs")
    parser.add_argument("--num-samples", type=int, default=100, help="Total number of questions to generate")
    parser.add_argument("--output-dir", default="data", help="Directory for benchmark files and images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-images", action="store_true", help="Do not generate synthetic images")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("VLM BENCHMARK GENERATOR")
    print("=" * 60)

    generator = BenchmarkGenerator(output_dir=args.output_dir)

    print(f"\nGenerating a benchmark with {args.num_samples} questions...")
    benchmark = generator.generate_benchmark(
        num_samples=args.num_samples,
        generate_images=not args.no_images,
    )

    print("\nSummary:")
    for split_name, entries in benchmark.items():
        print(f"  {split_name}: {len(entries)} entries")

    print("\nBy question type:")
    type_counts: Dict[str, int] = {}
    for entries in benchmark.values():
        for entry in entries:
            qtype = entry["question_type"]
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}")

    if not args.no_images:
        print(f"\nImages saved to: {generator.images_dir}")

    print("\nNext steps:")
    print("  1. Run src/evaluate_vlms.py on a GPU")
    print("  2. Use src/analyze_results.py to generate plots and reports")


if __name__ == "__main__":
    main()
