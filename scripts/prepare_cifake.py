#!/usr/bin/env python3
"""Create deterministic, leakage-safe CIFAKE manifests.

The script does not copy images. It records repository-relative image paths and
standardises the project-wide binary label convention:

    0 = authentic/real
    1 = AI-generated/fake
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


CLASS_DEFINITIONS = {
    "REAL": {"label": 0, "class_name": "real"},
    "FAKE": {"label": 1, "class_name": "ai_generated"},
}
FIELDNAMES = ["image_path", "label", "class_name", "source", "split"]


def _require_images(directory: Path, minimum: int) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required CIFAKE directory not found: {directory}")

    paths = sorted(directory.glob("*.jpg"))
    if len(paths) < minimum:
        raise ValueError(
            f"{directory} contains {len(paths)} JPEGs, but at least {minimum} "
            "are required."
        )
    return paths


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Image path {path} is outside project root {project_root}; "
            "portable manifests require data inside the project."
        ) from exc


def _record(path: Path, folder_name: str, split: str, project_root: Path) -> dict[str, Any]:
    definition = CLASS_DEFINITIONS[folder_name]
    return {
        "image_path": _relative_path(path, project_root),
        "label": definition["label"],
        "class_name": definition["class_name"],
        "source": "cifake",
        "split": split,
    }


def _verify_images(records: list[dict[str, Any]], project_root: Path) -> None:
    errors: list[str] = []
    for row in records:
        path = project_root / row["image_path"]
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:  # Pillow raises several format-specific errors.
            errors.append(f"{path}: {exc}")

    if errors:
        preview = "\n".join(errors[:10])
        raise ValueError(f"Found {len(errors)} unreadable images:\n{preview}")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def _validate_records(split_records: dict[str, list[dict[str, Any]]]) -> None:
    all_paths: list[str] = []
    for split, records in split_records.items():
        labels = {int(row["label"]) for row in records}
        if labels != {0, 1}:
            raise ValueError(f"{split} contains invalid label set: {sorted(labels)}")
        if any(row["split"] != split for row in records):
            raise ValueError(f"One or more rows in {split} have an incorrect split value")
        all_paths.extend(str(row["image_path"]) for row in records)

    duplicate_count = len(all_paths) - len(set(all_paths))
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} image paths shared across manifests")


def build_manifests(
    raw_root: Path,
    output_dir: Path,
    project_root: Path,
    train_per_class: int = 5_000,
    val_per_class: int = 1_000,
    test_per_class: int = 1_000,
    seed: int = 42,
    verify_images: bool = False,
) -> dict[str, Any]:
    """Build CIFAKE train/validation/test CSV manifests and return a summary."""

    if min(train_per_class, val_per_class, test_per_class) <= 0:
        raise ValueError("Every per-class sample count must be positive")

    split_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for folder_name, definition in CLASS_DEFINITIONS.items():
        label = int(definition["label"])
        official_train = _require_images(
            raw_root / "train" / folder_name,
            train_per_class + val_per_class,
        )
        official_test = _require_images(
            raw_root / "test" / folder_name,
            test_per_class,
        )

        train_rng = random.Random(seed + label)
        train_rng.shuffle(official_train)
        val_paths = official_train[:val_per_class]
        train_paths = official_train[val_per_class : val_per_class + train_per_class]

        test_rng = random.Random(seed + 100 + label)
        test_rng.shuffle(official_test)
        test_paths = official_test[:test_per_class]

        split_records["train"].extend(
            _record(path, folder_name, "train", project_root) for path in train_paths
        )
        split_records["val"].extend(
            _record(path, folder_name, "val", project_root) for path in val_paths
        )
        split_records["test"].extend(
            _record(path, folder_name, "test", project_root) for path in test_paths
        )

    # Avoid class-blocked CSVs while preserving deterministic row order.
    for offset, records in enumerate(split_records.values()):
        random.Random(seed + 200 + offset).shuffle(records)

    _validate_records(split_records)

    all_records = [row for records in split_records.values() for row in records]
    if verify_images:
        _verify_images(all_records, project_root)

    for split, records in split_records.items():
        _write_csv(output_dir / f"{split}.csv", records)

    summary: dict[str, Any] = {
        "dataset": "CIFAKE",
        "seed": seed,
        "label_mapping": {"real": 0, "ai_generated": 1},
        "raw_root": _relative_path(raw_root, project_root),
        "images_are_copied": False,
        "split_strategy": {
            "train": "sampled from CIFAKE official train split",
            "val": "disjoint sample from CIFAKE official train split",
            "test": "sampled from CIFAKE official test split",
        },
        "splits": {},
        "total_images": len(all_records),
        "duplicate_paths_across_splits": 0,
        "image_verification_requested": verify_images,
    }

    for split, records in split_records.items():
        counts = Counter(row["class_name"] for row in records)
        summary["splits"][split] = {
            "total": len(records),
            "real": counts["real"],
            "ai_generated": counts["ai_generated"],
        }

    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic CIFAKE CSV manifests without copying images."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/cifake"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-per-class", type=int, default=5_000)
    parser.add_argument("--val-per-class", type=int, default=1_000)
    parser.add_argument("--test-per-class", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Open and verify every selected image before writing manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    raw_root = args.raw_root if args.raw_root.is_absolute() else project_root / args.raw_root
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    )

    summary = build_manifests(
        raw_root=raw_root,
        output_dir=output_dir,
        project_root=project_root,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
        verify_images=args.verify_images,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
