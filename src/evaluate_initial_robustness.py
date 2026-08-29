"""Evaluate the clean CLIP linear probe under representative transformations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.clip_features import encode_normalized_images, load_frozen_clip
from src.device import choose_device
from src.image_transforms import (
    DEFAULT_ROBUSTNESS_CONDITIONS,
    TRANSFORM_SPECS,
    apply_evaluation_transform,
)
from src.linear_probe import load_feature_cache, load_linear_probe_checkpoint
from src.metrics import binary_classification_metrics


DEFAULT_TEST_CACHE = Path("data/features/clip_vit_b32_quickgelu_openai/test.npz")
DEFAULT_TEST_MANIFEST = Path("data/processed/test.csv")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


class TransformedPathDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Load cached-reference paths and apply one deterministic transformation."""

    def __init__(
        self,
        image_paths: np.ndarray,
        labels: np.ndarray,
        condition: str,
        clip_preprocess: Any,
        seed: int,
        project_root: Path,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.condition = condition
        self.clip_preprocess = clip_preprocess
        self.seed = seed
        self.project_root = project_root

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        relative_path = str(self.image_paths[index])
        absolute_path = self.project_root / relative_path
        if not absolute_path.is_file():
            raise FileNotFoundError(f"Reference image not found: {absolute_path}")
        with Image.open(absolute_path) as image:
            transformed = apply_evaluation_transform(
                image,
                self.condition,
                image_path=relative_path,
                seed=self.seed,
            )
            tensor = self.clip_preprocess(transformed)
        label = torch.tensor(int(self.labels[index]), dtype=torch.int64)
        return tensor, label, relative_path


def stability_metrics(
    labels: np.ndarray,
    clean_probabilities: np.ndarray,
    transformed_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Measure probability movement and classification changes from clean input."""

    if not (
        labels.shape == clean_probabilities.shape == transformed_probabilities.shape
    ):
        raise ValueError("Labels and probability arrays must have identical shapes")
    differences = transformed_probabilities - clean_probabilities
    clean_predictions = clean_probabilities >= threshold
    transformed_predictions = transformed_probabilities >= threshold
    return {
        "mean_absolute_probability_change": float(np.mean(np.abs(differences))),
        "median_absolute_probability_change": float(np.median(np.abs(differences))),
        "maximum_absolute_probability_change": float(np.max(np.abs(differences))),
        "prediction_flip_rate": float(np.mean(clean_predictions != transformed_predictions)),
        "real_mean_probability_shift": float(np.mean(differences[labels == 0])),
        "ai_generated_mean_probability_shift": float(np.mean(differences[labels == 1])),
    }


def metric_deltas(
    clean_metrics: dict[str, Any], transformed_metrics: dict[str, Any]
) -> dict[str, float]:
    keys = (
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "brier_score",
    )
    return {
        key: float(transformed_metrics[key] - clean_metrics[key]) for key in keys
    }


def evaluate_transformed_condition(
    *,
    condition: str,
    image_paths: np.ndarray,
    labels: np.ndarray,
    clean_probabilities: np.ndarray,
    model: torch.nn.Module,
    clip_preprocess: Any,
    linear_checkpoint: Any,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
    project_root: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    dataset = TransformedPathDataset(
        image_paths,
        labels,
        condition,
        clip_preprocess,
        seed,
        project_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    probability_batches: list[np.ndarray] = []
    observed_labels: list[np.ndarray] = []
    observed_paths: list[str] = []
    started = time.perf_counter()

    for images, batch_labels, batch_paths in tqdm(
        loader, desc=TRANSFORM_SPECS[condition].display_name, unit="batch"
    ):
        features = encode_normalized_images(model, images.to(device)).cpu().numpy()
        probability_batches.append(linear_checkpoint.probabilities(features))
        observed_labels.append(batch_labels.numpy())
        observed_paths.extend(batch_paths)

    probabilities = np.concatenate(probability_batches)
    elapsed_seconds = time.perf_counter() - started
    if observed_paths != image_paths.tolist():
        raise ValueError(f"Path order changed while evaluating {condition}")
    if not np.array_equal(np.concatenate(observed_labels), labels):
        raise ValueError(f"Label order changed while evaluating {condition}")

    metrics = binary_classification_metrics(labels, probabilities, threshold=0.5)
    result = {
        "condition": condition,
        "display_name": TRANSFORM_SPECS[condition].display_name,
        "parameters": TRANSFORM_SPECS[condition].parameters,
        "metrics": metrics,
        "stability_vs_clean": stability_metrics(
            labels, clean_probabilities, probabilities, threshold=0.5
        ),
        "runtime": {
            "device": device.type,
            "seconds": elapsed_seconds,
            "images_per_second": len(labels) / elapsed_seconds,
        },
    }
    return result, probabilities


def _configure_matplotlib() -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aigc-detector-matplotlib")
    )


def write_robustness_plot(results: list[dict[str, Any]], path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["display_name"] for row in results]
    positions = np.arange(len(results))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].bar(
        positions - width / 2,
        [row["metrics"]["roc_auc"] for row in results],
        width,
        label="ROC-AUC",
    )
    axes[0].bar(
        positions + width / 2,
        [row["metrics"]["balanced_accuracy"] for row in results],
        width,
        label="Balanced accuracy",
    )
    axes[0].set(
        title="Clean and representative transformed performance",
        ylabel="Metric value",
        xticks=positions,
        xticklabels=names,
        ylim=(0.0, 1.0),
    )
    axes[0].legend()

    axes[1].bar(
        positions - width / 2,
        [row["stability_vs_clean"]["mean_absolute_probability_change"] for row in results],
        width,
        label="Mean |probability change|",
    )
    axes[1].bar(
        positions + width / 2,
        [row["stability_vs_clean"]["prediction_flip_rate"] for row in results],
        width,
        label="Prediction flip rate",
    )
    axes[1].set(
        title="Prediction stability relative to clean input",
        ylabel="Fraction",
        xticks=positions,
        xticklabels=names,
        ylim=(0.0, 1.0),
    )
    axes[1].legend()
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_transform_sample_figure(
    image_paths: np.ndarray,
    labels: np.ndarray,
    conditions: list[str],
    project_root: Path,
    seed: int,
    path: Path,
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_indices = [int(np.flatnonzero(labels == label)[0]) for label in (0, 1)]
    figure, axes = plt.subplots(2, len(conditions), figsize=(3 * len(conditions), 6))
    if len(conditions) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for row, sample_index in enumerate(sample_indices):
        relative_path = str(image_paths[sample_index])
        with Image.open(project_root / relative_path) as source:
            for column, condition in enumerate(conditions):
                transformed = apply_evaluation_transform(
                    source,
                    condition,
                    image_path=relative_path,
                    seed=seed,
                )
                axes[row, column].imshow(transformed, interpolation="nearest")
                axes[row, column].axis("off")
                if row == 0:
                    axes[row, column].set_title(
                        TRANSFORM_SPECS[condition].display_name, fontsize=10
                    )
        axes[row, 0].text(
            -0.08,
            0.5,
            "Real" if row == 0 else "AI-generated",
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=11,
            clip_on=False,
        )
    figure.suptitle("Representative deterministic evaluation transformations")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate initial clean-versus-transformed CLIP robustness."
    )
    parser.add_argument("--test-cache", type=Path, default=DEFAULT_TEST_CACHE)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/clip_linear_probe.npz")
    )
    parser.add_argument(
        "--model-cache-dir", type=Path, default=Path("checkpoints/open_clip")
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(TRANSFORM_SPECS),
        default=list(DEFAULT_ROBUSTNESS_CONDITIONS),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/clip_initial_robustness.json")
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("reports/figures/clip_initial_robustness.png"),
    )
    parser.add_argument(
        "--sample-figure",
        type=Path,
        default=Path("reports/figures/clip_initial_transform_samples.png"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--maximum-samples",
        type=int,
        help="Debug only: evaluate the first N cached test samples.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative")
    if args.maximum_samples is not None and args.maximum_samples <= 0:
        raise ValueError("maximum-samples must be positive")

    project_root = Path(__file__).resolve().parents[1]
    reference = load_feature_cache(args.test_cache, "test")
    if sha256_file(args.test_manifest) != reference.manifest_sha256:
        raise ValueError("Test manifest no longer matches the cached test features")
    maximum_samples = args.maximum_samples or len(reference.labels)
    image_paths = reference.image_paths[:maximum_samples]
    labels = reference.labels[:maximum_samples]
    features = reference.features[:maximum_samples]
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Selected evaluation subset must contain both labels")

    linear_checkpoint = load_linear_probe_checkpoint(args.checkpoint)
    clean_probabilities = linear_checkpoint.probabilities(features)
    clean_metrics = binary_classification_metrics(labels, clean_probabilities, threshold=0.5)
    clean_result = {
        "condition": "clean",
        "display_name": TRANSFORM_SPECS["clean"].display_name,
        "parameters": {},
        "metrics": clean_metrics,
        "stability_vs_clean": {
            "mean_absolute_probability_change": 0.0,
            "median_absolute_probability_change": 0.0,
            "maximum_absolute_probability_change": 0.0,
            "prediction_flip_rate": 0.0,
            "real_mean_probability_shift": 0.0,
            "ai_generated_mean_probability_shift": 0.0,
        },
        "runtime": {"source": "validated clean test feature cache"},
    }
    results = [clean_result]

    transformed_conditions = [
        condition for condition in dict.fromkeys(args.conditions) if condition != "clean"
    ]
    if transformed_conditions:
        device = choose_device(args.device)
        print(
            f"Loading frozen CLIP for {len(transformed_conditions)} transformed "
            f"conditions on device={device.type}; samples={len(labels)}",
            flush=True,
        )
        model, clip_preprocess = load_frozen_clip(device, args.model_cache_dir)
        for condition in transformed_conditions:
            result, _ = evaluate_transformed_condition(
                condition=condition,
                image_paths=image_paths,
                labels=labels,
                clean_probabilities=clean_probabilities,
                model=model,
                clip_preprocess=clip_preprocess,
                linear_checkpoint=linear_checkpoint,
                device=device,
                batch_size=args.batch_size,
                workers=args.workers,
                seed=args.seed,
                project_root=project_root,
            )
            result["delta_vs_clean"] = metric_deltas(clean_metrics, result["metrics"])
            results.append(result)
            print(
                f"PASS {condition}: auc={result['metrics']['roc_auc']:.6f}, "
                f"bal_acc={result['metrics']['balanced_accuracy']:.6f}, "
                f"flip_rate={result['stability_vs_clean']['prediction_flip_rate']:.6f}",
                flush=True,
            )
    else:
        device = choose_device(args.device)

    write_robustness_plot(results, args.figure)
    figure_conditions = [result["condition"] for result in results]
    write_transform_sample_figure(
        image_paths,
        labels,
        figure_conditions,
        project_root,
        args.seed,
        args.sample_figure,
    )
    report = {
        "experiment": "initial_clean_vs_transformed_clip_evaluation",
        "purpose": (
            "Representative engineering-scale robustness check; not the full "
            "transformation matrix or held-out-generator evaluation."
        ),
        "seed": args.seed,
        "temporary_decision_threshold": 0.5,
        "samples": len(labels),
        "device": device.type,
        "test_manifest": {
            "path": args.test_manifest.as_posix(),
            "sha256": reference.manifest_sha256,
        },
        "test_feature_cache": {
            "path": args.test_cache.as_posix(),
            "sha256": sha256_file(args.test_cache),
        },
        "linear_checkpoint": {
            "path": args.checkpoint.as_posix(),
            "sha256": sha256_file(args.checkpoint),
            "regularization_c": linear_checkpoint.regularization_c,
        },
        "transformation_stage": "before official CLIP preprocessing",
        "results": results,
        "artifacts": {
            "robustness_figure": args.figure.as_posix(),
            "transformation_sample_figure": args.sample_figure.as_posix(),
        },
    }
    atomic_json_write(args.report, report)
    print(f"Updated report: {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
