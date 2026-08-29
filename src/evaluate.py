"""Evaluate a trained CIFAKE smoke-test checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import ManifestImageDataset, build_image_transform
from src.device import choose_device
from src.metrics import evaluate_model
from src.model import load_checkpoint


def _write_confusion_matrix(matrix: list[list[int]], output_path: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aigc-detector-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Real", "AI-generated"],
        yticklabels=["Real", "AI-generated"],
        xlabel="Predicted label",
        ylabel="True label",
        title="CIFAKE smoke-test confusion matrix",
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_predictions(
    output_path: Path,
    image_paths: list[str],
    labels: list[float],
    probabilities: list[float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "label", "pred"])
        writer.writeheader()
        writer.writerows(
            {
                "image_path": path,
                "label": int(label),
                "pred": round(float(probability), 6),
            }
            for path, label, probability in zip(
                image_paths, labels, probabilities, strict=True
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAKE smoke-test checkpoint.")
    parser.add_argument("--test-csv", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cifake_cnn.pt"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/cifake_smoke_test.json")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("reports/figures/cifake_confusion_matrix.png")
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("outputs/cifake_test_predictions.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative")

    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint)
    preprocessing = checkpoint["preprocessing"]
    transform = build_image_transform(
        image_size=int(preprocessing["image_size"]),
        mean=preprocessing["mean"],
        std=preprocessing["std"],
    )
    dataset = ManifestImageDataset(args.test_csv, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model.to(device)
    metrics, labels, probabilities, image_paths = evaluate_model(
        model,
        loader,
        nn.BCEWithLogitsLoss(),
        device,
        threshold=args.threshold,
    )

    result: dict[str, Any] = {
        "experiment": "cifake_smoke_cnn",
        "purpose": "pipeline smoke test; not a real-world robustness claim",
        "device": device.type,
        "test_manifest": args.test_csv.as_posix(),
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_metrics": checkpoint.get("metrics", {}),
        "test_metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_predictions(args.predictions_output, image_paths, labels, probabilities)
    _write_confusion_matrix(metrics["confusion_matrix"], args.figure)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
