"""Train the compact CIFAKE smoke-test CNN."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data import ManifestImageDataset, build_image_transform
from src.device import choose_device
from src.metrics import binary_classification_metrics, evaluate_model
from src.model import CIFAKESmokeCNN, save_checkpoint


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable smoke-test runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    """Train for one epoch and calculate metrics from the observed batches."""

    model.train()
    labels: list[float] = []
    probabilities: list[float] = []
    total_loss = 0.0
    total_examples = 0

    for images, batch_labels, _ in loader:
        images = images.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        batch_size = batch_labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        labels.extend(batch_labels.detach().cpu().tolist())
        probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())

    metrics = binary_classification_metrics(labels, probabilities)
    metrics["loss"] = total_loss / total_examples
    metrics["samples"] = total_examples
    return metrics


def _write_training_plot(history: list[dict[str, Any]], output_path: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aigc-detector-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train"]["loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["val"]["loss"] for row in history], marker="o", label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCE loss")
    axes[0].legend()

    axes[1].plot(
        epochs, [row["train"]["roc_auc"] for row in history], marker="o", label="train"
    )
    axes[1].plot(
        epochs, [row["val"]["roc_auc"] for row in history], marker="o", label="validation"
    )
    axes[1].set(title="ROC-AUC", xlabel="Epoch", ylabel="ROC-AUC", ylim=(0.0, 1.0))
    axes[1].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CIFAKE smoke-test CNN.")
    parser.add_argument("--train-csv", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/processed/val.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cifake_cnn.pt"))
    parser.add_argument(
        "--history-output", type=Path, default=Path("reports/cifake_training_history.json")
    )
    parser.add_argument(
        "--figure-output", type=Path, default=Path("reports/figures/cifake_training_history.png")
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs and batch-size must be positive; workers cannot be negative")

    seed_everything(args.seed)
    device = choose_device(args.device)
    transform = build_image_transform()
    train_dataset = ManifestImageDataset(args.train_csv, transform=transform)
    val_dataset = ManifestImageDataset(args.val_csv, transform=transform)
    generator = torch.Generator().manual_seed(args.seed)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = CIFAKESmokeCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(
        f"Training {parameter_count:,}-parameter model on device={device.type}; "
        f"train={len(train_dataset)}, val={len(val_dataset)}",
        flush=True,
    )

    history: list[dict[str, Any]] = []
    best_auc = float("-inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["roc_auc"] > best_auc:
            best_auc = float(val_metrics["roc_auc"])
            best_epoch = epoch
            save_checkpoint(args.checkpoint, model, epoch=epoch, metrics=val_metrics)

        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_auc={train_metrics['roc_auc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )

    summary = {
        "experiment": "cifake_smoke_cnn",
        "purpose": "pipeline smoke test; not a real-world robustness claim",
        "seed": args.seed,
        "device": device.type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "parameter_count": parameter_count,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "best_epoch": best_epoch,
        "best_val_roc_auc": best_auc,
        "checkpoint": args.checkpoint.as_posix(),
        "history": history,
    }
    args.history_output.parent.mkdir(parents=True, exist_ok=True)
    args.history_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_training_plot(history, args.figure_output)
    print(f"Best checkpoint: epoch={best_epoch}, val_auc={best_auc:.4f}, path={args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
