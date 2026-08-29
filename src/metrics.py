"""Binary-classification metrics and model evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader


def binary_classification_metrics(
    labels: list[float] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate thresholded and threshold-independent binary metrics."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    y_true = np.asarray(labels, dtype=np.int64)
    y_probability = np.asarray(probabilities, dtype=np.float64)
    if y_true.size == 0 or y_true.size != y_probability.size:
        raise ValueError("labels and probabilities must have the same non-zero length")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("binary metrics require both labels 0 and 1")
    if np.any((y_probability < 0.0) | (y_probability > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")

    y_pred = (y_probability >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "average_precision": float(average_precision_score(y_true, y_probability)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[dict[str, Any], list[float], list[float], list[str]]:
    """Evaluate a model and return metrics plus per-image outputs."""

    model.eval()
    labels: list[float] = []
    probabilities: list[float] = []
    image_paths: list[str] = []
    total_loss = 0.0
    total_examples = 0

    with torch.inference_mode():
        for images, batch_labels, batch_paths in loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(images)
            loss = criterion(logits, batch_labels)

            batch_size = batch_labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            labels.extend(batch_labels.detach().cpu().tolist())
            probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
            image_paths.extend(batch_paths)

    metrics = binary_classification_metrics(labels, probabilities, threshold=threshold)
    metrics["loss"] = total_loss / total_examples
    metrics["samples"] = total_examples
    return metrics, labels, probabilities, image_paths
