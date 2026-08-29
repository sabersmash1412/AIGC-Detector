"""Select, train, and evaluate a clean frozen-CLIP linear probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.linear_probe import (
    LINEAR_PROBE_NAME,
    LinearProbeCheckpoint,
    linear_probe_probabilities,
    load_feature_cache,
    load_linear_probe_checkpoint,
    save_linear_probe_checkpoint,
)
from src.metrics import binary_classification_metrics


DEFAULT_FEATURE_DIR = Path("data/features/clip_vit_b32_quickgelu_openai")
DEFAULT_C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


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


def metrics_with_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    metrics = binary_classification_metrics(labels, probabilities, threshold=0.5)
    metrics["log_loss"] = float(log_loss(labels, probabilities, labels=[0, 1]))
    metrics["samples"] = int(len(labels))
    return metrics


def _write_predictions(
    path: Path, image_paths: np.ndarray, labels: np.ndarray, probabilities: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "label", "pred"])
        writer.writeheader()
        writer.writerows(
            {
                "image_path": image_path,
                "label": int(label),
                "pred": round(float(probability), 8),
            }
            for image_path, label, probability in zip(
                image_paths.tolist(), labels.tolist(), probabilities.tolist(), strict=True
            )
        )
    temporary_path.replace(path)


def _configure_matplotlib() -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aigc-detector-matplotlib")
    )


def _write_selection_plot(candidates: list[dict[str, Any]], best_c: float, path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_values = [row["regularization_c"] for row in candidates]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.semilogx(
        c_values,
        [row["train_metrics"]["roc_auc"] for row in candidates],
        marker="o",
        label="Train ROC-AUC",
    )
    axis.semilogx(
        c_values,
        [row["validation_metrics"]["roc_auc"] for row in candidates],
        marker="o",
        label="Validation ROC-AUC",
    )
    axis.axvline(best_c, color="black", linestyle="--", alpha=0.65, label=f"Selected C={best_c:g}")
    axis.set(
        xlabel="C (inverse regularisation strength; logarithmic scale)",
        ylabel="ROC-AUC",
        title="Frozen CLIP linear-probe model selection",
        ylim=(0.5, 1.0),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    _configure_matplotlib()
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
        title="Frozen CLIP linear-probe confusion matrix",
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and evaluate a logistic-regression head on frozen CLIP features."
    )
    parser.add_argument("--train-cache", type=Path, default=DEFAULT_FEATURE_DIR / "train.npz")
    parser.add_argument("--val-cache", type=Path, default=DEFAULT_FEATURE_DIR / "val.npz")
    parser.add_argument("--test-cache", type=Path, default=DEFAULT_FEATURE_DIR / "test.npz")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/clip_linear_probe.npz")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/clip_linear_probe.json")
    )
    parser.add_argument(
        "--selection-figure",
        type=Path,
        default=Path("reports/figures/clip_linear_probe_selection.png"),
    )
    parser.add_argument(
        "--confusion-figure",
        type=Path,
        default=Path("reports/figures/clip_linear_probe_confusion_matrix.png"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("outputs/clip_linear_probe_test_predictions.csv"),
    )
    parser.add_argument("--c-values", type=float, nargs="+", default=list(DEFAULT_C_VALUES))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    c_values = sorted(set(args.c_values))
    if not c_values or any(value <= 0.0 for value in c_values):
        raise ValueError("Every C value must be positive")
    if args.max_iter <= 0:
        raise ValueError("max-iter must be positive")

    train = load_feature_cache(args.train_cache, "train")
    validation = load_feature_cache(args.val_cache, "val")
    test = load_feature_cache(args.test_cache, "test")
    cache_model_identities = {
        (cache.model_name, cache.pretrained) for cache in (train, validation, test)
    }
    if len(cache_model_identities) != 1:
        raise ValueError("Train, validation, and test caches use different CLIP models")

    print(
        f"Linear-probe search: train={len(train.labels)}, val={len(validation.labels)}, "
        f"features={train.features.shape[1]}, C={c_values}",
        flush=True,
    )
    candidates: list[dict[str, Any]] = []
    best_classifier: LogisticRegression | None = None
    best_validation_auc = float("-inf")
    best_c = 0.0

    for regularization_c in c_values:
        classifier = LogisticRegression(
            C=regularization_c,
            solver="lbfgs",
            max_iter=args.max_iter,
            random_state=args.seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(train.features, train.labels)

        coefficients = classifier.coef_[0]
        intercept = float(classifier.intercept_[0])
        train_probabilities = linear_probe_probabilities(
            train.features, coefficients, intercept
        )
        validation_probabilities = linear_probe_probabilities(
            validation.features, coefficients, intercept
        )
        train_metrics = metrics_with_log_loss(train.labels, train_probabilities)
        validation_metrics = metrics_with_log_loss(
            validation.labels, validation_probabilities
        )
        candidates.append(
            {
                "regularization_c": regularization_c,
                "iterations": int(classifier.n_iter_[0]),
                "coefficient_l2_norm": float(np.linalg.vector_norm(coefficients)),
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
            }
        )
        print(
            f"C={regularization_c:g}: train_auc={train_metrics['roc_auc']:.6f}, "
            f"val_auc={validation_metrics['roc_auc']:.6f}, "
            f"val_bal_acc={validation_metrics['balanced_accuracy']:.6f}",
            flush=True,
        )

        validation_auc = float(validation_metrics["roc_auc"])
        if validation_auc > best_validation_auc:
            best_validation_auc = validation_auc
            best_c = regularization_c
            best_classifier = classifier

    if best_classifier is None:
        raise RuntimeError("No linear-probe candidate was trained")

    selected_coefficients = best_classifier.coef_[0].astype(np.float64)
    selected_intercept = float(best_classifier.intercept_[0])
    selected_validation_probabilities = linear_probe_probabilities(
        validation.features, selected_coefficients, selected_intercept
    )
    selected_validation_metrics = metrics_with_log_loss(
        validation.labels, selected_validation_probabilities
    )

    checkpoint = LinearProbeCheckpoint(
        coefficients=selected_coefficients,
        intercept=selected_intercept,
        regularization_c=best_c,
        threshold=0.5,
        seed=args.seed,
        selected_validation_roc_auc=best_validation_auc,
        train_cache_sha256=sha256_file(args.train_cache),
        validation_cache_sha256=sha256_file(args.val_cache),
    )
    save_linear_probe_checkpoint(args.checkpoint, checkpoint)
    loaded_checkpoint = load_linear_probe_checkpoint(args.checkpoint)

    test_probabilities = loaded_checkpoint.probabilities(test.features)
    test_metrics = metrics_with_log_loss(test.labels, test_probabilities)
    _write_predictions(
        args.predictions_output, test.image_paths, test.labels, test_probabilities
    )
    _write_selection_plot(candidates, best_c, args.selection_figure)
    _write_confusion_matrix(test_metrics["confusion_matrix"], args.confusion_figure)

    report = {
        "experiment": "clean_frozen_clip_linear_probe",
        "purpose": "Clean-image linear baseline before robustness training.",
        "seed": args.seed,
        "classifier": {
            "name": LINEAR_PROBE_NAME,
            "library": "scikit-learn",
            "library_version": sklearn.__version__,
            "solver": "lbfgs",
            "penalty": "L2",
            "max_iter": args.max_iter,
            "trainable_parameters": int(selected_coefficients.size + 1),
            "selection_metric": "validation_roc_auc",
            "tie_break": "smallest C (strongest regularisation)",
            "selected_c": best_c,
            "temporary_decision_threshold": 0.5,
        },
        "feature_model": {
            "model_name": train.model_name,
            "pretrained": train.pretrained,
            "feature_dimension": train.features.shape[1],
            "l2_normalized": True,
            "frozen": True,
        },
        "data": {
            "train": {
                "cache_path": args.train_cache.as_posix(),
                "cache_sha256": sha256_file(args.train_cache),
                "samples": len(train.labels),
            },
            "validation": {
                "cache_path": args.val_cache.as_posix(),
                "cache_sha256": sha256_file(args.val_cache),
                "samples": len(validation.labels),
            },
            "test": {
                "cache_path": args.test_cache.as_posix(),
                "cache_sha256": sha256_file(args.test_cache),
                "samples": len(test.labels),
            },
        },
        "candidates": candidates,
        "selected_validation_metrics": selected_validation_metrics,
        "test_metrics": test_metrics,
        "checkpoint": {
            "path": args.checkpoint.as_posix(),
            "sha256": sha256_file(args.checkpoint),
            "safe_object_free_npz": True,
        },
        "artifacts": {
            "selection_figure": args.selection_figure.as_posix(),
            "confusion_figure": args.confusion_figure.as_posix(),
            "test_predictions": args.predictions_output.as_posix(),
        },
        "interpretation": (
            "Test metrics measure only clean CIFAKE performance. The threshold 0.5 is "
            "temporary; formal validation-based threshold selection occurs in Section 3."
        ),
    }
    atomic_json_write(args.report, report)
    print(
        f"Selected C={best_c:g} with val_auc={best_validation_auc:.6f}; "
        f"test_auc={test_metrics['roc_auc']:.6f}, "
        f"test_bal_acc={test_metrics['balanced_accuracy']:.6f}",
        flush=True,
    )
    print(f"Checkpoint={args.checkpoint}; report={args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
