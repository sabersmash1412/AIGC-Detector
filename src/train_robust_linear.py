"""Train the controlled Section 3 E2 or E3 robust linear head."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.extract_clip_features import atomic_json_write
from src.device import choose_device
from src.image_transforms import DEFAULT_ROBUSTNESS_CONDITIONS
from src.linear_probe import load_linear_probe_checkpoint
from src.robust_linear_training import (
    RobustTrainingResult,
    load_paired_feature_set,
    save_robust_linear_checkpoint,
    sha256_file,
    train_paired_linear_head,
)


DEFAULT_PROTOCOL = Path("configs/section3_experiment.json")
DEFAULT_INITIAL_CHECKPOINT = Path("checkpoints/clip_linear_probe.npz")
EXPERIMENT_SPECS = {
    "e2": {
        "display_name": "E2",
        "artifact_name": "E2_supervised_clean_plus_transformed",
        "consistency_weight": 0.0,
        "checkpoint": Path("checkpoints/clip_linear_e2_supervised.npz"),
        "report": Path("reports/section3_e2_training.json"),
        "figure": Path("reports/figures/section3_e2_validation_curve.png"),
        "purpose": "Measure supervised transformation training before adding consistency loss.",
    },
    "e3": {
        "display_name": "E3",
        "artifact_name": "E3_supervised_plus_consistency",
        "consistency_weight": 1.0,
        "checkpoint": Path("checkpoints/clip_linear_e3_consistency.npz"),
        "report": Path("reports/section3_e3_training.json"),
        "figure": Path("reports/figures/section3_e3_validation_curve.png"),
        "purpose": "Isolate the effect of prediction consistency beyond E2 supervision.",
    },
}


def _load_protocol(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Section 3 protocol not found: {path}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected_conditions = list(DEFAULT_ROBUSTNESS_CONDITIONS)
    if protocol.get("representative_conditions") != expected_conditions:
        raise ValueError("Protocol conditions do not match the transform registry")
    if protocol.get("model_selection", {}).get("data") != "validation only":
        raise ValueError("Section 3 model selection must remain validation-only")
    return protocol


def _configure_matplotlib() -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aigc-detector-matplotlib")
    )


def _write_training_figure(
    result: RobustTrainingResult,
    path: Path,
    *,
    display_name: str,
    consistency_weight: float,
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in result.history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if consistency_weight > 0.0:
        axes[0].plot(
            epochs,
            [row["total_loss"] for row in result.history],
            marker="o",
            markersize=3,
            label="Total loss",
        )
        axes[0].plot(
            epochs,
            [row["supervised_loss"] for row in result.history],
            marker="o",
            markersize=3,
            label="Supervised component",
        )
        axes[0].plot(
            epochs,
            [row["consistency_loss_diagnostic"] for row in result.history],
            marker="o",
            markersize=3,
            label="Consistency component",
        )
        axes[0].legend()
    else:
        axes[0].plot(
            epochs,
            [row["supervised_loss"] for row in result.history],
            marker="o",
            markersize=3,
        )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Loss",
        title=f"{display_name} training loss",
    )
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        epochs,
        [row["validation"]["selection_mean_roc_auc"] for row in result.history],
        marker="o",
        markersize=3,
        label="Mean validation ROC-AUC",
    )
    axes[1].plot(
        epochs,
        [row["validation"]["worst_condition_roc_auc"] for row in result.history],
        marker="o",
        markersize=3,
        label="Worst-condition ROC-AUC",
    )
    axes[1].axvline(
        result.best_epoch,
        color="black",
        linestyle="--",
        alpha=0.65,
        label=f"Selected epoch {result.best_epoch}",
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel="ROC-AUC",
        title=f"Validation-only {display_name} model selection",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train E2 supervision or E3 supervision plus consistency."
    )
    parser.add_argument("--experiment", choices=tuple(EXPERIMENT_SPECS), required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--initial-checkpoint", type=Path, default=DEFAULT_INITIAL_CHECKPOINT
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--debug-maximum-epochs",
        type=int,
        default=None,
        help="Debug only: override the locked epoch budget and mark the report non-production.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_spec = EXPERIMENT_SPECS[args.experiment]
    display_name = str(experiment_spec["display_name"])
    artifact_name = str(experiment_spec["artifact_name"])
    consistency_weight = float(experiment_spec["consistency_weight"])
    checkpoint_path = args.checkpoint or experiment_spec["checkpoint"]
    report_path = args.report or experiment_spec["report"]
    figure_path = args.figure or experiment_spec["figure"]
    if not all(isinstance(path, Path) for path in (checkpoint_path, report_path, figure_path)):
        raise TypeError("Experiment artifact paths must be pathlib.Path values")
    protocol = _load_protocol(args.protocol)
    paired_config = protocol["paired_training"]
    seed = int(protocol["random_seed"])
    conditions = tuple(protocol["representative_conditions"][1:])
    clean_feature_dir = Path(protocol["data"]["clean_feature_directory"])
    transformed_feature_dir = Path(
        protocol["data"]["transformed_feature_directory"]
    )
    maximum_epochs = int(paired_config["maximum_epochs"])
    protocol_consistency_weight = float(
        protocol["experiments"]["E3"]["consistency_weight"]
    )
    if args.experiment == "e3" and consistency_weight != protocol_consistency_weight:
        raise ValueError("E3 consistency weight does not match the locked protocol")
    production_run = args.debug_maximum_epochs is None
    if args.debug_maximum_epochs is not None:
        if args.debug_maximum_epochs <= 0:
            raise ValueError("debug-maximum-epochs must be positive")
        maximum_epochs = args.debug_maximum_epochs

    print(f"Loading and validating {display_name} paired feature caches", flush=True)
    train = load_paired_feature_set(
        split="train",
        clean_cache_path=clean_feature_dir / "train.npz",
        transformed_feature_dir=transformed_feature_dir,
        conditions=conditions,
        seed=seed,
    )
    validation = load_paired_feature_set(
        split="val",
        clean_cache_path=clean_feature_dir / "val.npz",
        transformed_feature_dir=transformed_feature_dir,
        conditions=conditions,
        seed=seed,
    )
    initialization = load_linear_probe_checkpoint(args.initial_checkpoint)
    device = choose_device(args.device)
    print(
        f"{display_name} training: device={device.type}, train_images={train.samples}, "
        f"paired_examples={train.pairs}, val_images={validation.samples}, "
        f"conditions={len(conditions)}, trainable_parameters=513, "
        f"consistency_weight={consistency_weight:g}",
        flush=True,
    )

    started = time.perf_counter()
    result = train_paired_linear_head(
        train=train,
        validation=validation,
        initialization=initialization,
        device=device,
        seed=seed,
        batch_size=int(paired_config["batch_size"]),
        maximum_epochs=maximum_epochs,
        learning_rate=float(paired_config["learning_rate"]),
        weight_decay=float(paired_config["weight_decay"]),
        early_stopping_patience=int(paired_config["early_stopping_patience"]),
        consistency_weight=consistency_weight,
    )
    elapsed_seconds = time.perf_counter() - started
    save_robust_linear_checkpoint(
        checkpoint_path,
        result=result,
        initialization=initialization,
        experiment=artifact_name,
        protocol_sha256=sha256_file(args.protocol),
        train=train,
        validation=validation,
        seed=seed,
        consistency_weight=consistency_weight,
    )
    _write_training_figure(
        result,
        figure_path,
        display_name=display_name,
        consistency_weight=consistency_weight,
    )

    if consistency_weight > 0.0:
        loss_description = (
            "0.5 * (BCE clean + BCE transformed) + "
            f"{consistency_weight:g} * MSE(clean probability, transformed probability)"
        )
    else:
        loss_description = "0.5 * (BCE clean + BCE transformed)"

    report = {
        "experiment": artifact_name,
        "production_protocol_run": production_run,
        "purpose": experiment_spec["purpose"],
        "protocol": {
            "path": args.protocol.as_posix(),
            "sha256": sha256_file(args.protocol),
            "protocol_version": protocol["protocol_version"],
        },
        "data": {
            "train_clean_images": train.samples,
            "train_pairs_per_epoch": train.pairs,
            "validation_clean_images": validation.samples,
            "validation_conditions": ["clean", *conditions],
            "transformed_train_cache_sha256": dict(
                zip(conditions, train.transformed_cache_sha256, strict=True)
            ),
            "transformed_validation_cache_sha256": dict(
                zip(conditions, validation.transformed_cache_sha256, strict=True)
            ),
            "test_data_loaded": False,
            "organiser_validation_subset_used": False,
        },
        "model": {
            "feature_encoder": "ViT-B-32-quickgelu/openai",
            "feature_encoder_frozen": True,
            "linear_head_trainable_parameters": 513,
            "initial_checkpoint": args.initial_checkpoint.as_posix(),
            "initial_checkpoint_sha256": sha256_file(args.initial_checkpoint),
        },
        "training": {
            "seed": seed,
            "device": device.type,
            "loss": loss_description,
            "consistency_weight": consistency_weight,
            "optimizer": paired_config["optimizer"],
            "learning_rate": paired_config["learning_rate"],
            "weight_decay": paired_config["weight_decay"],
            "weight_decay_applies_to": paired_config["weight_decay_applies_to"],
            "batch_size": paired_config["batch_size"],
            "maximum_epochs": maximum_epochs,
            "early_stopping_patience": paired_config["early_stopping_patience"],
            "epochs_completed": result.epochs_completed,
            "stopped_early": result.stopped_early,
            "elapsed_seconds": elapsed_seconds,
        },
        "model_selection": {
            "data": "validation only",
            "metric": "mean ROC-AUC across clean and six transformed conditions",
            "best_epoch": result.best_epoch,
            "initial_e1_validation": result.initial_validation,
            f"selected_{args.experiment}_validation": result.best_validation,
        },
        "history": result.history,
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "sha256": sha256_file(checkpoint_path),
            "safe_object_free_npz": True,
            "temporary_threshold": 0.5,
        },
        "artifacts": {"training_figure": figure_path.as_posix()},
        "interpretation": (
            "No test data was loaded. Threshold 0.5 remains temporary until the "
            "validation-only threshold selection in Section 3E."
        ),
    }
    atomic_json_write(report_path, report)
    print(
        f"PASS {display_name}: best_epoch={result.best_epoch}, "
        f"val_mean_auc={result.best_validation['selection_mean_roc_auc']:.6f}, "
        f"val_worst_auc={result.best_validation['worst_condition_roc_auc']:.6f}",
        flush=True,
    )
    print(
        f"Checkpoint={checkpoint_path}; report={report_path}; figure={figure_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Robust linear training failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
