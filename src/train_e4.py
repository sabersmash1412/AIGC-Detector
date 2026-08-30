"""Train the frozen-protocol E4 source-balanced linear detector."""

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

from scripts.extract_clip_features import atomic_json_write
from scripts.prepare_e4_sid_real import validate_protocol
from src.device import choose_device
from src.e4_training import (
    E4TrainingResult,
    combined_cache_digest,
    save_e4_checkpoint,
    train_e4_linear_head,
)
from src.linear_probe import load_linear_probe_checkpoint
from src.robust_linear_training import load_paired_feature_set, sha256_file


DEFAULT_PROTOCOL = Path("configs/e4_domain_adaptation.json")
DEFAULT_CHECKPOINT = Path("checkpoints/clip_linear_e4_domain_adapted.npz")
DEFAULT_REPORT = Path("reports/e4_training.json")
DEFAULT_FIGURE = Path("reports/figures/e4_validation_training_curve.png")


def _configure_matplotlib() -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aigc-detector-matplotlib")
    )


def _write_figure(result: E4TrainingResult, path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in result.history]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(epochs, [row["total_loss"] for row in result.history], marker="o")
    axes[0].set(title="Source-balanced E4 loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(
        epochs,
        [row["validation"]["mean_cifake_balanced_accuracy"] for row in result.history],
        marker="o",
        label="Mean CIFAKE BA",
    )
    axes[1].plot(
        epochs,
        [row["validation"]["worst_cifake_balanced_accuracy"] for row in result.history],
        marker="o",
        label="Worst CIFAKE BA",
    )
    axes[1].set(title="Validation utility", xlabel="Epoch", ylabel="Balanced accuracy")
    axes[1].legend()
    axes[2].plot(
        epochs,
        [row["validation"]["worst_sid_real_fpr"] for row in result.history],
        marker="o",
        label="Worst SID-real FPR",
    )
    axes[2].axhline(0.05, color="#dc2626", linestyle="--", label="5% constraint")
    axes[2].set(title="Validation safety constraint", xlabel="Epoch", ylabel="FPR")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("E4 validation-only model and threshold selection", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train E4 with balanced CIFAKE-fake/CIFAKE-real/SID-real groups."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--debug-maximum-epochs", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = _load_protocol(args.protocol)
    training = protocol["training"]
    selection = protocol["validation_and_threshold_selection"]
    cifake_config = protocol["development_data"]["cifake"]
    sid_config = protocol["development_data"]["sid_set_real"]
    conditions = tuple(protocol["representative_conditions"][1:])
    transform_seed = 42
    maximum_epochs = int(training["maximum_epochs"])
    production_run = args.debug_maximum_epochs is None
    if args.debug_maximum_epochs is not None:
        if args.debug_maximum_epochs <= 0:
            raise ValueError("debug-maximum-epochs must be positive")
        maximum_epochs = args.debug_maximum_epochs

    print("Loading and validating E4 CIFAKE and SID-real paired caches", flush=True)
    cifake_train = load_paired_feature_set(
        split="train",
        clean_cache_path=Path(cifake_config["clean_feature_directory"]) / "train.npz",
        transformed_feature_dir=Path(cifake_config["transformed_feature_directory"]),
        conditions=conditions,
        seed=transform_seed,
    )
    cifake_validation = load_paired_feature_set(
        split="val",
        clean_cache_path=Path(cifake_config["clean_feature_directory"]) / "val.npz",
        transformed_feature_dir=Path(cifake_config["transformed_feature_directory"]),
        conditions=conditions,
        seed=transform_seed,
    )
    sid_train = load_paired_feature_set(
        split="train",
        clean_cache_path=Path(sid_config["clean_feature_directory"]) / "train.npz",
        transformed_feature_dir=Path(sid_config["transformed_feature_directory"]),
        conditions=conditions,
        seed=transform_seed,
        require_both_labels=False,
    )
    sid_validation = load_paired_feature_set(
        split="val",
        clean_cache_path=Path(sid_config["clean_feature_directory"]) / "val.npz",
        transformed_feature_dir=Path(sid_config["transformed_feature_directory"]),
        conditions=conditions,
        seed=transform_seed,
        require_both_labels=False,
    )
    if cifake_train.samples != int(cifake_config["train_samples"]):
        raise ValueError("E4 CIFAKE training sample count changed")
    if cifake_validation.samples != int(cifake_config["validation_samples"]):
        raise ValueError("E4 CIFAKE validation sample count changed")
    if sid_train.samples != int(protocol["sid_real_sampling"]["train_count"]):
        raise ValueError("E4 SID training sample count changed")
    if sid_validation.samples != int(protocol["sid_real_sampling"]["validation_count"]):
        raise ValueError("E4 SID validation sample count changed")

    initialization_path = Path(protocol["initialization"]["checkpoint"])
    if sha256_file(initialization_path) != protocol["initialization"]["checkpoint_sha256"]:
        raise ValueError("Frozen E3 initialization checkpoint changed")
    initialization = load_linear_probe_checkpoint(initialization_path)
    threshold_values = np.arange(
        float(selection["candidate_threshold_minimum"]),
        float(selection["candidate_threshold_maximum"]) + 1e-12,
        float(selection["candidate_threshold_step"]),
    )
    device = choose_device(args.device)
    print(
        "E4 training: "
        f"device={device.type}, CIFAKE={cifake_train.samples}, "
        f"SID-real={sid_train.samples}, conditions={len(conditions)}, "
        "epoch_groups=30000/15000/15000, trainable_parameters=513",
        flush=True,
    )
    started = time.perf_counter()
    result = train_e4_linear_head(
        cifake_train=cifake_train,
        sid_train_real=sid_train,
        cifake_validation=cifake_validation,
        sid_validation_real=sid_validation,
        initialization=initialization,
        device=device,
        seed=int(training["random_seed"]),
        batch_size=int(training["batch_size"]),
        maximum_epochs=maximum_epochs,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        consistency_weight=float(training["consistency_weight"]),
        thresholds=threshold_values,
        sid_real_fpr_constraint=float(selection["sid_real_fpr_constraint"]),
    )
    elapsed_seconds = time.perf_counter() - started
    train_digest = combined_cache_digest(
        {"cifake": cifake_train, "sid_real": sid_train}, "train"
    )
    validation_digest = combined_cache_digest(
        {"cifake": cifake_validation, "sid_real": sid_validation}, "validation"
    )
    save_e4_checkpoint(
        args.checkpoint,
        result=result,
        initialization=initialization,
        protocol_sha256=sha256_file(args.protocol),
        initial_checkpoint_sha256=sha256_file(initialization_path),
        train_cache_sha256=train_digest,
        validation_cache_sha256=validation_digest,
        seed=int(training["random_seed"]),
        consistency_weight=float(training["consistency_weight"]),
        sid_real_fpr_constraint=float(selection["sid_real_fpr_constraint"]),
    )
    _write_figure(result, args.figure)
    report = {
        "experiment": protocol["experiment_name"],
        "production_protocol_run": production_run,
        "research_question": protocol["research_question"],
        "protocol": {
            "path": args.protocol.as_posix(),
            "sha256": sha256_file(args.protocol),
            "version": protocol["protocol_version"],
        },
        "data": {
            "cifake_train_images": cifake_train.samples,
            "sid_real_train_images": sid_train.samples,
            "cifake_validation_images": cifake_validation.samples,
            "sid_real_validation_images": sid_validation.samples,
            "conditions": ["clean", *conditions],
            "training_cache_digest": train_digest,
            "validation_cache_digest": validation_digest,
            "test_or_audit_data_loaded": False,
            "sid_set_synthetic_or_tampered_used": False,
            "organiser_validation_subset_used": False,
        },
        "model": {
            "encoder": "OpenCLIP ViT-B-32-quickgelu/openai",
            "encoder_frozen": True,
            "linear_head_trainable_parameters": 513,
            "initial_checkpoint": initialization_path.as_posix(),
            "initial_checkpoint_sha256": sha256_file(initialization_path),
        },
        "training": {
            "device": device.type,
            "seed": training["random_seed"],
            "source_group_weights": training["group_sampling_per_epoch"],
            "examples_per_epoch": 60000,
            "batch_size": training["batch_size"],
            "maximum_epochs": maximum_epochs,
            "epochs_completed": result.epochs_completed,
            "stopped_early": result.stopped_early,
            "learning_rate": training["learning_rate"],
            "weight_decay": training["weight_decay"],
            "consistency_weight": training["consistency_weight"],
            "elapsed_seconds": elapsed_seconds,
        },
        "model_and_threshold_selection": {
            "data": selection["data"],
            "sid_real_fpr_constraint": selection["sid_real_fpr_constraint"],
            "objective": selection["epoch_and_threshold_objective"],
            "best_epoch": result.best_epoch,
            "selected_threshold": result.threshold,
            "initial_e3_validation": result.initial_validation,
            "selected_e4_validation": result.best_validation,
        },
        "history": result.history,
        "checkpoint": {
            "path": args.checkpoint.as_posix(),
            "sha256": sha256_file(args.checkpoint),
            "safe_object_free_npz": True,
            "threshold": result.threshold,
        },
        "artifacts": {
            "training_figure": args.figure.as_posix(),
            "training_figure_sha256": sha256_file(args.figure),
        },
        "guardrails": {
            "E3_remains_original_primary_model": True,
            "E4_is_post_hoc_follow_up": True,
            "existing_section4_results_overwritten": False,
            "fresh_external_test_still_required": True,
        },
    }
    atomic_json_write(args.report, report)
    print(
        f"PASS E4: best_epoch={result.best_epoch}, threshold={result.threshold:.3f}, "
        f"sid_worst_fpr={result.best_validation['worst_sid_real_fpr']:.3f}, "
        f"cifake_mean_ba={result.best_validation['mean_cifake_balanced_accuracy']:.6f}",
        flush=True,
    )
    print(
        f"Checkpoint={args.checkpoint}; report={args.report}; figure={args.figure}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"E4 training failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
