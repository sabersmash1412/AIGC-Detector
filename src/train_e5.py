"""Train all frozen E5 candidates and enforce risk-controlled acceptance gates."""

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
from src.device import choose_device
from src.e5_protocol import validate_e5_protocol
from src.e5_training import (
    E5TrainingOutcome,
    combined_cache_digest,
    save_e5_checkpoint,
    train_e5_candidates,
)
from src.linear_probe import load_linear_probe_checkpoint
from src.robust_linear_training import load_paired_feature_set, sha256_file


DEFAULT_PROTOCOL = Path("configs/e5_source_matched_adaptation.json")
DEFAULT_CHECKPOINT = Path("checkpoints/clip_linear_e5_source_matched.npz")
DEFAULT_REPORT = Path("reports/e5_training.json")
DEFAULT_FIGURE = Path("reports/figures/e5_validation_training.png")


def _configure_matplotlib() -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aigc-detector-matplotlib")
    )


def _write_figure(outcome: E5TrainingOutcome, path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = ("#64748b", "#2563eb", "#7c3aed", "#dc2626")
    for candidate, color in zip(outcome.candidates, colors, strict=True):
        epochs = [row["epoch"] for row in candidate.history]
        label = f"anchor={candidate.anchor_weight:g}"
        axes[0, 0].plot(
            epochs,
            [row["total_loss"] for row in candidate.history],
            label=label,
            color=color,
        )
        axes[0, 1].plot(
            epochs,
            [row["validation"]["worst_coverage"] for row in candidate.history],
            label=label,
            color=color,
        )
        axes[1, 0].plot(
            epochs,
            [row["validation"]["worst_real_error_upper"] for row in candidate.history],
            label=f"{label} real→AI",
            color=color,
        )
        axes[1, 0].plot(
            epochs,
            [row["validation"]["worst_ai_error_upper"] for row in candidate.history],
            linestyle="--",
            label=f"{label} AI→real",
            color=color,
        )
        axes[1, 1].plot(
            epochs,
            [
                row["validation"]["binary_worst_balanced_accuracy"]
                for row in candidate.history
            ],
            label=label,
            color=color,
        )
    axes[0, 0].set(title="Training objective", xlabel="Epoch", ylabel="Loss")
    axes[0, 1].set(
        title="Worst source-condition decisive coverage",
        xlabel="Epoch",
        ylabel="Coverage",
    )
    axes[0, 1].axhline(0.25, color="black", linestyle=":", label="25% minimum")
    axes[1, 0].set(
        title="Worst Wilson upper confident-error bounds",
        xlabel="Epoch",
        ylabel="Upper bound",
    )
    axes[1, 0].axhline(0.05, color="#991b1b", linestyle=":")
    axes[1, 0].axhline(0.10, color="#92400e", linestyle=":")
    axes[1, 1].set(
        title="Binary benchmark (not deployment rule)",
        xlabel="Epoch",
        ylabel="Worst balanced accuracy",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle(
        f"E5 validation-only candidate selection — accepted={outcome.accepted}",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _load_dataset(
    *,
    split: str,
    clean_dir: Path,
    transformed_dir: Path,
    conditions: tuple[str, ...],
    require_both_labels: bool,
) -> Any:
    return load_paired_feature_set(
        split=split,
        clean_cache_path=clean_dir / f"{split}.npz",
        transformed_feature_dir=transformed_dir,
        conditions=conditions,
        seed=42,
        require_both_labels=require_both_labels,
    )


def _candidate_report(candidate: Any) -> dict[str, Any]:
    return {
        "anchor_weight": candidate.anchor_weight,
        "feasible_candidate_found": candidate.best_validation is not None,
        "best_epoch": candidate.best_epoch,
        "epochs_completed": candidate.epochs_completed,
        "stopped_early": candidate.stopped_early,
        "best_validation": candidate.best_validation,
        "history": candidate.history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train all frozen E5 source-matched candidates."
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
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_e5_protocol(protocol)
    training = protocol["training"]
    selection = protocol["validation_and_decision_selection"]
    errors = selection["confident_error_constraints"]
    coverage = selection["anti_trivial_abstention_constraints"]
    maximum_epochs = int(training["maximum_epochs"])
    production_run = args.debug_maximum_epochs is None
    if args.debug_maximum_epochs is not None:
        if args.debug_maximum_epochs <= 0:
            raise ValueError("debug-maximum-epochs must be positive")
        if (
            args.checkpoint == DEFAULT_CHECKPOINT
            or args.report == DEFAULT_REPORT
            or args.figure == DEFAULT_FIGURE
        ):
            raise ValueError("Debug E5 runs must use non-production checkpoint/report/figure paths")
        maximum_epochs = args.debug_maximum_epochs

    conditions = tuple(protocol["representative_conditions"][1:])
    data = protocol["development_data"]
    cifake_config = data["cifake"]
    real_config = data["sid_set_train_real"]
    flux_config = data["sid_set_train_flux"]
    print("Loading and validating E5 CIFAKE, SID-real, and SID-FLUX caches", flush=True)
    cifake_train = _load_dataset(
        split="train",
        clean_dir=Path(cifake_config["clean_feature_directory"]),
        transformed_dir=Path(cifake_config["transformed_feature_directory"]),
        conditions=conditions,
        require_both_labels=True,
    )
    cifake_validation = _load_dataset(
        split="val",
        clean_dir=Path(cifake_config["clean_feature_directory"]),
        transformed_dir=Path(cifake_config["transformed_feature_directory"]),
        conditions=conditions,
        require_both_labels=True,
    )
    real_train = _load_dataset(
        split="train",
        clean_dir=Path(real_config["clean_feature_directory"]),
        transformed_dir=Path(real_config["transformed_feature_directory"]),
        conditions=conditions,
        require_both_labels=False,
    )
    real_validation = _load_dataset(
        split="val",
        clean_dir=Path(real_config["clean_feature_directory"]),
        transformed_dir=Path(real_config["transformed_feature_directory"]),
        conditions=conditions,
        require_both_labels=False,
    )
    flux_train = _load_dataset(
        split="train",
        clean_dir=Path(flux_config["clean_feature_directory"]),
        transformed_dir=Path(flux_config["transformed_feature_directory"]),
        conditions=conditions,
        require_both_labels=False,
    )
    flux_validation = _load_dataset(
        split="val",
        clean_dir=Path(flux_config["clean_feature_directory"]),
        transformed_dir=Path(flux_config["transformed_feature_directory"]),
        conditions=conditions,
        require_both_labels=False,
    )
    expected_counts = (
        (cifake_train.samples, int(cifake_config["train_samples"]), "CIFAKE train"),
        (
            cifake_validation.samples,
            int(cifake_config["validation_samples"]),
            "CIFAKE validation",
        ),
        (real_train.samples, int(real_config["train_samples"]), "SID real train"),
        (
            real_validation.samples,
            int(real_config["validation_samples"]),
            "SID real validation",
        ),
        (
            flux_train.samples,
            int(protocol["sid_flux_sampling"]["train_count"]),
            "SID FLUX train",
        ),
        (
            flux_validation.samples,
            int(protocol["sid_flux_sampling"]["validation_count"]),
            "SID FLUX validation",
        ),
    )
    for observed, expected, name in expected_counts:
        if observed != expected:
            raise ValueError(f"E5 {name} count changed: {observed} != {expected}")
    if set(np.unique(real_train.clean.labels).tolist()) != {0} or set(
        np.unique(real_validation.clean.labels).tolist()
    ) != {0}:
        raise ValueError("E5 SID-real caches changed labels")
    if set(np.unique(flux_train.clean.labels).tolist()) != {1} or set(
        np.unique(flux_validation.clean.labels).tolist()
    ) != {1}:
        raise ValueError("E5 SID-FLUX caches changed labels")

    initialization_path = Path(protocol["initialization"]["checkpoint"])
    if sha256_file(initialization_path) != protocol["initialization"]["checkpoint_sha256"]:
        raise ValueError("Frozen E3 initialization checkpoint changed")
    initialization = load_linear_probe_checkpoint(initialization_path)
    threshold_values = np.arange(
        float(selection["score_grid_minimum"]),
        float(selection["score_grid_maximum"]) + 1e-12,
        float(selection["score_grid_step"]),
    )
    device = choose_device(args.device)
    anchor_weights = tuple(float(value) for value in training["anchor_weights"])
    print(
        "E5 training: "
        f"device={device.type}, CIFAKE={cifake_train.samples}, "
        f"SID-real={real_train.samples}, SID-FLUX={flux_train.samples}, "
        f"conditions={len(conditions)}, anchors={anchor_weights}, "
        "epoch_groups=15000/15000/15000/15000, trainable_parameters=513",
        flush=True,
    )
    started = time.perf_counter()
    outcome = train_e5_candidates(
        cifake_train=cifake_train,
        sid_real_train=real_train,
        sid_flux_train=flux_train,
        cifake_validation=cifake_validation,
        sid_real_validation=real_validation,
        sid_flux_validation=flux_validation,
        initialization=initialization,
        device=device,
        anchor_weights=anchor_weights,
        examples_per_epoch=int(training["examples_per_epoch"]),
        seed=int(training["random_seed"]),
        batch_size=int(training["batch_size"]),
        maximum_epochs=maximum_epochs,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        consistency_weight=float(training["consistency_weight"]),
        thresholds=threshold_values,
        confidence_level=float(selection["confidence_level"]),
        maximum_real_called_ai_upper=float(
            errors["maximum_wilson_upper_real_called_ai"]
        ),
        maximum_ai_called_real_upper=float(
            errors["maximum_wilson_upper_ai_called_real"]
        ),
        minimum_clean_coverage=float(
            coverage["minimum_clean_decisive_coverage_per_group"]
        ),
        minimum_worst_coverage=float(
            coverage["minimum_worst_source_condition_decisive_coverage"]
        ),
        minimum_mean_coverage=float(
            coverage["minimum_mean_source_condition_decisive_coverage"]
        ),
    )
    elapsed = time.perf_counter() - started
    train_digest = combined_cache_digest(
        {"cifake": cifake_train, "sid_real": real_train, "sid_flux": flux_train},
        "train",
    )
    validation_digest = combined_cache_digest(
        {
            "cifake": cifake_validation,
            "sid_real": real_validation,
            "sid_flux": flux_validation,
        },
        "validation",
    )
    checkpoint_record: dict[str, Any] | None = None
    if outcome.accepted:
        save_e5_checkpoint(
            args.checkpoint,
            outcome=outcome,
            initialization=initialization,
            protocol_sha256=sha256_file(args.protocol),
            initial_checkpoint_sha256=sha256_file(initialization_path),
            train_cache_sha256=train_digest,
            validation_cache_sha256=validation_digest,
            seed=int(training["random_seed"]),
            consistency_weight=float(training["consistency_weight"]),
        )
        checkpoint_record = {
            "path": args.checkpoint.as_posix(),
            "sha256": sha256_file(args.checkpoint),
            "safe_object_free_npz": True,
        }
    _write_figure(outcome, args.figure)
    selected = outcome.selected
    report = {
        "experiment": protocol["experiment_name"],
        "production_protocol_run": production_run,
        "training_completed": True,
        "accepted_for_fresh_external_evaluation": bool(
            production_run and outcome.accepted
        ),
        "protocol": {
            "path": args.protocol.as_posix(),
            "sha256": sha256_file(args.protocol),
            "version": protocol["protocol_version"],
        },
        "development_data": {
            "cifake_train": cifake_train.samples,
            "cifake_validation": cifake_validation.samples,
            "sid_real_train": real_train.samples,
            "sid_real_validation": real_validation.samples,
            "sid_flux_train": flux_train.samples,
            "sid_flux_validation": flux_validation.samples,
            "conditions": ["clean", *conditions],
            "train_cache_digest": train_digest,
            "validation_cache_digest": validation_digest,
            "test_images_or_features_loaded": False,
            "prior_audit_images_or_features_loaded": False,
            "organiser_validation_subset_used": False,
        },
        "training": {
            "device": device.type,
            "seed": int(training["random_seed"]),
            "anchor_weights": list(anchor_weights),
            "all_anchor_candidates_completed": len(outcome.candidates)
            == len(anchor_weights),
            "examples_per_epoch": int(training["examples_per_epoch"]),
            "batch_size": int(training["batch_size"]),
            "maximum_epochs": maximum_epochs,
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "consistency_weight": float(training["consistency_weight"]),
            "elapsed_seconds": elapsed,
        },
        "initial_e3_validation": outcome.initial_validation,
        "candidates": [_candidate_report(candidate) for candidate in outcome.candidates],
        "selection": {
            "outcome": "accepted" if outcome.accepted else "rejected",
            "no_feasible_candidate_rule_applied": not outcome.accepted,
            "selected_anchor_weight": None if selected is None else selected.anchor_weight,
            "selected_epoch": None if selected is None else selected.best_epoch,
            "selected_validation": None
            if selected is None
            else selected.best_validation,
            "checkpoint_written": checkpoint_record is not None,
        },
        "checkpoint": checkpoint_record,
        "artifacts": {
            "figure": args.figure.as_posix(),
            "figure_sha256": sha256_file(args.figure),
        },
        "guardrails": {
            "E3_remains_original_primary_binary_baseline": True,
            "E4_remains_documented_failed_replacement": True,
            "risk_controlled_decision_is_primary": True,
            "binary_threshold_is_benchmark_only": True,
            "fresh_external_test_required_before_success_claim": True,
        },
    }
    atomic_json_write(args.report, report)
    if outcome.accepted and selected is not None and selected.best_validation is not None:
        risk = selected.best_validation["risk_controlled"]
        print(
            "ACCEPT E5 validation: "
            f"anchor={selected.anchor_weight:g}, epoch={selected.best_epoch}, "
            f"thresholds={risk['real_threshold']:.3f}/{risk['ai_threshold']:.3f}, "
            f"errors={risk['worst_real_called_ai_wilson_upper']:.3f}/"
            f"{risk['worst_ai_called_real_wilson_upper']:.3f}, "
            f"coverage={risk['worst_source_condition_decisive_coverage']:.3f}/"
            f"{risk['mean_source_condition_decisive_coverage']:.3f}",
            flush=True,
        )
        print(f"Checkpoint={args.checkpoint}", flush=True)
    else:
        print(
            "REJECT E5 validation: no epoch/anchor/threshold pair satisfied every "
            "frozen error and coverage constraint; no checkpoint written",
            flush=True,
        )
    print(f"Report={args.report}; figure={args.figure}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"E5 training failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
