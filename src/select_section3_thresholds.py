"""Select and freeze E1-E3 operating thresholds from validation data only."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.extract_clip_features import atomic_json_write
from src.image_transforms import DEFAULT_ROBUSTNESS_CONDITIONS
from src.linear_probe import load_linear_probe_checkpoint
from src.robust_linear_training import (
    load_paired_feature_set,
    sha256_file,
)
from src.threshold_selection import (
    ThresholdSelectionResult,
    select_balanced_accuracy_threshold,
    write_thresholded_checkpoint,
)


DEFAULT_PROTOCOL = Path("configs/section3_experiment.json")
DEFAULT_REPORT = Path("reports/section3_threshold_selection.json")
DEFAULT_FIGURE = Path("reports/figures/section3_threshold_selection.png")
MODEL_PATHS = {
    "E1": Path("checkpoints/clip_linear_probe.npz"),
    "E2": Path("checkpoints/clip_linear_e2_supervised.npz"),
    "E3": Path("checkpoints/clip_linear_e3_consistency.npz"),
}
OUTPUT_PATHS = {
    "E1": Path("checkpoints/clip_linear_e1_thresholded.npz"),
    "E2": Path("checkpoints/clip_linear_e2_thresholded.npz"),
    "E3": Path("checkpoints/clip_linear_e3_thresholded.npz"),
}


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["threshold_selection"]["data"] != "validation only":
        raise ValueError("Threshold selection must remain validation-only")
    if tuple(protocol["representative_conditions"]) != DEFAULT_ROBUSTNESS_CONDITIONS:
        raise ValueError("Protocol conditions do not match the transform registry")
    return protocol


def _candidate_thresholds(config: dict[str, Any]) -> np.ndarray:
    minimum = float(config["candidate_minimum"])
    maximum = float(config["candidate_maximum"])
    step = float(config["candidate_step"])
    if not 0.0 < minimum <= maximum < 1.0 or step <= 0.0:
        raise ValueError("Invalid threshold grid in protocol")
    count = int(round((maximum - minimum) / step)) + 1
    thresholds = minimum + np.arange(count, dtype=np.float64) * step
    if not np.isclose(thresholds[-1], maximum, rtol=0.0, atol=1e-12):
        raise ValueError("Threshold grid endpoints are not divisible by the step")
    return thresholds


def _probabilities_by_condition(dataset: Any, checkpoint: Any) -> dict[str, np.ndarray]:
    return {
        "clean": checkpoint.probabilities(dataset.clean.features),
        **{
            condition: checkpoint.probabilities(cache.features)
            for condition, cache in zip(
                dataset.conditions, dataset.transformed, strict=True
            )
        },
    }


def _configure_matplotlib() -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aigc-detector-matplotlib")
    )


def _write_figure(
    selections: dict[str, ThresholdSelectionResult], path: Path
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for model_name, selection in selections.items():
        line = axis.plot(
            selection.curve[:, 0],
            selection.curve[:, 1],
            label=f"{model_name} mean balanced accuracy",
        )[0]
        axis.axvline(
            selection.threshold,
            color=line.get_color(),
            linestyle="--",
            alpha=0.65,
            label=f"{model_name} selected {selection.threshold:.3f}",
        )
    axis.axvline(0.5, color="black", linestyle=":", alpha=0.7, label="Default 0.500")
    axis.set(
        xlabel="AI-generated probability threshold",
        ylabel="Mean validation balanced accuracy",
        title="Validation-only operating-threshold selection",
        xlim=(0.0, 1.0),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze E1-E3 thresholds using transformed validation data only."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = _load_protocol(args.protocol)
    seed = int(protocol["random_seed"])
    conditions = tuple(protocol["representative_conditions"][1:])
    clean_feature_dir = Path(protocol["data"]["clean_feature_directory"])
    transformed_feature_dir = Path(
        protocol["data"]["transformed_feature_directory"]
    )
    validation = load_paired_feature_set(
        split="val",
        clean_cache_path=clean_feature_dir / "val.npz",
        transformed_feature_dir=transformed_feature_dir,
        conditions=conditions,
        seed=seed,
    )
    thresholds = _candidate_thresholds(protocol["threshold_selection"])
    protocol_digest = sha256_file(args.protocol)
    selections: dict[str, ThresholdSelectionResult] = {}
    model_reports: dict[str, Any] = {}

    print(
        f"Threshold selection: validation_images={validation.samples}, "
        f"conditions={len(conditions) + 1}, candidates={len(thresholds)}",
        flush=True,
    )
    for model_name, checkpoint_path in MODEL_PATHS.items():
        checkpoint = load_linear_probe_checkpoint(checkpoint_path)
        probabilities = _probabilities_by_condition(validation, checkpoint)
        selection = select_balanced_accuracy_threshold(
            validation.clean.labels, probabilities, thresholds
        )
        destination_path = OUTPUT_PATHS[model_name]
        source_digest = sha256_file(checkpoint_path)
        write_thresholded_checkpoint(
            checkpoint_path,
            destination_path,
            threshold=selection.threshold,
            objective_score=selection.objective_score,
            protocol_sha256=protocol_digest,
            source_checkpoint_sha256=source_digest,
        )
        selections[model_name] = selection
        model_reports[model_name] = {
            "source_checkpoint": checkpoint_path.as_posix(),
            "source_checkpoint_sha256": source_digest,
            "thresholded_checkpoint": destination_path.as_posix(),
            "thresholded_checkpoint_sha256": sha256_file(destination_path),
            "selected_threshold": selection.threshold,
            "selection_objective_score": selection.objective_score,
            "selected_metrics": selection.selected_metrics,
            "baseline_0_5_metrics": selection.baseline_0_5_metrics,
            "mean_balanced_accuracy_delta_vs_0_5": (
                selection.selected_metrics["mean_balanced_accuracy"]
                - selection.baseline_0_5_metrics["mean_balanced_accuracy"]
            ),
        }
        print(
            f"PASS {model_name}: threshold={selection.threshold:.3f}, "
            f"mean_bal_acc={selection.objective_score:.6f}, "
            f"worst_bal_acc={selection.worst_condition_balanced_accuracy:.6f}",
            flush=True,
        )

    _write_figure(selections, args.figure)
    report = {
        "experiment": "section3_validation_only_threshold_selection",
        "purpose": "Freeze E1-E3 operating thresholds before transformed test evaluation.",
        "protocol": {
            "path": args.protocol.as_posix(),
            "sha256": protocol_digest,
            "protocol_version": protocol["protocol_version"],
        },
        "selection": {
            "split": "validation",
            "samples": validation.samples,
            "class_counts": {
                "real_0": int(np.sum(validation.clean.labels == 0)),
                "ai_generated_1": int(np.sum(validation.clean.labels == 1)),
            },
            "conditions": ["clean", *conditions],
            "candidate_minimum": float(thresholds[0]),
            "candidate_maximum": float(thresholds[-1]),
            "candidate_step": float(thresholds[1] - thresholds[0]),
            "candidate_count": len(thresholds),
            "objective": protocol["threshold_selection"]["objective"],
            "tie_break_1": protocol["threshold_selection"]["tie_break_1"],
            "tie_break_2": protocol["threshold_selection"]["tie_break_2"],
            "test_data_loaded": False,
            "organiser_validation_subset_used": False,
        },
        "models": model_reports,
        "artifacts": {"selection_figure": args.figure.as_posix()},
        "next_step_lock": (
            "Use these checkpoint thresholds unchanged for all Section 3 test results."
        ),
    }
    atomic_json_write(args.report, report)
    print(f"Updated report: {args.report}; figure={args.figure}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Threshold selection failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
