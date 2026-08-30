"""One-pass frozen E1-E3 evaluation on identical Section 3 test views."""

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
from src.evaluate_initial_robustness import stability_metrics
from src.image_transforms import DEFAULT_ROBUSTNESS_CONDITIONS, TRANSFORM_SPECS
from src.linear_probe import load_linear_probe_checkpoint
from src.metrics import binary_classification_metrics
from src.robust_linear_training import (
    PairedFeatureSet,
    load_paired_feature_set,
    sha256_file,
)
from src.threshold_selection import THRESHOLD_SELECTION_KIND


DEFAULT_PROTOCOL = Path("configs/section3_experiment.json")
DEFAULT_THRESHOLD_REPORT = Path("reports/section3_threshold_selection.json")
DEFAULT_REPORT = Path("reports/section3_e1_e3_test_comparison.json")
DEFAULT_COMPARISON_FIGURE = Path(
    "reports/figures/section3_e1_e3_test_comparison.png"
)
DEFAULT_CONFUSION_FIGURE = Path(
    "reports/figures/section3_key_confusion_matrices.png"
)
DEFAULT_PROBABILITY_OUTPUT = Path("outputs/section3_test_probabilities.npz")

SUMMARY_METRIC_KEYS = (
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "brier_score",
)


def _scalar(archive: np.lib.npyio.NpzFile, key: str) -> object:
    if key not in archive.files:
        raise ValueError(f"Thresholded checkpoint is missing metadata {key!r}")
    value = archive[key]
    if value.ndim != 0:
        raise ValueError(f"Checkpoint metadata {key!r} must be scalar")
    return value.item()


def validate_frozen_threshold_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_threshold: float,
    expected_protocol_sha256: str,
    expected_source_sha256: str,
) -> Any:
    """Prove that a checkpoint came from the frozen validation-only selection."""

    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Thresholded checkpoint hash changed: {path}")
    checkpoint = load_linear_probe_checkpoint(path)
    with np.load(path, allow_pickle=False) as archive:
        kind = str(_scalar(archive, "threshold_selection_kind"))
        split = str(_scalar(archive, "threshold_selection_split"))
        protocol_sha256 = str(
            _scalar(archive, "threshold_selection_protocol_sha256")
        )
        source_sha256 = str(_scalar(archive, "threshold_source_checkpoint_sha256"))
    if kind != THRESHOLD_SELECTION_KIND or split != "validation":
        raise ValueError("Checkpoint threshold was not selected from validation data")
    if protocol_sha256 != expected_protocol_sha256:
        raise ValueError("Checkpoint uses a different Section 3 protocol")
    if source_sha256 != expected_source_sha256:
        raise ValueError("Checkpoint refers to a different trained source model")
    if not np.isclose(checkpoint.threshold, expected_threshold, rtol=0.0, atol=1e-12):
        raise ValueError("Checkpoint threshold does not match the frozen report")
    return checkpoint


def _probabilities_by_condition(
    dataset: PairedFeatureSet, checkpoint: Any
) -> dict[str, np.ndarray]:
    return {
        "clean": checkpoint.probabilities(dataset.clean.features),
        **{
            condition: checkpoint.probabilities(cache.features)
            for condition, cache in zip(
                dataset.conditions, dataset.transformed, strict=True
            )
        },
    }


def evaluate_model_conditions(
    labels: np.ndarray,
    probabilities_by_condition: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    """Evaluate one frozen model and summarise transformed robustness."""

    clean_probabilities = probabilities_by_condition["clean"]
    by_condition: dict[str, Any] = {}
    for condition, probabilities in probabilities_by_condition.items():
        metrics = binary_classification_metrics(labels, probabilities, threshold)
        metrics["samples"] = len(labels)
        if condition == "clean":
            stability = {
                "mean_absolute_probability_change": 0.0,
                "median_absolute_probability_change": 0.0,
                "maximum_absolute_probability_change": 0.0,
                "prediction_flip_rate": 0.0,
                "real_mean_probability_shift": 0.0,
                "ai_generated_mean_probability_shift": 0.0,
            }
        else:
            stability = stability_metrics(
                labels, clean_probabilities, probabilities, threshold
            )
        by_condition[condition] = {
            "display_name": TRANSFORM_SPECS[condition].display_name,
            "parameters": TRANSFORM_SPECS[condition].parameters,
            "metrics": metrics,
            "stability_vs_clean": stability,
        }

    transformed_names = [
        condition for condition in probabilities_by_condition if condition != "clean"
    ]
    transformed_summary = {
        f"mean_{metric}": float(
            np.mean(
                [by_condition[name]["metrics"][metric] for name in transformed_names]
            )
        )
        for metric in SUMMARY_METRIC_KEYS
    }
    transformed_summary.update(
        {
            "mean_prediction_flip_rate": float(
                np.mean(
                    [
                        by_condition[name]["stability_vs_clean"][
                            "prediction_flip_rate"
                        ]
                        for name in transformed_names
                    ]
                )
            ),
            "worst_balanced_accuracy_condition": min(
                transformed_names,
                key=lambda name: by_condition[name]["metrics"][
                    "balanced_accuracy"
                ],
            ),
            "worst_transformed_balanced_accuracy": float(
                min(
                    by_condition[name]["metrics"]["balanced_accuracy"]
                    for name in transformed_names
                )
            ),
            "worst_roc_auc_condition": min(
                transformed_names,
                key=lambda name: by_condition[name]["metrics"]["roc_auc"],
            ),
            "worst_transformed_roc_auc": float(
                min(
                    by_condition[name]["metrics"]["roc_auc"]
                    for name in transformed_names
                )
            ),
        }
    )
    all_condition_balanced_accuracies = [
        row["metrics"]["balanced_accuracy"] for row in by_condition.values()
    ]
    return {
        "threshold": threshold,
        "clean": by_condition["clean"],
        "transformed_summary": transformed_summary,
        "all_condition_mean_balanced_accuracy": float(
            np.mean(all_condition_balanced_accuracies)
        ),
        "by_condition": by_condition,
    }


def comparison_deltas(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Return candidate-minus-baseline test deltas for controlled comparisons."""

    return {
        "clean_roc_auc": (
            candidate["clean"]["metrics"]["roc_auc"]
            - baseline["clean"]["metrics"]["roc_auc"]
        ),
        "clean_balanced_accuracy": (
            candidate["clean"]["metrics"]["balanced_accuracy"]
            - baseline["clean"]["metrics"]["balanced_accuracy"]
        ),
        "mean_transformed_roc_auc": (
            candidate["transformed_summary"]["mean_roc_auc"]
            - baseline["transformed_summary"]["mean_roc_auc"]
        ),
        "mean_transformed_balanced_accuracy": (
            candidate["transformed_summary"]["mean_balanced_accuracy"]
            - baseline["transformed_summary"]["mean_balanced_accuracy"]
        ),
        "worst_transformed_balanced_accuracy": (
            candidate["transformed_summary"][
                "worst_transformed_balanced_accuracy"
            ]
            - baseline["transformed_summary"][
                "worst_transformed_balanced_accuracy"
            ]
        ),
        "mean_prediction_flip_rate": (
            candidate["transformed_summary"]["mean_prediction_flip_rate"]
            - baseline["transformed_summary"]["mean_prediction_flip_rate"]
        ),
        "by_condition": {
            condition: {
                "roc_auc": (
                    candidate["by_condition"][condition]["metrics"]["roc_auc"]
                    - baseline["by_condition"][condition]["metrics"]["roc_auc"]
                ),
                "balanced_accuracy": (
                    candidate["by_condition"][condition]["metrics"][
                        "balanced_accuracy"
                    ]
                    - baseline["by_condition"][condition]["metrics"][
                        "balanced_accuracy"
                    ]
                ),
                "prediction_flip_rate": (
                    candidate["by_condition"][condition]["stability_vs_clean"][
                        "prediction_flip_rate"
                    ]
                    - baseline["by_condition"][condition]["stability_vs_clean"][
                        "prediction_flip_rate"
                    ]
                ),
            }
            for condition in baseline["by_condition"]
        },
    }


def _atomic_probability_write(
    path: Path,
    dataset: PairedFeatureSet,
    probabilities: dict[str, dict[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "image_paths": dataset.clean.image_paths,
        "labels": dataset.clean.labels,
    }
    for model_name, model_probabilities in probabilities.items():
        for condition, values in model_probabilities.items():
            arrays[f"{model_name}_{condition}"] = np.asarray(values, dtype=np.float64)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        np.savez(temporary_path, **arrays)
        temporary_path.replace(path)


def _configure_matplotlib() -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aigc-detector-matplotlib")
    )


def _write_comparison_figure(
    model_results: dict[str, dict[str, Any]], path: Path
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(DEFAULT_ROBUSTNESS_CONDITIONS)
    display_names = [TRANSFORM_SPECS[name].display_name for name in conditions]
    positions = np.arange(len(conditions))
    width = 0.25
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for model_index, (model_name, result) in enumerate(model_results.items()):
        offset = (model_index - 1) * width
        axes[0].bar(
            positions + offset,
            [result["by_condition"][name]["metrics"]["roc_auc"] for name in conditions],
            width,
            label=model_name,
        )
        axes[1].bar(
            positions + offset,
            [
                result["by_condition"][name]["metrics"]["balanced_accuracy"]
                for name in conditions
            ],
            width,
            label=model_name,
        )

    transformed_conditions = conditions[1:]
    transformed_positions = np.arange(len(transformed_conditions))
    for model_index, (model_name, result) in enumerate(model_results.items()):
        offset = (model_index - 1) * width
        axes[2].bar(
            transformed_positions + offset,
            [
                result["by_condition"][name]["stability_vs_clean"][
                    "prediction_flip_rate"
                ]
                for name in transformed_conditions
            ],
            width,
            label=model_name,
        )

    for axis, title, ylabel in (
        (axes[0], "Threshold-independent ranking", "ROC-AUC"),
        (axes[1], "Frozen-threshold decisions", "Balanced accuracy"),
    ):
        axis.set(
            xticks=positions,
            xticklabels=display_names,
            title=title,
            ylabel=ylabel,
            ylim=(0.5, 1.0),
        )
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    axes[2].set(
        xticks=transformed_positions,
        xticklabels=[TRANSFORM_SPECS[name].display_name for name in transformed_conditions],
        title="Decision instability versus clean",
        ylabel="Prediction flip rate",
        ylim=(0.0, 0.5),
    )
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_confusion_figure(
    model_results: dict[str, dict[str, Any]], path: Path
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key_conditions = ("clean", "gaussian_blur_sigma1", "resize_0_5x")
    figure, axes = plt.subplots(3, 3, figsize=(11, 10))
    for row_index, (model_name, result) in enumerate(model_results.items()):
        for column_index, condition in enumerate(key_conditions):
            axis = axes[row_index, column_index]
            matrix = result["by_condition"][condition]["metrics"]["confusion_matrix"]
            axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1000)
            for row in range(2):
                for column in range(2):
                    axis.text(
                        column,
                        row,
                        str(matrix[row][column]),
                        ha="center",
                        va="center",
                    )
            axis.set(
                xticks=[0, 1],
                yticks=[0, 1],
                xticklabels=["Real", "AI"],
                yticklabels=["Real", "AI"],
                xlabel="Predicted",
                ylabel="True",
                title=f"{model_name} — {TRANSFORM_SPECS[condition].display_name}",
            )
    figure.suptitle("Section 3 key test confusion matrices", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen E1-E3 models once on identical transformed test caches."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--threshold-report", type=Path, default=DEFAULT_THRESHOLD_REPORT
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--comparison-figure", type=Path, default=DEFAULT_COMPARISON_FIGURE
    )
    parser.add_argument(
        "--confusion-figure", type=Path, default=DEFAULT_CONFUSION_FIGURE
    )
    parser.add_argument(
        "--probability-output", type=Path, default=DEFAULT_PROBABILITY_OUTPUT
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    threshold_report = json.loads(args.threshold_report.read_text(encoding="utf-8"))
    protocol_digest = sha256_file(args.protocol)
    if threshold_report["protocol"]["sha256"] != protocol_digest:
        raise ValueError("Threshold report and Section 3 protocol do not match")
    if threshold_report["selection"]["split"] != "validation":
        raise ValueError("Threshold report was not selected from validation data")

    conditions = tuple(protocol["representative_conditions"][1:])
    if ("clean", *conditions) != DEFAULT_ROBUSTNESS_CONDITIONS:
        raise ValueError("Protocol conditions do not match the transform registry")
    seed = int(protocol["random_seed"])
    clean_feature_dir = Path(protocol["data"]["clean_feature_directory"])
    transformed_feature_dir = Path(
        protocol["data"]["transformed_feature_directory"]
    )
    test = load_paired_feature_set(
        split="test",
        clean_cache_path=clean_feature_dir / "test.npz",
        transformed_feature_dir=transformed_feature_dir,
        conditions=conditions,
        seed=seed,
    )

    pre_test_primary_model = max(
        threshold_report["models"],
        key=lambda name: threshold_report["models"][name][
            "selection_objective_score"
        ],
    )
    print(
        f"Frozen test evaluation: samples={test.samples}, conditions={len(conditions) + 1}, "
        f"pre_test_primary_model={pre_test_primary_model}",
        flush=True,
    )
    model_results: dict[str, dict[str, Any]] = {}
    all_probabilities: dict[str, dict[str, np.ndarray]] = {}
    checkpoint_records: dict[str, Any] = {}
    for model_name, selection_record in threshold_report["models"].items():
        checkpoint_path = Path(selection_record["thresholded_checkpoint"])
        checkpoint = validate_frozen_threshold_checkpoint(
            checkpoint_path,
            expected_sha256=selection_record["thresholded_checkpoint_sha256"],
            expected_threshold=float(selection_record["selected_threshold"]),
            expected_protocol_sha256=protocol_digest,
            expected_source_sha256=selection_record["source_checkpoint_sha256"],
        )
        probabilities = _probabilities_by_condition(test, checkpoint)
        result = evaluate_model_conditions(
            test.clean.labels, probabilities, checkpoint.threshold
        )
        model_results[model_name] = result
        all_probabilities[model_name] = probabilities
        checkpoint_records[model_name] = {
            "path": checkpoint_path.as_posix(),
            "sha256": sha256_file(checkpoint_path),
            "frozen_validation_threshold": checkpoint.threshold,
            "source_checkpoint_sha256": selection_record[
                "source_checkpoint_sha256"
            ],
        }
        print(
            f"PASS {model_name}: clean_auc={result['clean']['metrics']['roc_auc']:.6f}, "
            f"mean_transformed_auc={result['transformed_summary']['mean_roc_auc']:.6f}, "
            f"mean_transformed_bal_acc={result['transformed_summary']['mean_balanced_accuracy']:.6f}, "
            f"worst_bal_acc={result['transformed_summary']['worst_transformed_balanced_accuracy']:.6f}",
            flush=True,
        )

    comparisons = {
        "E2_minus_E1": comparison_deltas(model_results["E1"], model_results["E2"]),
        "E3_minus_E1": comparison_deltas(model_results["E1"], model_results["E3"]),
        "E3_minus_E2": comparison_deltas(model_results["E2"], model_results["E3"]),
    }
    _atomic_probability_write(args.probability_output, test, all_probabilities)
    _write_comparison_figure(model_results, args.comparison_figure)
    _write_confusion_figure(model_results, args.confusion_figure)

    report = {
        "experiment": "section3_frozen_e1_e3_test_comparison",
        "purpose": "One-time controlled test comparison after validation-only model and threshold freezing.",
        "protocol": {
            "path": args.protocol.as_posix(),
            "sha256": protocol_digest,
            "protocol_version": protocol["protocol_version"],
        },
        "pre_test_selection": {
            "primary_model": pre_test_primary_model,
            "basis": "Highest validation threshold-selection objective before test evaluation.",
            "validation_objective_scores": {
                name: row["selection_objective_score"]
                for name, row in threshold_report["models"].items()
            },
            "threshold_report": args.threshold_report.as_posix(),
            "threshold_report_sha256": sha256_file(args.threshold_report),
        },
        "test_data": {
            "split": "CIFAKE internal test",
            "samples": test.samples,
            "class_counts": {
                "real_0": int(np.sum(test.clean.labels == 0)),
                "ai_generated_1": int(np.sum(test.clean.labels == 1)),
            },
            "conditions": ["clean", *conditions],
            "transform_seed": seed,
            "clean_cache": {
                "path": test.clean_cache_path.as_posix(),
                "sha256": test.clean_cache_sha256,
            },
            "transformed_cache_sha256": dict(
                zip(conditions, test.transformed_cache_sha256, strict=True)
            ),
            "path_and_label_alignment_verified": True,
            "organiser_validation_subset_used": False,
        },
        "checkpoints": checkpoint_records,
        "models": model_results,
        "comparisons": comparisons,
        "artifacts": {
            "comparison_figure": args.comparison_figure.as_posix(),
            "key_confusion_matrices": args.confusion_figure.as_posix(),
            "per_image_probabilities": args.probability_output.as_posix(),
            "per_image_probabilities_sha256": sha256_file(args.probability_output),
        },
        "interpretation_guardrail": (
            "These test results must not be used to retrain E1-E3, change their frozen "
            "thresholds, or retroactively choose a different primary model."
        ),
    }
    atomic_json_write(args.report, report)
    print(
        f"Updated report: {args.report}; figures={args.comparison_figure}, "
        f"{args.confusion_figure}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Section 3 test evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
