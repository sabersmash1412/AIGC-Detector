"""Frozen Section 4A evaluation over the complete transformation matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts.extract_clip_features import atomic_json_write
from src.evaluate_section3 import (
    _atomic_probability_write,
    _configure_matplotlib,
    _probabilities_by_condition,
    comparison_deltas,
    evaluate_model_conditions,
    validate_frozen_threshold_checkpoint,
)
from src.image_transforms import FULL_ROBUSTNESS_CONDITIONS, TRANSFORM_SPECS
from src.robust_linear_training import load_paired_feature_set, sha256_file


DEFAULT_PROTOCOL = Path("configs/section4_evaluation.json")
DEFAULT_THRESHOLD_REPORT = Path("reports/section3_threshold_selection.json")
DEFAULT_REPORT = Path("reports/section4_full_transformation_matrix.json")
DEFAULT_SEVERITY_FIGURE = Path("reports/figures/section4_severity_curves.png")
DEFAULT_HEATMAP_FIGURE = Path("reports/figures/section4_full_matrix_heatmap.png")
DEFAULT_CONFUSION_FIGURE = Path(
    "reports/figures/section4_worst_severity_confusion_matrices.png"
)
DEFAULT_PROBABILITY_OUTPUT = Path("outputs/section4_full_matrix_probabilities.npz")

MODEL_NAMES = ("E1", "E2", "E3")
SEVERE_CONDITIONS = (
    "jpeg_q30",
    "gaussian_blur_sigma2",
    "resize_0_25x",
    "gaussian_noise_sigma0_10",
)


def validate_section4_protocol(protocol: dict[str, Any]) -> None:
    """Reject any protocol drift that could turn the stress test into tuning."""

    full_conditions = tuple(protocol["full_matrix_conditions"])
    if full_conditions != FULL_ROBUSTNESS_CONDITIONS:
        raise ValueError("Section 4 conditions do not match the transform registry")
    if tuple(protocol["models"]) != MODEL_NAMES:
        raise ValueError("Section 4 must evaluate exactly E1, E2, and E3")
    if protocol["preselected_primary_model"] != "E3":
        raise ValueError("E3 must remain the primary model selected before Section 4")
    if protocol["data"]["organiser_validation_subset_used"] is not False:
        raise ValueError("The organiser validation subset must remain untouched")

    rules = protocol["frozen_evaluation_rules"]
    for rule in (
        "retraining_allowed",
        "threshold_changes_allowed",
        "model_reselection_from_matrix_results_allowed",
    ):
        if rules[rule] is not False:
            raise ValueError(f"Frozen evaluation rule changed: {rule}")
    if rules["same_paths_labels_and_seed_for_all_models"] is not True:
        raise ValueError("All models must use identical paired examples")

    representative = tuple(protocol["previously_evaluated_representative_conditions"])
    new_severities = tuple(protocol["new_severity_conditions"])
    if representative[0] != "clean":
        raise ValueError("Representative conditions must start with clean")
    if set(representative).intersection(new_severities):
        raise ValueError("Representative and new-severity conditions must be disjoint")
    if set(representative).union(new_severities) != set(full_conditions):
        raise ValueError("Representative and new severities must cover the full matrix")

    flattened_families = tuple(
        condition
        for conditions in protocol["severity_families"].values()
        for condition in conditions
    )
    if flattened_families != full_conditions[1:]:
        raise ValueError("Severity families must cover transformed conditions in order")


def subset_summary(
    model_result: dict[str, Any], conditions: tuple[str, ...]
) -> dict[str, Any]:
    """Summarise a named, predeclared subset of evaluation conditions."""

    if not conditions:
        raise ValueError("A condition subset cannot be empty")
    missing = [name for name in conditions if name not in model_result["by_condition"]]
    if missing:
        raise ValueError(f"Missing conditions from model result: {missing}")

    rows = model_result["by_condition"]
    return {
        "conditions": list(conditions),
        "mean_roc_auc": float(
            np.mean([rows[name]["metrics"]["roc_auc"] for name in conditions])
        ),
        "mean_balanced_accuracy": float(
            np.mean(
                [rows[name]["metrics"]["balanced_accuracy"] for name in conditions]
            )
        ),
        "mean_prediction_flip_rate": float(
            np.mean(
                [
                    rows[name]["stability_vs_clean"]["prediction_flip_rate"]
                    for name in conditions
                ]
            )
        ),
        "worst_roc_auc_condition": min(
            conditions, key=lambda name: rows[name]["metrics"]["roc_auc"]
        ),
        "worst_roc_auc": float(
            min(rows[name]["metrics"]["roc_auc"] for name in conditions)
        ),
        "worst_balanced_accuracy_condition": min(
            conditions,
            key=lambda name: rows[name]["metrics"]["balanced_accuracy"],
        ),
        "worst_balanced_accuracy": float(
            min(rows[name]["metrics"]["balanced_accuracy"] for name in conditions)
        ),
        "maximum_prediction_flip_rate_condition": max(
            conditions,
            key=lambda name: rows[name]["stability_vs_clean"][
                "prediction_flip_rate"
            ],
        ),
        "maximum_prediction_flip_rate": float(
            max(
                rows[name]["stability_vs_clean"]["prediction_flip_rate"]
                for name in conditions
            )
        ),
    }


def add_predeclared_summaries(
    model_result: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    """Attach full, seen/new, and family summaries without selecting post hoc groups."""

    representative_transformed = tuple(
        protocol["previously_evaluated_representative_conditions"][1:]
    )
    new_severities = tuple(protocol["new_severity_conditions"])
    all_transformed = tuple(protocol["full_matrix_conditions"][1:])
    return {
        "all_14_transformed": subset_summary(model_result, all_transformed),
        "six_representative_transforms": subset_summary(
            model_result, representative_transformed
        ),
        "eight_new_severities": subset_summary(model_result, new_severities),
        "by_predeclared_family": {
            family: subset_summary(model_result, tuple(conditions))
            for family, conditions in protocol["severity_families"].items()
        },
    }


def _write_severity_figure(
    model_results: dict[str, dict[str, Any]],
    families: dict[str, list[str]],
    path: Path,
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve_families = tuple(name for name in families if name != "single_setting")
    x_labels = {
        "jpeg": ("90", "70", "50", "30"),
        "gaussian_blur": ("0.5", "1", "2"),
        "resize": ("0.5×", "0.25×"),
        "gaussian_noise": ("0.02", "0.05", "0.10"),
    }
    titles = {
        "jpeg": "JPEG quality (lower = stronger)",
        "gaussian_blur": "Gaussian blur σ",
        "resize": "Downscale factor",
        "gaussian_noise": "Gaussian noise σ",
    }
    colors = {"E1": "#6b7280", "E2": "#2563eb", "E3": "#dc2626"}
    figure, axes = plt.subplots(2, 4, figsize=(18, 8), sharey="row")
    for column, family in enumerate(curve_families):
        conditions = families[family]
        positions = np.arange(len(conditions))
        for model_name, result in model_results.items():
            axes[0, column].plot(
                positions,
                [result["by_condition"][name]["metrics"]["roc_auc"] for name in conditions],
                marker="o",
                linewidth=2,
                label=model_name,
                color=colors[model_name],
            )
            axes[1, column].plot(
                positions,
                [
                    result["by_condition"][name]["metrics"]["balanced_accuracy"]
                    for name in conditions
                ],
                marker="o",
                linewidth=2,
                label=model_name,
                color=colors[model_name],
            )
        for row in range(2):
            axes[row, column].set_xticks(positions, x_labels[family])
            axes[row, column].grid(alpha=0.25)
            axes[row, column].set_ylim(0.45, 1.01)
        axes[0, column].set_title(titles[family])
        axes[1, column].set_xlabel("Frozen test severity")
    axes[0, 0].set_ylabel("ROC-AUC")
    axes[1, 0].set_ylabel("Balanced accuracy")
    axes[0, -1].legend(loc="lower left")
    figure.suptitle("Section 4A — severity-dependent robustness", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_heatmap(
    model_results: dict[str, dict[str, Any]], conditions: tuple[str, ...], path: Path
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = tuple(model_results)
    figure, axes = plt.subplots(2, 1, figsize=(22, 6.8))
    metric_rows = (
        ("roc_auc", "ROC-AUC"),
        ("balanced_accuracy", "Balanced accuracy at frozen threshold"),
    )
    for axis, (metric, title) in zip(axes, metric_rows, strict=True):
        values = np.asarray(
            [
                [
                    model_results[model]["by_condition"][condition]["metrics"][metric]
                    for condition in conditions
                ]
                for model in model_names
            ]
        )
        image = axis.imshow(values, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if values[row, column] > 0.62 else "white",
                )
        axis.set(
            yticks=np.arange(len(model_names)),
            yticklabels=model_names,
            xticks=np.arange(len(conditions)),
            xticklabels=[TRANSFORM_SPECS[name].display_name for name in conditions],
            title=title,
        )
        axis.tick_params(axis="x", rotation=35)
        figure.colorbar(image, ax=axis, fraction=0.018, pad=0.01)
    figure.suptitle("Section 4A — complete frozen transformation matrix", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_confusion_figure(
    model_results: dict[str, dict[str, Any]], path: Path
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 4, figsize=(14, 10))
    for row, (model_name, result) in enumerate(model_results.items()):
        for column, condition in enumerate(SEVERE_CONDITIONS):
            axis = axes[row, column]
            matrix = np.asarray(
                result["by_condition"][condition]["metrics"]["confusion_matrix"]
            )
            axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1000)
            for true_label in range(2):
                for predicted_label in range(2):
                    axis.text(
                        predicted_label,
                        true_label,
                        str(matrix[true_label, predicted_label]),
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
    figure.suptitle("Section 4A — strongest-severity confusion matrices", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen E1-E3 models on the full 15-condition matrix."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--threshold-report", type=Path, default=DEFAULT_THRESHOLD_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--severity-figure", type=Path, default=DEFAULT_SEVERITY_FIGURE)
    parser.add_argument("--heatmap-figure", type=Path, default=DEFAULT_HEATMAP_FIGURE)
    parser.add_argument("--confusion-figure", type=Path, default=DEFAULT_CONFUSION_FIGURE)
    parser.add_argument(
        "--probability-output", type=Path, default=DEFAULT_PROBABILITY_OUTPUT
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_section4_protocol(protocol)
    threshold_report = json.loads(args.threshold_report.read_text(encoding="utf-8"))

    section3_protocol_path = Path(threshold_report["protocol"]["path"])
    section3_protocol_digest = sha256_file(section3_protocol_path)
    if threshold_report["protocol"]["sha256"] != section3_protocol_digest:
        raise ValueError("Threshold report no longer matches its Section 3 protocol")
    if threshold_report["selection"]["split"] != "validation":
        raise ValueError("Frozen thresholds must come only from validation data")

    conditions = tuple(protocol["full_matrix_conditions"][1:])
    test = load_paired_feature_set(
        split=protocol["data"]["split"],
        clean_cache_path=Path(protocol["data"]["clean_feature_cache"]),
        transformed_feature_dir=Path(protocol["data"]["transformed_feature_directory"]),
        conditions=conditions,
        seed=int(protocol["transform_seed"]),
    )
    expected_samples = int(protocol["data"]["samples"])
    if test.samples != expected_samples:
        raise ValueError(f"Expected {expected_samples} test examples, found {test.samples}")

    print(
        f"Frozen Section 4A evaluation: samples={test.samples}, "
        f"conditions={len(conditions) + 1}, primary={protocol['preselected_primary_model']}",
        flush=True,
    )
    model_results: dict[str, dict[str, Any]] = {}
    all_probabilities: dict[str, dict[str, np.ndarray]] = {}
    checkpoint_records: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        selection_record = threshold_report["models"][model_name]
        checkpoint_path = Path(protocol["models"][model_name])
        if checkpoint_path.as_posix() != selection_record["thresholded_checkpoint"]:
            raise ValueError(f"Section 4 checkpoint drift for {model_name}")
        checkpoint = validate_frozen_threshold_checkpoint(
            checkpoint_path,
            expected_sha256=selection_record["thresholded_checkpoint_sha256"],
            expected_threshold=float(selection_record["selected_threshold"]),
            expected_protocol_sha256=section3_protocol_digest,
            expected_source_sha256=selection_record["source_checkpoint_sha256"],
        )
        probabilities = _probabilities_by_condition(test, checkpoint)
        result = evaluate_model_conditions(
            test.clean.labels, probabilities, checkpoint.threshold
        )
        result["predeclared_summaries"] = add_predeclared_summaries(result, protocol)
        model_results[model_name] = result
        all_probabilities[model_name] = probabilities
        checkpoint_records[model_name] = {
            "path": checkpoint_path.as_posix(),
            "sha256": sha256_file(checkpoint_path),
            "frozen_validation_threshold": checkpoint.threshold,
            "source_checkpoint_sha256": selection_record["source_checkpoint_sha256"],
        }
        summary = result["predeclared_summaries"]["all_14_transformed"]
        print(
            f"PASS {model_name}: mean_auc={summary['mean_roc_auc']:.6f}, "
            f"mean_bal_acc={summary['mean_balanced_accuracy']:.6f}, "
            f"worst_bal_acc={summary['worst_balanced_accuracy']:.6f} "
            f"({summary['worst_balanced_accuracy_condition']}), "
            f"mean_flip={summary['mean_prediction_flip_rate']:.6f}",
            flush=True,
        )

    comparisons = {
        "E2_minus_E1": comparison_deltas(model_results["E1"], model_results["E2"]),
        "E3_minus_E1": comparison_deltas(model_results["E1"], model_results["E3"]),
        "E3_minus_E2": comparison_deltas(model_results["E2"], model_results["E3"]),
    }
    _atomic_probability_write(args.probability_output, test, all_probabilities)
    _write_severity_figure(model_results, protocol["severity_families"], args.severity_figure)
    _write_heatmap(model_results, tuple(protocol["full_matrix_conditions"]), args.heatmap_figure)
    _write_confusion_figure(model_results, args.confusion_figure)

    report = {
        "experiment": "section4a_frozen_full_transformation_matrix",
        "purpose": protocol["purpose"],
        "protocol": {
            "path": args.protocol.as_posix(),
            "sha256": sha256_file(args.protocol),
            "protocol_version": protocol["protocol_version"],
        },
        "frozen_before_evaluation": {
            "primary_model": protocol["preselected_primary_model"],
            "threshold_source": "Section 3 validation data only",
            "threshold_report": args.threshold_report.as_posix(),
            "threshold_report_sha256": sha256_file(args.threshold_report),
            "rules": protocol["frozen_evaluation_rules"],
        },
        "test_data": {
            "dataset": protocol["data"]["dataset"],
            "split": protocol["data"]["split"],
            "samples": test.samples,
            "class_counts": {
                "real_0": int(np.sum(test.clean.labels == 0)),
                "ai_generated_1": int(np.sum(test.clean.labels == 1)),
            },
            "conditions": list(protocol["full_matrix_conditions"]),
            "transform_seed": int(protocol["transform_seed"]),
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
        "condition_groups": {
            "previously_evaluated_representative_conditions": protocol[
                "previously_evaluated_representative_conditions"
            ],
            "new_severity_conditions": protocol["new_severity_conditions"],
            "severity_families": protocol["severity_families"],
        },
        "checkpoints": checkpoint_records,
        "models": model_results,
        "comparisons": comparisons,
        "artifacts": {
            "severity_curves": args.severity_figure.as_posix(),
            "severity_curves_sha256": sha256_file(args.severity_figure),
            "full_matrix_heatmap": args.heatmap_figure.as_posix(),
            "full_matrix_heatmap_sha256": sha256_file(args.heatmap_figure),
            "worst_severity_confusion_matrices": args.confusion_figure.as_posix(),
            "worst_severity_confusion_matrices_sha256": sha256_file(args.confusion_figure),
            "per_image_probabilities": args.probability_output.as_posix(),
            "per_image_probabilities_sha256": sha256_file(args.probability_output),
        },
        "interpretation_guardrail": (
            "This final stress-test matrix must not be used to retrain E1-E3, "
            "change their validation-selected thresholds, or reselect the primary model."
        ),
    }
    atomic_json_write(args.report, report)
    print(
        f"Updated report: {args.report}; figures={args.severity_figure}, "
        f"{args.heatmap_figure}, {args.confusion_figure}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Section 4A evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
