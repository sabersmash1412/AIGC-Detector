"""Frozen E1-E3 evaluation on the external SID-Set FLUX subset."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.extract_clip_features import atomic_json_write
from scripts.prepare_sid_set_heldout import validate_protocol
from src.evaluate_section3 import (
    _configure_matplotlib,
    validate_frozen_threshold_checkpoint,
)
from src.linear_probe import load_feature_cache
from src.metrics import binary_classification_metrics
from src.robust_linear_training import sha256_file


DEFAULT_PROTOCOL = Path("configs/section4b_held_out_generator.json")
DEFAULT_EMBEDDING_SUMMARY = Path("reports/section4b_clip_embedding_summary.json")
DEFAULT_CIFAKE_REPORT = Path("reports/section3_e1_e3_test_comparison.json")
DEFAULT_REPORT = Path("reports/section4b_held_out_generator_evaluation.json")
DEFAULT_GENERALISATION_FIGURE = Path(
    "reports/figures/section4b_domain_generalisation.png"
)
DEFAULT_CONFUSION_FIGURE = Path(
    "reports/figures/section4b_heldout_confusion_matrices.png"
)
DEFAULT_DISTRIBUTION_FIGURE = Path(
    "reports/figures/section4b_e3_probability_distribution.png"
)
DEFAULT_PROBABILITY_OUTPUT = Path("outputs/section4b_heldout_probabilities.npz")
MODEL_NAMES = ("E1", "E2", "E3")


def class_rates(metrics: dict[str, Any]) -> dict[str, float]:
    """Return class-specific rates from a standard binary confusion matrix."""

    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    if matrix.shape != (2, 2) or np.any(matrix < 0):
        raise ValueError("Expected a non-negative 2x2 confusion matrix")
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    negative_total = int(true_negative + false_positive)
    positive_total = int(false_negative + true_positive)
    if negative_total == 0 or positive_total == 0:
        raise ValueError("Both classes must be present in the confusion matrix")
    return {
        "true_negative_rate_real_recall": float(true_negative / negative_total),
        "false_positive_rate_real_called_ai": float(false_positive / negative_total),
        "true_positive_rate_ai_recall": float(true_positive / positive_total),
        "false_negative_rate_ai_called_real": float(false_negative / positive_total),
    }


def probability_distribution_summary(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    """Summarise score distributions without publishing per-image data."""

    label_array = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if label_array.shape != probability_array.shape or set(np.unique(label_array)) != {
        0,
        1,
    }:
        raise ValueError("Probability summary requires aligned binary classes")
    if not bool(np.isfinite(probability_array).all()) or np.any(
        (probability_array < 0.0) | (probability_array > 1.0)
    ):
        raise ValueError("Probabilities must be finite and lie in [0, 1]")

    def summarise(values: np.ndarray) -> dict[str, float]:
        quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        return {
            "samples": int(len(values)),
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            "minimum": float(np.min(values)),
            "q05": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "q95": float(quantiles[4]),
            "maximum": float(np.max(values)),
        }

    return {
        "real_0": summarise(probability_array[label_array == 0]),
        "flux_ai_generated_1": summarise(probability_array[label_array == 1]),
    }


def domain_metric_deltas(
    cifake_clean: dict[str, Any], heldout: dict[str, Any]
) -> dict[str, float]:
    """Return held-out FLUX minus internal clean-CIFAKE metric differences."""

    deltas = {
        key: float(heldout[key] - cifake_clean[key])
        for key in (
            "roc_auc",
            "average_precision",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "brier_score",
        )
    }
    cifake_rates = class_rates(cifake_clean)
    heldout_rates = class_rates(heldout)
    deltas.update(
        {
            key: float(heldout_rates[key] - cifake_rates[key])
            for key in cifake_rates
        }
    )
    return deltas


def _atomic_probability_write(
    path: Path,
    *,
    image_paths: np.ndarray,
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "image_paths": image_paths,
        "labels": labels,
        **{f"{name}_probabilities": values for name, values in probabilities.items()},
    }
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        np.savez(temporary_path, **arrays)
        temporary_path.replace(path)


def _write_generalisation_figure(
    cifake_results: dict[str, Any],
    heldout_results: dict[str, Any],
    path: Path,
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("roc_auc", "ROC-AUC"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("true_positive_rate_ai_recall", "AI recall"),
        ("true_negative_rate_real_recall", "Real recall"),
    )
    positions = np.arange(len(MODEL_NAMES))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        if metric in {"roc_auc", "balanced_accuracy"}:
            cifake_values = [
                cifake_results[name]["metrics"][metric] for name in MODEL_NAMES
            ]
            heldout_values = [heldout_results[name]["metrics"][metric] for name in MODEL_NAMES]
        else:
            cifake_values = [cifake_results[name]["class_rates"][metric] for name in MODEL_NAMES]
            heldout_values = [heldout_results[name]["class_rates"][metric] for name in MODEL_NAMES]
        axis.bar(
            positions - width / 2,
            cifake_values,
            width,
            label="CIFAKE clean",
            color="#64748b",
        )
        axis.bar(
            positions + width / 2,
            heldout_values,
            width,
            label="SID-Set FLUX",
            color="#dc2626",
        )
        axis.set(
            title=title,
            xticks=positions,
            xticklabels=MODEL_NAMES,
            ylim=(0.0, 1.0),
        )
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(cifake_values):
            axis.text(index - width / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
        for index, value in enumerate(heldout_values):
            axis.text(index + width / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
    axes[0, 0].legend(loc="lower left")
    figure.suptitle("Section 4B — clean-domain versus held-out FLUX generalisation", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_confusion_figure(results: dict[str, Any], path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, model_name in zip(axes, MODEL_NAMES, strict=True):
        matrix = np.asarray(results[model_name]["metrics"]["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1000)
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Real", "FLUX"],
            yticklabels=["Real", "FLUX"],
            xlabel="Predicted",
            ylabel="True",
            title=f"{model_name} — threshold {results[model_name]['threshold']:.3f}",
        )
    figure.suptitle("Section 4B — frozen-threshold SID-Set confusion matrices", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_distribution_figure(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float, path: Path
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0.0, 1.0, 41)
    axis.hist(
        probabilities[labels == 0],
        bins=bins,
        alpha=0.65,
        label="SID-Set real",
        color="#2563eb",
    )
    axis.hist(
        probabilities[labels == 1],
        bins=bins,
        alpha=0.65,
        label="SID-Set FLUX",
        color="#dc2626",
    )
    axis.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Frozen E3 threshold = {threshold:.3f}",
    )
    axis.set(
        xlabel="Predicted probability of AI generation",
        ylabel="Images",
        title="E3 score distribution on held-out SID-Set/FLUX",
        xlim=(0.0, 1.0),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen E1-E3 heads on held-out SID-Set FLUX features."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--embedding-summary", type=Path, default=DEFAULT_EMBEDDING_SUMMARY)
    parser.add_argument("--cifake-report", type=Path, default=DEFAULT_CIFAKE_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--generalisation-figure", type=Path, default=DEFAULT_GENERALISATION_FIGURE)
    parser.add_argument("--confusion-figure", type=Path, default=DEFAULT_CONFUSION_FIGURE)
    parser.add_argument("--distribution-figure", type=Path, default=DEFAULT_DISTRIBUTION_FIGURE)
    parser.add_argument("--probability-output", type=Path, default=DEFAULT_PROBABILITY_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    embedding_summary = json.loads(args.embedding_summary.read_text(encoding="utf-8"))
    if embedding_summary["protocol"]["sha256"] != sha256_file(args.protocol):
        raise ValueError("Embedding summary was produced from a different Section 4B protocol")
    if embedding_summary["frozen_guardrails"]["organiser_validation_subset_used"] is not False:
        raise ValueError("The organiser validation subset must remain untouched")

    cache_path = Path(embedding_summary["feature_cache"]["cache_path"])
    if sha256_file(cache_path) != embedding_summary["feature_cache"]["cache_sha256"]:
        raise ValueError("Held-out feature cache changed after extraction")
    cache = load_feature_cache(cache_path, "heldout")
    if cache.manifest_sha256 != embedding_summary["preparation_validation"][
        "manifest_sha256"
    ]:
        raise ValueError("Held-out cache and audited manifest do not match")

    threshold_report_path = Path(protocol["frozen_evaluation"]["threshold_report"])
    threshold_report = json.loads(threshold_report_path.read_text(encoding="utf-8"))
    section3_protocol_path = Path(threshold_report["protocol"]["path"])
    section3_protocol_sha256 = sha256_file(section3_protocol_path)
    if threshold_report["protocol"]["sha256"] != section3_protocol_sha256:
        raise ValueError("Section 3 threshold report protocol hash changed")
    if threshold_report["selection"]["split"] != "validation":
        raise ValueError("Frozen thresholds did not come from validation data")

    cifake_report = json.loads(args.cifake_report.read_text(encoding="utf-8"))
    if cifake_report["pre_test_selection"]["primary_model"] != "E3":
        raise ValueError("E3 was not preselected before held-out evaluation")
    print(
        f"Frozen held-out evaluation: samples={len(cache.labels)}, generator=FLUX, "
        f"primary={protocol['frozen_evaluation']['primary_model']}",
        flush=True,
    )

    model_results: dict[str, Any] = {}
    probabilities_by_model: dict[str, np.ndarray] = {}
    checkpoint_records: dict[str, Any] = {}
    cifake_clean_results: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        selection = threshold_report["models"][model_name]
        checkpoint_path = Path(protocol["frozen_evaluation"]["models"][model_name])
        if checkpoint_path.as_posix() != selection["thresholded_checkpoint"]:
            raise ValueError(f"Held-out checkpoint path drift for {model_name}")
        checkpoint = validate_frozen_threshold_checkpoint(
            checkpoint_path,
            expected_sha256=selection["thresholded_checkpoint_sha256"],
            expected_threshold=float(selection["selected_threshold"]),
            expected_protocol_sha256=section3_protocol_sha256,
            expected_source_sha256=selection["source_checkpoint_sha256"],
        )
        probabilities = checkpoint.probabilities(cache.features)
        metrics = binary_classification_metrics(
            cache.labels, probabilities, checkpoint.threshold
        )
        metrics["samples"] = len(cache.labels)
        rates = class_rates(metrics)
        distributions = probability_distribution_summary(cache.labels, probabilities)
        cifake_clean = cifake_report["models"][model_name]["clean"]
        cifake_clean_metrics = cifake_clean["metrics"]
        cifake_clean_rates = class_rates(cifake_clean_metrics)
        cifake_clean_results[model_name] = {
            "metrics": cifake_clean_metrics,
            "class_rates": cifake_clean_rates,
        }
        model_results[model_name] = {
            "threshold": checkpoint.threshold,
            "metrics": metrics,
            "class_rates": rates,
            "probability_distributions": distributions,
            "heldout_minus_cifake_clean": domain_metric_deltas(
                cifake_clean_metrics, metrics
            ),
        }
        probabilities_by_model[model_name] = probabilities
        checkpoint_records[model_name] = {
            "path": checkpoint_path.as_posix(),
            "sha256": sha256_file(checkpoint_path),
            "frozen_validation_threshold": checkpoint.threshold,
            "source_checkpoint_sha256": selection["source_checkpoint_sha256"],
        }
        print(
            f"PASS {model_name}: auc={metrics['roc_auc']:.6f}, "
            f"bal_acc={metrics['balanced_accuracy']:.6f}, "
            f"real_recall={rates['true_negative_rate_real_recall']:.6f}, "
            f"flux_recall={rates['true_positive_rate_ai_recall']:.6f}",
            flush=True,
        )

    _atomic_probability_write(
        args.probability_output,
        image_paths=cache.image_paths,
        labels=cache.labels,
        probabilities=probabilities_by_model,
    )
    _write_generalisation_figure(
        cifake_clean_results, model_results, args.generalisation_figure
    )
    _write_confusion_figure(model_results, args.confusion_figure)
    _write_distribution_figure(
        cache.labels,
        probabilities_by_model["E3"],
        model_results["E3"]["threshold"],
        args.distribution_figure,
    )

    report = {
        "experiment": "section4b_frozen_sid_set_flux_held_out_generator",
        "purpose": protocol["purpose"],
        "protocol": {
            "path": args.protocol.as_posix(),
            "sha256": sha256_file(args.protocol),
            "version": protocol["protocol_version"],
        },
        "pre_evaluation_freeze": {
            "primary_model": protocol["frozen_evaluation"]["primary_model"],
            "retraining_allowed": False,
            "threshold_changes_allowed": False,
            "model_reselection_allowed": False,
            "threshold_source": "Section 3 CIFAKE validation data only",
            "threshold_report": threshold_report_path.as_posix(),
            "threshold_report_sha256": sha256_file(threshold_report_path),
        },
        "held_out_data": {
            "dataset": protocol["dataset"]["name"],
            "source_repository": protocol["dataset"]["repository"],
            "source_revision": protocol["dataset"]["source_revision"],
            "source_split": protocol["dataset"]["source_split"],
            "held_out_generator": protocol["task"]["held_out_generator"],
            "samples": len(cache.labels),
            "class_counts": {
                "real_0": int(np.sum(cache.labels == 0)),
                "flux_ai_generated_1": int(np.sum(cache.labels == 1)),
            },
            "content_unique_images": embedding_summary["preparation_validation"][
                "unique_image_sha256"
            ],
            "excluded_exact_duplicates": embedding_summary[
                "preparation_validation"
            ]["excluded_exact_duplicates"],
            "feature_cache": cache_path.as_posix(),
            "feature_cache_sha256": sha256_file(cache_path),
            "manifest_sha256": cache.manifest_sha256,
            "organiser_validation_subset_used": False,
        },
        "licensing_and_publication": {
            "declared_dataset_license": protocol["dataset"]["declared_license"],
            "underlying_real_image_license": protocol["dataset"][
                "underlying_real_image_license"
            ],
            "individual_images_used_in_public_artifacts": False,
            "aggregate_metrics_and_non-image_figures_only": True,
        },
        "checkpoints": checkpoint_records,
        "internal_reference": {
            "dataset": "CIFAKE internal clean test",
            "report": args.cifake_report.as_posix(),
            "report_sha256": sha256_file(args.cifake_report),
            "models": cifake_clean_results,
        },
        "models": model_results,
        "artifacts": {
            "domain_generalisation_figure": args.generalisation_figure.as_posix(),
            "domain_generalisation_figure_sha256": sha256_file(args.generalisation_figure),
            "confusion_matrices": args.confusion_figure.as_posix(),
            "confusion_matrices_sha256": sha256_file(args.confusion_figure),
            "e3_probability_distribution": args.distribution_figure.as_posix(),
            "e3_probability_distribution_sha256": sha256_file(args.distribution_figure),
            "per_image_probabilities": args.probability_output.as_posix(),
            "per_image_probabilities_sha256": sha256_file(args.probability_output),
        },
        "interpretation_guardrails": [
            "The SID-Set result cannot be used to retrain E1-E3, change thresholds, or reselect the primary model.",
            "Differences in content, resolution, and real-image source can confound pure generator attribution.",
            "This deterministic 2,000-image subset is not the complete SID-Set validation split.",
            "The result measures FLUX generalisation on clean external images, not local tampering detection.",
        ],
    }
    atomic_json_write(args.report, report)
    print(
        f"Updated report: {args.report}; figures={args.generalisation_figure}, "
        f"{args.confusion_figure}, {args.distribution_figure}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Held-out generator evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
