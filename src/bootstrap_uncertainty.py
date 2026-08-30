"""Deterministic paired-bootstrap uncertainty for Section 4 evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from scripts.extract_clip_features import atomic_json_write
from src.evaluate_section3 import _configure_matplotlib
from src.image_transforms import FULL_ROBUSTNESS_CONDITIONS, TRANSFORM_SPECS
from src.metrics import binary_classification_metrics
from src.robust_linear_training import sha256_file


DEFAULT_PROTOCOL = Path("configs/section4c_bootstrap.json")
DEFAULT_REPORT = Path("reports/section4c_bootstrap_uncertainty.json")
DEFAULT_CONDITION_FIGURE = Path(
    "reports/figures/section4c_e3_condition_confidence_intervals.png"
)
DEFAULT_SUMMARY_FIGURE = Path(
    "reports/figures/section4c_model_summary_confidence_intervals.png"
)
DEFAULT_DELTA_FIGURE = Path(
    "reports/figures/section4c_paired_model_differences.png"
)
MODEL_NAMES = ("E1", "E2", "E3")


@dataclass(frozen=True)
class StratifiedBootstrapDraws:
    """Class-stratified source indices shared across all paired predictions."""

    negative: np.ndarray
    positive: np.ndarray
    seed: int

    @property
    def replicates(self) -> int:
        return int(self.negative.shape[0])


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Reject uncertainty-protocol drift or any post-test tuning permission."""

    bootstrap = protocol["bootstrap"]
    if int(bootstrap["replicates"]) <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    confidence = float(bootstrap["confidence_level"])
    if not 0.0 < confidence < 1.0:
        raise ValueError("Bootstrap confidence level must lie in (0, 1)")
    if bootstrap["method"] != "nonparametric stratified paired percentile bootstrap":
        raise ValueError("Unexpected bootstrap method")
    if tuple(protocol["paired_comparisons"]) != ("E3_minus_E1", "E3_minus_E2"):
        raise ValueError("Section 4C paired comparisons changed")
    guardrails = protocol["frozen_guardrails"]
    if tuple(guardrails["models"]) != MODEL_NAMES or guardrails["primary_model"] != "E3":
        raise ValueError("E3 and the E1-E3 comparison set must remain frozen")
    for rule in (
        "retraining_allowed",
        "threshold_changes_allowed",
        "model_reselection_allowed",
        "organiser_validation_subset_used",
        "bootstrap_results_allowed_to_change_models_or_thresholds",
    ):
        if guardrails[rule] is not False:
            raise ValueError(f"Frozen uncertainty rule changed: {rule}")


def make_stratified_draws(
    labels: np.ndarray, *, replicates: int, seed: int
) -> StratifiedBootstrapDraws:
    """Generate deterministic class-preserving paired bootstrap source indices."""

    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1 or set(np.unique(label_array)) != {0, 1}:
        raise ValueError("Stratified bootstrap requires one-dimensional binary labels")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    negative = np.flatnonzero(label_array == 0)
    positive = np.flatnonzero(label_array == 1)
    generator = np.random.default_rng(seed)
    return StratifiedBootstrapDraws(
        negative=generator.choice(
            negative, size=(replicates, len(negative)), replace=True
        ),
        positive=generator.choice(
            positive, size=(replicates, len(positive)), replace=True
        ),
        seed=seed,
    )


def _bootstrap_auc(
    probabilities: np.ndarray,
    draws: StratifiedBootstrapDraws,
    *,
    chunk_size: int = 100,
) -> np.ndarray:
    """Compute tie-corrected binary AUCs efficiently from stratified draws."""

    probability_array = np.asarray(probabilities, dtype=np.float64)
    if probability_array.ndim != 1 or not bool(np.isfinite(probability_array).all()):
        raise ValueError("Probabilities must be a finite vector")
    negative_count = draws.negative.shape[1]
    positive_count = draws.positive.shape[1]
    aucs = np.empty(draws.replicates, dtype=np.float64)
    for start in range(0, draws.replicates, chunk_size):
        stop = min(start + chunk_size, draws.replicates)
        scores = np.concatenate(
            (
                probability_array[draws.negative[start:stop]],
                probability_array[draws.positive[start:stop]],
            ),
            axis=1,
        )
        ranks = rankdata(scores, method="average", axis=1)
        positive_rank_sum = ranks[:, negative_count:].sum(axis=1)
        aucs[start:stop] = (
            positive_rank_sum - positive_count * (positive_count + 1) / 2
        ) / (positive_count * negative_count)
    return aucs


def bootstrap_binary_metrics(
    probabilities: np.ndarray,
    *,
    threshold: float,
    draws: StratifiedBootstrapDraws,
    clean_probabilities: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Bootstrap AUC, class rates, balanced accuracy, and optional flip rate."""

    probability_array = np.asarray(probabilities, dtype=np.float64)
    if probability_array.ndim != 1 or not bool(np.isfinite(probability_array).all()):
        raise ValueError("Probabilities must be a finite vector")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1]")
    maximum_index = max(int(draws.negative.max()), int(draws.positive.max()))
    if maximum_index >= len(probability_array):
        raise ValueError("Bootstrap draws exceed the probability vector")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")

    real_recall = np.mean(
        probability_array[draws.negative] < threshold, axis=1
    )
    ai_recall = np.mean(
        probability_array[draws.positive] >= threshold, axis=1
    )
    result = {
        "roc_auc": _bootstrap_auc(probability_array, draws),
        "balanced_accuracy": 0.5 * (real_recall + ai_recall),
        "real_recall": real_recall,
        "real_false_positive_rate": 1.0 - real_recall,
        "ai_recall": ai_recall,
    }
    if clean_probabilities is not None:
        clean = np.asarray(clean_probabilities, dtype=np.float64)
        if clean.shape != probability_array.shape:
            raise ValueError("Clean and transformed probabilities must align")
        clean_predictions = clean >= threshold
        transformed_predictions = probability_array >= threshold
        negative_flips = clean_predictions[draws.negative] != transformed_predictions[
            draws.negative
        ]
        positive_flips = clean_predictions[draws.positive] != transformed_predictions[
            draws.positive
        ]
        result["prediction_flip_rate"] = 0.5 * (
            negative_flips.mean(axis=1) + positive_flips.mean(axis=1)
        )
    return result


def confidence_interval(
    point_estimate: float,
    replicates: np.ndarray,
    *,
    confidence_level: float,
) -> dict[str, Any]:
    """Summarise one point estimate and equal-tailed percentile interval."""

    values = np.asarray(replicates, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not bool(np.isfinite(values).all()):
        raise ValueError("Bootstrap replicates must be a non-empty finite vector")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(values, [alpha / 2, 1.0 - alpha / 2])
    return {
        "point_estimate": float(point_estimate),
        "confidence_level": float(confidence_level),
        "lower": float(lower),
        "upper": float(upper),
        "bootstrap_standard_error": float(np.std(values, ddof=1)),
        "replicates": int(len(values)),
    }


def _point_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    clean_probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = binary_classification_metrics(labels, probabilities, threshold)
    matrix = np.asarray(metrics["confusion_matrix"])
    point = {
        "roc_auc": float(metrics["roc_auc"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "real_recall": float(matrix[0, 0] / matrix[0].sum()),
        "real_false_positive_rate": float(matrix[0, 1] / matrix[0].sum()),
        "ai_recall": float(matrix[1, 1] / matrix[1].sum()),
    }
    if clean_probabilities is not None:
        point["prediction_flip_rate"] = float(
            np.mean(
                (np.asarray(clean_probabilities) >= threshold)
                != (np.asarray(probabilities) >= threshold)
            )
        )
    return point


def _validate_input_artifact(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact is missing: {path}")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Frozen artifact hash changed: {path}")


def _load_probabilities(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if "labels" not in arrays or "image_paths" not in arrays:
        raise ValueError(f"Probability archive lacks identity arrays: {path}")
    labels = arrays["labels"]
    if labels.shape != (2000,) or set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"Expected 2,000 balanced binary examples: {path}")
    if np.sum(labels == 0) != 1000 or np.sum(labels == 1) != 1000:
        raise ValueError(f"Expected exactly 1,000 examples per class: {path}")
    if len(set(arrays["image_paths"].tolist())) != 2000:
        raise ValueError(f"Probability archive paths are not unique: {path}")
    return arrays


def _intervals_for_series(
    point: dict[str, float],
    bootstrap: dict[str, np.ndarray],
    metrics: tuple[str, ...],
    confidence_level: float,
) -> dict[str, Any]:
    return {
        metric: confidence_interval(
            point[metric], bootstrap[metric], confidence_level=confidence_level
        )
        for metric in metrics
    }


def _full_matrix_uncertainty(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    draws: StratifiedBootstrapDraws,
    confidence_level: float,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    labels = arrays["labels"]
    conditions = tuple(report["test_data"]["conditions"])
    if conditions != FULL_ROBUSTNESS_CONDITIONS:
        raise ValueError("Full-matrix report conditions changed")
    output_models: dict[str, Any] = {}
    summary_replicates: dict[str, dict[str, np.ndarray]] = {}
    for model_name in MODEL_NAMES:
        threshold = float(report["models"][model_name]["threshold"])
        clean = arrays[f"{model_name}_clean"]
        by_condition: dict[str, Any] = {}
        condition_bootstrap: dict[str, dict[str, np.ndarray]] = {}
        condition_points: dict[str, dict[str, float]] = {}
        for condition in conditions:
            probabilities = arrays[f"{model_name}_{condition}"]
            point = _point_metrics(labels, probabilities, threshold, clean)
            boot = bootstrap_binary_metrics(
                probabilities,
                threshold=threshold,
                draws=draws,
                clean_probabilities=clean,
            )
            condition_points[condition] = point
            condition_bootstrap[condition] = boot
            by_condition[condition] = _intervals_for_series(
                point,
                boot,
                ("roc_auc", "balanced_accuracy", "prediction_flip_rate"),
                confidence_level,
            )

        transformed = conditions[1:]
        summary_boot = {
            "mean_transformed_roc_auc": np.mean(
                [condition_bootstrap[name]["roc_auc"] for name in transformed], axis=0
            ),
            "mean_transformed_balanced_accuracy": np.mean(
                [condition_bootstrap[name]["balanced_accuracy"] for name in transformed],
                axis=0,
            ),
            "worst_transformed_balanced_accuracy": np.min(
                [condition_bootstrap[name]["balanced_accuracy"] for name in transformed],
                axis=0,
            ),
            "mean_transformed_prediction_flip_rate": np.mean(
                [condition_bootstrap[name]["prediction_flip_rate"] for name in transformed],
                axis=0,
            ),
        }
        summary_point = {
            "mean_transformed_roc_auc": float(
                np.mean([condition_points[name]["roc_auc"] for name in transformed])
            ),
            "mean_transformed_balanced_accuracy": float(
                np.mean(
                    [condition_points[name]["balanced_accuracy"] for name in transformed]
                )
            ),
            "worst_transformed_balanced_accuracy": float(
                np.min(
                    [condition_points[name]["balanced_accuracy"] for name in transformed]
                )
            ),
            "mean_transformed_prediction_flip_rate": float(
                np.mean(
                    [condition_points[name]["prediction_flip_rate"] for name in transformed]
                )
            ),
        }
        output_models[model_name] = {
            "threshold": threshold,
            "by_condition": by_condition,
            "summary": _intervals_for_series(
                summary_point,
                summary_boot,
                tuple(summary_boot),
                confidence_level,
            ),
        }
        summary_replicates[model_name] = summary_boot

    comparisons = _paired_comparisons(
        {name: output_models[name]["summary"] for name in MODEL_NAMES},
        summary_replicates,
        confidence_level,
    )
    return {
        "samples": len(labels),
        "conditions": list(conditions),
        "models": output_models,
        "paired_comparisons": comparisons,
    }, summary_replicates


def _heldout_uncertainty(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    draws: StratifiedBootstrapDraws,
    confidence_level: float,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    labels = arrays["labels"]
    models: dict[str, Any] = {}
    bootstrap_by_model: dict[str, dict[str, np.ndarray]] = {}
    metrics = (
        "roc_auc",
        "balanced_accuracy",
        "real_recall",
        "real_false_positive_rate",
        "flux_recall",
    )
    for model_name in MODEL_NAMES:
        threshold = float(report["models"][model_name]["threshold"])
        probabilities = arrays[f"{model_name}_probabilities"]
        point = _point_metrics(labels, probabilities, threshold)
        point["flux_recall"] = point.pop("ai_recall")
        boot = bootstrap_binary_metrics(
            probabilities, threshold=threshold, draws=draws
        )
        boot["flux_recall"] = boot.pop("ai_recall")
        models[model_name] = {
            "threshold": threshold,
            "metrics": _intervals_for_series(
                point, boot, metrics, confidence_level
            ),
        }
        bootstrap_by_model[model_name] = boot
    comparisons = _paired_comparisons(
        {name: models[name]["metrics"] for name in MODEL_NAMES},
        bootstrap_by_model,
        confidence_level,
    )
    return {
        "samples": len(labels),
        "generator": report["held_out_data"]["held_out_generator"],
        "models": models,
        "paired_comparisons": comparisons,
    }, bootstrap_by_model


def _paired_comparisons(
    interval_records: dict[str, dict[str, Any]],
    bootstrap_by_model: dict[str, dict[str, np.ndarray]],
    confidence_level: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for candidate, baseline in (("E3", "E1"), ("E3", "E2")):
        name = f"{candidate}_minus_{baseline}"
        shared_metrics = tuple(
            metric
            for metric in bootstrap_by_model[candidate]
            if metric in bootstrap_by_model[baseline]
        )
        output[name] = {
            metric: confidence_interval(
                interval_records[candidate][metric]["point_estimate"]
                - interval_records[baseline][metric]["point_estimate"],
                bootstrap_by_model[candidate][metric]
                - bootstrap_by_model[baseline][metric],
                confidence_level=confidence_level,
            )
            for metric in shared_metrics
        }
    return output


def _errorbar_values(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray([row["point_estimate"] for row in records])
    errors = np.asarray(
        [
            [point - row["lower"] for point, row in zip(points, records, strict=True)],
            [row["upper"] - point for point, row in zip(points, records, strict=True)],
        ]
    )
    return points, errors


def _write_condition_figure(full: dict[str, Any], path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = tuple(full["conditions"])
    e3 = full["models"]["E3"]["by_condition"]
    positions = np.arange(len(conditions))
    figure, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
    for axis, metric, title in (
        (axes[0], "roc_auc", "ROC-AUC"),
        (axes[1], "balanced_accuracy", "Balanced accuracy at frozen threshold"),
    ):
        records = [e3[name][metric] for name in conditions]
        points, errors = _errorbar_values(records)
        axis.errorbar(
            positions,
            points,
            yerr=errors,
            fmt="o",
            capsize=4,
            color="#dc2626",
            ecolor="#64748b",
        )
        axis.set(title=title, ylabel=metric.replace("_", " ").title(), ylim=(0.45, 1.01))
        axis.grid(axis="y", alpha=0.25)
    axes[1].set(
        xticks=positions,
        xticklabels=[TRANSFORM_SPECS[name].display_name for name in conditions],
    )
    axes[1].tick_params(axis="x", rotation=35)
    figure.suptitle("Section 4C — E3 95% paired-bootstrap confidence intervals", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_summary_figure(full: dict[str, Any], heldout: dict[str, Any], path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        (full, "summary", "mean_transformed_roc_auc", "Full matrix mean ROC-AUC"),
        (
            full,
            "summary",
            "mean_transformed_balanced_accuracy",
            "Full matrix mean balanced accuracy",
        ),
        (heldout, "metrics", "roc_auc", "Held-out FLUX ROC-AUC"),
        (heldout, "metrics", "balanced_accuracy", "Held-out FLUX balanced accuracy"),
    )
    positions = np.arange(len(MODEL_NAMES))
    colors = ("#64748b", "#2563eb", "#dc2626")
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, (dataset, section, metric, title) in zip(axes.flat, panels, strict=True):
        records = [dataset["models"][name][section][metric] for name in MODEL_NAMES]
        points, errors = _errorbar_values(records)
        axis.bar(positions, points, color=colors, alpha=0.85)
        axis.errorbar(positions, points, yerr=errors, fmt="none", capsize=5, color="black")
        axis.set(
            title=title,
            xticks=positions,
            xticklabels=MODEL_NAMES,
            ylim=(0.5, 1.0),
        )
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Section 4C — model summary uncertainty", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_delta_figure(full: dict[str, Any], heldout: dict[str, Any], path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        ("Matrix AUC", full, "mean_transformed_roc_auc"),
        ("Matrix BA", full, "mean_transformed_balanced_accuracy"),
        ("FLUX AUC", heldout, "roc_auc"),
        ("FLUX BA", heldout, "balanced_accuracy"),
        ("FLUX real recall", heldout, "real_recall"),
    ]
    comparisons = ("E3_minus_E1", "E3_minus_E2")
    figure, axis = plt.subplots(figsize=(10, 6))
    offsets = (-0.12, 0.12)
    colors = ("#dc2626", "#2563eb")
    for comparison, offset, color in zip(comparisons, offsets, colors, strict=True):
        records = [dataset["paired_comparisons"][comparison][metric] for _, dataset, metric in rows]
        points, errors = _errorbar_values(records)
        positions = np.arange(len(rows)) + offset
        axis.errorbar(
            points,
            positions,
            xerr=errors,
            fmt="o",
            capsize=4,
            label=comparison.replace("_", " "),
            color=color,
        )
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.5)
    axis.set(
        yticks=np.arange(len(rows)),
        yticklabels=[name for name, _, _ in rows],
        xlabel="Paired metric difference (E3 minus baseline)",
        title="Section 4C — paired 95% bootstrap differences",
    )
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen paired-bootstrap uncertainty for Section 4A and 4B."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--condition-figure", type=Path, default=DEFAULT_CONDITION_FIGURE)
    parser.add_argument("--summary-figure", type=Path, default=DEFAULT_SUMMARY_FIGURE)
    parser.add_argument("--delta-figure", type=Path, default=DEFAULT_DELTA_FIGURE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    for source in ("full_matrix", "held_out_flux"):
        _validate_input_artifact(
            Path(protocol[source]["report"]), protocol[source]["report_sha256"]
        )
        _validate_input_artifact(
            Path(protocol[source]["probabilities"]),
            protocol[source]["probabilities_sha256"],
        )
    full_report = json.loads(Path(protocol["full_matrix"]["report"]).read_text())
    heldout_report = json.loads(Path(protocol["held_out_flux"]["report"]).read_text())
    if full_report["test_data"]["organiser_validation_subset_used"] is not False:
        raise ValueError("Full-matrix report used the organiser subset")
    if heldout_report["held_out_data"]["organiser_validation_subset_used"] is not False:
        raise ValueError("Held-out report used the organiser subset")
    if full_report["frozen_before_evaluation"]["primary_model"] != "E3":
        raise ValueError("Full-matrix primary model changed")
    if heldout_report["pre_evaluation_freeze"]["primary_model"] != "E3":
        raise ValueError("Held-out primary model changed")

    full_arrays = _load_probabilities(Path(protocol["full_matrix"]["probabilities"]))
    heldout_arrays = _load_probabilities(Path(protocol["held_out_flux"]["probabilities"]))
    bootstrap = protocol["bootstrap"]
    replicates = int(bootstrap["replicates"])
    seed = int(bootstrap["seed"])
    confidence_level = float(bootstrap["confidence_level"])
    full_draws = make_stratified_draws(
        full_arrays["labels"], replicates=replicates, seed=seed
    )
    heldout_draws = make_stratified_draws(
        heldout_arrays["labels"], replicates=replicates, seed=seed
    )
    print(
        f"Paired bootstrap: replicates={replicates}, confidence={confidence_level:.0%}, "
        f"seed={seed}, class_draws=1000+1000",
        flush=True,
    )
    full, _ = _full_matrix_uncertainty(
        full_report, full_arrays, full_draws, confidence_level
    )
    print("PASS full transformation matrix uncertainty", flush=True)
    heldout, _ = _heldout_uncertainty(
        heldout_report, heldout_arrays, heldout_draws, confidence_level
    )
    print("PASS held-out FLUX uncertainty", flush=True)

    _write_condition_figure(full, args.condition_figure)
    _write_summary_figure(full, heldout, args.summary_figure)
    _write_delta_figure(full, heldout, args.delta_figure)
    report = {
        "experiment": protocol["experiment_name"],
        "purpose": protocol["purpose"],
        "protocol": {
            "path": args.protocol.as_posix(),
            "sha256": sha256_file(args.protocol),
            "version": protocol["protocol_version"],
        },
        "method": protocol["bootstrap"],
        "frozen_inputs": {
            source: {
                "report": protocol[source]["report"],
                "report_sha256": protocol[source]["report_sha256"],
                "probabilities": protocol[source]["probabilities"],
                "probabilities_sha256": protocol[source]["probabilities_sha256"],
            }
            for source in ("full_matrix", "held_out_flux")
        },
        "full_matrix": full,
        "held_out_flux": heldout,
        "artifacts": {
            "e3_condition_confidence_intervals": args.condition_figure.as_posix(),
            "e3_condition_confidence_intervals_sha256": sha256_file(args.condition_figure),
            "model_summary_confidence_intervals": args.summary_figure.as_posix(),
            "model_summary_confidence_intervals_sha256": sha256_file(args.summary_figure),
            "paired_model_differences": args.delta_figure.as_posix(),
            "paired_model_differences_sha256": sha256_file(args.delta_figure),
        },
        "guardrails": protocol["frozen_guardrails"],
        "interpretation": protocol["interpretation"],
    }
    atomic_json_write(args.report, report)
    e3_flux = heldout["models"]["E3"]["metrics"]
    print(
        "E3 held-out: "
        f"auc={e3_flux['roc_auc']['point_estimate']:.6f} "
        f"[{e3_flux['roc_auc']['lower']:.6f}, {e3_flux['roc_auc']['upper']:.6f}], "
        f"real_fpr={e3_flux['real_false_positive_rate']['point_estimate']:.6f} "
        f"[{e3_flux['real_false_positive_rate']['lower']:.6f}, "
        f"{e3_flux['real_false_positive_rate']['upper']:.6f}]",
        flush=True,
    )
    print(
        f"Updated report: {args.report}; figures={args.condition_figure}, "
        f"{args.summary_figure}, {args.delta_figure}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Bootstrap uncertainty failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
