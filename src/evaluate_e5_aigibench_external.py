"""Single-use frozen E5 evaluation on AIGIBench Open Images/Midjourney V6."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.extract_clip_features import atomic_json_write
from src.e5_aigibench_amendment import validate_amendment
from src.e5_aigibench_protocol import validate_e5_aigibench_protocol
from src.e5_protocol import wilson_interval
from src.evaluate_heldout_generator import class_rates
from src.evaluate_section3 import _configure_matplotlib
from src.image_transforms import FULL_ROBUSTNESS_CONDITIONS, TRANSFORM_SPECS
from src.linear_probe import FeatureCache, load_feature_cache, load_linear_probe_checkpoint
from src.metrics import binary_classification_metrics
from src.robust_linear_training import sha256_file
from src.transformed_features import load_transformed_feature_cache


DEFAULT_RUN_LOCK = Path("configs/e5_aigibench_external_evaluation_run.json")
MODEL_NAMES = ("E3", "E4", "E5")
CLASS_NAMES = {0: "real_0", 1: "midjourney_v6_ai_generated_1"}
EXPECTED_PRIMARY = (
    "clean",
    "jpeg_q50",
    "gaussian_blur_sigma1",
    "resize_0_5x",
    "gaussian_noise_sigma0_05",
    "color_jitter_seeded_20pct",
    "center_crop_80pct",
)
EXPECTED_STRESS = (
    "jpeg_q90",
    "jpeg_q70",
    "jpeg_q30",
    "gaussian_blur_sigma0_5",
    "gaussian_blur_sigma2",
    "resize_0_25x",
    "gaussian_noise_sigma0_02",
    "gaussian_noise_sigma0_10",
)


def _scalar(archive: np.lib.npyio.NpzFile, key: str) -> object:
    if key not in archive.files or archive[key].ndim != 0:
        raise ValueError(f"Checkpoint metadata {key!r} is missing or non-scalar")
    return archive[key].item()


def _load_verified_json(root: Path, record: dict[str, str], description: str) -> dict[str, Any]:
    path = root / record["path"]
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"Frozen {description} changed or is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run_lock(run: dict[str, Any]) -> None:
    """Reject any mutation of the final evaluation design before scoring."""

    if run["status"] != (
        "frozen_after_all_feature_extraction_before_any_checkpoint_load_or_score_computation"
    ):
        raise ValueError("Final external evaluation was not frozen before scoring")
    if "universal" not in run["research_claim_limit"].lower():
        raise ValueError("External-result claim limitation changed")
    conditions = run["conditions"]
    if tuple(conditions["full_matrix"]) != FULL_ROBUSTNESS_CONDITIONS:
        raise ValueError("Frozen full transformation matrix changed")
    if tuple(conditions["primary_claim_conditions"]) != EXPECTED_PRIMARY:
        raise ValueError("Frozen primary conditions changed")
    if tuple(conditions["mandatory_stress_only_conditions"]) != EXPECTED_STRESS:
        raise ValueError("Frozen stress conditions changed")

    expected_gates = {
        "confidence_level": 0.95,
        "maximum_wilson_upper_real_called_ai": 0.05,
        "maximum_wilson_upper_ai_called_real": 0.1,
        "minimum_clean_decisive_coverage_per_class": 0.6,
        "minimum_worst_primary_class_condition_decisive_coverage": 0.25,
        "minimum_mean_primary_class_condition_decisive_coverage": 0.5,
        "minimum_worst_primary_condition_roc_auc": 0.8,
        "all_criteria_required_for_pass": True,
    }
    if run["frozen_success_criteria"] != expected_gates:
        raise ValueError("Frozen external success gates changed")
    data = run["frozen_external_data"]
    if data["class_counts"] != {
        "real_0": 1000,
        "midjourney_v6_ai_generated_1": 1000,
    }:
        raise ValueError("Frozen external class counts changed")
    if int(data["samples"]) != 2000 or int(data["feature_dimension"]) != 512:
        raise ValueError("Frozen external feature dimensions changed")
    if int(data["transform_seed"]) != 42:
        raise ValueError("Frozen transform seed changed")

    expected_models = {
        "E3": (0.437, "1523bc38fc4626a3e252464da146ad66af074c4196ebb2f44719ec149fd3482a"),
        "E4": (0.6930000000000001, "7379642904d17627d8e9df708e352ec141aa9c609ef7400cd4b9e8896254f542"),
        "E5": (0.52, "b6c25a38a86692a74280650f516105c01efbaabe91f8da728b1a455cbf1756c4"),
    }
    for name, (threshold, digest) in expected_models.items():
        record = run["comparison_models"][name]
        threshold_key = "binary_benchmark_threshold" if name == "E5" else "threshold"
        if not np.isclose(float(record[threshold_key]), threshold, rtol=0.0, atol=1e-12):
            raise ValueError(f"Frozen {name} threshold changed")
        if record["checkpoint_sha256"] != digest:
            raise ValueError(f"Frozen {name} checkpoint identity changed")
    e5 = run["comparison_models"]["E5"]
    for key, expected in {
        "selected_anchor_weight": 0.01,
        "selected_epoch": 40,
        "real_threshold": 0.23700000000000002,
        "ai_threshold": 0.8170000000000001,
    }.items():
        if not np.isclose(float(e5[key]), expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"Frozen E5 selection changed: {key}")

    guardrails = run["single_use_guardrails"]
    for key in (
        "checkpoint_loaded_or_scores_computed_at_freeze",
        "evaluation_outputs_present_at_freeze",
        "retraining_after_results_allowed",
        "threshold_changes_after_results_allowed",
        "model_reselection_after_results_allowed",
        "dataset_or_condition_removal_after_results_allowed",
        "organiser_validation_subset_used",
    ):
        if guardrails[key] is not False:
            raise ValueError(f"Single-use guardrail changed: {key}")


def triage_class_metrics(
    probabilities: np.ndarray,
    *,
    true_label: int,
    real_threshold: float,
    ai_threshold: float,
    confidence_level: float,
) -> dict[str, Any]:
    """Return risk/coverage statistics for one true class."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not bool(np.isfinite(values).all()):
        raise ValueError("Triage probabilities must be a finite non-empty vector")
    if true_label not in (0, 1) or not 0.0 <= real_threshold < ai_threshold <= 1.0:
        raise ValueError("Invalid triage class or thresholds")
    called_real = int(np.count_nonzero(values <= real_threshold))
    called_ai = int(np.count_nonzero(values >= ai_threshold))
    uncertain = int(len(values) - called_real - called_ai)
    decisive = called_real + called_ai
    correct = called_real if true_label == 0 else called_ai
    errors = called_ai if true_label == 0 else called_real
    lower, upper = wilson_interval(errors, len(values), confidence_level)
    return {
        "samples": int(len(values)),
        "called_real": called_real,
        "uncertain": uncertain,
        "called_ai_generated": called_ai,
        "decisive": decisive,
        "decisive_coverage": float(decisive / len(values)),
        "uncertain_rate": float(uncertain / len(values)),
        "decisive_accuracy": float(correct / decisive) if decisive else None,
        "confident_errors": errors,
        "confident_error_rate": float(errors / len(values)),
        "confident_error_wilson_interval": [float(lower), float(upper)],
        "confident_error_wilson_upper": float(upper),
    }


def evaluate_frozen_gates(
    by_condition: dict[str, Any],
    *,
    primary_conditions: tuple[str, ...],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate all predeclared E5 gates without access to stress-only selection."""

    groups = [
        by_condition[condition]["triage_by_class"][CLASS_NAMES[label]]
        for condition in primary_conditions
        for label in (0, 1)
    ]
    real_uppers = [
        by_condition[c]["triage_by_class"][CLASS_NAMES[0]][
            "confident_error_wilson_upper"
        ]
        for c in primary_conditions
    ]
    ai_uppers = [
        by_condition[c]["triage_by_class"][CLASS_NAMES[1]][
            "confident_error_wilson_upper"
        ]
        for c in primary_conditions
    ]
    clean_coverages = {
        CLASS_NAMES[label]: by_condition["clean"]["triage_by_class"][CLASS_NAMES[label]][
            "decisive_coverage"
        ]
        for label in (0, 1)
    }
    coverages = [float(group["decisive_coverage"]) for group in groups]
    aucs = [float(by_condition[c]["binary_metrics"]["roc_auc"]) for c in primary_conditions]
    observed = {
        "worst_real_called_ai_wilson_upper": float(max(real_uppers)),
        "worst_ai_called_real_wilson_upper": float(max(ai_uppers)),
        "clean_decisive_coverage_by_class": clean_coverages,
        "minimum_clean_decisive_coverage": float(min(clean_coverages.values())),
        "worst_primary_class_condition_decisive_coverage": float(min(coverages)),
        "mean_primary_class_condition_decisive_coverage": float(np.mean(coverages)),
        "worst_primary_condition_roc_auc": float(min(aucs)),
    }
    checks = {
        "real_called_ai_risk": observed["worst_real_called_ai_wilson_upper"]
        <= float(criteria["maximum_wilson_upper_real_called_ai"]),
        "ai_called_real_risk": observed["worst_ai_called_real_wilson_upper"]
        <= float(criteria["maximum_wilson_upper_ai_called_real"]),
        "clean_decisive_coverage": observed["minimum_clean_decisive_coverage"]
        >= float(criteria["minimum_clean_decisive_coverage_per_class"]),
        "worst_primary_decisive_coverage": observed[
            "worst_primary_class_condition_decisive_coverage"
        ]
        >= float(criteria["minimum_worst_primary_class_condition_decisive_coverage"]),
        "mean_primary_decisive_coverage": observed[
            "mean_primary_class_condition_decisive_coverage"
        ]
        >= float(criteria["minimum_mean_primary_class_condition_decisive_coverage"]),
        "worst_primary_roc_auc": observed["worst_primary_condition_roc_auc"]
        >= float(criteria["minimum_worst_primary_condition_roc_auc"]),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "all_criteria_required": True,
        "checks": checks,
        "observed": observed,
        "criteria": criteria,
        "stress_conditions_affect_pass_decision": False,
    }


def _validate_feature_inputs(
    root: Path, run: dict[str, Any]
) -> tuple[FeatureCache, dict[str, np.ndarray], dict[str, Any]]:
    """Prove all feature identities and alignment before any checkpoint is loaded."""

    dependencies = run["protocol_dependencies"]
    base = _load_verified_json(root, dependencies["base_protocol"], "base protocol")
    validate_e5_aigibench_protocol(base)
    base_lock = _load_verified_json(root, dependencies["base_lock"], "base lock")
    if base_lock["status"] != "PASS":
        raise ValueError("Base AIGIBench lock did not pass")
    amendment = _load_verified_json(
        root, dependencies["deduplication_amendment"], "deduplication amendment"
    )
    validate_amendment(amendment, root)
    amendment_lock = _load_verified_json(
        root, dependencies["amendment_lock"], "deduplication amendment lock"
    )
    if amendment_lock["status"] != "PASS":
        raise ValueError("Deduplication amendment lock did not pass")

    frozen = run["frozen_external_data"]
    manifest = root / frozen["manifest"]["path"]
    provenance = _load_verified_json(root, frozen["provenance"], "external provenance")
    if not manifest.is_file() or sha256_file(manifest) != frozen["manifest"]["sha256"]:
        raise ValueError("Frozen external manifest changed or is missing")
    if provenance["selection"]["final_selected_member_list_sha256"] != frozen[
        "final_selected_member_list_sha256"
    ]:
        raise ValueError("Final deduplicated member selection changed")
    if provenance["counts"]["development_or_prior_audit_overlap"] != 0:
        raise ValueError("External set overlaps development or prior audit data")
    if provenance["dataset"]["organiser_validation_subset_used"] is not False:
        raise ValueError("Organiser validation data was used")

    clean_summary = _load_verified_json(root, frozen["clean_summary"], "clean summary")
    transformed_summary = _load_verified_json(
        root, frozen["transformed_summary"], "transformed summary"
    )
    for summary_name, summary in (
        ("clean", clean_summary),
        ("transformed", transformed_summary),
    ):
        guardrails = summary["frozen_guardrails"]
        for key in (
            "e5_checkpoint_loaded",
            "classifier_training_performed",
            "threshold_selection_performed",
            "model_selection_performed",
            "predictions_or_metrics_computed",
            "organiser_validation_subset_used",
        ):
            if guardrails[key] is not False:
                raise ValueError(f"{summary_name.title()} feature guardrail failed: {key}")

    clean_path = root / frozen["clean_cache"]["path"]
    clean_digest = sha256_file(clean_path)
    if clean_digest != frozen["clean_cache"]["sha256"]:
        raise ValueError("Frozen clean feature cache changed")
    if clean_summary["feature_cache"]["cache_sha256"] != clean_digest:
        raise ValueError("Clean cache and clean summary differ")
    clean = load_feature_cache(clean_path, "external_test", require_both_labels=True)
    if clean.features.shape != (2000, 512):
        raise ValueError("Clean feature shape changed")
    if np.bincount(clean.labels, minlength=2).tolist() != [1000, 1000]:
        raise ValueError("Clean external labels are not 1,000 per class")
    if clean.manifest_sha256 != frozen["manifest"]["sha256"]:
        raise ValueError("Clean features refer to a different manifest")

    conditions = tuple(run["conditions"]["full_matrix"])
    if transformed_summary["full_matrix"] != list(conditions):
        raise ValueError("Transformed summary matrix changed")
    if int(transformed_summary["transform_seed"]) != int(frozen["transform_seed"]):
        raise ValueError("Transformed summary seed changed")
    features = {"clean": clean.features}
    cache_hashes = {"clean": clean_digest}
    expected_non_clean = tuple(c for c in conditions if c != "clean")
    if tuple(transformed_summary["transformed_conditions"]) != expected_non_clean:
        raise ValueError("Transformed condition order changed")
    if set(transformed_summary["caches"]) != set(expected_non_clean):
        raise ValueError("Transformed cache set is incomplete")
    root_path = root / frozen["transformed_cache_root"]
    for condition in expected_non_clean:
        path = root_path / f"{condition}.npz"
        expected_digest = transformed_summary["caches"][condition]["cache_sha256"]
        observed_digest = sha256_file(path)
        if observed_digest != expected_digest:
            raise ValueError(f"Frozen transformed cache changed: {condition}")
        cache = load_transformed_feature_cache(
            path,
            expected_split="external_test",
            expected_condition=condition,
            expected_seed=int(frozen["transform_seed"]),
            expected_clean_cache_sha256=clean_digest,
            reference=clean,
            require_both_labels=True,
        )
        if cache.features.shape != (2000, 512):
            raise ValueError(f"Transformed feature shape changed: {condition}")
        features[condition] = cache.features
        cache_hashes[condition] = observed_digest
    return clean, features, {
        "manifest_sha256": sha256_file(manifest),
        "provenance_sha256": frozen["provenance"]["sha256"],
        "clean_summary_sha256": frozen["clean_summary"]["sha256"],
        "transformed_summary_sha256": frozen["transformed_summary"]["sha256"],
        "cache_sha256_by_condition": cache_hashes,
    }


def _validate_checkpoint(path: Path, record: dict[str, Any], name: str) -> Any:
    if sha256_file(path) != record["checkpoint_sha256"]:
        raise ValueError(f"Frozen {name} checkpoint changed")
    checkpoint = load_linear_probe_checkpoint(path)
    expected_threshold = (
        record["binary_benchmark_threshold"] if name == "E5" else record["threshold"]
    )
    if not np.isclose(checkpoint.threshold, expected_threshold, rtol=0.0, atol=1e-12):
        raise ValueError(f"Frozen {name} checkpoint threshold changed")
    with np.load(path, allow_pickle=False) as archive:
        kind = str(_scalar(archive, "robust_checkpoint_kind"))
        epoch = int(_scalar(archive, "selected_best_epoch"))
        if name == "E3" and kind != "section3_robust_linear_head_v1":
            raise ValueError("Unexpected E3 checkpoint kind")
        if name == "E4" and kind != "e4_domain_adapted_linear_head_v1":
            raise ValueError("Unexpected E4 checkpoint kind")
        if name == "E5":
            if kind != record["checkpoint_kind"]:
                raise ValueError("Unexpected E5 checkpoint kind")
            for key in ("real_threshold", "ai_threshold", "anchor_weight"):
                expected = record[
                    "selected_anchor_weight" if key == "anchor_weight" else key
                ]
                if not np.isclose(
                    float(_scalar(archive, key)), float(expected), rtol=0.0, atol=1e-12
                ):
                    raise ValueError(f"E5 checkpoint metadata changed: {key}")
            if epoch != int(record["selected_epoch"]):
                raise ValueError("E5 selected epoch changed")
    return checkpoint


def _score_models(
    run: dict[str, Any],
    labels: np.ndarray,
    features: dict[str, np.ndarray],
    root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    checkpoints = {}
    checkpoint_records = {}
    for name in MODEL_NAMES:
        record = run["comparison_models"][name]
        path = root / record["checkpoint"]
        checkpoints[name] = _validate_checkpoint(path, record, name)
        checkpoint_records[name] = {
            "path": record["checkpoint"],
            "sha256": record["checkpoint_sha256"],
            "threshold": float(
                record["binary_benchmark_threshold"] if name == "E5" else record["threshold"]
            ),
            "role": record["role"],
        }

    conditions = tuple(run["conditions"]["full_matrix"])
    probabilities: dict[str, dict[str, np.ndarray]] = {name: {} for name in MODEL_NAMES}
    comparison: dict[str, Any] = {}
    for name in MODEL_NAMES:
        threshold = checkpoint_records[name]["threshold"]
        rows = {}
        for condition in conditions:
            scores = checkpoints[name].probabilities(features[condition])
            probabilities[name][condition] = scores
            metrics = binary_classification_metrics(labels, scores, threshold=threshold)
            rows[condition] = {"metrics": metrics, "class_rates": class_rates(metrics)}
        balanced = [rows[c]["metrics"]["balanced_accuracy"] for c in conditions]
        aucs = [rows[c]["metrics"]["roc_auc"] for c in conditions]
        comparison[name] = {
            **checkpoint_records[name],
            "by_condition": rows,
            "summary": {
                "clean_balanced_accuracy": rows["clean"]["metrics"]["balanced_accuracy"],
                "mean_full_matrix_balanced_accuracy": float(np.mean(balanced)),
                "worst_full_matrix_balanced_accuracy": float(np.min(balanced)),
                "mean_full_matrix_roc_auc": float(np.mean(aucs)),
                "worst_full_matrix_roc_auc": float(np.min(aucs)),
            },
        }
    return comparison, probabilities, checkpoint_records


def _e5_results(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    run: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = run["comparison_models"]["E5"]
    real_threshold = float(record["real_threshold"])
    ai_threshold = float(record["ai_threshold"])
    binary_threshold = float(record["binary_benchmark_threshold"])
    confidence = float(run["frozen_success_criteria"]["confidence_level"])
    by_condition: dict[str, Any] = {}
    for condition in run["conditions"]["full_matrix"]:
        scores = probabilities[condition]
        triage = {
            CLASS_NAMES[label]: triage_class_metrics(
                scores[labels == label],
                true_label=label,
                real_threshold=real_threshold,
                ai_threshold=ai_threshold,
                confidence_level=confidence,
            )
            for label in (0, 1)
        }
        decisive = sum(row["decisive"] for row in triage.values())
        correct = triage[CLASS_NAMES[0]]["called_real"] + triage[CLASS_NAMES[1]][
            "called_ai_generated"
        ]
        by_condition[condition] = {
            "condition_role": (
                "primary_claim"
                if condition in run["conditions"]["primary_claim_conditions"]
                else "mandatory_stress_only"
            ),
            "binary_metrics": binary_classification_metrics(
                labels, scores, threshold=binary_threshold
            ),
            "triage_by_class": triage,
            "triage_aggregate": {
                "samples": int(len(labels)),
                "decisive": int(decisive),
                "uncertain": int(len(labels) - decisive),
                "decisive_coverage": float(decisive / len(labels)),
                "uncertain_rate": float((len(labels) - decisive) / len(labels)),
                "decisive_accuracy": float(correct / decisive) if decisive else None,
            },
        }
    gates = evaluate_frozen_gates(
        by_condition,
        primary_conditions=tuple(run["conditions"]["primary_claim_conditions"]),
        criteria=run["frozen_success_criteria"],
    )
    return by_condition, gates


def _write_probability_output(
    path: Path,
    *,
    image_paths: np.ndarray,
    labels: np.ndarray,
    probabilities: dict[str, dict[str, np.ndarray]],
) -> None:
    arrays: dict[str, np.ndarray] = {"image_paths": image_paths, "labels": labels}
    for model, rows in probabilities.items():
        for condition, values in rows.items():
            arrays[f"{model}_{condition}"] = np.asarray(values, dtype=np.float64)
    np.savez(path, **arrays)


def _write_robustness_figure(by_condition: dict[str, Any], path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = tuple(by_condition)
    positions = np.arange(len(conditions))
    figure, axis = plt.subplots(figsize=(17, 6))
    axis.plot(
        positions,
        [by_condition[c]["binary_metrics"]["roc_auc"] for c in conditions],
        marker="o",
        linewidth=2,
        label="ROC-AUC",
        color="#2563eb",
    )
    axis.plot(
        positions,
        [by_condition[c]["binary_metrics"]["balanced_accuracy"] for c in conditions],
        marker="o",
        linewidth=2,
        label="Balanced accuracy at 0.520",
        color="#dc2626",
    )
    axis.set(
        ylabel="Metric value",
        ylim=(0.0, 1.02),
        xticks=positions,
        xticklabels=[TRANSFORM_SPECS[c].display_name for c in conditions],
        title="E5 on fresh AIGIBench Open Images V7 / Midjourney V6",
    )
    axis.tick_params(axis="x", rotation=35)
    for label in axis.get_xticklabels():
        label.set_ha("right")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_triage_figure(by_condition: dict[str, Any], path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = tuple(by_condition)
    columns = (
        (CLASS_NAMES[0], "called_real", "Real → real"),
        (CLASS_NAMES[0], "uncertain", "Real → uncertain"),
        (CLASS_NAMES[0], "called_ai_generated", "Real → AI"),
        (CLASS_NAMES[1], "called_real", "AI → real"),
        (CLASS_NAMES[1], "uncertain", "AI → uncertain"),
        (CLASS_NAMES[1], "called_ai_generated", "AI → AI"),
    )
    matrix = np.asarray(
        [
            [by_condition[c]["triage_by_class"][group][key] / 1000.0 for group, key, _ in columns]
            for c in conditions
        ]
    )
    figure, axis = plt.subplots(figsize=(11, 10))
    image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=8,
            )
    axis.set(
        xticks=np.arange(len(columns)),
        xticklabels=[column[2] for column in columns],
        yticks=np.arange(len(conditions)),
        yticklabels=[TRANSFORM_SPECS[c].display_name for c in conditions],
        title="E5 three-way outcomes by true class and transformation",
    )
    axis.tick_params(axis="x", rotation=30)
    for label in axis.get_xticklabels():
        label.set_ha("right")
    figure.colorbar(image, ax=axis, label="Fraction of true-class images")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_distribution_figure(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    real_threshold: float,
    ai_threshold: float,
    path: Path,
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5.8))
    bins = np.linspace(0.0, 1.0, 51)
    axis.hist(probabilities[labels == 0], bins=bins, alpha=0.65, label="Open Images V7 real", color="#2563eb")
    axis.hist(probabilities[labels == 1], bins=bins, alpha=0.65, label="Midjourney V6", color="#dc2626")
    axis.axvspan(real_threshold, ai_threshold, alpha=0.13, color="#f59e0b", label="Uncertain region")
    axis.axvline(real_threshold, color="#0f172a", linestyle="--", linewidth=1.8)
    axis.axvline(ai_threshold, color="#0f172a", linestyle="--", linewidth=1.8)
    axis.set(
        xlim=(0.0, 1.0),
        xlabel="E5 probability of AI generation",
        ylabel="Images",
        title="Clean external E5 score distributions and frozen triage thresholds",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_comparison_figure(comparison: dict[str, Any], path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("clean_balanced_accuracy", "Clean BA"),
        ("mean_full_matrix_balanced_accuracy", "Mean matrix BA"),
        ("worst_full_matrix_balanced_accuracy", "Worst matrix BA"),
        ("worst_full_matrix_roc_auc", "Worst matrix AUC"),
    )
    positions = np.arange(len(metrics))
    width = 0.24
    colors = {"E3": "#64748b", "E4": "#f59e0b", "E5": "#2563eb"}
    figure, axis = plt.subplots(figsize=(11, 6))
    for index, model in enumerate(MODEL_NAMES):
        values = [comparison[model]["summary"][key] for key, _ in metrics]
        offset = (index - 1) * width
        axis.bar(positions + offset, values, width, label=model, color=colors[model])
        for position, value in zip(positions + offset, values, strict=True):
            axis.text(position, value + 0.012, f"{value:.3f}", ha="center", fontsize=8)
    axis.set(
        xticks=positions,
        xticklabels=[label for _, label in metrics],
        ylim=(0.0, 1.08),
        ylabel="Metric value",
        title="Frozen binary comparison on identical fresh external features",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-lock", type=Path, default=DEFAULT_RUN_LOCK)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_lock_path = args.run_lock if args.run_lock.is_absolute() else root / args.run_lock
    run = json.loads(run_lock_path.read_text(encoding="utf-8"))
    validate_run_lock(run)
    outputs = {key: root / value for key, value in run["outputs"].items()}
    if outputs["report"].exists() or outputs["probabilities"].exists():
        raise ValueError(
            "Single-use AIGIBench evaluation output already exists; refusing to rescore"
        )

    print("Validating every frozen data artifact before loading any checkpoint", flush=True)
    clean, features, data_identity = _validate_feature_inputs(root, run)
    print(
        "PASS data preflight: samples=2000, conditions=15, aligned=True; "
        "loading frozen E3/E4/E5 heads",
        flush=True,
    )
    comparison, probabilities, checkpoint_records = _score_models(
        run, clean.labels, features, root
    )
    by_condition, gates = _e5_results(clean.labels, probabilities["E5"], run)

    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temporary_directory:
        temporary = Path(temporary_directory)
        temporary_outputs = {
            key: temporary / Path(value).name for key, value in run["outputs"].items()
        }
        _write_probability_output(
            temporary_outputs["probabilities"],
            image_paths=clean.image_paths,
            labels=clean.labels,
            probabilities=probabilities,
        )
        _write_robustness_figure(by_condition, temporary_outputs["robustness_figure"])
        _write_triage_figure(by_condition, temporary_outputs["triage_figure"])
        e5_record = run["comparison_models"]["E5"]
        _write_distribution_figure(
            clean.labels,
            probabilities["E5"]["clean"],
            real_threshold=float(e5_record["real_threshold"]),
            ai_threshold=float(e5_record["ai_threshold"]),
            path=temporary_outputs["distribution_figure"],
        )
        _write_comparison_figure(comparison, temporary_outputs["comparison_figure"])
        for key in (
            "probabilities",
            "robustness_figure",
            "triage_figure",
            "distribution_figure",
            "comparison_figure",
        ):
            temporary_outputs[key].replace(outputs[key])

    report = {
        "experiment": "e5_aigibench_midjourney_single_use_external_evaluation",
        "status": gates["status"],
        "interpretation": (
            "E5 passed every predeclared external risk, coverage, and ranking gate."
            if gates["status"] == "PASS"
            else "E5 is not externally validated on this one-time benchmark because one or more predeclared gates failed."
        ),
        "claim_limit": run["research_claim_limit"],
        "run_lock": {
            "path": run_lock_path.relative_to(root).as_posix(),
            "sha256": sha256_file(run_lock_path),
            "frozen_at_utc": run["frozen_at_utc"],
        },
        "dataset": {
            "benchmark": "AIGIBench Midjourney V6 test subset",
            "real_source": "Open Images V7",
            "ai_generator": "Midjourney V6",
            "samples": 2000,
            "class_counts": run["frozen_external_data"]["class_counts"],
            "license": "CC-BY-NC-SA-4.0",
            "organiser_validation_subset_used": False,
        },
        "data_identity": data_identity,
        "models": checkpoint_records,
        "e5_decision_rule": {
            "real": f"score <= {run['comparison_models']['E5']['real_threshold']:.3f}",
            "uncertain": (
                f"{run['comparison_models']['E5']['real_threshold']:.3f} < score < "
                f"{run['comparison_models']['E5']['ai_threshold']:.3f}"
            ),
            "ai_generated": f"score >= {run['comparison_models']['E5']['ai_threshold']:.3f}",
            "binary_benchmark_threshold": run["comparison_models"]["E5"][
                "binary_benchmark_threshold"
            ],
        },
        "frozen_gate_decision": gates,
        "e5_by_condition": by_condition,
        "binary_model_comparison": comparison,
        "outputs": {
            key: {
                "path": run["outputs"][key],
                "sha256": sha256_file(outputs[key]),
                "committable": key != "probabilities",
            }
            for key in outputs
            if key != "report"
        },
        "post_result_guardrails": {
            "retraining_allowed": False,
            "threshold_changes_allowed": False,
            "model_reselection_allowed": False,
            "dataset_or_condition_removal_allowed": False,
            "stress_conditions_included_in_report": True,
        },
    }
    atomic_json_write(outputs["report"], report)
    clean_metrics = by_condition["clean"]["binary_metrics"]
    print(
        f"{gates['status']} E5 external audit: clean_auc={clean_metrics['roc_auc']:.6f}, "
        f"clean_ba={clean_metrics['balanced_accuracy']:.6f}, "
        f"worst_primary_auc={gates['observed']['worst_primary_condition_roc_auc']:.6f}",
        flush=True,
    )
    print(f"Updated report: {run['outputs']['report']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"E5 AIGIBench external evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
