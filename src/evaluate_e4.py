"""Frozen post-hoc E3-versus-E4 evaluation on cached Section 4 features."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.extract_clip_features import atomic_json_write
from scripts.prepare_e4_sid_real import validate_protocol as validate_e4_protocol
from scripts.prepare_sid_set_heldout import validate_protocol as validate_sid_protocol
from src.evaluate_full_matrix import add_predeclared_summaries, validate_section4_protocol
from src.evaluate_heldout_generator import class_rates, probability_distribution_summary
from src.evaluate_section3 import (
    _configure_matplotlib,
    _probabilities_by_condition,
    comparison_deltas,
    evaluate_model_conditions,
)
from src.image_transforms import FULL_ROBUSTNESS_CONDITIONS, TRANSFORM_SPECS
from src.linear_probe import load_feature_cache, load_linear_probe_checkpoint
from src.metrics import binary_classification_metrics
from src.robust_linear_training import load_paired_feature_set, sha256_file


DEFAULT_PROTOCOL = Path("configs/e4_posthoc_evaluation.json")
MODEL_NAMES = ("E3", "E4")
E4_CHECKPOINT_KIND = "e4_domain_adapted_linear_head_v1"


def _scalar(archive: np.lib.npyio.NpzFile, key: str) -> object:
    if key not in archive.files or archive[key].ndim != 0:
        raise ValueError(f"E4 checkpoint metadata {key!r} is missing or non-scalar")
    return archive[key].item()


def validate_evaluation_protocol(protocol: dict[str, Any]) -> None:
    """Reject post-training evaluation drift before any E4 test scores are read."""

    if protocol["status"] != "frozen_after_e4_training_before_e4_test_evaluation":
        raise ValueError("E4 evaluation protocol was not frozen at the correct stage")
    positioning = protocol["positioning"]
    required_positioning = {
        "E3_remains_original_primary_model": True,
        "E4_is_post_hoc_follow_up": True,
        "E4_results_are_design_independent_fresh_test": False,
        "fresh_external_test_still_required": True,
    }
    for key, expected in required_positioning.items():
        if positioning[key] is not expected:
            raise ValueError(f"E4 research positioning changed: {key}")

    rules = protocol["frozen_rules"]
    for key in (
        "retraining_allowed",
        "threshold_changes_allowed",
        "epoch_reselection_allowed",
        "condition_selection_after_results_allowed",
        "overwrite_original_section4_artifacts_allowed",
    ):
        if rules[key] is not False:
            raise ValueError(f"Frozen E4 evaluation rule changed: {key}")
    if rules["same_paths_labels_and_features_for_E3_and_E4"] is not True:
        raise ValueError("E3 and E4 must use identical cached examples")
    if rules["organiser_validation_subset_used"] is not False:
        raise ValueError("The organiser validation subset must remain untouched")

    if tuple(protocol["evaluation"]["cifake_conditions"]) != FULL_ROBUSTNESS_CONDITIONS:
        raise ValueError("E4 CIFAKE evaluation must cover the complete frozen matrix")
    if protocol["evaluation"]["comparison_direction"] != "E4 minus E3":
        raise ValueError("E4 comparison direction changed")
    inputs = protocol["frozen_inputs"]
    if not np.isclose(inputs["E3"]["threshold"], 0.437, rtol=0.0, atol=1e-12):
        raise ValueError("Frozen E3 threshold changed")
    if not 0.0 < float(inputs["E4"]["threshold"]) < 1.0:
        raise ValueError("Frozen E4 threshold is invalid")

    outputs = protocol["outputs"]
    original_paths = {
        inputs["cifake_full_matrix"]["original_report"],
        inputs["sid_flux_audit"]["original_report"],
    }
    if any(path in original_paths for path in outputs.values()):
        raise ValueError("E4 outputs may not overwrite original Section 4 artifacts")


def _verify_hash(path: Path, expected: str, description: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"Frozen {description} changed: {path}")


def validate_e4_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_threshold: float,
    expected_protocol_sha256: str,
    expected_epoch: int,
) -> Any:
    """Load E4 only after proving its validation-selected provenance."""

    _verify_hash(path, expected_sha256, "E4 checkpoint")
    checkpoint = load_linear_probe_checkpoint(path)
    with np.load(path, allow_pickle=False) as archive:
        kind = str(_scalar(archive, "robust_checkpoint_kind"))
        experiment = str(_scalar(archive, "experiment"))
        protocol_sha256 = str(_scalar(archive, "protocol_sha256"))
        selection_kind = str(_scalar(archive, "threshold_selection_kind"))
        selected_epoch = int(_scalar(archive, "selected_best_epoch"))
        validation = json.loads(str(_scalar(archive, "selected_validation_json")))
    if kind != E4_CHECKPOINT_KIND or experiment != "E4_sid_real_domain_adaptation":
        raise ValueError("Unexpected E4 checkpoint type")
    if protocol_sha256 != expected_protocol_sha256:
        raise ValueError("E4 checkpoint refers to a different development protocol")
    if selection_kind != "validation_sid_real_fpr_constrained":
        raise ValueError("E4 threshold was not selected by the frozen validation rule")
    if selected_epoch != expected_epoch:
        raise ValueError("E4 checkpoint epoch changed after training")
    if validation["constraint_satisfied"] is not True or validation["fallback_used"] is not False:
        raise ValueError("E4 checkpoint did not satisfy its validation constraint")
    if not np.isclose(checkpoint.threshold, expected_threshold, rtol=0.0, atol=1e-12):
        raise ValueError("E4 checkpoint threshold changed")
    return checkpoint


def assert_metrics_reproduced(
    expected: dict[str, Any], observed: dict[str, Any], *, context: str
) -> None:
    """Prove that recalculated E3 metrics reproduce the already frozen report."""

    for key in (
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "brier_score",
    ):
        if not np.isclose(float(expected[key]), float(observed[key]), rtol=0.0, atol=1e-12):
            raise ValueError(f"Frozen E3 metric did not reproduce for {context}: {key}")
    if expected["confusion_matrix"] != observed["confusion_matrix"]:
        raise ValueError(f"Frozen E3 confusion matrix did not reproduce for {context}")


def heldout_comparison_deltas(
    baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any]
) -> dict[str, float]:
    """Return E4-minus-E3 audit changes, including both class-specific rates."""

    baseline_rates = class_rates(baseline_metrics)
    candidate_rates = class_rates(candidate_metrics)
    deltas = {
        key: float(candidate_metrics[key] - baseline_metrics[key])
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
    deltas.update(
        {key: float(candidate_rates[key] - baseline_rates[key]) for key in baseline_rates}
    )
    return deltas


def _atomic_probability_write(
    path: Path,
    *,
    cifake_paths: np.ndarray,
    cifake_labels: np.ndarray,
    cifake_probabilities: dict[str, dict[str, np.ndarray]],
    audit_paths: np.ndarray,
    audit_labels: np.ndarray,
    audit_probabilities: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "cifake_image_paths": cifake_paths,
        "cifake_labels": cifake_labels,
        "audit_image_paths": audit_paths,
        "audit_labels": audit_labels,
    }
    for model, rows in cifake_probabilities.items():
        for condition, values in rows.items():
            arrays[f"cifake_{model}_{condition}"] = np.asarray(values, dtype=np.float64)
    for model, values in audit_probabilities.items():
        arrays[f"audit_{model}"] = np.asarray(values, dtype=np.float64)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        np.savez(temporary_path, **arrays)
        temporary_path.replace(path)


def _write_full_matrix_figure(
    results: dict[str, Any], conditions: tuple[str, ...], path: Path
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.arange(len(conditions))
    colors = {"E3": "#64748b", "E4": "#2563eb"}
    figure, axes = plt.subplots(2, 1, figsize=(17, 8), sharex=True)
    for model in MODEL_NAMES:
        axes[0].plot(
            positions,
            [results[model]["by_condition"][c]["metrics"]["balanced_accuracy"] for c in conditions],
            marker="o",
            linewidth=2,
            label=f"{model} (threshold {results[model]['threshold']:.3f})",
            color=colors[model],
        )
        axes[1].plot(
            positions,
            [results[model]["by_condition"][c]["metrics"]["roc_auc"] for c in conditions],
            marker="o",
            linewidth=2,
            label=model,
            color=colors[model],
        )
    axes[0].set_ylabel("Balanced accuracy")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_xticks(
        positions,
        [TRANSFORM_SPECS[c].display_name for c in conditions],
        rotation=35,
        ha="right",
    )
    for axis in axes:
        axis.set_ylim(0.45, 1.01)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower left")
    figure.suptitle("Post-hoc E3 versus E4 — frozen CIFAKE transformation matrix", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_audit_figure(results: dict[str, Any], path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, model in zip(axes[:2], MODEL_NAMES, strict=True):
        matrix = np.asarray(results[model]["metrics"]["confusion_matrix"])
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
            title=f"{model} — threshold {results[model]['threshold']:.3f}",
        )
    metrics = ("false_positive_rate_real_called_ai", "true_positive_rate_ai_recall")
    labels = ("Real FPR", "FLUX recall")
    positions = np.arange(2)
    width = 0.34
    for index, model in enumerate(MODEL_NAMES):
        values = [results[model]["class_rates"][metric] for metric in metrics]
        axes[2].bar(positions + (index - 0.5) * width, values, width, label=model)
        for position, value in zip(positions + (index - 0.5) * width, values, strict=True):
            axes[2].text(position, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
    axes[2].set(xticks=positions, xticklabels=labels, ylim=(0, 1.08), title="Class-specific audit rates")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()
    figure.suptitle("Post-hoc E3 versus E4 — SID-Set real / FLUX audit", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_distribution_figure(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    path: Path,
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(0.0, 1.0, 41)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, model in zip(axes, MODEL_NAMES, strict=True):
        values = probabilities[model]
        axis.hist(values[labels == 0], bins=bins, alpha=0.65, label="SID-Set real", color="#2563eb")
        axis.hist(values[labels == 1], bins=bins, alpha=0.65, label="FLUX", color="#dc2626")
        axis.axvline(thresholds[model], color="black", linestyle="--", linewidth=2)
        axis.set(
            title=f"{model} — threshold {thresholds[model]:.3f}",
            xlabel="Predicted probability of AI generation",
            xlim=(0, 1),
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Images")
    figure.suptitle("Post-hoc SID-Set / FLUX score distributions", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen post-hoc E3-versus-E4 evaluation.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_evaluation_protocol(protocol)
    frozen = protocol["frozen_inputs"]

    for record, path_key, hash_key, description in (
        (frozen["e4_development_protocol"], "path", "sha256", "E4 development protocol"),
        (frozen["e4_training_report"], "path", "sha256", "E4 training report"),
        (frozen["cifake_full_matrix"], "protocol", "protocol_sha256", "Section 4A protocol"),
        (frozen["cifake_full_matrix"], "original_report", "original_report_sha256", "Section 4A report"),
        (frozen["sid_flux_audit"], "protocol", "protocol_sha256", "Section 4B protocol"),
        (frozen["sid_flux_audit"], "embedding_summary", "embedding_summary_sha256", "Section 4B embedding summary"),
        (frozen["sid_flux_audit"], "feature_cache", "feature_cache_sha256", "Section 4B feature cache"),
        (frozen["sid_flux_audit"], "original_report", "original_report_sha256", "Section 4B report"),
    ):
        _verify_hash(Path(record[path_key]), record[hash_key], description)

    e4_development_protocol = json.loads(Path(frozen["e4_development_protocol"]["path"]).read_text())
    validate_e4_protocol(e4_development_protocol)
    e4_training = json.loads(Path(frozen["e4_training_report"]["path"]).read_text())
    if e4_training["production_protocol_run"] is not True:
        raise ValueError("E4 checkpoint did not come from a production protocol run")
    if e4_training["data"]["test_or_audit_data_loaded"] is not False:
        raise ValueError("E4 training report indicates test/audit leakage")

    full_protocol_path = Path(frozen["cifake_full_matrix"]["protocol"])
    full_protocol = json.loads(full_protocol_path.read_text())
    validate_section4_protocol(full_protocol)
    original_full = json.loads(Path(frozen["cifake_full_matrix"]["original_report"]).read_text())
    sid_protocol = json.loads(Path(frozen["sid_flux_audit"]["protocol"]).read_text())
    validate_sid_protocol(sid_protocol)
    original_audit = json.loads(Path(frozen["sid_flux_audit"]["original_report"]).read_text())
    embedding_summary = json.loads(Path(frozen["sid_flux_audit"]["embedding_summary"]).read_text())

    checkpoints = {
        "E3": load_linear_probe_checkpoint(Path(frozen["E3"]["checkpoint"])),
        "E4": validate_e4_checkpoint(
            Path(frozen["E4"]["checkpoint"]),
            expected_sha256=frozen["E4"]["checkpoint_sha256"],
            expected_threshold=float(frozen["E4"]["threshold"]),
            expected_protocol_sha256=frozen["e4_development_protocol"]["sha256"],
            expected_epoch=int(e4_training["model_and_threshold_selection"]["best_epoch"]),
        ),
    }
    _verify_hash(Path(frozen["E3"]["checkpoint"]), frozen["E3"]["checkpoint_sha256"], "E3 checkpoint")
    for name in MODEL_NAMES:
        if not np.isclose(checkpoints[name].threshold, frozen[name]["threshold"], rtol=0.0, atol=1e-12):
            raise ValueError(f"Frozen {name} threshold changed")

    conditions = tuple(full_protocol["full_matrix_conditions"][1:])
    cifake = load_paired_feature_set(
        split=full_protocol["data"]["split"],
        clean_cache_path=Path(full_protocol["data"]["clean_feature_cache"]),
        transformed_feature_dir=Path(full_protocol["data"]["transformed_feature_directory"]),
        conditions=conditions,
        seed=int(full_protocol["transform_seed"]),
    )
    if cifake.samples != int(full_protocol["data"]["samples"]):
        raise ValueError("Frozen CIFAKE sample count changed")
    audit = load_feature_cache(Path(frozen["sid_flux_audit"]["feature_cache"]), "heldout")
    if audit.manifest_sha256 != embedding_summary["preparation_validation"]["manifest_sha256"]:
        raise ValueError("SID/FLUX cache no longer matches its frozen manifest")

    print(
        f"Frozen E4 post-hoc evaluation: CIFAKE={cifake.samples}x{len(conditions) + 1}, "
        f"SID/FLUX={len(audit.labels)}, models=E3/E4",
        flush=True,
    )
    full_results: dict[str, Any] = {}
    full_probabilities: dict[str, dict[str, np.ndarray]] = {}
    audit_results: dict[str, Any] = {}
    audit_probabilities: dict[str, np.ndarray] = {}
    for model in MODEL_NAMES:
        condition_probabilities = _probabilities_by_condition(cifake, checkpoints[model])
        result = evaluate_model_conditions(cifake.clean.labels, condition_probabilities, checkpoints[model].threshold)
        result["predeclared_summaries"] = add_predeclared_summaries(result, full_protocol)
        full_results[model] = result
        full_probabilities[model] = condition_probabilities

        heldout_probabilities = checkpoints[model].probabilities(audit.features)
        metrics = binary_classification_metrics(audit.labels, heldout_probabilities, checkpoints[model].threshold)
        metrics["samples"] = len(audit.labels)
        audit_results[model] = {
            "threshold": checkpoints[model].threshold,
            "metrics": metrics,
            "class_rates": class_rates(metrics),
            "probability_distributions": probability_distribution_summary(audit.labels, heldout_probabilities),
        }
        audit_probabilities[model] = heldout_probabilities

    for condition in FULL_ROBUSTNESS_CONDITIONS:
        assert_metrics_reproduced(
            original_full["models"]["E3"]["by_condition"][condition]["metrics"],
            full_results["E3"]["by_condition"][condition]["metrics"],
            context=f"CIFAKE/{condition}",
        )
    assert_metrics_reproduced(
        original_audit["models"]["E3"]["metrics"],
        audit_results["E3"]["metrics"],
        context="SID-Set/FLUX audit",
    )

    full_delta = comparison_deltas(full_results["E3"], full_results["E4"])
    audit_delta = heldout_comparison_deltas(
        audit_results["E3"]["metrics"], audit_results["E4"]["metrics"]
    )
    outputs = protocol["outputs"]
    probability_path = Path(outputs["probabilities"])
    _atomic_probability_write(
        probability_path,
        cifake_paths=cifake.clean.image_paths,
        cifake_labels=cifake.clean.labels,
        cifake_probabilities=full_probabilities,
        audit_paths=audit.image_paths,
        audit_labels=audit.labels,
        audit_probabilities=audit_probabilities,
    )
    full_figure = Path(outputs["full_matrix_figure"])
    audit_figure = Path(outputs["audit_figure"])
    distribution_figure = Path(outputs["score_distribution_figure"])
    _write_full_matrix_figure(full_results, FULL_ROBUSTNESS_CONDITIONS, full_figure)
    _write_audit_figure(audit_results, audit_figure)
    _write_distribution_figure(
        audit.labels,
        audit_probabilities,
        {name: checkpoints[name].threshold for name in MODEL_NAMES},
        distribution_figure,
    )

    report = {
        "experiment": protocol["experiment_name"],
        "research_question": protocol["research_question"],
        "protocol": {"path": args.protocol.as_posix(), "sha256": sha256_file(args.protocol), "version": protocol["protocol_version"]},
        "pre_evaluation_freeze": {
            "status": protocol["status"],
            "rules": protocol["frozen_rules"],
            "models": {
                name: {
                    "checkpoint": frozen[name]["checkpoint"],
                    "checkpoint_sha256": frozen[name]["checkpoint_sha256"],
                    "threshold": float(checkpoints[name].threshold),
                }
                for name in MODEL_NAMES
            },
        },
        "cifake_full_matrix": {
            "post_hoc_not_fresh": True,
            "samples": cifake.samples,
            "conditions": list(FULL_ROBUSTNESS_CONDITIONS),
            "E3_reproduction_verified": True,
            "models": full_results,
            "E4_minus_E3": full_delta,
        },
        "sid_set_flux_audit": {
            "post_hoc_not_fresh": True,
            "samples": len(audit.labels),
            "class_counts": {
                "real_0": int(np.sum(audit.labels == 0)),
                "flux_ai_generated_1": int(np.sum(audit.labels == 1)),
            },
            "row_and_content_disjoint_from_E4_development": True,
            "E3_reproduction_verified": True,
            "models": audit_results,
            "E4_minus_E3": audit_delta,
        },
        "guardrails": {
            "organiser_validation_subset_used": False,
            "original_section4_artifacts_overwritten": False,
            "E3_remains_original_primary_model": True,
            "E4_is_post_hoc_follow_up": True,
            "fresh_external_test_still_required": True,
        },
        "artifacts": {
            "full_matrix_figure": full_figure.as_posix(),
            "full_matrix_figure_sha256": sha256_file(full_figure),
            "audit_figure": audit_figure.as_posix(),
            "audit_figure_sha256": sha256_file(audit_figure),
            "score_distribution_figure": distribution_figure.as_posix(),
            "score_distribution_figure_sha256": sha256_file(distribution_figure),
            "probabilities": probability_path.as_posix(),
            "probabilities_sha256": sha256_file(probability_path),
        },
        "interpretation_guardrails": [
            "E4 was designed after inspecting both evaluation domains, so these are post-hoc results rather than a fresh test.",
            "The frozen E4 threshold must not be changed from these results.",
            "A new generator and authentic-image source are required for an unbiased external claim.",
            "Aggregate metrics and non-image figures may be published; SID images may not be placed in the public demo.",
        ],
    }
    report_path = Path(outputs["report"])
    atomic_json_write(report_path, report)

    full_summary = full_results["E4"]["predeclared_summaries"]["all_14_transformed"]
    e4_rates = audit_results["E4"]["class_rates"]
    print(
        f"PASS E4 CIFAKE: clean_ba={full_results['E4']['clean']['metrics']['balanced_accuracy']:.6f}, "
        f"mean_transformed_ba={full_summary['mean_balanced_accuracy']:.6f}, "
        f"worst_ba={full_summary['worst_balanced_accuracy']:.6f} "
        f"({full_summary['worst_balanced_accuracy_condition']})",
        flush=True,
    )
    print(
        f"PASS E4 SID/FLUX: auc={audit_results['E4']['metrics']['roc_auc']:.6f}, "
        f"bal_acc={audit_results['E4']['metrics']['balanced_accuracy']:.6f}, "
        f"real_fpr={e4_rates['false_positive_rate_real_called_ai']:.6f}, "
        f"flux_recall={e4_rates['true_positive_rate_ai_recall']:.6f}",
        flush=True,
    )
    print(f"Updated post-hoc report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"E4 post-hoc evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
