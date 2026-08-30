"""Aggregate, privacy-safe error analysis for the frozen E3 detector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

from scripts.extract_clip_features import atomic_json_write
from src.evaluate_section3 import _configure_matplotlib
from src.image_transforms import FULL_ROBUSTNESS_CONDITIONS, TRANSFORM_SPECS
from src.robust_linear_training import sha256_file


DEFAULT_PROTOCOL = Path("configs/section4d_error_analysis.json")
DEFAULT_REPORT = Path("reports/section4d_error_analysis.json")
DEFAULT_MARKDOWN = Path("reports/section4d_error_analysis.md")
DEFAULT_DIRECTION_FIGURE = Path(
    "reports/figures/section4d_condition_error_directions.png"
)
DEFAULT_PERSISTENCE_FIGURE = Path(
    "reports/figures/section4d_error_persistence.png"
)
DEFAULT_COOCCURRENCE_FIGURE = Path(
    "reports/figures/section4d_error_cooccurrence.png"
)
DEFAULT_PROPERTY_FIGURE = Path(
    "reports/figures/section4d_sidset_real_property_profile.png"
)
MODEL_NAME = "E3"


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Reject privacy, model, threshold, or analysis-protocol drift."""

    model = protocol["model"]
    if model["name"] != MODEL_NAME or not np.isclose(
        float(model["frozen_threshold"]), 0.437, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Section 4D must analyse frozen E3 at threshold 0.437")
    privacy = protocol["privacy_and_licensing"]
    if privacy["sid_set_images_publicly_displayed"] is not False:
        raise ValueError("SID-Set images must not be displayed publicly")
    if privacy["sid_set_paths_or_img_ids_published"] is not False:
        raise ValueError("SID-Set paths and image IDs must remain private")
    if privacy["only_aggregate_properties_and_anonymised_identifiers_published"] is not True:
        raise ValueError("Public error analysis must remain aggregate and anonymised")
    if privacy["per_image_records_must_remain_git_ignored"] is not True:
        raise ValueError("Per-image error records must remain Git-ignored")
    guardrails = protocol["frozen_guardrails"]
    for rule in (
        "retraining_allowed",
        "threshold_changes_allowed",
        "model_reselection_allowed",
        "error_analysis_allowed_to_change_models_or_thresholds",
        "organiser_validation_subset_used",
    ):
        if guardrails[rule] is not False:
            raise ValueError(f"Frozen error-analysis rule changed: {rule}")
    if int(protocol["analysis"]["property_thumbnail_size"]) <= 0:
        raise ValueError("Property thumbnail size must be positive")
    if int(protocol["analysis"]["representative_records_per_error_type"]) <= 0:
        raise ValueError("Representative error count must be positive")


def anonymised_identifier(image_path: str) -> str:
    """Return a stable identifier that cannot expose a local or source path."""

    return hashlib.sha256(image_path.encode("utf-8")).hexdigest()[:12]


def condition_error_analysis(
    labels: np.ndarray,
    clean_probabilities: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Describe false-positive/negative direction and clean-relative transitions."""

    label_array = np.asarray(labels, dtype=np.int64)
    clean = np.asarray(clean_probabilities, dtype=np.float64)
    transformed = np.asarray(probabilities, dtype=np.float64)
    if label_array.shape != clean.shape or label_array.shape != transformed.shape:
        raise ValueError("Labels and clean/transformed probabilities must align")
    if set(np.unique(label_array)) != {0, 1}:
        raise ValueError("Condition error analysis requires binary labels")
    clean_predictions = clean >= threshold
    predictions = transformed >= threshold
    errors = predictions != label_array
    clean_errors = clean_predictions != label_array
    real = label_array == 0
    ai = label_array == 1
    clean_correct = ~clean_errors
    condition_correct = ~errors
    return {
        "samples": int(len(labels)),
        "false_positives_real_called_ai": int(np.sum(real & predictions)),
        "false_negatives_ai_called_real": int(np.sum(ai & ~predictions)),
        "real_error_rate": float(np.mean(predictions[real])),
        "ai_error_rate": float(np.mean(~predictions[ai])),
        "total_errors": int(np.sum(errors)),
        "balanced_error_rate": float(
            0.5 * (np.mean(predictions[real]) + np.mean(~predictions[ai]))
        ),
        "new_errors_from_clean": {
            "total": int(np.sum(clean_correct & errors)),
            "real_became_false_positive": int(np.sum(real & clean_correct & errors)),
            "ai_became_false_negative": int(np.sum(ai & clean_correct & errors)),
        },
        "recovered_clean_errors": {
            "total": int(np.sum(clean_errors & condition_correct)),
            "real_false_positive_recovered": int(
                np.sum(real & clean_errors & condition_correct)
            ),
            "ai_false_negative_recovered": int(
                np.sum(ai & clean_errors & condition_correct)
            ),
        },
        "prediction_flips_vs_clean": {
            "total": int(np.sum(clean_predictions != predictions)),
            "real_to_ai": int(np.sum(real & ~clean_predictions & predictions)),
            "ai_to_real": int(np.sum(ai & clean_predictions & ~predictions)),
        },
        "probability_shift_vs_clean": {
            "all_mean": float(np.mean(transformed - clean)),
            "real_mean": float(np.mean((transformed - clean)[real])),
            "ai_mean": float(np.mean((transformed - clean)[ai])),
            "mean_absolute": float(np.mean(np.abs(transformed - clean))),
        },
    }


def error_persistence_analysis(
    labels: np.ndarray,
    error_matrix: np.ndarray,
    *,
    clean_condition_index: int = 0,
) -> dict[str, Any]:
    """Count how many frozen conditions misclassify each paired test image."""

    label_array = np.asarray(labels, dtype=np.int64)
    errors = np.asarray(error_matrix, dtype=bool)
    if errors.ndim != 2 or errors.shape[1] != len(label_array):
        raise ValueError("Error matrix must be conditions by aligned samples")
    if not 0 <= clean_condition_index < errors.shape[0]:
        raise ValueError("Invalid clean condition index")
    error_counts = errors.sum(axis=0)
    clean_errors = errors[clean_condition_index]
    maximum = errors.shape[0]

    def group(mask: np.ndarray) -> dict[str, Any]:
        counts = error_counts[mask]
        histogram = np.bincount(counts, minlength=maximum + 1)
        return {
            "samples": int(np.sum(mask)),
            "mean_failed_conditions": float(np.mean(counts)),
            "median_failed_conditions": float(np.median(counts)),
            "never_failed": int(np.sum(counts == 0)),
            "failed_at_least_once": int(np.sum(counts > 0)),
            "failed_majority_of_conditions": int(np.sum(counts > maximum / 2)),
            "failed_all_conditions": int(np.sum(counts == maximum)),
            "clean_correct_but_transformation_failed": int(
                np.sum(~clean_errors[mask] & (counts > 0))
            ),
            "histogram_failed_condition_count": histogram.tolist(),
        }

    return {
        "conditions": maximum,
        "all": group(np.ones(len(label_array), dtype=bool)),
        "real_0": group(label_array == 0),
        "ai_generated_1": group(label_array == 1),
        "per_image_failed_condition_count": error_counts,
    }


def error_jaccard_matrix(error_matrix: np.ndarray) -> np.ndarray:
    """Return pairwise Jaccard similarity of condition-level error sets."""

    errors = np.asarray(error_matrix, dtype=bool)
    if errors.ndim != 2:
        raise ValueError("Error matrix must be two-dimensional")
    count = errors.shape[0]
    matrix = np.zeros((count, count), dtype=np.float64)
    for row in range(count):
        for column in range(count):
            union = np.sum(errors[row] | errors[column])
            intersection = np.sum(errors[row] & errors[column])
            matrix[row, column] = 1.0 if union == 0 else intersection / union
    return matrix


def image_properties(image: Image.Image, *, thumbnail_size: int) -> dict[str, float]:
    """Extract inexpensive low-level properties from one local image."""

    if thumbnail_size <= 0:
        raise ValueError("thumbnail_size must be positive")
    rgb = image.convert("RGB")
    width, height = rgb.size
    thumbnail = rgb.resize(
        (thumbnail_size, thumbnail_size), resample=Image.Resampling.BILINEAR
    )
    array = np.asarray(thumbnail, dtype=np.float32) / 255.0
    luminance = (
        0.2126 * array[:, :, 0]
        + 0.7152 * array[:, :, 1]
        + 0.0722 * array[:, :, 2]
    )
    maximum = array.max(axis=2)
    minimum = array.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    histogram, _ = np.histogram(luminance, bins=64, range=(0.0, 1.0))
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    horizontal_edges = np.abs(np.diff(luminance, axis=1)).mean()
    vertical_edges = np.abs(np.diff(luminance, axis=0)).mean()
    return {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(width / height),
        "megapixels": float(width * height / 1_000_000),
        "mean_luminance": float(np.mean(luminance)),
        "luminance_standard_deviation": float(np.std(luminance)),
        "mean_saturation": float(np.mean(saturation)),
        "grayscale_entropy": float(entropy),
        "edge_strength": float(0.5 * (horizontal_edges + vertical_edges)),
    }


def summarise_property_groups(
    records: list[dict[str, Any]], property_names: tuple[str, ...]
) -> dict[str, Any]:
    """Summarise image properties by error category and real-image association."""

    if not records:
        raise ValueError("Property records cannot be empty")
    categories = sorted({str(row["category"]) for row in records})

    def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            property_name: {
                "mean": float(np.mean([row["properties"][property_name] for row in rows])),
                "median": float(
                    np.median([row["properties"][property_name] for row in rows])
                ),
                "standard_deviation": float(
                    np.std([row["properties"][property_name] for row in rows], ddof=1)
                    if len(rows) > 1
                    else 0.0
                ),
            }
            for property_name in property_names
        }

    by_category: dict[str, Any] = {}
    for category in categories:
        rows = [row for row in records if row["category"] == category]
        by_category[category] = {"samples": len(rows), "properties": stats(rows)}

    real_rows = [row for row in records if row["label"] == 0]
    true_negative = [row for row in real_rows if row["category"] == "true_negative"]
    false_positive = [row for row in real_rows if row["category"] == "false_positive"]
    if not true_negative or not false_positive:
        raise ValueError("Real-image analysis requires true negatives and false positives")
    associations: dict[str, Any] = {}
    scores = np.asarray([row["probability"] for row in real_rows])
    for property_name in property_names:
        tn = np.asarray([row["properties"][property_name] for row in true_negative])
        fp = np.asarray([row["properties"][property_name] for row in false_positive])
        pooled_variance = (
            ((len(tn) - 1) * np.var(tn, ddof=1) + (len(fp) - 1) * np.var(fp, ddof=1))
            / (len(tn) + len(fp) - 2)
        )
        pooled_standard_deviation = math.sqrt(max(float(pooled_variance), 0.0))
        standardised_difference = (
            0.0
            if pooled_standard_deviation == 0.0
            else float((np.mean(fp) - np.mean(tn)) / pooled_standard_deviation)
        )
        values = np.asarray(
            [row["properties"][property_name] for row in real_rows], dtype=np.float64
        )
        correlation = spearmanr(values, scores).statistic
        associations[property_name] = {
            "false_positive_minus_true_negative_standardised_mean_difference": standardised_difference,
            "spearman_correlation_with_e3_ai_probability_among_real": float(
                0.0 if not np.isfinite(correlation) else correlation
            ),
        }
    return {"by_category": by_category, "real_image_associations": associations}


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _validate_hash(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"Frozen Section 4D input changed or is missing: {path}")


def _category(label: int, prediction: int) -> str:
    if label == 0:
        return "true_negative" if prediction == 0 else "false_positive"
    return "true_positive" if prediction == 1 else "false_negative"


def _representative_errors(
    records: list[dict[str, Any]], category: str, count: int, threshold: float
) -> list[dict[str, Any]]:
    errors = [row for row in records if row["category"] == category]
    errors.sort(
        key=lambda row: abs(float(row["probability"]) - threshold), reverse=True
    )
    return [
        {
            "anonymous_id": row["anonymous_id"],
            "category": row["category"],
            "probability": row["probability"],
            "confidence_margin_from_frozen_threshold": abs(
                float(row["probability"]) - threshold
            ),
            "properties": row.get("properties"),
        }
        for row in errors[:count]
    ]


def _write_direction_figure(
    by_condition: dict[str, Any], conditions: tuple[str, ...], path: Path
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.arange(len(conditions))
    false_positives = [
        by_condition[name]["false_positives_real_called_ai"] for name in conditions
    ]
    false_negatives = [
        by_condition[name]["false_negatives_ai_called_real"] for name in conditions
    ]
    induced = [by_condition[name]["new_errors_from_clean"]["total"] for name in conditions]
    recovered = [by_condition[name]["recovered_clean_errors"]["total"] for name in conditions]
    figure, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
    width = 0.38
    axes[0].bar(positions - width / 2, false_positives, width, label="False positives", color="#dc2626")
    axes[0].bar(positions + width / 2, false_negatives, width, label="False negatives", color="#2563eb")
    axes[1].bar(positions - width / 2, induced, width, label="New errors vs clean", color="#f97316")
    axes[1].bar(positions + width / 2, recovered, width, label="Recovered clean errors", color="#16a34a")
    for axis, title in (
        (axes[0], "Error direction at frozen E3 threshold"),
        (axes[1], "Transformation-induced and recovered errors"),
    ):
        axis.set(title=title, ylabel="Images")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    axes[1].set(
        xticks=positions,
        xticklabels=[TRANSFORM_SPECS[name].display_name for name in conditions],
    )
    axes[1].tick_params(axis="x", rotation=35)
    figure.suptitle("Section 4D — CIFAKE transformation error analysis", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_persistence_figure(persistence: dict[str, Any], path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(persistence["conditions"] + 1)
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(
        x,
        persistence["real_0"]["histogram_failed_condition_count"],
        marker="o",
        label="Real images",
        color="#2563eb",
    )
    axis.plot(
        x,
        persistence["ai_generated_1"]["histogram_failed_condition_count"],
        marker="o",
        label="AI-generated images",
        color="#dc2626",
    )
    axis.set(
        xlabel="Number of conditions misclassified (out of 15)",
        ylabel="Images",
        title="Section 4D — cross-condition error persistence",
        xticks=x,
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_cooccurrence_figure(
    matrix: np.ndarray, conditions: tuple[str, ...], path: Path
) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0)
    labels = [TRANSFORM_SPECS[name].display_name for name in conditions]
    axis.set(
        xticks=np.arange(len(conditions)),
        yticks=np.arange(len(conditions)),
        xticklabels=labels,
        yticklabels=labels,
        title="Section 4D — Jaccard similarity of E3 error sets",
    )
    axis.tick_params(axis="x", rotation=45)
    figure.colorbar(image, ax=axis, label="Error-set Jaccard similarity")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_property_figure(property_summary: dict[str, Any], path: Path) -> None:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    associations = property_summary["real_image_associations"]
    names = tuple(associations)
    display = [name.replace("_", " ") for name in names]
    standardised = [
        associations[name][
            "false_positive_minus_true_negative_standardised_mean_difference"
        ]
        for name in names
    ]
    correlations = [
        associations[name]["spearman_correlation_with_e3_ai_probability_among_real"]
        for name in names
    ]
    positions = np.arange(len(names))
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].bar(positions, standardised, color="#dc2626")
    axes[1].bar(positions, correlations, color="#2563eb")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[0].set(
        ylabel="Standardised difference",
        title="False-positive minus correctly-real property profile",
    )
    axes[1].set(
        ylabel="Spearman correlation",
        title="Property correlation with E3 AI probability among real images",
        xticks=positions,
        xticklabels=display,
    )
    axes[1].tick_params(axis="x", rotation=35)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Section 4D — aggregate SID-Set real-image error profile", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_local_records(path: Path, records: list[dict[str, Any]]) -> None:
    atomic_json_write(
        path,
        {
            "warning": "Local per-image research artifact. Git-ignored; do not publish.",
            "records": records,
        },
    )


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    matrix = report["cifake_full_matrix"]
    heldout = report["held_out_flux"]
    persistence = matrix["error_persistence"]
    e3 = heldout["confusion_and_rates"]
    blur = matrix["by_condition"]["gaussian_blur_sigma2"]
    resize = matrix["by_condition"]["resize_0_25x"]
    conditions = tuple(matrix["conditions"])
    jaccard = np.asarray(matrix["error_set_jaccard"])
    off_diagonal = [
        (float(jaccard[row, column]), conditions[row], conditions[column])
        for row in range(len(conditions))
        for column in range(row + 1, len(conditions))
    ]
    top_overlap, top_overlap_a, top_overlap_b = max(off_diagonal)
    property_associations = heldout["aggregate_property_analysis"][
        "real_image_associations"
    ]
    strongest_property, strongest_property_values = max(
        property_associations.items(),
        key=lambda item: abs(
            item[1][
                "false_positive_minus_true_negative_standardised_mean_difference"
            ]
        ),
    )
    strongest_property_difference = strongest_property_values[
        "false_positive_minus_true_negative_standardised_mean_difference"
    ]
    lines = [
        "# Section 4D Error Analysis",
        "",
        "This note analyses the already-frozen E3 detector. No result in this document was used to retrain the model, alter threshold 0.437, or reselect the primary model.",
        "",
        "## Main findings",
        "",
        f"- On held-out SID-Set/FLUX, E3 correctly detects **{e3['true_positives']} / 1,000** FLUX images but falsely flags **{e3['false_positives']} / 1,000** real images.",
        f"- The real-image false-positive rate is **{e3['real_false_positive_rate']:.1%}**; the paired-bootstrap 95% interval is **{heldout['bootstrap_reference']['real_false_positive_rate_lower']:.1%}–{heldout['bootstrap_reference']['real_false_positive_rate_upper']:.1%}**.",
        f"- Severe blur σ=2 produces **{blur['false_positives_real_called_ai']} false positives** and **{blur['false_negatives_ai_called_real']} false negatives** on CIFAKE.",
        f"- Severe 0.25× resizing produces **{resize['false_positives_real_called_ai']} false positives** and **{resize['false_negatives_ai_called_real']} false negatives**.",
        f"- Across all 15 CIFAKE conditions, **{persistence['real_0']['failed_at_least_once']} / 1,000 real** and **{persistence['ai_generated_1']['failed_at_least_once']} / 1,000 AI-generated** images fail at least once.",
        f"- **{persistence['all']['clean_correct_but_transformation_failed']} / 2,000** CIFAKE images are correct when clean but fail under at least one transformation.",
        "",
        "## Transformation failure direction",
        "",
        "![Condition error directions](figures/section4d_condition_error_directions.png)",
        "",
        "The strongest blur and resize settings create many false positives, meaning destructive resampling can remove or alter cues that E3 associates with authentic images. Strong noise instead produces more missed AI images, showing that different transformations shift scores in different directions.",
        "",
        "## Cross-condition persistence",
        "",
        "![Error persistence](figures/section4d_error_persistence.png)",
        "",
        "Repeated failures are not uniformly distributed. A subset of images fails under several conditions, while many remain correct throughout. The error-set co-occurrence heatmap helps distinguish shared failure groups from transformation-specific failures.",
        "",
        "![Error co-occurrence](figures/section4d_error_cooccurrence.png)",
        "",
        f"The largest off-diagonal overlap is **{top_overlap:.3f} Jaccard similarity** between **{TRANSFORM_SPECS[top_overlap_a].display_name}** and **{TRANSFORM_SPECS[top_overlap_b].display_name}**. Those two destructive resampling conditions therefore tend to break many of the same images.",
        "",
        "## Held-out real-image false positives",
        "",
        "![SID-Set aggregate property profile](figures/section4d_sidset_real_property_profile.png)",
        "",
        f"All measured low-level associations are weak: the largest absolute standardised mean difference is **{abs(strongest_property_difference):.3f}** for **{strongest_property.replace('_', ' ')}**. Simple size, brightness, colour, entropy, and edge measurements therefore do not explain the high external-real false-positive rate.",
        "",
        "These property associations are descriptive, not causal. They cannot separate generator artefacts from differences in dataset source, content, image resolution, or processing pipeline. Individual SID-Set images and paths are not included because their source attribution is unavailable through the sampled dataset interface.",
        "",
        "## Deployment implication",
        "",
        "E3 is suitable as a research prototype and ranking signal, but not as an automatic moderation decision-maker. A deployment-oriented version needs source-diverse real images, separate external calibration data, a low-false-positive operating point, and an abstention or uncertainty outcome.",
        "",
        "## Guardrails and limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["interpretation_limits"])
    lines.extend(
        [
            "- The organiser validation subset was never used.",
            "- No SID-Set images are displayed or committed.",
            "- Per-image records remain under the Git-ignored `outputs/` directory.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create frozen E3 aggregate error analysis for Section 4D."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--direction-figure", type=Path, default=DEFAULT_DIRECTION_FIGURE)
    parser.add_argument("--persistence-figure", type=Path, default=DEFAULT_PERSISTENCE_FIGURE)
    parser.add_argument("--cooccurrence-figure", type=Path, default=DEFAULT_COOCCURRENCE_FIGURE)
    parser.add_argument("--property-figure", type=Path, default=DEFAULT_PROPERTY_FIGURE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    for name, record in protocol["inputs"].items():
        if isinstance(record, dict) and "sha256" in record:
            _validate_hash(Path(record["path"]), record["sha256"])

    full_report = json.loads(
        Path(protocol["inputs"]["full_matrix_report"]["path"]).read_text()
    )
    heldout_report = json.loads(
        Path(protocol["inputs"]["held_out_report"]["path"]).read_text()
    )
    bootstrap_report = json.loads(
        Path(protocol["inputs"]["bootstrap_report"]["path"]).read_text()
    )
    if full_report["frozen_before_evaluation"]["primary_model"] != MODEL_NAME:
        raise ValueError("Full-matrix primary model changed")
    if heldout_report["pre_evaluation_freeze"]["primary_model"] != MODEL_NAME:
        raise ValueError("Held-out primary model changed")
    if bootstrap_report["guardrails"]["organiser_validation_subset_used"] is not False:
        raise ValueError("Bootstrap analysis used organiser data")

    full_arrays = _load_archive(
        Path(protocol["inputs"]["full_matrix_probabilities"]["path"])
    )
    heldout_arrays = _load_archive(
        Path(protocol["inputs"]["held_out_probabilities"]["path"])
    )
    threshold = float(protocol["model"]["frozen_threshold"])
    labels = full_arrays["labels"].astype(np.int64)
    conditions = tuple(full_report["test_data"]["conditions"])
    if conditions != FULL_ROBUSTNESS_CONDITIONS:
        raise ValueError("Full-matrix conditions changed")
    clean = full_arrays[f"{MODEL_NAME}_clean"]
    by_condition: dict[str, Any] = {}
    errors: list[np.ndarray] = []
    for condition in conditions:
        probabilities = full_arrays[f"{MODEL_NAME}_{condition}"]
        by_condition[condition] = condition_error_analysis(
            labels, clean, probabilities, threshold=threshold
        )
        errors.append((probabilities >= threshold) != labels)
    error_matrix = np.stack(errors)
    persistence = error_persistence_analysis(labels, error_matrix)
    per_image_failed_count = persistence.pop("per_image_failed_condition_count")
    jaccard = error_jaccard_matrix(error_matrix)

    heldout_labels = heldout_arrays["labels"].astype(np.int64)
    heldout_paths = heldout_arrays["image_paths"].astype(str)
    heldout_probabilities = heldout_arrays[f"{MODEL_NAME}_probabilities"]
    if len(heldout_labels) != 2000 or np.sum(heldout_labels == 0) != 1000:
        raise ValueError("Held-out identity arrays changed")
    manifest_path = Path(protocol["inputs"]["held_out_manifest"])
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if [row["image_path"] for row in manifest_rows] != heldout_paths.tolist():
        raise ValueError("Held-out manifest and probabilities are misaligned")

    thumbnail_size = int(protocol["analysis"]["property_thumbnail_size"])
    property_names = tuple(protocol["analysis"]["held_out_image_properties"])
    local_records: list[dict[str, Any]] = []
    print("Extracting privacy-safe SID-Set aggregate image properties", flush=True)
    for index, (label, image_path, probability) in enumerate(
        zip(heldout_labels, heldout_paths, heldout_probabilities, strict=True)
    ):
        prediction = int(probability >= threshold)
        with Image.open(image_path) as image:
            properties = image_properties(image, thumbnail_size=thumbnail_size)
        if tuple(properties) != property_names:
            raise ValueError("Configured and extracted image properties differ")
        local_records.append(
            {
                "index": index,
                "image_path": image_path,
                "anonymous_id": anonymised_identifier(image_path),
                "label": int(label),
                "prediction": prediction,
                "probability": float(probability),
                "category": _category(int(label), prediction),
                "properties": properties,
            }
        )
    property_summary = summarise_property_groups(local_records, property_names)
    category_counts = Counter(row["category"] for row in local_records)
    representative_count = int(
        protocol["analysis"]["representative_records_per_error_type"]
    )
    representatives = {
        "false_positives": _representative_errors(
            local_records, "false_positive", representative_count, threshold
        ),
        "false_negatives": _representative_errors(
            local_records, "false_negative", representative_count, threshold
        ),
    }

    local_output = Path(protocol["privacy_and_licensing"]["per_image_records"])
    cifake_local_records = [
        {
            "anonymous_id": anonymised_identifier(str(path)),
            "label": int(label),
            "failed_condition_count": int(failed),
        }
        for path, label, failed in zip(
            full_arrays["image_paths"], labels, per_image_failed_count, strict=True
        )
    ]
    _write_local_records(
        local_output,
        [
            {"dataset": "cifake", **row} for row in cifake_local_records
        ]
        + [{"dataset": "sid_set", **row} for row in local_records],
    )

    _write_direction_figure(by_condition, conditions, args.direction_figure)
    _write_persistence_figure(persistence, args.persistence_figure)
    _write_cooccurrence_figure(jaccard, conditions, args.cooccurrence_figure)
    _write_property_figure(property_summary, args.property_figure)

    heldout_matrix = heldout_report["models"][MODEL_NAME]["metrics"][
        "confusion_matrix"
    ]
    bootstrap_e3 = bootstrap_report["held_out_flux"]["models"][MODEL_NAME][
        "metrics"
    ]
    report = {
        "experiment": protocol["experiment_name"],
        "purpose": protocol["purpose"],
        "protocol": {
            "path": args.protocol.as_posix(),
            "sha256": sha256_file(args.protocol),
            "version": protocol["protocol_version"],
        },
        "model": protocol["model"],
        "cifake_full_matrix": {
            "samples": len(labels),
            "conditions": list(conditions),
            "by_condition": by_condition,
            "error_persistence": persistence,
            "error_set_jaccard": jaccard.tolist(),
        },
        "held_out_flux": {
            "samples": len(heldout_labels),
            "confusion_and_rates": {
                "true_negatives": int(heldout_matrix[0][0]),
                "false_positives": int(heldout_matrix[0][1]),
                "false_negatives": int(heldout_matrix[1][0]),
                "true_positives": int(heldout_matrix[1][1]),
                "real_false_positive_rate": float(category_counts["false_positive"] / 1000),
                "flux_false_negative_rate": float(category_counts["false_negative"] / 1000),
            },
            "category_counts": dict(category_counts),
            "aggregate_property_analysis": property_summary,
            "anonymised_representative_errors": representatives,
            "bootstrap_reference": {
                "real_false_positive_rate_lower": bootstrap_e3[
                    "real_false_positive_rate"
                ]["lower"],
                "real_false_positive_rate_upper": bootstrap_e3[
                    "real_false_positive_rate"
                ]["upper"],
                "flux_recall_lower": bootstrap_e3["flux_recall"]["lower"],
                "flux_recall_upper": bootstrap_e3["flux_recall"]["upper"],
            },
            "individual_images_or_paths_published": False,
        },
        "artifacts": {
            "public_markdown_note": args.markdown.as_posix(),
            "condition_error_directions": args.direction_figure.as_posix(),
            "condition_error_directions_sha256": sha256_file(args.direction_figure),
            "error_persistence": args.persistence_figure.as_posix(),
            "error_persistence_sha256": sha256_file(args.persistence_figure),
            "error_cooccurrence": args.cooccurrence_figure.as_posix(),
            "error_cooccurrence_sha256": sha256_file(args.cooccurrence_figure),
            "sidset_real_property_profile": args.property_figure.as_posix(),
            "sidset_real_property_profile_sha256": sha256_file(args.property_figure),
            "local_per_image_records": local_output.as_posix(),
            "local_per_image_records_sha256": sha256_file(local_output),
        },
        "privacy_and_licensing": protocol["privacy_and_licensing"],
        "guardrails": protocol["frozen_guardrails"],
        "interpretation_limits": protocol["interpretation_limits"],
    }
    _write_markdown(report, args.markdown)
    report["artifacts"]["public_markdown_note_sha256"] = sha256_file(args.markdown)
    atomic_json_write(args.report, report)
    print(
        f"PASS Section 4D: CIFAKE conditions={len(conditions)}, "
        f"SID real false positives={category_counts['false_positive']}, "
        f"FLUX false negatives={category_counts['false_negative']}",
        flush=True,
    )
    print(
        f"Updated report: {args.report}; note={args.markdown}; local={local_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Section 4D error analysis failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
