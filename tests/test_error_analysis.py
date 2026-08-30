import copy
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.error_analysis import (
    anonymised_identifier,
    condition_error_analysis,
    error_jaccard_matrix,
    error_persistence_analysis,
    image_properties,
    summarise_property_groups,
    validate_protocol,
)


PROTOCOL_PATH = Path("configs/section4d_error_analysis.json")


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_checked_in_protocol_preserves_frozen_model_and_privacy() -> None:
    protocol = _protocol()

    validate_protocol(protocol)

    assert protocol["model"]["name"] == "E3"
    assert protocol["model"]["frozen_threshold"] == pytest.approx(0.437)
    assert protocol["frozen_guardrails"]["organiser_validation_subset_used"] is False
    assert protocol["privacy_and_licensing"]["sid_set_images_publicly_displayed"] is False
    assert protocol["privacy_and_licensing"]["sid_set_paths_or_img_ids_published"] is False


@pytest.mark.parametrize(
    ("section", "rule", "unsafe_value"),
    [
        ("frozen_guardrails", "threshold_changes_allowed", True),
        ("frozen_guardrails", "organiser_validation_subset_used", True),
        ("privacy_and_licensing", "sid_set_images_publicly_displayed", True),
        ("privacy_and_licensing", "sid_set_paths_or_img_ids_published", True),
        (
            "privacy_and_licensing",
            "only_aggregate_properties_and_anonymised_identifiers_published",
            False,
        ),
    ],
)
def test_protocol_rejects_unsafe_changes(
    section: str, rule: str, unsafe_value: bool
) -> None:
    protocol = copy.deepcopy(_protocol())
    protocol[section][rule] = unsafe_value

    with pytest.raises(ValueError):
        validate_protocol(protocol)


def test_anonymised_identifier_is_stable_and_does_not_expose_path() -> None:
    source_path = "/private/dataset/person-name/image-123.jpg"

    identifier = anonymised_identifier(source_path)

    assert identifier == anonymised_identifier(source_path)
    assert re.fullmatch(r"[0-9a-f]{12}", identifier)
    assert "person-name" not in identifier
    assert "image-123" not in identifier


def test_condition_error_analysis_counts_directions_and_transitions() -> None:
    labels = np.asarray([0, 0, 1, 1])
    clean = np.asarray([0.1, 0.2, 0.8, 0.9])
    transformed = np.asarray([0.1, 0.8, 0.2, 0.9])

    result = condition_error_analysis(
        labels, clean, transformed, threshold=0.5
    )

    assert result["false_positives_real_called_ai"] == 1
    assert result["false_negatives_ai_called_real"] == 1
    assert result["real_error_rate"] == pytest.approx(0.5)
    assert result["ai_error_rate"] == pytest.approx(0.5)
    assert result["new_errors_from_clean"]["total"] == 2
    assert result["recovered_clean_errors"]["total"] == 0
    assert result["prediction_flips_vs_clean"]["total"] == 2


def test_condition_error_analysis_counts_recovered_clean_errors() -> None:
    labels = np.asarray([0, 1])
    clean = np.asarray([0.8, 0.2])
    transformed = np.asarray([0.1, 0.9])

    result = condition_error_analysis(
        labels, clean, transformed, threshold=0.5
    )

    assert result["total_errors"] == 0
    assert result["new_errors_from_clean"]["total"] == 0
    assert result["recovered_clean_errors"]["total"] == 2
    assert result["recovered_clean_errors"]["real_false_positive_recovered"] == 1
    assert result["recovered_clean_errors"]["ai_false_negative_recovered"] == 1


def test_error_persistence_summarises_aligned_images() -> None:
    labels = np.asarray([0, 0, 1, 1])
    errors = np.asarray(
        [
            [False, True, False, False],
            [True, True, False, False],
            [False, True, True, False],
        ]
    )

    result = error_persistence_analysis(labels, errors)

    assert result["conditions"] == 3
    assert result["all"]["histogram_failed_condition_count"] == [1, 2, 0, 1]
    assert result["real_0"]["failed_at_least_once"] == 2
    assert result["real_0"]["failed_all_conditions"] == 1
    assert result["real_0"]["clean_correct_but_transformation_failed"] == 1
    assert result["ai_generated_1"]["never_failed"] == 1
    np.testing.assert_array_equal(
        result["per_image_failed_condition_count"], np.asarray([1, 3, 1, 0])
    )


def test_error_jaccard_matrix_matches_known_sets() -> None:
    errors = np.asarray(
        [
            [True, True, False, False],
            [False, True, True, False],
            [False, False, False, False],
        ]
    )

    matrix = error_jaccard_matrix(errors)

    assert matrix[0, 1] == pytest.approx(1 / 3)
    assert matrix[1, 0] == pytest.approx(1 / 3)
    assert matrix[0, 2] == 0.0
    assert matrix[2, 2] == 1.0


def test_image_properties_are_finite_for_a_solid_image() -> None:
    image = Image.new("RGB", (10, 20), color=(128, 128, 128))

    properties = image_properties(image, thumbnail_size=8)

    assert properties["width"] == 10
    assert properties["height"] == 20
    assert properties["aspect_ratio"] == pytest.approx(0.5)
    assert properties["megapixels"] == pytest.approx(0.0002)
    assert properties["grayscale_entropy"] == pytest.approx(0.0)
    assert properties["edge_strength"] == pytest.approx(0.0)
    assert all(np.isfinite(value) for value in properties.values())


def test_property_summary_is_aggregate_and_reports_real_association() -> None:
    records = [
        {
            "label": 0,
            "category": "true_negative",
            "probability": 0.1,
            "properties": {"width": 100.0},
        },
        {
            "label": 0,
            "category": "true_negative",
            "probability": 0.2,
            "properties": {"width": 110.0},
        },
        {
            "label": 0,
            "category": "false_positive",
            "probability": 0.8,
            "properties": {"width": 200.0},
        },
        {
            "label": 0,
            "category": "false_positive",
            "probability": 0.9,
            "properties": {"width": 210.0},
        },
        {
            "label": 1,
            "category": "true_positive",
            "probability": 0.9,
            "properties": {"width": 150.0},
        },
        {
            "label": 1,
            "category": "false_negative",
            "probability": 0.1,
            "properties": {"width": 140.0},
        },
    ]

    summary = summarise_property_groups(records, ("width",))

    assert summary["by_category"]["false_positive"]["samples"] == 2
    association = summary["real_image_associations"]["width"]
    assert (
        association[
            "false_positive_minus_true_negative_standardised_mean_difference"
        ]
        > 0
    )
    assert association["spearman_correlation_with_e3_ai_probability_among_real"] > 0
    assert "image_path" not in json.dumps(summary)
