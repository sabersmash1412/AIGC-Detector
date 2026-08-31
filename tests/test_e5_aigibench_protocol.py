from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.e5_aigibench_protocol import (
    validate_artifacts_absent,
    validate_e5_aigibench_protocol,
    validate_frozen_inputs,
)


PROTOCOL_PATH = Path("configs/e5_fresh_external_aigibench_midjourney.json")


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_checked_in_aigibench_protocol_and_e5_inputs_are_frozen() -> None:
    protocol = _protocol()
    validate_e5_aigibench_protocol(protocol)
    validate_frozen_inputs(protocol, Path.cwd())
    validate_artifacts_absent(protocol, Path.cwd())


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("synthetic", "selected_generator", "DALL-E 3", "Midjourney V6"),
        ("synthetic", "selected_images", 500, "1,000 images per class"),
        ("real", "source_name", "COCO val2017", "Open Images V7"),
        ("real", "selected_images", 500, "1,000 images per class"),
    ],
)
def test_aigibench_lock_rejects_source_or_count_drift(
    section: str, key: str, value: object, match: str
) -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["external_dataset"][section][key] = value
    with pytest.raises(ValueError, match=match):
        validate_e5_aigibench_protocol(protocol)


def test_aigibench_lock_rejects_member_selection_drift() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["external_dataset"]["selection"]["selection_seed"] = 7
    with pytest.raises(ValueError, match="selection seed"):
        validate_e5_aigibench_protocol(protocol)


def test_aigibench_lock_rejects_threshold_drift() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["frozen_e5"]["ai_threshold"] = 0.7
    with pytest.raises(ValueError, match="selection changed"):
        validate_e5_aigibench_protocol(protocol)


def test_aigibench_lock_rejects_organiser_data() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["external_dataset"]["organiser_validation_subset_used"] = True
    with pytest.raises(ValueError, match="Organiser validation exclusion"):
        validate_e5_aigibench_protocol(protocol)


def test_aigibench_lock_rejects_relaxed_success_gate() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["frozen_success_criteria"][
        "maximum_wilson_upper_real_called_ai"
    ] = 0.2
    with pytest.raises(ValueError, match="success criterion"):
        validate_e5_aigibench_protocol(protocol)


def test_aigibench_lock_detects_preexisting_artifact(tmp_path: Path) -> None:
    protocol = copy.deepcopy(_protocol())
    raw_root = tmp_path / protocol["acquisition_and_integrity"]["raw_root"]
    raw_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="existed before protocol freeze"):
        validate_artifacts_absent(protocol, tmp_path)
