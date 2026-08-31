from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.prepare_e5_external_aigibench_deduplicated import (
    validate_amendment_lock,
)
from src.e5_aigibench_amendment import (
    validate_amendment,
    validate_outputs_absent,
)
from src.robust_linear_training import sha256_file


AMENDMENT_PATH = Path("configs/e5_aigibench_deduplication_amendment.json")


def _amendment() -> dict:
    return json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))


def test_checked_in_amendment_preserves_frozen_e5_and_dataset() -> None:
    amendment = _amendment()
    validate_amendment(amendment, Path.cwd())


def test_amendment_rejects_model_threshold_change() -> None:
    amendment = copy.deepcopy(_amendment())
    amendment["frozen_e5"]["real_threshold"] = 0.5
    with pytest.raises(ValueError, match="E5 changed"):
        validate_amendment(amendment, Path.cwd())


def test_amendment_rejects_manual_replacement() -> None:
    amendment = copy.deepcopy(_amendment())
    amendment["deduplication_selection"]["manual_replacement_allowed"] = True
    with pytest.raises(ValueError, match="Forbidden amendment selection"):
        validate_amendment(amendment, Path.cwd())


def test_amendment_rejects_candidate_order_drift() -> None:
    amendment = copy.deepcopy(_amendment())
    amendment["candidate_population"]["selection_seed"] = 7
    with pytest.raises(ValueError, match="candidate population"):
        validate_amendment(amendment, Path.cwd())


def test_amendment_outputs_must_be_absent_before_continuation(tmp_path: Path) -> None:
    amendment = copy.deepcopy(_amendment())
    validate_outputs_absent(amendment, tmp_path)
    path = tmp_path / amendment["outputs"]["manifest"]
    path.parent.mkdir(parents=True)
    path.write_text("image_path,label\n", encoding="utf-8")
    with pytest.raises(ValueError, match="existed before amendment lock"):
        validate_outputs_absent(amendment, tmp_path)


def test_amendment_lock_rejects_drift(tmp_path: Path) -> None:
    amendment = tmp_path / "amendment.json"
    amendment.write_text('{"version": 1}\n', encoding="utf-8")
    receipt = tmp_path / "lock.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "PASS",
                "amendment": {
                    "path": amendment.as_posix(),
                    "sha256": sha256_file(amendment),
                },
                "observed_prefeature_state": {
                    "features_predictions_or_metrics_present": False
                },
            }
        ),
        encoding="utf-8",
    )
    validate_amendment_lock(amendment, receipt)
    amendment.write_text('{"version": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed after"):
        validate_amendment_lock(amendment, receipt)
