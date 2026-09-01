from __future__ import annotations

import json

import pytest

from scripts.audit_submission import ROOT, _walk_json, run_audit


def test_recursive_json_walk_exposes_nested_organiser_flag() -> None:
    record = {"outer": [{"organiser_validation_subset_used": False}]}
    observed = dict(_walk_json(record))
    assert observed[("outer", "0", "organiser_validation_subset_used")] is False


def test_submission_audit_passes_checked_in_frozen_state() -> None:
    report = run_audit(ROOT)
    assert report["status"] == "PASS"
    assert all(check["status"] == "PASS" for check in report["checks"].values())
    assert (
        report["checks"]["organiser_validation_isolation"]["evidence_records"]
        >= 10
    )


def test_submission_audit_rejects_changed_checkpoint_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.audit_submission as audit

    monkeypatch.setattr(audit, "EXPECTED_E5_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="checkpoint identity changed"):
        run_audit(ROOT)


def test_written_audit_report_is_valid_json() -> None:
    path = ROOT / "reports/submission_audit.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8"))["status"] == "PASS"
