from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_names_exact_frozen_e5_identity_and_thresholds() -> None:
    assert "b6c25a38a86692a74280650f516105c01efbaabe91f8da728b1a455cbf1756c4" in README
    assert "real `≤0.237`" in README
    assert "AI `≥0.817`" in README
    assert "0.52" in README


def test_readme_discloses_external_failure_and_responsible_use() -> None:
    assert "official external decision is still FAIL" in README
    assert "not externally validated for safety-critical" in README
    assert "Do not use this prototype by itself" in README


def test_readme_contains_fresh_clone_and_audit_commands() -> None:
    required_commands = (
        "python3.12 -m venv .venv",
        ".venv/bin/python -m pip install -r requirements.txt",
        ".venv/bin/python -m src.demo_app --device auto --inbrowser",
        ".venv/bin/python -m scripts.audit_submission --check-only",
        ".venv/bin/python -m pytest -q",
    )
    for command in required_commands:
        assert command in README


def test_project_has_mit_license() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License")
    assert "Copyright (c) 2026 Annamalai" in license_text


def test_readme_headline_e5_results_match_frozen_report() -> None:
    report = json.loads(
        (ROOT / "reports/e5_aigibench_external_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    e5 = report["binary_model_comparison"]["E5"]
    clean = e5["by_condition"]["clean"]["metrics"]
    summary = e5["summary"]
    for value in (
        clean["roc_auc"],
        clean["balanced_accuracy"],
        summary["mean_full_matrix_roc_auc"],
        summary["worst_full_matrix_roc_auc"],
    ):
        assert f"{value:.4f}" in README
    assert report["status"] == "FAIL"


def test_all_relative_readme_links_exist() -> None:
    targets = re.findall(r"\]\(([^)]+)\)", README)
    relative_targets = [
        target.split("#", maxsplit=1)[0]
        for target in targets
        if not target.startswith(("http://", "https://", "#"))
    ]
    assert relative_targets
    for target in relative_targets:
        assert (ROOT / target).exists(), target
