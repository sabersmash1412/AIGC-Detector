"""Audit the frozen submission for leakage, provenance, and tracked-data safety."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("reports/submission_audit.json")
EXPECTED_E5_SHA256 = (
    "b6c25a38a86692a74280650f516105c01efbaabe91f8da728b1a455cbf1756c4"
)
FORBIDDEN_TRACKED_PREFIXES = ("data/raw/", "data/features/", "outputs/")
PRIVATE_TEXT_MARKERS = (
    "/Users/annamalai/",
    "annamalai@",
    "/Documents/AI generated image/",
)
AUDITED_JSON_GLOBS = (
    "configs/*.json",
    "reports/*.json",
    "data/processed/*provenance.json",
)
MANIFEST_EXPECTATIONS = {
    "data/processed/e4_sid_real/train.csv": {
        "rows": 3000,
        "labels": {"0": 3000},
    },
    "data/processed/e4_sid_real/val.csv": {
        "rows": 1000,
        "labels": {"0": 1000},
    },
    "data/processed/e5_sid_flux/train.csv": {
        "rows": 3000,
        "labels": {"1": 3000},
    },
    "data/processed/e5_sid_flux/val.csv": {
        "rows": 1000,
        "labels": {"1": 1000},
    },
    "data/processed/sid_set_flux_heldout.csv": {
        "rows": 2000,
        "labels": {"0": 1000, "1": 1000},
    },
    "data/processed/e5_aigibench_midjourney.csv": {
        "rows": 2000,
        "labels": {"0": 1000, "1": 1000},
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(path for path in result.stdout.decode().split("\0") if path)


def _walk_json(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            yield child_path, child
            yield from _walk_json(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, (*path, str(index)))


def _audit_organiser_exclusion(root: Path) -> dict[str, Any]:
    files: set[Path] = set()
    for pattern in AUDITED_JSON_GLOBS:
        files.update(root.glob(pattern))
    files.discard(root / DEFAULT_OUTPUT)

    evidence: list[str] = []
    violations: list[str] = []
    for path in sorted(files):
        record = json.loads(path.read_text(encoding="utf-8"))
        for key_path, value in _walk_json(record):
            key = key_path[-1]
            dotted = ".".join(key_path)
            relative = path.relative_to(root).as_posix()
            if key == "organiser_validation_subset_used":
                evidence.append(f"{relative}:{dotted}=false")
                if value is not False:
                    violations.append(f"{relative}:{dotted} must be false")
            elif key in {
                "organiser_validation_subset_forbidden",
                "organiser_coco_val2017_forbidden",
                "organiser_dalle_advanced_forbidden",
            }:
                evidence.append(f"{relative}:{dotted}=true")
                if value is not True:
                    violations.append(f"{relative}:{dotted} must be true")
            elif key == "organiser_validation_subset" and isinstance(value, dict):
                if "used" in value:
                    evidence.append(f"{relative}:{dotted}.used=false")
                    if value["used"] is not False:
                        violations.append(f"{relative}:{dotted}.used must be false")

    if len(evidence) < 10:
        violations.append("Too little organiser-exclusion evidence was found")
    if violations:
        raise ValueError("; ".join(violations))
    return {
        "status": "PASS",
        "evidence_records": len(evidence),
        "json_files_scanned": len(files),
        "rule": "All recorded organiser-use flags are false and forbidden flags are true.",
    }


def _audit_tracked_files(root: Path, tracked: list[str]) -> dict[str, Any]:
    forbidden = [
        path
        for path in tracked
        if any(path.startswith(prefix) for prefix in FORBIDDEN_TRACKED_PREFIXES)
    ]
    private_hits: list[str] = []
    for relative in tracked:
        path = root / relative
        try:
            payload = path.read_bytes()
        except (FileNotFoundError, IsADirectoryError):
            continue
        if b"\0" in payload[:8192]:
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if any(marker in text for marker in PRIVATE_TEXT_MARKERS):
            private_hits.append(relative)
    if forbidden:
        raise ValueError(f"Forbidden generated data is tracked: {forbidden}")
    if private_hits:
        raise ValueError(f"Private local text is tracked: {private_hits}")
    return {
        "status": "PASS",
        "raw_images_tracked": 0,
        "feature_caches_tracked": 0,
        "prediction_outputs_tracked": 0,
        "private_local_markers_found": 0,
    }


def _audit_manifests(root: Path) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for relative, expected in MANIFEST_EXPECTATIONS.items():
        path = root / relative
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        labels = Counter(row["label"] for row in rows)
        observed_labels = dict(sorted(labels.items()))
        if len(rows) != expected["rows"] or observed_labels != expected["labels"]:
            raise ValueError(f"Frozen manifest composition changed: {relative}")
        if any(Path(row["image_path"]).is_absolute() for row in rows):
            raise ValueError(f"Manifest contains absolute paths: {relative}")
        summaries[relative] = {
            "rows": len(rows),
            "labels": observed_labels,
            "sha256": sha256_file(path),
        }
    return {"status": "PASS", "manifests": summaries}


def _audit_licenses(root: Path) -> dict[str, Any]:
    sid = json.loads(
        (root / "configs/section4b_held_out_generator.json").read_text(
            encoding="utf-8"
        )
    )["dataset"]
    aigibench = json.loads(
        (root / "configs/e5_fresh_external_aigibench_midjourney.json").read_text(
            encoding="utf-8"
        )
    )["external_dataset"]
    expected = {
        "CIFAKE": "MIT (as declared by the official Kaggle dataset page)",
        "SID-Set": "CC-BY-4.0",
        "AIGIBench": "CC-BY-NC-SA-4.0",
        "OpenCLIP": "MIT",
        "OpenAI CLIP": "MIT",
    }
    if sid["declared_license"] != expected["SID-Set"]:
        raise ValueError("SID-Set license record changed")
    if aigibench["license"] != expected["AIGIBench"]:
        raise ValueError("AIGIBench license record changed")
    notices = root / "docs/DATASETS_AND_LICENSES.md"
    if not notices.is_file():
        raise ValueError("Dataset and license notices are missing")
    notice_text = notices.read_text(encoding="utf-8")
    for name, license_name in expected.items():
        if name not in notice_text or license_name.split(" (")[0] not in notice_text:
            raise ValueError(f"License notice is incomplete for {name}")
    return {"status": "PASS", "declared_assets": expected}


def run_audit(root: Path = ROOT) -> dict[str, Any]:
    tracked = _git_tracked_files(root)
    checkpoint = root / "checkpoints/clip_linear_e5_source_matched.npz"
    observed_checkpoint_sha = sha256_file(checkpoint)
    if observed_checkpoint_sha != EXPECTED_E5_SHA256:
        raise ValueError("Frozen E5 checkpoint identity changed")
    checks = {
        "organiser_validation_isolation": _audit_organiser_exclusion(root),
        "tracked_data_safety": _audit_tracked_files(root, tracked),
        "frozen_manifest_integrity": _audit_manifests(root),
        "licenses_and_citations": _audit_licenses(root),
        "frozen_e5_checkpoint": {
            "status": "PASS",
            "path": checkpoint.relative_to(root).as_posix(),
            "sha256": observed_checkpoint_sha,
        },
    }
    return {
        "audit": "submission_provenance_leakage_and_license_audit_v1",
        "status": "PASS",
        "checks": checks,
        "disclosures": [
            "No third-party image bytes are tracked by Git.",
            "Tracked selection manifests contain repository-relative paths and dataset-derived identifiers.",
            "AIGIBench was used only as a non-commercial held-out external evaluation set.",
            "This automated audit records technical evidence and is not legal advice.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_audit(ROOT)
        if not args.check_only:
            output = ROOT / args.output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            "PASS submission audit: organiser isolation, tracked data, manifests, "
            "licenses, and frozen E5 identity",
            flush=True,
        )
        if not args.check_only:
            print(f"Report={args.output}", flush=True)
        return 0
    except Exception as exc:
        print(f"Submission audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
