#!/usr/bin/env python3
"""Prepare deterministic SID-Set-train real-only manifests for E4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from scripts.extract_clip_features import atomic_json_write, sha256_file
from scripts.prepare_sid_set_heldout import (
    Candidate,
    _download_image,
    _project_relative,
    _safe_stem,
    _session,
)
from src.image_transforms import DEFAULT_ROBUSTNESS_CONDITIONS


DEFAULT_PROTOCOL = Path("configs/e4_domain_adaptation.json")
MANIFEST_FIELDS = ("image_path", "label", "class_name", "source", "split")
MODEL_NAME = "E3"


@dataclass(frozen=True)
class AuditExclusions:
    """Identifiers and byte hashes that E4 development must not reuse."""

    image_ids: frozenset[str]
    content_sha256: frozenset[str]


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Reject E4 data leakage, task drift, and training-protocol drift."""

    if protocol["status"] != "frozen_before_e4_data_download":
        raise ValueError("E4 protocol must be frozen before data preparation")
    positioning = protocol["research_positioning"]
    if positioning["E3_remains_original_primary_model"] is not True:
        raise ValueError("E3 must remain the original primary model")
    if positioning["E4_is_post_hoc_follow_up"] is not True:
        raise ValueError("E4 must remain explicitly post-hoc")
    if positioning["universal_detector_claim_allowed"] is not False:
        raise ValueError("E4 cannot claim universal detection")
    if positioning["existing_section4_results_may_be_overwritten"] is not False:
        raise ValueError("E4 cannot overwrite frozen Section 4 results")

    dataset = protocol["development_data"]["sid_set_real"]
    if dataset["repository"] != "saberzl/SID_Set":
        raise ValueError("Unexpected E4 source repository")
    if not re.fullmatch(r"[0-9a-f]{40}", dataset["source_revision"]):
        raise ValueError("E4 source revision must be a full Git hash")
    if dataset["source_split"] != "train" or int(dataset["source_rows"]) != 210000:
        raise ValueError("E4 SID data must come only from the 210,000-row train split")
    if int(dataset["allowed_source_label"]) != 0:
        raise ValueError("E4 SID development data must be real-only label 0")
    if set(dataset["forbidden_source_labels"]) != {"1", "2"}:
        raise ValueError("Synthetic and tampered SID labels must remain forbidden")
    if dataset["declared_license"] != "CC-BY-4.0":
        raise ValueError("Unexpected SID-Set declared license")

    sample = protocol["sid_real_sampling"]
    if int(sample["page_size"]) <= 0 or int(sample["candidate_buffer"]) <= 0:
        raise ValueError("E4 sample page size and buffer must be positive")
    expected_pages = (
        int(dataset["source_rows"]) + int(sample["page_size"]) - 1
    ) // int(sample["page_size"])
    if int(sample["page_count"]) != expected_pages:
        raise ValueError("E4 page count does not cover the pinned train split")
    if int(sample["train_count"]) + int(sample["validation_count"]) != int(
        sample["total_count"]
    ):
        raise ValueError("E4 train and validation counts do not match total count")
    if min(
        int(sample["train_count"]),
        int(sample["validation_count"]),
        int(sample["total_count"]),
    ) <= 0:
        raise ValueError("E4 sample counts must be positive")
    if sample["manual_cherry_picking_allowed"] is not False:
        raise ValueError("Manual E4 sample cherry-picking is forbidden")

    audit = protocol["frozen_section4_audit"]
    if audit["dataset_split"] != "SID-Set validation":
        raise ValueError("Frozen audit must remain SID-Set validation")
    for rule in (
        "allowed_for_training",
        "allowed_for_validation",
        "allowed_for_epoch_or_threshold_selection",
    ):
        if audit[rule] is not False:
            raise ValueError(f"Frozen audit leakage rule changed: {rule}")
    for key in ("manifest_sha256", "provenance_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", audit[key]):
            raise ValueError(f"Frozen audit {key} is not a SHA-256 hash")

    organiser = protocol["organiser_validation_subset"]
    for rule in (
        "allowed_for_training",
        "allowed_for_validation",
        "allowed_for_model_or_threshold_selection",
        "used",
    ):
        if organiser[rule] is not False:
            raise ValueError(f"Organiser exclusion rule changed: {rule}")

    initialization = protocol["initialization"]
    if MODEL_NAME not in initialization["model"]:
        raise ValueError("E4 must initialise from E3")
    if not re.fullmatch(r"[0-9a-f]{64}", initialization["checkpoint_sha256"]):
        raise ValueError("E4 initialization checkpoint hash is invalid")
    if tuple(protocol["representative_conditions"]) != DEFAULT_ROBUSTNESS_CONDITIONS:
        raise ValueError("E4 representative conditions changed from the registry")

    groups = protocol["training"]["group_sampling_per_epoch"]
    if set(groups) != {"cifake_ai_generated", "cifake_real", "sid_set_train_real"}:
        raise ValueError("E4 training groups changed")
    if not abs(sum(float(value) for value in groups.values()) - 1.0) < 1e-12:
        raise ValueError("E4 group-sampling weights must sum to one")
    if groups["cifake_ai_generated"] != 0.5:
        raise ValueError("E4 per-epoch class balance changed")

    threshold = protocol["validation_and_threshold_selection"]
    if float(threshold["sid_real_fpr_constraint"]) != 0.05:
        raise ValueError("E4 SID-real false-positive constraint changed")
    if "no test or audit data" not in threshold["data"]:
        raise ValueError("E4 selection must exclude test and audit data")

    privacy = protocol["privacy_and_publication"]
    if privacy["raw_sid_images_must_not_be_committed"] is not True:
        raise ValueError("Raw SID images must remain out of Git")
    if privacy["individual_sid_images_allowed_in_public_demo"] is not False:
        raise ValueError("E4 SID images cannot appear in the public demo")
    if privacy["source_urls_persisted"] is not False:
        raise ValueError("E4 must not persist source URLs")


def deterministic_page_order(page_count: int, seed: int) -> list[int]:
    """Return a reproducible pseudorandom order of metadata pages."""

    if page_count <= 0:
        raise ValueError("page_count must be positive")
    return sorted(
        range(page_count),
        key=lambda page: hashlib.sha256(f"e4:{seed}:page:{page}".encode()).digest(),
    )


def candidate_rank(candidate: Candidate, seed: int) -> bytes:
    """Return the frozen deterministic rank for one eligible real row."""

    payload = f"e4:{seed}:row:{candidate.row_idx}:{candidate.img_id}".encode()
    return hashlib.sha256(payload).digest()


def _validate_frozen_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"Frozen E4 input changed or is missing: {path}")


def load_audit_exclusions(
    protocol: dict[str, Any], project_root: Path
) -> AuditExclusions:
    """Load exact Section 4B audit identities after validating frozen hashes."""

    audit = protocol["frozen_section4_audit"]
    manifest_path = project_root / audit["manifest"]
    provenance_path = project_root / audit["provenance"]
    _validate_frozen_file(manifest_path, audit["manifest_sha256"])
    _validate_frozen_file(provenance_path, audit["provenance_sha256"])
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    images = provenance.get("images", [])
    if len(images) != int(audit["samples"]):
        raise ValueError("Frozen Section 4B provenance sample count changed")
    image_ids = frozenset(str(row["img_id"]) for row in images)
    hashes = frozenset(str(row["sha256"]) for row in images)
    if len(image_ids) != len(images) or len(hashes) != len(images):
        raise ValueError("Frozen audit identities or contents are not unique")
    return AuditExclusions(image_ids=image_ids, content_sha256=hashes)


def parse_real_page(
    payload: dict[str, Any],
    *,
    expected_rows: int,
    expected_revision: str,
    observed_revision: str | None,
) -> list[Candidate]:
    """Validate one rows-API payload and retain only real source label 0."""

    if observed_revision != expected_revision:
        raise ValueError(
            f"SID-Set revision drift: expected {expected_revision}, "
            f"received {observed_revision!r}"
        )
    if payload.get("partial") is not False:
        raise ValueError("Hugging Face returned a partial E4 metadata page")
    if int(payload.get("num_rows_total", -1)) != expected_rows:
        raise ValueError("SID-Set E4 train row count changed")
    candidates: list[Candidate] = []
    for envelope in payload.get("rows", []):
        row = envelope["row"]
        if int(row["label"]) != 0:
            continue
        image = row.get("image")
        if not isinstance(image, dict) or not image.get("src"):
            raise ValueError(f"SID-Set E4 row {envelope['row_idx']} has no image asset")
        candidates.append(
            Candidate(
                row_idx=int(envelope["row_idx"]),
                img_id=str(row["img_id"]),
                source_label=0,
                project_label=0,
                class_name="real",
                image_url=str(image["src"]),
                source_width=int(row["width"]),
                source_height=int(row["height"]),
            )
        )
    return candidates


def fetch_real_page(
    session: requests.Session,
    *,
    protocol: dict[str, Any],
    page_index: int,
) -> list[Candidate]:
    """Fetch and validate one pinned SID-Set train metadata page."""

    dataset = protocol["development_data"]["sid_set_real"]
    sample = protocol["sid_real_sampling"]
    page_size = int(sample["page_size"])
    response = session.get(
        dataset["rows_api"],
        params={
            "dataset": dataset["repository"],
            "config": dataset["config"],
            "split": dataset["source_split"],
            "offset": page_index * page_size,
            "length": page_size,
        },
        timeout=(30, 180),
    )
    response.raise_for_status()
    return parse_real_page(
        response.json(),
        expected_rows=int(dataset["source_rows"]),
        expected_revision=dataset["source_revision"],
        observed_revision=response.headers.get("X-Revision"),
    )


def collect_candidate_pool(
    session: requests.Session,
    protocol: dict[str, Any],
    exclusions: AuditExclusions,
) -> tuple[list[Candidate], list[int], int]:
    """Collect a deterministic, audit-excluded candidate pool."""

    sample = protocol["sid_real_sampling"]
    required = int(sample["total_count"]) + int(sample["candidate_buffer"])
    pages = deterministic_page_order(int(sample["page_count"]), int(sample["seed"]))
    candidates: list[Candidate] = []
    pages_fetched: list[int] = []
    audit_ids_rejected = 0
    observed_row_indices: set[int] = set()
    observed_image_ids: set[str] = set()
    for page_index in tqdm(pages, desc="E4 SID real metadata", unit="page"):
        for candidate in fetch_real_page(
            session, protocol=protocol, page_index=page_index
        ):
            if candidate.row_idx in observed_row_indices:
                raise ValueError("SID-Set rows API returned a duplicate row index")
            observed_row_indices.add(candidate.row_idx)
            if candidate.img_id in exclusions.image_ids:
                audit_ids_rejected += 1
                continue
            if candidate.img_id in observed_image_ids:
                continue
            observed_image_ids.add(candidate.img_id)
            candidates.append(candidate)
        pages_fetched.append(page_index)
        if len(candidates) >= required:
            break
    if len(candidates) < required:
        raise ValueError(
            f"Only {len(candidates)} eligible E4 real candidates found; need {required}"
        )
    candidates.sort(key=lambda row: candidate_rank(row, int(sample["seed"])))
    return candidates, pages_fetched, audit_ids_rejected


def allocate_manifest_rows(
    accepted: list[tuple[Candidate, Path, dict[str, Any]]],
    *,
    train_count: int,
    validation_count: int,
    project_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Allocate ranked accepted images into disjoint train and validation rows."""

    if len(accepted) != train_count + validation_count:
        raise ValueError("Accepted E4 image count does not match split targets")
    manifests = {"train": [], "val": []}
    provenance_rows: list[dict[str, Any]] = []
    for index, (candidate, destination, verification) in enumerate(accepted):
        split = "train" if index < train_count else "val"
        relative_path = _project_relative(destination, project_root)
        manifests[split].append(
            {
                "image_path": relative_path,
                "label": 0,
                "class_name": "real",
                "source": "sid_set_train_real_e4",
                "split": split,
            }
        )
        provenance_rows.append(
            {
                "row_idx": candidate.row_idx,
                "img_id": candidate.img_id,
                "source_split": "train",
                "source_label": 0,
                "project_label": 0,
                "e4_split": split,
                "image_path": relative_path,
                **verification,
            }
        )
    return manifests, provenance_rows


def _atomic_csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def prepare_e4_real_data(protocol_path: Path, project_root: Path) -> dict[str, Any]:
    """Download and verify the frozen E4 real-only development sample."""

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    dataset = protocol["development_data"]["sid_set_real"]
    sample = protocol["sid_real_sampling"]
    initialization_path = project_root / protocol["initialization"]["checkpoint"]
    _validate_frozen_file(
        initialization_path, protocol["initialization"]["checkpoint_sha256"]
    )
    exclusions = load_audit_exclusions(protocol, project_root)

    session = _session()
    candidates, pages_fetched, audit_ids_rejected = collect_candidate_pool(
        session, protocol, exclusions
    )
    target = int(sample["total_count"])
    image_root = project_root / dataset["image_root"]
    accepted: list[tuple[Candidate, Path, dict[str, Any]]] = []
    accepted_hashes: set[str] = set()
    duplicate_content_rejected = 0
    audit_content_rejected = 0
    progress = tqdm(total=target, desc="E4 SID unique real images", unit="image")
    try:
        for candidate in candidates:
            destination = image_root / f"{_safe_stem(candidate)}.jpg"
            verification = _download_image(session, candidate, destination)
            content_hash = str(verification["sha256"])
            if content_hash in exclusions.content_sha256:
                audit_content_rejected += 1
                continue
            if content_hash in accepted_hashes:
                duplicate_content_rejected += 1
                continue
            accepted_hashes.add(content_hash)
            accepted.append((candidate, destination, verification))
            progress.update(1)
            if len(accepted) == target:
                break
    finally:
        progress.close()
    if len(accepted) != target:
        raise ValueError(
            f"Only {len(accepted)} unique, audit-disjoint E4 images accepted; need {target}"
        )

    manifests, provenance_rows = allocate_manifest_rows(
        accepted,
        train_count=int(sample["train_count"]),
        validation_count=int(sample["validation_count"]),
        project_root=project_root,
    )
    manifest_paths = {
        "train": project_root / dataset["train_manifest"],
        "val": project_root / dataset["validation_manifest"],
    }
    for split, path in manifest_paths.items():
        _atomic_csv_write(path, manifests[split])

    all_rows = manifests["train"] + manifests["val"]
    if Counter(int(row["label"]) for row in all_rows) != Counter({0: target}):
        raise ValueError("E4 manifests are not real-only")
    if len({row["image_path"] for row in all_rows}) != target:
        raise ValueError("E4 manifests contain duplicate paths")
    if set(row["img_id"] for row in provenance_rows) & exclusions.image_ids:
        raise ValueError("E4 provenance overlaps frozen audit image IDs")
    if set(row["sha256"] for row in provenance_rows) & exclusions.content_sha256:
        raise ValueError("E4 provenance overlaps frozen audit content")

    provenance_path = project_root / dataset["provenance"]
    provenance = {
        "experiment": protocol["experiment_name"],
        "protocol": {
            "path": _project_relative(protocol_path, project_root),
            "sha256": sha256_file(protocol_path),
            "version": protocol["protocol_version"],
        },
        "dataset": {
            "repository": dataset["repository"],
            "source_revision": dataset["source_revision"],
            "source_split": dataset["source_split"],
            "source_rows": dataset["source_rows"],
            "allowed_source_label": dataset["allowed_source_label"],
            "declared_license": dataset["declared_license"],
        },
        "sampling": {
            "seed": sample["seed"],
            "pages_fetched": pages_fetched,
            "candidate_pool": len(candidates),
            "audit_image_ids_rejected_during_metadata": audit_ids_rejected,
            "train_count": len(manifests["train"]),
            "validation_count": len(manifests["val"]),
            "synthetic_rows_downloaded": 0,
            "tampered_rows_downloaded": 0,
            "manual_cherry_picking": False,
        },
        "isolation": {
            "section4b_audit_manifest_sha256": protocol["frozen_section4_audit"][
                "manifest_sha256"
            ],
            "section4b_audit_provenance_sha256": protocol["frozen_section4_audit"][
                "provenance_sha256"
            ],
            "audit_image_id_overlap": 0,
            "audit_content_sha256_overlap": 0,
            "train_validation_path_overlap": 0,
        },
        "manifests": {
            split: {
                "path": _project_relative(path, project_root),
                "sha256": sha256_file(path),
                "rows": len(manifests[split]),
            }
            for split, path in manifest_paths.items()
        },
        "download": {
            "source_urls_persisted": False,
            "all_images_verified": True,
            "total_bytes": sum(int(row["bytes"]) for row in provenance_rows),
            "content_unique_images": len(accepted_hashes),
            "duplicate_content_rejected": duplicate_content_rejected,
            "audit_content_rejected": audit_content_rejected,
            "unreferenced_downloads_may_remain_in_ignored_raw_directory": True,
        },
        "images": provenance_rows,
        "organiser_validation_subset_used": False,
        "sid_set_synthetic_or_tampered_used": False,
    }
    atomic_json_write(provenance_path, provenance)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare frozen, real-only SID-Set train data for E4."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    protocol_path = (
        args.protocol if args.protocol.is_absolute() else project_root / args.protocol
    )
    result = prepare_e4_real_data(protocol_path, project_root)
    print(
        "PASS E4 SID real preparation: "
        f"train={result['sampling']['train_count']}, "
        f"val={result['sampling']['validation_count']}, "
        f"bytes={result['download']['total_bytes']}, "
        f"audit_overlap={result['isolation']['audit_content_sha256_overlap']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"E4 SID real preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
