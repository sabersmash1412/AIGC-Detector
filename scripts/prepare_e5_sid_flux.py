#!/usr/bin/env python3
"""Prepare deterministic, audit-disjoint SID-train FLUX manifests for E5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
from src.e5_protocol import validate_e5_protocol, validate_frozen_e5_inputs


DEFAULT_PROTOCOL = Path("configs/e5_source_matched_adaptation.json")
MANIFEST_FIELDS = ("image_path", "label", "class_name", "source", "split")


@dataclass(frozen=True)
class DevelopmentExclusions:
    """Frozen identities and byte hashes unavailable to E5 FLUX sampling."""

    audit_image_ids: frozenset[str]
    audit_content_sha256: frozenset[str]
    real_image_ids: frozenset[str]
    real_content_sha256: frozenset[str]

    @property
    def all_image_ids(self) -> frozenset[str]:
        return self.audit_image_ids | self.real_image_ids

    @property
    def all_content_sha256(self) -> frozenset[str]:
        return self.audit_content_sha256 | self.real_content_sha256


def deterministic_page_order(page_count: int, seed: int) -> list[int]:
    """Return the complete E5-specific deterministic page order."""

    if page_count <= 0:
        raise ValueError("page_count must be positive")
    return sorted(
        range(page_count),
        key=lambda page: hashlib.sha256(f"e5:{seed}:page:{page}".encode()).digest(),
    )


def candidate_rank(candidate: Candidate, seed: int) -> bytes:
    """Return the frozen E5 rank for one eligible FLUX row."""

    payload = f"e5:{seed}:row:{candidate.row_idx}:{candidate.img_id}".encode()
    return hashlib.sha256(payload).digest()


def _load_provenance_images(path: Path, *, expected: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    if len(images) != expected:
        raise ValueError(f"Frozen provenance count changed: {path}")
    ids = [str(row["img_id"]) for row in images]
    hashes = [str(row["sha256"]) for row in images]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError(f"Frozen provenance is not identity/content unique: {path}")
    return images


def load_development_exclusions(
    protocol: dict[str, Any], project_root: Path
) -> DevelopmentExclusions:
    """Load audited real and held-out identities only after hash validation."""

    validate_frozen_e5_inputs(protocol, project_root)
    real = protocol["development_data"]["sid_set_train_real"]
    audit = protocol["forbidden_development_data"]["section4_sid_flux_audit"]
    real_images = _load_provenance_images(
        project_root / real["provenance"],
        expected=int(real["train_samples"]) + int(real["validation_samples"]),
    )
    audit_images = _load_provenance_images(
        project_root / audit["provenance"], expected=2000
    )
    exclusions = DevelopmentExclusions(
        audit_image_ids=frozenset(str(row["img_id"]) for row in audit_images),
        audit_content_sha256=frozenset(str(row["sha256"]) for row in audit_images),
        real_image_ids=frozenset(str(row["img_id"]) for row in real_images),
        real_content_sha256=frozenset(str(row["sha256"]) for row in real_images),
    )
    if exclusions.audit_image_ids & exclusions.real_image_ids:
        raise ValueError("Frozen E5 real data overlaps the prior SID audit by image ID")
    if exclusions.audit_content_sha256 & exclusions.real_content_sha256:
        raise ValueError("Frozen E5 real data overlaps the prior SID audit by content")
    return exclusions


def parse_flux_page(
    payload: dict[str, Any],
    *,
    expected_rows: int,
    expected_revision: str,
    observed_revision: str | None,
) -> list[Candidate]:
    """Validate one rows-API response and retain only full-synthetic label 1."""

    if observed_revision != expected_revision:
        raise ValueError(
            f"SID-Set revision drift: expected {expected_revision}, "
            f"received {observed_revision!r}"
        )
    if payload.get("partial") is not False:
        raise ValueError("Hugging Face returned a partial E5 metadata page")
    if int(payload.get("num_rows_total", -1)) != expected_rows:
        raise ValueError("SID-Set E5 train row count changed")
    candidates: list[Candidate] = []
    for envelope in payload.get("rows", []):
        row = envelope["row"]
        if int(row["label"]) != 1:
            continue
        image = row.get("image")
        if not isinstance(image, dict) or not image.get("src"):
            raise ValueError(f"SID-Set E5 row {envelope['row_idx']} has no image asset")
        candidates.append(
            Candidate(
                row_idx=int(envelope["row_idx"]),
                img_id=str(row["img_id"]),
                source_label=1,
                project_label=1,
                class_name="ai_generated",
                image_url=str(image["src"]),
                source_width=int(row["width"]),
                source_height=int(row["height"]),
            )
        )
    return candidates


def fetch_flux_page(
    session: requests.Session, *, protocol: dict[str, Any], page_index: int
) -> list[Candidate]:
    """Fetch one pinned SID-Set train page and retain label-1 metadata."""

    dataset = protocol["development_data"]["sid_set_train_flux"]
    sample = protocol["sid_flux_sampling"]
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
    return parse_flux_page(
        response.json(),
        expected_rows=int(dataset["source_rows"]),
        expected_revision=dataset["source_revision"],
        observed_revision=response.headers.get("X-Revision"),
    )


def collect_candidate_pool(
    session: requests.Session,
    protocol: dict[str, Any],
    exclusions: DevelopmentExclusions,
) -> tuple[list[Candidate], list[int], dict[str, int]]:
    """Collect a ranked label-1 pool while excluding frozen identities."""

    sample = protocol["sid_flux_sampling"]
    required = int(sample["total_count"]) + int(sample["candidate_buffer"])
    pages = deterministic_page_order(int(sample["page_count"]), int(sample["seed"]))
    candidates: list[Candidate] = []
    pages_fetched: list[int] = []
    observed_row_indices: set[int] = set()
    observed_image_ids: set[str] = set()
    rejected = {"audit_image_id": 0, "real_image_id": 0, "duplicate_image_id": 0}
    for page_index in tqdm(pages, desc="E5 SID FLUX metadata", unit="page"):
        for candidate in fetch_flux_page(session, protocol=protocol, page_index=page_index):
            if candidate.row_idx in observed_row_indices:
                raise ValueError("SID-Set rows API returned a duplicate row index")
            observed_row_indices.add(candidate.row_idx)
            if candidate.img_id in exclusions.audit_image_ids:
                rejected["audit_image_id"] += 1
                continue
            if candidate.img_id in exclusions.real_image_ids:
                rejected["real_image_id"] += 1
                continue
            if candidate.img_id in observed_image_ids:
                rejected["duplicate_image_id"] += 1
                continue
            observed_image_ids.add(candidate.img_id)
            candidates.append(candidate)
        pages_fetched.append(page_index)
        if len(candidates) >= required:
            break
    if len(candidates) < required:
        raise ValueError(
            f"Only {len(candidates)} eligible E5 FLUX candidates found; need {required}"
        )
    candidates.sort(key=lambda row: candidate_rank(row, int(sample["seed"])))
    return candidates, pages_fetched, rejected


def allocate_manifest_rows(
    accepted: list[tuple[Candidate, Path, dict[str, Any]]],
    *,
    train_count: int,
    validation_count: int,
    project_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Allocate ranked FLUX images into deterministic train/validation splits."""

    if len(accepted) != train_count + validation_count:
        raise ValueError("Accepted E5 FLUX count does not match split targets")
    manifests = {"train": [], "val": []}
    provenance_rows: list[dict[str, Any]] = []
    for index, (candidate, destination, verification) in enumerate(accepted):
        split = "train" if index < train_count else "val"
        relative_path = _project_relative(destination, project_root)
        manifests[split].append(
            {
                "image_path": relative_path,
                "label": 1,
                "class_name": "ai_generated",
                "source": "sid_set_train_flux_e5",
                "split": split,
            }
        )
        provenance_rows.append(
            {
                "row_idx": candidate.row_idx,
                "img_id": candidate.img_id,
                "source_split": "train",
                "source_label": 1,
                "project_label": 1,
                "e5_split": split,
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


def prepare_e5_flux_data(protocol_path: Path, project_root: Path) -> dict[str, Any]:
    """Download and verify the frozen E5 SID-train FLUX development sample."""

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_e5_protocol(protocol)
    validate_frozen_e5_inputs(protocol, project_root)
    dataset = protocol["development_data"]["sid_set_train_flux"]
    sample = protocol["sid_flux_sampling"]
    exclusions = load_development_exclusions(protocol, project_root)

    session = _session()
    candidates, pages_fetched, identity_rejections = collect_candidate_pool(
        session, protocol, exclusions
    )
    target = int(sample["total_count"])
    image_root = project_root / dataset["image_root"]
    accepted: list[tuple[Candidate, Path, dict[str, Any]]] = []
    accepted_hashes: set[str] = set()
    duplicate_content_rejected = 0
    audit_content_rejected = 0
    real_content_rejected = 0
    progress = tqdm(total=target, desc="E5 SID unique FLUX images", unit="image")
    try:
        for candidate in candidates:
            destination = image_root / f"{_safe_stem(candidate)}.jpg"
            verification = _download_image(session, candidate, destination)
            content_hash = str(verification["sha256"])
            if content_hash in exclusions.audit_content_sha256:
                audit_content_rejected += 1
                continue
            if content_hash in exclusions.real_content_sha256:
                real_content_rejected += 1
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
            f"Only {len(accepted)} unique, development-disjoint E5 FLUX images "
            f"accepted; need {target}"
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
    if Counter(int(row["label"]) for row in all_rows) != Counter({1: target}):
        raise ValueError("E5 manifests are not FLUX-only label 1")
    if len({row["image_path"] for row in all_rows}) != target:
        raise ValueError("E5 FLUX manifests contain duplicate paths")
    image_ids = {str(row["img_id"]) for row in provenance_rows}
    content_hashes = {str(row["sha256"]) for row in provenance_rows}
    if image_ids & exclusions.all_image_ids:
        raise ValueError("E5 FLUX provenance overlaps frozen development identities")
    if content_hashes & exclusions.all_content_sha256:
        raise ValueError("E5 FLUX provenance overlaps frozen development content")

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
            "identity_rejections": identity_rejections,
            "train_count": len(manifests["train"]),
            "validation_count": len(manifests["val"]),
            "real_rows_downloaded": 0,
            "tampered_rows_downloaded": 0,
            "manual_cherry_picking": False,
        },
        "isolation": {
            "frozen_real_provenance_sha256": protocol["development_data"][
                "sid_set_train_real"
            ]["provenance_sha256"],
            "section4b_audit_manifest_sha256": protocol[
                "forbidden_development_data"
            ]["section4_sid_flux_audit"]["manifest_sha256"],
            "section4b_audit_provenance_sha256": protocol[
                "forbidden_development_data"
            ]["section4_sid_flux_audit"]["provenance_sha256"],
            "real_image_id_overlap": 0,
            "real_content_sha256_overlap": 0,
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
            "real_content_rejected": real_content_rejected,
            "unreferenced_downloads_may_remain_in_ignored_raw_directory": True,
        },
        "images": provenance_rows,
        "organiser_validation_subset_used": False,
        "prior_sid_flux_audit_used_for_selection": False,
    }
    atomic_json_write(provenance_path, provenance)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare frozen SID-Set train FLUX-only data for E5."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    protocol_path = (
        args.protocol if args.protocol.is_absolute() else project_root / args.protocol
    )
    result = prepare_e5_flux_data(protocol_path, project_root)
    print(
        "PASS E5 SID FLUX preparation: "
        f"train={result['sampling']['train_count']}, "
        f"val={result['sampling']['validation_count']}, "
        f"bytes={result['download']['total_bytes']}, "
        f"real_overlap={result['isolation']['real_content_sha256_overlap']}, "
        f"audit_overlap={result['isolation']['audit_content_sha256_overlap']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"E5 SID FLUX preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
