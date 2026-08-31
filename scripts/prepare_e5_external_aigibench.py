#!/usr/bin/env python3
"""Selectively acquire the frozen AIGIBench Open Images/Midjourney audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile, ZipInfo

import requests
from tqdm import tqdm

from scripts.extract_clip_features import atomic_json_write, sha256_file
from scripts.prepare_e5_external_synthbuster import (
    HTTPRangeReader,
    IMAGE_SUFFIXES,
    MAX_IMAGE_BYTES,
    _atomic_csv_write,
    _project_relative,
    _session,
    extract_member,
    load_development_content_hashes,
)
from src.e5_aigibench_protocol import (
    validate_e5_aigibench_protocol,
    validate_frozen_inputs,
)


DEFAULT_PROTOCOL = Path("configs/e5_fresh_external_aigibench_midjourney.json")
DEFAULT_LOCK_REPORT = Path("reports/e5_aigibench_external_evaluation_lock.json")


def _selection_key(seed: int, member: ZipInfo) -> tuple[bytes, str]:
    digest = hashlib.sha256(f"{seed}:{member.filename}".encode("utf-8")).digest()
    return digest, member.filename


def select_class_members(
    archive: ZipFile,
    *,
    prefix: str,
    available_images: int,
    selected_images: int,
    seed: int,
) -> list[ZipInfo]:
    """Apply the frozen hash-ranked selection to one exact archive class."""

    path = PurePosixPath(prefix)
    if not prefix.endswith("/") or prefix.startswith("/") or ".." in path.parts:
        raise ValueError(f"Unsafe or non-directory ZIP prefix: {prefix!r}")
    members: list[ZipInfo] = []
    seen_basenames: set[str] = set()
    for member in archive.infolist():
        member_path = PurePosixPath(member.filename)
        if member.is_dir() or not member.filename.startswith(prefix):
            continue
        if member_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if member_path.name in seen_basenames:
            raise ValueError(f"Duplicate basename under ZIP prefix {prefix}: {member_path.name}")
        if member.file_size <= 0 or member.file_size > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Unsafe image size in ZIP member {member.filename}: {member.file_size}"
            )
        seen_basenames.add(member_path.name)
        members.append(member)
    if len(members) != available_images:
        raise ValueError(
            f"ZIP prefix {prefix!r} contains {len(members)} images; "
            f"the lock requires {available_images} available images"
        )
    if not 0 < selected_images <= available_images:
        raise ValueError("Frozen selected-image count is invalid")
    return sorted(members, key=lambda member: _selection_key(seed, member))[
        :selected_images
    ]


def canonical_member_list_sha256(groups: list[list[ZipInfo]]) -> str:
    names = [member.filename for group in groups for member in group]
    payload = ("\n".join(names) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_lock_receipt(
    protocol_path: Path, lock_report_path: Path
) -> dict[str, Any]:
    receipt = json.loads(lock_report_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise ValueError("E5 AIGIBench lock receipt did not pass")
    frozen = receipt.get("protocol", {})
    if frozen.get("path") != protocol_path.as_posix():
        raise ValueError("E5 AIGIBench receipt references a different protocol")
    if frozen.get("sha256") != sha256_file(protocol_path):
        raise ValueError("E5 AIGIBench protocol changed after its pre-download lock")
    if receipt.get("external_artifacts_present_at_freeze") is not False:
        raise ValueError("AIGIBench artifacts were present when the lock was recorded")
    return receipt


def _manifest_and_provenance_rows(
    *,
    selected: list[ZipInfo],
    verification_by_member: dict[str, dict[str, Any]],
    destination_root: Path,
    label: int,
    class_name: str,
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for selection_rank, member in enumerate(selected):
        destination = destination_root / PurePosixPath(member.filename).name
        relative = _project_relative(destination, project_root)
        manifest_rows.append(
            {
                "image_path": relative,
                "label": label,
                "class_name": class_name,
                "source": "aigibench_midjourney_v6_external",
                "split": "external_test",
            }
        )
        provenance_rows.append(
            {
                "image_path": relative,
                "label": label,
                "class_name": class_name,
                "selection_rank": selection_rank,
                **verification_by_member[member.filename],
            }
        )
    return manifest_rows, provenance_rows


def prepare_external_set(
    protocol_path: Path,
    lock_report_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_e5_aigibench_protocol(protocol)
    validate_frozen_inputs(protocol, project_root)
    lock_receipt = validate_lock_receipt(protocol_path, lock_report_path)
    integrity = protocol["acquisition_and_integrity"]
    dataset = protocol["external_dataset"]
    archive_lock = dataset["archive"]
    selection_lock = dataset["selection"]
    raw_root = project_root / integrity["raw_root"]
    manifest_path = project_root / integrity["manifest"]
    provenance_path = project_root / integrity["provenance"]
    real_root = raw_root / "real"
    ai_root = raw_root / "ai_generated"

    session = _session()
    reader = HTTPRangeReader(session, str(archive_lock["source_url"]))
    if reader.size != int(archive_lock["remote_bytes"]):
        reader.close()
        session.close()
        raise ValueError(
            f"Pinned archive size changed: {reader.size} != {archive_lock['remote_bytes']}"
        )
    try:
        archive = ZipFile(reader, "r")
    except Exception:
        reader.close()
        session.close()
        raise

    progress = tqdm(total=2000, desc="E5 AIGIBench images", unit="image")
    try:
        seed = int(selection_lock["selection_seed"])
        real = dataset["real"]
        synthetic = dataset["synthetic"]
        real_members = select_class_members(
            archive,
            prefix=real["expected_archive_prefix"],
            available_images=int(real["available_images"]),
            selected_images=int(real["selected_images"]),
            seed=seed,
        )
        ai_members = select_class_members(
            archive,
            prefix=synthetic["expected_archive_prefix"],
            available_images=int(synthetic["available_images"]),
            selected_images=int(synthetic["selected_images"]),
            seed=seed,
        )
        selection_digest = canonical_member_list_sha256([real_members, ai_members])
        expected_digest = selection_lock["canonical_member_list_sha256"]
        if selection_digest != expected_digest:
            raise ValueError(
                "Deterministic AIGIBench selection changed: "
                f"{selection_digest} != {expected_digest}"
            )

        destination_by_member = {
            member.filename: real_root / PurePosixPath(member.filename).name
            for member in real_members
        }
        destination_by_member.update(
            {
                member.filename: ai_root / PurePosixPath(member.filename).name
                for member in ai_members
            }
        )
        verification_by_member: dict[str, dict[str, Any]] = {}
        # Archive-offset order substantially reduces repeated HTTP range traffic while
        # the final manifest remains in the separately frozen selection order.
        for member in sorted(real_members + ai_members, key=lambda item: item.header_offset):
            verification_by_member[member.filename] = extract_member(
                archive, member, destination_by_member[member.filename]
            )
            progress.update(1)
    finally:
        progress.close()
        archive.close()
        reader.close()
        session.close()

    real_manifest, real_provenance = _manifest_and_provenance_rows(
        selected=real_members,
        verification_by_member=verification_by_member,
        destination_root=real_root,
        label=0,
        class_name="real",
        project_root=project_root,
    )
    ai_manifest, ai_provenance = _manifest_and_provenance_rows(
        selected=ai_members,
        verification_by_member=verification_by_member,
        destination_root=ai_root,
        label=1,
        class_name="ai_generated",
        project_root=project_root,
    )
    provenance_rows = real_provenance + ai_provenance
    external_hashes = [str(row["sha256"]) for row in provenance_rows]
    if len(set(external_hashes)) != 2000:
        raise ValueError("The selected AIGIBench set contains byte-identical images")
    development_hashes = load_development_content_hashes(project_root)
    overlap = set(external_hashes) & development_hashes
    if overlap:
        raise ValueError(
            f"AIGIBench content overlaps E5 development/audit data: {len(overlap)} files"
        )

    manifest_rows = real_manifest + ai_manifest
    _atomic_csv_write(manifest_path, manifest_rows)
    provenance = {
        "experiment": "e5_fresh_aigibench_midjourney_preparation",
        "protocol": {
            "path": protocol_path.as_posix(),
            "sha256": sha256_file(protocol_path),
        },
        "lock_receipt": {
            "path": lock_report_path.as_posix(),
            "sha256": sha256_file(lock_report_path),
            "frozen_at_utc": lock_receipt["frozen_at_utc"],
        },
        "dataset": {
            "benchmark": dataset["benchmark_name"],
            "repository_commit": archive_lock["repository_commit"],
            "license": dataset["license"],
            "real_source": dataset["real"]["source_name"],
            "synthetic_generator": dataset["synthetic"]["selected_generator"],
            "organiser_validation_subset_used": False,
            "manual_image_inspection_before_scoring": False,
        },
        "archive": {
            "source_archive": archive_lock["source_archive"],
            "remote_bytes": archive_lock["remote_bytes"],
            "repository_commit": archive_lock["repository_commit"],
            "xet_content_hash": archive_lock["xet_content_hash"],
            "linked_etag": archive_lock["linked_etag"],
            "selected_member_list_sha256": selection_digest,
            "selected_images": 2000,
            "integrity": "Pinned URL and byte size; frozen member-list SHA-256; ZIP CRC32 per entry; SHA-256 and decode validation per extracted image.",
            "complete_archive_downloaded": False,
        },
        "images": provenance_rows,
        "counts": {
            "total": len(provenance_rows),
            "real": len(real_provenance),
            "ai_generated": len(ai_provenance),
            "content_unique": len(set(external_hashes)),
            "development_or_prior_audit_overlap": len(overlap),
            "development_and_audit_exclusion_hashes": len(development_hashes),
        },
        "manifest": {
            "path": _project_relative(manifest_path, project_root),
            "sha256": None,
        },
        "provenance": {"path": _project_relative(provenance_path, project_root)},
        "publication": {
            "raw_images_committed_to_git": False,
            "individual_images_allowed_in_public_demo": False,
            "aggregate_outputs_only": True,
        },
    }
    provenance["manifest"]["sha256"] = sha256_file(manifest_path)
    atomic_json_write(provenance_path, provenance)
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Selectively prepare frozen AIGIBench real + Midjourney V6 data."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--lock-report", type=Path, default=DEFAULT_LOCK_REPORT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        free_gib = shutil.disk_usage(Path.cwd()).free / (1024**3)
        if free_gib < 6.0:
            raise ValueError(
                f"At least 6 GiB free space is required; found {free_gib:.2f} GiB"
            )
        result = prepare_external_set(args.protocol, args.lock_report, Path.cwd())
        counts = result["counts"]
        print(
            "PASS E5 AIGIBench preparation: "
            f"real={counts['real']}, midjourney={counts['ai_generated']}, "
            f"overlap={counts['development_or_prior_audit_overlap']}"
        )
        print(
            f"Manifest={result['manifest']['path']}; "
            f"provenance={result['provenance']['path']}"
        )
        return 0
    except (
        BadZipFile,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        requests.RequestException,
    ) as exc:
        print(f"E5 AIGIBench preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
