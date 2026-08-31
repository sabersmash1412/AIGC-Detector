#!/usr/bin/env python3
"""Complete the frozen AIGIBench audit using content-unique candidates."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile, ZipInfo

import requests
from tqdm import tqdm

from scripts.extract_clip_features import atomic_json_write, sha256_file
from scripts.prepare_e5_external_aigibench import (
    _manifest_and_provenance_rows,
    canonical_member_list_sha256,
    select_class_members,
)
from scripts.prepare_e5_external_synthbuster import (
    HTTPRangeReader,
    _atomic_csv_write,
    _project_relative,
    _session,
    extract_member,
    load_development_content_hashes,
)
from src.e5_aigibench_amendment import (
    validate_amendment,
    validate_outputs_absent,
)


DEFAULT_AMENDMENT = Path("configs/e5_aigibench_deduplication_amendment.json")
DEFAULT_LOCK_REPORT = Path("reports/e5_aigibench_deduplication_amendment_lock.json")


def validate_amendment_lock(
    amendment_path: Path, lock_report_path: Path
) -> dict[str, Any]:
    receipt = json.loads(lock_report_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise ValueError("AIGIBench deduplication amendment lock did not pass")
    frozen = receipt.get("amendment", {})
    if frozen.get("path") != amendment_path.as_posix():
        raise ValueError("Amendment lock references a different protocol")
    if frozen.get("sha256") != sha256_file(amendment_path):
        raise ValueError("AIGIBench amendment changed after its lock")
    if receipt["observed_prefeature_state"][
        "features_predictions_or_metrics_present"
    ] is not False:
        raise ValueError("Amendment lock was not made before model access")
    return receipt


def _complete_class_selection(
    *,
    archive: ZipFile,
    candidates: list[ZipInfo],
    destination_root: Path,
    target_unique: int,
    progress: tqdm,
) -> tuple[list[ZipInfo], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    selected: list[ZipInfo] = []
    verification_by_member: dict[str, dict[str, Any]] = {}
    first_member_by_hash: dict[str, tuple[int, str]] = {}
    excluded: list[dict[str, Any]] = []
    for candidate_rank, member in enumerate(candidates):
        destination = destination_root / PurePosixPath(member.filename).name
        verification = extract_member(archive, member, destination)
        digest = str(verification["sha256"])
        retained = first_member_by_hash.get(digest)
        if retained is not None:
            excluded.append(
                {
                    "candidate_rank": candidate_rank,
                    "archive_member": member.filename,
                    "sha256": digest,
                    "duplicate_of_candidate_rank": retained[0],
                    "duplicate_of_archive_member": retained[1],
                }
            )
            continue
        first_member_by_hash[digest] = (candidate_rank, member.filename)
        selected.append(member)
        verification_by_member[member.filename] = {
            "candidate_rank": candidate_rank,
            **verification,
        }
        progress.update(1)
        if len(selected) == target_unique:
            break
    if len(selected) != target_unique:
        raise ValueError(
            f"Only {len(selected)} unique candidates were available; required {target_unique}"
        )
    return selected, verification_by_member, excluded


def prepare_deduplicated_external_set(
    amendment_path: Path,
    lock_report_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    validate_amendment(amendment, project_root)
    validate_outputs_absent(amendment, project_root)
    amendment_lock = validate_amendment_lock(amendment_path, lock_report_path)
    base_path = project_root / amendment["base_protocol"]["path"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    dataset = base["external_dataset"]
    archive_lock = dataset["archive"]
    population = amendment["candidate_population"]
    selection = amendment["deduplication_selection"]
    raw_root = project_root / amendment["known_prefeature_state"]["raw_root"]
    real_root = raw_root / "real"
    ai_root = raw_root / "ai_generated"
    outputs = amendment["outputs"]
    manifest_path = project_root / outputs["manifest"]
    provenance_path = project_root / outputs["provenance"]

    session = _session()
    reader = HTTPRangeReader(session, str(archive_lock["source_url"]))
    if reader.size != int(population["archive_remote_bytes"]):
        reader.close()
        session.close()
        raise ValueError("Pinned AIGIBench archive byte size changed")
    try:
        archive = ZipFile(reader, "r")
    except Exception:
        reader.close()
        session.close()
        raise

    target = int(selection["target_unique_images_per_class"])
    progress = tqdm(total=target * 2, desc="E5 unique AIGIBench images", unit="image")
    try:
        seed = int(population["selection_seed"])
        real_candidates = select_class_members(
            archive,
            prefix=population["real_prefix"],
            available_images=int(population["real_candidates"]),
            selected_images=int(population["real_candidates"]),
            seed=seed,
        )
        ai_candidates = select_class_members(
            archive,
            prefix=population["ai_prefix"],
            available_images=int(population["ai_candidates"]),
            selected_images=int(population["ai_candidates"]),
            seed=seed,
        )
        if canonical_member_list_sha256([real_candidates]) != population[
            "real_candidate_order_sha256"
        ]:
            raise ValueError("Frozen real candidate ordering changed")
        if canonical_member_list_sha256([ai_candidates]) != population[
            "ai_candidate_order_sha256"
        ]:
            raise ValueError("Frozen AI candidate ordering changed")

        real_members, real_verification, real_excluded = _complete_class_selection(
            archive=archive,
            candidates=real_candidates,
            destination_root=real_root,
            target_unique=target,
            progress=progress,
        )
        ai_members, ai_verification, ai_excluded = _complete_class_selection(
            archive=archive,
            candidates=ai_candidates,
            destination_root=ai_root,
            target_unique=target,
            progress=progress,
        )
    finally:
        progress.close()
        archive.close()
        reader.close()
        session.close()

    real_hashes = {str(item["sha256"]) for item in real_verification.values()}
    ai_hashes = {str(item["sha256"]) for item in ai_verification.values()}
    if len(real_hashes) != target or len(ai_hashes) != target:
        raise ValueError("Content-deduplicated class count changed unexpectedly")
    cross_class = real_hashes & ai_hashes
    if cross_class:
        raise ValueError(f"AIGIBench has {len(cross_class)} cross-class duplicate images")

    real_manifest, real_provenance = _manifest_and_provenance_rows(
        selected=real_members,
        verification_by_member=real_verification,
        destination_root=real_root,
        label=0,
        class_name="real",
        project_root=project_root,
    )
    ai_manifest, ai_provenance = _manifest_and_provenance_rows(
        selected=ai_members,
        verification_by_member=ai_verification,
        destination_root=ai_root,
        label=1,
        class_name="ai_generated",
        project_root=project_root,
    )
    provenance_rows = real_provenance + ai_provenance
    selected_hashes = real_hashes | ai_hashes
    development_hashes = load_development_content_hashes(project_root)
    overlap = selected_hashes & development_hashes
    if overlap:
        raise ValueError(
            f"AIGIBench content overlaps E5 development/audit data: {len(overlap)} files"
        )

    manifest_rows = real_manifest + ai_manifest
    _atomic_csv_write(manifest_path, manifest_rows)
    final_selection_digest = canonical_member_list_sha256([real_members, ai_members])
    provenance = {
        "experiment": "e5_fresh_aigibench_midjourney_deduplicated_preparation",
        "amendment": {
            "path": amendment_path.as_posix(),
            "sha256": sha256_file(amendment_path),
        },
        "amendment_lock": {
            "path": lock_report_path.as_posix(),
            "sha256": sha256_file(lock_report_path),
            "frozen_at_utc": amendment_lock["frozen_at_utc"],
        },
        "base_protocol": amendment["base_protocol"],
        "duplicate_audit": amendment["duplicate_audit"],
        "dataset": {
            "benchmark": dataset["benchmark_name"],
            "repository_commit": archive_lock["repository_commit"],
            "license": dataset["license"],
            "real_source": dataset["real"]["source_name"],
            "synthetic_generator": dataset["synthetic"]["selected_generator"],
            "organiser_validation_subset_used": False,
            "manual_image_inspection_before_scoring": False,
        },
        "selection": {
            "algorithm": selection["algorithm"],
            "target_unique_images_per_class": target,
            "real_candidates_examined": max(
                item["candidate_rank"] for item in real_verification.values()
            )
            + 1,
            "ai_candidates_examined": max(
                item["candidate_rank"] for item in ai_verification.values()
            )
            + 1,
            "real_duplicates_excluded": real_excluded,
            "ai_duplicates_excluded": ai_excluded,
            "final_selected_member_list_sha256": final_selection_digest,
        },
        "archive": {
            "source_archive": archive_lock["source_archive"],
            "remote_bytes": archive_lock["remote_bytes"],
            "repository_commit": archive_lock["repository_commit"],
            "xet_content_hash": archive_lock["xet_content_hash"],
            "linked_etag": archive_lock["linked_etag"],
            "complete_archive_downloaded": False,
        },
        "images": provenance_rows,
        "counts": {
            "total_selected_unique": len(provenance_rows),
            "real_selected_unique": len(real_provenance),
            "ai_selected_unique": len(ai_provenance),
            "cross_class_duplicate_groups": len(cross_class),
            "duplicates_excluded": len(real_excluded) + len(ai_excluded),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--lock-report", type=Path, default=DEFAULT_LOCK_REPORT)
    args = parser.parse_args()
    try:
        free_gib = shutil.disk_usage(Path.cwd()).free / (1024**3)
        if free_gib < 2.0:
            raise ValueError(f"At least 2 GiB free space is required; found {free_gib:.2f} GiB")
        result = prepare_deduplicated_external_set(
            args.amendment, args.lock_report, Path.cwd()
        )
        counts = result["counts"]
        selection = result["selection"]
        print(
            "PASS E5 deduplicated AIGIBench preparation: "
            f"real={counts['real_selected_unique']}, "
            f"midjourney={counts['ai_selected_unique']}, "
            f"duplicates_excluded={counts['duplicates_excluded']}, "
            f"overlap={counts['development_or_prior_audit_overlap']}"
        )
        print(
            f"Candidates examined: real={selection['real_candidates_examined']}, "
            f"ai={selection['ai_candidates_examined']}"
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
        print(f"E5 deduplicated AIGIBench preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
