#!/usr/bin/env python3
"""Download a deterministic, balanced SID-Set real-versus-FLUX subset."""

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
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from scripts.extract_clip_features import atomic_json_write, sha256_file


DEFAULT_PROTOCOL = Path("configs/section4b_held_out_generator.json")
MANIFEST_FIELDS = ("image_path", "label", "class_name", "source", "split")
USER_AGENT = "AIGC-Detector/section4b (+https://github.com/sabersmash1412/AIGC-Detector)"
MAX_IMAGE_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class Candidate:
    """One source row eligible for deterministic held-out sampling."""

    row_idx: int
    img_id: str
    source_label: int
    project_label: int
    class_name: str
    image_url: str
    source_width: int
    source_height: int


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Validate licensing, label, sampling, and frozen-evaluation guardrails."""

    dataset = protocol["dataset"]
    if dataset["name"] != "SID-Set" or dataset["repository"] != "saberzl/SID_Set":
        raise ValueError("Section 4B must use the audited SID-Set repository")
    if not re.fullmatch(r"[0-9a-f]{40}", dataset["source_revision"]):
        raise ValueError("SID-Set source revision must be a full Git commit hash")
    audit = dataset["license_audit"]
    if dataset["declared_license"] != "CC-BY-4.0":
        raise ValueError("Unexpected SID-Set declared licence")
    if audit["status"] != "usable_with_attribution" or not audit[
        "attribution_required"
    ]:
        raise ValueError("SID-Set attribution requirements must remain explicit")
    if audit["raw_images_must_not_be_committed"] is not True:
        raise ValueError("Raw SID-Set images must remain out of Git")
    if audit[
        "individual_images_must_not_be_published_or_shown_without_source_attribution"
    ] is not True:
        raise ValueError("Individual SID-Set publication restrictions must remain explicit")
    if protocol["download"]["individual_images_allowed_in_public_demo"] is not False:
        raise ValueError("SID-Set images cannot be shown in the public demo")

    task = protocol["task"]
    included = task["included_source_labels"]
    if set(included) != {"0", "1"}:
        raise ValueError("Only SID-Set source labels 0 and 1 may be included")
    if included["0"]["project_label"] != 0 or included["1"]["project_label"] != 1:
        raise ValueError("Project label mapping must remain real=0 and AI-generated=1")
    if set(task["excluded_source_labels"]) != {"2"}:
        raise ValueError("SID-Set tampered label 2 must be explicitly excluded")
    if task["held_out_generator"] != "FLUX":
        raise ValueError("The held-out-generator identity must remain FLUX")

    sample = protocol["sample"]
    if sample["target_per_class"] <= 0 or sample["page_size"] <= 0:
        raise ValueError("Sample and page sizes must be positive")
    expected_pages = (
        int(dataset["source_rows"]) + int(sample["page_size"]) - 1
    ) // int(sample["page_size"])
    if int(sample["page_count"]) != expected_pages:
        raise ValueError("Page count does not cover the frozen source split")
    if int(sample["total_target"]) != 2 * int(sample["target_per_class"]):
        raise ValueError("Held-out sample must be exactly class balanced")

    evaluation = protocol["frozen_evaluation"]
    if tuple(evaluation["models"]) != ("E1", "E2", "E3"):
        raise ValueError("Frozen evaluation must compare exactly E1, E2, and E3")
    if evaluation["primary_model"] != "E3":
        raise ValueError("E3 must remain the preselected primary model")
    for rule in (
        "retraining_allowed",
        "threshold_changes_allowed",
        "model_reselection_allowed",
        "organiser_validation_subset_used",
    ):
        if evaluation[rule] is not False:
            raise ValueError(f"Frozen Section 4B rule changed: {rule}")


def deterministic_page_order(page_count: int, seed: int) -> list[int]:
    """Return a cross-platform deterministic pseudorandom ordering of pages."""

    if page_count <= 0:
        raise ValueError("page_count must be positive")
    return sorted(
        range(page_count),
        key=lambda page: hashlib.sha256(f"{seed}:page:{page}".encode()).digest(),
    )


def candidate_rank(candidate: Candidate, seed: int) -> bytes:
    payload = f"{seed}:row:{candidate.row_idx}:{candidate.img_id}".encode()
    return hashlib.sha256(payload).digest()


def select_balanced_candidates(
    candidates: list[Candidate], *, target_per_class: int, seed: int
) -> list[Candidate]:
    """Select an exactly balanced subset using only frozen identifiers."""

    if target_per_class <= 0:
        raise ValueError("target_per_class must be positive")
    by_label = {0: [], 1: []}
    for candidate in candidates:
        if candidate.project_label not in by_label:
            raise ValueError("Candidate has a non-binary project label")
        by_label[candidate.project_label].append(candidate)
    for label in (0, 1):
        if len(by_label[label]) < target_per_class:
            raise ValueError(
                f"Only {len(by_label[label])} candidates found for project label {label}; "
                f"need {target_per_class}"
            )
        by_label[label].sort(key=lambda row: candidate_rank(row, seed))

    selected = by_label[0][:target_per_class] + by_label[1][:target_per_class]
    selected.sort(key=lambda row: candidate_rank(row, seed))
    if len({row.row_idx for row in selected}) != len(selected):
        raise ValueError("Selected SID-Set source rows are not unique")
    return selected


def register_unique_content(
    candidate: Candidate,
    image_sha256: str,
    owners_by_sha256: dict[str, Candidate],
) -> Candidate | None:
    """Register unique bytes, return a same-label duplicate, or reject label conflict."""

    if not re.fullmatch(r"[0-9a-f]{64}", image_sha256):
        raise ValueError("Image SHA-256 must be a lowercase hexadecimal digest")
    existing = owners_by_sha256.get(image_sha256)
    if existing is None:
        owners_by_sha256[image_sha256] = candidate
        return None
    if existing.project_label != candidate.project_label:
        raise ValueError(
            "Byte-identical SID-Set images have conflicting project labels: "
            f"rows {existing.row_idx} and {candidate.row_idx}"
        )
    return existing


def _session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _fetch_page(
    session: requests.Session,
    *,
    protocol: dict[str, Any],
    page_index: int,
) -> list[Candidate]:
    dataset = protocol["dataset"]
    sample = protocol["sample"]
    page_size = int(sample["page_size"])
    response = session.get(
        protocol["download"]["rows_api"],
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
    revision = response.headers.get("X-Revision")
    if revision != dataset["source_revision"]:
        raise ValueError(
            f"SID-Set source revision drift: expected {dataset['source_revision']}, "
            f"received {revision!r}"
        )
    payload = response.json()
    if payload.get("partial") is not False:
        raise ValueError("Hugging Face returned a partial SID-Set page")
    if int(payload.get("num_rows_total", -1)) != int(dataset["source_rows"]):
        raise ValueError("SID-Set validation row count changed")

    included = protocol["task"]["included_source_labels"]
    candidates: list[Candidate] = []
    for envelope in payload["rows"]:
        row = envelope["row"]
        source_label = int(row["label"])
        definition = included.get(str(source_label))
        if definition is None:
            continue
        image = row.get("image")
        if not isinstance(image, dict) or not image.get("src"):
            raise ValueError(f"SID-Set row {envelope['row_idx']} has no image asset")
        candidates.append(
            Candidate(
                row_idx=int(envelope["row_idx"]),
                img_id=str(row["img_id"]),
                source_label=source_label,
                project_label=int(definition["project_label"]),
                class_name=str(definition["project_class_name"]),
                image_url=str(image["src"]),
                source_width=int(row["width"]),
                source_height=int(row["height"]),
            )
        )
    return candidates


def collect_candidates(
    session: requests.Session, protocol: dict[str, Any]
) -> tuple[list[Candidate], list[int], Counter[int]]:
    """Fetch deterministic source pages until both binary classes are large enough."""

    sample = protocol["sample"]
    target = int(sample["target_per_class"])
    candidates: list[Candidate] = []
    pages_fetched: list[int] = []
    counts: Counter[int] = Counter()
    order = deterministic_page_order(int(sample["page_count"]), int(sample["seed"]))
    for page_index in tqdm(order, desc="SID-Set metadata", unit="page"):
        page_candidates = _fetch_page(
            session, protocol=protocol, page_index=page_index
        )
        candidates.extend(page_candidates)
        counts.update(row.project_label for row in page_candidates)
        pages_fetched.append(page_index)
        if counts[0] >= target and counts[1] >= target:
            break
    if counts[0] < target or counts[1] < target:
        raise ValueError(
            f"Could not collect a balanced candidate pool: real={counts[0]}, "
            f"AI={counts[1]}"
        )
    return candidates, pages_fetched, counts


def _safe_stem(candidate: Candidate) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", candidate.img_id)[:80]
    if not cleaned:
        cleaned = hashlib.sha256(candidate.img_id.encode()).hexdigest()[:16]
    return f"row-{candidate.row_idx:05d}_{cleaned}"


def _verify_image(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        image_format = str(image.format or "unknown")
        image.convert("RGB").getpixel((0, 0))
    return width, height, image_format


def is_allowed_image_content_type(content_type: str) -> bool:
    """Allow declared images and generic binary assets verified later by Pillow."""

    normalized = content_type.lower().split(";", maxsplit=1)[0].strip()
    return (
        not normalized
        or normalized.startswith("image/")
        or normalized in {"application/octet-stream", "binary/octet-stream"}
    )


def _download_image(
    session: requests.Session, candidate: Candidate, destination: Path
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        temporary_path = destination.with_suffix(destination.suffix + ".part")
        response = session.get(candidate.image_url, stream=True, timeout=(30, 180))
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if not is_allowed_image_content_type(content_type):
            raise ValueError(
                f"Expected image response for SID-Set row {candidate.row_idx}, "
                f"received {content_type!r}"
            )
        downloaded_bytes = 0
        with temporary_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > MAX_IMAGE_BYTES:
                        raise ValueError(
                            f"SID-Set row {candidate.row_idx} exceeded the "
                            f"{MAX_IMAGE_BYTES}-byte safety limit"
                        )
                    handle.write(chunk)
        width, height, _ = _verify_image(temporary_path)
        if width != candidate.source_width or height != candidate.source_height:
            raise ValueError(
                f"Downloaded dimensions changed for row {candidate.row_idx}: "
                f"expected {(candidate.source_width, candidate.source_height)}, "
                f"found {(width, height)}"
            )
        temporary_path.replace(destination)

    width, height, image_format = _verify_image(destination)
    if width != candidate.source_width or height != candidate.source_height:
        raise ValueError(
            f"Downloaded dimensions changed for row {candidate.row_idx}: "
            f"expected {(candidate.source_width, candidate.source_height)}, "
            f"found {(width, height)}"
        )
    return {
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "width": width,
        "height": height,
        "format": image_format,
    }


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("SID-Set outputs must remain inside the project") from exc


def _atomic_csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def prepare_subset(protocol_path: Path, project_root: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    image_root = project_root / protocol["download"]["image_root"]
    manifest_path = project_root / protocol["download"]["manifest"]
    provenance_path = project_root / protocol["download"]["provenance"]

    session = _session()
    candidates, pages_fetched, candidate_counts = collect_candidates(session, protocol)
    target_per_class = int(protocol["sample"]["target_per_class"])
    seed = int(protocol["sample"]["seed"])
    candidates_by_label = {
        label: sorted(
            (row for row in candidates if row.project_label == label),
            key=lambda row: candidate_rank(row, seed),
        )
        for label in (0, 1)
    }
    owners_by_sha256: dict[str, Candidate] = {}
    accepted: list[tuple[Candidate, Path, dict[str, Any]]] = []
    excluded_duplicates: list[dict[str, Any]] = []
    progress = tqdm(
        total=2 * target_per_class,
        desc="SID-Set unique images",
        unit="image",
    )
    try:
        for label in (0, 1):
            accepted_for_label = 0
            for candidate in candidates_by_label[label]:
                destination = (
                    image_root / candidate.class_name / f"{_safe_stem(candidate)}.jpg"
                )
                verification = _download_image(session, candidate, destination)
                duplicate_of = register_unique_content(
                    candidate, verification["sha256"], owners_by_sha256
                )
                if duplicate_of is not None:
                    excluded_duplicates.append(
                        {
                            "row_idx": candidate.row_idx,
                            "img_id": candidate.img_id,
                            "project_label": candidate.project_label,
                            "sha256": verification["sha256"],
                            "duplicate_of_row_idx": duplicate_of.row_idx,
                            "reason": "byte_identical_same_label_content",
                        }
                    )
                    continue
                accepted.append((candidate, destination, verification))
                accepted_for_label += 1
                progress.update(1)
                if accepted_for_label == target_per_class:
                    break
            if accepted_for_label != target_per_class:
                raise ValueError(
                    f"Only {accepted_for_label} content-unique images available for "
                    f"project label {label}; need {target_per_class}"
                )
    finally:
        progress.close()

    accepted.sort(key=lambda row: candidate_rank(row[0], seed))
    manifest_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for candidate, destination, verification in accepted:
        relative_path = _project_relative(destination, project_root)
        manifest_rows.append(
            {
                "image_path": relative_path,
                "label": candidate.project_label,
                "class_name": candidate.class_name,
                "source": "sid_set_flux_heldout",
                "split": "heldout",
            }
        )
        provenance_rows.append(
            {
                "row_idx": candidate.row_idx,
                "img_id": candidate.img_id,
                "source_label": candidate.source_label,
                "project_label": candidate.project_label,
                "image_path": relative_path,
                **verification,
            }
        )

    labels = Counter(int(row["label"]) for row in manifest_rows)
    if labels != Counter({0: int(protocol["sample"]["target_per_class"]), 1: int(protocol["sample"]["target_per_class"])}):
        raise ValueError(f"Downloaded manifest is not balanced: {labels}")
    if len({row["image_path"] for row in manifest_rows}) != len(manifest_rows):
        raise ValueError("Downloaded manifest contains duplicate paths")

    _atomic_csv_write(manifest_path, manifest_rows)
    provenance = {
        "experiment": "section4b_sid_set_flux_heldout_preparation",
        "protocol": {
            "path": _project_relative(protocol_path, project_root),
            "sha256": sha256_file(protocol_path),
            "version": protocol["protocol_version"],
        },
        "dataset": {
            "name": protocol["dataset"]["name"],
            "repository": protocol["dataset"]["repository"],
            "source_revision": protocol["dataset"]["source_revision"],
            "source_split": protocol["dataset"]["source_split"],
            "declared_license": protocol["dataset"]["declared_license"],
            "attribution_required": True,
        },
        "sampling": {
            "seed": int(protocol["sample"]["seed"]),
            "pages_fetched": pages_fetched,
            "candidate_counts": {
                "real_0": candidate_counts[0],
                "ai_generated_1": candidate_counts[1],
            },
            "selected_counts": {
                "real_0": labels[0],
                "ai_generated_1": labels[1],
            },
            "excluded_source_label_2": True,
        },
        "manifest": {
            "path": _project_relative(manifest_path, project_root),
            "sha256": sha256_file(manifest_path),
            "rows": len(manifest_rows),
        },
        "download": {
            "source_urls_persisted": False,
            "all_images_verified": True,
            "total_bytes": sum(row["bytes"] for row in provenance_rows),
            "content_unique_image_sha256": len(
                {row["sha256"] for row in provenance_rows}
            ),
            "excluded_exact_duplicates": excluded_duplicates,
            "unreferenced_downloads_may_remain_in_ignored_raw_directory": True,
        },
        "images": provenance_rows,
        "organiser_validation_subset_used": False,
    }
    atomic_json_write(provenance_path, provenance)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic 1,000-real/1,000-FLUX SID-Set subset."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    protocol_path = (
        args.protocol if args.protocol.is_absolute() else project_root / args.protocol
    )
    result = prepare_subset(protocol_path, project_root)
    print(
        "PASS SID-Set held-out preparation: "
        f"samples={result['manifest']['rows']}, "
        f"bytes={result['download']['total_bytes']}, "
        f"manifest={result['manifest']['path']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SID-Set held-out preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
