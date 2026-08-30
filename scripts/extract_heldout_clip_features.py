"""Extract frozen CLIP embeddings for the audited SID-Set/FLUX subset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import open_clip

from scripts.extract_clip_features import atomic_json_write, extract_split, sha256_file
from scripts.prepare_sid_set_heldout import validate_protocol
from src.clip_features import (
    CLIP_FEATURE_DIMENSION,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    load_frozen_clip,
)
from src.device import choose_device


DEFAULT_PROTOCOL = Path("configs/section4b_held_out_generator.json")
DEFAULT_OUTPUT = Path(
    "data/features/clip_vit_b32_quickgelu_openai/sid_set_flux_heldout.npz"
)
DEFAULT_SUMMARY = Path("reports/section4b_clip_embedding_summary.json")
HELDOUT_SPLIT = "heldout"


def validate_manifest_rows(
    rows: list[dict[str, str]], *, target_per_class: int
) -> dict[str, Any]:
    """Validate balanced binary labels and identities without loading images."""

    if len(rows) != 2 * target_per_class:
        raise ValueError("SID-Set held-out manifest has an unexpected row count")
    labels = Counter(row.get("label") for row in rows)
    classes = Counter(row.get("class_name") for row in rows)
    if labels != Counter({"0": target_per_class, "1": target_per_class}):
        raise ValueError(f"SID-Set held-out manifest is not label balanced: {labels}")
    if classes != Counter({"real": target_per_class, "ai_generated": target_per_class}):
        raise ValueError(f"SID-Set held-out class names changed: {classes}")
    if {row.get("source") for row in rows} != {"sid_set_flux_heldout"}:
        raise ValueError("SID-Set held-out manifest has an unexpected source")
    if {row.get("split") for row in rows} != {HELDOUT_SPLIT}:
        raise ValueError("SID-Set held-out manifest has an unexpected split")
    paths = [str(row.get("image_path")) for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("SID-Set held-out manifest paths are not unique")
    return {
        "samples": len(rows),
        "class_counts": {
            "real_0": labels["0"],
            "ai_generated_1": labels["1"],
        },
        "unique_paths": len(set(paths)),
    }


def validate_preparation_provenance(
    protocol: dict[str, Any],
    provenance: dict[str, Any],
    *,
    protocol_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Prove that the local subset matches the frozen source and dedup protocol."""

    if provenance["protocol"]["sha256"] != protocol_sha256:
        raise ValueError("SID-Set provenance was made from a different protocol")
    if provenance["manifest"]["sha256"] != manifest_sha256:
        raise ValueError("SID-Set provenance and manifest hashes do not match")
    if provenance["dataset"]["source_revision"] != protocol["dataset"][
        "source_revision"
    ]:
        raise ValueError("SID-Set source revision changed")
    if provenance["sampling"]["excluded_source_label_2"] is not True:
        raise ValueError("SID-Set tampered label 2 was not excluded")
    if provenance["organiser_validation_subset_used"] is not False:
        raise ValueError("The organiser validation subset must remain untouched")

    images = provenance["images"]
    expected = int(protocol["sample"]["total_target"])
    if len(images) != expected:
        raise ValueError("SID-Set provenance has an unexpected image count")
    if len({int(row["row_idx"]) for row in images}) != expected:
        raise ValueError("SID-Set provenance source rows are not unique")
    if len({str(row["sha256"]) for row in images}) != expected:
        raise ValueError("SID-Set provenance contains duplicate image content")
    if provenance["download"]["content_unique_image_sha256"] != expected:
        raise ValueError("SID-Set content-deduplication audit is incomplete")
    return {
        "source_revision": provenance["dataset"]["source_revision"],
        "unique_source_rows": expected,
        "unique_image_sha256": expected,
        "excluded_exact_duplicates": len(
            provenance["download"]["excluded_exact_duplicates"]
        ),
        "organiser_validation_subset_used": False,
    }


def validate_prepared_subset(
    *,
    project_root: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    manifest_path = project_root / protocol["download"]["manifest"]
    provenance_path = project_root / protocol["download"]["provenance"]
    if not manifest_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(
            "Run scripts.prepare_sid_set_heldout before extracting held-out features"
        )
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifest_validation = validate_manifest_rows(
        rows, target_per_class=int(protocol["sample"]["target_per_class"])
    )
    missing_files = [
        row["image_path"]
        for row in rows
        if not (project_root / row["image_path"]).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"SID-Set manifest refers to {len(missing_files)} missing images"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance_validation = validate_preparation_provenance(
        protocol,
        provenance,
        protocol_sha256=sha256_file(protocol_path),
        manifest_sha256=sha256_file(manifest_path),
    )
    return manifest_path, {
        **manifest_validation,
        **provenance_validation,
        "manifest_path": manifest_path.relative_to(project_root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "provenance_path": provenance_path.relative_to(project_root).as_posix(),
        "provenance_sha256": sha256_file(provenance_path),
        "all_manifest_images_present": True,
    }


def _relative_artifact_path(path: Path, project_root: Path) -> str:
    """Return a portable repository-relative artifact path."""

    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Artifact path is outside the project: {path}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen CLIP features for SID-Set real versus FLUX images."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model-cache-dir", type=Path, default=Path("checkpoints/open_clip"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative")
    project_root = Path(__file__).resolve().parents[1]
    protocol_path = args.protocol if args.protocol.is_absolute() else project_root / args.protocol
    output_path = args.output if args.output.is_absolute() else project_root / args.output
    summary_path = (
        args.summary_output
        if args.summary_output.is_absolute()
        else project_root / args.summary_output
    )
    model_cache_dir = (
        args.model_cache_dir
        if args.model_cache_dir.is_absolute()
        else project_root / args.model_cache_dir
    )

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    manifest_path, preparation = validate_prepared_subset(
        project_root=project_root,
        protocol_path=protocol_path,
        protocol=protocol,
    )

    device = choose_device(args.device)
    print(
        f"Loading frozen OpenCLIP model={CLIP_MODEL_NAME}, pretrained={CLIP_PRETRAINED}, "
        f"device={device.type}; SID-Set real=1000, FLUX=1000",
        flush=True,
    )
    model, preprocess = load_frozen_clip(
        device,
        model_cache_dir,
        model_name=CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
    )
    previous_summary: dict[str, Any] = {}
    if summary_path.is_file():
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = extract_split(
        split=HELDOUT_SPLIT,
        manifest_path=manifest_path,
        output_path=output_path,
        model=model,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        overwrite=args.overwrite,
        maximum_samples=None,
        model_name=CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
    )
    previous_result = previous_summary.get("feature_cache", {})
    if (
        result["status"] == "validated_existing"
        and previous_result.get("cache_sha256") == result["cache_sha256"]
    ):
        for preserved_key in (
            "extraction_device",
            "extraction_seconds",
            "images_per_second",
        ):
            if preserved_key in previous_result:
                result[preserved_key] = previous_result[preserved_key]
    result["manifest"] = _relative_artifact_path(manifest_path, project_root)
    result["cache_path"] = _relative_artifact_path(output_path, project_root)
    summary = {
        "experiment": "section4b_sid_set_flux_frozen_clip_embeddings",
        "purpose": "External held-out-generator embeddings only; no training, threshold selection, or evaluation.",
        "protocol": {
            "path": protocol_path.relative_to(project_root).as_posix(),
            "sha256": sha256_file(protocol_path),
            "version": protocol["protocol_version"],
        },
        "dataset": {
            "name": protocol["dataset"]["name"],
            "source_split": protocol["dataset"]["source_split"],
            "source_revision": protocol["dataset"]["source_revision"],
            "held_out_generator": protocol["task"]["held_out_generator"],
            "declared_license": protocol["dataset"]["declared_license"],
        },
        "preparation_validation": preparation,
        "model": {
            "library": "open_clip_torch",
            "library_version": open_clip.__version__,
            "model_name": CLIP_MODEL_NAME,
            "pretrained": CLIP_PRETRAINED,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": 0,
            "feature_dimension": CLIP_FEATURE_DIMENSION,
            "l2_normalized": True,
        },
        "feature_cache": result,
        "frozen_guardrails": {
            "classifier_training_performed": False,
            "threshold_selection_performed": False,
            "model_selection_performed": False,
            "organiser_validation_subset_used": False,
        },
    }
    atomic_json_write(summary_path, summary)
    print(
        f"PASS SID-Set CLIP: samples={result['samples']}, "
        f"shape={result['feature_shape']}, cache={result['cache_path']}",
        flush=True,
    )
    print(f"Updated summary: {summary_path.relative_to(project_root)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SID-Set CLIP extraction failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
