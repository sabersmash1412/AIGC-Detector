#!/usr/bin/env python3
"""Extract frozen CLIP embeddings for the amended AIGIBench external audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import open_clip
from tqdm import tqdm

from scripts.extract_clip_features import atomic_json_write, extract_split, sha256_file
from src.clip_features import (
    CLIP_FEATURE_DIMENSION,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    load_frozen_clip,
)
from src.device import choose_device
from src.e5_aigibench_amendment import validate_amendment


DEFAULT_AMENDMENT = Path("configs/e5_aigibench_deduplication_amendment.json")
DEFAULT_OUTPUT = Path(
    "data/features/e5_aigibench_midjourney_clip_vit_b32_quickgelu_openai.npz"
)
DEFAULT_SUMMARY = Path("reports/e5_aigibench_clip_embedding_summary.json")
EXTERNAL_SPLIT = "external_test"


def validate_manifest_rows(
    rows: list[dict[str, str]], *, target_per_class: int
) -> dict[str, Any]:
    if len(rows) != 2 * target_per_class:
        raise ValueError("AIGIBench manifest has an unexpected row count")
    labels = Counter(row.get("label") for row in rows)
    classes = Counter(row.get("class_name") for row in rows)
    if labels != Counter({"0": target_per_class, "1": target_per_class}):
        raise ValueError(f"AIGIBench manifest is not label balanced: {labels}")
    if classes != Counter({"real": target_per_class, "ai_generated": target_per_class}):
        raise ValueError(f"AIGIBench manifest class names changed: {classes}")
    if {row.get("source") for row in rows} != {
        "aigibench_midjourney_v6_external"
    }:
        raise ValueError("AIGIBench manifest source changed")
    if {row.get("split") for row in rows} != {EXTERNAL_SPLIT}:
        raise ValueError("AIGIBench manifest split changed")
    paths = [str(row.get("image_path")) for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("AIGIBench manifest paths are not unique")
    return {
        "samples": len(rows),
        "class_counts": {
            "real_0": labels["0"],
            "ai_generated_1": labels["1"],
        },
        "unique_paths": len(set(paths)),
    }


def validate_preparation_provenance(
    amendment: dict[str, Any],
    provenance: dict[str, Any],
    rows: list[dict[str, str]],
    *,
    amendment_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    if provenance["amendment"]["sha256"] != amendment_sha256:
        raise ValueError("AIGIBench provenance was made from another amendment")
    if provenance["manifest"]["sha256"] != manifest_sha256:
        raise ValueError("AIGIBench provenance and manifest hashes do not match")
    if provenance["base_protocol"] != amendment["base_protocol"]:
        raise ValueError("AIGIBench base protocol changed")
    if provenance["duplicate_audit"] != amendment["duplicate_audit"]:
        raise ValueError("AIGIBench duplicate audit changed")
    if provenance["dataset"]["organiser_validation_subset_used"] is not False:
        raise ValueError("Organiser validation data must remain unused")
    if provenance["dataset"]["manual_image_inspection_before_scoring"] is not False:
        raise ValueError("AIGIBench images were manually inspected before scoring")

    counts = provenance["counts"]
    target = int(
        amendment["deduplication_selection"]["target_unique_images_per_class"]
    )
    expected_counts = {
        "total_selected_unique": 2 * target,
        "real_selected_unique": target,
        "ai_selected_unique": target,
        "cross_class_duplicate_groups": 0,
        "duplicates_excluded": 4,
        "development_or_prior_audit_overlap": 0,
    }
    for key, expected in expected_counts.items():
        if int(counts[key]) != expected:
            raise ValueError(f"AIGIBench provenance count changed: {key}")
    selection = provenance["selection"]
    if int(selection["real_candidates_examined"]) != 1000:
        raise ValueError("AIGIBench real candidate count changed")
    if int(selection["ai_candidates_examined"]) != 1004:
        raise ValueError("AIGIBench AI candidate count changed")
    if selection["algorithm"] != amendment["deduplication_selection"]["algorithm"]:
        raise ValueError("AIGIBench deduplication algorithm changed")
    if selection["real_duplicates_excluded"]:
        raise ValueError("Unexpected real-image duplicate exclusion")
    if len(selection["ai_duplicates_excluded"]) != 4:
        raise ValueError("Expected exactly four AI duplicate exclusions")

    images = provenance["images"]
    if len(images) != len(rows):
        raise ValueError("AIGIBench provenance image count changed")
    image_paths = [str(item["image_path"]) for item in images]
    content_hashes = [str(item["sha256"]) for item in images]
    if image_paths != [row["image_path"] for row in rows]:
        raise ValueError("AIGIBench provenance order differs from the manifest")
    if [int(item["label"]) for item in images] != [int(row["label"]) for row in rows]:
        raise ValueError("AIGIBench provenance labels differ from the manifest")
    if len(set(content_hashes)) != len(content_hashes):
        raise ValueError("AIGIBench selected provenance contains duplicate content")
    return {
        "unique_image_sha256": len(set(content_hashes)),
        "duplicates_excluded": 4,
        "real_candidates_examined": int(selection["real_candidates_examined"]),
        "ai_candidates_examined": int(selection["ai_candidates_examined"]),
        "final_selected_member_list_sha256": selection[
            "final_selected_member_list_sha256"
        ],
        "development_or_prior_audit_overlap": 0,
        "organiser_validation_subset_used": False,
    }


def validate_prepared_external_set(
    *,
    project_root: Path,
    amendment_path: Path,
    amendment: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    manifest_path = project_root / amendment["outputs"]["manifest"]
    provenance_path = project_root / amendment["outputs"]["provenance"]
    if not manifest_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(
            "Run scripts.prepare_e5_external_aigibench_deduplicated first"
        )
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    target = int(
        amendment["deduplication_selection"]["target_unique_images_per_class"]
    )
    manifest_validation = validate_manifest_rows(rows, target_per_class=target)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance_validation = validate_preparation_provenance(
        amendment,
        provenance,
        rows,
        amendment_sha256=sha256_file(amendment_path),
        manifest_sha256=sha256_file(manifest_path),
    )
    provenance_by_path = {
        str(item["image_path"]): str(item["sha256"])
        for item in provenance["images"]
    }
    for row in tqdm(rows, desc="E5 external file hashes", unit="image"):
        path = project_root / row["image_path"]
        if not path.is_file():
            raise FileNotFoundError(f"AIGIBench manifest image is missing: {path}")
        if sha256_file(path) != provenance_by_path[row["image_path"]]:
            raise ValueError(f"AIGIBench image changed after preparation: {path}")
    return manifest_path, {
        **manifest_validation,
        **provenance_validation,
        "manifest_path": manifest_path.relative_to(project_root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "provenance_path": provenance_path.relative_to(project_root).as_posix(),
        "provenance_sha256": sha256_file(provenance_path),
        "all_manifest_images_present_and_hash_matched": True,
    }


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Artifact is outside the project: {path}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
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
    root = Path(__file__).resolve().parents[1]
    amendment_path = args.amendment if args.amendment.is_absolute() else root / args.amendment
    output_path = args.output if args.output.is_absolute() else root / args.output
    summary_path = (
        args.summary_output if args.summary_output.is_absolute() else root / args.summary_output
    )
    model_cache_dir = (
        args.model_cache_dir
        if args.model_cache_dir.is_absolute()
        else root / args.model_cache_dir
    )
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    validate_amendment(amendment, root)
    manifest_path, preparation = validate_prepared_external_set(
        project_root=root,
        amendment_path=amendment_path,
        amendment=amendment,
    )

    device = choose_device(args.device)
    print(
        f"Loading frozen OpenCLIP model={CLIP_MODEL_NAME}, pretrained={CLIP_PRETRAINED}, "
        f"device={device.type}; AIGIBench real=1000, Midjourney V6=1000",
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
        split=EXTERNAL_SPLIT,
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
        for key in ("extraction_device", "extraction_seconds", "images_per_second"):
            if key in previous_result:
                result[key] = previous_result[key]
    result["manifest"] = _relative(manifest_path, root)
    result["cache_path"] = _relative(output_path, root)
    summary = {
        "experiment": "e5_aigibench_midjourney_frozen_clip_embeddings",
        "purpose": "Fresh external embeddings only; no classifier training, model scoring, threshold selection, or evaluation.",
        "amendment": {
            "path": amendment_path.relative_to(root).as_posix(),
            "sha256": sha256_file(amendment_path),
            "version": amendment["protocol_version"],
        },
        "dataset": {
            "benchmark": "AIGIBench Midjourney V6 test subset",
            "real_source": "Open Images V7",
            "held_out_generator": "Midjourney V6",
            "license": "CC-BY-NC-SA-4.0",
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
            "e5_checkpoint_loaded": False,
            "classifier_training_performed": False,
            "threshold_selection_performed": False,
            "model_selection_performed": False,
            "predictions_or_metrics_computed": False,
            "organiser_validation_subset_used": False,
        },
    }
    atomic_json_write(summary_path, summary)
    print(
        f"PASS E5 AIGIBench CLIP: samples={result['samples']}, "
        f"shape={result['feature_shape']}, cache={result['cache_path']}",
        flush=True,
    )
    print(f"Updated summary: {summary_path.relative_to(root)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"E5 AIGIBench CLIP extraction failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
