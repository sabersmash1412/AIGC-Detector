#!/usr/bin/env python3
"""Extract the frozen 14-condition AIGIBench transformed CLIP matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open_clip

from scripts.extract_clip_features import atomic_json_write, sha256_file
from scripts.extract_transformed_clip_features import (
    AVAILABLE_CONDITIONS,
    extract_condition,
)
from src.clip_features import (
    CLIP_FEATURE_DIMENSION,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    load_frozen_clip,
)
from src.device import choose_device
from src.e5_aigibench_amendment import validate_amendment
from src.image_transforms import FULL_ROBUSTNESS_CONDITIONS
from src.linear_probe import FeatureCache, load_feature_cache


DEFAULT_AMENDMENT = Path("configs/e5_aigibench_deduplication_amendment.json")
DEFAULT_CLEAN_CACHE = Path(
    "data/features/e5_aigibench_midjourney_clip_vit_b32_quickgelu_openai.npz"
)
DEFAULT_CLEAN_SUMMARY = Path("reports/e5_aigibench_clip_embedding_summary.json")
DEFAULT_OUTPUT_DIR = Path("data/features/e5_aigibench_midjourney_transformed_seed42")
DEFAULT_SUMMARY = Path("reports/e5_aigibench_transformed_embedding_summary.json")
EXTERNAL_SPLIT = "external_test"
TRANSFORM_SEED = 42
FROZEN_CONDITIONS = tuple(
    condition for condition in FULL_ROBUSTNESS_CONDITIONS if condition != "clean"
)


def validate_clean_reference(
    *,
    amendment: dict[str, Any],
    amendment_path: Path,
    clean_summary: dict[str, Any],
    clean_cache_path: Path,
) -> tuple[FeatureCache, str]:
    if clean_summary["amendment"]["sha256"] != sha256_file(amendment_path):
        raise ValueError("Clean AIGIBench features use a different amendment")
    guardrails = clean_summary["frozen_guardrails"]
    for key in (
        "e5_checkpoint_loaded",
        "classifier_training_performed",
        "threshold_selection_performed",
        "model_selection_performed",
        "predictions_or_metrics_computed",
        "organiser_validation_subset_used",
    ):
        if guardrails[key] is not False:
            raise ValueError(f"Clean AIGIBench guardrail failed: {key}")
    result = clean_summary["feature_cache"]
    clean_digest = sha256_file(clean_cache_path)
    if result["cache_sha256"] != clean_digest:
        raise ValueError("Clean AIGIBench cache hash differs from its summary")
    if result["manifest_sha256"] != clean_summary["preparation_validation"][
        "manifest_sha256"
    ]:
        raise ValueError("Clean AIGIBench manifest identity changed")
    if result["feature_shape"] != [2000, CLIP_FEATURE_DIMENSION]:
        raise ValueError("Clean AIGIBench feature shape changed")
    if result["class_counts"] != {"real_0": 1000, "ai_generated_1": 1000}:
        raise ValueError("Clean AIGIBench class counts changed")
    reference = load_feature_cache(
        clean_cache_path, EXTERNAL_SPLIT, require_both_labels=True
    )
    if reference.features.shape != (2000, CLIP_FEATURE_DIMENSION):
        raise ValueError("Loaded clean AIGIBench cache has the wrong shape")
    if np.bincount(reference.labels, minlength=2).tolist() != [1000, 1000]:
        raise ValueError("Loaded clean AIGIBench labels are not balanced")
    if reference.manifest_sha256 != result["manifest_sha256"]:
        raise ValueError("Loaded clean cache has a different manifest hash")
    if amendment["candidate_population"]["selection_seed"] != TRANSFORM_SEED:
        raise ValueError("Amendment seed differs from transformed-feature seed")
    return reference, clean_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--clean-cache", type=Path, default=DEFAULT_CLEAN_CACHE)
    parser.add_argument("--clean-summary", type=Path, default=DEFAULT_CLEAN_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model-cache-dir", type=Path, default=Path("checkpoints/open_clip"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _project_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Artifact is outside the project: {path}") from exc


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative")
    if tuple(AVAILABLE_CONDITIONS) != FROZEN_CONDITIONS:
        raise ValueError("Transform registry changed after the external audit was frozen")
    root = Path(__file__).resolve().parents[1]
    amendment_path = _project_path(args.amendment, root)
    clean_cache_path = _project_path(args.clean_cache, root)
    clean_summary_path = _project_path(args.clean_summary, root)
    output_dir = _project_path(args.output_dir, root)
    summary_path = _project_path(args.summary_output, root)
    model_cache_dir = _project_path(args.model_cache_dir, root)

    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    validate_amendment(amendment, root)
    clean_summary = json.loads(clean_summary_path.read_text(encoding="utf-8"))
    reference, clean_digest = validate_clean_reference(
        amendment=amendment,
        amendment_path=amendment_path,
        clean_summary=clean_summary,
        clean_cache_path=clean_cache_path,
    )

    device = choose_device(args.device)
    print(
        f"Loading frozen OpenCLIP model={CLIP_MODEL_NAME}, pretrained={CLIP_PRETRAINED}, "
        f"device={device.type}; samples=2000, transformed_conditions={len(FROZEN_CONDITIONS)}",
        flush=True,
    )
    model, preprocess = load_frozen_clip(
        device,
        model_cache_dir,
        model_name=CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
    )
    existing_summary: dict[str, Any] = {}
    if summary_path.is_file():
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cache_summaries = dict(existing_summary.get("caches", {}))

    for condition in FROZEN_CONDITIONS:
        output_path = output_dir / f"{condition}.npz"
        result = extract_condition(
            reference=reference,
            clean_cache_sha256=clean_digest,
            output_path=output_path,
            condition=condition,
            model=model,
            preprocess=preprocess,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=TRANSFORM_SEED,
            project_root=root,
            overwrite=args.overwrite,
            require_both_labels=True,
        )
        previous = cache_summaries.get(condition, {})
        if (
            result["status"] == "validated_existing"
            and previous.get("cache_sha256") == result["cache_sha256"]
        ):
            for key in ("extraction_device", "extraction_seconds", "images_per_second"):
                if key in previous:
                    result[key] = previous[key]
        result["cache_path"] = _relative(output_path, root)
        cache_summaries[condition] = result
        print(
            f"PASS external/{condition}: samples={result['samples']}, "
            f"shape={result['feature_shape']}, cache={result['cache_path']}",
            flush=True,
        )

    summary = {
        "experiment": "e5_aigibench_midjourney_frozen_transformed_clip_embeddings",
        "purpose": "Complete frozen robustness-matrix embeddings only; no E5 checkpoint, scores, thresholds, predictions, or metrics.",
        "amendment": {
            "path": _relative(amendment_path, root),
            "sha256": sha256_file(amendment_path),
            "version": amendment["protocol_version"],
        },
        "clean_reference": {
            "summary_path": _relative(clean_summary_path, root),
            "summary_sha256": sha256_file(clean_summary_path),
            "cache_path": _relative(clean_cache_path, root),
            "cache_sha256": clean_digest,
            "manifest_sha256": reference.manifest_sha256,
            "samples": 2000,
            "class_counts": {"real_0": 1000, "ai_generated_1": 1000},
        },
        "model": {
            "library": "open_clip_torch",
            "library_version": open_clip.__version__,
            "model_name": CLIP_MODEL_NAME,
            "pretrained": CLIP_PRETRAINED,
            "feature_dimension": CLIP_FEATURE_DIMENSION,
            "trainable_parameters": 0,
            "l2_normalized": True,
        },
        "transform_seed": TRANSFORM_SEED,
        "full_matrix": list(FULL_ROBUSTNESS_CONDITIONS),
        "transformed_conditions": list(FROZEN_CONDITIONS),
        "caches": cache_summaries,
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
    print(f"Updated summary: {_relative(summary_path, root)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"E5 AIGIBench transformed CLIP extraction failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
