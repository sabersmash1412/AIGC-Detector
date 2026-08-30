"""Cache deterministic transformed CLIP embeddings paired to clean caches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.extract_clip_features import atomic_json_write, sha256_file
from src.clip_features import (
    CLIP_FEATURE_DIMENSION,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    encode_normalized_images,
    load_frozen_clip,
)
from src.device import choose_device
from src.image_transforms import DEFAULT_ROBUSTNESS_CONDITIONS, TRANSFORM_SPECS
from src.linear_probe import FeatureCache, load_feature_cache
from src.transformed_dataset import TransformedPathDataset
from src.transformed_features import (
    atomic_transformed_feature_cache_write,
    load_transformed_feature_cache,
)


DEFAULT_CLEAN_FEATURE_DIR = Path("data/features/clip_vit_b32_quickgelu_openai")
DEFAULT_OUTPUT_DIR = Path("data/features/clip_transformed_seed42")
DEFAULT_SUMMARY_OUTPUT = Path("reports/clip_transformed_embedding_summary.json")
VALID_SPLITS = ("train", "val", "test")
DEFAULT_CONDITIONS = tuple(
    condition for condition in DEFAULT_ROBUSTNESS_CONDITIONS if condition != "clean"
)


def _limited_reference(reference: FeatureCache, maximum_samples: int | None) -> FeatureCache:
    if maximum_samples is None:
        return reference
    stop = min(maximum_samples, len(reference.labels))
    return FeatureCache(
        features=reference.features[:stop],
        labels=reference.labels[:stop],
        image_paths=reference.image_paths[:stop],
        split=reference.split,
        model_name=reference.model_name,
        pretrained=reference.pretrained,
        manifest_sha256=reference.manifest_sha256,
    )


def _cache_path(
    output_dir: Path, split: str, condition: str, maximum_samples: int | None
) -> Path:
    suffix = "" if maximum_samples is None else f".first-{maximum_samples}"
    return output_dir / split / f"{condition}{suffix}.npz"


def extract_condition(
    *,
    reference: FeatureCache,
    clean_cache_sha256: str,
    output_path: Path,
    condition: str,
    model: torch.nn.Module,
    preprocess: Any,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
    project_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Extract or validate one paired transformed-feature cache."""

    if output_path.exists() and not overwrite:
        cache = load_transformed_feature_cache(
            output_path,
            expected_split=reference.split,
            expected_condition=condition,
            expected_seed=seed,
            expected_clean_cache_sha256=clean_cache_sha256,
            reference=reference,
        )
        print(f"Validated existing cache; skipping extraction: {output_path}", flush=True)
        status = "validated_existing"
        elapsed_seconds = None
    else:
        dataset = TransformedPathDataset(
            reference.image_paths,
            reference.labels,
            condition,
            preprocess,
            seed,
            project_root,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        )
        feature_batches: list[np.ndarray] = []
        observed_label_batches: list[np.ndarray] = []
        observed_paths: list[str] = []
        started = time.perf_counter()
        for images, labels, paths in tqdm(
            loader,
            desc=f"CLIP {reference.split} {condition}",
            unit="batch",
        ):
            feature_batches.append(
                encode_normalized_images(model, images.to(device))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            observed_label_batches.append(labels.numpy().astype(np.int64))
            observed_paths.extend(paths)

        features = np.concatenate(feature_batches)
        observed_labels = np.concatenate(observed_label_batches)
        if observed_paths != reference.image_paths.tolist():
            raise ValueError("Transformed DataLoader changed the reference image order")
        if not np.array_equal(observed_labels, reference.labels):
            raise ValueError("Transformed DataLoader changed the reference labels")
        atomic_transformed_feature_cache_write(
            output_path,
            features=features,
            reference=reference,
            condition=condition,
            transform_seed=seed,
            clean_cache_sha256=clean_cache_sha256,
        )
        elapsed_seconds = time.perf_counter() - started
        cache = load_transformed_feature_cache(
            output_path,
            expected_split=reference.split,
            expected_condition=condition,
            expected_seed=seed,
            expected_clean_cache_sha256=clean_cache_sha256,
            reference=reference,
        )
        status = "extracted"

    result: dict[str, Any] = {
        "status": status,
        "split": reference.split,
        "condition": condition,
        "display_name": TRANSFORM_SPECS[condition].display_name,
        "parameters": TRANSFORM_SPECS[condition].parameters,
        "transform_seed": seed,
        "samples": int(len(cache.labels)),
        "feature_shape": list(cache.features.shape),
        "feature_dimension": CLIP_FEATURE_DIMENSION,
        "cache_path": output_path.as_posix(),
        "cache_bytes": output_path.stat().st_size,
        "cache_sha256": sha256_file(output_path),
        "clean_cache_sha256": clean_cache_sha256,
        "path_and_label_alignment_verified": True,
    }
    if elapsed_seconds is not None:
        result["extraction_device"] = device.type
        result["extraction_seconds"] = elapsed_seconds
        result["images_per_second"] = len(cache.labels) / elapsed_seconds
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract paired transformed embeddings for controlled E1-E3 training."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=VALID_SPLITS,
        default=["train", "val"],
        help="Defaults to train and val; test requires --allow-test.",
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=DEFAULT_CONDITIONS, default=list(DEFAULT_CONDITIONS)
    )
    parser.add_argument("--clean-feature-dir", type=Path, default=DEFAULT_CLEAN_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--model-cache-dir", type=Path, default=Path("checkpoints/open_clip"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--maximum-samples",
        type=int,
        default=None,
        help="Debug only: use the first N clean-cache rows and a distinct filename.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative")
    if args.maximum_samples is not None and args.maximum_samples <= 0:
        raise ValueError("maximum-samples must be positive")
    if "test" in args.splits and not args.allow_test:
        raise ValueError(
            "Test-cache extraction is locked during model development; "
            "pass --allow-test only after E2/E3 settings are frozen."
        )

    device = choose_device(args.device)
    print(
        f"Loading frozen OpenCLIP model={CLIP_MODEL_NAME}, pretrained={CLIP_PRETRAINED}, "
        f"device={device.type}",
        flush=True,
    )
    model, preprocess = load_frozen_clip(device, args.model_cache_dir)
    project_root = Path(__file__).resolve().parents[1]

    existing_summary: dict[str, Any] = {}
    if args.summary_output.is_file():
        existing_summary = json.loads(args.summary_output.read_text(encoding="utf-8"))
    cache_summaries = dict(existing_summary.get("caches", {}))

    for split in dict.fromkeys(args.splits):
        clean_cache_path = args.clean_feature_dir / f"{split}.npz"
        full_reference = load_feature_cache(clean_cache_path, split)
        reference = _limited_reference(full_reference, args.maximum_samples)
        clean_cache_sha256 = sha256_file(clean_cache_path)
        for condition in dict.fromkeys(args.conditions):
            output_path = _cache_path(
                args.output_dir, split, condition, args.maximum_samples
            )
            result = extract_condition(
                reference=reference,
                clean_cache_sha256=clean_cache_sha256,
                output_path=output_path,
                condition=condition,
                model=model,
                preprocess=preprocess,
                device=device,
                batch_size=args.batch_size,
                workers=args.workers,
                seed=args.seed,
                project_root=project_root,
                overwrite=args.overwrite,
            )
            cache_key = f"{split}/{condition}"
            if args.maximum_samples is not None:
                cache_key += f"/first_{args.maximum_samples}"
            previous_result = cache_summaries.get(cache_key, {})
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
            cache_summaries[cache_key] = result
            print(
                f"PASS {split}/{condition}: samples={result['samples']}, "
                f"shape={result['feature_shape']}, cache={result['cache_path']}",
                flush=True,
            )

    summary = {
        "experiment": "paired_transformed_clip_embedding_cache",
        "purpose": "Train/validation inputs for controlled E1-E3 robustness experiments.",
        "test_cache_policy": "Explicit --allow-test required after model settings are frozen.",
        "model": {
            "library": "open_clip_torch",
            "library_version": open_clip.__version__,
            "model_name": CLIP_MODEL_NAME,
            "pretrained": CLIP_PRETRAINED,
            "feature_dimension": CLIP_FEATURE_DIMENSION,
            "trainable_parameters": 0,
            "l2_normalized": True,
        },
        "transform_seed": args.seed,
        "conditions": list(dict.fromkeys(args.conditions)),
        "caches": cache_summaries,
    }
    atomic_json_write(args.summary_output, summary)
    print(f"Updated summary: {args.summary_output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Transformed CLIP extraction failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
