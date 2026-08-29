"""Extract deterministic frozen-CLIP embeddings from dataset manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.clip_features import (
    CLIP_FEATURE_DIMENSION,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    encode_normalized_images,
    load_frozen_clip,
)
from src.data import ManifestImageDataset
from src.device import choose_device


CACHE_FORMAT_VERSION = 1
DEFAULT_OUTPUT_DIR = Path("data/features/clip_vit_b32_quickgelu_openai")
DEFAULT_SUMMARY_OUTPUT = Path("reports/clip_embedding_summary.json")
VALID_SPLITS = ("train", "val", "test")
REQUIRED_CACHE_KEYS = {
    "format_version",
    "features",
    "labels",
    "image_paths",
    "split",
    "model_name",
    "pretrained",
    "manifest_sha256",
    "normalized",
}


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(output_path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON object."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)


def atomic_feature_cache_write(
    output_path: Path,
    *,
    features: np.ndarray,
    labels: np.ndarray,
    image_paths: np.ndarray,
    split: str,
    model_name: str,
    pretrained: str,
    manifest_sha256: str,
) -> None:
    """Atomically save arrays and cache identity metadata without pickle."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory) / output_path.name
        np.savez(
            temporary_path,
            format_version=np.asarray(CACHE_FORMAT_VERSION, dtype=np.int64),
            features=features,
            labels=labels,
            image_paths=image_paths,
            split=np.asarray(split),
            model_name=np.asarray(model_name),
            pretrained=np.asarray(pretrained),
            manifest_sha256=np.asarray(manifest_sha256),
            normalized=np.asarray(True),
        )
        temporary_path.replace(output_path)


def _scalar(cache: Any, key: str) -> Any:
    value = cache[key]
    if value.ndim != 0:
        raise ValueError(f"Cache metadata {key!r} must be a scalar")
    return value.item()


def validate_feature_cache(
    cache_path: Path,
    *,
    expected_split: str,
    expected_manifest_sha256: str,
    expected_paths: Sequence[str],
    expected_labels: Sequence[int],
    model_name: str = CLIP_MODEL_NAME,
    pretrained: str = CLIP_PRETRAINED,
) -> dict[str, Any]:
    """Validate cache contents, order, provenance, and numerical invariants."""

    if not cache_path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")

    with np.load(cache_path, allow_pickle=False) as cache:
        missing = REQUIRED_CACHE_KEYS.difference(cache.files)
        if missing:
            raise ValueError(f"Feature cache is missing keys: {sorted(missing)}")

        if int(_scalar(cache, "format_version")) != CACHE_FORMAT_VERSION:
            raise ValueError("Unsupported feature-cache format version")
        if str(_scalar(cache, "split")) != expected_split:
            raise ValueError("Feature-cache split does not match the requested split")
        if str(_scalar(cache, "model_name")) != model_name:
            raise ValueError("Feature-cache model name does not match")
        if str(_scalar(cache, "pretrained")) != pretrained:
            raise ValueError("Feature-cache pretrained tag does not match")
        if str(_scalar(cache, "manifest_sha256")) != expected_manifest_sha256:
            raise ValueError("Feature cache was created from a different manifest")
        if bool(_scalar(cache, "normalized")) is not True:
            raise ValueError("Feature cache must contain normalized embeddings")

        features = cache["features"]
        labels = cache["labels"]
        image_paths = cache["image_paths"]

        expected_shape = (len(expected_paths), CLIP_FEATURE_DIMENSION)
        if features.shape != expected_shape or features.dtype != np.float32:
            raise ValueError(
                f"Expected float32 features with shape {expected_shape}, "
                f"got {features.dtype} {features.shape}"
            )
        if labels.shape != (len(expected_paths),) or labels.dtype != np.int64:
            raise ValueError("Feature-cache labels have the wrong shape or dtype")
        if image_paths.shape != (len(expected_paths),):
            raise ValueError("Feature-cache image paths have the wrong shape")
        if not np.array_equal(labels, np.asarray(expected_labels, dtype=np.int64)):
            raise ValueError("Feature-cache labels do not match manifest order")
        if image_paths.tolist() != list(expected_paths):
            raise ValueError("Feature-cache paths do not match manifest order")
        if len(set(image_paths.tolist())) != len(image_paths):
            raise ValueError("Feature-cache image paths are not unique")
        if not bool(np.isfinite(features).all()):
            raise ValueError("Feature cache contains NaN or infinite values")

        norms = np.linalg.vector_norm(features, axis=1)
        maximum_norm_error = float(np.max(np.abs(norms - 1.0)))
        if maximum_norm_error > 2e-5:
            raise ValueError(
                f"Feature cache is not L2-normalized; max error={maximum_norm_error}"
            )

        class_counts = np.bincount(labels, minlength=2)
        return {
            "samples": int(len(labels)),
            "feature_shape": list(features.shape),
            "feature_dtype": str(features.dtype),
            "all_finite": True,
            "maximum_l2_norm_error": maximum_norm_error,
            "class_counts": {
                "real_0": int(class_counts[0]),
                "ai_generated_1": int(class_counts[1]),
            },
        }


def _dataset_identity(
    dataset: ManifestImageDataset, maximum_samples: int | None
) -> tuple[list[str], list[int]]:
    records = dataset.records[:maximum_samples]
    return (
        [row["image_path"] for row in records],
        [int(row["label"]) for row in records],
    )


def _cache_name(split: str, maximum_samples: int | None) -> str:
    if maximum_samples is None:
        return f"{split}.npz"
    return f"{split}.first-{maximum_samples}.npz"


def extract_split(
    *,
    split: str,
    manifest_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    preprocess: Any,
    device: torch.device,
    batch_size: int,
    workers: int,
    overwrite: bool,
    maximum_samples: int | None,
    model_name: str,
    pretrained: str,
) -> dict[str, Any]:
    """Extract or validate one manifest split and return summary metadata."""

    dataset = ManifestImageDataset(manifest_path, transform=preprocess)
    expected_paths, expected_labels = _dataset_identity(dataset, maximum_samples)
    if maximum_samples is not None:
        dataset_for_loader = Subset(dataset, range(len(expected_paths)))
    else:
        dataset_for_loader = dataset
    manifest_digest = sha256_file(manifest_path)

    if output_path.exists() and not overwrite:
        validation = validate_feature_cache(
            output_path,
            expected_split=split,
            expected_manifest_sha256=manifest_digest,
            expected_paths=expected_paths,
            expected_labels=expected_labels,
            model_name=model_name,
            pretrained=pretrained,
        )
        print(f"Validated existing cache; skipping extraction: {output_path}", flush=True)
        return {
            **validation,
            "status": "validated_existing",
            "manifest": manifest_path.as_posix(),
            "manifest_sha256": manifest_digest,
            "cache_path": output_path.as_posix(),
            "cache_bytes": output_path.stat().st_size,
            "cache_sha256": sha256_file(output_path),
            "validation_device": device.type,
        }

    loader = DataLoader(
        dataset_for_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    observed_paths: list[str] = []
    started = time.perf_counter()

    for images, labels, image_paths in tqdm(loader, desc=f"CLIP {split}", unit="batch"):
        images = images.to(device)
        features = encode_normalized_images(model, images).cpu().numpy().astype(np.float32)
        feature_batches.append(features)
        label_batches.append(labels.numpy().astype(np.int64))
        observed_paths.extend(image_paths)

    features = np.concatenate(feature_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    if observed_paths != expected_paths:
        raise ValueError("DataLoader image order does not match manifest order")
    if not np.array_equal(labels, np.asarray(expected_labels, dtype=np.int64)):
        raise ValueError("DataLoader label order does not match manifest order")

    atomic_feature_cache_write(
        output_path,
        features=features,
        labels=labels,
        image_paths=np.asarray(observed_paths),
        split=split,
        model_name=model_name,
        pretrained=pretrained,
        manifest_sha256=manifest_digest,
    )
    elapsed_seconds = time.perf_counter() - started
    validation = validate_feature_cache(
        output_path,
        expected_split=split,
        expected_manifest_sha256=manifest_digest,
        expected_paths=expected_paths,
        expected_labels=expected_labels,
        model_name=model_name,
        pretrained=pretrained,
    )
    return {
        **validation,
        "status": "extracted",
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": manifest_digest,
        "cache_path": output_path.as_posix(),
        "cache_bytes": output_path.stat().st_size,
        "cache_sha256": sha256_file(output_path),
        "extraction_device": device.type,
        "extraction_seconds": elapsed_seconds,
        "images_per_second": len(expected_paths) / elapsed_seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract validated, normalized frozen-CLIP embeddings."
    )
    parser.add_argument(
        "--splits", nargs="+", choices=VALID_SPLITS, default=list(VALID_SPLITS)
    )
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--model-name", default=CLIP_MODEL_NAME)
    parser.add_argument("--pretrained", default=CLIP_PRETRAINED)
    parser.add_argument("--model-cache-dir", type=Path, default=Path("checkpoints/open_clip"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--maximum-samples",
        type=int,
        default=None,
        help="Debug only: extract the first N records into a distinct cache filename.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative")
    if args.maximum_samples is not None and args.maximum_samples <= 0:
        raise ValueError("maximum-samples must be positive")

    device = choose_device(args.device)
    print(
        f"Loading frozen OpenCLIP model={args.model_name}, pretrained={args.pretrained}, "
        f"device={device.type}",
        flush=True,
    )
    model, preprocess = load_frozen_clip(
        device,
        args.model_cache_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    existing_summary: dict[str, Any] = {}
    if args.summary_output.is_file():
        existing_summary = json.loads(args.summary_output.read_text(encoding="utf-8"))
    matching_existing = (
        existing_summary.get("model", {}).get("model_name") == args.model_name
        and existing_summary.get("model", {}).get("pretrained") == args.pretrained
    )
    split_summaries = dict(existing_summary.get("splits", {})) if matching_existing else {}

    for split in args.splits:
        manifest_path = args.manifest_dir / f"{split}.csv"
        output_path = args.output_dir / _cache_name(split, args.maximum_samples)
        split_key = split if args.maximum_samples is None else f"{split}_first_{args.maximum_samples}"
        result = extract_split(
            split=split,
            manifest_path=manifest_path,
            output_path=output_path,
            model=model,
            preprocess=preprocess,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            overwrite=args.overwrite,
            maximum_samples=args.maximum_samples,
            model_name=args.model_name,
            pretrained=args.pretrained,
        )
        previous_result = split_summaries.get(split_key, {})
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
        split_summaries[split_key] = result
        print(
            f"PASS {split}: samples={result['samples']}, "
            f"shape={result['feature_shape']}, cache={result['cache_path']}",
            flush=True,
        )

    summary = {
        "experiment": "frozen_clip_embedding_cache",
        "purpose": "Reusable clean-image embeddings; no classifier training or evaluation.",
        "cache_format_version": CACHE_FORMAT_VERSION,
        "model": {
            "library": "open_clip_torch",
            "library_version": open_clip.__version__,
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "total_parameters": total_parameters,
            "trainable_parameters": 0,
            "feature_dimension": CLIP_FEATURE_DIMENSION,
            "l2_normalized": True,
        },
        "splits": split_summaries,
    }
    atomic_json_write(args.summary_output, summary)
    print(f"Updated summary: {args.summary_output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CLIP feature extraction failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
