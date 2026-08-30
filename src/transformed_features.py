"""Safe paired transformed-feature cache interfaces."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.image_transforms import TRANSFORM_SPECS
from src.linear_probe import FeatureCache, load_feature_cache


TRANSFORMED_CACHE_KIND = "paired_transformed_clip_features_v1"
REQUIRED_TRANSFORMED_KEYS = {
    "cache_kind",
    "condition",
    "transform_parameters_json",
    "transform_seed",
    "clean_cache_sha256",
}


def canonical_transform_parameters(condition: str) -> str:
    """Return stable JSON for the registered transformation parameters."""

    if condition not in TRANSFORM_SPECS or condition == "clean":
        raise ValueError(f"Expected a registered non-clean condition, got {condition!r}")
    return json.dumps(TRANSFORM_SPECS[condition].parameters, sort_keys=True, separators=(",", ":"))


def atomic_transformed_feature_cache_write(
    path: Path,
    *,
    features: np.ndarray,
    reference: FeatureCache,
    condition: str,
    transform_seed: int,
    clean_cache_sha256: str,
) -> None:
    """Atomically write a pickle-free transformed cache with pairing provenance."""

    parameters_json = canonical_transform_parameters(condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        np.savez(
            temporary_path,
            format_version=np.asarray(1, dtype=np.int64),
            features=np.asarray(features, dtype=np.float32),
            labels=np.asarray(reference.labels, dtype=np.int64),
            image_paths=np.asarray(reference.image_paths),
            split=np.asarray(reference.split),
            model_name=np.asarray(reference.model_name),
            pretrained=np.asarray(reference.pretrained),
            manifest_sha256=np.asarray(reference.manifest_sha256),
            normalized=np.asarray(True),
            cache_kind=np.asarray(TRANSFORMED_CACHE_KIND),
            condition=np.asarray(condition),
            transform_parameters_json=np.asarray(parameters_json),
            transform_seed=np.asarray(transform_seed, dtype=np.int64),
            clean_cache_sha256=np.asarray(clean_cache_sha256),
        )
        temporary_path.replace(path)


@dataclass(frozen=True)
class TransformedFeatureCache:
    """Validated transformed features paired to one clean reference cache."""

    features: np.ndarray
    labels: np.ndarray
    image_paths: np.ndarray
    split: str
    condition: str
    transform_seed: int
    clean_cache_sha256: str


def _scalar(archive: np.lib.npyio.NpzFile, key: str) -> object:
    value = archive[key]
    if value.ndim != 0:
        raise ValueError(f"Transformed-cache metadata {key!r} must be a scalar")
    return value.item()


def load_transformed_feature_cache(
    path: Path,
    *,
    expected_split: str,
    expected_condition: str,
    expected_seed: int,
    expected_clean_cache_sha256: str,
    reference: FeatureCache,
) -> TransformedFeatureCache:
    """Load a transformed cache and prove exact clean/transformed alignment."""

    base_cache = load_feature_cache(path, expected_split)
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_TRANSFORMED_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"Transformed cache is missing keys: {sorted(missing)}")
        cache_kind = str(_scalar(archive, "cache_kind"))
        condition = str(_scalar(archive, "condition"))
        parameters_json = str(_scalar(archive, "transform_parameters_json"))
        transform_seed = int(_scalar(archive, "transform_seed"))
        clean_cache_sha256 = str(_scalar(archive, "clean_cache_sha256"))

    if cache_kind != TRANSFORMED_CACHE_KIND:
        raise ValueError("Unsupported transformed-cache kind")
    if condition != expected_condition:
        raise ValueError(
            f"Expected transformed condition {expected_condition!r}, found {condition!r}"
        )
    if parameters_json != canonical_transform_parameters(condition):
        raise ValueError("Transformed-cache parameters do not match the transform registry")
    if transform_seed != expected_seed:
        raise ValueError(
            f"Expected transformation seed {expected_seed}, found {transform_seed}"
        )
    if clean_cache_sha256 != expected_clean_cache_sha256:
        raise ValueError("Transformed cache refers to a different clean feature cache")
    if base_cache.manifest_sha256 != reference.manifest_sha256:
        raise ValueError("Clean and transformed caches use different manifests")
    if base_cache.model_name != reference.model_name or base_cache.pretrained != reference.pretrained:
        raise ValueError("Clean and transformed caches use different CLIP models")
    if not np.array_equal(base_cache.labels, reference.labels):
        raise ValueError("Clean and transformed cache labels are not aligned")
    if not np.array_equal(base_cache.image_paths, reference.image_paths):
        raise ValueError("Clean and transformed cache paths are not aligned")

    return TransformedFeatureCache(
        features=base_cache.features,
        labels=base_cache.labels,
        image_paths=base_cache.image_paths,
        split=base_cache.split,
        condition=condition,
        transform_seed=transform_seed,
        clean_cache_sha256=clean_cache_sha256,
    )
