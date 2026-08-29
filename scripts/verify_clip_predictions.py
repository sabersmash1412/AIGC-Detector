"""Verify required JSON output against cached frozen-CLIP probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.linear_probe import load_feature_cache, load_linear_probe_checkpoint


REQUIRED_PREDICTION_KEYS = {"image_path", "pred"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prediction_json(path: Path) -> list[dict[str, Any]]:
    """Load and strictly validate the submission prediction schema."""

    if not path.is_file():
        raise FileNotFoundError(f"Prediction JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Prediction JSON must be a non-empty array")

    seen_paths: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or set(row) != REQUIRED_PREDICTION_KEYS:
            raise ValueError(
                f"Prediction row {index} must contain exactly {sorted(REQUIRED_PREDICTION_KEYS)}"
            )
        image_path = row["image_path"]
        probability = row["pred"]
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(f"Prediction row {index} has an invalid image_path")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError(f"Prediction row {index} has a non-numeric pred")
        if not np.isfinite(probability) or not 0.0 <= float(probability) <= 1.0:
            raise ValueError(f"Prediction row {index} has pred outside [0, 1]")
        if image_path in seen_paths:
            raise ValueError(f"Prediction JSON contains duplicate path: {image_path}")
        seen_paths.add(image_path)
        validated.append({"image_path": image_path, "pred": float(probability)})
    return validated


def verify_predictions(
    predictions: list[dict[str, Any]],
    reference_cache_path: Path,
    checkpoint_path: Path,
    tolerance: float,
) -> dict[str, Any]:
    """Compare JSON probabilities with the selected model on cached test features."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    reference = load_feature_cache(reference_cache_path, "test")
    checkpoint = load_linear_probe_checkpoint(checkpoint_path)
    expected_probabilities = checkpoint.probabilities(reference.features)
    prediction_by_path = {row["image_path"]: row["pred"] for row in predictions}
    missing_paths = [
        path for path in reference.image_paths.tolist() if path not in prediction_by_path
    ]
    if missing_paths:
        raise ValueError(
            f"Prediction JSON is missing {len(missing_paths)} reference paths; "
            f"first={missing_paths[0]}"
        )

    observed_probabilities = np.asarray(
        [prediction_by_path[path] for path in reference.image_paths.tolist()],
        dtype=np.float64,
    )
    absolute_differences = np.abs(observed_probabilities - expected_probabilities)
    maximum_difference = float(np.max(absolute_differences))
    if maximum_difference > tolerance:
        raise ValueError(
            f"JSON probabilities differ from cached-model probabilities; "
            f"max={maximum_difference}, tolerance={tolerance}"
        )

    all_probabilities = np.asarray([row["pred"] for row in predictions], dtype=np.float64)
    return {
        "status": "passed",
        "json_predictions": len(predictions),
        "schema": {"required_keys": ["image_path", "pred"], "exact_keys": True},
        "paths_unique": True,
        "probability_range": [
            float(np.min(all_probabilities)),
            float(np.max(all_probabilities)),
        ],
        "reference_predictions_compared": int(len(reference.labels)),
        "comparison_tolerance": tolerance,
        "maximum_absolute_probability_difference": maximum_difference,
        "mean_absolute_probability_difference": float(np.mean(absolute_differences)),
        "reference_cache": reference_cache_path.as_posix(),
        "linear_checkpoint": checkpoint_path.as_posix(),
    }


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate directory-inference JSON and compare reference probabilities."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=Path("data/features/clip_vit_b32_quickgelu_openai/test.npz"),
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/clip_linear_probe.npz")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/clip_inference_contract.json")
    )
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--inference-device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--inference-seconds", type=float)
    parser.add_argument("--inference-skipped", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        predictions = load_prediction_json(args.predictions)
        report = verify_predictions(
            predictions,
            args.reference_cache,
            args.checkpoint,
            args.tolerance,
        )
        report["prediction_json"] = {
            "path": args.predictions.as_posix(),
            "bytes": args.predictions.stat().st_size,
            "sha256": sha256_file(args.predictions),
        }
        if args.inference_batch_size is not None and args.inference_batch_size <= 0:
            raise ValueError("inference-batch-size must be positive")
        if args.inference_seconds is not None and args.inference_seconds <= 0.0:
            raise ValueError("inference-seconds must be positive")
        if args.inference_skipped is not None and args.inference_skipped < 0:
            raise ValueError("inference-skipped cannot be negative")
        report["observed_inference_run"] = {
            "device": args.inference_device,
            "batch_size": args.inference_batch_size,
            "wall_time_seconds": args.inference_seconds,
            "skipped_images": args.inference_skipped,
        }
        atomic_json_write(args.report, report)
    except Exception as exc:
        print(f"Prediction verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
