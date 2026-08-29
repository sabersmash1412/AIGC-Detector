"""Validate the frozen OpenCLIP feature extractor on two CIFAKE images."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import open_clip
import torch
from PIL import Image

from src.clip_features import (
    CLIP_FEATURE_DIMENSION,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    encode_normalized_images,
    load_frozen_clip,
)
from src.device import choose_device


DEFAULT_MODEL_NAME = CLIP_MODEL_NAME
DEFAULT_PRETRAINED = CLIP_PRETRAINED
EXPECTED_FEATURE_DIMENSION = CLIP_FEATURE_DIMENSION
REQUIRED_MANIFEST_COLUMNS = {"image_path", "label", "class_name", "source", "split"}


def select_real_and_fake(manifest_path: Path) -> list[dict[str, str]]:
    """Return the first real and first generated-image record in a manifest."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    selected: dict[str, dict[str, str]] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_MANIFEST_COLUMNS.difference(columns)
        if missing:
            raise ValueError(
                f"Manifest {manifest_path} is missing columns: {sorted(missing)}"
            )

        for row in reader:
            label = row["label"]
            if label in {"0", "1"} and label not in selected:
                selected[label] = row
            if selected.keys() == {"0", "1"}:
                break

    missing_labels = {"0", "1"}.difference(selected)
    if missing_labels:
        raise ValueError(
            f"Manifest must contain labels 0 and 1; missing: {sorted(missing_labels)}"
        )
    return [selected["0"], selected["1"]]


def validate_features(features: torch.Tensor, expected_rows: int = 2) -> dict[str, Any]:
    """Validate normalized CLIP features and return report-safe diagnostics."""

    expected_shape = (expected_rows, EXPECTED_FEATURE_DIMENSION)
    if tuple(features.shape) != expected_shape:
        raise ValueError(
            f"Expected CLIP feature shape {expected_shape}, got {tuple(features.shape)}"
        )
    if not bool(torch.isfinite(features).all()):
        raise ValueError("CLIP features contain NaN or infinite values")

    norms = torch.linalg.vector_norm(features, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
        raise ValueError(f"CLIP features are not L2-normalized; norms={norms.tolist()}")

    return {
        "shape": list(features.shape),
        "dtype": str(features.dtype).removeprefix("torch."),
        "all_finite": True,
        "l2_norms": [round(float(value), 6) for value in norms.tolist()],
        "real_fake_cosine_similarity": round(
            float(torch.sum(features[0] * features[1]).item()), 6
        ),
    }


def atomic_json_write(output_path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON report without leaving a partially written destination."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check frozen OpenCLIP feature extraction on one real and one fake image."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/val.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/clip_environment_check.json")
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("checkpoints/open_clip")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    device = choose_device(args.device)
    rows = select_real_and_fake(args.manifest)

    print(
        f"Loading OpenCLIP model={args.model_name}, pretrained={args.pretrained}, "
        f"device={device.type}",
        flush=True,
    )
    model, preprocess = load_frozen_clip(
        device,
        args.cache_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
    )

    tensors: list[torch.Tensor] = []
    samples: list[dict[str, Any]] = []
    for row in rows:
        image_path = project_root / row["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Manifest image not found: {image_path}")
        with Image.open(image_path) as image:
            tensors.append(preprocess(image.convert("RGB")))
        samples.append(
            {
                "image_path": row["image_path"],
                "label": int(row["label"]),
                "class_name": row["class_name"],
                "source": row["source"],
                "split": row["split"],
            }
        )

    image_batch = torch.stack(tensors).to(device)
    features = encode_normalized_images(model, image_batch).cpu()

    feature_diagnostics = validate_features(features)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters != 0:
        raise ValueError(f"CLIP must be frozen, but {trainable_parameters:,} parameters can train")

    report = {
        "check": "frozen_clip_environment_and_feature_sanity",
        "status": "passed",
        "model": {
            "library": "open_clip_torch",
            "library_version": open_clip.__version__,
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "feature_dimension": features.shape[1],
        },
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "requested_device": args.device,
            "selected_device": device.type,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "input_batch_shape": list(image_batch.shape),
        "preprocessing": repr(preprocess),
        "samples": samples,
        "features": feature_diagnostics,
        "interpretation": (
            "This verifies feature extraction only; the cosine similarity is not a "
            "classification result and no classifier was trained."
        ),
    }
    atomic_json_write(args.output, report)

    print(
        f"PASS: input={tuple(image_batch.shape)}, features={tuple(features.shape)}, "
        f"finite={feature_diagnostics['all_finite']}, "
        f"norms={feature_diagnostics['l2_norms']}",
        flush=True,
    )
    print(
        f"Frozen parameters: {trainable_parameters:,}/{total_parameters:,}; "
        f"report={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CLIP check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
