"""Run image-directory inference and write AIGC probabilities to JSON."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from src.clip_features import encode_normalized_images, load_frozen_clip
from src.data import build_image_transform
from src.device import choose_device
from src.linear_probe import load_linear_probe_checkpoint
from src.model import load_checkpoint


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
SUPPORTED_BACKENDS = ("auto", "cnn", "clip_linear")


def resolve_backend(checkpoint_path: Path, requested_backend: str = "auto") -> str:
    """Resolve an explicit backend or infer it from the safe checkpoint suffix."""

    if requested_backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend {requested_backend!r}; choose from {SUPPORTED_BACKENDS}"
        )
    if requested_backend != "auto":
        return requested_backend
    if checkpoint_path.suffix.lower() == ".npz":
        return "clip_linear"
    if checkpoint_path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        return "cnn"
    raise ValueError(
        "Cannot infer backend from checkpoint suffix; use --backend cnn or clip_linear"
    )


def discover_images(input_dir: Path, recursive: bool = False) -> list[Path]:
    """Return deterministically ordered supported image paths."""

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _batches(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _atomic_json_write(output_path: Path, predictions: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)


def predict_directory(
    input_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    batch_size: int = 64,
    device_name: str = "auto",
    recursive: bool = False,
    strict: bool = False,
    backend_name: str = "auto",
    model_cache_dir: Path = Path("checkpoints/open_clip"),
    show_progress: bool = False,
) -> tuple[list[dict[str, Any]], list[str], torch.device]:
    """Predict every readable image and save the required JSON array.

    Returns the predictions, skipped-image messages, and selected device to aid
    testing and command-line reporting.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    paths = discover_images(input_dir, recursive=recursive)
    if not paths:
        raise ValueError(f"No supported image files found in: {input_dir}")

    device = choose_device(device_name)
    backend = resolve_backend(checkpoint_path, backend_name)
    linear_checkpoint = None
    if backend == "cnn":
        model, checkpoint = load_checkpoint(checkpoint_path)
        preprocessing = checkpoint["preprocessing"]
        transform = build_image_transform(
            image_size=int(preprocessing["image_size"]),
            mean=preprocessing["mean"],
            std=preprocessing["std"],
        )
        model.to(device)
        model.eval()
    else:
        linear_checkpoint = load_linear_probe_checkpoint(checkpoint_path)
        model, transform = load_frozen_clip(device, model_cache_dir)

    predictions: list[dict[str, Any]] = []
    skipped: list[str] = []
    path_batches = _batches(paths, batch_size)
    if show_progress:
        path_batches = tqdm(
            path_batches,
            total=math.ceil(len(paths) / batch_size),
            desc=f"Predict {backend}",
            unit="batch",
        )

    with torch.inference_mode():
        for path_batch in path_batches:
            tensors: list[torch.Tensor] = []
            readable_paths: list[Path] = []

            for path in path_batch:
                try:
                    with Image.open(path) as image:
                        tensors.append(transform(image.convert("RGB")))
                    readable_paths.append(path)
                except Exception as exc:
                    message = f"{path}: {exc}"
                    if strict:
                        raise ValueError(f"Failed to read image: {message}") from exc
                    skipped.append(message)

            if not tensors:
                continue

            images = torch.stack(tensors).to(device)
            if backend == "cnn":
                probabilities = torch.sigmoid(model(images)).detach().cpu().tolist()
            else:
                if linear_checkpoint is None:
                    raise RuntimeError("CLIP linear checkpoint was not loaded")
                features = encode_normalized_images(model, images).cpu().numpy()
                probabilities = linear_checkpoint.probabilities(features).tolist()
            predictions.extend(
                {
                    "image_path": path.as_posix(),
                    "pred": round(float(probability), 6),
                }
                for path, probability in zip(readable_paths, probabilities, strict=True)
            )

    if not predictions:
        raise ValueError("No readable images were available for inference")

    _atomic_json_write(output_path, predictions)
    return predictions, skipped, device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the probability that each image in a directory is AI-generated."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default="auto")
    parser.add_argument(
        "--model-cache-dir", type=Path, default=Path("checkpoints/open_clip")
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on the first unreadable image instead of skipping it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        predictions, skipped, device = predict_directory(
            input_dir=args.input_dir,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            batch_size=args.batch_size,
            device_name=args.device,
            recursive=args.recursive,
            strict=args.strict,
            backend_name=args.backend,
            model_cache_dir=args.model_cache_dir,
            show_progress=True,
        )
    except Exception as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 1

    for message in skipped:
        print(f"Skipped unreadable image: {message}", file=sys.stderr)
    backend = resolve_backend(args.checkpoint, args.backend)
    print(
        f"Wrote {len(predictions)} predictions to {args.output} "
        f"using backend={backend}, device={device.type}; skipped={len(skipped)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
