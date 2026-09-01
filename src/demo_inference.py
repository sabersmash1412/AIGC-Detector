"""Frozen E5 single-image and deterministic multi-view demo inference."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

from src.clip_features import encode_normalized_images, load_frozen_clip
from src.device import choose_device
from src.image_transforms import (
    DEFAULT_ROBUSTNESS_CONDITIONS,
    TRANSFORM_SPECS,
    apply_evaluation_transform,
)
from src.linear_probe import LinearProbeCheckpoint, load_linear_probe_checkpoint
from src.robust_linear_training import sha256_file


DEFAULT_E5_CHECKPOINT = Path("checkpoints/clip_linear_e5_source_matched.npz")
DEFAULT_MODEL_CACHE = Path("checkpoints/open_clip")
EXPECTED_E5_CHECKPOINT_SHA256 = (
    "b6c25a38a86692a74280650f516105c01efbaabe91f8da728b1a455cbf1756c4"
)
EXPECTED_E5_KIND = "e5_source_matched_risk_controlled_linear_head_v1"
REAL_THRESHOLD = 0.23700000000000002
AI_THRESHOLD = 0.8170000000000001
BINARY_BENCHMARK_THRESHOLD = 0.52
TRANSFORM_SEED = 42
MAX_INPUT_PIXELS = 50_000_000
VIEW_CONDITIONS = DEFAULT_ROBUSTNESS_CONDITIONS
DECISION_REAL = "real"
DECISION_UNCERTAIN = "uncertain"
DECISION_AI = "ai_generated"
DECISION_DISPLAY = {
    DECISION_REAL: "Likely real",
    DECISION_UNCERTAIN: "Uncertain",
    DECISION_AI: "Likely AI-generated",
}


def _scalar(archive: np.lib.npyio.NpzFile, key: str) -> object:
    if key not in archive.files or archive[key].ndim != 0:
        raise ValueError(f"E5 checkpoint metadata {key!r} is missing or non-scalar")
    return archive[key].item()


def decision_for_score(
    score: float,
    *,
    real_threshold: float = REAL_THRESHOLD,
    ai_threshold: float = AI_THRESHOLD,
) -> str:
    """Map an E5 score to its frozen risk-controlled three-way decision."""

    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("E5 score must be finite and lie in [0, 1]")
    if not 0.0 <= real_threshold < ai_threshold <= 1.0:
        raise ValueError("E5 thresholds must satisfy 0 <= real < AI <= 1")
    if score <= real_threshold:
        return DECISION_REAL
    if score >= ai_threshold:
        return DECISION_AI
    return DECISION_UNCERTAIN


def validate_e5_demo_checkpoint(path: Path) -> LinearProbeCheckpoint:
    """Load the exact frozen E5 head and prove its inference metadata."""

    if sha256_file(path) != EXPECTED_E5_CHECKPOINT_SHA256:
        raise ValueError(
            "Demo requires the exact frozen E5 checkpoint; its SHA-256 identity changed"
        )
    checkpoint = load_linear_probe_checkpoint(path)
    with np.load(path, allow_pickle=False) as archive:
        expected_scalars = {
            "robust_checkpoint_kind": EXPECTED_E5_KIND,
            "experiment": "E5_source_matched_domain_adaptation",
            "selected_best_epoch": 40,
            "anchor_weight": 0.01,
            "real_threshold": REAL_THRESHOLD,
            "ai_threshold": AI_THRESHOLD,
            "binary_benchmark_threshold": BINARY_BENCHMARK_THRESHOLD,
        }
        for key, expected in expected_scalars.items():
            observed = _scalar(archive, key)
            if isinstance(expected, float):
                if not np.isclose(
                    float(observed), expected, rtol=0.0, atol=1e-12
                ):
                    raise ValueError(f"Frozen E5 checkpoint metadata changed: {key}")
            elif observed != expected:
                raise ValueError(f"Frozen E5 checkpoint metadata changed: {key}")
    if not np.isclose(
        checkpoint.threshold, BINARY_BENCHMARK_THRESHOLD, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Frozen E5 binary benchmark threshold changed")
    return checkpoint


def _validated_rgb_image(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("Upload must decode as a Pillow image")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if width * height > MAX_INPUT_PIXELS:
        raise ValueError(
            f"Image is too large ({width}×{height}); maximum is {MAX_INPUT_PIXELS:,} pixels"
        )
    return ImageOps.exif_transpose(image).convert("RGB").copy()


def _stable_image_identity(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.width}x{image.height}:".encode("utf-8"))
    digest.update(image.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ViewPrediction:
    condition: str
    display_name: str
    score: float
    decision: str
    delta_from_clean: float


@dataclass(frozen=True)
class DemoAnalysis:
    width: int
    height: int
    clean_score: float
    clean_decision: str
    robust_decision: str
    consensus_decision: str
    all_views_agree: bool
    decision_flip_count: int
    score_minimum: float
    score_maximum: float
    score_mean: float
    score_standard_deviation: float
    decision_counts: dict[str, int]
    views: tuple[ViewPrediction, ...]


def summarise_view_scores(
    conditions: tuple[str, ...], probabilities: np.ndarray
) -> DemoAnalysis:
    """Create the demo's clean, consensus, and disagreement interpretation."""

    values = np.asarray(probabilities, dtype=np.float64)
    if not conditions or conditions[0] != "clean":
        raise ValueError("Multi-view demo conditions must begin with clean")
    if values.shape != (len(conditions),):
        raise ValueError("Multi-view probabilities do not align with conditions")
    if not bool(np.isfinite(values).all()) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Multi-view probabilities must be finite and lie in [0, 1]")

    decisions = tuple(decision_for_score(float(score)) for score in values)
    counts = {
        decision: decisions.count(decision)
        for decision in (DECISION_REAL, DECISION_UNCERTAIN, DECISION_AI)
    }
    maximum_votes = max(counts.values())
    winners = [decision for decision, count in counts.items() if count == maximum_votes]
    consensus = winners[0] if len(winners) == 1 else DECISION_UNCERTAIN
    all_agree = len(set(decisions)) == 1
    clean_decision = decisions[0]
    robust_decision = clean_decision if all_agree else DECISION_UNCERTAIN
    clean_score = float(values[0])
    views = tuple(
        ViewPrediction(
            condition=condition,
            display_name=TRANSFORM_SPECS[condition].display_name,
            score=float(score),
            decision=decision,
            delta_from_clean=float(score - clean_score),
        )
        for condition, score, decision in zip(
            conditions, values, decisions, strict=True
        )
    )
    return DemoAnalysis(
        width=0,
        height=0,
        clean_score=clean_score,
        clean_decision=clean_decision,
        robust_decision=robust_decision,
        consensus_decision=consensus,
        all_views_agree=all_agree,
        decision_flip_count=sum(decision != clean_decision for decision in decisions[1:]),
        score_minimum=float(np.min(values)),
        score_maximum=float(np.max(values)),
        score_mean=float(np.mean(values)),
        score_standard_deviation=float(np.std(values)),
        decision_counts=counts,
        views=views,
    )


class E5DemoPredictor:
    """Load frozen E5 once and score clean plus six representative views."""

    def __init__(
        self,
        *,
        checkpoint_path: Path = DEFAULT_E5_CHECKPOINT,
        device_name: str = "auto",
        model_cache_dir: Path = DEFAULT_MODEL_CACHE,
        model_loader: Callable[
            [torch.device, Path], tuple[nn.Module, Callable[[Image.Image], torch.Tensor]]
        ] = load_frozen_clip,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = choose_device(device_name)
        self.checkpoint = validate_e5_demo_checkpoint(checkpoint_path)
        self.model, self.preprocess = model_loader(self.device, model_cache_dir)
        self._inference_lock = threading.Lock()

    def analyze(self, image: Image.Image) -> DemoAnalysis:
        rgb = _validated_rgb_image(image)
        image_identity = f"upload:{_stable_image_identity(rgb)}"
        transformed = [
            apply_evaluation_transform(
                rgb,
                condition,
                image_path=image_identity,
                seed=TRANSFORM_SEED,
            )
            for condition in VIEW_CONDITIONS
        ]
        tensors = torch.stack([self.preprocess(view) for view in transformed]).to(
            self.device
        )
        with self._inference_lock, torch.inference_mode():
            features = encode_normalized_images(self.model, tensors).cpu().numpy()
            probabilities = self.checkpoint.probabilities(features)
        summary = summarise_view_scores(VIEW_CONDITIONS, probabilities)
        return DemoAnalysis(
            **{
                **summary.__dict__,
                "width": rgb.width,
                "height": rgb.height,
            }
        )


def analysis_as_json(analysis: DemoAnalysis) -> str:
    """Return a compact machine-readable record for debugging or screenshots."""

    record: dict[str, Any] = {
        "image": {"width": analysis.width, "height": analysis.height},
        "clean_score": analysis.clean_score,
        "clean_decision": analysis.clean_decision,
        "robust_decision": analysis.robust_decision,
        "consensus_decision": analysis.consensus_decision,
        "all_views_agree": analysis.all_views_agree,
        "decision_flip_count": analysis.decision_flip_count,
        "score_range": [analysis.score_minimum, analysis.score_maximum],
        "score_mean": analysis.score_mean,
        "score_standard_deviation": analysis.score_standard_deviation,
        "thresholds": {
            "real_maximum": REAL_THRESHOLD,
            "ai_minimum": AI_THRESHOLD,
        },
        "views": [view.__dict__ for view in analysis.views],
    }
    return json.dumps(record, indent=2)
