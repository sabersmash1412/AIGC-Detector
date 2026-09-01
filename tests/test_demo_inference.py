from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from src.demo_inference import (
    AI_THRESHOLD,
    DECISION_AI,
    DECISION_REAL,
    DECISION_UNCERTAIN,
    DEFAULT_E5_CHECKPOINT,
    REAL_THRESHOLD,
    VIEW_CONDITIONS,
    DemoInputError,
    E5DemoPredictor,
    decision_for_score,
    summarise_view_scores,
    validate_e5_demo_checkpoint,
)


class FakeClip(nn.Module):
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = torch.zeros((len(images), 512), device=images.device)
        features[:, 0] = images.mean(dim=(1, 2, 3))
        features[:, 1] = images.std(dim=(1, 2, 3)) + 0.1
        return features


def _fake_loader(device: torch.device, cache_dir: Path):
    del cache_dir
    preprocess = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )
    return FakeClip().to(device).eval(), preprocess


def test_three_way_threshold_boundaries_are_inclusive() -> None:
    assert decision_for_score(0.0) == DECISION_REAL
    assert decision_for_score(REAL_THRESHOLD) == DECISION_REAL
    assert decision_for_score(REAL_THRESHOLD + 0.001) == DECISION_UNCERTAIN
    assert decision_for_score(AI_THRESHOLD - 0.001) == DECISION_UNCERTAIN
    assert decision_for_score(AI_THRESHOLD) == DECISION_AI
    assert decision_for_score(1.0) == DECISION_AI


def test_multi_view_disagreement_downgrades_robust_conclusion() -> None:
    probabilities = np.asarray([0.1, 0.2, 0.5, 0.9, 0.15, 0.85, 0.4])
    analysis = summarise_view_scores(VIEW_CONDITIONS, probabilities)
    assert analysis.clean_decision == DECISION_REAL
    assert analysis.robust_decision == DECISION_UNCERTAIN
    assert analysis.all_views_agree is False
    assert analysis.decision_flip_count == 4
    assert analysis.decision_counts == {
        DECISION_REAL: 3,
        DECISION_UNCERTAIN: 2,
        DECISION_AI: 2,
    }


def test_multi_view_agreement_preserves_decision() -> None:
    analysis = summarise_view_scores(
        VIEW_CONDITIONS, np.full(len(VIEW_CONDITIONS), 0.9)
    )
    assert analysis.clean_decision == DECISION_AI
    assert analysis.robust_decision == DECISION_AI
    assert analysis.consensus_decision == DECISION_AI
    assert analysis.all_views_agree is True
    assert analysis.decision_flip_count == 0


def test_exact_e5_checkpoint_passes_demo_validation() -> None:
    checkpoint = validate_e5_demo_checkpoint(DEFAULT_E5_CHECKPOINT)
    assert checkpoint.threshold == pytest.approx(0.52)


def test_demo_predictor_is_deterministic_for_uploaded_image() -> None:
    predictor = E5DemoPredictor(
        checkpoint_path=DEFAULT_E5_CHECKPOINT,
        device_name="cpu",
        model_loader=_fake_loader,
    )
    image = Image.new("RGB", (96, 80), (120, 80, 200))
    first = predictor.analyze(image)
    second = predictor.analyze(image)
    assert first == second
    assert first is second
    assert predictor.cached_analysis_count == 1
    assert (first.width, first.height) == (96, 80)
    assert len(first.views) == len(VIEW_CONDITIONS)
    assert all(0.0 <= view.score <= 1.0 for view in first.views)


def test_demo_predictor_cache_is_bounded_and_can_be_cleared() -> None:
    predictor = E5DemoPredictor(
        checkpoint_path=DEFAULT_E5_CHECKPOINT,
        device_name="cpu",
        model_loader=_fake_loader,
        analysis_cache_size=2,
    )
    for colour in ((1, 2, 3), (4, 5, 6), (7, 8, 9)):
        predictor.analyze(Image.new("RGB", (32, 32), colour))
    assert predictor.cached_analysis_count == 2
    predictor.clear_analysis_cache()
    assert predictor.cached_analysis_count == 0


def test_invalid_image_object_has_safe_input_error() -> None:
    predictor = E5DemoPredictor(
        checkpoint_path=DEFAULT_E5_CHECKPOINT,
        device_name="cpu",
        model_loader=_fake_loader,
    )
    with pytest.raises(DemoInputError, match="did not decode"):
        predictor.analyze(object())  # type: ignore[arg-type]
