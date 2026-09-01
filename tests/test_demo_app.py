from __future__ import annotations

import gradio as gr

from src.demo_app import TABLE_HEADERS, create_demo, present_analysis
from src.demo_inference import (
    DECISION_AI,
    DECISION_REAL,
    DECISION_UNCERTAIN,
    DemoAnalysis,
    ViewPrediction,
)


def _analysis() -> DemoAnalysis:
    return DemoAnalysis(
        width=1024,
        height=768,
        clean_score=0.9,
        clean_decision=DECISION_AI,
        robust_decision=DECISION_UNCERTAIN,
        consensus_decision=DECISION_AI,
        all_views_agree=False,
        decision_flip_count=1,
        score_minimum=0.5,
        score_maximum=0.9,
        score_mean=0.8,
        score_standard_deviation=0.12,
        decision_counts={DECISION_REAL: 0, DECISION_UNCERTAIN: 1, DECISION_AI: 6},
        views=(
            ViewPrediction("clean", "Clean", 0.9, DECISION_AI, 0.0),
            ViewPrediction(
                "gaussian_blur_sigma1",
                "Blur σ=1",
                0.5,
                DECISION_UNCERTAIN,
                -0.4,
            ),
        ),
    )


class FakePredictor:
    def analyze(self, image):
        del image
        return _analysis()


def test_present_analysis_exposes_disagreement_and_view_rows() -> None:
    verdict, details, table = present_analysis(_analysis())
    assert "Uncertain" in verdict
    assert "downgraded" in verdict
    assert "Stability warning" in details
    assert table == [
        ["Clean", 0.9, "Likely AI-generated", 0.0],
        ["Blur σ=1", 0.5, "Uncertain", -0.4],
    ]
    assert TABLE_HEADERS == [
        "View",
        "AI-likelihood score",
        "Decision",
        "Δ from clean",
    ]


def test_gradio_demo_builds_without_loading_a_model() -> None:
    demo = create_demo(FakePredictor())
    assert isinstance(demo, gr.Blocks)
    config = demo.get_config_file()
    assert any(component.get("props", {}).get("label") == "Image to analyse" for component in config["components"])
    assert any(dependency.get("api_name") == "analyse_image" for dependency in config["dependencies"])

