"""Gradio interface for the frozen E5 multi-view research demo."""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Any, Protocol

import gradio as gr
from PIL import Image

from src.demo_inference import (
    AI_THRESHOLD,
    DECISION_AI,
    DECISION_DISPLAY,
    DECISION_REAL,
    DECISION_UNCERTAIN,
    DEFAULT_E5_CHECKPOINT,
    DEFAULT_MODEL_CACHE,
    REAL_THRESHOLD,
    DemoAnalysis,
    E5DemoPredictor,
)


INITIAL_VERDICT = """
<div class="verdict-card verdict-empty">
  <div class="verdict-kicker">READY</div>
  <div class="verdict-title">Upload an image to begin</div>
  <div class="verdict-score">E5 will evaluate the clean image and six altered views.</div>
</div>
"""
INITIAL_DETAILS = (
    "The demo reports a three-way research decision and exposes transformation "
    "disagreement instead of presenting one score as proof."
)
TABLE_HEADERS = ["View", "AI-likelihood score", "Decision", "Δ from clean"]


class Predictor(Protocol):
    def analyze(self, image: Image.Image) -> DemoAnalysis: ...


def _verdict_html(analysis: DemoAnalysis) -> str:
    decision = analysis.robust_decision
    css_class = {
        DECISION_REAL: "verdict-real",
        DECISION_UNCERTAIN: "verdict-uncertain",
        DECISION_AI: "verdict-ai",
    }[decision]
    stable_text = (
        "All seven views agree."
        if analysis.all_views_agree
        else "Views disagree, so the robust conclusion is downgraded to uncertain."
    )
    return f"""
<div class="verdict-card {css_class}">
  <div class="verdict-kicker">ROBUST CONCLUSION</div>
  <div class="verdict-title">{html.escape(DECISION_DISPLAY[decision])}</div>
  <div class="verdict-score">Clean AI-likelihood score: {analysis.clean_score:.1%}</div>
  <div class="verdict-note">{html.escape(stable_text)}</div>
</div>
"""


def _details_markdown(analysis: DemoAnalysis) -> str:
    clean = DECISION_DISPLAY[analysis.clean_decision]
    consensus = DECISION_DISPLAY[analysis.consensus_decision]
    score_range = analysis.score_maximum - analysis.score_minimum
    low_resolution = min(analysis.width, analysis.height) < 128
    lines = [
        f"**Clean decision:** {clean}  ",
        f"**Majority view:** {consensus}  ",
        f"**Input:** {analysis.width} × {analysis.height} pixels  ",
        (
            f"**Multi-view range:** {analysis.score_minimum:.3f}–"
            f"{analysis.score_maximum:.3f} (spread {score_range:.3f})  "
        ),
        (
            f"**View decisions:** {analysis.decision_counts[DECISION_REAL]} real, "
            f"{analysis.decision_counts[DECISION_UNCERTAIN]} uncertain, "
            f"{analysis.decision_counts[DECISION_AI]} AI-generated  "
        ),
    ]
    if analysis.all_views_agree:
        lines.append("✅ **Stability:** no transformed view changed the three-way decision.")
    else:
        lines.append(
            f"⚠️ **Stability warning:** {analysis.decision_flip_count} of the six "
            "transformed views changed the clean decision."
        )
    if low_resolution:
        lines.append(
            "\n⚠️ This is a low-resolution image. Upscaling can remove or create "
            "signals, so interpret the result especially cautiously."
        )
    lines.append(
        "\nThis score is a model output, not proof of provenance. E5 was the strongest "
        "tested model but did not satisfy every frozen external safety gate."
    )
    return "\n".join(lines)


def _view_table(analysis: DemoAnalysis) -> list[list[Any]]:
    return [
        [
            view.display_name,
            round(view.score, 4),
            DECISION_DISPLAY[view.decision],
            round(view.delta_from_clean, 4),
        ]
        for view in analysis.views
    ]


def present_analysis(analysis: DemoAnalysis) -> tuple[str, str, list[list[Any]]]:
    """Convert a core inference result into stable Gradio component values."""

    return _verdict_html(analysis), _details_markdown(analysis), _view_table(analysis)


def create_demo(predictor: Predictor) -> gr.Blocks:
    """Build the interface without loading or mutating any model state."""

    def run(image: Image.Image | None, progress: gr.Progress = gr.Progress()):
        if image is None:
            return (
                """
<div class="verdict-card verdict-error">
  <div class="verdict-kicker">INPUT NEEDED</div>
  <div class="verdict-title">Please upload an image</div>
</div>
""",
                "Choose a JPEG, PNG, WebP, BMP or TIFF image, then select **Analyse image**.",
                [],
            )
        try:
            progress(0.05, desc="Preparing seven deterministic views")
            analysis = predictor.analyze(image)
            progress(0.9, desc="Summarising robustness")
            result = present_analysis(analysis)
            progress(1.0, desc="Complete")
            return result
        except Exception as exc:
            message = html.escape(str(exc))
            return (
                f"""
<div class="verdict-card verdict-error">
  <div class="verdict-kicker">INFERENCE ERROR</div>
  <div class="verdict-title">The image could not be analysed</div>
  <div class="verdict-note">{message}</div>
</div>
""",
                "Try another image. If the problem continues, restart the local demo and "
                "check that the frozen E5 checkpoint and OpenCLIP weights are available.",
                [],
            )

    with gr.Blocks(title="AIGC Detector — E5 Research Demo") as demo:
        gr.Markdown(
            """
# AIGC Detector
### Frozen E5 · source-matched CLIP · transformation-aware uncertainty

Upload one image. The detector evaluates its clean form plus JPEG compression,
blur, resize, noise, colour jitter and centre crop. It reveals disagreement
instead of hiding instability behind a single prediction.
""",
            elem_classes="hero-copy",
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=5, min_width=320):
                image_input = gr.Image(
                    type="pil",
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                    label="Image to analyse",
                    height=430,
                )
                with gr.Row():
                    analyse_button = gr.Button(
                        "Analyse image", variant="primary", scale=3
                    )
                    clear_button = gr.ClearButton(
                        value="Clear", components=[image_input], scale=1
                    )
            with gr.Column(scale=5, min_width=320):
                verdict = gr.HTML(INITIAL_VERDICT)
                details = gr.Markdown(INITIAL_DETAILS, elem_classes="details-panel")

        gr.Markdown("## Multi-view disagreement audit")
        view_table = gr.Dataframe(
            headers=TABLE_HEADERS,
            datatype=["str", "number", "str", "number"],
            value=[],
            interactive=False,
            type="array",
            max_height=360,
            wrap=True,
            show_row_numbers=False,
        )

        with gr.Accordion("How to interpret this demo", open=False):
            gr.Markdown(
                f"""
- Scores at or below **{REAL_THRESHOLD:.3f}** are labelled **Likely real**.
- Scores at or above **{AI_THRESHOLD:.3f}** are labelled **Likely AI-generated**.
- Scores between them are labelled **Uncertain**.
- If views cross decision regions, the robust conclusion becomes **Uncertain**.
- The encoder is frozen OpenCLIP ViT-B/32; only a 513-parameter linear head was trained.
- E5 substantially outperformed E3 and E4 externally, but it is not universally or
  production-safety validated. Use provenance metadata and human review for consequential cases.
"""
            )

        analyse_button.click(
            fn=run,
            inputs=image_input,
            outputs=[verdict, details, view_table],
            show_progress="full",
            concurrency_limit=1,
            api_name="analyse_image",
            api_description="Return frozen E5 clean and multi-view research predictions.",
        )
        clear_button.click(
            fn=lambda: (INITIAL_VERDICT, INITIAL_DETAILS, []),
            inputs=None,
            outputs=[verdict, details, view_table],
            show_progress="hidden",
            queue=False,
        )
    return demo.queue(default_concurrency_limit=1, max_size=8)


CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
.hero-copy { text-align: center; margin-bottom: 0.8rem; }
.verdict-card {
  border-radius: 18px;
  padding: 1.45rem;
  min-height: 185px;
  border: 1px solid var(--border-color-primary);
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.verdict-kicker { font-size: 0.78rem; font-weight: 750; letter-spacing: 0.13em; opacity: 0.72; }
.verdict-title { font-size: 2rem; font-weight: 750; margin: 0.3rem 0 0.25rem; }
.verdict-score { font-size: 1.05rem; font-weight: 600; }
.verdict-note { margin-top: 0.6rem; line-height: 1.45; }
.verdict-real { background: color-mix(in srgb, #16a34a 13%, var(--background-fill-primary)); border-color: #16a34a; }
.verdict-uncertain { background: color-mix(in srgb, #d97706 14%, var(--background-fill-primary)); border-color: #d97706; }
.verdict-ai { background: color-mix(in srgb, #dc2626 12%, var(--background-fill-primary)); border-color: #dc2626; }
.verdict-error { background: color-mix(in srgb, #dc2626 10%, var(--background-fill-primary)); border-color: #dc2626; }
.verdict-empty { background: var(--background-fill-secondary); }
.details-panel { padding: 0.25rem 0.25rem; }
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_E5_CHECKPOINT)
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        print(
            f"Loading frozen E5 demo on device={args.device}; checkpoint={args.checkpoint}",
            flush=True,
        )
        predictor = E5DemoPredictor(
            checkpoint_path=args.checkpoint,
            device_name=args.device,
            model_cache_dir=args.model_cache_dir,
        )
        demo = create_demo(predictor)
        print(
            "PASS E5 demo preflight: exact checkpoint, thresholds, and seven views validated",
            flush=True,
        )
        demo.launch(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,
            inbrowser=args.inbrowser,
            show_error=True,
            max_threads=2,
            max_file_size="25mb",
            theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
            css=CSS,
            ssr_mode=False,
            footer_links=["settings"],
        )
        return 0
    except Exception as exc:
        print(f"E5 demo failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
