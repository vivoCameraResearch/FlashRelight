#!/usr/bin/env python3
"""Gradio demo for FlashRelight.

    python app.py                       # http://0.0.0.0:7860
    python app.py --share               # public gradio link
    python app.py --device cuda:1 --steps 30
    python app.py --delight_only            # load just FlashDelight
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gradio as gr  # noqa: E402

from flashrelight.pipeline import (  # noqa: E402
    DEFAULT_FPS,
    FlashRelightPipeline,
    RelightRequest,
)
from flashrelight.utils import video_io as vio  # noqa: E402

ASSETS = REPO_ROOT / "assets"
PIPE: FlashRelightPipeline | None = None
TMP = Path(tempfile.gettempdir()) / "flashrelight_demo"
TMP.mkdir(parents=True, exist_ok=True)

EXAMPLE_PROMPTS = {
    "Dappled light": "Across the face, irregular dappled light patches wander with an organic motion.",
    "Warm firelight": "Warm orange firelight flickers from the lower left, casting "
                      "soft moving shadows across the face while the right side stays in cool shadow.",
    "Passing headlights": "A cool blue key light sweeps slowly from the left to the right of the face, "
                          "like a passing car headlight at night.",
    "Studio pulse": "Bright white studio light pulses gently from the front, with a soft rim "
                    "light separating the subject from the dark background.",
    "Golden hour": "Golden hour sunlight from a low angle on the right, slowly dimming as clouds pass by.",
    "Red strobe": "A harsh red strobe pulses rapidly from the right, plunging the face into darkness "
                  "between flashes.",
}

DELIGHT_MODE = "Delighting only"
MODES = ["Env map", "Light reference", "Text", DELIGHT_MODE]
MODE_KEY = {"Env map": "env", "Text": "text", "Light reference": "light_ref"}

_HEADER = """
<div align="center">
<h1>FlashRelight</h1>
<p>Change the lighting in a portrait video with an HDR panorama, an environment-map sequence or a lighting reference.
Text control supports a limited range of lighting descriptions. Select Delighting only to suppress the source illumination.</p>
</div>
"""

_FOOTNOTE = """
<div align="center">
<sub>Output: 49 frames at 480 × 832 pixels.
<a href="https://github.com/vivoCameraResearch/FlashRelight">GitHub</a></sub>
</div>
"""


def _env_choices() -> list[str]:
    env_dir = ASSETS / "envmaps"
    if not env_dir.exists():
        return []
    dirs = [str(p) for p in sorted(env_dir.iterdir()) if p.is_dir()]
    files = [str(p) for p in sorted(env_dir.iterdir())
             if p.is_file() and p.suffix.lower() in vio.HDR_EXTS | {".npy"}]
    return dirs + files


def _asset_videos(sub: str) -> list[str]:
    d = ASSETS / sub
    return [str(p) for p in sorted(d.glob("*.mp4"))] if d.exists() else []


def run(
    source,
    mode,
    env_choice,
    env_upload,
    prompt,
    light_ref_choice,
    light_ref_upload,
    matting,
    steps,
    seed,
    use_image_ref,
    env_rotation_speed=0.0,
    env_scale=vio.DEFAULT_ENV_SCALE,
    progress=gr.Progress(track_tqdm=False),
):
    if PIPE is None:
        raise gr.Error("pipeline is not loaded")
    if not source:
        raise gr.Error("Please upload or select a source video first.")

    if mode == DELIGHT_MODE:
        progress(0.03, desc="delighting the portrait")
        frames = PIPE.preprocess(source, matting=matting)
        delit_u8 = PIPE.delight_frames(frames, seed=int(seed), steps=int(steps))
        run_dir = Path(tempfile.mkdtemp(dir=TMP))
        stem = f"{Path(source).stem}_{int(seed)}"
        delit = vio.save_mp4(delit_u8, run_dir / f"{stem}_delit.mp4", fps=DEFAULT_FPS)
        compare = vio.save_mp4(vio.delight_comparison(frames, delit_u8),
                               run_dir / f"{stem}_delit_compare.mp4", fps=DEFAULT_FPS)
        progress(1.0, desc="finished")
        return ([(delit, "Delighted portrait"), (compare, "Source and delighted portrait")],
                f"Delighting complete · {int(steps)} steps · seed {int(seed)}", None, delit, compare)

    mode_key = MODE_KEY[mode]
    env_map = light_ref = None
    if mode_key == "env":
        env_map = env_upload or env_choice
        if not env_map:
            raise gr.Error("Select an environment map or upload an .exr, .hdr or .npy file.")
    elif mode_key == "text":
        if not str(prompt).strip():
            raise gr.Error("Enter a lighting description (or pick one below).")
    else:
        light_ref = light_ref_upload or light_ref_choice
        if not light_ref:
            raise gr.Error("Select or upload a lighting-reference video.")

    req = RelightRequest(
        source=source,
        mode=mode_key,
        env_map=env_map,
        env_rotation_speed=(env_rotation_speed or 0.0) if mode_key == "env" else 0.0,
        env_scale=vio.DEFAULT_ENV_SCALE if env_scale is None else env_scale,
        prompt=prompt or "",
        light_ref=light_ref,
        seed=int(seed),
        use_image_ref=bool(use_image_ref),
    ).validate()

    progress(0.03, desc="preprocessing input")
    res = PIPE.run(req, matting=matting, steps=int(steps), progress=False)

    run_dir = Path(tempfile.mkdtemp(dir=TMP))
    stem = f"{res.sample_id}_{res.mode}_{int(seed)}"
    relit = vio.save_mp4(res.relit_u8, run_dir / f"{stem}_relit.mp4", fps=DEFAULT_FPS)
    delit = vio.save_mp4(res.delit_u8, run_dir / f"{stem}_delit.mp4", fps=DEFAULT_FPS)
    compare = vio.save_mp4(res.comparison(), run_dir / f"{stem}_compare.mp4", fps=DEFAULT_FPS)

    gallery = [
        (compare, "source ▸ control ▸ delit ▸ relit"),
        (relit, "relit result"),
        (delit, "stage 1 · delit"),
    ]
    info = (f"**done** · delight {res.timings['delight']:.0f}s · "
            f"relight {res.timings['relight']:.0f}s · {int(steps)} steps · seed {int(seed)}")
    progress(1.0, desc="finished")
    return gallery, info, relit, delit, compare


def build_ui(default_steps: int = 50, default_matting: str = "auto",
             delight_only: bool = False) -> gr.Blocks:
    env_choices = _env_choices()
    light_choices = _asset_videos("light_refs")
    source_examples = _asset_videos("sources")

    title = "FlashDelight" if delight_only else "FlashRelight"
    with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# FlashDelight\nSuppress the source lighting in a portrait video."
                    if delight_only else _HEADER)

        with gr.Row(equal_height=False):
            # ------------------------------------------------ left: inputs
            with gr.Column(scale=5):
                gr.Markdown("### 1 · Source")
                source = gr.Video(label="Portrait video", height=300,
                                  sources=["upload", "webcam"])
                if source_examples:
                    gr.Examples(source_examples, inputs=source,
                                label="Bundled examples", examples_per_page=8)

                gr.Markdown("### 2 · Delighting" if delight_only else "### 2 · Task")
                mode = gr.Radio([DELIGHT_MODE] if delight_only else MODES,
                                value=DELIGHT_MODE if delight_only else "Env map",
                                label=None, container=False, visible=not delight_only)

                with gr.Group(visible=not delight_only) as env_group:
                    env_choice = gr.Dropdown(
                        [(Path(path).name, path) for path in env_choices],
                        value=env_choices[0] if env_choices else None,
                        label="Environment map")
                    env_upload = gr.File(label="…or upload a panorama or sequence (.exr / .hdr / .npy)",
                                         file_types=[".npy", ".exr", ".hdr"],
                                         type="filepath")
                    env_scale = gr.Number(value=vio.DEFAULT_ENV_SCALE, minimum=0, step=0.1,
                                          label="HDR intensity scale",
                                          info="Default 0.1. Use 1 for a map already scaled by 0.1.")
                    env_rotation_speed = gr.Number(value=0, label="Panorama rotation (°/s)",
                                                   info="0 keeps a single map fixed. Positive moves right, negative left. "
                                                        "For an existing sequence, use 0.")

                with gr.Group(visible=False) as text_group:
                    prompt = gr.Textbox(label="Lighting description", lines=3,
                                        value=EXAMPLE_PROMPTS["Dappled light"])
                    gr.Examples([[text] for text in EXAMPLE_PROMPTS.values()], inputs=[prompt],
                                label="Prompt presets", examples_per_page=5)

                with gr.Group(visible=False) as ref_group:
                    light_ref_choice = gr.Dropdown(
                        light_choices, value=light_choices[0] if light_choices else None,
                        label="Built-in lighting reference",
                        info="The lighting of this clip is transferred onto your subject")
                    light_ref_upload = gr.Video(label="…or upload a reference video", height=200)

                with gr.Accordion("⚙️ Advanced", open=False):
                    with gr.Row():
                        steps = gr.Slider(10, 50, value=default_steps, step=1,
                                          label="Steps per stage",
                                          info="Fewer steps reduce sampling time")
                        seed = gr.Number(value=1, precision=0, label="Seed")
                    with gr.Row():
                        matting = gr.Radio(["auto", "birefnet", "none"], value=default_matting,
                                           label="Matting",
                                           info="'none' if the clip already has a black background")
                        use_image_ref = gr.Checkbox(True, label="Appearance anchor", visible=not delight_only,
                                                    info="preserves skin tone & texture")

                run_btn = gr.Button("Delight" if delight_only else "Relight", variant="primary", size="lg")

            # ------------------------------------------------ right: results
            with gr.Column(scale=6):
                gr.Markdown("### 3 · Result")
                gallery = gr.Gallery(label=None, columns=1, rows=1, height=430,
                                     object_fit="contain", show_download_button=True,
                                     interactive=False)
                info = gr.Markdown()
                with gr.Accordion("All outputs", open=False):
                    out_relit = gr.Video(label="Relit result", height=300, visible=not delight_only)
                    out_delit = gr.Video(label="Stage 1 · delit", height=300)
                    out_compare = gr.Video(label="Comparison", height=300)

        gr.Markdown(_FOOTNOTE)

        def _toggle(choice):
            return (gr.update(visible=choice == "Env map"),
                    gr.update(visible=choice == "Text"),
                    gr.update(visible=choice == "Light reference"),
                    gr.update(visible=choice != DELIGHT_MODE),
                    gr.update(visible=choice != DELIGHT_MODE),
                    gr.update(value="Delight" if choice == DELIGHT_MODE else "Relight"))

        mode.change(_toggle, mode, [env_group, text_group, ref_group, use_image_ref,
                                    out_relit, run_btn])
        run_btn.click(
            run,
            inputs=[source, mode, env_choice, env_upload, prompt, light_ref_choice,
                    light_ref_upload, matting, steps, seed, use_image_ref, env_rotation_speed, env_scale],
            outputs=[gallery, info, out_relit, out_delit, out_compare],
        )
    return demo


def main():
    global PIPE
    p = argparse.ArgumentParser()
    p.add_argument("--delight_only", action="store_true", help="launch the standalone FlashDelight demo")
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--offload", action="store_true")
    p.add_argument("--matting", type=str, default="auto",
                   choices=["auto", "birefnet", "none"])
    p.add_argument("--matting_model", type=str, default=None)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    PIPE = FlashRelightPipeline(
        weights_dir=args.weights or "weights",
        device=args.device,
        steps=args.steps,
        offload=args.offload,
        matting=args.matting,
        matting_model=args.matting_model,
        delight_only=args.delight_only,
    )
    demo = build_ui(default_steps=args.steps, default_matting=args.matting, delight_only=args.delight_only)
    demo.queue(max_size=8).launch(
        server_name=args.host, server_port=args.port, share=args.share,
        allowed_paths=[str(ASSETS), str(TMP)],
    )


if __name__ == "__main__":
    main()