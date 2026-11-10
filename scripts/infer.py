#!/usr/bin/env python3
"""FlashRelight inference CLI.

Single sample
-------------
    # dynamic HDR environment map
    python scripts/infer.py --source assets/sources/generated_gesture.mp4 \
        --mode env --env_map assets/envmaps/04_random_great_circle --matting none --out_dir outputs

    # natural-language lighting control
    python scripts/infer.py --source assets/sources/generated_portrait.mp4 \
        --mode text --prompt "Across the face, irregular dappled light patches wander with an organic motion." \
        --matting none --out_dir outputs

    # lighting-reference video
    python scripts/infer.py --source assets/sources/generated_gesture.mp4 \
        --mode light_ref --light_ref assets/light_refs/studio_sweep.mp4 --matting none --out_dir outputs

Batch
-----
    python scripts/infer.py --samples assets/samples.json --out_dir outputs

Delighting only
---------------
    python scripts/infer.py --delight_only --source assets/sources/generated_portrait.mp4 --matting none
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flashrelight.pipeline import (  # noqa: E402
    DEFAULT_FPS,
    FlashRelightPipeline,
    RelightRequest,
)
from flashrelight.utils import video_io as vio  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="FlashRelight: dynamic portrait video relighting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # --- inputs ---------------------------------------------------------- #
    p.add_argument("--source", type=str, default=None,
                   help="input portrait video or image (any resolution / length)")
    p.add_argument("--samples", type=str, default=None,
                   help="JSON list of jobs; overrides the single-sample flags")
    p.add_argument("--mode", type=str, default="env",
                   choices=["env", "text", "light_ref"],
                   help="lighting control modality")
    p.add_argument("--env_map", type=str, default=None,
                   help="folder of .exr frames, a single .exr/.hdr, or a .npy [T,3,H,W]")
    p.add_argument("--env_scale", type=float, default=vio.DEFAULT_ENV_SCALE, metavar="GAIN",
                   help="linear HDR multiplier (default: %(default)s). "
                        "Use 1 for a map already scaled by 0.1.")
    p.add_argument("--env_rotation_speed", type=float, default=0.0, metavar="DEG/S",
                   help="yaw a single panorama in degrees per output second "
                        "(0: fixed, positive: right, negative: left)")
    p.add_argument("--prompt", type=str, default="",
                   help="lighting description for --mode text")
    p.add_argument("--light_ref", type=str, default=None,
                   help="lighting-reference video for --mode light_ref")
    p.add_argument("--image_ref", type=str, default=None,
                   help="appearance anchor image (default: mid-frame of the source)")
    p.add_argument("--no_image_ref", action="store_true",
                   help="disable the appearance anchor branch")
    p.add_argument("--start_seconds", type=float, default=0.0,
                   help="seek into the source clip before taking 49 frames")

    # --- outputs --------------------------------------------------------- #
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--no_comparison", action="store_true",
                   help="do not write the side-by-side comparison video")
    p.add_argument("--delight_only", action="store_true",
                   help="load only FlashDelight and save the delighted portrait video")

    # --- model ----------------------------------------------------------- #
    p.add_argument("--weights", type=str, default=str(REPO_ROOT / "weights"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--steps", type=int, default=50, help="flow-matching steps")
    p.add_argument("--shift", type=float, default=5.0, help="timestep shift")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--offload", action="store_true",
                   help="keep the DiT on CPU between calls (less VRAM, slower)")
    p.add_argument("--matting", type=str, default="auto",
                   choices=["auto", "birefnet", "none"],
                   help="portrait matting for in-the-wild inputs")
    p.add_argument("--matting_model", type=str, default=None,
                   help="local BiRefNet dir or HF repo id")
    args = p.parse_args()
    if args.fps < 1:
        p.error("--fps must be positive")
    return args


def build_requests(args) -> list:
    if args.samples:
        raw = json.loads(Path(args.samples).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("samples", [])
        requests = []
        base = Path(args.samples).resolve().parent
        for item in raw:
            req = RelightRequest.from_dict(item)
            # allow sample JSONs to use paths relative to their own location
            for attr in ("source", "env_map", "light_ref", "image_ref"):
                val = getattr(req, attr, None)
                if val and not Path(val).is_absolute():
                    candidate = base / val
                    if candidate.exists() or not Path(val).exists():
                        setattr(req, attr, str(candidate))
            req.seed = item.get("seed", args.seed)
            if str(req.mode).lower() == "env" and item.get("env_scale") in (None, ""):
                req.env_scale = args.env_scale
            requests.append(req)
        if not requests:
            raise SystemExit(f"no samples found in {args.samples}")
        return requests

    if not args.source:
        raise SystemExit("provide --source or --samples (see --help)")

    return [RelightRequest(
        source=args.source,
        mode=args.mode,
        env_map=args.env_map,
        env_scale=args.env_scale,
        env_rotation_speed=args.env_rotation_speed,
        prompt=args.prompt,
        light_ref=args.light_ref,
        image_ref=args.image_ref,
        use_image_ref=not args.no_image_ref,
        seed=args.seed,
        start_seconds=args.start_seconds,
    )]


def main():
    args = parse_args()
    requests = build_requests(args)
    for request in requests:
        if args.delight_only:
            if not Path(request.source).is_file():
                raise FileNotFoundError(f"source not found: {request.source}")
        else:
            request.validate()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = FlashRelightPipeline(
        weights_dir=args.weights,
        device=args.device,
        steps=args.steps,
        shift=args.shift,
        offload=args.offload,
        matting=args.matting,
        matting_model=args.matting_model,
        delight_only=args.delight_only,
    )

    if args.delight_only:
        for req in requests:
            source_u8 = pipe.preprocess(req.source, matting=req.matting, start_seconds=req.start_seconds)
            delit_u8 = pipe.delight_frames(source_u8, seed=req.seed)
            name = f"{req.resolved_id()}_delit.mp4"
            vio.save_mp4(delit_u8, out_dir / name, fps=args.fps)
            if not args.no_comparison:
                vio.save_mp4(
                    vio.delight_comparison(source_u8, delit_u8),
                    out_dir / f"{req.resolved_id()}_delit_compare.mp4", fps=args.fps)
            print(f"[done] {out_dir / name}")
        return

    pipe.run_batch(
        requests,
        out_dir=str(out_dir),
        save_comparison=not args.no_comparison,
        fps=args.fps,
    )
    print(f"\nAll done -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
