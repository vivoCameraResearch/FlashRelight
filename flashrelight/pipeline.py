# flashrelight/pipeline.py
#
# High-level two-stage FlashRelight pipeline:
#
#      in-the-wild clip
#            |
#     [optional] portrait matting + crop to 832x480 @ 49 frames
#            |
#   stage 1: FlashDelight  -- removes the baked-in lighting -> flat "delit" video
#            |
#   stage 2: FlashRelight  -- re-lights it under one of three controls:
#              * env   : dynamic HDR environment map sequence
#              * text   : natural-language lighting description
#              * light_ref : lighting-reference video
#            |
#        relit portrait video
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from flashrelight.engine import FlashRelightEngine
from flashrelight.models.text_encoder.t5 import T5EncoderModel
from flashrelight.models.wan_vae import Wan2_1_VAE
from flashrelight.utils import video_io as vio

MODES = ("env", "text", "light_ref")

# Native training resolution / length. Changing these degrades quality.
NUM_FRAMES = 49
PIXEL_H, PIXEL_W = 832, 480
LATENT_F, LATENT_H, LATENT_W = 13, 104, 60
ENV_H, ENV_W = 64, 128
DEFAULT_FPS = 16


@dataclass
class RelightRequest:
    """One relighting job."""

    source: str                                   # video or image path
    mode: str = "env"                             # env | text | light_ref
    env_map: Optional[str] = None                 # .exr folder / .exr / .hdr / .npy
    prompt: str = ""                              # lighting description (mode="text")
    light_ref: Optional[str] = None               # lighting reference video
    sample_id: Optional[str] = None
    seed: int = 1
    use_image_ref: bool = True                    # identity/appearance anchor frame
    image_ref: Optional[str] = None               # defaults to the source mid-frame
    start_seconds: float = 0.0
    matting: Optional[str] = None                 # auto | birefnet | none (per sample)
    env_rotation_speed: float = 0.0               # single panorama yaw, degrees per output second
    env_scale: float = vio.DEFAULT_ENV_SCALE      # linear HDR multiplier

    def resolved_id(self) -> str:
        return self.sample_id or Path(self.source).stem

    def validate(self) -> "RelightRequest":
        self.mode = str(self.mode).lower()
        if self.mode == "ref_light":
            self.mode = "light_ref"
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        self.env_rotation_speed = float(self.env_rotation_speed)
        if not np.isfinite(self.env_rotation_speed):
            raise ValueError("env_rotation_speed must be finite")
        if self.env_rotation_speed and self.mode != "env":
            raise ValueError("env_rotation_speed requires mode='env'")
        if not Path(self.source).exists():
            raise FileNotFoundError(f"source not found: {self.source}")
        if self.mode == "env":
            if not self.env_map:
                raise ValueError("mode='env' requires env_map")
            if not Path(self.env_map).exists():
                raise FileNotFoundError(f"env_map not found: {self.env_map}")
            self.env_scale = vio.validate_env_scale(self.env_scale)
        if self.mode == "text" and not str(self.prompt).strip():
            raise ValueError("mode='text' requires a non-empty prompt")
        if self.mode == "light_ref":
            if not self.light_ref:
                raise ValueError("mode='light_ref' requires light_ref")
            if not Path(self.light_ref).exists():
                raise FileNotFoundError(f"light_ref not found: {self.light_ref}")
        return self

    @classmethod
    def from_dict(cls, d: Dict) -> "RelightRequest":
        known = set(cls.__dataclass_fields__)
        alias = {"id": "sample_id", "source_video": "source", "env_maps": "env_map",
                 "light_ref_video": "light_ref", "vlm_prompt": "prompt",
                 "text_prompt": "prompt"}
        kwargs = {}
        for k, v in d.items():
            k = alias.get(k, k)
            if k in known and v not in (None, ""):
                kwargs[k] = v
        return cls(**kwargs)


@dataclass
class RelightResult:
    sample_id: str
    mode: str
    source_u8: np.ndarray
    delit_u8: np.ndarray
    relit_u8: np.ndarray
    env_u8: Optional[np.ndarray] = None
    light_ref_u8: Optional[np.ndarray] = None
    prompt: str = ""
    timings: Dict[str, float] = field(default_factory=dict)

    def comparison(self) -> np.ndarray:
        """Side-by-side panel: source | control | delit | relit."""
        panels = [vio.add_label(self.source_u8, "SOURCE")]
        if self.env_u8 is not None:
            panels.append(vio.add_label(self.env_u8, "ENV MAP"))
        if self.light_ref_u8 is not None:
            panels.append(vio.add_label(self.light_ref_u8, "LIGHT REF"))
        if self.mode == "text" and self.prompt:
            h, w = self.relit_u8.shape[1], self.relit_u8.shape[2]
            panels.append(vio.make_text_panel(self.prompt, h, w, self.relit_u8.shape[0]))
        panels.append(vio.add_label(self.delit_u8, "DELIT"))
        panels.append(vio.add_label(self.relit_u8, "RELIT"))
        return vio.hstack_panels(panels)


class FlashRelightPipeline:
    """Portrait video delighting and relighting. Use delight_only=True to load just stage 1.

    Example
    -------
    >>> pipe = FlashRelightPipeline("weights")
    >>> res = pipe.run(RelightRequest(source="assets/sources/generated_portrait.mp4",
    ...                               mode="env", env_map="assets/envmaps/02_zwartkops_start_sunset",
    ...                               matting="none"))
    >>> vio.save_mp4(res.relit_u8, "out.mp4")
    """

    @staticmethod
    def resolve_weights_dir(weights_dir: Optional[str] = None) -> str:
        """Find the weights directory.

        Order: explicit argument -> ``./weights`` -> ``<repo>/weights`` ->
        ``<repo>/../weights`` (handy when the weights sit next to a git clone,
        as in our release bundle).
        """
        repo_root = Path(__file__).resolve().parents[1]
        candidates = []
        if weights_dir:
            candidates.append(Path(weights_dir))
        candidates += [repo_root / "weights", repo_root.parent / "weights"]
        for cand in candidates:
            if (cand / "config.json").exists():
                return str(cand)
        raise FileNotFoundError(
            "FlashRelight weights not found. Looked in:\n  " +
            "\n  ".join(str(c) for c in candidates) +
            "\nRun `python scripts/download_weights.py --dest weights` or pass "
            "--weights /path/to/weights (see README §2).")

    def __init__(
        self,
        weights_dir: str = "weights",
        delight_ckpt: Optional[str] = None,
        relight_ckpt: Optional[str] = None,
        device: str = "cuda:0",
        steps: int = 50,
        shift: float = 5.0,
        offload: bool = False,
        t5_cpu: bool = True,
        matting: str = "auto",                      # auto | birefnet | none
        matting_model: Optional[str] = None,
        param_dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
        delight_only: bool = False,
    ):
        try:
            weights = Path(self.resolve_weights_dir(weights_dir))
        except FileNotFoundError:
            weights = Path(weights_dir)
        if not weights.exists():
            raise FileNotFoundError(
                f"weights dir not found: {weights}. See README -> 'Download weights'.")

        self.weights_dir = str(weights)
        self.device = torch.device(device)
        self.steps = int(steps)
        self.shift = float(shift)
        self.matting_mode = str(matting).lower()
        self.matting_model = matting_model
        self.verbose = bool(verbose)
        self._matter = None

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        delight_ckpt = delight_ckpt or str(weights / "flashdelight_rank0_full.pt")

        # The VAE and the umT5 text encoder are identical for both stages:
        # build once and share (saves ~11 GB of host RAM and 0.5 GB of VRAM).
        if verbose:
            print("[pipeline] loading shared VAE + umT5 text encoder ...")
        shared_vae = Wan2_1_VAE(vae_pth=str(weights / "Wan2.1_VAE.pth"), device=self.device)
        shared_t5 = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            checkpoint_path=str(weights / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(weights / "google/umt5-xxl"),
            shard_fn=None,
        )

        common = dict(
            wan_dir=str(weights),
            device=device,
            param_dtype=param_dtype,
            offload=offload,
            t5_cpu=t5_cpu,
            vae=shared_vae,
            text_encoder=shared_t5,
            verbose=verbose,
        )
        if verbose:
            print("[pipeline] loading FlashDelight (stage 1) ...")
        self.delight = FlashRelightEngine("delight", delight_ckpt, **common)
        self.relight: Optional[FlashRelightEngine] = None
        if not delight_only:
            if verbose:
                print("[pipeline] loading FlashRelight (stage 2) ...")
            relight_ckpt = relight_ckpt or str(weights / "flashrelight_rank0_full.pt")
            self.relight = FlashRelightEngine("relight", relight_ckpt, **common)

        self.vae = shared_vae
        if verbose:
            mem = torch.cuda.memory_allocated(self.device) / 1024 ** 3
            print(f"[pipeline] ready (VRAM allocated: {mem:.2f} GiB)")

    # ------------------------------------------------------------------ input
    def _resolve_matting_model(self) -> str:
        """Prefer a local BiRefNet next to the weights, else the HF repo id."""
        from flashrelight.utils.matting import DEFAULT_BIREFNET_ID

        if self.matting_model:
            return self.matting_model
        local = Path(self.weights_dir) / "BiRefNet-portrait"
        if (local / "config.json").exists():
            return str(local)
        return DEFAULT_BIREFNET_ID

    def _get_matter(self):
        if self._matter is None:
            from flashrelight.utils.matting import get_matter
            path = self._resolve_matting_model()
            if self.verbose:
                print(f"[pipeline] loading portrait matting model: {path}")
            self._matter = get_matter(model_path=path, device=str(self.device))
        return self._matter

    def preprocess(
        self,
        source: str,
        matting: Optional[str] = None,
        start_seconds: float = 0.0,
        num_frames: int = NUM_FRAMES,
    ) -> np.ndarray:
        """Any clip/image -> uint8 ``[49,832,480,3]`` subject-on-black input.

        ``matting``: ``"birefnet"`` forces matting, ``"none"`` disables it,
        ``"auto"`` (default) runs matting and silently falls back to the raw
        frames if the segmenter is unavailable.
        """
        mode = (matting or self.matting_mode).lower()
        # the model consumes exactly 49 frames; cap the read at 49/16 s
        frames = vio.read_source_rgb(
            source,
            num_frames=num_frames,
            target_h=PIXEL_H,
            target_w=PIXEL_W,
            max_seconds=num_frames / DEFAULT_FPS,
            start_seconds=start_seconds,
        )
        if mode == "none":
            return frames
        try:
            return self._get_matter().matte_frames(frames)
        except Exception as exc:
            if mode == "birefnet":
                raise
            if self.verbose:
                print(f"[pipeline] matting unavailable ({type(exc).__name__}: {exc}); "
                      f"using raw frames. Pass --matting none to silence this.")
            return frames

    # ------------------------------------------------------------- conditions
    @torch.no_grad()
    def encode_env(self, env_path: str, *, env_rotation_speed: float = 0.0,
                   fps: float = DEFAULT_FPS, env_scale: float = vio.DEFAULT_ENV_SCALE) -> torch.Tensor:
        """HDR env sequence -> 48-channel env latent ``[48,13,8,16]``."""
        env_hdr = vio.read_env_hdr(env_path, num_frames=NUM_FRAMES,
                                   target_h=ENV_H, target_w=ENV_W,
                                   env_rotation_speed=env_rotation_speed, fps=fps, env_scale=env_scale)
        return self._encode_env_hdr(env_hdr)

    @torch.no_grad()
    def _encode_env_hdr(self, env_hdr: np.ndarray) -> torch.Tensor:
        ldr, log, direction = vio.env_hdr_to_3buffer(env_hdr)
        dev = self.device

        def enc(buf: np.ndarray, to_signed: bool) -> torch.Tensor:
            x = torch.from_numpy(buf).to(dev)
            if to_signed:
                x = x * 2.0 - 1.0
            return self.vae.encode([x])[0]

        return torch.cat([enc(ldr, True), enc(log, True), enc(direction, False)], dim=0)

    # ---------------------------------------------------------------- stage 1
    @torch.no_grad()
    def delight_frames(self, source_u8: np.ndarray, seed: int = 1,
                       steps: Optional[int] = None, progress: bool = False) -> np.ndarray:
        """Suppress the source lighting and return decoded uint8 RGB frames."""
        ref_latent = self.delight.encode_video(source_u8)
        video = self.delight.generate(
            ref_latent=ref_latent,
            prompt="",
            latent_shape=(16, LATENT_F, LATENT_H, LATENT_W),
            steps=steps or self.steps,
            shift=self.shift,
            seed=int(seed),
            decode=True,
            progress=progress,
        )
        return self._to_u8(video)

    @torch.no_grad()
    def run_delight(self, source_u8: np.ndarray, seed: int = 1,
                    steps: Optional[int] = None, progress: bool = False):
        """Prepare the delighted video and its conditioning latent for stage 2.

        Returns ``(delit_u8, delit_latent)``. ``delit_u8`` are the decoded uint8
        frames; ``delit_latent`` is the VAE latent of *those decoded frames*
        (not the raw sampler output), so that stage 2 sees exactly the same
        condition distribution it was trained on -- at training time the
        ref latent was always encoded from mp4-decoded frames.
        """
        delit_u8 = self.delight_frames(source_u8, seed=seed, steps=steps, progress=progress)
        delit_latent = self.delight.encode_video(delit_u8)
        return delit_u8, delit_latent

    # ---------------------------------------------------------------- stage 2
    def _relight_engine(self) -> FlashRelightEngine:
        if self.relight is None:
            raise RuntimeError("Relighting requires a pipeline initialized with delight_only=False.")
        return self.relight

    @torch.no_grad()
    def run_relight(
        self,
        delit_latent: torch.Tensor,
        mode: str = "env",
        env_latent: Optional[torch.Tensor] = None,
        prompt: str = "",
        light_ref_u8: Optional[np.ndarray] = None,
        image_ref_u8: Optional[np.ndarray] = None,
        seed: int = 1,
        steps: Optional[int] = None,
        progress: bool = False,
    ) -> np.ndarray:
        """Relight a delit latent under exactly one lighting control.

        Mirrors the paper's evaluation protocol: the two unused control inputs
        are zeroed so each mode is exercised in isolation.
        """
        engine = self._relight_engine()
        zeros_env = torch.zeros(48, LATENT_F, 8, 16, device=self.device,
                                dtype=torch.float32)
        env_in, light_in, text_in, zero_text = zeros_env, None, "", True

        if mode == "env":
            if env_latent is None:
                raise ValueError("mode='env' requires env_latent")
            env_in = env_latent
        elif mode == "text":
            text_in, zero_text = prompt, False
        elif mode == "light_ref":
            if light_ref_u8 is None:
                raise ValueError("mode='light_ref' requires light_ref_u8")
            light_in = engine.encode_video(light_ref_u8)
        else:
            raise ValueError(f"unknown mode: {mode}")

        image_ref_latent = None
        if image_ref_u8 is not None:
            image_ref_latent = engine.encode_image(image_ref_u8)

        video = engine.generate(
            ref_latent=delit_latent,
            prompt=text_in,
            latent_shape=(16, LATENT_F, LATENT_H, LATENT_W),
            steps=steps or self.steps,
            shift=self.shift,
            seed=int(seed),
            env_3buffer_latent=env_in,
            light_ref_latent=light_in,
            image_ref_latent=image_ref_latent,
            zero_text=zero_text,
            decode=True,
            progress=progress,
        )
        return self._to_u8(video)

    # --------------------------------------------------------------- end-to-end
    @torch.no_grad()
    def run(
        self,
        request: Union[RelightRequest, Dict],
        matting: Optional[str] = None,
        steps: Optional[int] = None,
        progress: bool = True,
        delit_cache: Optional[Dict[str, tuple]] = None,
        *,
        fps: float = DEFAULT_FPS,
    ) -> RelightResult:
        """Full delight -> relight pass for one request."""
        self._relight_engine()
        req = (request if isinstance(request, RelightRequest)
               else RelightRequest.from_dict(request)).validate()
        sid = req.resolved_id()
        timings: Dict[str, float] = {}
        # explicit call argument > per-sample field > pipeline default
        matting_mode = matting or req.matting or self.matting_mode

        env_hdr = None
        if req.mode == "env":
            env_hdr = vio.read_env_hdr(req.env_map, num_frames=NUM_FRAMES, target_h=ENV_H,
                                       target_w=ENV_W, env_rotation_speed=req.env_rotation_speed,
                                       fps=fps, env_scale=req.env_scale)

        t0 = time.perf_counter()
        source_u8 = self.preprocess(req.source, matting=matting_mode,
                                    start_seconds=req.start_seconds)
        timings["preprocess"] = time.perf_counter() - t0

        cache_key = f"{req.source}|{req.start_seconds}|{matting_mode}|{req.seed}"
        if delit_cache is not None and cache_key in delit_cache:
            delit_latent, delit_u8 = delit_cache[cache_key]
            timings["delight"] = 0.0
        else:
            t0 = time.perf_counter()
            if self.verbose:
                print(f"[{sid}] stage 1/2  delighting ...")
            delit_u8, delit_latent = self.run_delight(
                source_u8, seed=req.seed, steps=steps, progress=progress)
            timings["delight"] = time.perf_counter() - t0
            if delit_cache is not None:
                delit_cache[cache_key] = (delit_latent, delit_u8)

        env_latent, env_u8, light_ref_u8 = None, None, None
        if env_hdr is not None:
            # visualisation only: keep the 2:1 panorama aspect, on black
            env_u8 = vio.letterbox(vio.env_hdr_to_u8(env_hdr, target_h=ENV_H * 4),
                                   PIXEL_H, PIXEL_W)
            env_latent = self._encode_env_hdr(env_hdr)
        elif req.mode == "light_ref":
            light_ref_u8 = vio.read_video_rgb(req.light_ref, num_frames=NUM_FRAMES,
                                              target_h=PIXEL_H, target_w=PIXEL_W)

        image_ref_u8 = None
        if req.use_image_ref:
            if req.image_ref:
                image_ref_u8 = vio.read_image_rgb(req.image_ref, PIXEL_H, PIXEL_W)
            else:
                # appearance anchor = middle frame of the (matted) source clip
                image_ref_u8 = source_u8[len(source_u8) // 2]

        t0 = time.perf_counter()
        if self.verbose:
            print(f"[{sid}] stage 2/2  relighting (mode={req.mode}) ...")
        relit_u8 = self.run_relight(
            delit_latent,
            mode=req.mode,
            env_latent=env_latent,
            prompt=req.prompt,
            light_ref_u8=light_ref_u8,
            image_ref_u8=image_ref_u8,
            seed=req.seed,
            steps=steps,
            progress=progress,
        )
        timings["relight"] = time.perf_counter() - t0

        return RelightResult(
            sample_id=sid,
            mode=req.mode,
            source_u8=source_u8,
            delit_u8=delit_u8,
            relit_u8=relit_u8,
            env_u8=env_u8,
            light_ref_u8=light_ref_u8,
            prompt=req.prompt,
            timings=timings,
        )

    @torch.no_grad()
    def run_batch(
        self,
        requests: List[Union[RelightRequest, Dict]],
        out_dir: str,
        save_comparison: bool = True,
        fps: int = DEFAULT_FPS,
        matting: Optional[str] = None,
        steps: Optional[int] = None,
        reuse_delight: bool = True,
    ) -> List[str]:
        """Run many requests, reusing the delit video across identical sources."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cache: Optional[Dict[str, tuple]] = {} if reuse_delight else None
        written: List[str] = []

        for i, raw in enumerate(requests, 1):
            req = (raw if isinstance(raw, RelightRequest)
                   else RelightRequest.from_dict(raw)).validate()
            print(f"\n=== [{i}/{len(requests)}] {req.resolved_id()} (mode={req.mode}) ===")
            res = self.run(req, matting=matting, steps=steps, delit_cache=cache, fps=fps)

            stem = f"{res.sample_id}_{res.mode}"
            written.append(vio.save_mp4(res.relit_u8, out / f"{stem}_relit.mp4", fps=fps))
            vio.save_mp4(res.delit_u8, out / f"{stem}_delit.mp4", fps=fps)
            if save_comparison:
                vio.save_mp4(res.comparison(), out / f"{stem}_compare.mp4", fps=fps)
            print(f"    delight {res.timings['delight']:.1f}s | "
                  f"relight {res.timings['relight']:.1f}s -> {out / (stem + '_relit.mp4')}")
            gc.collect()
            torch.cuda.empty_cache()

        return written

    # ------------------------------------------------------------------ helper
    @staticmethod
    def _to_u8(video: torch.Tensor) -> np.ndarray:
        """Decoded ``[3,T,H,W]`` in [-1,1] -> uint8 ``[T,H,W,3]``."""
        if isinstance(video, (list, tuple)):
            video = video[0]
        if video.ndim == 5:
            video = video[0]
        if video.shape[0] == 3:
            video = video.permute(1, 2, 3, 0)
        elif video.shape[1] == 3:
            video = video.permute(0, 2, 3, 1)
        else:
            raise ValueError(f"unexpected video shape {tuple(video.shape)}")
        video = video.detach().float().cpu()
        if float(video.min()) >= -1.1 and float(video.max()) <= 1.1:
            video = (video + 1.0) * 0.5
        return (video.clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
