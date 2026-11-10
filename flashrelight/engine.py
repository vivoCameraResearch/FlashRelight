# flashrelight/engine.py
#
# Single-stage sampler shared by FlashDelight (delighting) and FlashRelight
# (relighting). Both stages are Wan2.1-1.3B DiT video models fine-tuned with
# LoRA; they only differ in input channel layout and lighting conditions.
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from flashrelight.models.flashrelight_dit import (
    FlashRelightSystem,
    FlashRelightWanBackbone,
)
from flashrelight.models.lora import apply_lora_to_wan
from flashrelight.models.text_encoder.t5 import T5EncoderModel
from flashrelight.models.wan import WanModel
from flashrelight.models.wan_vae import Wan2_1_VAE
from flashrelight.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler

# Per-stage settings; these must match the training configuration.
STAGE_CONFIG = {
    "delight": dict(in_dim=32, cond_dim=64, lora_rank=16, lora_alpha=16.0, env_fusion=False),
    "relight": dict(in_dim=64, cond_dim=48, lora_rank=32, lora_alpha=32.0, env_fusion=True),
}

LORA_TARGETS = "q,k,v,o,ffn.0,ffn.2"


def load_wan_config(config_path: str, in_dim: int, cond_dim: int) -> dict:
    """Read the Wan2.1 `config.json` and override the FlashRelight-specific dims."""
    cfg = json.loads(Path(config_path).read_text())

    def pick(*names, default=None):
        for n in names:
            if n in cfg:
                return cfg[n]
        return default

    return dict(
        model_type=pick("model_type", default="t2v"),
        patch_size=tuple(pick("patch_size", default=[1, 2, 2])),
        text_len=int(pick("text_len", "text_length", default=512)),
        in_dim=int(in_dim),
        dim=int(pick("dim", "hidden_size", default=1536)),
        ffn_dim=int(pick("ffn_dim", "intermediate_size", default=8960)),
        freq_dim=int(pick("freq_dim", default=256)),
        text_dim=int(pick("text_dim", default=4096)),
        out_dim=int(pick("out_dim", default=16)),
        num_heads=int(pick("num_heads", "num_attention_heads", default=12)),
        num_layers=int(pick("num_layers", "num_hidden_layers", default=30)),
        window_size=tuple(pick("window_size", default=[-1, -1])),
        qk_norm=bool(pick("qk_norm", default=True)),
        cross_attn_norm=bool(pick("cross_attn_norm", default=True)),
        eps=float(pick("eps", default=1e-6)),
        cond_dim=int(cond_dim),
    )


class FlashRelightEngine:
    """Flow-matching sampler for one FlashRelight stage.

    Parameters
    ----------
    stage:
        ``"delight"`` or ``"relight"``.
    checkpoint:
        Path to the released stage checkpoint (``*.pt`` holding ``{"model": state_dict}``
        or a bare ``state_dict``).
    wan_dir:
        Directory of the Wan2.1-T2V-1.3B release: ``config.json``,
        ``models_t5_umt5-xxl-enc-bf16.pth``, ``google/umt5-xxl/`` and ``Wan2.1_VAE.pth``.
    vae:
        Optional pre-built :class:`Wan2_1_VAE` to share between stages.
    text_encoder:
        Optional pre-built :class:`T5EncoderModel` to share between stages.
    offload:
        Move the DiT to CPU between calls (saves ~4 GB of VRAM, slightly slower).
    """

    def __init__(
        self,
        stage: str,
        checkpoint: str,
        wan_dir: str,
        vae_ckpt: Optional[str] = None,
        t5_ckpt: str = "models_t5_umt5-xxl-enc-bf16.pth",
        t5_tokenizer: str = "google/umt5-xxl",
        device: str | torch.device = "cuda:0",
        param_dtype: torch.dtype = torch.bfloat16,
        offload: bool = False,
        t5_cpu: bool = True,
        num_train_timesteps: int = 1000,
        vae: Optional[Wan2_1_VAE] = None,
        text_encoder: Optional[T5EncoderModel] = None,
        verbose: bool = True,
    ):
        stage = str(stage).lower()
        if stage not in STAGE_CONFIG:
            raise ValueError(f"stage must be one of {sorted(STAGE_CONFIG)}, got {stage!r}")
        self.stage = stage
        self.scfg = STAGE_CONFIG[stage]

        self.device = torch.device(device)
        self.param_dtype = param_dtype
        self.offload = bool(offload)
        self.t5_cpu = bool(t5_cpu)
        self.num_train_timesteps = int(num_train_timesteps)
        self.verbose = bool(verbose)

        wan_dir = str(wan_dir)
        self.wan_dir = wan_dir
        vae_ckpt = vae_ckpt or str(Path(wan_dir) / "Wan2.1_VAE.pth")

        # ---------------- text encoder (umT5-XXL, kept on CPU by default) -----
        self.text_encoder = text_encoder or T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            checkpoint_path=str(Path(wan_dir) / t5_ckpt),
            tokenizer_path=str(Path(wan_dir) / t5_tokenizer),
            shard_fn=None,
        )

        # ---------------- VAE -------------------------------------------------
        self.vae = vae or Wan2_1_VAE(vae_pth=vae_ckpt, device=self.device)

        # ---------------- DiT + LoRA -----------------------------------------
        self.wan_cfg = load_wan_config(
            str(Path(wan_dir) / "config.json"),
            in_dim=self.scfg["in_dim"],
            cond_dim=self.scfg["cond_dim"],
        )
        self.dit = self._build_and_load(checkpoint)
        self.dit.to("cpu" if self.offload else self.device)

    # ------------------------------------------------------------------ build
    def _build_and_load(self, checkpoint: str) -> FlashRelightSystem:
        wan = WanModel(**self.wan_cfg)
        apply_lora_to_wan(
            wan,
            target_substrings=[s.strip() for s in LORA_TARGETS.split(",") if s.strip()],
            rank=int(self.scfg["lora_rank"]),
            alpha=float(self.scfg["lora_alpha"]),
            dropout=0.0,
        )
        dit = FlashRelightSystem(
            FlashRelightWanBackbone(wan),
            with_env_fusion=bool(self.scfg["env_fusion"]),
        )
        # cast before load_state_dict, otherwise fp32 params blow up host RAM
        dit = dit.to(dtype=self.param_dtype)

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"{self.stage} checkpoint not found: {ckpt_path}")
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = dit.load_state_dict(state, strict=False)
        del ckpt, state
        gc.collect()

        if self.verbose:
            print(f"[{self.stage}] loaded {ckpt_path.name}: "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"[{self.stage}]   missing e.g. {list(missing)[:3]}")
            if unexpected:
                print(f"[{self.stage}]   unexpected e.g. {list(unexpected)[:3]}")
        # Only the zero-gated env branch may legitimately be missing in the
        # delighting checkpoint; anything else means a config mismatch.
        hard_missing = [k for k in missing if "cond_cross_attn" not in k
                        and "cond_norm" not in k and "cond_gate" not in k
                        and "cond_embedding" not in k and not k.startswith("fusion")]
        if hard_missing:
            raise RuntimeError(
                f"[{self.stage}] checkpoint does not match the model definition; "
                f"{len(hard_missing)} unexpected missing keys, e.g. {hard_missing[:5]}")

        dit.eval()
        for p in dit.parameters():
            p.requires_grad_(False)
        return dit

    # -------------------------------------------------------------- utilities
    def _encode_text(self, prompt: str) -> list:
        if self.t5_cpu:
            ctx = self.text_encoder([prompt], torch.device("cpu"))
            return [t.to(self.device, dtype=torch.bfloat16) for t in ctx]
        self.text_encoder.model.to(self.device)
        ctx = self.text_encoder([prompt], self.device)
        if self.offload:
            self.text_encoder.model.cpu()
        return [t.to(self.device, dtype=torch.bfloat16) for t in ctx]

    @torch.no_grad()
    def encode_video(self, frames_u8) -> torch.Tensor:
        """uint8 ``[T,H,W,3]`` -> VAE latent ``[16,T',H/8,W/8]``."""
        arr = np.asarray(frames_u8).astype(np.float32) / 127.5 - 1.0
        video = torch.from_numpy(arr).permute(3, 0, 1, 2).to(self.device)
        return self.vae.encode([video])[0]

    @torch.no_grad()
    def encode_image(self, frame_u8) -> torch.Tensor:
        """uint8 ``[H,W,3]`` -> single-frame latent ``[16,H/8,W/8]``."""
        arr = np.asarray(frame_u8).astype(np.float32) / 127.5 - 1.0
        img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(1).to(self.device)
        return self.vae.encode([img])[0][:, 0].contiguous()

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=self.param_dtype):
            out = self.vae.decode([latent.to(self.device)])
        return out[0] if isinstance(out, (list, tuple)) else out

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def generate(
        self,
        ref_latent: torch.Tensor,
        prompt: str = "",
        latent_shape: Tuple[int, int, int, int] = (16, 13, 104, 60),
        steps: int = 50,
        shift: float = 5.0,
        seed: int = 1,
        env_3buffer_latent: Optional[torch.Tensor] = None,
        light_ref_latent: Optional[torch.Tensor] = None,
        image_ref_latent: Optional[torch.Tensor] = None,
        zero_text: bool = False,
        decode: bool = True,
        progress: bool = False,
    ) -> torch.Tensor:
        """Run the flow-matching sampler.

        Returns a decoded video tensor ``[3,T,H,W]`` in ``[-1,1]`` when
        ``decode`` is True, otherwise the raw latent ``[16,F,h,w]``.
        """
        if ref_latent is None:
            raise ValueError("ref_latent is required")

        context = self._encode_text(prompt or "")
        if zero_text:
            context = [torch.zeros_like(t) for t in context]

        C, F, H, W = latent_shape
        if F != 13:
            raise ValueError("FlashRelight is trained for 13 latent frames (49 video frames)")

        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        latents = torch.randn(C, F, H, W, device=self.device, dtype=torch.float32, generator=g)

        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(int(steps), device=self.device, shift=float(shift))
        timesteps = scheduler.timesteps

        def _b5(x, dtype=None):
            if x is None:
                return None
            if x.ndim == 4:
                x = x.unsqueeze(0)
            return x.to(self.device, dtype=dtype, non_blocking=True)

        ref = _b5(ref_latent)
        light_ref = _b5(light_ref_latent)
        env_tokens = _b5(env_3buffer_latent, dtype=self.param_dtype)
        if image_ref_latent is not None:
            image_ref = image_ref_latent
            if image_ref.ndim == 3:
                image_ref = image_ref.unsqueeze(0)
            image_ref = image_ref.to(self.device, non_blocking=True)
        else:
            image_ref = None

        self.dit.to(self.device)
        tty = sys.stdout.isatty()
        with torch.amp.autocast("cuda", dtype=self.param_dtype):
            for i, t in enumerate(timesteps):
                pred = self.dit(
                    latents.unsqueeze(0),
                    ref,
                    torch.stack([t]),
                    context=context,
                    env_map_3buffer_latent=env_tokens,
                    ref_light_latent=light_ref,
                    image_ref_latent=image_ref,
                )
                latents = scheduler.step(
                    pred, t, latents.unsqueeze(0), return_dict=False, generator=g
                )[0].squeeze(0)
                if progress and (tty or i + 1 == len(timesteps)):
                    end = "" if tty else "\n"
                    print(f"\r[{self.stage}] step {i + 1}/{len(timesteps)}",
                          end=end, flush=True)
        if progress and tty:
            print()

        if self.offload:
            self.dit.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        return self.decode_latent(latents) if decode else latents
