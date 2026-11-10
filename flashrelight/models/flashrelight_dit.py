# flashrelight/models/flashrelight_dit.py
#
# FlashRelight / FlashDelight wrapper around the Wan2.1 DiT (``WanModel``).
#
# Latent channel layout (16 latent channels per video, Wan2.1 VAE):
#
#   FlashDelight  (in_dim=32):  [ noisy | source            ]
#   FlashRelight  (in_dim=64):  [ noisy | delit | light-ref | image-ref ]
#
# Lighting conditions:
#   * env-map    -> 48-ch 3-buffer env latent -> ``cond_tokens`` -> time-aligned
#                   cross-attention inside every block
#   * text       -> T5 context -> the original Wan text cross-attention
#   * light-ref  -> 16-ch latent concatenated on the input channels
from __future__ import annotations

import torch
import torch.nn as nn

from flashrelight.models.wan import WanModel
from flashrelight.models.condition_fusion import EnvPhysEncoder

# env-map latent grid is 8x16 tokens per latent frame
ENV_TOKENS_PER_FRAME = 8 * 16


def _to_list_cthw(x_bcthw: torch.Tensor):
    """[B,C,T,H,W] -> list of [C,T,H,W] (the layout WanModel expects)."""
    if x_bcthw.ndim != 5:
        raise ValueError(f"Expected 5D [B,C,T,H,W], got {tuple(x_bcthw.shape)}")
    return [x_bcthw[b].contiguous() for b in range(x_bcthw.shape[0])]


class FlashRelightWanBackbone(nn.Module):
    """Adapts FlashRelight tensors to the Wan DiT interface.

    Inputs (all latents, Wan2.1 VAE space, F=13 for 49 input frames):
      noisy_gt  : [B,16,F,H,W]
      ref       : [B,16,F,H,W]   source video (delight) / delit video (relight)
      t         : [B]            flow-matching timestep
      context   : [B,L,4096]     T5 text embedding
      cond_tokens: [B,F*128,48]  env-map tokens, or None to disable the branch
      ref_light_latent : [B,16,F,H,W] optional lighting-reference video
      image_ref_latent : [B,16,H,W]   optional appearance anchor frame

    Output: [B,16,F,H,W] predicted flow-matching velocity.
    """

    def __init__(self, wan: WanModel):
        super().__init__()
        self.wan = wan
        self.patch_size = getattr(wan, "patch_size", (1, 2, 2))
        self.expected_in_dim = int(getattr(wan, "in_dim", 32))

    def _seq_len_for(self, F: int, H: int, W: int) -> int:
        tp, hp, wp = self.patch_size
        return (F // tp) * (H // hp) * (W // wp)

    def forward(
        self,
        noisy_gt: torch.Tensor,
        ref: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        cond_tokens: torch.Tensor | None = None,
        ref_light_latent: torch.Tensor | None = None,
        image_ref_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.expected_in_dim == 64:
            if ref_light_latent is None:
                ref_light_latent = torch.zeros_like(ref)
            if image_ref_latent is None:
                image_ref_latent = torch.zeros_like(ref[:, :, 0])
            if image_ref_latent.ndim != 4:
                raise ValueError(
                    f"image_ref_latent must be [B,C,H,W], got {tuple(image_ref_latent.shape)}")
            # broadcast the single appearance frame over time
            image_ref_latent = image_ref_latent.unsqueeze(2).expand(
                -1, -1, ref.shape[2], -1, -1).contiguous()
            x_in = torch.cat([noisy_gt, ref, ref_light_latent, image_ref_latent], dim=1)
        elif self.expected_in_dim == 48:
            if ref_light_latent is None:
                ref_light_latent = torch.zeros_like(ref)
            x_in = torch.cat([noisy_gt, ref, ref_light_latent], dim=1)
        elif self.expected_in_dim == 32:
            x_in = torch.cat([noisy_gt, ref], dim=1)
        else:
            raise ValueError(
                f"Unsupported Wan in_dim={self.expected_in_dim}; expected 32, 48 or 64")

        if x_in.shape[1] != self.expected_in_dim:
            raise ValueError(
                f"Input channel mismatch: got {x_in.shape[1]}, expected {self.expected_in_dim}")

        _, _, F, H, W = x_in.shape
        tp, hp, wp = self.patch_size
        if F < tp or H < hp or W < wp:
            raise ValueError(f"Input smaller than patch_size={self.patch_size}: F,H,W={F,H,W}")
        if (F % tp) != 0 or (H % hp) != 0 or (W % wp) != 0:
            raise ValueError(
                f"Input not divisible by patch_size={self.patch_size}: F,H,W={F,H,W}")

        seq_len = self._seq_len_for(F, H, W)
        if seq_len <= 0:
            raise ValueError(f"seq_len <= 0 (patch_size={self.patch_size}, F,H,W={F,H,W})")

        t = t.to(device=x_in.device)
        if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            t = t.float()

        out_list = self.wan(
            x=_to_list_cthw(x_in),
            t=t,
            context=context,
            seq_len=seq_len,
            y=None,
            cond_tokens=cond_tokens,
            cond_lens=None,
            cond_already_projected=False,
            env_per_frame=ENV_TOKENS_PER_FRAME,
        )
        return torch.stack([u.contiguous() for u in out_list], dim=0)


class FlashRelightSystem(nn.Module):
    """Top-level module whose ``state_dict`` matches the released checkpoints.

    * ``backbone`` -> Wan DiT (+ LoRA)
    * ``fusion``   -> env-map token encoder (relighting stage only)
    """

    def __init__(self, backbone: nn.Module, with_env_fusion: bool = True):
        super().__init__()
        self.backbone = backbone
        # EnvPhysEncoder only holds a LayerNorm; it is absent from the
        # delighting checkpoint, so we only create it when needed.
        self.fusion = EnvPhysEncoder(out_dim=48, env_grid_hw=(8, 16)) if with_env_fusion else None

    def forward(
        self,
        noisy_gt: torch.Tensor,
        ref: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        env_map_3buffer_latent: torch.Tensor | None = None,
        ref_light_latent: torch.Tensor | None = None,
        image_ref_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is None:
            raise ValueError("text context is required (pass zeros to disable it)")

        cond_tokens = None
        if self.fusion is not None and env_map_3buffer_latent is not None:
            cond_tokens = self.fusion(env_map_3buffer_latent)  # [B, F*128, 48]

        return self.backbone(
            noisy_gt,
            ref,
            t,
            context=context,
            cond_tokens=cond_tokens,
            ref_light_latent=ref_light_latent,
            image_ref_latent=image_ref_latent,
        )
