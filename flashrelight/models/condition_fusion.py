# flashrelight/models/condition_fusion.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnvPhysEncoder(nn.Module):
    """Lightweight env-map tokenizer (relighting stage).

    The 48-channel env 3-buffer latent (LDR tone-mapped / log-radiance /
    direction buffers, each encoded by the Wan2.1 VAE into 16 channels) is

      1) average-pooled to an ``env_grid_hw`` spatial grid,
      2) flattened to tokens ``[B, T*gh*gw, 48]``,
      3) layer-normalised.

    No extra MLP is used here: the projection to the DiT width happens in
    ``WanModel.cond_embedding``.
    """

    def __init__(
        self,
        out_dim: int = 48,
        env_grid_hw: tuple[int, int] = (8, 16),
    ):
        super().__init__()
        self.out_dim = out_dim
        self.env_grid_hw = env_grid_hw
        self.env_norm = nn.LayerNorm(out_dim)

    def encode_env_tokens(self, env_map: torch.Tensor) -> torch.Tensor:
        """``[B,T,48,H,W]`` or ``[B,48,T,H,W]`` -> ``[B, T*gh*gw, 48]``."""
        if env_map.ndim != 5:
            raise ValueError(f"env_map must be 5D, got {tuple(env_map.shape)}")

        if env_map.shape[2] == self.out_dim:          # [B,T,C,H,W]
            x_in = env_map
        elif env_map.shape[1] == self.out_dim:        # [B,C,T,H,W]
            x_in = env_map.permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise ValueError(
                f"Cannot infer channel dim for env_map; expected C={self.out_dim} "
                f"in [B,T,C,H,W] or [B,C,T,H,W], got {tuple(env_map.shape)}")

        B, T, C, H, W = x_in.shape
        gh, gw = self.env_grid_hw

        x = x_in.reshape(B * T, C, H, W).contiguous()
        x = F.adaptive_avg_pool2d(x, (gh, gw))
        x = x.flatten(2).transpose(1, 2).contiguous()      # [B*T, gh*gw, C]
        return x.view(B, T * gh * gw, self.out_dim)

    def forward(self, env_map: torch.Tensor) -> torch.Tensor:
        return self.env_norm(self.encode_env_tokens(env_map))
