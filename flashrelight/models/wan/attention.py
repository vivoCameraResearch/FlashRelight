# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# FlashRelight release note
# -------------------------
# The original Wan implementation hard-requires FlashAttention-2/3. For the
# public release we keep the FlashAttention fast path but add an exact
# `scaled_dot_product_attention` (SDPA) fallback so that the model runs on any
# PyTorch >= 2.1 install without compiling FlashAttention.
#
# The fallback is numerically equivalent to the varlen FlashAttention call used
# here (keys are masked with an explicit key-padding mask, padded query rows are
# zeroed), so results match the paper up to float round-off.
import os
import warnings

import torch
import torch.nn.functional as F

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# Set FLASHRELIGHT_ATTENTION=sdpa to force the SDPA path even when FA is installed.
_FORCE_SDPA = os.environ.get("FLASHRELIGHT_ATTENTION", "auto").lower() in ("sdpa", "torch", "math")

FLASH_ATTN_AVAILABLE = (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE) and not _FORCE_SDPA

__all__ = [
    'flash_attention',
    'attention',
    'sdpa_attention',
    'FLASH_ATTN_AVAILABLE',
]


def sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    dtype=torch.bfloat16,
):
    """Exact SDPA replacement for the varlen FlashAttention call.

    q: [B, Lq, Nq, C]   k: [B, Lk, Nk, C]   v: [B, Lk, Nk, C2]
    q_lens / k_lens: [B] int tensors or None.
    """
    out_dtype = q.dtype
    b, lq, nq = q.shape[0], q.shape[1], q.shape[2]
    lk, nk = k.shape[1], k.shape[2]

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    if q_scale is not None:
        q = q * q_scale

    # grouped-query attention: replicate kv heads
    if nq != nk:
        if nq % nk != 0:
            raise ValueError(f"Nq ({nq}) must be divisible by Nk ({nk})")
        rep = nq // nk
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)

    q = q.transpose(1, 2)  # [B, N, Lq, C]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    attn_mask = None
    if k_lens is not None:
        pos = torch.arange(lk, device=q.device)
        valid = pos[None, :] < k_lens.to(device=q.device).view(-1, 1)  # [B, Lk]
        attn_mask = valid[:, None, None, :]  # broadcast over heads / queries

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )

    out = out.transpose(1, 2).contiguous()  # [B, Lq, N, C2]

    if q_lens is not None:
        pos = torch.arange(lq, device=out.device)
        q_valid = pos[None, :] < q_lens.to(device=out.device).view(-1, 1)
        out = out * q_valid[:, :, None, None].to(out.dtype)

    return out.type(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    if not FLASH_ATTN_AVAILABLE:
        if window_size != (-1, -1):
            raise NotImplementedError(
                "sliding-window attention requires FlashAttention; "
                "FlashRelight never uses it (window_size=(-1,-1))")
        return sdpa_attention(
            q, k, v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )
