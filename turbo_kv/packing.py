"""int4 bit-packing + per-token affine quantization for ``TurboKVCache`` (PLAN §M3).

Two 4-bit codes (0–15) share one ``uint8`` byte → the packed store is exactly
**0.5 bytes/value**, the 4× reduction vs BF16 (2 bytes/value) that M3 must show
at the *allocator* level. Quantization is **per-token** (a scale + zero-point per
``head_dim`` vector), the granularity M2 proved is the one a rotation helps.

``torch`` is imported lazily so the module imports without a GPU.
"""
from __future__ import annotations

INT4_LEVELS = 16


def pack_int4(codes):
    """Pack a ``uint8`` tensor of 4-bit codes ``[..., n]`` → ``[..., ceil(n/2)]``.

    Even indices go in the low nibble, odd indices in the high nibble. ``n`` is
    padded with a zero code when odd (the caller tracks the true length).
    """
    import torch

    codes = codes.to(torch.uint8)
    n = codes.shape[-1]
    if n % 2:
        pad = torch.zeros(*codes.shape[:-1], 1, dtype=torch.uint8, device=codes.device)
        codes = torch.cat([codes, pad], dim=-1)
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo | (hi << 4)).contiguous()


def unpack_int4(packed, n: int):
    """Inverse of :func:`pack_int4`; ``n`` is the true (pre-pad) last-dim length."""
    import torch

    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)
    return out[..., :n].contiguous()


def quantize_int4_per_token(x):
    """Per-token affine quant of ``x`` ``[..., d]`` → ``(codes uint8, scale, lo)``.

    ``scale`` and ``lo`` keep a trailing singleton dim (``[..., 1]``) so dequant
    broadcasts. Codes are integers in ``[0, 15]``.
    """
    import torch

    x = x.to(torch.float32)
    lo = x.amin(dim=-1, keepdim=True)
    hi = x.amax(dim=-1, keepdim=True)
    scale = ((hi - lo) / (INT4_LEVELS - 1)).clamp_min(1e-8)
    codes = torch.clamp(torch.round((x - lo) / scale), 0, INT4_LEVELS - 1).to(torch.uint8)
    return codes, scale, lo


def dequantize_int4_per_token(codes, scale, lo):
    """Inverse of :func:`quantize_int4_per_token` → float32 reconstruction."""
    return codes.to(scale.dtype) * scale + lo


def quantize_int4_per_channel(x):
    """Per-**channel** affine quant of ``x`` ``[..., T, D]`` → ``(codes, scale, lo)``.

    The scale/zero are computed per coordinate (channel) over the token dim, so
    ``scale``/``lo`` keep shape ``[..., 1, D]``. This is the KIVI key-cache scheme:
    keys have persistent *channel* outliers, and a per-channel scale gives each
    outlier channel its own range instead of letting it inflate a whole token's
    scale (the failure mode of per-token key quantization). Dequant reuses
    :func:`dequantize_int4_per_token` (the ``codes*scale+lo`` formula broadcasts).
    """
    import torch

    x = x.to(torch.float32)
    lo = x.amin(dim=-2, keepdim=True)
    hi = x.amax(dim=-2, keepdim=True)
    scale = ((hi - lo) / (INT4_LEVELS - 1)).clamp_min(1e-8)
    codes = torch.clamp(torch.round((x - lo) / scale), 0, INT4_LEVELS - 1).to(torch.uint8)
    return codes, scale, lo

