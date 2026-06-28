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


def quantize_int4_per_token_outliers(x, n_outliers: int):
    """Dense-and-sparse per-token quant (KVQuant/QJL) of ``x`` ``[..., T, D]``.

    The per-token *key* failure mode (M4): a few outlier coordinates inflate the
    whole token's int4 range, crushing the other ~60 coordinates. We keep the top
    ``n_outliers`` coordinates per token in fp16 (sparse side-channel) and compute
    the affine min/max over the **non-outlier** coordinates only, so the dense
    rest gets the full int4 grid. Outlier positions are still quantized (clamped)
    to keep the packed shape uniform, but are overwritten on reconstruction.

    Returns ``(codes uint8, scale, lo, out_idx int16, out_val)`` where ``out_idx``
    is ``[..., T, n_outliers]`` and ``out_val`` holds the original outlier values
    (kept in the input dtype). ``n_outliers=0`` degenerates to the dense per-token
    path with empty outlier tensors.
    """
    import torch

    in_dtype = x.dtype
    xf = x.to(torch.float32)
    if n_outliers <= 0:
        codes, scale, lo = quantize_int4_per_token(xf)
        out_idx = torch.empty(*xf.shape[:-1], 0, dtype=torch.int16, device=xf.device)
        out_val = torch.empty(*xf.shape[:-1], 0, dtype=in_dtype, device=xf.device)
        return codes, scale, lo, out_idx, out_val

    n = min(int(n_outliers), xf.shape[-1])
    _, out_idx = torch.topk(xf.abs(), n, dim=-1)             # [..., T, n]
    mask = torch.zeros_like(xf, dtype=torch.bool)
    mask.scatter_(-1, out_idx, True)                          # True at outliers
    # affine range over the NON-outlier coords only.
    lo = torch.where(mask, xf.new_full((), float("inf")), xf).amin(dim=-1, keepdim=True)
    hi = torch.where(mask, xf.new_full((), float("-inf")), xf).amax(dim=-1, keepdim=True)
    scale = ((hi - lo) / (INT4_LEVELS - 1)).clamp_min(1e-8)
    codes = torch.clamp(torch.round((xf - lo) / scale), 0, INT4_LEVELS - 1).to(torch.uint8)
    out_val = xf.gather(-1, out_idx).to(in_dtype)            # fp16 outlier values
    return codes, scale, lo, out_idx.to(torch.int16), out_val


def dequantize_int4_per_token_outliers(codes, scale, lo, out_idx, out_val):
    """Inverse of :func:`quantize_int4_per_token_outliers` → float32.

    Dequant the dense per-token grid, then ``scatter_`` the fp16 outlier values
    back into their ``out_idx`` positions along the last dim (exact at outliers).
    """
    deq = dequantize_int4_per_token(codes.to(scale.dtype), scale, lo).to(torch.float32)
    if out_idx.numel() == 0:
        return deq
    deq.scatter_(-1, out_idx.to(torch.int64), out_val.to(torch.float32))
    return deq

