"""Per-channel / per-token scalar quantizers for TurboQuant KV-cache (PLAN §3, M2).

Two quantization granularities, because the *axis* is what makes rotation matter:

* **per-channel** (``axis="channel"``): one scale per coordinate, over all tokens.
  Already adapts to channel outliers, so an orthogonal rotation does **not** help.
* **per-token** (``axis="token"``): one scale per token vector, over its
  coordinates. A few large channels otherwise inflate the whole token's scale —
  this is exactly where a rotation (which spreads energy across coordinates)
  sharply reduces quantization error. This is the TurboQuant / QuaRot premise.

Bit-widths in the M2 sweep are ``{2.5, 3, 3.5, 4}``. A fractional bit-width maps
to an integer number of reconstruction levels ``L = round(2**bits)`` (e.g. 2.5 →
6 levels, 3.5 → 11 levels); :func:`effective_bits` reports ``log2(L)`` so plots
can use the *realized* rate.

``torch`` is imported lazily so the module imports without a GPU.
"""
from __future__ import annotations

import math


def num_levels(bits: float) -> int:
    """Integer reconstruction levels for a (possibly fractional) bit-width."""
    return max(2, int(round(2.0 ** bits)))


def effective_bits(bits: float) -> float:
    """Realized rate ``log2(L)`` for the integer level count of ``bits``."""
    return math.log2(num_levels(bits))


def _reduce_dims(x, axis: str):
    """Dims to reduce over when computing scales for the given granularity."""
    if axis in ("channel", "coord", "per_channel"):
        # Scale per coordinate (last dim): reduce over all the token dims.
        return tuple(range(x.dim() - 1))
    if axis in ("token", "row", "per_token"):
        # Scale per token (a row): reduce over the coordinate dim only.
        return (x.dim() - 1,)
    raise ValueError(f"unknown quant axis: {axis!r}")


def fake_quantize(x, bits: float, *, axis: str = "channel", symmetric: bool = False):
    """Quantize→dequantize ``x`` and return ``x̂`` (same shape, float32).

    ``axis`` ∈ {``channel``, ``token``}. ``symmetric=False`` uses an affine
    min/max grid (zero-point); ``symmetric=True`` an absmax grid centered at 0.
    """
    import torch

    x = torch.as_tensor(x, dtype=torch.float32)
    dims = _reduce_dims(x, axis)
    L = num_levels(bits)

    if symmetric:
        amax = x.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8)
        qmax = max(1, (L - 1) // 2)
        scale = amax / qmax
        q = torch.clamp(torch.round(x / scale), -qmax, qmax)
        deq = q * scale
    else:
        lo = x.amin(dim=dims, keepdim=True)
        hi = x.amax(dim=dims, keepdim=True)
        scale = ((hi - lo) / (L - 1)).clamp_min(1e-8)
        q = torch.clamp(torch.round((x - lo) / scale), 0, L - 1)
        deq = q * scale + lo

    return deq.to(torch.float32)


def fake_quantize_per_coord(x, bits: float, *, symmetric: bool = False):
    """Per-channel quantizer (back-compat alias for ``axis='channel'``)."""
    return fake_quantize(x, bits, axis="channel", symmetric=symmetric)


def quantization_error(x, bits: float, *, axis: str = "channel", symmetric: bool = False) -> dict:
    """Quantize ``x`` at the given axis → dict of MSE, RMSE, cosine, max-abs error."""
    import torch

    x = torch.as_tensor(x, dtype=torch.float32)
    x_hat = fake_quantize(x, bits, axis=axis, symmetric=symmetric)
    err = x_hat - x
    xf = x.flatten()
    hf = x_hat.flatten()
    cos = torch.nn.functional.cosine_similarity(xf, hf, dim=0).item()
    return {
        "mse": torch.mean(err**2).item(),
        "rmse": torch.sqrt(torch.mean(err**2)).item(),
        "cosine": cos,
        "max_abs": err.abs().max().item(),
        "effective_bits": effective_bits(bits),
    }
