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


# --------------------------------------------------------------------------- #
# Non-uniform quantization (NUQ) — KVQuant-style codebook reconstruction (M12).
#
# Uniform affine quant spaces ``L`` levels evenly between min/max. NUQ instead
# *fits* the ``L`` reconstruction levels to the data density — by empirical
# quantiles or by 1-D k-means (Lloyd–Max) — so heavy-tailed activations get fine
# resolution near the mode and coarse resolution in the tails. Levels are fit
# PER GROUP: one codebook per coordinate (``axis="channel"``, codebook shared
# over the token dim) or one codebook per token vector (``axis="token"``,
# codebook shared over the coordinate dim). This is a numerical study; storing
# the per-group codebook (``L`` fp16 levels) is extra cost handled separately.
# --------------------------------------------------------------------------- #


def _nuq_groups(x, axis: str):
    """Reshape ``x`` to a 2-D ``[G, n]`` matrix of independent codebook groups.

    Returns ``(mat, g_dim)`` where each row of ``mat`` is one group's samples:

    * ``axis="token"``:   ``mat = x.reshape(-1, d)`` — ``G`` token rows, ``n=d``.
    * ``axis="channel"``: ``mat = x.reshape(-1, d).t()`` — ``G=d`` coords, ``n``
      tokens. ``g_dim`` records the original coordinate count ``d`` so callers
      can reshape levels back to a shape that broadcasts against ``x``.
    """
    import torch

    d = x.shape[-1]
    flat = x.reshape(-1, d)
    if axis in ("token", "row", "per_token"):
        return flat, "token"
    if axis in ("channel", "coord", "per_channel"):
        return flat.t().contiguous(), "channel"
    raise ValueError(f"unknown quant axis: {axis!r}")


def _fit_levels_2d(mat, L: int, method: str, iters: int):
    """Fit ``L`` reconstruction levels for each row (group) of ``mat`` ``[G, n]``.

    Returns ascending levels ``[G, L]``. ``method="quantile"`` uses the ``L``
    midpoint empirical quantiles; ``method="kmeans"`` runs 1-D Lloyd from that
    quantile init for ``iters`` steps (empty clusters keep their old level).
    """
    import torch

    G, n = mat.shape
    # Quantile init at the L midpoints (i+0.5)/L — deterministic, ascending.
    qs = (torch.arange(L, dtype=mat.dtype, device=mat.device) + 0.5) / L
    levels = torch.quantile(mat, qs, dim=1).t().contiguous()  # [G, L]

    if method == "quantile":
        return levels
    if method != "kmeans":
        raise ValueError(f"unknown nuq method: {method!r}")

    for _ in range(max(0, iters)):
        # Assign each sample to its nearest level, then move each level to the
        # mean of its assigned samples (Lloyd update). Vectorized over groups.
        dist = (mat[:, :, None] - levels[:, None, :]).abs()  # [G, n, L]
        idx = dist.argmin(dim=-1)                            # [G, n]
        onehot = torch.nn.functional.one_hot(idx, L).to(mat.dtype)  # [G, n, L]
        counts = onehot.sum(dim=1)                           # [G, L]
        sums = (onehot * mat[:, :, None]).sum(dim=1)         # [G, L]
        new_levels = sums / counts.clamp_min(1.0)            # safe: empty→0/1
        # Empty clusters keep their previous level (no div-by-zero, no collapse).
        new_levels = torch.where(counts > 0, new_levels, levels)
        levels = new_levels

    levels, _ = levels.sort(dim=1)  # nearest-assignment is permutation-invariant
    return levels


def fit_nuq_levels(x, bits: float, *, axis: str = "channel",
                   method: str = "kmeans", iters: int = 10):
    """Fit per-group NUQ reconstruction levels for ``x``.

    Returns a levels tensor in a layout that **broadcasts against** ``x[..., None]``
    (shape ``[..., d, 1]``) for nearest-level lookup:

    * ``axis="token"``:   ``[*x.shape[:-1], 1, L]`` — one codebook per token row.
    * ``axis="channel"``: ``[d, L]`` — one codebook per coordinate (broadcasts
      over every leading token dim).

    ``method`` ∈ {``"quantile"``, ``"kmeans"``}; ``L = num_levels(bits)``.
    """
    import torch

    x = torch.as_tensor(x, dtype=torch.float32)
    L = num_levels(bits)
    mat, kind = _nuq_groups(x, axis)
    levels = _fit_levels_2d(mat, L, method, iters)  # [G, L]
    if kind == "token":
        return levels.reshape(*x.shape[:-1], 1, L)
    return levels  # [d, L]


def fake_quantize_nuq(x, bits: float, *, axis: str = "channel",
                      method: str = "kmeans", iters: int = 10):
    """NUQ analog of :func:`fake_quantize`: fit levels, snap to nearest, deq.

    Returns ``x̂`` (same shape as ``x``, float32) where each value is replaced by
    its nearest fitted reconstruction level within its group.
    """
    import torch

    x = torch.as_tensor(x, dtype=torch.float32)
    levels = fit_nuq_levels(x, bits, axis=axis, method=method, iters=iters)
    dist = (x[..., None] - levels).abs()                 # [..., d, L]
    idx = dist.argmin(dim=-1, keepdim=True)              # [..., d, 1]
    levels_exp = levels.expand(dist.shape)               # [..., d, L]
    x_hat = torch.gather(levels_exp, -1, idx).squeeze(-1)
    return x_hat.to(torch.float32)


def quantization_error_nuq(x, bits: float, *, axis: str = "channel",
                           method: str = "kmeans", iters: int = 10) -> dict:
    """:func:`quantization_error` using NUQ — same dict of MSE/cosine metrics."""
    import torch

    x = torch.as_tensor(x, dtype=torch.float32)
    x_hat = fake_quantize_nuq(x, bits, axis=axis, method=method, iters=iters)
    err = x_hat - x
    cos = torch.nn.functional.cosine_similarity(x.flatten(), x_hat.flatten(), dim=0).item()
    return {
        "mse": torch.mean(err**2).item(),
        "rmse": torch.sqrt(torch.mean(err**2)).item(),
        "cosine": cos,
        "max_abs": err.abs().max().item(),
        "effective_bits": effective_bits(bits),
    }
