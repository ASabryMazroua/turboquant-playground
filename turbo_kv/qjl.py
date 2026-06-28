"""``turbo_kv/qjl.py`` — TurboQuant-prod: 1-bit QJL residual (PLAN §4, M6).

A scalar MSE quantizer minimises ``‖Rk - \\widehat{Rk}‖`` but its reconstruction is
a *biased* estimate of the inner product ``(Rq)·(Rk)`` — the residual
``r = Rk - \\widehat{Rk}`` correlates with the quantization grid, so attention
logits are systematically off (this is exactly what M4 saw end-to-end: 4-bit
MSE-only KV inflates WikiText perplexity 15-57×).

TurboQuant-prod fixes the *bias* with a 1-bit **Quantized Johnson–Lindenstrauss**
sketch of the residual. Draw a fixed Gaussian sketch ``S ∈ ℝ^{m×d}`` and store
only ``sign(S r) ∈ {±1}^m`` (1 bit/coordinate when ``m = d``) plus ``‖r‖``. For a
standard normal row ``s``::

    E[ sign(sᵀr) · (sᵀq') ] = √(2/π) · (q'ᵀr)/‖r‖

so ``q'ᵀr`` is recovered **unbiasedly** by

    \\widehat{q'ᵀr} = ‖r‖ · √(π/2) · (1/m) Σ_j sign(s_jᵀr) (s_jᵀq')

and the decode-time logit estimate is

    qᵀk ≈ (Rq)ᵀ\\widehat{Rk} + \\widehat{(Rq)ᵀr}_{QJL}

at ``(b-1)``-bit MSE recon + 1-bit sketch = ``b`` bits total, but unbiased.

``torch`` is imported lazily so this module imports without a GPU.
"""
from __future__ import annotations

import math

from turbo_kv import quantizers as Q


class QJLSketch:
    """Fixed Gaussian sign-sketch for unbiased inner-product estimation of a residual.

    ``m`` projection rows (default ``m = dim`` → 1 bit per coordinate). The sketch
    matrix ``S`` is built lazily on first use and cached on the right device/dtype.
    """

    def __init__(self, dim: int, m: int | None = None, seed: int = 0) -> None:
        self.dim = int(dim)
        self.m = int(m) if m is not None else int(dim)
        self.seed = int(seed)
        self._S = None  # [m, dim]

    def _ensure(self, device, dtype):
        if self._S is None:
            import torch

            g = torch.Generator(device="cpu").manual_seed(self.seed)
            S = torch.randn(self.m, self.dim, generator=g)
            self._S = S.to(device=device, dtype=dtype)
        return self._S

    def sketch(self, r):
        """Residuals ``r`` ``[..., d]`` → ``(signs bool [..., m], norm [..., 1])``."""
        import torch

        S = self._ensure(r.device, r.dtype)
        proj = r @ S.t()                       # [..., m]
        signs = proj >= 0                      # bool, 1 bit each
        norm = r.norm(dim=-1, keepdim=True)    # [..., 1]
        return signs, norm

    def estimate_matrix(self, q, signs, norm):
        """Unbiased estimate of every ``q_i · r_j`` → ``[nq, nk]``.

        ``q`` ``[nq, d]``; ``signs`` ``[nk, m]`` bool; ``norm`` ``[nk, 1]``.
        """
        import torch

        S = self._ensure(q.device, q.dtype)
        qproj = q @ S.t()                              # [nq, m]
        s = signs.to(q.dtype) * 2.0 - 1.0              # [nk, m] in {-1,+1}
        coeff = math.sqrt(math.pi / 2.0) / self.m
        return coeff * (qproj @ s.t()) * norm.reshape(1, -1)   # [nq, nk]

    def sketch_bits_per_value(self) -> float:
        """Sketch storage in bits per stored key *value* (m sign bits over d coords)."""
        return self.m / self.dim


def quantize_key_mse(Rk, bits: float, *, axis: str = "token"):
    """MSE-only baseline: reconstruct ``Rk`` at ``bits`` (returns ``\\widehat{Rk}``)."""
    return Q.fake_quantize(Rk, bits, axis=axis, symmetric=False)


def encode_key_prod(Rk, bits: float, sketch: QJLSketch, *, axis: str = "token"):
    """TurboQuant-prod key encode: ``(b-1)``-bit MSE recon + 1-bit QJL of residual.

    Returns ``(Rk_hat, signs, norm)`` — ``Rk_hat`` is the dequantized recon,
    ``signs``/``norm`` the residual sketch consumed by :meth:`estimate_matrix`.
    """
    Rk_hat = Q.fake_quantize(Rk, max(1.0, bits - 1.0), axis=axis, symmetric=False).to(Rk.dtype)
    signs, norm = sketch.sketch(Rk - Rk_hat)
    return Rk_hat, signs, norm


def logits_mse(Rq, Rk, bits: float, *, axis: str = "token"):
    """Biased MSE-only logit matrix ``(Rq)·\\widehat{Rk}`` at ``bits``."""
    Rk_hat = quantize_key_mse(Rk, bits, axis=axis).to(Rq.dtype)
    return Rq @ Rk_hat.t()


def logits_prod(Rq, Rk, bits: float, sketch: QJLSketch, *, axis: str = "token"):
    """Unbiased prod logit matrix: ``(Rq)·\\widehat{Rk} + \\widehat{(Rq)·r}_{QJL}``."""
    Rk_hat, signs, norm = encode_key_prod(Rk, bits, sketch, axis=axis)
    base = Rq @ Rk_hat.t()
    resid = sketch.estimate_matrix(Rq, signs, norm)
    return base + resid


def prod_bits_per_value(bits: float, sketch: QJLSketch, *, head_dim: int = 64) -> float:
    """Realized rate of prod: ``(b-1)``-bit recon + sketch + per-token ‖r‖ (bf16)."""
    recon = Q.effective_bits(max(1.0, bits - 1.0))
    sketch_bits = sketch.sketch_bits_per_value()
    norm_bits = 16.0 / head_dim  # one bf16 ‖r‖ per token, amortized over d coords
    return recon + sketch_bits + norm_bits
