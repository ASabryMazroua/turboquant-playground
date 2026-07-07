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

    def estimate_batched(self, q, signs, norm):
        """Batched unbiased IP estimate over arbitrary leading (broadcast) dims.

        ``q`` ``[..., nq, d]``; ``signs`` ``[..., nk, m]`` bool; ``norm``
        ``[..., nk, 1]`` → ``[..., nq, nk]``. The leading batch dims of ``q`` and
        the sketch tensors must be broadcastable (e.g. ``q`` carries a GQA group
        axis the per-kv-head ``signs``/``norm`` broadcast over). This is the
        end-to-end (M16) entry point: it never materialises a reconstructed key,
        only the ``[..., nq, nk]`` logit block (which the caller chunks over
        ``nq`` to bound prefill memory).
        """
        import torch

        S = self._ensure(q.device, q.dtype)
        qproj = q @ S.t()                                  # [..., nq, m]
        s = signs.to(q.dtype) * 2.0 - 1.0                  # [..., nk, m]
        coeff = math.sqrt(math.pi / 2.0) / self.m
        est = coeff * torch.matmul(qproj, s.transpose(-1, -2))   # [..., nq, nk]
        return est * norm.transpose(-1, -2)                # norm [...,nk,1]->[...,1,nk]

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


# ---------------------------------------------------------------------------
# M11 — "QJL done right": direct large-m sign sketch of the key (no MSE base).
#
# M6 found the 1-bit *residual* sketch is variance-limited (only neutral near
# ≈3.5-4 bits). The practical QJL / TurboQuant-prod of Zandieh et al. never
# builds an MSE reconstruction base at all: it sketches the **rotated key
# itself** with a LARGE Gaussian sign sketch (``m=256``, ``m=512`` early layers)
# and keeps only ``sign(S·Rk) ∈ {±1}^m`` plus the scalar norm ``‖Rk‖``. Variance
# of the unbiased IP estimator scales as ``1/m``, so a wide sketch — not a recon
# base — is what buys low attention-KL. A handful of fp16 *outlier* coordinates
# (the few extreme |coords| of ``Rk``) are stored exactly to remove their large
# contribution to the sketch variance.
#
# Design decision (documented): we **zero the outlier coordinates of ``Rk``
# before sketching**, so the sign sketch models only the dense/non-outlier part
# (smaller ‖·‖ → lower variance), and the decode-time estimate adds back the
# *exact* outlier inner-product term:
#
#     qᵀk ≈ \\widehat{(Rq)ᵀ Rk_dense}_{QJL}  +  (Rq)ᵀ Rk_outliers
#
# where ``Rk_dense`` is ``Rk`` with the top-``n_outliers`` |coords| set to 0.
# ---------------------------------------------------------------------------


def encode_key_direct(Rk, sketch: QJLSketch, n_outliers: int = 0):
    """Direct QJL key encode — LARGE-m sign sketch of ``Rk`` itself, no MSE base.

    Returns ``(signs, norm, out_idx, out_val)``:

    * ``signs``/``norm`` — :meth:`QJLSketch.sketch` of ``Rk`` with the top
      ``n_outliers`` |coords| of each key zeroed (so the sketch models the dense
      part only; ``norm = ‖Rk_dense‖``).
    * ``out_idx`` ``[nk, n_outliers]`` long / ``out_val`` ``[nk, n_outliers]``
      fp16 — the exact (signed) outlier coordinates, or ``(None, None)`` when
      ``n_outliers == 0``.
    """
    import torch

    n_outliers = int(n_outliers)
    if n_outliers <= 0:
        signs, norm = sketch.sketch(Rk)
        return signs, norm, None, None

    k = min(n_outliers, Rk.shape[-1])
    out_idx = torch.topk(Rk.abs(), k=k, dim=-1).indices        # [nk, k] long
    out_val = torch.gather(Rk, -1, out_idx).to(torch.float16)  # signed, fp16
    dense = Rk.clone()
    dense.scatter_(-1, out_idx, torch.zeros_like(out_val, dtype=Rk.dtype))
    signs, norm = sketch.sketch(dense)
    return signs, norm, out_idx, out_val


def logits_direct(Rq, signs, norm, sketch: QJLSketch, *, out_idx=None, out_val=None,
                  Rk_for_outliers=None):
    """Unbiased direct-QJL logit matrix ``[nq, nk]`` = sketch IP estimate of the
    dense part ``+`` exact outlier IP.

    ``out_idx``/``out_val`` come from :func:`encode_key_direct`. The outlier term
    is built as a sparse ``[nk, D]`` matrix (zeros except each key's outlier
    coords ``= out_val``) so ``Rq @ outlier_mat.t()`` adds the exact, per-key
    ``Rq[:, idx]·val`` contribution with the correct ``[nq, nk]`` shape. If
    ``out_val`` is ``None`` but ``Rk_for_outliers`` is given, the exact values
    are gathered from it instead.
    """
    import torch

    base = sketch.estimate_matrix(Rq, signs, norm)             # [nq, nk]
    if out_idx is None or (out_val is None and Rk_for_outliers is None):
        return base

    nk = base.shape[1]
    D = Rq.shape[-1]
    out_idx = out_idx.to(torch.long)
    if out_val is not None:
        vals = out_val.to(Rq.dtype)
    else:
        vals = torch.gather(Rk_for_outliers, -1, out_idx).to(Rq.dtype)
    outlier_mat = torch.zeros(nk, D, device=Rq.device, dtype=Rq.dtype)
    outlier_mat.scatter_(-1, out_idx, vals)                    # [nk, D] sparse
    return base + Rq @ outlier_mat.t()


def direct_bits_per_value(sketch: QJLSketch, *, head_dim: int = 64, n_outliers: int = 0) -> float:
    """Realized rate of direct QJL: ``m`` sign bits + bf16 ‖Rk‖ + fp16 outliers
    (value + index), all amortized over ``head_dim`` coordinates."""
    sign_bits = sketch.m / head_dim
    norm_bits = 16.0 / head_dim
    idx_bits = math.ceil(math.log2(head_dim)) if head_dim > 1 else 0
    outlier_bits = int(n_outliers) * (16.0 + idx_bits) / head_dim
    return sign_bits + norm_bits + outlier_bits
