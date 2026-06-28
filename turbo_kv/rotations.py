"""Orthogonal rotations for TurboQuant KV-cache (PLAN §4, M2).

Two rotation families, both orthogonal (``R Rᵀ = I``) so that inner products are
preserved exactly before quantization::

    qᵀk = (Rq)ᵀ(Rk)

* :class:`DenseRotation` — a Haar-ish random orthogonal matrix (QR of a Gaussian).
  Apply cost is ``O(d²)`` per vector. Used as the *reference* rotation quality.
* :class:`RandomizedHadamard` — the structured randomized Hadamard transform
  ``R = D₁ H D₂ H`` (PLAN §4): two random sign diagonals interleaved with the
  normalized Walsh–Hadamard transform. Apply cost is ``O(d log d)`` via the fast
  Walsh–Hadamard transform (:func:`fwht`), with the *same* energy-spreading
  property as a dense rotation but far cheaper at large ``head_dim``.

``torch`` is imported lazily so the module can be imported by non-GPU tooling.
``head_dim`` for Qwen2.5-0.5B is 64 (a power of two), which the RHT requires.
"""
from __future__ import annotations

import math


# --------------------------------------------------------------------------- #
# Walsh–Hadamard transform
# --------------------------------------------------------------------------- #


def is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def fwht(x, *, normalize: bool = True):
    """Fast Walsh–Hadamard transform along the last dim (``O(d log d)``).

    Operates on a tensor ``x`` of shape ``[..., d]`` with ``d`` a power of two.
    With ``normalize=True`` the transform is orthonormal (``H Hᵀ = I``), so
    applying :func:`fwht` twice is the identity.
    """
    import torch

    x = torch.as_tensor(x)
    n = x.shape[-1]
    if not is_pow2(n):
        raise ValueError(f"fwht needs a power-of-two last dim, got {n}")
    orig_shape = x.shape
    y = x.clone()
    h = 1
    while h < n:
        y = y.view(*orig_shape[:-1], -1, 2 * h)
        a = y[..., :h]
        b = y[..., h : 2 * h]
        y = torch.cat([a + b, a - b], dim=-1).reshape(orig_shape)
        h *= 2
    if normalize:
        y = y / math.sqrt(n)
    return y


def hadamard_matrix(n: int, *, device=None, dtype=None):
    """Dense normalized Walsh–Hadamard matrix ``H`` (``H = Hᵀ``, ``H H = I``)."""
    import torch

    if not is_pow2(n):
        raise ValueError(f"hadamard_matrix needs power-of-two n, got {n}")
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    h = h / math.sqrt(n)
    if dtype is None:
        dtype = torch.float32
    return h.to(device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# Rotation interface
# --------------------------------------------------------------------------- #


class Rotation:
    """Base interface: ``rotate`` applies ``R``, ``inverse`` applies ``Rᵀ``.

    Convention: vectors live in the last dim. For a batch ``x`` of shape
    ``[..., d]``, ``rotate(x)`` returns ``x`` with each row ``v`` mapped to
    ``R v``.
    """

    kind: str = "base"
    dim: int = 0

    def rotate(self, x):  # pragma: no cover - interface
        raise NotImplementedError

    def inverse(self, x):  # pragma: no cover - interface
        raise NotImplementedError


class IdentityRotation(Rotation):
    kind = "none"

    def __init__(self, dim: int):
        self.dim = dim

    def rotate(self, x):
        return x

    def inverse(self, x):
        return x


class DenseRotation(Rotation):
    """Random orthogonal matrix via QR of a Gaussian (sign-fixed for determinism)."""

    kind = "dense"

    def __init__(self, dim: int, *, seed: int = 0, device=None, dtype=None):
        import torch

        self.dim = dim
        gen = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        # Fix column signs by sign(diag(R)) so the factorization is deterministic.
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        if dtype is None:
            dtype = torch.float32
        self.R = q.to(device=device, dtype=dtype)

    def rotate(self, x):
        import torch

        x = torch.as_tensor(x)
        return x @ self.R.t().to(x.dtype)

    def inverse(self, x):
        import torch

        x = torch.as_tensor(x)
        return x @ self.R.to(x.dtype)

    def as_matrix(self):
        return self.R


class RandomizedHadamard(Rotation):
    """Structured rotation ``R = D₁ H D₂ H`` with ``O(d log d)`` apply via FWHT.

    ``D₁, D₂`` are random ±1 diagonals; ``H`` is the orthonormal Walsh–Hadamard
    transform. Orthogonal because ``H H = I`` and ``Dᵢ Dᵢ = I``::

        R Rᵀ = D₁ H D₂ H · H D₂ H D₁ = I
    """

    kind = "rht"

    def __init__(self, dim: int, *, seed: int = 0, device=None, dtype=None):
        import torch

        if not is_pow2(dim):
            raise ValueError(f"RHT needs power-of-two dim, got {dim}")
        self.dim = dim
        gen = torch.Generator(device="cpu").manual_seed(seed)
        if dtype is None:
            dtype = torch.float32
        signs1 = torch.randint(0, 2, (dim,), generator=gen) * 2 - 1
        signs2 = torch.randint(0, 2, (dim,), generator=gen) * 2 - 1
        self.d1 = signs1.to(device=device, dtype=dtype)
        self.d2 = signs2.to(device=device, dtype=dtype)

    def rotate(self, x):
        # R x = D₁ H D₂ H x
        import torch

        x = torch.as_tensor(x)
        d1 = self.d1.to(x.dtype)
        d2 = self.d2.to(x.dtype)
        y = fwht(x)
        y = y * d2
        y = fwht(y)
        y = y * d1
        return y

    def inverse(self, x):
        # Rᵀ x = H D₂ H D₁ x
        import torch

        x = torch.as_tensor(x)
        d1 = self.d1.to(x.dtype)
        d2 = self.d2.to(x.dtype)
        y = x * d1
        y = fwht(y)
        y = y * d2
        y = fwht(y)
        return y


def make_rotation(kind: str, dim: int, *, seed: int = 0, device=None, dtype=None) -> Rotation:
    """Factory: ``kind`` ∈ {``none``, ``dense``, ``rht``}."""
    kind = kind.lower()
    if kind in ("none", "identity", "id"):
        return IdentityRotation(dim)
    if kind == "dense":
        return DenseRotation(dim, seed=seed, device=device, dtype=dtype)
    if kind in ("rht", "hadamard", "randomized_hadamard"):
        return RandomizedHadamard(dim, seed=seed, device=device, dtype=dtype)
    raise ValueError(f"unknown rotation kind: {kind!r}")
