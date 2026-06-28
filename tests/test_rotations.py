"""Unit tests for turbo_kv.rotations (M2 gate: RRᵀ=I, Hadamard correctness).

Run on a torch-enabled env (AML): ``python -m pytest tests/ -q``.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from turbo_kv import rotations as R  # noqa: E402


DIMS = [16, 32, 64, 128, 256]


@pytest.mark.parametrize("n", DIMS)
def test_hadamard_orthonormal(n):
    H = R.hadamard_matrix(n, dtype=torch.float64)
    I = H @ H.t()
    assert torch.allclose(I, torch.eye(n, dtype=torch.float64), atol=1e-9)


@pytest.mark.parametrize("n", DIMS)
def test_fwht_matches_dense_hadamard(n):
    x = torch.randn(7, n, dtype=torch.float64)
    H = R.hadamard_matrix(n, dtype=torch.float64)
    dense = x @ H.t()  # H symmetric, so H xᵀ == (x Hᵀ)
    fast = R.fwht(x)
    assert torch.allclose(dense, fast, atol=1e-9)


@pytest.mark.parametrize("n", DIMS)
def test_fwht_involution(n):
    x = torch.randn(5, n, dtype=torch.float64)
    assert torch.allclose(R.fwht(R.fwht(x)), x, atol=1e-9)


def test_fwht_rejects_non_pow2():
    with pytest.raises(ValueError):
        R.fwht(torch.randn(3, 48))


@pytest.mark.parametrize("kind", ["dense", "rht"])
@pytest.mark.parametrize("n", DIMS)
def test_rotation_orthogonal_inverse(kind, n):
    rot = R.make_rotation(kind, n, seed=1, dtype=torch.float64)
    x = torch.randn(11, n, dtype=torch.float64)
    # inverse(rotate(x)) == x
    assert torch.allclose(rot.inverse(rot.rotate(x)), x, atol=1e-9)


@pytest.mark.parametrize("kind", ["dense", "rht"])
@pytest.mark.parametrize("n", DIMS)
def test_rotation_preserves_norm_and_inner_product(kind, n):
    rot = R.make_rotation(kind, n, seed=2, dtype=torch.float64)
    q = torch.randn(13, n, dtype=torch.float64)
    k = torch.randn(13, n, dtype=torch.float64)
    # Norm preserved
    assert torch.allclose(rot.rotate(q).norm(dim=-1), q.norm(dim=-1), atol=1e-9)
    # Inner product preserved: qᵀk == (Rq)ᵀ(Rk)
    ip = (q * k).sum(-1)
    ip_rot = (rot.rotate(q) * rot.rotate(k)).sum(-1)
    assert torch.allclose(ip, ip_rot, atol=1e-8)


def test_dense_matrix_is_orthogonal():
    rot = R.DenseRotation(64, seed=3, dtype=torch.float64)
    M = rot.as_matrix()
    assert torch.allclose(M @ M.t(), torch.eye(64, dtype=torch.float64), atol=1e-9)


def test_rht_rejects_non_pow2():
    with pytest.raises(ValueError):
        R.make_rotation("rht", 48)
