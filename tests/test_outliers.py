"""Unit tests for dense-and-sparse outlier keys (M10, KVQuant/QJL).

Sparse fp16 outliers rescue per-token int4 keys: keeping the few largest-magnitude
coordinates per token in fp16 keeps them out of the affine min/max so the dense
remainder uses the full int4 grid. Scoped to the per-token KEY axis only.
"""
import pytest

torch = pytest.importorskip("torch")

from turbo_kv import packing as P  # noqa: E402
from turbo_kv.cache import TurboKVCache  # noqa: E402


def _relerr(rec, ref):
    return float((rec - ref).norm() / ref.norm().clamp_min(1e-9))


def test_outlier_positions_are_exact_and_relerr_drops():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 5, 64)
    n = 8
    codes, scale, lo, out_idx, out_val = P.quantize_int4_per_token_outliers(x, n)
    rec = P.dequantize_int4_per_token_outliers(codes, scale, lo, out_idx, out_val)
    # outlier positions reconstruct exactly (stored fp16, scatter overwrites).
    gathered = rec.gather(-1, out_idx.to(torch.int64))
    assert torch.allclose(gathered, x.gather(-1, out_idx.to(torch.int64)), atol=1e-3)
    # overall relerr is lower than the no-outlier dense-only path.
    c0, s0, l0 = P.quantize_int4_per_token(x)
    rec0 = P.dequantize_int4_per_token(c0.float(), s0, l0)
    assert _relerr(rec, x) < _relerr(rec0, x)


def test_outliers_rescue_extreme_coordinate():
    x = torch.full((1, 1, 1, 64), 0.01)
    x[..., 17] = 1000.0  # one extreme coordinate inflates the whole token's range
    # without outliers: the small coords are crushed to a single bin.
    c0, s0, l0 = P.quantize_int4_per_token(x)
    rec0 = P.dequantize_int4_per_token(c0.float(), s0, l0)
    small = torch.arange(64) != 17
    err0 = _relerr(rec0[..., small], x[..., small])
    # with 1 outlier: the extreme coord is sparse, the small coords get the grid.
    c1, s1, l1, oi, ov = P.quantize_int4_per_token_outliers(x, 1)
    rec1 = P.dequantize_int4_per_token_outliers(c1, s1, l1, oi, ov)
    err1 = _relerr(rec1[..., small], x[..., small])
    assert err1 < err0 * 0.1  # sharp drop on the non-outlier coordinates


def test_n_outliers_zero_matches_dense_path():
    torch.manual_seed(1)
    x = torch.randn(1, 2, 4, 64)
    codes, scale, lo, out_idx, out_val = P.quantize_int4_per_token_outliers(x, 0)
    c0, s0, l0 = P.quantize_int4_per_token(x)
    assert torch.equal(codes, c0)
    assert torch.allclose(scale, s0) and torch.allclose(lo, l0)
    assert out_idx.numel() == 0 and out_val.numel() == 0


def _feed(cache, T, D=64, seed=0):
    torch.manual_seed(seed)
    k = torch.randn(1, 2, T, D, dtype=torch.bfloat16)
    v = torch.randn(1, 2, T, D, dtype=torch.bfloat16)
    fk, fv = cache.update(k, v, 0)
    return k, fk


def test_cache_outliers_improve_per_token_keys():
    base = TurboKVCache(residual_length=8, rotation="none", head_dim=64,
                        key_quant="per_token", key_outliers=0)
    out = TurboKVCache(residual_length=8, rotation="none", head_dim=64,
                       key_quant="per_token", key_outliers=8)
    k0, fk0 = _feed(base, 40, seed=3)
    k1, fk1 = _feed(out, 40, seed=3)
    assert _relerr(fk1.float(), k1.float()) < _relerr(fk0.float(), k0.float())


def test_cache_key_outliers_zero_has_no_outlier_store():
    cache = TurboKVCache(residual_length=8, rotation="none", head_dim=64,
                         key_quant="per_token", key_outliers=0)
    _feed(cache, 40, seed=4)
    store = cache._cK[0]
    assert store is not None
    assert "out_idx" not in store
    assert store.get("outliers", 0) == 0
