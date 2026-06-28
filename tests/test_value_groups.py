"""Unit tests for group-wise VALUE quantization (M13, KIVI/AWQ).

Values are quantized per-token but in GROUPS of ``group_size`` consecutive
coordinates (one int4 scale/zero per group) instead of one scale per whole token.
Finer resolution when intra-head magnitude varies across the value head. Scoped
to VALUES only; keys keep their per-channel / per-token-outlier schemes. The
``residual_length`` BF16 window is the KIVI residual buffer (recent tokens kept
full precision, evicted in chunks).
"""
import pytest

torch = pytest.importorskip("torch")

from turbo_kv import packing as P  # noqa: E402
from turbo_kv.cache import TurboKVCache  # noqa: E402


def _relerr(rec, ref):
    return float((rec - ref).norm() / ref.norm().clamp_min(1e-9))


def test_grouped_beats_whole_head_when_magnitude_varies():
    # first 32 coords ~1.0, last 32 coords ~100.0 — a single whole-head scale is
    # set by the large block and crushes the small block; per-group scales fix it.
    torch.manual_seed(0)
    x = torch.empty(1, 1, 4, 64)
    x[..., :32] = 1.0 + 0.05 * torch.randn(1, 1, 4, 32)
    x[..., 32:] = 100.0 + 5.0 * torch.randn(1, 1, 4, 32)

    cg, sg, lg = P.quantize_int4_per_token_grouped(x, 32)
    rec_g = P.dequantize_int4_per_token_grouped(cg, sg, lg, 32)

    c0, s0, l0 = P.quantize_int4_per_token(x)
    rec_0 = P.dequantize_int4_per_token(c0.float(), s0, l0)

    assert _relerr(rec_g, x) < _relerr(rec_0, x)


def test_group_size_equals_dim_matches_whole_head():
    torch.manual_seed(1)
    x = torch.randn(1, 2, 5, 64)
    cg, sg, lg = P.quantize_int4_per_token_grouped(x, 64)  # one group == whole head
    rec_g = P.dequantize_int4_per_token_grouped(cg, sg, lg, 64)
    c0, s0, l0 = P.quantize_int4_per_token(x)
    rec_0 = P.dequantize_int4_per_token(c0.float(), s0, l0)
    assert torch.equal(cg, c0)
    assert torch.allclose(rec_g, rec_0, atol=1e-5)


def _feed_values(cache, T, D=64, seed=0):
    torch.manual_seed(seed)
    # values with intra-head magnitude variation, the regime grouping helps.
    k = torch.randn(1, 2, T, D, dtype=torch.bfloat16)
    v = torch.empty(1, 2, T, D)
    v[..., :32] = 1.0 + 0.05 * torch.randn(1, 2, T, 32)
    v[..., 32:] = 50.0 + 3.0 * torch.randn(1, 2, T, 32)
    v = v.to(torch.bfloat16)
    fk, fv = cache.update(k, v, 0)
    return k, v, fk, fv


def test_cache_grouped_values_reconstruct_better_keys_unchanged():
    base = TurboKVCache(residual_length=8, rotation="none", head_dim=64,
                        value_group_size=0)
    grp = TurboKVCache(residual_length=8, rotation="none", head_dim=64,
                       value_group_size=32)
    k0, v0, fk0, fv0 = _feed_values(base, 40, seed=3)
    k1, v1, fk1, fv1 = _feed_values(grp, 40, seed=3)
    # grouped values reconstruct at least as well (better here, magnitude varies).
    assert _relerr(fv1.float(), v1.float()) <= _relerr(fv0.float(), v0.float())
    # keys are unaffected by the value group size.
    assert _relerr(fk1.float(), k1.float()) == pytest.approx(
        _relerr(fk0.float(), k0.float()), rel=1e-6)


def test_value_group_size_zero_has_no_vgroup_store():
    cache = TurboKVCache(residual_length=8, rotation="none", head_dim=64,
                         value_group_size=0)
    _feed_values(cache, 40, seed=4)
    store = cache._cV[0]
    assert store is not None
    assert "vgroup" not in store
    assert store.get("vgroup", 0) == 0
