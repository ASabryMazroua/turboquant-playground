"""M9 — per-layer bit allocation: keep the first ``bf16_layers`` layers in BF16.

The most quantization-sensitive early layers (M2 found layer 0 a huge
inner-product RMSE outlier) are kept entirely in BF16 — never evicted/compressed
— while the remaining layers stay int4. ``bf16_layers=0`` is byte-for-byte the
all-int4 cache. These run on a GPU node; locally they only need to py_compile.
"""
import pytest

torch = pytest.importorskip("torch")

from turbo_kv.cache import TurboKVCache  # noqa: E402


def _rand_kv(B=1, H=2, T=1, D=64):
    return (torch.randn(B, H, T, D, dtype=torch.bfloat16),
            torch.randn(B, H, T, D, dtype=torch.bfloat16))


def test_bf16_layer_is_exact_past_window():
    # A BF16-allocated layer keeps ALL tokens exact even past residual_length,
    # and never populates the compressed store.
    cache = TurboKVCache(bf16_layers=1, residual_length=8, head_dim=64,
                         rotation="none", key_quant="per_channel")
    k, v = _rand_kv(T=40)
    fk, fv = cache.update(k, v, 0)
    assert fk.shape == k.shape
    assert torch.equal(fk, k) and torch.equal(fv, v)
    assert cache._cK[0] is None and cache._cV[0] is None
    assert cache.get_seq_length(0) == 40


def test_non_bf16_layer_still_compresses():
    # The same cache: a layer at/above bf16_layers still evicts + compresses.
    cache = TurboKVCache(bf16_layers=1, residual_length=8, head_dim=64,
                         rotation="none", key_quant="per_channel")
    k, v = _rand_kv(T=40)
    fk, _ = cache.update(k, v, 1)
    assert cache._cK[1] is not None  # history was compressed
    assert fk.shape[2] == 40


def test_default_bf16_layers_zero_still_compresses_layer0():
    # Default bf16_layers=0 is the unchanged all-int4 cache: layer 0 compresses.
    cache = TurboKVCache(residual_length=8, head_dim=64, rotation="none",
                         key_quant="per_channel")
    k, v = _rand_kv(T=40)
    fk, _ = cache.update(k, v, 0)
    assert cache._cK[0] is not None
    assert fk.shape[2] == 40
