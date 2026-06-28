"""M8 — pre-RoPE key quantization (KVQuant fix) unit tests.

Keys are stored/quantized in their RAW pre-rotary basis and RoPE is re-applied to
the reconstructed keys at attention time (see :class:`turbo_kv.cache.TurboKVCache`
with ``pre_rope=True``). These tests validate the reconstruction numerics against
a HuggingFace-style reference RoPE. ``torch`` is imported lazily; they run on the
GPU node and ``py_compile`` on CPU.
"""
import pytest

torch = pytest.importorskip("torch")

from turbo_kv.cache import TurboKVCache  # noqa: E402


def _inv_freq(D, base=10000.0):
    # HuggingFace rotary base frequencies: theta^{-2i/D}, i in [0, D/2).
    return 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float32) / D))


def _rotate_half_ref(x):
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope_ref(k, positions, inv_freq):
    """HuggingFace-style RoPE on ``[B,H,T,D]`` keys for absolute ``positions``."""
    pos = positions.to(torch.float32)
    freqs = pos[:, None] * inv_freq[None, :]          # [T, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)           # [T, D]
    cos = emb.cos()[None, None, :, :]                 # [1, 1, T, D]
    sin = emb.sin()[None, None, :, :]
    kf = k.to(torch.float32)
    return kf * cos + _rotate_half_ref(kf) * sin


# --------------------------------------------------------------------------- #
# 1) rotate_half + cos/sin reconstruction round-trips vs an inline HF reference
# --------------------------------------------------------------------------- #
def test_cache_rope_matches_hf_reference():
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 12, 64
    inv_freq = _inv_freq(D)
    k_pre = torch.randn(B, H, T, D, dtype=torch.float32)
    positions = torch.arange(T, dtype=torch.int32)

    cache = TurboKVCache(residual_length=64, rotation="none", head_dim=D,
                         key_quant="per_channel", pre_rope=True)
    cache._rope_inv_freq = inv_freq
    got = cache._apply_rope(k_pre, positions, torch.float32)
    ref = _apply_rope_ref(k_pre, positions, inv_freq)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)

    # rotate_half itself round-trips: rotate_half(rotate_half(x)) == -x.
    x = torch.randn(B, H, T, D)
    assert torch.allclose(cache._rotate_half(cache._rotate_half(x)), -x, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2) short context (no eviction): returned keys equal manually-RoPE'd input keys
# --------------------------------------------------------------------------- #
def test_pre_rope_short_context_equals_manual_rope():
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 10, 64
    inv_freq = _inv_freq(D)
    k_pre = torch.randn(B, H, T, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, dtype=torch.bfloat16)
    positions = torch.arange(T, dtype=torch.int32)

    cache = TurboKVCache(residual_length=64, rotation="none", head_dim=D,
                         key_quant="per_channel", pre_rope=True)
    fk, fv = cache.update(k_pre, v, 0, {"cache_position": positions,
                                        "rope_inv_freq": inv_freq})
    assert fk.shape == (B, H, T, D)
    # Window is exact BF16 (< residual_length), so reconstruction == RoPE(k_pre).
    ref = _apply_rope_ref(k_pre, positions, inv_freq).to(torch.bfloat16)
    assert torch.allclose(fk.float(), ref.float(), atol=2e-2, rtol=2e-2)
    # Values are unaffected by RoPE.
    assert torch.equal(fv, v)


# --------------------------------------------------------------------------- #
# 3) eviction path: token-by-token > residual_length, positions stay aligned
# --------------------------------------------------------------------------- #
def test_pre_rope_eviction_positions_align():
    torch.manual_seed(0)
    B, H, D = 1, 2, 64
    inv_freq = _inv_freq(D)
    cache = TurboKVCache(residual_length=16, rotation="none", head_dim=D,
                         key_quant="per_channel", key_group_size=8, pre_rope=True)

    # prefill 40 tokens, then decode 20 one at a time → 60 total, plenty evicted.
    pos = 0
    k0 = torch.randn(B, H, 40, D, dtype=torch.bfloat16)
    v0 = torch.randn(B, H, 40, D, dtype=torch.bfloat16)
    cp0 = torch.arange(40, dtype=torch.int32)
    fk, fv = cache.update(k0, v0, 0, {"cache_position": cp0, "rope_inv_freq": inv_freq})
    pos = 40
    for _ in range(20):
        k = torch.randn(B, H, 1, D, dtype=torch.bfloat16)
        v = torch.randn(B, H, 1, D, dtype=torch.bfloat16)
        cp = torch.tensor([pos], dtype=torch.int32)
        fk, fv = cache.update(k, v, 0, {"cache_position": cp, "rope_inv_freq": inv_freq})
        pos += 1

    assert cache.get_seq_length(0) == 60
    assert fk.shape[2] == 60
    # positions buffer aligns 1:1 with the reconstructed key length.
    assert cache._pos[0].shape[0] == fk.shape[2] == 60
    assert torch.isfinite(fk.float()).all()


# --------------------------------------------------------------------------- #
# 4) pre_rope=False is unaffected (no positions buffer, no inv_freq needed)
# --------------------------------------------------------------------------- #
def test_pre_rope_off_is_unchanged():
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 50, 64
    k = torch.randn(B, H, T, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, dtype=torch.bfloat16)
    cache = TurboKVCache(residual_length=16, rotation="none", head_dim=D,
                         key_quant="per_channel")
    fk, fv = cache.update(k, v, 0)
    assert fk.shape == (B, H, T, D)
    assert cache._pos[0] is None
    assert cache.memory_bytes()["position_bytes"] == 0
