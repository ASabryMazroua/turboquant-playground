"""M14 — attention-sink + per-channel-keys combination (KVQuant/StreamingLLM).

The field's actual recipe keeps the first few tokens in fp16 (the attention sink)
AND quantizes keys per-channel. These tests VERIFY the BF16 sink composes
correctly with the post-M7 per-channel keys, M8 pre-RoPE, and M13 value groups —
the sink holds the stream's first ``sink_length`` tokens (positions 0..sink-1),
kept exact, and stays position-aligned with the reconstruction order
(sink -> history -> window) and the arrival-order ``_pos`` buffer used by pre_rope.

``torch`` is imported lazily; these run on the GPU node and ``py_compile`` on CPU.
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
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    kf = k.to(torch.float32)
    return kf * cos + _rotate_half_ref(kf) * sin


def _relerr(rec, ref):
    return float((rec - ref).norm() / ref.norm().clamp_min(1e-9))


# --------------------------------------------------------------------------- #
# 1) sink tokens are kept EXACT with per-channel keys; later tokens quantized
# --------------------------------------------------------------------------- #
def test_sink_exact_with_per_channel_keys():
    torch.manual_seed(0)
    B, H, T, D, S = 1, 2, 40, 64, 4
    k = torch.randn(B, H, T, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, dtype=torch.bfloat16)

    cache = TurboKVCache(residual_length=8, rotation="none", head_dim=D,
                         key_quant="per_channel", key_group_size=8, sink_length=S)
    fk, fv = cache.update(k, v, 0)

    assert fk.shape == (B, H, T, D)
    # the sink buffer holds exactly the first S tokens, byte-identical (BF16).
    assert cache._sK[0] is not None and cache._sK[0].shape[2] == S
    assert torch.equal(fk[:, :, :S, :], k[:, :, :S, :])
    # a token in the compressed history (index 10, well before the window) is
    # quantized, i.e. NOT exactly equal to its input key.
    assert not torch.equal(fk[:, :, 10, :], k[:, :, 10, :])
    assert _relerr(fk[:, :, 10, :].float(), k[:, :, 10, :].float()) > 0.0


# --------------------------------------------------------------------------- #
# 2) sink + pre_rope: position alignment; sink tokens get RoPE positions 0..S-1
# --------------------------------------------------------------------------- #
def test_sink_pre_rope_position_alignment():
    torch.manual_seed(0)
    B, H, T, D, S = 1, 2, 40, 64, 4
    inv_freq = _inv_freq(D)
    k = torch.randn(B, H, T, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, dtype=torch.bfloat16)
    positions = torch.arange(T, dtype=torch.int32)

    cache = TurboKVCache(residual_length=8, rotation="none", head_dim=D,
                         key_quant="per_channel", key_group_size=8,
                         pre_rope=True, sink_length=S)
    # returns without the pre_rope alignment assertion firing.
    fk, fv = cache.update(k, v, 0, {"cache_position": positions,
                                    "rope_inv_freq": inv_freq})
    assert fk.shape[2] == T
    assert cache._pos[0].shape[0] == fk.shape[2] == T
    # sink tokens are RoPE'd for positions 0..S-1 (raw BF16 keys -> RoPE on recon).
    ref = _apply_rope_ref(k[:, :, :S, :], positions[:S], inv_freq).to(torch.bfloat16)
    assert torch.allclose(fk[:, :, :S, :].float(), ref.float(), atol=2e-2, rtol=2e-2)
    assert torch.isfinite(fk.float()).all()


# --------------------------------------------------------------------------- #
# 3) sink + group-wise values compose: finite recon, first S values exact
# --------------------------------------------------------------------------- #
def test_sink_with_value_groups():
    torch.manual_seed(0)
    B, H, T, D, S = 1, 2, 40, 64, 4
    k = torch.randn(B, H, T, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, dtype=torch.bfloat16)

    cache = TurboKVCache(residual_length=8, rotation="none", head_dim=D,
                         key_quant="per_channel", key_group_size=8,
                         value_group_size=32, sink_length=S)
    fk, fv = cache.update(k, v, 0)

    assert fk.shape == fv.shape == (B, H, T, D)
    assert torch.isfinite(fv.float()).all()
    # first S value positions are kept exact in the BF16 sink.
    assert cache._sV[0] is not None and cache._sV[0].shape[2] == S
    assert torch.equal(fv[:, :, :S, :], v[:, :, :S, :])


# --------------------------------------------------------------------------- #
# 4) sink_length=0 is unchanged: no sink buffers populated
# --------------------------------------------------------------------------- #
def test_sink_length_zero_no_sink_buffers():
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 40, 64
    k = torch.randn(B, H, T, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, dtype=torch.bfloat16)

    cache = TurboKVCache(residual_length=8, rotation="none", head_dim=D,
                         key_quant="per_channel", key_group_size=8, sink_length=0)
    fk, fv = cache.update(k, v, 0)

    assert cache._sK[0] is None and cache._sV[0] is None
    assert fk.shape[2] == T
