"""Unit tests for turbo_kv.packing + turbo_kv.cache (M3 int4 TurboKVCache)."""
import pytest

torch = pytest.importorskip("torch")

from turbo_kv import packing as P  # noqa: E402
from turbo_kv.cache import TurboKVCache  # noqa: E402


# --------------------------------------------------------------------------- #
# packing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [2, 8, 63, 64, 128])
def test_pack_unpack_roundtrip(n):
    codes = torch.randint(0, 16, (3, 5, n), dtype=torch.uint8)
    packed = P.pack_int4(codes)
    assert packed.shape[-1] == (n + 1) // 2
    out = P.unpack_int4(packed, n)
    assert out.shape == codes.shape
    assert torch.equal(out, codes)


def test_packed_is_half_byte_per_value():
    codes = torch.randint(0, 16, (4, 64), dtype=torch.uint8)
    packed = P.pack_int4(codes)
    # 64 codes per row -> 32 bytes per row -> exactly 0.5 bytes/value.
    assert packed.shape[-1] == 32
    assert packed.numel() == codes.numel() // 2


def test_quantize_dequantize_within_step():
    torch.manual_seed(0)
    x = torch.randn(8, 2, 16, 64)
    codes, scale, lo = P.quantize_int4_per_token(x)
    assert codes.dtype == torch.uint8 and codes.max() <= 15
    rec = P.dequantize_int4_per_token(codes.float(), scale, lo)
    # Per-token affine: |x - x̂| ≤ scale/2.
    assert (rec - x).abs().amax(dim=-1, keepdim=True).le(scale / 2 + 1e-4).all()


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def _rand_kv(B=1, H=2, T=1, D=64):
    return torch.randn(B, H, T, D, dtype=torch.bfloat16), torch.randn(B, H, T, D, dtype=torch.bfloat16)


def test_cache_short_context_is_exact():
    # With seq ≤ residual_length nothing is compressed → exact BF16 passthrough.
    cache = TurboKVCache(residual_length=64, rotation="dense", head_dim=64)
    k, v = _rand_kv(T=10)
    fk, fv = cache.update(k, v, 0)
    assert torch.equal(fk, k) and torch.equal(fv, v)
    assert cache.get_seq_length(0) == 10


def test_cache_compresses_overflow_and_reconstructs():
    cache = TurboKVCache(residual_length=32, rotation="dense", head_dim=64)
    k, v = _rand_kv(T=200)
    fk, fv = cache.update(k, v, 0)
    assert cache.get_seq_length(0) == 200
    assert fk.shape == k.shape
    # Reconstruction error is bounded (int4 lossy but coherent).
    rel = ((fk.float() - k.float()).norm() / k.float().norm()).item()
    assert rel < 0.5


def test_cache_decode_appends_one_token_at_a_time():
    cache = TurboKVCache(residual_length=16, rotation="rht", head_dim=64)
    k0, v0 = _rand_kv(T=20)
    cache.update(k0, v0, 0)
    for _ in range(10):
        k, v = _rand_kv(T=1)
        fk, fv = cache.update(k, v, 0)
    assert cache.get_seq_length(0) == 30
    assert fk.shape[2] == 30


def test_sink_tokens_kept_exact():
    # The first ``sink_length`` tokens must be reconstructed exactly (BF16),
    # while the same tokens are lossy without a sink — attention sinks receive
    # massive attention and are catastrophic to quantize.
    sink = 4
    k, v = _rand_kv(T=300)
    fk_sink, _ = TurboKVCache(residual_length=32, rotation="none", head_dim=64,
                              sink_length=sink).update(k, v, 0)
    fk_nosink, _ = TurboKVCache(residual_length=32, rotation="none", head_dim=64,
                                sink_length=0).update(k, v, 0)
    assert fk_sink.shape == k.shape
    # Sink tokens are bit-exact; non-sink reconstruction is lossy (int4).
    assert torch.allclose(fk_sink[:, :, :sink, :].float(), k[:, :, :sink, :].float(), atol=1e-2)
    sink_err = (fk_sink[:, :, :sink, :].float() - k[:, :, :sink, :].float()).abs().mean()
    nosink_err = (fk_nosink[:, :, :sink, :].float() - k[:, :, :sink, :].float()).abs().mean()
    assert sink_err < nosink_err


def test_memory_is_roughly_4x_smaller_than_bf16():
    # Long context, small window → compressed store dominates → ≈4× reduction.
    cache = TurboKVCache(residual_length=8, rotation="dense", head_dim=64)
    B, H, T, D = 1, 2, 1024, 64
    k, v = torch.randn(B, H, T, D, dtype=torch.bfloat16), torch.randn(B, H, T, D, dtype=torch.bfloat16)
    cache.update(k, v, 0)
    mb = cache.memory_bytes()
    bf16_bytes = 2 * B * H * T * D * 2  # keys+values, 2 bytes/val
    ratio = bf16_bytes / mb["total_bytes"]
    assert ratio > 3.0  # 4× minus scale/zero + window overhead


def test_multi_layer_independent():
    cache = TurboKVCache(residual_length=16, rotation="none", head_dim=64)
    for li in range(3):
        k, v = _rand_kv(T=50)
        cache.update(k, v, li)
    assert len(cache) == 3
    for li in range(3):
        assert cache.get_seq_length(li) == 50
