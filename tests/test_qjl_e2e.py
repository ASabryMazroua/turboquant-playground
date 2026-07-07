"""M16 — end-to-end QJL key cache: ``TurboKVCache.qjl_update_and_attend``.

These tests exercise the custom attention path that estimates ``qᵀk`` from the
QJL sign sketch (no key reconstruction → no SDPA). They validate the batched
attend against the M11-tested 2D primitives, that query-chunking is numerically
invariant, that an all-window cache reproduces exact attention, GQA shapes, and
the ``pre_rope`` incompatibility guard. No transformers / GPU required.
"""
import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from turbo_kv import qjl as QJL  # noqa: E402
from turbo_kv.cache import TurboKVCache  # noqa: E402


def _exact_attention(q, k, v, scaling):
    """Reference causal SDPA in the (already-rotated) basis → [B,Hq,T,D]."""
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    kr = k.repeat_interleave(G, dim=1)
    vr = v.repeat_interleave(G, dim=1)
    scores = torch.matmul(q, kr.transpose(-1, -2)) * scaling
    cm = torch.full((T, T), float("-inf")).triu(1)
    probs = F.softmax(scores + cm, dim=-1)
    return torch.matmul(probs, vr)


def _reference_all_history(cache, q, scaling, layer_idx=0):
    """Manual QJL attention when every token is in the history store (rl=0)."""
    B, Hq, T, D = q.shape
    cK = cache._cK[layer_idx]
    Hkv = cK["signs"].shape[1]
    G = Hq // Hkv
    sketch = cache._sketch
    Vfull = cache._decompress(cache._cV[layer_idx], torch.float32)  # [B,Hkv,Th,D]
    out = torch.zeros(B, Hq, T, D, dtype=torch.float32)
    cm = torch.full((T, T), float("-inf")).triu(1)
    for h in range(Hq):
        kv = h // G
        Rq = q[0, h].to(torch.float32)
        signs = cK["signs"][0, kv]
        norm = cK["norm"][0, kv].to(torch.float32)
        if "out_idx" in cK:
            logits = QJL.logits_direct(Rq, signs, norm, sketch,
                                       out_idx=cK["out_idx"][0, kv],
                                       out_val=cK["out_val"][0, kv])
        else:
            logits = sketch.estimate_matrix(Rq, signs, norm)
        probs = F.softmax(logits * scaling + cm, dim=-1)
        out[0, h] = probs @ Vfull[0, kv]
    return out


@pytest.mark.parametrize("n_out", [0, 4])
def test_qjl_attend_matches_2d_primitives(n_out):
    # The batched attend (estimate_batched + chunking + GQA broadcast) must equal
    # a per-head loop over the M11-validated 2D logits_direct / estimate_matrix.
    torch.manual_seed(0)
    B, Hq, Hkv, T, D, m = 1, 4, 2, 7, 16, 32
    G = Hq // Hkv
    q = torch.randn(B, Hq, T, D)
    k = torch.randn(B, Hkv, T, D)
    v = torch.randn(B, Hkv, T, D)
    cache = TurboKVCache(residual_length=0, rotation="none", head_dim=D,
                         key_quant="qjl", qjl_m=m, qjl_outliers=n_out, qjl_chunk=1000)
    o = cache.qjl_update_and_attend(q, k, v, 0, num_key_value_groups=G,
                                    scaling=D ** -0.5)
    ref = _reference_all_history(cache, q, D ** -0.5)
    assert o.shape == (B, Hq, T, D)
    assert torch.allclose(o, ref, atol=1e-4, rtol=1e-4)


def test_qjl_chunking_is_numerically_invariant():
    # Query chunking is purely a memory optimisation: each output row is computed
    # over the full key axis independently, so a tiny chunk must match a huge one.
    torch.manual_seed(1)
    B, Hq, Hkv, T, D, m = 1, 4, 2, 13, 16, 32
    G = Hq // Hkv
    q = torch.randn(B, Hq, T, D)
    k = torch.randn(B, Hkv, T, D)
    v = torch.randn(B, Hkv, T, D)

    def run(chunk):
        c = TurboKVCache(residual_length=3, rotation="none", head_dim=D,
                         key_quant="qjl", qjl_m=m, qjl_outliers=2, qjl_chunk=chunk,
                         seed=7)
        return c.qjl_update_and_attend(q, k, v, 0, num_key_value_groups=G,
                                       scaling=D ** -0.5)

    assert torch.allclose(run(2), run(10_000), atol=1e-6)


def test_qjl_all_window_reproduces_exact_attention():
    # With residual_length >= T nothing is evicted, so the QJL store is empty and
    # attention runs entirely over the exact BF16 window → must equal exact SDPA.
    torch.manual_seed(2)
    B, Hq, Hkv, T, D = 1, 6, 2, 9, 8
    G = Hq // Hkv
    q = torch.randn(B, Hq, T, D)
    k = torch.randn(B, Hkv, T, D)
    v = torch.randn(B, Hkv, T, D)
    cache = TurboKVCache(residual_length=100, rotation="none", head_dim=D,
                         key_quant="qjl", qjl_m=64)
    o = cache.qjl_update_and_attend(q, k, v, 0, num_key_value_groups=G,
                                    scaling=D ** -0.5)
    assert cache._cK[0] is None  # nothing evicted
    assert torch.allclose(o, _exact_attention(q, k, v, D ** -0.5), atol=1e-5)


def test_qjl_gqa_shapes_and_finite():
    # Real Qwen2.5-0.5B GQA factor: 14 query heads, 2 kv heads.
    torch.manual_seed(3)
    B, Hq, Hkv, T, D = 1, 14, 2, 20, 64
    G = Hq // Hkv
    q = torch.randn(B, Hq, T, D)
    k = torch.randn(B, Hkv, T, D)
    v = torch.randn(B, Hkv, T, D)
    cache = TurboKVCache(residual_length=4, rotation="none", head_dim=D,
                         key_quant="qjl", qjl_m=128, qjl_outliers=8, qjl_chunk=8)
    o = cache.qjl_update_and_attend(q, k, v, 0, num_key_value_groups=G,
                                    scaling=D ** -0.5)
    assert o.shape == (B, Hq, T, D)
    assert torch.isfinite(o).all()


def test_qjl_decode_token_by_token_matches_prefill():
    # Prefilling T tokens then reading must match feeding the same tokens one at a
    # time: the stored sketch is identical (fixed seed) and the per-row attention
    # is order-independent, so the final query's output agrees.
    torch.manual_seed(4)
    B, Hq, Hkv, T, D, m = 1, 4, 2, 10, 16, 64
    G = Hq // Hkv
    k = torch.randn(B, Hkv, T, D)
    v = torch.randn(B, Hkv, T, D)
    q = torch.randn(B, Hq, T, D)

    def fresh():
        return TurboKVCache(residual_length=3, rotation="none", head_dim=D,
                            key_quant="qjl", qjl_m=m, seed=11, qjl_chunk=4)

    o_prefill = fresh().qjl_update_and_attend(q, k, v, 0, num_key_value_groups=G,
                                              scaling=D ** -0.5)
    inc = fresh()
    last = None
    for t in range(T):
        last = inc.qjl_update_and_attend(
            q[:, :, t:t + 1], k[:, :, t:t + 1], v[:, :, t:t + 1], 0,
            num_key_value_groups=G, scaling=D ** -0.5)
    assert torch.allclose(o_prefill[:, :, -1:], last, atol=1e-4, rtol=1e-4)


def test_qjl_beats_per_token_int4_on_channel_outliers():
    # The scientific premise of M16: per-token int4 keys are wrecked by outlier
    # channels (M4), while the QJL unbiased-IP estimate is not. On a key matrix
    # with a couple of fat channels, QJL attention is closer to exact than the
    # per-token int4 cache at the same context.
    torch.manual_seed(5)
    B, Hq, Hkv, T, D = 1, 4, 2, 96, 64
    G = Hq // Hkv
    q = torch.randn(B, Hq, T, D)
    k = torch.randn(B, Hkv, T, D)
    v = torch.randn(B, Hkv, T, D)
    k[..., 7] *= 25.0          # persistent outlier channels
    k[..., 31] *= 18.0
    scaling = D ** -0.5
    exact = _exact_attention(q, k, v, scaling)

    qjl_cache = TurboKVCache(residual_length=0, rotation="none", head_dim=D,
                             key_quant="qjl", qjl_m=256, qjl_outliers=4)
    o_qjl = qjl_cache.qjl_update_and_attend(q, k, v, 0, num_key_value_groups=G,
                                            scaling=scaling)

    pt_cache = TurboKVCache(residual_length=0, rotation="none", head_dim=D,
                            key_quant="per_token")
    full_k, full_v = pt_cache.update(k, v, 0)
    o_pt = _exact_attention(q, full_k, full_v, scaling)

    rel_qjl = (o_qjl - exact).norm() / exact.norm()
    rel_pt = (o_pt - exact).norm() / exact.norm()
    assert rel_qjl < rel_pt


def test_qjl_pre_rope_incompatible():
    with pytest.raises(ValueError):
        TurboKVCache(key_quant="qjl", pre_rope=True)


def test_qjl_memory_accounting_counts_sketch():
    torch.manual_seed(6)
    B, Hq, Hkv, T, D = 1, 4, 2, 40, 64
    G = Hq // Hkv
    q = torch.randn(B, Hq, T, D)
    k = torch.randn(B, Hkv, T, D)
    v = torch.randn(B, Hkv, T, D)
    cache = TurboKVCache(residual_length=8, rotation="none", head_dim=D,
                         key_quant="qjl", qjl_m=128, qjl_outliers=8)
    cache.qjl_update_and_attend(q, k, v, 0, num_key_value_groups=G, scaling=D ** -0.5)
    mem = cache.memory_bytes()
    assert mem["qjl_bytes"] > 0          # sketch sign bits + norm + outliers
    assert mem["packed_bytes"] > 0       # int4 values still stored
    assert mem["total_bytes"] >= mem["qjl_bytes"] + mem["packed_bytes"]
