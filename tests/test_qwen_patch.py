"""Unit tests for the M4 rotate-query identity (turbo_kv.qwen_patch math).

The transformers patch itself needs a GPU model, but its *correctness* rests on
two orthogonal-rotation identities that we test here at the tensor level:

* logits:  qᵀk == (Rq)ᵀ(Rk)
* output:  Rᵀ(softmax(qkᵀ)·(Rv)) == softmax(qkᵀ)·v

plus a full single-head attention equivalence (rotated vs unrotated) with no
quantization, and that the patch module imports cleanly.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from turbo_kv import rotations as R  # noqa: E402


def _sdpa(q, k, v):
    d = q.shape[-1]
    w = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(d), dim=-1)
    return w @ v


@pytest.mark.parametrize("kind", ["rht", "dense", "none"])
def test_rotate_query_preserves_logits(kind):
    torch.manual_seed(0)
    B, H, T, D = 2, 3, 16, 64
    q = torch.randn(B, H, T, D, dtype=torch.float64)
    k = torch.randn(B, H, T, D, dtype=torch.float64)
    rot = R.make_rotation(kind, D, dtype=torch.float64)
    logits = q @ k.transpose(-1, -2)
    logits_rot = rot.rotate(q) @ rot.rotate(k).transpose(-1, -2)
    assert torch.allclose(logits, logits_rot, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("kind", ["rht", "dense", "none"])
def test_value_rotation_inverse_recovers_output(kind):
    torch.manual_seed(1)
    B, H, T, D = 2, 3, 16, 64
    q = torch.randn(B, H, T, D, dtype=torch.float64)
    k = torch.randn(B, H, T, D, dtype=torch.float64)
    v = torch.randn(B, H, T, D, dtype=torch.float64)
    rot = R.make_rotation(kind, D, dtype=torch.float64)
    o = _sdpa(q, k, v)
    # accumulate in rotated space then inverse-rotate the output once
    o_rot = _sdpa(rot.rotate(q), rot.rotate(k), rot.rotate(v))
    o_rec = rot.inverse(o_rot)
    assert torch.allclose(o, o_rec, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("kind", ["rht", "dense"])
def test_full_attention_rotated_equiv_unrotated(kind):
    torch.manual_seed(2)
    B, H, T, D = 1, 2, 32, 64
    q = torch.randn(B, H, T, D, dtype=torch.float64)
    k = torch.randn(B, H, T, D, dtype=torch.float64)
    v = torch.randn(B, H, T, D, dtype=torch.float64)
    rot = R.make_rotation(kind, D, dtype=torch.float64)
    o_ref = _sdpa(q, k, v)
    o_turbo = rot.inverse(_sdpa(rot.rotate(q), rot.rotate(k), rot.rotate(v)))
    assert torch.allclose(o_ref, o_turbo, atol=1e-9, rtol=1e-9)


def test_qwen_patch_module_imports():
    # Importing must not require transformers (lazy imports inside the forward).
    from turbo_kv import qwen_patch

    assert hasattr(qwen_patch, "patch_qwen2_attention")
    assert hasattr(qwen_patch, "unpatch_qwen2_attention")
    assert hasattr(qwen_patch, "_turbo_sdpa_forward")
