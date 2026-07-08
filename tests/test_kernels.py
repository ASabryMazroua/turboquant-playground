"""Reference-equality tests for the fused int4 Triton kernels (M5).

These need a GPU + Triton, so they are skipped on CPU-only machines (they run in
the GPU job). Each kernel is checked against the dequantize-then-matmul PyTorch
reference by relative Frobenius error.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():  # pragma: no cover - CPU import guard
    pytest.skip("kernels require CUDA", allow_module_level=True)

from kernels.int4_logits_triton import int4_logits, int4_logits_reference  # noqa: E402
from kernels.int4_values_triton import int4_values_rot, int4_values_reference  # noqa: E402
from turbo_kv import packing as P  # noqa: E402


def _quantize_rows(x):
    codes, scale, lo = P.quantize_int4_per_token(x)
    packed = P.pack_int4(codes)
    return packed, scale, lo


def _relerr(a, b):
    return (a - b).norm() / b.norm().clamp_min(1e-9)


@pytest.mark.parametrize("nq,nk", [(64, 1024), (96, 4096), (1, 2048), (128, 257)])
def test_int4_logits_matches_reference(nq, nk):
    torch.manual_seed(0)
    D = 64
    Rq = torch.randn(nq, D, device="cuda", dtype=torch.float32)
    Rk = torch.randn(nk, D, device="cuda", dtype=torch.float32)
    packed, scale, lo = _quantize_rows(Rk)
    out = int4_logits(Rq, packed, scale, lo)
    ref = int4_logits_reference(Rq, packed, scale, lo, D)
    assert out.shape == (nq, nk)
    assert _relerr(out, ref) < 1e-3


@pytest.mark.parametrize("nq,nk", [(64, 1024), (96, 4096), (1, 2048), (128, 257)])
def test_int4_values_matches_reference(nq, nk):
    torch.manual_seed(1)
    D = 64
    attn = torch.softmax(torch.randn(nq, nk, device="cuda", dtype=torch.float32), dim=-1)
    Rv = torch.randn(nk, D, device="cuda", dtype=torch.float32)
    packed, scale, lo = _quantize_rows(Rv)
    out = int4_values_rot(attn, packed, scale, lo, D)
    ref = int4_values_reference(attn, packed, scale, lo, D)
    assert out.shape == (nq, D)
    assert _relerr(out, ref) < 1e-3


def test_logits_then_values_roundtrip_attention():
    # End-to-end fused attention in the rotated space vs the dequant reference.
    torch.manual_seed(2)
    nq, nk, D = 64, 4096, 64
    Rq = torch.randn(nq, D, device="cuda", dtype=torch.float32)
    Rk = torch.randn(nk, D, device="cuda", dtype=torch.float32)
    Rv = torch.randn(nk, D, device="cuda", dtype=torch.float32)
    pK, sK, lK = _quantize_rows(Rk)
    pV, sV, lV = _quantize_rows(Rv)
    scale = 1.0 / (D ** 0.5)

    logits = int4_logits(Rq, pK, sK, lK) * scale
    attn = torch.softmax(logits, dim=-1)
    o = int4_values_rot(attn, pV, sV, lV, D)

    ref_logits = int4_logits_reference(Rq, pK, sK, lK, D) * scale
    ref_attn = torch.softmax(ref_logits, dim=-1)
    ref_o = int4_values_reference(ref_attn, pV, sV, lV, D)
    assert _relerr(o, ref_o) < 2e-3
