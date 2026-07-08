"""Reference-equality tests for the tensor-core int4 logits kernel (M15).

Needs a GPU + Triton, so skipped on CPU-only machines (runs in the GPU job).
The tensor-core kernel reconstructs the key tile to bf16 and contracts with
``allow_tf32=True``, so it is checked against the M5 fp32 dequant reference with
a LOOSER tolerance (relerr < 1e-2) than the exact M5 kernel (1e-3).
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():  # pragma: no cover - CPU import guard
    pytest.skip("kernels require CUDA", allow_module_level=True)

from kernels.int4_logits_tc_triton import int4_logits_tc  # noqa: E402
from kernels.int4_logits_triton import int4_logits_reference  # noqa: E402
from turbo_kv import packing as P  # noqa: E402

TC_RELERR_TOL = 1e-2


def _quantize_rows(x):
    codes, scale, lo = P.quantize_int4_per_token(x)
    return P.pack_int4(codes), scale, lo


def _relerr(a, b):
    return (a - b).norm() / b.norm().clamp_min(1e-9)


@pytest.mark.parametrize("nq,nk", [(64, 1024), (96, 4096), (1, 2048), (128, 257)])
def test_int4_logits_tc_matches_reference(nq, nk):
    torch.manual_seed(0)
    D = 64
    Rq = torch.randn(nq, D, device="cuda", dtype=torch.float32)
    Rk = torch.randn(nk, D, device="cuda", dtype=torch.float32)
    packed, scale, lo = _quantize_rows(Rk)
    out = int4_logits_tc(Rq, packed, scale, lo)
    ref = int4_logits_reference(Rq, packed, scale, lo, D)
    assert out.shape == (nq, nk)
    assert _relerr(out, ref) < TC_RELERR_TOL


def test_int4_logits_tc_no_full_dequant_alloc():
    """Structural: the TC wrapper must not materialize an ``[nk, D]`` key matrix.

    The fused path's only large output is the ``[nq, nk]`` logits; peak allocation
    must stay well below the reference, which builds the dequantized ``[nk, D]``
    keys. We compare peak allocator bytes of the TC kernel vs the dequant reference.
    """
    torch.manual_seed(3)
    nq, nk, D = 64, 8192, 64
    Rq = torch.randn(nq, D, device="cuda", dtype=torch.float32)
    Rk = torch.randn(nk, D, device="cuda", dtype=torch.float32)
    packed, scale, lo = _quantize_rows(Rk)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _ = int4_logits_tc(Rq, packed, scale, lo)
    torch.cuda.synchronize()
    tc_peak = torch.cuda.max_memory_allocated()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _ = int4_logits_reference(Rq, packed, scale, lo, D)
    torch.cuda.synchronize()
    ref_peak = torch.cuda.max_memory_allocated()

    assert tc_peak < ref_peak
