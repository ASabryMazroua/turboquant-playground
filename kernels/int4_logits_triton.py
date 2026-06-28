"""Fused int4 attention-logits Triton kernel (PLAN §M5).

Computes ``logits[i, j] = Rq[i] · dequant(packed_Kj)`` **without** ever
materializing the dequantized ``[nk, D]`` key matrix — the int4 nibbles are
unpacked inside the kernel and consumed directly, so HBM traffic is the packed
0.5-byte/value store (+ per-token scale/lo) instead of the 2-byte/value BF16
reconstruction the M4 cache builds.

Per-token affine dequant is ``k̂[d] = code[d]·scale + lo``, so

    Rq·k̂ = scale·(Rq·code) + lo·Σ_d Rq[d].

With the M3 packing (even dims → low nibble, odd dims → high nibble),
``Rq·code = Rq_even·lo_nib + Rq_odd·hi_nib``.

NOTE: ``@triton.jit`` resolves global names (``tl`` …) from this module's
namespace, so triton is imported at module level. These kernels are only
imported on a GPU/Triton host (tests ``importorskip('triton')`` first).
"""
from __future__ import annotations

import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 64


@triton.jit
def _int4_logits_kernel(
    qe_ptr, qo_ptr, qrow_ptr,          # [nq, H] even/odd, [nq] row-sum (f32)
    packed_ptr, scale_ptr, lo_ptr,     # [nk, H] uint8, [nk] f32, [nk] f32
    out_ptr,                           # [nq, nk] f32
    nq, nk,
    sqm, sqh, spk, sph, som, son,
    H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_h = tl.arange(0, H)
    m_mask = offs_m < nq
    n_mask = offs_n < nk

    qe = tl.load(qe_ptr + offs_m[:, None] * sqm + offs_h[None, :] * sqh,
                 mask=m_mask[:, None], other=0.0)
    qo = tl.load(qo_ptr + offs_m[:, None] * sqm + offs_h[None, :] * sqh,
                 mask=m_mask[:, None], other=0.0)
    pk = tl.load(packed_ptr + offs_n[:, None] * spk + offs_h[None, :] * sph,
                 mask=n_mask[:, None], other=0)
    lo_nib = (pk & 0x0F).to(tl.float32)
    hi_nib = ((pk >> 4) & 0x0F).to(tl.float32)

    # [BM, H] · [H, BN] → [BM, BN]
    acc = tl.dot(qe, tl.trans(lo_nib), allow_tf32=False)
    acc += tl.dot(qo, tl.trans(hi_nib), allow_tf32=False)

    sc = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
    loj = tl.load(lo_ptr + offs_n, mask=n_mask, other=0.0)
    qrow = tl.load(qrow_ptr + offs_m, mask=m_mask, other=0.0)
    out = sc[None, :] * acc + loj[None, :] * qrow[:, None]

    out_ptrs = out_ptr + offs_m[:, None] * som + offs_n[None, :] * son
    tl.store(out_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])


def int4_logits(Rq, packed_K, scale_K, lo_K):
    """Fused int4 logits ``[nq, nk]`` = ``Rq @ dequant(packed_K).T``.

    ``Rq`` ``[nq, D]`` f32; ``packed_K`` ``[nk, D//2]`` uint8; ``scale_K``/``lo_K``
    ``[nk]`` f32. Never allocates the ``[nk, D]`` dequantized keys.
    """
    import torch

    nq, D = Rq.shape
    nk = packed_K.shape[0]
    H = D // 2
    Rq = Rq.contiguous()
    qe = Rq[:, 0::2].contiguous()
    qo = Rq[:, 1::2].contiguous()
    qrow = Rq.sum(-1).contiguous()
    scale_K = scale_K.reshape(-1).contiguous().to(torch.float32)
    lo_K = lo_K.reshape(-1).contiguous().to(torch.float32)
    out = torch.empty((nq, nk), device=Rq.device, dtype=torch.float32)

    grid = (triton.cdiv(nq, BLOCK_M), triton.cdiv(nk, BLOCK_N))
    _int4_logits_kernel[grid](
        qe, qo, qrow, packed_K, scale_K, lo_K, out,
        nq, nk,
        qe.stride(0), qe.stride(1), packed_K.stride(0), packed_K.stride(1),
        out.stride(0), out.stride(1),
        H=H, BM=BLOCK_M, BN=BLOCK_N,
    )
    return out


def int4_logits_reference(Rq, packed_K, scale_K, lo_K, D):
    """Reference: dequantize the full ``[nk, D]`` keys, then ``Rq @ Kĥ.T``."""
    from turbo_kv import packing as P

    codes = P.unpack_int4(packed_K, D).to(Rq.dtype)
    Rk_hat = codes * scale_K.reshape(-1, 1) + lo_K.reshape(-1, 1)
    return Rq @ Rk_hat.t()
