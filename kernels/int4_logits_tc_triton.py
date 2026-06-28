"""Tensor-core int4 attention-logits Triton kernel (PLAN §M15).

M5's :mod:`kernels.int4_logits_triton` was *correct* but ran ``tl.dot(...,
allow_tf32=False)`` on fp32 operands, so the multiply executed on the CUDA cores
(no tensor cores) and lost to cuBLAS. This M15 variant keeps M5's algorithm and
its **memory** property — the int4 nibbles are unpacked and the key tile is
reconstructed only in registers/SRAM, never as a global ``[nk, D]`` bf16 matrix —
but reconstructs the key **TILE to bf16** and runs ``tl.dot(..., allow_tf32=True)``
so the GEMM lands on the **tensor cores** (bf16 inputs, fp32 accumulate).

Math (per-token affine dequant ``k̂[d] = code[d]·scale + lo``):

    logits[i, j] = Rq[i] · k̂[j] = Σ_d Rq[i,d]·(code_j[d]·scale_j + lo_j).

With the M3 packing (even dims → low nibble, odd dims → high nibble) the tile
is reconstructed as two ``[BN, H]`` bf16 half-tiles
``k_even = lo_nib·scale + lo`` / ``k_odd = hi_nib·scale + lo`` and contracted
against the matching ``[BM, H]`` bf16 query halves. Summing the two
``tl.dot``s reproduces ``Σ_d`` exactly (the affine ``lo`` term is folded into the
reconstructed tile, so no separate row-sum pass is needed):

    Σ_d Rq·k̂ = (Rq_even · k_even) + (Rq_odd · k_odd).

The contraction dim is ``H = D//2 = 32`` (a multiple of 16 → tensor-core legal);
``BM = BN = 64``. Because operands are bf16 and the affine is rounded into the
bf16 tile, the result matches the fp32 reference to ``< 1e-2`` (vs M5's 1e-3),
which is the expected tensor-core / bf16 accuracy envelope.

NOTE (the M5 lesson): ``@triton.jit`` resolves global names (``tl`` …) from this
module's namespace, so triton is imported at MODULE level and the kernel is
defined at MODULE level. This module is only imported on a GPU/Triton host (the
tests ``importorskip('triton')`` first; ``report_m15`` never imports it).
``python -m py_compile`` only *compiles* this file — it does not execute the
module-level ``import triton`` — so it stays compilable without triton installed,
exactly like the M5 kernel files.
"""
from __future__ import annotations

import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 64


@triton.jit
def _int4_logits_tc_kernel(
    qe_ptr, qo_ptr,                    # [nq, H] even/odd query halves (f32 in HBM)
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

    # Query halves → bf16 (tensor-core operand). [BM, H]
    qe = tl.load(qe_ptr + offs_m[:, None] * sqm + offs_h[None, :] * sqh,
                 mask=m_mask[:, None], other=0.0).to(tl.bfloat16)
    qo = tl.load(qo_ptr + offs_m[:, None] * sqm + offs_h[None, :] * sqh,
                 mask=m_mask[:, None], other=0.0).to(tl.bfloat16)

    # Unpack the int4 nibbles and reconstruct the key TILE to bf16 *in SRAM*
    # (never written back to global → preserves the no-full-dequant property).
    pk = tl.load(packed_ptr + offs_n[:, None] * spk + offs_h[None, :] * sph,
                 mask=n_mask[:, None], other=0)
    lo_nib = (pk & 0x0F).to(tl.float32)            # [BN, H] code in {0..15} (exact)
    hi_nib = ((pk >> 4) & 0x0F).to(tl.float32)
    sc = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)   # [BN]
    loj = tl.load(lo_ptr + offs_n, mask=n_mask, other=0.0)     # [BN]
    # k̂ = code·scale + lo, affine folded into the bf16 tile (per-token broadcast).
    ke = (lo_nib * sc[:, None] + loj[:, None]).to(tl.bfloat16)  # [BN, H]
    ko = (hi_nib * sc[:, None] + loj[:, None]).to(tl.bfloat16)

    # Tensor-core GEMM: [BM, H] · [H, BN] → [BM, BN], fp32 accumulate.
    acc = tl.dot(qe, tl.trans(ke), allow_tf32=True)
    acc += tl.dot(qo, tl.trans(ko), allow_tf32=True)

    out_ptrs = out_ptr + offs_m[:, None] * som + offs_n[None, :] * son
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def int4_logits_tc(Rq, packed_K, scale_K, lo_K):
    """Tensor-core fused int4 logits ``[nq, nk]`` = ``Rq @ dequant(packed_K).T``.

    Drop-in replacement for :func:`kernels.int4_logits_triton.int4_logits` that
    reconstructs each key tile to **bf16 in SRAM** and contracts on the tensor
    cores (``allow_tf32=True``). ``Rq`` ``[nq, D]`` f32; ``packed_K`` ``[nk, D//2]``
    uint8; ``scale_K``/``lo_K`` ``[nk]`` f32. Never allocates the ``[nk, D]``
    dequantized keys.
    """
    import torch

    nq, D = Rq.shape
    nk = packed_K.shape[0]
    H = D // 2
    Rq = Rq.contiguous()
    qe = Rq[:, 0::2].contiguous()
    qo = Rq[:, 1::2].contiguous()
    scale_K = scale_K.reshape(-1).contiguous().to(torch.float32)
    lo_K = lo_K.reshape(-1).contiguous().to(torch.float32)
    out = torch.empty((nq, nk), device=Rq.device, dtype=torch.float32)

    grid = (triton.cdiv(nq, BLOCK_M), triton.cdiv(nk, BLOCK_N))
    _int4_logits_tc_kernel[grid](
        qe, qo, packed_K, scale_K, lo_K, out,
        nq, nk,
        qe.stride(0), qe.stride(1), packed_K.stride(0), packed_K.stride(1),
        out.stride(0), out.stride(1),
        H=H, BM=BLOCK_M, BN=BLOCK_N,
    )
    return out


# NOTE — values kernel is intentionally NOT given a tensor-core variant here.
# The fused int4 *values* op (``attn @ V̂``) at decode is nq=1, so its GEMM M
# dimension is 1 and tensor cores (which need M≥8/16 to fill an MMA tile) are
# structurally underutilized; a bf16 TC values kernel would not beat the M5
# CUDA-core path there. We focus M15 on the *logits* op, where nq≥64 prefill /
# scoring shapes actually fill the tensor-core M tile. (See report_m15 gate.)
