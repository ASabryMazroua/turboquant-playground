"""Fused int4 attention-values Triton kernel (PLAN §M5).

Computes the rotated attention output ``o_rot[i] = Σ_j a[i, j] · dequant(packed_Vj)``
**without** materializing the dequantized ``[nk, D]`` value matrix. With per-token
affine dequant ``v̂[d] = code[d]·scale + lo``:

    o_rot[i, d] = Σ_j (a[i,j]·scale_j)·code_j[d] + (Σ_j a[i,j]·lo_j)

so with ``b = a·scale`` and the even/odd nibble split,
``o_even = b @ lo_nib + arowlo`` and ``o_odd = b @ hi_nib + arowlo``. The wrapper
interleaves the two halves back to ``[nq, D]``; the caller applies the single
inverse rotation ``o = Rᵀ o_rot`` (per head, the M4 structure).

triton is imported at module level so ``@triton.jit`` can resolve ``tl``.
"""
from __future__ import annotations

import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 64


@triton.jit
def _int4_values_kernel(
    a_ptr,                              # [nq, nk] f32 attention weights
    packed_ptr, scale_ptr, lo_ptr,      # [nk, H] uint8, [nk] f32, [nk] f32
    oe_ptr, oo_ptr,                     # [nq, H] f32 even/odd outputs
    nq, nk,
    sam, san, spk, sph, som, soh,
    H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_h = tl.arange(0, H)
    m_mask = offs_m < nq

    acc_e = tl.zeros((BM, H), dtype=tl.float32)
    acc_o = tl.zeros((BM, H), dtype=tl.float32)
    acc_lo = tl.zeros((BM,), dtype=tl.float32)

    for n0 in range(0, nk, BN):
        offs_n = n0 + tl.arange(0, BN)
        n_mask = offs_n < nk
        a_blk = tl.load(a_ptr + offs_m[:, None] * sam + offs_n[None, :] * san,
                        mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        sc = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
        loj = tl.load(lo_ptr + offs_n, mask=n_mask, other=0.0)
        b = a_blk * sc[None, :]                                   # [BM, BN]

        pk = tl.load(packed_ptr + offs_n[:, None] * spk + offs_h[None, :] * sph,
                     mask=n_mask[:, None], other=0)
        lo_nib = (pk & 0x0F).to(tl.float32)                       # [BN, H]
        hi_nib = ((pk >> 4) & 0x0F).to(tl.float32)

        acc_e += tl.dot(b, lo_nib, allow_tf32=False)              # [BM, H]
        acc_o += tl.dot(b, hi_nib, allow_tf32=False)
        acc_lo += tl.sum(a_blk * loj[None, :], axis=1)            # [BM]

    oe = acc_e + acc_lo[:, None]
    oo = acc_o + acc_lo[:, None]
    tl.store(oe_ptr + offs_m[:, None] * som + offs_h[None, :] * soh, oe, mask=m_mask[:, None])
    tl.store(oo_ptr + offs_m[:, None] * som + offs_h[None, :] * soh, oo, mask=m_mask[:, None])


def int4_values_rot(attn, packed_V, scale_V, lo_V, D):
    """Fused int4 rotated output ``o_rot`` ``[nq, D]`` = ``attn @ dequant(packed_V)``.

    ``attn`` ``[nq, nk]`` f32; ``packed_V`` ``[nk, D//2]`` uint8; ``scale_V``/``lo_V``
    ``[nk]`` f32. Never allocates the ``[nk, D]`` dequantized values.
    """
    import torch

    nq, nk = attn.shape
    H = D // 2
    attn = attn.contiguous()
    scale_V = scale_V.reshape(-1).contiguous().to(torch.float32)
    lo_V = lo_V.reshape(-1).contiguous().to(torch.float32)
    oe = torch.empty((nq, H), device=attn.device, dtype=torch.float32)
    oo = torch.empty((nq, H), device=attn.device, dtype=torch.float32)

    grid = (triton.cdiv(nq, BLOCK_M),)
    _int4_values_kernel[grid](
        attn, packed_V, scale_V, lo_V, oe, oo,
        nq, nk,
        attn.stride(0), attn.stride(1), packed_V.stride(0), packed_V.stride(1),
        oe.stride(0), oe.stride(1),
        H=H, BM=BLOCK_M, BN=BLOCK_N,
    )
    o_rot = torch.empty((nq, D), device=attn.device, dtype=torch.float32)
    o_rot[:, 0::2] = oe
    o_rot[:, 1::2] = oo
    return o_rot


def int4_values_reference(attn, packed_V, scale_V, lo_V, D):
    """Reference: dequantize the full ``[nk, D]`` values, then ``attn @ V̂``."""
    from turbo_kv import packing as P

    codes = P.unpack_int4(packed_V, D).to(attn.dtype)
    Rv_hat = codes * scale_V.reshape(-1, 1) + lo_V.reshape(-1, 1)
    return attn @ Rv_hat
