"""M5 — fused int4 attention kernel microbenchmark (PLAN §M5).

For each ``(nq, nk)`` shape, compares the fused Triton int4 kernels against the
dequantize-then-matmul PyTorch reference on:

* **correctness** — relative Frobenius error (gate: < 1e-3);
* **latency** — ``triton.testing.do_bench`` median / p10 / p90 (ms);
* **memory** — peak allocator bytes, to verify the fused path never materializes
  the ``[nk, D]`` dequantized K/V (the reference does);
* **bandwidth** — effective GB/s of the fused kernel (HBM-bound at decode).

Also runs a fused-vs-reference *decode step* (logits → softmax → values) at
``nq=1`` to give the honest M4-dequant-vs-M5-fused decode latency comparison.

Writes ``turbo_kernels.csv`` and ``turbo_kernels_decode.csv``. GPU-only::

    python benchmarks/benchmark_attention_micro.py --out-dir outputs
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbo_kv import metrics, packing, reporting  # noqa: E402
from kernels.int4_logits_triton import int4_logits, int4_logits_reference  # noqa: E402
from kernels.int4_values_triton import int4_values_rot, int4_values_reference  # noqa: E402


def _quantize_rows(x):
    codes, scale, lo = packing.quantize_int4_per_token(x)
    return packing.pack_int4(codes), scale, lo


def _relerr(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-9))


def _bench(fn):
    import triton

    p50, p10, p90 = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.1, 0.9])
    return float(p50), float(p10), float(p90)


def _peak(fn):
    import torch

    torch.cuda.synchronize()
    metrics.reset_peak_memory()
    out = fn()
    torch.cuda.synchronize()
    return metrics.peak_allocated_mb(), out


def bench_shapes(nq_list, nk_list, D, kern_csv):
    import torch

    for nk in nk_list:
        for nq in nq_list:
            Rq = torch.randn(nq, D, device="cuda", dtype=torch.float32)
            Rk = torch.randn(nk, D, device="cuda", dtype=torch.float32)
            Rv = torch.randn(nk, D, device="cuda", dtype=torch.float32)
            pK, sK, lK = _quantize_rows(Rk)
            pV, sV, lV = _quantize_rows(Rv)
            attn = torch.softmax(torch.randn(nq, nk, device="cuda", dtype=torch.float32), dim=-1)

            # --- logits ---
            relerr_l = _relerr(int4_logits(Rq, pK, sK, lK),
                               int4_logits_reference(Rq, pK, sK, lK, D))
            f50, f10, f90 = _bench(lambda: int4_logits(Rq, pK, sK, lK))
            r50, _, _ = _bench(lambda: int4_logits_reference(Rq, pK, sK, lK, D))
            # plain BF16 attention logits (NO quantization) — the original baseline
            Rq_bf, Rk_bf = Rq.to(torch.bfloat16), Rk.to(torch.bfloat16)
            b50, _, _ = _bench(lambda: Rq_bf @ Rk_bf.t())
            fpeak, _ = _peak(lambda: int4_logits(Rq, pK, sK, lK))
            rpeak, _ = _peak(lambda: int4_logits_reference(Rq, pK, sK, lK, D))
            # fused HBM traffic: packed K + scale/lo + Rq + out
            fbytes = nk * (D // 2) + nk * 8 + nq * D * 4 + nq * nk * 4
            reporting.append_row(kern_csv, dict(
                op="logits", nq=nq, nk=nk, relerr=round(relerr_l, 6),
                fused_ms=round(f50, 4), fused_p10=round(f10, 4), fused_p90=round(f90, 4),
                ref_ms=round(r50, 4), bf16_ms=round(b50, 4),
                speedup=round(r50 / f50, 3), speedup_vs_bf16=round(b50 / f50, 3),
                fused_peak_mb=round(fpeak, 3), ref_peak_mb=round(rpeak, 3),
                fused_gbs=round(fbytes / 1e9 / (f50 / 1e3), 1)))

            # --- values ---
            relerr_v = _relerr(int4_values_rot(attn, pV, sV, lV, D),
                               int4_values_reference(attn, pV, sV, lV, D))
            f50, f10, f90 = _bench(lambda: int4_values_rot(attn, pV, sV, lV, D))
            r50, _, _ = _bench(lambda: int4_values_reference(attn, pV, sV, lV, D))
            attn_bf, Rv_bf = attn.to(torch.bfloat16), Rv.to(torch.bfloat16)
            b50, _, _ = _bench(lambda: attn_bf @ Rv_bf)
            fpeak, _ = _peak(lambda: int4_values_rot(attn, pV, sV, lV, D))
            rpeak, _ = _peak(lambda: int4_values_reference(attn, pV, sV, lV, D))
            fbytes = nk * (D // 2) + nk * 8 + nq * nk * 4 + nq * D * 4
            reporting.append_row(kern_csv, dict(
                op="values", nq=nq, nk=nk, relerr=round(relerr_v, 6),
                fused_ms=round(f50, 4), fused_p10=round(f10, 4), fused_p90=round(f90, 4),
                ref_ms=round(r50, 4), bf16_ms=round(b50, 4),
                speedup=round(r50 / f50, 3), speedup_vs_bf16=round(b50 / f50, 3),
                fused_peak_mb=round(fpeak, 3), ref_peak_mb=round(rpeak, 3),
                fused_gbs=round(fbytes / 1e9 / (f50 / 1e3), 1)))
            print(f"[m5] nq={nq} nk={nk}: logits relerr={relerr_l:.2e} "
                  f"values relerr={relerr_v:.2e}")


def bench_decode(nk_list, D, decode_csv):
    """Honest M4-dequant vs M5-fused decode step (nq=1): logits→softmax→values."""
    import torch

    inv = 1.0 / (D ** 0.5)
    for nk in nk_list:
        Rq = torch.randn(1, D, device="cuda", dtype=torch.float32)
        Rk = torch.randn(nk, D, device="cuda", dtype=torch.float32)
        Rv = torch.randn(nk, D, device="cuda", dtype=torch.float32)
        pK, sK, lK = _quantize_rows(Rk)
        pV, sV, lV = _quantize_rows(Rv)

        def fused():
            attn = torch.softmax(int4_logits(Rq, pK, sK, lK) * inv, dim=-1)
            return int4_values_rot(attn, pV, sV, lV, D)

        def reference():
            attn = torch.softmax(int4_logits_reference(Rq, pK, sK, lK, D) * inv, dim=-1)
            return int4_values_reference(attn, pV, sV, lV, D)

        # plain BF16 attention (NO quantization) — the original decode path
        Rq_bf, Rk_bf, Rv_bf = Rq.to(torch.bfloat16), Rk.to(torch.bfloat16), Rv.to(torch.bfloat16)

        def bf16():
            attn = torch.softmax((Rq_bf @ Rk_bf.t()).float() * inv, dim=-1).to(torch.bfloat16)
            return attn @ Rv_bf

        relerr = _relerr(fused(), reference())
        f50, f10, f90 = _bench(fused)
        r50, _, _ = _bench(reference)
        b50, _, _ = _bench(bf16)
        fpeak, _ = _peak(fused)
        rpeak, _ = _peak(reference)
        bpeak, _ = _peak(bf16)
        reporting.append_row(decode_csv, dict(
            nk=nk, relerr=round(relerr, 6),
            fused_ms=round(f50, 4), fused_p10=round(f10, 4), fused_p90=round(f90, 4),
            ref_ms=round(r50, 4), bf16_ms=round(b50, 4),
            speedup=round(r50 / f50, 3), speedup_vs_bf16=round(b50 / f50, 3),
            fused_peak_mb=round(fpeak, 3), ref_peak_mb=round(rpeak, 3),
            bf16_peak_mb=round(bpeak, 3)))
        print(f"[m5][decode] nk={nk}: fused {f50:.4f} ref {r50:.4f} bf16 {b50:.4f} ms | "
              f"fused/bf16 {f50 / b50:.2f}x slower | relerr {relerr:.2e} | "
              f"peak fused {fpeak:.1f} ref {rpeak:.1f} bf16 {bpeak:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nq", default="1,64")
    ap.add_argument("--nk", default="1024,4096,8192,16384")
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    import torch

    print("device:", torch.cuda.get_device_name(0))
    os.makedirs(args.out_dir, exist_ok=True)
    kern_csv = os.path.join(args.out_dir, "turbo_kernels.csv")
    decode_csv = os.path.join(args.out_dir, "turbo_kernels_decode.csv")

    nq_list = [int(x) for x in args.nq.split(",")]
    nk_list = [int(x) for x in args.nk.split(",")]
    bench_shapes(nq_list, nk_list, args.head_dim, kern_csv)
    bench_decode(nk_list, args.head_dim, decode_csv)
    print(f"\nwrote {kern_csv}\nwrote {decode_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
