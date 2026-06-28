"""M15 — tensor-core int4 logits microbenchmark (PLAN §M15).

For each ``(nq, nk)`` shape (D=64), compares three logits paths on latency
(``triton.testing.do_bench`` p50/p10/p90), correctness vs the fp32 dequant
reference, and peak memory:

* **bf16 cuBLAS** — plain ``Rq_bf @ Rk_bf.T`` (no quantization), the bar to beat;
* **M5 int4 (exact)** — :func:`kernels.int4_logits_triton.int4_logits`
  (``allow_tf32=False``, CUDA cores);
* **M15 int4 (tensor-core)** — :func:`kernels.int4_logits_tc_triton.int4_logits_tc`
  (bf16 tile dequant in SRAM, ``allow_tf32=True``, tensor cores).

The honest question M15 answers: does running the int4 GEMM on tensor cores
NARROW the gap to cuBLAS that M5 left open — and where (nq≥64 prefill/scoring vs
nq=1 decode, where the tensor-core M tile is starved)?

Writes ``turbo_kernels_tc.csv``. GPU-only::

    python benchmarks/benchmark_attention_micro_tc.py --out-dir outputs
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbo_kv import metrics, packing, reporting  # noqa: E402
from kernels.int4_logits_triton import int4_logits, int4_logits_reference  # noqa: E402
from kernels.int4_logits_tc_triton import int4_logits_tc  # noqa: E402


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


def bench_shapes(nq_list, nk_list, D, csv):
    import torch

    for nk in nk_list:
        for nq in nq_list:
            Rq = torch.randn(nq, D, device="cuda", dtype=torch.float32)
            Rk = torch.randn(nk, D, device="cuda", dtype=torch.float32)
            pK, sK, lK = _quantize_rows(Rk)

            ref = int4_logits_reference(Rq, pK, sK, lK, D)
            relerr_m5 = _relerr(int4_logits(Rq, pK, sK, lK), ref)
            relerr_tc = _relerr(int4_logits_tc(Rq, pK, sK, lK), ref)

            m5_50, m5_10, m5_90 = _bench(lambda: int4_logits(Rq, pK, sK, lK))
            tc_50, tc_10, tc_90 = _bench(lambda: int4_logits_tc(Rq, pK, sK, lK))
            Rq_bf, Rk_bf = Rq.to(torch.bfloat16), Rk.to(torch.bfloat16)
            bf_50, _, _ = _bench(lambda: Rq_bf @ Rk_bf.t())

            tc_peak, _ = _peak(lambda: int4_logits_tc(Rq, pK, sK, lK))
            m5_peak, _ = _peak(lambda: int4_logits(Rq, pK, sK, lK))
            ref_peak, _ = _peak(lambda: int4_logits_reference(Rq, pK, sK, lK, D))
            # fused HBM traffic: packed K + scale/lo + Rq + out
            tc_bytes = nk * (D // 2) + nk * 8 + nq * D * 4 + nq * nk * 4

            reporting.append_row(csv, dict(
                op="logits", nq=nq, nk=nk,
                relerr_tc=round(relerr_tc, 6), relerr_m5=round(relerr_m5, 6),
                tc_ms=round(tc_50, 4), tc_p10=round(tc_10, 4), tc_p90=round(tc_90, 4),
                m5_ms=round(m5_50, 4), bf16_ms=round(bf_50, 4),
                speedup_vs_m5=round(m5_50 / tc_50, 3),
                speedup_vs_bf16=round(bf_50 / tc_50, 3),
                tc_peak_mb=round(tc_peak, 3), m5_peak_mb=round(m5_peak, 3),
                ref_peak_mb=round(ref_peak, 3),
                tc_gbs=round(tc_bytes / 1e9 / (tc_50 / 1e3), 1)))
            print(f"[m15] nq={nq} nk={nk}: tc {tc_50:.4f} m5 {m5_50:.4f} bf16 {bf_50:.4f} ms | "
                  f"tc/m5 {m5_50 / tc_50:.2f}x  tc/bf16 {bf_50 / tc_50:.2f}x | "
                  f"relerr tc {relerr_tc:.2e} m5 {relerr_m5:.2e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nq", default="1,16,64,256")
    ap.add_argument("--nk", default="1024,4096,16384")
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    import torch

    print("device:", torch.cuda.get_device_name(0))
    os.makedirs(args.out_dir, exist_ok=True)
    csv = os.path.join(args.out_dir, "turbo_kernels_tc.csv")

    nq_list = [int(x) for x in args.nq.split(",")]
    nk_list = [int(x) for x in args.nk.split(",")]
    bench_shapes(nq_list, nk_list, args.head_dim, csv)
    print(f"\nwrote {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
