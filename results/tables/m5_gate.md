# M5 gate — fused int4 attention kernels

- correctness: all relerr < 0.001 → YES (max 3.22e-04)
- no full dequant materialized: fused peak < reference peak → YES
- speedup range (reference/fused): 0.04× (values nk=16384) … 0.61× (logits nk=8192)
- decode step (nq=1): fused vs M4 dequant speedup 0.12×–0.69× over nk=1024–16384
- decode step (nq=1): fused vs **plain BF16 (no quant)** speedup 0.02×–0.08× (fused is 12.2–47.6× slower)

**Finding.** The fused kernels unpack int4 nibbles in-register and compute logits / value-sums directly from the 0.5-byte/value packed store, so they **never materialize the dequantized K/V** (peak memory confirms it). Correctness matches the dequant reference to < 1e-3. On **latency** the honest result is that the fused int4 kernel **loses to both** the M4 dequant→cuBLAS path **and** the original plain-BF16 attention: at head_dim=64 the workload is tiny and cuBLAS GEMM/GEMV (tensor cores) is hard to beat, while the Triton kernel runs `allow_tf32=False` for exactness and is launch/occupancy-bound (achieved GB/s ≪ A100 peak). The int4 fusion's value here is **memory** (4× smaller KV, no BF16 reconstruction), not decode latency — a real, honest systems result.

## Verdict: PASS  (correct=True, no_dequant=True)
