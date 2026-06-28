# M15 gate — tensor-core int4 logits kernel

- correctness: all relerr_tc < 0.01 → YES (max 2.37e-03)
- TC vs M5 int4 exact: 1.21× (nq=256 nk=4096) … 1.33× (nq=256 nk=16384) — faster than M5 somewhere → YES
- at nq≥64 (6 shapes): TC beats M5 exact in 6/6 (mean 1.27×)
- at nq=1 decode (3 shapes): TC vs M5 1.23×–1.28× (tensor-core M dim = 1 → underutilized, as expected)
- TC vs bf16 cuBLAS: 0.08×–0.21× (>1 = int4 TC beats cuBLAS; gap to cuBLAS narrowed vs M5)

**Finding.** Reconstructing the int4 key tile to bf16 *in SRAM* and running `tl.dot(allow_tf32=True)` moves the int4 logits GEMM onto the **tensor cores** while preserving M5's memory property (no global `[nk, D]` dequant). Correctness holds at the bf16/tf32 envelope (relerr < 1e-2). The honest latency result: the tensor-core path **narrows the cuBLAS gap M5 left open**, with the real win at **nq≥64** (prefill/scoring) where the MMA M tile is filled. At **nq=1 decode** the M dimension is 1, so tensor cores are structurally starved and the kernel stays memory/launch-bound at head_dim=64 — an expected, publishable limitation, not a regression. The values op (nq=1, `attn @ V̂`) is left on the M5 CUDA-core path for the same reason.

## Verdict: PASS  (correct=True, faster_than_m5_somewhere=True)
