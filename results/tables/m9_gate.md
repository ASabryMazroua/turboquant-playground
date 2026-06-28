# M9 gate — per-layer bit allocation (QJL more-bits-for-early-layers)

- gate: for **per_token** keys (the hard setting), increasing bf16_layers gives non-increasing tf_kl at the largest context (keeping sensitive early layers BF16 helps)

- ctx=16384 per_token: bf16L=0: 4.136e+00  ->  bf16L=1: 3.581e-01  ->  bf16L=2: 2.197e-01
- per_token tf_kl reduced 18.82x from bf16_layers=0 to 2
- ctx=16384 per_channel (already near-lossless): tf_kl 5.553e-02..6.779e-02 — little to gain
- memory cost: peak 1586MB -> 1598MB (+11MB) for bf16_layers 0->2

**Finding.** Layers are not equally quantization-sensitive: M2 found layer 0 a huge inner-product RMSE outlier (~1565 vs ~250 elsewhere), and QJL spends more bits on early layers. The clean int4-only analog is to keep the first ``bf16_layers`` layers' KV entirely in BF16 and int4 the rest. The hard **per_token** key setting is rescued by early-layer BF16 (tf_kl falls as bf16_layers grows), confirming the sensitivity is concentrated early; **per_channel** keys are already near-lossless so they gain little. The cost is real and measured — the BF16 window inflates peak memory — so the interesting result is the per_token rescue traded against memory, visible in m9_quality_memory_tradeoff.

## Verdict: PASS  (per_token tf_kl non-increasing in bf16_layers=True)
