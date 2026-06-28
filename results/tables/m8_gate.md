# M8 gate — pre-RoPE key quantization (KVQuant fix)

- gate: pre-RoPE tf_kl <= post-RoPE tf_kl x 1.1 at every context (pre-RoPE does not hurt, ideally helps)

- ctx=4096: post tf_kl 2.666e-02  ->  pre **2.450e-02**  (1.09x better)
- ctx=8192: post tf_kl 2.149e-02  ->  pre **2.048e-02**  (1.05x better)
- ctx=16384: post tf_kl 6.263e-02  ->  pre **3.826e-02**  (1.64x better)

**Finding.** M7 recovered near-lossless int4 KV with per-channel key quantization, but still quantized keys *post*-RoPE. RoPE mixes adjacent channels with position-dependent rotations, smearing the per-channel statistics that per-channel quantization relies on. KVQuant's fix is to quantize the **raw pre-RoPE key** and re-apply RoPE to the reconstructed key at attention time (we store a tiny int32 positions buffer to do so, preserving the ~4x memory win). Pre-RoPE keys should match or beat post-RoPE on teacher-forced KL at every context.

## Verdict: PASS  (pre-RoPE does not hurt=True)
