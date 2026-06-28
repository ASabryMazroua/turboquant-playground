# M4 gate — patched Qwen2 attention (rotate-query int4 KV)

- thresholds: tf_kl<0.5, argmax-match>0.9, ppl_ratio<1.2 (RHT, sink=4)

- ctx=4096 dense sink=0: tf_kl=2.462e+00 match=0.461 ppl_ratio=10.966 decode 73.16ms/tok (bf16 24.25) peak 1116MB (bf16 1149) -> INCOHERENT
- ctx=4096 dense sink=4: tf_kl=1.952e+00 match=0.531 ppl_ratio=6.932 decode 73.29ms/tok (bf16 24.25) peak 1164MB (bf16 1149) -> INCOHERENT
- ctx=4096 none sink=0: tf_kl=4.115e+00 match=0.270 ppl_ratio=58.671 decode 68.33ms/tok (bf16 24.25) peak 1116MB (bf16 1149) -> INCOHERENT
- ctx=4096 none sink=4: tf_kl=2.875e+00 match=0.434 ppl_ratio=17.817 decode 68.40ms/tok (bf16 24.25) peak 1164MB (bf16 1149) -> INCOHERENT
- ctx=4096 rht sink=0: tf_kl=1.764e+00 match=0.566 ppl_ratio=5.948 decode 152.28ms/tok (bf16 24.25) peak 1116MB (bf16 1149) -> INCOHERENT
- ctx=4096 rht sink=4: tf_kl=1.799e+00 match=0.566 ppl_ratio=6.176 decode 195.46ms/tok (bf16 24.25) peak 1164MB (bf16 1149) -> INCOHERENT
- ctx=8192 dense sink=0: tf_kl=2.027e+00 match=0.559 ppl_ratio=7.195 decode 73.04ms/tok (bf16 30.51) peak 1272MB (bf16 1341) -> INCOHERENT
- ctx=8192 dense sink=4: tf_kl=1.824e+00 match=0.570 ppl_ratio=6.008 decode 73.62ms/tok (bf16 30.51) peak 1369MB (bf16 1341) -> INCOHERENT
- ctx=8192 none sink=0: tf_kl=1.473e+00 match=0.606 ppl_ratio=4.410 decode 68.50ms/tok (bf16 30.51) peak 1272MB (bf16 1341) -> INCOHERENT
- ctx=8192 none sink=4: tf_kl=1.606e+00 match=0.590 ppl_ratio=4.978 decode 68.47ms/tok (bf16 30.51) peak 1369MB (bf16 1341) -> INCOHERENT
- ctx=8192 rht sink=0: tf_kl=7.220e+00 match=0.062 ppl_ratio=1388.526 decode 196.50ms/tok (bf16 30.51) peak 1272MB (bf16 1341) -> INCOHERENT
- ctx=8192 rht sink=4: tf_kl=7.240e+00 match=0.051 ppl_ratio=1414.744 decode 195.81ms/tok (bf16 30.51) peak 1369MB (bf16 1341) -> INCOHERENT
- ctx=16384 dense sink=0: tf_kl=5.204e+00 match=0.227 ppl_ratio=188.117 decode 73.19ms/tok (bf16 30.67) peak 1586MB (bf16 1723) -> INCOHERENT
- ctx=16384 dense sink=4: tf_kl=5.144e+00 match=0.231 ppl_ratio=175.309 decode 73.51ms/tok (bf16 30.67) peak 1778MB (bf16 1723) -> INCOHERENT
- ctx=16384 none sink=0: tf_kl=5.071e+00 match=0.191 ppl_ratio=166.133 decode 68.62ms/tok (bf16 30.67) peak 1586MB (bf16 1723) -> INCOHERENT
- ctx=16384 none sink=4: tf_kl=4.413e+00 match=0.273 ppl_ratio=84.997 decode 68.59ms/tok (bf16 30.67) peak 1778MB (bf16 1723) -> INCOHERENT
- ctx=16384 rht sink=0: tf_kl=5.453e+00 match=0.156 ppl_ratio=259.280 decode 196.35ms/tok (bf16 30.67) peak 1586MB (bf16 1723) -> INCOHERENT
- ctx=16384 rht sink=4: tf_kl=5.329e+00 match=0.168 ppl_ratio=225.762 decode 196.71ms/tok (bf16 30.67) peak 1778MB (bf16 1723) -> INCOHERENT

**Finding.** The rotate-query *patch* is correct (pytest 74/74; the per-layer probe with a non-quantized cache shows `none` bit-exact and `rht`/`dense` only a stable ~3.5% bf16 rotated-basis floor). But the held-out **novel-text** eval (valid reference, ppl_bf16≈6.5) shows the 4-bit `TurboKVCache` is **pervasively lossy**: median next-token KL ~1–6 and ppl inflated 4–260×. This overturns M3's 'near-lossless' result, which was an artifact of the repeated-text eval — induction/copy heads hide KV corruption on repeated text but not on novel text. Rotation helps at ctx=4096 (rht ppl_ratio 5.9 < dense 11.0 < none 58.7, reproducing the M2 per-token result) but the relationship **inverts at long context**: at 8k–16k rht degrades badly (rht spikes to ppl_ratio 1389× at 8k while none is 4.4×) — rotation spreads each logit into a sum of many noisy int4 terms, so over thousands of keys a spurious maximum can dominate the softmax. BF16 **attention-sink** preservation (sink=4) was tested and gives only minor relief (none 58.7→17.8× at 4k, 166→85× at 16k; rht unchanged) — so the sink is **not** the primary cause. Conclusion: per-token int4 KV is insufficient for novel-text fidelity; the principled fix is the **QJL +1-bit residual (M6)** — the paper's unbiased-inner-product correction, since 4-bit MSE quant is biased — and/or per-channel key quantization (KIVI).

## Verdict: FAIL (quality) — patch validated, int4-only insufficient
