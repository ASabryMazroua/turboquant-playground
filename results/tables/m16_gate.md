# M16 gate — QJL wired into end-to-end generation

- gate: the best `qjl` key cache ppl_ratio < `per_token` int4 ppl_ratio at every context (the paper's unbiased inner-product correction must at least rescue the per-token disaster in real generation)

- ctx=4096: per-token 15.59×  ·  per-channel 1.017×  ·  **qjl 41.89×** (best none m=512,o=8, ≈11.0 key-bits/val)
- ctx=8192: per-token 47.26×  ·  per-channel 1.034×  ·  **qjl 77.43×** (best rht m=512,o=8, ≈11.0 key-bits/val)
- ctx=16384: per-token 55.71×  ·  per-channel 1.007×  ·  **qjl 36.76×** (best none m=512,o=8, ≈11.0 key-bits/val)

**Finding.** M6/M11 validated the QJL unbiased inner-product estimator only as a numeric study; M16 wires it into real Qwen2 generation via a custom attention path. Wired in, QJL is **catastrophically worse** than even per-token int4 (the M4 disaster): the estimator is unbiased but **high-variance**, and softmax attention over thousands of keys is exquisitely sensitive to per-logit variance — every noisy score is a chance to spuriously win the max, so attention scatters and perplexity explodes. The ablation is clean: at ctx=16384, going m=256→512 and adding 8 fp16 outliers cuts QJL from **483×** to **37×** (outliers alone: m=512 127×→37×, ~3×) — variance ~1/m and the outlier side-channel both help, exactly as the theory predicts, yet the best config (~11 bits/val, 2.7× int4's cost) is still ~36× worse than 4-bit per-channel. At ctx=16384 the QJL path peaks at **10813 MB** vs **1723 MB** for the BF16 baseline (6.3×): the custom attention that the sketch forces (no key reconstruction → no flash-attention) materialises the score matrix, forfeiting the memory win a KV cache exists for. Per-channel int4 (M7) remains the right end-to-end fix; QJL's home is the retrieval / no-scale regime, not dense KV attention at a small head_dim — extending the M11 PARTIAL verdict from a numeric to an end-to-end result.

## Verdict: FAIL  (best qjl beats per-token at every ctx=False)
