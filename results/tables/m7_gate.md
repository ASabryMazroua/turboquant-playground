# M7 gate — per-channel key quantization (the redemption)

- gate: per-channel key int4 ppl_ratio < 1.2× on WikiText at every context

- ctx=4096: per-token 15.6×  →  per-channel **1.017×**  (15× better)
- ctx=8192: per-token 47.3×  →  per-channel **1.034×**  (46× better)
- ctx=16384: per-token 55.7×  →  per-channel **1.007×**  (55× better)

**Finding.** M4 reported that 4-bit KV is 15-57× worse perplexity on real WikiText. The root cause was our own design choice: **per-token key quantization**. Keys have persistent outlier *channels*, and a per-token scale lets one outlier channel inflate every token's range, crushing the rest. Switching keys to **per-channel** quantization (KIVI's core fix) — the way KIVI/KVQuant/TurboQuant all do it — recovers near-lossless quality (ppl_ratio ≈ 1.01–1.03×), a 15–55× improvement from a single principled change, even though we quantize *post*-RoPE (pre-RoPE would help further). The earlier negative result was real and correctly measured — it was a demonstration of *why* the field quantizes keys per-channel.

## Verdict: PASS  (per-channel near-lossless=True)
