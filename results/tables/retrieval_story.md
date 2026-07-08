# Retrieval chapter — the estimator that killed attention *wins* here

**Setup.** Same QJL unbiased inner-product sketch from Finding 7 (M16), now in retrieval: score all vectors cheaply, keep a top-100 candidate pool, then **exact-rerank** those 100. recall@10 vs an exact `IndexFlatIP` ground truth, across 5 datasets, on quality · latency · RAM.

## 1. QJL beats SimHash by keeping the norm (and I can prove it)
Same `m=1024` sign bits; QJL decodes the *inner product* (using ‖x‖), SimHash decodes *Hamming* (angle only). recall@10 (reranked):

| dataset | QJL | SimHash | QJL advantage |
| --- | ---: | ---: | ---: |
| aniso (outlier dims) | 0.999 | 0.721 | 1.39× |
| iso (Gaussian control) | 0.890 | 0.686 | 1.30× |
| blobs (clustered) | 1.000 | 0.178 | 5.62× |
| unit (cosine) | 0.998 | 0.989 | 1.01× |
| mnist (real, 784-d) | 0.991 | 0.209 | 4.74× |

On max-inner-product data the norm decides the winner, so QJL wins big — but on **unit** vectors (every ‖x‖ = 1) the two **converge** (0.998 vs 0.989): remove the norm and QJL's edge vanishes. That is a clean controlled proof of the mechanism.

## 2. Rerank absorbs the variance that killed attention (Finding 7, vindicated)
QJL's *raw* scores are noisy (the same high variance from M16), but the shortlist-then-verify structure recovers near-exact recall:

| dataset | QJL m=1024 raw | after rerank |
| --- | ---: | ---: |
| aniso (outlier dims) | 0.712 | 0.999 |
| iso (Gaussian control) | 0.414 | 0.890 |
| blobs (clustered) | 0.707 | 1.000 |
| unit (cosine) | 0.667 | 0.998 |
| mnist (real, 784-d) | 0.657 | 0.991 |

The one-shot softmax in attention had no second chance; retrieval's exact rerank over a wide candidate pool is exactly that second chance.

## 3. Rotate-before-quantize (OPQ) helps *structured* data, not isotropic (the KV lesson)
PQ vs OPQ+PQ (a learned rotation before PQ), recall@10 reranked at 16 B/vec:

| dataset | PQ | OPQ+PQ | rotation effect |
| --- | ---: | ---: | ---: |
| aniso (outlier dims) | 0.843 | 0.996 | +0.153 |
| iso (Gaussian control) | 0.546 | 0.545 | -0.001 |
| blobs (clustered) | 0.334 | 0.342 | +0.008 |
| unit (cosine) | 0.921 | 0.998 | +0.076 |
| mnist (real, 784-d) | 1.000 | 0.999 | -0.001 |

Rotation clearly helps where PQ has headroom on anisotropic data (aniso, unit) and does essentially **nothing on the isotropic control** (`iso`, -0.001) — the same result as the KV cache (Findings 1 & 7): a rotation spreads outlier energy, and isotropic data has none to spread. (Where PQ already saturates — e.g. `mnist` at 1.000 — there is simply no room left to show the effect, though the rotation still lifts SQ4's raw recall there 0.74 → 0.91.)

## Verdict
The bias–variance lesson is symmetric. Attention *amplifies* variance (one-shot weighted blend of thousands of keys) → QJL was catastrophic. Retrieval *absorbs* it (shortlist + exact rerank) → the very same QJL sketch is near-lossless at a fraction of the bytes. **Unbiased-but-noisy is poison for a one-shot softmax and perfect for shortlist-then-verify.**
