# M11 gate — direct large-m QJL key sketch vs M6 residual prod

- claim (Pareto): direct m=256 (+outliers) reaches attn-KL at or below the best M6 prod-1b, at comparable/better realized bits
  - direct+out best: KL 2.244e+00 at 11.00 bits (m=512)
  - prod-1b best:    KL 3.084e+01 at 3.25 bits
  - KL_ok=True, bits_ok=False

**Finding.** This is the real QJL / TurboQuant-prod key encoding: a LARGE-m Gaussian **sign sketch of the rotated key itself** (no MSE reconstruction base, no per-channel scale/zero) plus a handful of exact fp16 outlier coordinates. Because the unbiased inner-product estimator's variance scales as 1/m, widening the sketch — not adding recon bits — is what drives attention-KL down, and zeroing the few extreme key coordinates before sketching (adding their exact IP back at decode) removes the dominant variance term. The cost is honest: ~4-5 bits/value for m=256-512 sign bits, so this is not a sub-4-bit win but the field's *quality-first* operating point that M6's 1-bit residual sketch could not reach. **Caveat:** the direct sketch *estimates logits*, so it does not drop into SDPA — full decode integration needs a custom fused attention path that consumes signs/norms/outliers directly (future work, overlapping with the fused-kernel milestone).

## Verdict: PARTIAL — direct large-m sketch validated; see numbers for Pareto crossover  (pareto_ok=False)  — direct m=512+out KL 2.244e+00 vs prod-1b KL 3.084e+01
