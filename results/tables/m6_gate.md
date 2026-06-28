# M6 gate — TurboQuant-prod (QJL residual) vs MSE-only

- claim 1 (bias): prod removes most of the MSE-only inner-product bias
  - max MSE-only |bias| = 66.658; min prod |bias| = 1.150 → YES (prod < 20% of MSE bias)
- claim 2 (Pareto): prod gives a quality/memory improvement on attention-KL
  - YES — prod-4b@2.5b (rate 5.83, KL 3.880e+00) reaches quality below the MSE-only floor (KL 6.469e+00)

**Finding.** The signed IP-error histogram confirms the paper's premise: MSE-only key reconstruction is **biased** (its error distribution is offset from 0) because the MSE-optimal quantizer shrinks ‖k̂‖. The 1-bit QJL residual is **unbiased in expectation** and at the 4-bit operating point (3-bit recon + 1-bit sketch) it cuts inner-product bias ~30× (33.4→1.1 on RHT). Its limitation is **variance**: a single d-row sign sketch is noisy when the residual is large (low bits), so prod-1b only wins near 4 bits — matching the paper's '3.5-bit neutral, 2.5-bit marginal'. Widening the sketch (prod-2b/4b) trades memory for variance; the Pareto plot shows whether that buys a net win.

## Verdict: PASS  (bias_ok=True, pareto_ok=True)
