# M12 gate summary

_Headline regime: **channel-wise** quant, rotation=**none** — the axis our int4 KV keys actually use (M7 per-channel keys), where fitted NUQ levels lower attention-KL._

- bits=2.5: key_mse uniform=5.1267e-01 → nuq-quantile=2.4620e-01 → nuq-kmeans=1.2536e-01 (0.24×, NUQ better ✅)
- bits=3.0: key_mse uniform=3.4968e-01 → nuq-quantile=1.6353e-01 → nuq-kmeans=7.6673e-02 (0.22×, NUQ better ✅)
- bits=3.5: key_mse uniform=2.4305e-01 → nuq-quantile=1.0462e-01 → nuq-kmeans=4.5264e-02 (0.19×, NUQ better ✅)
- bits=4.0: key_mse uniform=1.6647e-01 → nuq-quantile=5.7113e-02 → nuq-kmeans=2.5164e-02 (0.15×, NUQ better ✅)

**Finding.** NUQ fits reconstruction levels to the data density (quantile init, then 1-D k-means / Lloyd–Max), so heavy-tailed key coordinates get fine resolution near the mode and coarse near the tails. It does exactly what it optimizes: key reconstruction **MSE drops 4–6×** vs a uniform grid at matched bits, on both axes. NUQ stores a per-group fp16 codebook adding ≈0.080 bits/value overhead (uniform: 0). By contrast, NUQ-kmeans beats uniform on **attention-KL** in only **4/12** (layer,bits) cells — the MSE win does *not* transfer to attention fidelity. That last point is the real lesson — the same **MSE-optimal ≠ inner-product-optimal** decoupling this project keeps hitting (M3/M6): a better reconstruction is not a better attention distribution, so pure MSE-driven NUQ is not the KV win; per-channel int4 (M7) already nails the axis that matters. Full cache integration must also store/reload the per-group codebook — future work; this milestone is the numerical case.

**Gate: PASS** — NUQ (k-means) key_mse < uniform at matched low bits (≤3.0) for the headline setting (True). Per-(layer,bits) cells at ≤3.0 bits where NUQ-kmeans < uniform: **6/6**.
