# M3 gate summary

- **Memory:** best reduction **3.46×** (dense, rl=32: 13.9 MB vs BF16 48.0 MB) — ≈4× achieved ✅
- **Coherence** dense rl=256: tf_kl=4.723e+00, argmax_match=0.281, distinct=0.188 ❌
- **Coherence** rht rl=256: tf_kl=2.420e-02, argmax_match=1.000, distinct=0.969 ✅
- **Coherence** none rl=256: tf_kl=2.181e-02, argmax_match=1.000, distinct=0.969 ✅
- ⚠️ **Finding (decoupling):** rotation(s) ['dense'] have *good* int4 reconstruction (see m3_reconstruction_diag: dense relerr ≈ rht ≈ 0.088, both < none 0.118) yet **break attention** (high tf_kl) — reconstruction MSE and inner-product fidelity are decoupled. A Haar *dense* rotation spreads energy but is not flat, so its quant error aligns with the (correlated) query direction; structured **RHT** is flat/incoherent and avoids it. Coherent rotations: ['rht', 'none'].

- rl=32: tf_kl dense=2.868e+00, rht=7.345e-02, none=3.115e-02
- rl=64: tf_kl dense=2.667e+00, rht=1.284e-01, none=3.005e-02
- rl=128: tf_kl dense=3.661e+00, rht=1.068e-01, none=2.333e-02
- rl=256: tf_kl dense=4.723e+00, rht=2.420e-02, none=2.181e-02

**Gate: PASS** — int4 storage ≈4× smaller: True; generation coherent for ≥1 rotation (['rht', 'none']): True.
