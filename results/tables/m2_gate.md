# M2 gate summary

_Primary regime: **token-wise** quantization (where an orthogonal rotation removes channel-outlier inflation)._

- bits=2.5: ip_rmse none=5.3012e+02 → rht=9.1043e+01 (0.17×, reduced ✅)
- bits=3.0: ip_rmse none=3.6532e+02 → rht=1.2542e+02 (0.34×, reduced ✅)
- bits=3.5: ip_rmse none=1.5950e+02 → rht=6.1232e+01 (0.38×, reduced ✅)
- bits=4.0: ip_rmse none=1.1107e+02 → rht=7.2291e+01 (0.65×, reduced ✅)

- per-(layer,bits) cells where RHT < none: **12/12**; dense < none: **12/12** (layer 0 is the outlier layer — largest KV magnitudes).

- bits=2.5: ip_rmse dense=1.1904e+02 vs rht=9.1043e+01 (ratio 0.76×, mean over layers)
- bits=3.0: ip_rmse dense=6.8984e+01 vs rht=1.2542e+02 (ratio 1.82×, mean over layers)
- bits=3.5: ip_rmse dense=7.8919e+01 vs rht=6.1232e+01 (ratio 0.78×, mean over layers)
- bits=4.0: ip_rmse dense=1.7618e+01 vs rht=7.2291e+01 (ratio 4.10×, mean over layers)

- RHT becomes faster than dense at head_dim ≥ **4096** (at head_dim=8192: dense=14.619 ms vs rht=7.280 ms ✅)

_Contrast (per-channel, bits=4.0): none=1.6742e+00 vs rht=1.9693e+00 — rotation does **not** help per-channel quant._

**Gate: PASS** — RHT reduces inner-product error vs no-rotation at every per-token (layer,bits) cell (12/12) and at all bit-widths on average (True). Latency crossover head_dim: 4096.
