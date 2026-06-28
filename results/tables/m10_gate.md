# M10 gate — dense-and-sparse outlier keys (KVQuant + QJL)

- gate: for **per_token** keys, tf_kl at key_outliers=8 is lower than at key_outliers=0 at the largest context (sparse fp16 outliers rescue the dense int4 grid)

- ctx=16384 per_token: ko=0: 4.136e+00  ->  ko=8: 9.279e-01
- per_token tf_kl reduced 4.46x from key_outliers=0 to 8
- memory cost: peak 1586MB -> 1610MB (+24MB) for key_outliers 0->8

**Finding.** The per-token key failure mode (M4) is a few outlier coordinates inflating the whole token's int4 scale and crushing the other ~60 coordinates. KVQuant/QJL keep the top-N outlier coordinates per key in fp16 (a sparse side-channel) and compute the affine range over the dense remainder only, so those coordinates use the full int4 grid. This is the **complementary** rescue of per-token int4 keys — per-channel keys (M7) isolate *channel* outliers with their own scale and so gain nothing from per-token outliers (a deliberate no-op). The cost is honest: each kept outlier adds 2 bytes of index + 2 bytes of fp16 value per token, a real memory tax the table's peak_mb column reflects.

## Verdict: PASS  (per_token tf_kl lower at key_outliers=8 than 0=True)
