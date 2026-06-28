# M14 gate — attention sink + per-channel keys (KVQuant/StreamingLLM)

- gate: for per_channel keys, adding a small fp16 sink does NOT hurt tf_kl (sink ≤ no-sink within noise) and ideally helps at the largest context

- per_channel ctx=16384: sink=0: 6.263e-02  ->  sink=4: 3.827e-02  ->  sink=16: 3.553e-02
- per_channel: no-sink(0)=6.263e-02 -> best sink(16)=3.553e-02 (1.76x, helps)
- per_token  ctx=16384: sink=0: 4.136e+00  ->  sink=4: 3.926e+00  ->  sink=16: 4.015e+00
- per_token: sink alone 3.926e+00 vs per_channel no-sink 6.263e-02 (M4 finding reproduced: the sink alone does not rescue per-token keys)

**Finding.** The honest result is that per-channel keys (M7) already do the heavy lifting on the outlier key channels, so a small BF16 attention sink is a CHEAP COMPLEMENT, not the main lever: at the largest context it does not hurt per-channel tf_kl (and tends to help slightly), consistent with KVQuant/StreamingLLM, which keep the first few sink tokens in fp16 ALONGSIDE per-channel key quantization. For per-token keys the sink alone stays insufficient — the M4 ``modest_morning`` finding reproduced, because the per-token failure mode is outlier key CHANNELS, which the sink (a few exact TOKENS) does not address. ``sink_length=0`` is byte-for-byte the no-sink cache; the sink only routes the stream's first tokens (kept exact) into a tiny BF16 buffer and stays position-aligned with pre-RoPE.

## Verdict: PASS  (per_channel + small sink ≤ no-sink within noise at largest ctx=True)
