# M13 gate — group-wise VALUE quantization (KIVI/AWQ)

- gate: grouped values (value_group_size=32) tf_kl is ≤ whole-head (value_group_size=0) tf_kl at the largest context

- ctx=16384: vgs=0: 6.263e-02  ->  vgs=32: 4.601e-02
- tf_kl whole-head(0)=6.263e-02 -> grouped(32)=4.601e-02 (1.36x)
- memory cost: peak 1583MB -> 1586MB (+3MB) for the extra per-group value scales

**Finding.** Values were quantized per-token with ONE int4 scale over all 64 head coordinates. When a value head's coordinate magnitude varies across the head, that single scale is set by the largest sub-block and wastes the int4 grid on the rest. KIVI/AWQ quantize values per-token but in GROUPS (e.g. 32 coords), one affine scale/zero per group, so each sub-block gets its own range — finer value resolution exactly where intra-head magnitude varies. The cost is small and honest: ``ng = D/G`` scales+zeros per token instead of 1 (counted in the table's peak_mb / scale_zero bytes). This is a VALUES-only lever — keys keep their per-channel / per-token-outlier schemes, untouched. Separately, our ``residual_length`` BF16 window IS the KIVI **residual buffer**: the most recent tokens are kept full-precision and only evicted (and quantized) in chunks once they fall out of the window — already implemented.

## Verdict: PASS  (grouped values tf_kl ≤ whole-head at largest ctx=True)
