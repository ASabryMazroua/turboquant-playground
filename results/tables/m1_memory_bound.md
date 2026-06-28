# M1 memory-bound crossover (dynamic_bf16, batch=1)

context, decode_ms_tok_p50, x_vs_smallest
2048, 22.5302, 1.00x
8192, 21.8372, 0.97x
16384, 21.8586, 0.97x
30720, 21.8832, 0.97x

Decode latency stayed within 20% across the swept contexts (compute/overhead-bound here).
