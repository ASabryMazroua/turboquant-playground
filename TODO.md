# `turbo-kv-lab` — TODO

Tracks execution against [PLAN.md](PLAN.md). Each milestone ends with a **Codex
review → your approval** gate. Record every run (PASS **and** FAIL) in
`results/runs.md`.

Legend: ⬜ not started · 🟡 in progress · 🔵 awaiting Codex review · ✅ approved

---

## Gate 0 — Plan sign-off
- [x] You approved [PLAN.md](PLAN.md) Section 0 (rev 2)  ✅

---

## M0 — Scaffold + env + A100 connectivity  🔵
- [x] Create repo structure (PLAN Section 10)
- [x] Add `pyproject.toml`/`requirements.txt` (`torch`, `transformers`, `triton`, `azure-ai-ml`, `azure-identity`, `pandas`, `matplotlib`, `pynvml`)
- [x] Scaffold `turbo_kv/reporting.py` + `benchmarks/profiling/` (nvml_sampler, mem_snapshot, nsys/ncu wraps)
- [x] Adapt `benchmarks/_aml/submit_job_sdk.py` from ContentPlusCommerce
- [x] Write `benchmarks/_aml/hello_gpu.py` + `hello-a100-1gpu.yml`
- [x] `az ml` CLI working locally
- [x] Submit hello job — **note:** derived `ND12amrs_A100_v4` is INVALID ("instanceType is invalid"); valid SKU is 4-GPU `Singularity.ND48am_A100_v4`, 1 GPU via `CUDA_VISIBLE_DEVICES=0`
- [x] **Gate PASS:** logs show `NVIDIA A100-SXM4-80GB` (79.2 GB) + `torch.cuda.is_available()` True + Triton 2.2.0 imports
- [~] **Gate (partial):** `ncu` available ✅; **`nsys` NOT in the ACPT image** ⚠️ — must `apt-get install` nsys (or use a Nsight-bundled image) before M4/M5 timelines
- [x] Log job URL + device banner to `results/runs.md` (job `yellow_car_ph79x24pr7`)
- [ ] 🔵 Codex review → ✅ your approval  ← **current gate**

## M1 — Profiler + baselines  ✅
- [x] `turbo_kv/metrics.py` (memory, CUDA-event timing, exact-match, KL, MSE, IP error, perplexity)
- [x] `benchmarks/benchmark_cache_memory.py` (closed-form vs measured KV bytes)
- [x] `benchmarks/benchmark_generate.py` (prefill vs decode split)
- [x] Baselines: DynamicCache BF16, StaticCache BF16, HF QuantizedCache int4 (legacy `quanto==0.2.0`)
- [x] A100 sweep ctx {2k,8k,16k,30k} × batch {1,4} — job `polite_comb_n3h5wkf71v` ✅ PASS (5 attempts; see runs.md)
- [x] **Gate:** BF16 teacher-forced equivalence (KL-primary max KL 1.08e-3 <1e-2); `results/baseline.csv` fully populated; KV bytes 0.0% err
- [x] Record: context length where decode becomes memory-bound (dynamic flat ~22ms; static grows → `results/tables/m1_memory_bound.md`)
- [x] Profiler hooks: CUDA events + allocator stats + NVML sampler + 1 `torch.profiler` pass + mem snapshot
- [x] Report generator `benchmarks/report_m1.py` (7 plots + matrix table + crossover note)
- [x] Run report on downloaded outputs → plots/tables committed
- [x] Log runs (autonomous mode: continue unless a strong review comment lands)

## M2 — Torch TurboQuant-MSE  ✅
- [x] `turbo_kv/rotations.py` v1 dense orthogonal
- [x] `turbo_kv/rotations.py` v2 RHT ($D_1 H D_2 H$)
- [x] `turbo_kv/quantizers.py` scalar quant/dequant — per-channel **and** per-token (`fake_quantize(axis=...)`)
- [x] Unit tests: $RR^\top=I$, Hadamard correctness, FWHT==dense-H, + per-token rotation-benefit premise (52/52 pass)
- [~] `notebooks/01_rotation_quantization.ipynb` (deferred — benchmark + report cover the analysis)
- [x] **Gate PASS:** per-token RHT reduces IP error vs no-rotation at **12/12** (layer,bits) cells; RHT error ≈ dense at well-behaved layers; latency crossover identified (head_dim **4096**) — job `quirky_jelly_dpc46pyx2m`
- [x] Report MSE/cos/IP-error/KL at bits {2.5,3,3.5,4} × {token,channel} axis
- [x] Plots: error-vs-bits per rotation, axis-contrast (token vs channel), coord-magnitude histogram, rotation-latency crossover
- [x] Tables: recon-error × bits × rotation × axis → `results/tables/m2_error_matrix.md` + `m2_gate.md`
- [x] Log runs (autonomous mode: continue unless a strong review comment lands)
- **Key finding:** the *quantization axis* is the story — rotation helps **per-token** (channel outliers otherwise inflate the token scale), is neutral/harmful **per-channel**. At the model's head_dim=64, dense rotation is far cheaper than RHT (RHT wins only at head_dim≥4096).

## M3 — Custom TurboKVCache (int4)  ✅
- [x] `turbo_kv/packing.py` int4 pack/unpack
- [x] `turbo_kv/cache.py` (recent BF16 window + quantized older tokens)
- [x] **Gate PASS:** measured int4 storage **3.05–3.46×** smaller than BF16; generation coherent for **rht & none** (tf_kl ≤0.13, argmax 0.97–1.0); error within M2 tol — job `tidy_giraffe_rtdsr12xfj`
- [x] `results/turbo_mse.csv` (+ `turbo_memory_kv.csv`)
- [x] Profiler: allocator peak-alloc stats + memory-history snapshot (`results/traces/turbo_mem_snapshot_dense_rl128.pickle`)
- [x] Plots: theoretical-vs-measured KV-bytes bars, memory&KL-vs-residual_length, stored-bytes breakdown
- [x] pytest 64/64 (pack/unpack roundtrip, per-token quant, cache compression + ≈4× memory; 1 wrong-assert test fixed post-run)
- [x] Log runs (autonomous mode: continue unless a strong review comment lands)
- **Key finding:** real Qwen K/V per-token int4 is already near-lossless (`none` tf_kl 0.02); **`dense` orthogonal rotation breaks generation** (tf_kl 2.7–4.7) — but **not via reconstruction**: the direct diagnostic (`magenta_stamp_zv24bgwm37`) shows dense has the *best* reconstruction (relerr 0.088 ≈ rht < none 0.118, smallest int4 step). **Reconstruction MSE ⟂ inner-product fidelity** (TurboQuant thesis confirmed): RHT's structured-flat rotation gives isotropic int4 error decorrelated from the query; a Haar *dense* rotation isn't flat, so its error aligns with the correlated query → coherent logit distortion → blown-up KL. **M4 implication: use RHT, NOT dense.**

## M4 — Patched Qwen2 attention  🟡 (patch validated; quality gate FAIL — int4-only insufficient)
- [x] `turbo_kv/qwen_patch.py` (rotate-query identity, rotated-value accumulation, one inverse rotation) — **validated:** pytest 74/74, per-layer probe `none` bit-exact, `rht`/`dense` ~3.5% bf16 rotated-basis floor
- [x] Shape assertions for GQA `[B, H_kv, T, D]`
- [x] A100 generate vs BF16 at ctx {4k,8k,16k} (held-out novel-text window eval)
- [ ] **Gate FAIL (quality):** with a *valid* novel-text reference (ppl_bf16≈6.5) int4 `TurboKVCache` is pervasively lossy (median KL ~1–6, ppl ratio 4–260×). **Overturns M3's "near-lossless"** (that was a repeated-text artifact). Rotation helps at 4k (M2 reproduced) but inverts at long ctx (rht@8k 1389×). BF16 attention-sink (sink=4) gives only minor relief → not the fix.
- [x] Capture decode ms/tok + peak mem vs baseline (rht ≈ 2× bf16 decode from eager rotation; mem slightly lower)
- [x] Profiler: `torch.profiler` op breakdown (`m4_decode_ops`); nsys best-effort (apt blocked on non-root nodes)
- [x] Plots: `m4_quality_vs_rotation`, `m4_perplexity_ratio`, `m4_cost_bf16_vs_turbo`, `m4_layer_divergence`, `m4_sink_sweep`, `m4_decode_ops`
- [x] Tables: `m4_e2e` + `m4_gate.md` → `results/tables/`
- [x] Log runs (5 attempts in `runs.md`) → **finding:** int4-only insufficient → fix in M6 (QJL residual) and/or per-channel keys (KIVI)

## M5 — Fused int4 Triton kernels  ✅ (gate PASS — correct + memory-saving; honest latency loss)
- [x] `kernels/int4_logits_triton.py` (q_rot · packed_k → logits, unpack int4 in-kernel)
- [x] `kernels/int4_values_triton.py` (attn_weights · packed_v → o_rot, even/odd nibble split) + caller inverse RHT
- [x] Reference-equality tests `tests/test_kernels.py` (relerr < 1e-3 — **logits exactly 0.0**, values ≤ 3.4e-4)
- [x] `benchmarks/benchmark_attention_micro.py` (do_bench latency, peak-mem, bandwidth, decode M4-vs-M5)
- [x] **Gate PASS:** kernel ≈ reference (≤ 3e-4); **no full dequant materialized** (fused peak < ref peak, 16k: 25.4 vs 37.4 MB); decode latency reported **honestly** — fused **loses** to dequant+cuBLAS (0.11×–0.65×) because cuBLAS GEMM at head_dim=64 is hard to beat; the fused win is avoided BF16 reconstruction (memory), not GEMM latency
- [x] Profiler: `triton.testing.do_bench` (p50/p10/p90). nsys/ncu roofline skipped (nsys not in image; non-root apt blocked)
- [x] Plots: kernel-latency bars (`m5_latency_nq{1,64}`), speedup (`m5_speedup`), peak-mem (`m5_memory`), decode M4-vs-M5 (`m5_decode`)
- [x] Tables: reference-vs-fused (`m5_kernels.md`) → `results/tables/`
- [x] Log runs (`tender_machine…` compile-fix, `cool_basil…` PASS) → finding: int4 fusion saves the dequant materialization but doesn't beat cuBLAS latency at this head_dim

## M6 — TurboQuant-prod (QJL)  ✅ (gate PASS)
- [x] `turbo_kv/qjl.py` ($(b{-}1)$-bit recon + 1-bit QJL residual, keys first) + `tests/test_qjl.py`
- [x] **Gate PASS:** IP bias reduced vs MSE-only (max 66.7 → min **1.15**, <20%); prod extends the attn-KL Pareto **below the MSE-only floor** (MSE saturates ~6.5; `prod-4b@2.5b` → 3.88). Honest nuance: 1-bit sketch variance-limited at ≤3.5 b (matches paper "3.5 neutral / 2.5 marginal").
- [x] `results/turbo_prod.csv` (+ `turbo_prod_iperr.csv`) populated; sketch m-sweep {1,2,4}×d
- [x] Plots: signed IP-error histogram (`m6_ip_bias_hist`), IP-bias & attn-KL vs bits, **bits×quality Pareto** (`m6_pareto`)
- [x] Tables: MSE-only vs prod (IP-bias/RMSE/attn-KL) → `m6_prod.md`
- [x] Log runs (`magenta_pen…` 1-bit, `sharp_drawer…` m-sweep PASS) → key finding: QJL validates unbiased-IP claim + extends Pareto, at a memory cost

## M7 — The redemption: per-channel keys (KIVI fix)  ✅ (gate PASS)
- [x] Root-caused M4's 15–57× WikiText blowup via field research (KIVI/KVQuant/QJL): **per-token key quant was our mistake**; keys have outlier *channels*
- [x] `turbo_kv/packing.py` `quantize_int4_per_channel`; `TurboKVCache` `key_quant`/`key_group_size` (per-channel keys, group-aligned eviction); values stay per-token
- [x] Tests: `test_per_channel_keys_beat_per_token_on_channel_outlier`, `test_per_channel_decode_token_by_token` (pytest 19/19)
- [x] **Gate PASS:** per-channel int4 KV ppl_ratio **1.017/1.034/1.007×** at ctx 4k/8k/16k on WikiText (was 15.6/47.3/55.7×) — **15–55× better, near-lossless**, even post-RoPE
- [x] Plot `m7_per_channel_fix` + `m7_per_channel` table + `m7_gate.md`; run `epic_pear_1kpc0vwhp8`; README Finding 5 (redemption) + thesis updated
- [x] Next: deepen with the field's full recipe → **M8–M15 field-recipe fixes** (below)

## Final — Portfolio README  ✅ (live)
- [x] Compelling undergrad-level story with 5 findings incl. the redemption; thesis written
- [x] Embedded plots (`m2_axis_contrast`, `m4_corpus_comparison`, `m6_ip_bias_hist`, `m6_pareto`, `m5_decode`, `m7_per_channel_fix`)
- [x] Scrubbed secrets, MIT LICENSE, requirements.txt, reproducible (`report_*.py` rebuild plots on CPU); pushed to github.com/ASabryMazroua/turboquant-playground
- [x] **Retrieval chapter (Finding 8)** — FAISS multi-dataset (aniso/iso/blobs/unit/mnist) benchmark on quality·latency·RAM: the same QJL sketch that failed for attention is near-lossless for retrieval (shortlist + exact rerank absorbs the variance). QJL beats sign-LSH/SimHash up to 5× (uses the norm; converges on unit vectors = clean proof); OPQ rotate-before-quantize wins on structured data, no-op on isotropic (the KV lesson transfers). `retrieval/bench_faiss_turbo.py` + `retrieval/report_retrieval.py`; `results/retrieval_recall.csv` (51 rows) + plots `retrieval_qjl_vs_simhash`, `retrieval_tradeoff` + `retrieval_story.md`. No FAISS C++ fork (composable `IndexPreTransform` + 30-line QJL scorer).
- [ ] Optional: fill PLAN §11 master table

---

## Field-recipe fixes (M8–M15) — closing the gap to KIVI / KVQuant / QJL

M7 made int4 KV near-lossless with per-channel keys. These eight fixes apply the
rest of the field's recipe (the design choices we originally got wrong). Code is
implemented + `py_compile`-clean locally; **real numbers come from a batched A100
validation run** (local env has no torch/GPU). Legend: 🧩 code-complete (numbers
pending) · ✅ validated on A100.

### M8 — Pre-RoPE key quantization (KVQuant fix)  ✅
- [x] `TurboKVCache(pre_rope=True)`: store/quantize RAW pre-RoPE keys + a tiny int32 positions buffer; re-apply RoPE (HF-exact, from `inv_freq`) to reconstructed keys; values untouched; `memory_bytes` adds `position_bytes`
- [x] `qwen_patch.py`: `pre_rope` flag → RoPE on **query only**, pass raw key + `rope_inv_freq`+`cache_position` to the cache; `inv_freq` located for transformers 4.45.2 (model-level rotary, per-attn fallback)
- [x] `benchmark_qwen_turbo.py` `--rope-modes post,pre` (default `post`, no regression); `rope_mode` CSV column; re-patch per mode
- [x] `tests/test_pre_rope.py` (RoPE reconstruction vs HF reference, short-context exactness, eviction position alignment, pre_rope-off no-op); `report_m8.py` (`m8_pre_rope` plot/table + gate ≤ post-RoPE KL); `benchmarks/_aml/turbo-m8-1gpu.yml` (placeholders)
- [ ] **Gate (pending A100):** pre-RoPE tf_kl ≤ post-RoPE tf_kl at every context on WikiText (per-channel, rotation=none)

### M9 — Early-layer bit allocation (QJL)  ✅
- [x] `TurboKVCache(bf16_layers=N)` keeps the first N (sensitive) layers in full BF16, int4 the rest (M2 found layer 0 a huge outlier); `benchmark --bf16-layers 0,1,2`; `tests/test_early_layer_bits.py`; `report_m9.py` + `turbo-m9-1gpu.yml`
- [ ] **Gate (pending A100):** per_token tf_kl non-increasing as bf16_layers grows; per_channel already near-lossless (memory-cost tradeoff shown)

### M10 — Dense-and-sparse outliers (KVQuant + QJL)  ✅
- [x] `packing.quantize_int4_per_token_outliers` (top-N |coords| kept fp16, range over the dense rest); `TurboKVCache(key_outliers=N)` per-token KEY path; `tests/test_outliers.py`; `report_m10.py` + `turbo-m10-1gpu.yml`
- [ ] **Gate (pending A100):** per_token tf_kl at key_outliers=8 < at 0 (outliers rescue per-token int4, complementary to per-channel)

### M11 — QJL done right: large-m key sign-sketch (QJL)  ✅
- [x] `qjl.encode_key_direct`/`logits_direct`/`direct_bits_per_value` (sketch the rotated KEY directly at large m, no MSE base, +fp16 outliers); numerical study `benchmark_qjl_direct.py` (m∈{64..512}); `tests/test_qjl_direct.py`; `report_m11.py` + `turbo-m11-1gpu.yml`
- [ ] **Gate (pending A100):** direct m=256 attn-KL ≤ the M6 prod-1b result at comparable bits (the field's real QJL beats our M6 residual-1bit)

### M12 — Non-uniform quantization NUQ (KVQuant)  ✅
- [x] `quantizers.fit_nuq_levels`/`fake_quantize_nuq` (per-group k-means/quantile reconstruction levels); numerical study `benchmark_nuq.py` (uniform vs nuq-quantile vs nuq-kmeans); `tests/test_nuq.py`; `report_m12.py` + `turbo-m12-1gpu.yml`
- [ ] **Gate (pending A100):** nuq-kmeans attn-KL < uniform at matched bits (≤3) on the heavy-tailed regime

### M13 — Group-wise values + tuned residual buffer (KIVI/QJL)  ✅
- [x] `packing.quantize_int4_per_token_grouped` (one int4 scale per group of `value_group_size` coords); `TurboKVCache(value_group_size=G)` VALUE path; `residual_length` framed as the KIVI residual buffer; `benchmark --value-group-sizes 0,32`; `tests/test_value_groups.py`; `report_m13.py` + `turbo-m13-1gpu.yml`
- [ ] **Gate (pending A100):** grouped (32) value tf_kl ≤ whole-head at the largest ctx

### M14 — Attention-sink + per-channel combo (StreamingLLM/KVQuant)  ✅
- [x] Verified `sink_length` composes correctly with per-channel keys + M8 pre-RoPE (FIFO eviction keeps sink at positions 0..sink-1, aligned with the `_pos` buffer) + M13 value groups; `tests/test_sink_combo.py`; `benchmark --sink-lengths 0,4,16`; `report_m14.py` + `turbo-m14-1gpu.yml`
- [ ] **Gate (pending A100):** per-channel + small sink ≤ no-sink (cheap complement); per-token sink-alone insufficient (M4 reproduced)

### M15 — Fused int4 tensor-core kernel (best-effort Triton MMA)  ✅
- [x] `kernels/int4_logits_tc_triton.py` — int4→bf16 dequant per tile in SRAM + `tl.dot(allow_tf32=True)` on tensor cores (vs M5's exact fp32 `allow_tf32=False`); `tests/test_kernels_tc.py` (relerr < 1e-2); `benchmark_attention_micro_tc.py` (TC vs M5-exact vs bf16 cuBLAS); `report_m15.py` + `turbo-m15-1gpu.yml`
- [ ] **Gate (pending A100):** TC kernel correct (relerr < 1e-2) AND faster than M5-exact for nq≥64 (narrows the cuBLAS gap); nq=1 decode expected tensor-core-starved (honest)

> **Validation status (A100, 2026-06-28):** all eight ran on A100. **Gates PASS** for M8, M9, M10, M12, M13, M14, M15; **M11 PARTIAL** (QJL validated and beats the M6 1-bit sketch, but costs more bits than per-channel int4 — honest). M10 **failed first** (a missing lazy `import torch` in the outlier dequant path that `py_compile` can't catch) → fixed → rerun PASS. Headlines: **three independent rescues** of per-token int4 — per-channel keys (M7, →1.01×), one BF16 layer (M9, →1.34×), 8 fp16 outliers (M10, →2.27×); NUQ minimizing MSE without helping attention is the MSE≠inner-product lesson a third time. Per-fix plots in `results/plots/`, gates in `results/tables/`, full ledger in [results/runs.md](results/runs.md), narrative in README Finding 6.

---

## M16 — QJL wired into end-to-end generation (the README "natural next step")  🧩 (code-complete + locally validated; pending A100)

M6/M11 validated the unbiased inner-product QJL sign sketch only as a *numeric*
study. README §7 flagged the gap: "I did **not** wire QJL back into end-to-end
generation (the natural next step)." M16 closes that loop with a **custom
attention path** — QJL estimates `qᵀk` from `sign(S·Rk)` + `‖Rk‖`, so it can't use
the "reconstruct K → SDPA" path.

- [x] `turbo_kv/qjl.py`: `QJLSketch.estimate_batched` (broadcasts the GQA group axis; matmul-based unbiased IP estimate over arbitrary leading dims)
- [x] `turbo_kv/cache.py`: `key_quant="qjl"` (params `qjl_m=256`, `qjl_outliers`, `qjl_chunk=2048`); `_encode_qjl_key` (large-m sign sketch + fp16 outliers); `_append_store`/`memory_bytes` qjl branch (sign bits counted at the bit-packed ideal); **`qjl_update_and_attend`** + **`_qjl_attend`** — history via QJL estimate, sink+window exact (BF16), values int4-dequant, **query-chunked** so 16k prefill never builds the full `[q,T]` score matrix; pre_rope rejected (can't re-RoPE a sketch). `super().__init__()` made best-effort (version-portable Cache base)
- [x] `turbo_kv/qwen_patch.py`: QJL branch — bypass cache.update + SDPA, call `qjl_update_and_attend`, apply the single inverse rotation
- [x] `tests/test_qjl_e2e.py`: batched attend == 2D `logits_direct`/`estimate_matrix` (±outliers); chunking numerically invariant; all-window == exact SDPA; GQA 14/2 shapes; decode==prefill; **QJL beats per-token int4 on channel outliers**; pre_rope+qjl raises; memory counts sketch — **9/9 pass locally (CPU torch)**, full suite **120 passed / 2 skipped** (1 pre-existing unrelated `test_outliers` env edge)
- [x] `benchmarks/benchmark_qwen_turbo.py`: `--qjl-m`/`--qjl-outliers`, allow `qjl` in `--key-quants`, skip qjl+pre combo, CSV cols `qjl_m`/`qjl_outliers`
- [x] `benchmarks/report_m16.py` (per_token vs per_channel vs qjl ppl_ratio bars + realized key-bits + gate) + `benchmarks/_aml/turbo-m16-1gpu.yml` (WikiText sweep, pytest gate)
- [x] **Gate FAIL (honest negative — definitive):** on WikiText, even QJL's best config (RHT + m=512 + 8 fp16 outliers, ~11 bits/val) is ppl_ratio **42×/77×/37×** (4k/8k/16k) — worse than per-token int4 (15.6/47.3/55.7×) and **~36× off** per-channel (~1.01×). Two jobs: `plum_parrot_z14mw5hqnj` (outliers=0 worst case, 230×/2316×/483×) → comprehensive sweep `magenta_rail_0jtnplnjs2` (rotations {none,rht} × m {256,512} × outliers {0,8}). pytest 33/33 both. `per_token`/`per_channel` **exactly reproduce M7** → harness sound, catastrophe real.
- [x] Submitted + logged (`results/runs.md`); copied `turbo_e2e.csv` → `results/turbo_e2e_qjl.csv` (36 rows); ran `report_m16.py` (plots `m16_qjl_e2e` + `m16_qjl_ablation`, `m16_gate.md`); README **Finding 7** added + §7 limitation closed
- **Key finding:** the unbiased QJL estimator is **high-variance**, and softmax over thousands of keys is exquisitely sensitive to per-logit variance (spurious-max) → **unbiased ≠ good for attention**; low variance beats zero bias. Clean ablation: m + outliers cut 483×→37× (as theory predicts) but can't close the gap. Bonus failure: no key reconstruction → no flash-attention → 6.3× BF16 memory. **Per-channel int4 (M7) stays the answer; QJL's home is retrieval.** Extends M11 PARTIAL from numeric → end-to-end.


