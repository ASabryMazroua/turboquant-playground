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
- [ ] Optional: fill PLAN §11 master table; add retrieval chapter

---

## Field-recipe fixes (M8–M15) — closing the gap to KIVI / KVQuant / QJL

M7 made int4 KV near-lossless with per-channel keys. These eight fixes apply the
rest of the field's recipe (the design choices we originally got wrong). Code is
implemented + `py_compile`-clean locally; **real numbers come from a batched A100
validation run** (local env has no torch/GPU). Legend: 🧩 code-complete (numbers
pending) · ✅ validated on A100.

### M8 — Pre-RoPE key quantization (KVQuant fix)  🧩
- [x] `TurboKVCache(pre_rope=True)`: store/quantize RAW pre-RoPE keys + a tiny int32 positions buffer; re-apply RoPE (HF-exact, from `inv_freq`) to reconstructed keys; values untouched; `memory_bytes` adds `position_bytes`
- [x] `qwen_patch.py`: `pre_rope` flag → RoPE on **query only**, pass raw key + `rope_inv_freq`+`cache_position` to the cache; `inv_freq` located for transformers 4.45.2 (model-level rotary, per-attn fallback)
- [x] `benchmark_qwen_turbo.py` `--rope-modes post,pre` (default `post`, no regression); `rope_mode` CSV column; re-patch per mode
- [x] `tests/test_pre_rope.py` (RoPE reconstruction vs HF reference, short-context exactness, eviction position alignment, pre_rope-off no-op); `report_m8.py` (`m8_pre_rope` plot/table + gate ≤ post-RoPE KL); `benchmarks/_aml/turbo-m8-1gpu.yml` (placeholders)
- [ ] **Gate (pending A100):** pre-RoPE tf_kl ≤ post-RoPE tf_kl at every context on WikiText (per-channel, rotation=none)

### M9 — Early-layer bit allocation (QJL)  ⬜
### M10 — Dense-and-sparse outliers (KVQuant + QJL)  ⬜
### M11 — QJL done right: large-m key sign-sketch (QJL)  ⬜
### M12 — Non-uniform quantization NUQ (KVQuant)  ⬜
### M13 — Group-wise values + tuned residual buffer (KIVI/QJL)  ⬜
### M14 — Attention-sink + per-channel combo (StreamingLLM/KVQuant)  ⬜
### M15 — Fused int4 tensor-core kernel (best-effort Triton MMA)  ⬜
