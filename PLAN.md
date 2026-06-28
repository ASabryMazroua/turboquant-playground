# `turbo-kv-lab` — Project Plan (for sign-off)

> A small, rigorous repo that implements and **profiles** a TurboQuant-style
> quantized KV cache for one small causal LM on **A100 80GB** (via AML
> Singularity). The portfolio value is the path: *paper idea → measured decode
> bottleneck → low-bit rotated cache → fused int4 attention kernel → honest
> tradeoff table*.

---

## 0. Sign-off & review workflow

This document is the contract. **Nothing in Milestones M0–M6 starts until you sign
off Section 0.** After that, work proceeds one milestone at a time:

```
For each milestone Mi:
  1. Implement deliverables
  2. Run validation gates (record PASS/FAIL + evidence — including failures)
  3. Submit the A100 job(s) on AML, capture job URL + result CSVs
  4. Generate the milestone's plots + comparison tables (§8) from those CSVs
  5. Pause → Codex reviews the output → you approve → next milestone
```

| Gate | Owner | Status |
|------|-------|--------|
| Plan approved (this doc) | **You** | ⬜ Awaiting sign-off |
| M0 reviewed | Codex → You | ⬜ |
| M1 reviewed | Codex → You | ⬜ |
| M2 reviewed | Codex → You | ⬜ |
| M3 reviewed | Codex → You | ⬜ |
| M4 reviewed | Codex → You | ⬜ |
| M5 reviewed | Codex → You | ⬜ |
| M6 reviewed | Codex → You | ⬜ |

**Validation philosophy:** every experiment is logged whether it *succeeds or
fails*. Failures are first-class results — we record the job URL, the metric that
missed, and the hypothesis for why. No silent drops.

---

## 1. Thesis (the one-paragraph pitch)

> I identified the KV cache as the long-context decode bottleneck, implemented a
> TurboQuant-style **rotated low-bit cache**, avoided inverse-rotating cached keys
> by **rotating queries instead**, accumulated over **rotated values** and
> inverse-rotated the output only once, **fused int4 dequantization into the
> attention kernel**, and measured the memory/latency/quality tradeoff on A100.

This reads as *applied scientist + systems ML engineer*, not "I implemented
quantization."

---

## 2. Scope & non-goals

**In scope:** one model (Qwen2.5-0.5B-Instruct), one attention implementation
(Qwen2), single A100 80GB, one clean reproducible benchmark table, MSE → prod
(QJL) progression, Triton fused int4 kernels.

**Explicit non-goals (do NOT build first):** full vLLM integration, CUDA C++
extension, 2.5-bit packing as a first target, multi-model support, LongBench
reproduction, multi-GPU/TP. These are "later, if time" stretch items only.

**A100-specific constraint:** A100 Tensor Cores support BF16/FP16/INT8/INT4/INT1
but **not native FP8**. So our novelty lives in **int4 / bit-packed** kernels, not
FP8. vLLM FP8 KV cache is referenced only as an external serving baseline, not a
core deliverable.

---

## 3. Paper grounding (validated against arXiv:2504.19874)

Confirmed from the abstract (Zandieh, Daliri, Hadian, Mirrokni, Apr 2025):

- TurboQuant addresses **both** MSE and **inner-product** distortion; data-oblivious
  and online (suitable for a streaming KV cache).
- Mechanism: **randomly rotate** input vectors → coordinates follow a concentrated
  **Beta** distribution and are near-independent in high dim → apply an **optimal
  scalar quantizer per coordinate**.
- MSE-optimal quantizers are **biased** for inner products → **two-stage**: MSE
  quantizer + **1-bit QJL** transform on the residual → **unbiased** inner-product
  estimator (this is "TurboQuant-prod").
- Near-optimal: within a small (~**2.7×**) constant of the information-theoretic
  distortion lower bound.
- KV-cache result we target: **absolute quality neutrality at 3.5 bits/channel**,
  **marginal degradation at 2.5 bits/channel**.

---

## 4. Core technical insight (the math)

Cache rotated tensors instead of raw ones. Let $R$ be a (structured) orthogonal
rotation.

**Keys — rotate the query, never inverse-rotate cached keys:**

$$q^\top k = (Rq)^\top (Rk)$$

So we store $Rk$ quantized, rotate the *current* query $q \to Rq$ at decode, and the
logits are unchanged (up to quant error). No per-token inverse rotation of the cache.

**Values — accumulate rotated, inverse-rotate once:**

$$o_{\text{rot}} = \sum_i a_i (R v_i) = R\Big(\sum_i a_i v_i\Big), \qquad o = R^\top o_{\text{rot}}$$

One $R^\top$ at the end of the head, instead of inverse-rotating every cached value.

**TurboQuant-prod residual (keys first):** store $\tilde k$ in $b-1$ bits, plus a
1-bit QJL sketch of $r = Rk - \widehat{Rk}$. At decode:

$$q^\top k \approx (Rq)^\top \widehat{Rk} + \widehat{(Rq)^\top r}_{\text{QJL}}$$

**Structured rotation (practical $R$):** randomized Hadamard transform
$R = D_1 H D_2 H$ with Walsh–Hadamard $H$ and random sign diagonals $D_1, D_2$ —
$O(d \log d)$ instead of $O(d^2)$, keeps the "spread the energy" property.

---

## 5. Platform — A100 80GB via AML Singularity (verified recipe)

Derived from `ContentPlusCommerce/Experiments/AML` (verified working setup). Full
copy-paste recipe in **Appendix A**.

| Field | Value |
|-------|-------|
| Subscription | `<SUBSCRIPTION_ID>` (WebXT Shopping Singularity) |
| Resource group | `<RESOURCE_GROUP>` |
| Workspace | `<WORKSPACE>` |
| Virtual cluster | `<VIRTUAL_CLUSTER>` |
| **A100 instance (single-GPU work)** | `Singularity.ND48am_A100_v4` (4-GPU node — smallest **valid** SKU in this VC; pin to 1 GPU via `CUDA_VISIBLE_DEVICES=0`) |
| A100 instances (portal-verified) | 8-GPU `Singularity.ND96amrs_A100_v4`, 4-GPU `Singularity.ND48am_A100_v4` / `ND48amr_A100_v4-n1`. **Derived `ND12`/`ND24` names are INVALID** ("instanceType is invalid", verified by job `teal_berry_qfq7f822wr`). |
| SLA tier | `Standard` (no Premium A100 in this VC) |
| Environment | **CUDA** ACPT image `mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu121-py310-torch22x:biweekly.202410.1` + pip `torch/transformers/triton` (NOT the ROCm images used for MI300X) |
| Submit | `az ml job create --file <job>.yml` **or** Python SDK `submit_job_sdk.py` |
| Data | No node internet → stage model+data to datastore `shopping_prod_c09` under `local/Retail/...`, mount as `uri_folder` |

**Known gotchas (baked into every job):**
- YAML folded-scalar trap: keep each `python` invocation on **one** line under `command: >-`.
- A100 ⇒ use **CUDA** env vars, never `PYTORCH_ROCM_ARCH`/`HSA_OVERRIDE_GFX_VERSION`.
- Custom image ⇒ set `SINGULARITY_CUSTOM_IMAGE: "true"` and `SINGULARITY_CUSTOM_IMAGE_COMMANDS: "USER root"`.
- Always include the UAI env var `_AZUREML_SINGULARITY_JOB_UAI` and `AZUREML_COMPUTE_USE_COMMON_RUNTIME: "true"`.
- Single A100 80GB is enough for everything here (0.5B model); multi-GPU is out of scope.

---

## 6. Model

**Qwen2.5-0.5B-Instruct** — small, modern, HF-compatible, **GQA**, long context to
**32,768** tokens. GQA matters: the KV cache has `num_kv_heads < num_attn_heads`, so
quantizing K/V is exactly where the memory is. Stage the **base instruct** weights to
the datastore (distinct from the fine-tuned product-extraction Qwen2.5-0.5B already
present in the workspace datastore).

---

## 7. Profiler & instrumentation toolkit (which tool per step & why)

**Guiding principle — separate *measurement* from *diagnosis*.** Headline numbers
(decode ms/token, tokens/sec, peak memory) are taken with **lightweight in-process
instruments only**, so a heavy profiler never inflates the number we report.
*Diagnostic* runs (why it's slow / where memory goes) use heavy profilers **once per
config** with NVTX ranges. Every timed region: warmup ≥ 10 iters, fixed seeds,
`torch.cuda.synchronize()` around the region, report **median + p10/p90** over N reps.

| Tool | Measures | Used in | Why this tool specifically |
|------|----------|---------|----------------------------|
| `torch.cuda.Event(enable_timing=True)` + `synchronize` | GPU wall-time: prefill, decode/token | M1–M5 | Device-side timestamps, ~zero overhead → the only clean way to get ms/token without CPU-dispatch noise or profiler tax. |
| `torch.cuda` allocator stats (`reset_peak_memory_stats`, `max_memory_allocated`, `memory_reserved`, `memory_stats`) | Exact bytes, alloc retries, OOM counters | M1, M3, M4 | Proves *allocator-level* reduction, not just theory; separates **allocated vs reserved** so the caching allocator can't flatter us. |
| `torch.cuda.memory._record_memory_history()` + `_dump_snapshot()` → memory_viz | Memory **timeline** (stacked, per-allocation) | M1, M3 | The most convincing memory visual: KV cache dominating, then shrinking under int4. |
| `torch.profiler` (CPU+CUDA, `record_shapes`, `profile_memory`, `with_stack`) → Chrome/Perfetto + TensorBoard | Per-operator time & memory attribution | M1, M4 | Attributes decode time to attention/cache vs MLP and **confirms the rotate-query path adds no per-token inverse-rotate op**. Overhead ~1.2–2× → diagnostic runs only. |
| NVML via `pynvml` (≈ `nvidia-smi dmon`) | SM%, mem%, power, clocks over time | M0, M1, M4 | Pure sampler (no kernel overhead) → yields the *"GPU under-utilized during decode"* utilization-vs-time plot that motivates the memory-bound thesis. |
| **Nsight Systems** `nsys profile` + NVTX | Whole-system timeline: kernels, CUDA API, memcpy, host gaps | M4, M5 | Confirms fused kernels replaced the dequant→matmul sequence and exposes host/launch overhead that `torch.profiler` hides. |
| **Nsight Compute** `ncu` | Per-kernel: achieved DRAM GB/s, occupancy, warp-stall, **roofline** | M5 | Systems-credibility evidence for the Triton kernels: proves they're **memory-bound** and quantifies % of A100 peak HBM BW (~2 TB/s) + int4-unpack cost. Expensive → kernels only. |
| `triton.testing.do_bench` (+ `perf_report`) | Triton kernel latency w/ warmup, L2 flush, quantiles | M5 | Canonical low-variance Triton microbench; returns median/min/max feeding the GB/s & speedup plots. |
| `scalene` / `py-spy` (optional) | Python host hotspots | M3, M4 (if CPU-bound) | Cheap flame graph to catch Python overhead in the cache wrapper at short context. |

**Reporting stack:** `pandas` (tables → CSV + markdown) and `matplotlib` (static plots
in `results/plots/`), all driven by `turbo_kv/reporting.py` so every figure/table
regenerates from CSVs. Heavy traces (`.nsys-rep`, `.ncu-rep`, Chrome traces, memory
snapshots) live in `results/traces/`.

---

## 8. Visualization & reporting plan (plots + comparison tables)

**Rules.** Every quantitative finding ships with: (a) a CSV in `results/`, (b) ≥1 plot
in `results/plots/`, (c) a markdown comparison table auto-rendered into the README —
all produced by `turbo_kv/reporting.py` (no hand-built figures). Conventions: x-axis is
usually **context length** or **bits/channel**; always label units; **p10/p90 error
bars** over reps; one consistent color per cache type; log-scale memory where it helps.
**Failures are plotted too** (a quality cliff at 2-bit is a finding).

| Step | Comparison tables | Plots |
|------|-------------------|-------|
| **M1** | baseline matrix: cache × ctx × batch → peak mem, decode ms/tok, tokens/s, exact-match | (a) memory vs context (line/cache); (b) decode ms/tok vs context; (c) tokens/s vs batch; (d) **SM-util vs time** during decode (NVML); (e) **memory-timeline snapshot** (KV dominating); (f) torch.profiler top-op breakdown |
| **M2** | recon error (MSE, cosine, IP-error, attn-KL) × bits {2.5,3,3.5,4} × rotation {none, dense, RHT} | (a) **error vs bits** curves per rotation; (b) attn-KL vs bits; (c) **coordinate-magnitude histogram** before/after rotation (Beta concentration); (d) rotation latency vs `head_dim`: dense $O(d^2)$ vs RHT $O(d\log d)$ |
| **M3** | theoretical vs measured KV-bytes per bit-width {4,3,2.5} + reduction ×; memory vs `residual_length` | (a) grouped bar theoretical-vs-measured; (b) **before/after memory-timeline**; (c) memory & quality vs `residual_length` (dual-axis) |
| **M4** | BF16 vs TurboKV-MSE → exact-match %, perplexity, attn-KL, decode ms/tok, peak mem @ ctx {8k,16k} | (a) quality vs bits; (b) decode ms/tok BF16 vs TurboKV vs context; (c) peak mem BF16 vs TurboKV; (d) **per-layer attn-KL heatmap**; (e) annotated **nsys timeline** excerpt |
| **M5** | kernel: reference dequant-path vs fused → latency (median/p10/p90), achieved GB/s, occupancy, % peak BW; end-to-end ms/tok M4 vs M5 | (a) **kernel-latency bars w/ quantiles** (do_bench); (b) **roofline** for both kernels (ncu); (c) achieved-vs-peak HBM bandwidth bar; (d) decode ms/tok M4 vs M5 vs BF16 |
| **M6** | MSE-only vs prod → IP-bias (mean signed err), IP-RMSE, attn-KL @ bits {2.5,3,3.5}; final §11 table | (a) **signed IP-error histogram** MSE vs prod (prod centered at 0 = unbiased); (b) attn-KL vs bits MSE vs prod; (c) **bits×quality×memory Pareto scatter** |
| **Final** | filled §11 portfolio table | Pareto scatter as README hero image |

---

## 9. Milestones

Each milestone lists **Deliverables**, **Validation gates (PASS/FAIL)**, the **A100 job**,
and its **Profiler + Artifacts** (cross-referencing §7 and §8). Status starts ⬜.

### M0 — Scaffold + env + A100 connectivity smoke test  ⬜
*De-risk the platform before any algorithm work.*
- **Deliverables:** repo structure (Section 10); Python env (`torch`, `transformers`,
  `triton`, `azure-ai-ml`, `azure-identity`, `pandas`, `matplotlib`, `pynvml`);
  `turbo_kv/reporting.py` + `benchmarks/profiling/` scaffold; `benchmarks/_aml/hello_gpu.py` +
  `hello-a100-1gpu.yml`.
- **A100 job:** run `python -c "import torch; print(torch.cuda.get_device_name())"`
  + `nvidia-smi` on `Singularity.ND12amrs_A100_v4`.
- **Validation gates:**
  - PASS if job reaches **Completed** and logs show **"NVIDIA A100 80GB"**.
  - PASS if `torch.cuda.is_available()` is True and Triton imports.
  - PASS if `nsys` and `ncu` are present on the node (needed in M4/M5); if blocked,
    record the gap and plan the `torch.profiler` + NVML fallback.
  - FAIL handling: if the 1-GPU SKU is rejected, record it and retry with the
    portal-verified 8-GPU node (use 1 GPU).
- **Profiler:** `nvidia-smi` + NVML only (smoke test, no plots). Save the device
  banner + driver/CUDA versions to `results/runs.md`.

### M1 — Profiler + baselines  ⬜
- **Deliverables:** `benchmarks/benchmark_generate.py`, `benchmark_cache_memory.py`;
  metric library (Appendix B). Baselines: `DynamicCache` BF16, `StaticCache` BF16,
  HF `QuantizedCache` int4 (HQQ/quanto).
- **A100 job:** sweep context ∈ {2k, 8k, 16k, 32k} × batch ∈ {1, 4} on 1×A100.
- **Validation gates:**
  - PASS if BF16 exact-next-token match == **100%** vs itself (sanity).
  - PASS if `results/baseline.csv` is populated for all (cache, ctx, batch) cells with
    peak memory, decode ms/token, tokens/sec.
  - PASS if measured BF16 KV bytes match the closed-form (Appendix B) within ~5%.
  - **Decision recorded:** which context length first shows decode as memory-bound
    (this justifies the whole project).
- **Profiler:** CUDA events (headline ms/token), allocator stats + memory-history
  snapshot, NVML sampler, one `torch.profiler` diagnostic pass. **Artifacts:** baseline
  matrix table + 6 plots, incl. SM-util-vs-time and the memory-timeline (§8 · M1).

### M2 — Torch TurboQuant-MSE (rotation + scalar quant)  ⬜
*No custom attention yet — pure numerics.*
- **Deliverables:** `turbo_kv/rotations.py` (v1 dense orthogonal, v2 RHT $D_1HD_2H$),
  `turbo_kv/quantizers.py` (per-coordinate scalar quant/dequant), error notebook.
- **Validation gates:**
  - PASS if unit tests confirm $RR^\top = I$ and Hadamard transform correctness.
  - PASS if RHT **reduces** inner-product error vs no-rotation at matched bits.
  - PASS if structured RHT error ≈ dense-orthogonal error but is **measurably faster**
    (both timed).
  - Report MSE, cosine sim, $|q^\top k - \hat q^\top \hat k|$, attention KL at bits ∈
    {2.5, 3, 3.5, 4}. Expect ≈ neutral at 3.5, marginal at 2.5 (paper anchor).
- **Profiler:** `torch.cuda.Event` (dense-R vs RHT timing). **Artifacts:** error-vs-bits
  curves per rotation, coordinate-magnitude histogram (Beta concentration), rotation
  latency vs `head_dim` (§8 · M2).

### M3 — Custom `TurboKVCache` (int4 rotated K/V)  ⬜
*May dequantize to BF16 before attention — correctness/memory MVP.*
- **Deliverables:** `turbo_kv/cache.py` (recent BF16 window `residual_length` + quantized
  older tokens), `turbo_kv/packing.py` (int4 bit-pack/unpack).
- **Validation gates:**
  - PASS if raw KV memory drops ≈ **4×** at int4 (measured allocator stats, not just theory).
  - PASS if generation stays coherent and quant error is within M2 tolerances.
  - `results/turbo_mse.csv` populated. Record latency honestly (may be ≥ BF16 here — expected).
- **Profiler:** allocator stats + memory-history snapshot (before/after). **Artifacts:**
  theoretical-vs-measured KV-bytes bars, before/after memory-timeline, memory-vs-`residual_length`
  tradeoff (§8 · M3).

### M4 — Patched Qwen2 attention end-to-end  ⬜
- **Deliverables:** `turbo_kv/qwen_patch.py` patching `Qwen2Attention.forward` to use the
  **rotate-query** identity (no key inverse-rotation) + **rotated-value** accumulation with a
  single output inverse-rotation.
- **A100 job:** generate with TurboKVCache vs BF16 baseline at ctx ∈ {8k, 16k}.
- **Validation gates:**
  - PASS if generated text matches/close to BF16 (exact-match % + perplexity on a small set).
  - PASS if attention KL within tolerance set in M2.
  - Capture decode ms/token + peak memory vs baseline (improvement not yet required).
- **Profiler:** `torch.profiler` (op breakdown + `profile_memory`) + Nsight Systems with
  NVTX ranges (confirm one inverse-rotation per head, not per token). **Artifacts:**
  quality-vs-bits, decode-ms/tok & peak-mem BF16-vs-TurboKV, per-layer attn-KL heatmap,
  annotated nsys timeline (§8 · M4).

### M5 — Fused int4 Triton kernels  ⬜
*Where real latency wins (if any) appear.*
- **Deliverables:** `kernels/int4_logits_triton.py` (q_rot · packed_k → logits, unpack in-kernel),
  `kernels/int4_values_triton.py` (attn_weights · packed_v → o_rot) + inverse RHT.
- **A100 job:** `benchmarks/benchmark_attention_micro.py` microbench + end-to-end decode.
- **Validation gates:**
  - PASS if kernel outputs match the PyTorch reference within set atol/rtol.
  - PASS if **no full dequantized K/V** is materialized (verified via peak memory).
  - Record decode latency vs M4 dequant path — **report honestly** whether int4 fusion
    wins, ties, or loses on A100, with the reason.
- **Profiler:** `triton.testing.do_bench` (kernel microbench) + Nsight Compute (roofline,
  achieved DRAM GB/s, occupancy) + Nsight Systems (end-to-end). **Artifacts:** kernel-latency
  bars with quantiles, roofline plot, achieved-vs-peak HBM bandwidth, decode-ms/tok M4 vs M5 (§8 · M5).

### M6 — TurboQuant-prod (QJL residual, keys first)  ⬜
- **Deliverables:** `turbo_kv/qjl.py` — $(b{-}1)$-bit MSE recon + 1-bit QJL sketch of the
  key residual; unbiased decode-time inner-product estimate.
- **Validation gates:**
  - PASS if inner-product **bias** is measurably reduced vs MSE-only (the paper's central claim).
  - PASS if attention KL improves at low bits (2.5–3) vs MSE-only.
  - `results/turbo_prod.csv` populated; **final benchmark table** (Section 11) filled end-to-end.
- **Profiler:** `torch.cuda.Event` (added QJL cost) + reuse the M2 numeric harness.
  **Artifacts:** signed IP-error histogram (MSE biased vs prod unbiased), attn-KL-vs-bits
  MSE-vs-prod, and the **bits×quality×memory Pareto scatter** (§8 · M6).

### Final — Portfolio README  ⬜
Fill the Section 11 table, write the thesis (Section 1), embed the Pareto scatter as the
README hero image, link job URLs + CSVs, and a short "what worked / what didn't" retro.

---

## 10. Repo structure

```text
turbo-kv-lab/
  README.md
  PLAN.md                      # this file
  TODO.md
  pyproject.toml               # or requirements.txt
  benchmarks/
    benchmark_generate.py
    benchmark_attention_micro.py
    benchmark_cache_memory.py
    profiling/
      nvml_sampler.py          # SM%/mem%/power vs time (pynvml)
      mem_snapshot.py          # record_memory_history -> snapshot.pickle
      nsys_wrap.sh             # Nsight Systems capture (+ NVTX ranges)
      ncu_wrap.sh              # Nsight Compute per-kernel roofline
    _aml/
      hello_gpu.py
      hello-a100-1gpu.yml
      turbo-bench-1gpu.yml
      submit_job_sdk.py        # adapted from ContentPlusCommerce
  turbo_kv/
    rotations.py
    quantizers.py
    packing.py
    cache.py
    qwen_patch.py
    qjl.py
    metrics.py
    reporting.py               # pandas tables + matplotlib plots from CSVs
  kernels/
    int4_logits_triton.py
    int4_values_triton.py
  notebooks/
    01_rotation_quantization.ipynb
    02_attention_error.ipynb
    03_decode_profile.ipynb
    04_kernel_roofline.ipynb
  results/
    baseline.csv
    turbo_mse.csv
    turbo_prod.csv
    runs.md                    # job URLs + PASS/FAIL log (incl. failures)
    plots/                     # committed .png/.svg figures
    tables/                    # auto-rendered markdown tables
    traces/                    # .nsys-rep, .ncu-rep, chrome traces, mem snapshots
```

---

## 11. The benchmark table (the portfolio artifact)

| Cache | Bits/value | Context | Peak mem | Decode ms/tok | Attn KL | Exact next-tok |
|-------|-----------:|--------:|---------:|--------------:|--------:|---------------:|
| BF16 DynamicCache | 16 | 8k | — | — | 0 | 100% |
| HF QuantizedCache | 4 | 8k | — | — | — | — |
| TurboKV-MSE Torch | 4 | 8k | — | — | — | — |
| TurboKV-MSE fused | 4 | 8k | — | — | — | — |
| TurboKV-prod fused | 4 | 8k | — | — | — | — |

> Auto-generated by `turbo_kv/reporting.py` from `results/*.csv`; add a `tokens/s`
> column at fill-in time and pair with the **bits×quality×memory Pareto** plot (§8 · M6).

---

## 12. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| 1-GPU A100 SKU name rejected | Fall back to portal-verified 8-GPU `ND96amrs` node, use 1 GPU; record in `runs.md`. |
| No node internet for weights | Stage Qwen2.5-0.5B-Instruct + eval data to `shopping_prod_c09` datastore, mount as `uri_folder`. |
| int4 fusion doesn't beat BF16 latency on A100 | Expected per HF guidance; the **memory** win is the headline, latency reported honestly. |
| Triton kernel correctness drift | Gate every kernel against a PyTorch reference with fixed atol/rtol before benchmarking. |
| GQA head bookkeeping bugs | Add shape assertions on `[B, num_kv_heads, T, head_dim]` throughout. |
| Heavy profiler skews headline latency | Take ms/token with CUDA events only; run `torch.profiler`/`nsys`/`ncu` on **separate** diagnostic passes (§7). |
| `nsys`/`ncu` restricted on Singularity nodes | Confirm availability in M0; if blocked, fall back to `torch.profiler` + NVML and note the gap in `runs.md`. |
| Scope creep (vLLM/CUDA C++/multi-model) | Hard non-goals in Section 2; revisit only after M6. |

---

## Appendix A — AML A100 submission recipe (copy-paste)

```powershell
# one-time
$env:PYTHONNOUSERSITE = "1"; $env:PIP_ONLY_BINARY = ":all:"
az extension add --name ml --yes
az login
az account set --subscription "<SUBSCRIPTION_ID>"
az configure --defaults group=<RESOURCE_GROUP> workspace=<WORKSPACE>

# submit + monitor
az ml job create --file benchmarks/_aml/hello-a100-1gpu.yml
az ml job stream --name <job-name>
az ml job show --name <job-name> --query "{Status:status, Name:name}" -o table
```

A100 CUDA command-job skeleton (single GPU):

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
type: command
experiment_name: turbo-kv-lab
display_name: hello-a100-1gpu
compute: /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.MachineLearningServices/virtualclusters/<VIRTUAL_CLUSTER>
environment:
  image: mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu121-py310-torch22x:biweekly.202410.1
code: ./src
command: >-
  python -u -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())" && nvidia-smi
resources:
  instance_count: 1
  instance_type: Singularity.ND12amrs_A100_v4   # fallback: Singularity.ND96amrs_A100_v4 (use 1 GPU)
  properties:
    AISuperComputer:
      interactive: false
      virtualClusterArmId: /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.MachineLearningServices/virtualclusters/<VIRTUAL_CLUSTER>
      slaTier: Standard
      priority: Medium
environment_variables:
  _AZUREML_SINGULARITY_JOB_UAI: /subscriptions/<SUBSCRIPTION_ID>/resourcegroups/<RESOURCE_GROUP>/providers/Microsoft.ManagedIdentity/userAssignedIdentities/<USER_ASSIGNED_IDENTITY>
  AZUREML_COMPUTE_USE_COMMON_RUNTIME: "true"
  SINGULARITY_CUSTOM_IMAGE: "true"
  SINGULARITY_CUSTOM_IMAGE_COMMANDS: "USER root"
```

---

## Appendix B — Metrics & formulas

**KV-cache size (bytes):**

$$\text{KV bytes} = 2 \times L \times B \times T \times H_{kv} \times D_{\text{head}} \times \text{bytes/val}$$

BF16 ⇒ 2 bytes/val; int4 packed ⇒ 0.5 ⇒ raw reduction $2/0.5 = 4\times$; 3-bit ⇒
$2/(3/8) = 5.33\times$.

**Memory (PyTorch):** `reset_peak_memory_stats`, `max_memory_allocated`,
`memory_allocated`, `memory_reserved`, `memory_stats` (watch alloc retries / OOM counters).

**Speed:** CUDA events (`torch.cuda.Event(enable_timing=True)`); report **prefill latency**
and **decode ms/token** separately, plus tokens/sec.

**Quality:** exact next-token match, attention-distribution KL, attention-output MSE,
inner-product error $|q^\top k - \hat q^\top \hat k|$, and end-to-end perplexity on a small set.

---

## Appendix C — Profiler cookbook (copy-paste)

**CUDA-event timing (headline ms/token):**

```python
torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True)
s.record(); step(); e.record(); torch.cuda.synchronize()
ms = s.elapsed_time(e)   # report median + p10/p90 over N reps
```

**Memory snapshot (timeline plot → view at pytorch.org/memory_viz):**

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
generate(...)
torch.cuda.memory._dump_snapshot("results/traces/mem.pickle")
```

**torch.profiler (op breakdown + trace):**

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True, with_stack=True) as p:
    decode_some_tokens()
p.export_chrome_trace("results/traces/decode.json")
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

**NVML utilization sampler:**
`python benchmarks/profiling/nvml_sampler.py --hz 10 --out results/traces/nvml.csv`

**Nsight Systems (timeline, wrap the job command):**

```bash
nsys profile -t cuda,nvtx,osrt -o results/traces/decode --force-overwrite true \
  python benchmarks/benchmark_generate.py ...
```

**Nsight Compute (per-kernel roofline, target the Triton kernels):**

```bash
ncu --set roofline -k "regex:int4_(logits|values)" -o results/traces/kernels \
  python benchmarks/benchmark_attention_micro.py ...
```

**Triton microbench:**

```python
import triton
ms = triton.testing.do_bench(lambda: int4_logits(qr, packed_k),
                             warmup=25, rep=100, quantiles=[0.5, 0.1, 0.9])
```
