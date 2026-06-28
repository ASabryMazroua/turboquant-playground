# Azure ML A100 job specs — **scrubbed templates**

These YAML files are the exact Singularity command‑jobs used to run each milestone on a single
A100‑80GB. They are kept for transparency (they show the real environment, the `torch` pin trick, and
the Singularity env vars), but every internal identifier has been replaced with a placeholder:

| Placeholder | Replace with |
| --- | --- |
| `<SUBSCRIPTION_ID>` | your Azure subscription GUID |
| `<RESOURCE_GROUP>` | your resource group |
| `<WORKSPACE>` | your AML workspace |
| `<VIRTUAL_CLUSTER>` | your Singularity virtual cluster |
| `<USER_ASSIGNED_IDENTITY>` | your UAI name |
| `<TENANT_ID>` | your AAD tenant GUID |
| `<STORAGE_ACCOUNT>` | the workspace's default storage account (for downloading artifacts) |

They will **not** run as‑is. To reproduce on your own AML workspace, fill the placeholders and submit
with `az ml job create --file <spec>.yml`. If you don't have AML, ignore this folder entirely — the
same experiments run on **any** CUDA GPU via the plain `python benchmarks/...` commands in the top‑level
[README](../../README.md#6-reproduce-it).

| Spec | Milestone |
| --- | --- |
| `hello-a100-1gpu.yml` | M0 — GPU connectivity / environment smoke test |
| `turbo-bench-1gpu.yml` | M1 — BF16 / quantized-cache baselines + profiling |
| `turbo-m2-1gpu.yml` | M2 — rotation × quantization numeric study |
| `turbo-m3-1gpu.yml` | M3 — `TurboKVCache` int4 (memory + quality) |
| `turbo-m4-1gpu.yml`, `turbo-m4-wikitext-1gpu.yml` | M4 — patched Qwen2 attention; real‑text eval |
| `turbo-m5-1gpu.yml` | M5 — fused int4 Triton kernels |
| `turbo-m6-1gpu.yml` | M6 — QJL residual (unbiased inner products) |
