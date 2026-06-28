"""M1 — generation benchmark: prefill vs decode, memory, and KV-cache baselines.

For each ``(cache, context, batch)`` cell it runs a manual prefill + greedy
decode loop (so prefill latency and **per-token** decode latency are measured
separately with CUDA events), records allocator peak memory, and checks
exact-next-token agreement against the BF16 ``DynamicCache`` reference.

Baselines (PLAN M1):
  * ``dynamic_bf16``   — ``transformers.DynamicCache`` (reference)
  * ``static_bf16``    — ``transformers.StaticCache`` (pre-allocated)
  * ``quantized_int4`` — ``transformers.QuantoQuantizedCache`` (HF int4; best-effort)

Optional diagnostic passes (separate from the timed numbers): ``--nvml`` (SM-util
sampler thread), ``--profile`` (one ``torch.profiler`` op-breakdown), and
``--mem-snapshot`` (allocator history pickle). Writes ``outputs/baseline.csv``.
"""
from __future__ import annotations

import argparse
import inspect
import pathlib
import sys
import threading
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turbo_kv import metrics  # noqa: E402
from turbo_kv.reporting import append_row  # noqa: E402

DECODE_WARMUP = 8  # decode steps discarded before collecting per-token samples

# Forward kwargs shared by every model call. ``num_logits_to_keep=1`` makes the
# model compute logits only for the last position (added in transformers 4.45):
# this is what real decoding needs and it avoids a giant [B, T, vocab] fp32
# logits tensor on long-context prefill (which OOMs at ctx=32768, batch=4).
# Populated in main() once the model is loaded and the kwarg is confirmed.
_FWD_KW: dict = {}


# --------------------------------------------------------------------------- #
# NVML sampler thread (pure sampler — safe to run alongside the timed workload)
# --------------------------------------------------------------------------- #
class NvmlThread(threading.Thread):
    def __init__(self, out_csv: pathlib.Path, gpu: int = 0, hz: float = 10.0):
        super().__init__(daemon=True)
        self.out_csv = out_csv
        self.gpu = gpu
        self.period = 1.0 / hz
        self._stop_event = threading.Event()  # NB: not `_stop` — Thread._stop is internal
        self.ok = False

    def run(self) -> None:
        try:
            import pynvml
        except Exception as exc:  # pragma: no cover
            print(f"[nvml] disabled: {exc}")
            return
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(self.gpu)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        self.ok = True
        with self.out_csv.open("w", encoding="utf-8") as fh:
            fh.write("t_s,util_gpu_pct,util_mem_pct,mem_used_mb,power_w,sm_clock_mhz\n")
            while not self._stop_event.is_set():
                now = time.perf_counter()
                try:
                    u = pynvml.nvmlDeviceGetUtilizationRates(h)
                    m = pynvml.nvmlDeviceGetMemoryInfo(h)
                    p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    c = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                    fh.write(
                        f"{now - t0:.4f},{u.gpu},{u.memory},"
                        f"{m.used / 1024**2:.1f},{p:.1f},{c}\n"
                    )
                    fh.flush()
                except Exception:
                    pass
                time.sleep(max(0.0, self.period - (time.perf_counter() - now)))
        pynvml.nvmlShutdown()

    def stop(self) -> None:
        self._stop_event.set()


# --------------------------------------------------------------------------- #
# Model / cache construction
# --------------------------------------------------------------------------- #
def load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()
    return model, tok


def build_cache(kind: str, model, batch: int, max_len: int):
    """Return a fresh cache object for the given kind, or raise for int4 errors."""
    import torch
    from transformers import DynamicCache, StaticCache

    if kind == "dynamic_bf16":
        return DynamicCache()
    if kind == "static_bf16":
        return StaticCache(
            config=model.config,
            max_batch_size=batch,
            max_cache_len=max_len,
            device=model.device,
            dtype=torch.bfloat16,
        )
    if kind == "quantized_int4":
        from transformers import QuantizedCacheConfig, QuantoQuantizedCache

        cfg = QuantizedCacheConfig(
            backend="quanto", nbits=4, q_group_size=64, residual_length=128,
            compute_dtype=torch.bfloat16, device=model.device,
        )
        return QuantoQuantizedCache(cfg)
    raise ValueError(f"unknown cache kind: {kind}")


def make_inputs(batch: int, ctx: int, vocab: int):
    import torch

    g = torch.Generator(device="cpu").manual_seed(1234)
    # Keep ids well inside the vocab and away from id 0 to avoid pad artefacts.
    ids = torch.randint(5, vocab - 1, (batch, ctx), generator=g)
    return ids.to("cuda")


# --------------------------------------------------------------------------- #
# One (cache, ctx, batch) measurement
# --------------------------------------------------------------------------- #
def run_cell(model, kind, ctx, batch, decode_tokens, prefill_reps):
    import torch

    vocab = model.config.vocab_size
    max_len = ctx + decode_tokens + 1
    input_ids = make_inputs(batch, ctx, vocab)

    # --- prefill timing (fresh cache each rep) ---
    def one_prefill():
        cache = build_cache(kind, model, batch, max_len)
        pos = torch.arange(ctx, device=model.device)
        with torch.no_grad():
            model(input_ids=input_ids, past_key_values=cache, use_cache=True, cache_position=pos, **_FWD_KW)

    prefill = metrics.time_cuda_ms(one_prefill, warmup=1, reps=prefill_reps, quantiles=(0.5, 0.1, 0.9))

    # --- decode timing + memory + generated tokens (one fresh run) ---
    torch.cuda.empty_cache()
    metrics.reset_peak_memory()
    cache = build_cache(kind, model, batch, max_len)
    with torch.no_grad():
        pos = torch.arange(ctx, device=model.device)
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, cache_position=pos, **_FWD_KW)
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)  # [B,1]

        gen_tokens = [next_tok]
        decode_ms: list[float] = []
        for i in range(decode_tokens):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            cp = torch.tensor([ctx + i], device=model.device)
            s.record()
            out = model(input_ids=next_tok, past_key_values=cache, use_cache=True, cache_position=cp, **_FWD_KW)
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            e.record()
            torch.cuda.synchronize()
            if i >= DECODE_WARMUP:
                decode_ms.append(s.elapsed_time(e))
            gen_tokens.append(next_tok)

    peak_mb = metrics.peak_allocated_mb()
    reserved_mb = metrics.peak_reserved_mb()
    dec = metrics.summarize_ms(decode_ms)
    # tokens/sec from the median per-token decode latency for this batch.
    toks_per_s = (batch * 1000.0 / dec["ms_p50"]) if dec["ms_p50"] and dec["ms_p50"] == dec["ms_p50"] else float("nan")
    gen = torch.cat(gen_tokens, dim=1)  # [B, decode_tokens+1]

    # GQA shape sanity (PLAN risk mitigation): keys are [B, num_kv_heads, T, head_dim].
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim", None) or (model.config.hidden_size // model.config.num_attention_heads)

    return dict(
        prefill_ms_p50=round(prefill["ms_p50"], 4),
        prefill_ms_p10=round(prefill["ms_p10"], 4),
        prefill_ms_p90=round(prefill["ms_p90"], 4),
        decode_ms_tok_p50=round(dec["ms_p50"], 4),
        decode_ms_tok_p10=round(dec["ms_p10"], 4),
        decode_ms_tok_p90=round(dec["ms_p90"], 4),
        tokens_per_s=round(toks_per_s, 2),
        peak_mem_mb=round(peak_mb, 1),
        reserved_mem_mb=round(reserved_mb, 1),
        num_kv_heads=n_kv,
        head_dim=head_dim,
    ), gen


# --------------------------------------------------------------------------- #
# Diagnostic passes (not part of the headline timings)
# --------------------------------------------------------------------------- #
def profiler_pass(model, ctx, batch, decode_tokens, out_csv):
    import torch
    from torch.profiler import ProfilerActivity, profile

    vocab = model.config.vocab_size
    input_ids = make_inputs(batch, ctx, vocab)
    cache = build_cache("dynamic_bf16", model, batch, ctx + decode_tokens + 1)
    with torch.no_grad():
        pos = torch.arange(ctx, device=model.device)
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, cache_position=pos, **_FWD_KW)
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True,
        ) as prof:
            for i in range(decode_tokens):
                cp = torch.tensor([ctx + i], device=model.device)
                out = model(input_ids=next_tok, past_key_values=cache, use_cache=True, cache_position=cp, **_FWD_KW)
                next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    trace_path = pathlib.Path("outputs/traces/decode_profile.json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    ka = prof.key_averages()
    print(ka.table(sort_by="self_cuda_time_total", row_limit=20))
    # Persist the top ops as a CSV for the report bar chart.
    out_csv = pathlib.Path(out_csv)
    if out_csv.exists():
        out_csv.unlink()
    rows = sorted(ka, key=lambda e: e.self_cuda_time_total, reverse=True)[:15]
    for e in rows:
        append_row(out_csv, dict(
            op=e.key,
            self_cuda_ms=round(e.self_cuda_time_total / 1000.0, 4),
            cuda_ms=round(e.cuda_time_total / 1000.0, 4),
            cpu_ms=round(e.self_cpu_time_total / 1000.0, 4),
            count=e.count,
        ))
    print(f"profiler trace -> {trace_path}; op breakdown -> {out_csv}")


def nvml_pass(model, ctx, batch, decode_tokens, out_csv):
    """Sample SM-/mem-utilisation with NVML during a dedicated decode loop.

    Run *after* the timed sweep so the sampler thread never shares the GIL with a
    measured cell (concurrent sampling corrupted CUDA-event timing previously).
    Decodes enough tokens to give the 10 Hz sampler a populated trace.
    """
    import torch

    vocab = model.config.vocab_size
    max_pos = getattr(model.config, "max_position_embeddings", 32768)
    steps = max(decode_tokens, min(256, max_pos - ctx - 1))
    input_ids = make_inputs(batch, ctx, vocab)
    nvml = NvmlThread(pathlib.Path(out_csv))
    nvml.start()
    try:
        cache = build_cache("dynamic_bf16", model, batch, ctx + steps + 1)
        with torch.no_grad():
            pos = torch.arange(ctx, device=model.device)
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, cache_position=pos, **_FWD_KW)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            for i in range(steps):
                cp = torch.tensor([ctx + i], device=model.device)
                out = model(input_ids=nxt, past_key_values=cache, use_cache=True, cache_position=cp, **_FWD_KW)
                nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
    finally:
        nvml.stop()
        nvml.join(timeout=2.0)
    print(f"nvml SM-util trace ({steps} decode steps) -> {out_csv}")


def teacher_forced_decode(model, kind, ctx, batch, force_tokens):
    """Feed the *reference* next-token stream through ``kind``'s cache and record
    this cache's per-step next-token logits + argmax.

    Because every cache consumes identical inputs at every step, any disagreement
    is pure cache numerics — not the chaotic greedy cascade you get when each
    cache decodes its own argmax (a single bf16 tie-break early flips the whole
    tail). This is the correct way to test cache *equivalence* and to measure
    quantization quality (argmax agreement + next-token KL) for later milestones.

    Returns ``(logits[S, B, vocab] cpu-f32, argmax[B, S] cpu)``.
    """
    import torch

    vocab = model.config.vocab_size
    steps = force_tokens.shape[1]
    input_ids = make_inputs(batch, ctx, vocab)  # same seed => same prefill as the timed run
    cache = build_cache(kind, model, batch, ctx + steps + 1)
    logits_steps, preds = [], []
    with torch.no_grad():
        pos = torch.arange(ctx, device=model.device)
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, cache_position=pos, **_FWD_KW)
        lg = out.logits[:, -1, :].float()
        logits_steps.append(lg.cpu())
        preds.append(lg.argmax(-1).cpu())
        for i in range(steps - 1):
            tok = force_tokens[:, i:i + 1]  # feed the reference token, not our own argmax
            cp = torch.tensor([ctx + i], device=model.device)
            out = model(input_ids=tok, past_key_values=cache, use_cache=True, cache_position=cp, **_FWD_KW)
            lg = out.logits[:, -1, :].float()
            logits_steps.append(lg.cpu())
            preds.append(lg.argmax(-1).cpu())
    return torch.stack(logits_steps), torch.stack(preds, dim=1)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--contexts", default="2048,8192,16384,32768")
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--caches", default="dynamic_bf16,static_bf16,quantized_int4")
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--prefill-reps", type=int, default=3)
    ap.add_argument("--out", default="outputs/baseline.csv")
    ap.add_argument("--nvml", action="store_true", help="sample SM-util during one representative decode")
    ap.add_argument("--profile", action="store_true", help="one torch.profiler op-breakdown pass")
    ap.add_argument("--mem-snapshot", action="store_true", help="dump allocator history for one config")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — M1 benchmark requires the A100 node.")
        return 2

    contexts = [int(x) for x in args.contexts.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    caches = [c.strip() for c in args.caches.split(",")]

    print(f"loading {args.model_path} ...")
    model, _tok = load_model(args.model_path)
    print(f"loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_key_value_heads} kv-heads, vocab {model.config.vocab_size}")

    # Only last-token logits are needed for greedy decode; enabling this avoids a
    # giant [B, T, vocab] fp32 logits tensor on long-context prefill (OOM at 32k/b4).
    if "num_logits_to_keep" in inspect.signature(model.forward).parameters:
        _FWD_KW["num_logits_to_keep"] = 1
        print("forward kwarg: num_logits_to_keep=1 (last-token logits only)")
    else:
        print("forward kwarg: num_logits_to_keep unsupported on this model/version")

    # Decoding must stay inside the model's positional range, otherwise rotary
    # embeddings go out of distribution and static/dynamic caches disagree.
    max_pos = getattr(model.config, "max_position_embeddings", 32768)
    ctx_cap = max_pos - args.decode_tokens - 1
    clamped = sorted({min(c, ctx_cap) for c in contexts})
    if clamped != sorted(contexts):
        print(f"contexts clamped to <= {ctx_cap} (max_pos {max_pos} - decode {args.decode_tokens} - 1): "
              f"{sorted(contexts)} -> {clamped}")
    contexts = clamped

    out_path = pathlib.Path(args.out)
    if out_path.exists():
        out_path.unlink()

    # Representative config for the heavy diagnostic passes.
    rep_ctx = max(c for c in contexts if c <= 16384) if any(c <= 16384 for c in contexts) else min(contexts)

    static_tf_match = []  # teacher-forced static==dynamic argmax agreement (the gate)
    static_tf_kl = []     # teacher-forced static-vs-dynamic next-token KL
    for batch in batches:
        for ctx in contexts:
            ref_tokens = None
            ref_logits = None
            for kind in caches:
                tag = f"{kind} ctx={ctx} b={batch}"
                try:
                    row, gen = run_cell(model, kind, ctx, batch, args.decode_tokens, args.prefill_reps)

                    # Greedy-cascade match (informational): each cache decodes its own
                    # argmax, so a single bf16 tie-break flips the whole tail at long ctx.
                    if kind == "dynamic_bf16":
                        ref_tokens = gen
                        greedy_match = 1.0
                    else:
                        greedy_match = metrics.exact_match(gen, ref_tokens) if ref_tokens is not None else float("nan")

                    # Teacher-forced equivalence (the real metric): feed the reference
                    # token stream through this cache and compare per-step logits.
                    if ref_tokens is not None:
                        logits_k, pred_k = teacher_forced_decode(model, kind, ctx, batch, ref_tokens)
                        tf_match = metrics.exact_match(pred_k, ref_tokens.cpu())
                        if kind == "dynamic_bf16":
                            ref_logits = logits_k
                            tf_kl = 0.0
                        else:
                            vocab = model.config.vocab_size
                            tf_kl = metrics.attention_kl(
                                ref_logits.reshape(-1, vocab), logits_k.reshape(-1, vocab))
                        if kind == "static_bf16":
                            static_tf_match.append(tf_match)
                            static_tf_kl.append(tf_kl)
                    else:
                        tf_match, tf_kl = float("nan"), float("nan")

                    append_row(out_path, dict(
                        cache=kind, context=ctx, batch=batch, status="ok",
                        exact_match_vs_bf16=round(greedy_match, 5),
                        tf_argmax_match=round(tf_match, 5),
                        tf_logit_kl=round(tf_kl, 8),
                        **row,
                    ))
                    print(f"[ok]   {tag}: peak={row['peak_mem_mb']}MB "
                          f"decode={row['decode_ms_tok_p50']}ms/tok tf_match={tf_match:.3f} "
                          f"tf_kl={tf_kl:.2e} greedy={greedy_match:.3f}")
                except Exception as exc:  # failures are first-class results
                    append_row(out_path, dict(
                        cache=kind, context=ctx, batch=batch, status=f"FAIL: {type(exc).__name__}",
                        exact_match_vs_bf16=None, tf_argmax_match=None, tf_logit_kl=None,
                        prefill_ms_p50=None, prefill_ms_p10=None, prefill_ms_p90=None,
                        decode_ms_tok_p50=None, decode_ms_tok_p10=None, decode_ms_tok_p90=None,
                        tokens_per_s=None, peak_mem_mb=None, reserved_mem_mb=None,
                        num_kv_heads=None, head_dim=None,
                    ))
                    print(f"[FAIL] {tag}: {type(exc).__name__}: {exc}")

    # --- diagnostic passes (separate from headline numbers) ---
    if args.nvml:
        try:
            nvml_pass(model, rep_ctx, batches[0], args.decode_tokens, "outputs/nvml_decode.csv")
        except Exception as exc:
            print(f"[warn] nvml pass failed: {exc}")

    if args.mem_snapshot:
        try:
            from benchmarks.profiling.mem_snapshot import record_memory

            input_ids = make_inputs(batches[0], rep_ctx, model.config.vocab_size)
            with record_memory("outputs/traces/decode_mem.pickle"):
                cache = build_cache("dynamic_bf16", model, batches[0], rep_ctx + args.decode_tokens + 1)
                with torch.no_grad():
                    pos = torch.arange(rep_ctx, device=model.device)
                    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, cache_position=pos, **_FWD_KW)
                    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
                    for i in range(args.decode_tokens):
                        cp = torch.tensor([rep_ctx + i], device=model.device)
                        out = model(input_ids=nxt, past_key_values=cache, use_cache=True, cache_position=cp, **_FWD_KW)
                        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        except Exception as exc:
            print(f"[warn] mem-snapshot failed: {exc}")

    if args.profile:
        try:
            profiler_pass(model, rep_ctx, batches[0], args.decode_tokens, "outputs/op_breakdown.csv")
        except Exception as exc:
            print(f"[warn] profiler pass failed: {exc}")

    # --- gate summary ---
    # Equivalence is judged teacher-forced: identical inputs => disagreement is pure
    # cache numerics. The rigorous equivalence metric is the next-token KL between
    # the static and dynamic caches (distribution identity); it is tiny everywhere.
    # Argmax-agreement is a secondary diagnostic that is inherently noisy at very
    # long context: ~1-2% of tokens are near-ties whose winner flips on sub-ulp bf16
    # rounding in the attention reduction order over the padded StaticCache buffer
    # (two runs of the SAME static cache disagree similarly). So the gate is
    # KL-primary with a loose 0.98 argmax sanity floor; greedy-cascade exact-match is
    # reported per-row but is NOT the gate (chaotic at long context even for
    # numerically-equivalent caches).
    if static_tf_match:
        worst = min(static_tf_match)
        worst_kl = max(static_tf_kl)
        ok = worst_kl < 1e-2 and worst >= 0.98
        print(f"\nBF16 self-consistency gate (teacher-forced static==dynamic): "
              f"{'PASS' if ok else 'FAIL'} (KL-primary: max KL {worst_kl:.2e} < 1e-2, "
              f"min argmax-agree {worst:.4f} >= 0.98)")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
