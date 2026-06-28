"""M3 — Custom ``TurboKVCache`` (int4 rotated K/V): memory + coherence on Qwen.

Builds the int4 rotated KV cache on a real Qwen2.5-0.5B context and validates
the two M3 gates:

* **Memory** — resident KV storage drops ≈ 4× vs BF16 (measured from the cache's
  own byte accounting *and* the CUDA allocator), swept over ``residual_length``.
* **Coherence** — teacher-forced next-token logits stay close to the BF16
  reference (argmax agreement + KL), and greedy continuation is non-degenerate.

Outputs (CSV, under ``--out-dir``):
    turbo_mse.csv         config × stored-bytes breakdown × reduction× × quality
    turbo_memory_kv.csv   closed-form vs measured KV bytes per bit-width
    (traces/turbo_mem_snapshot_*.pickle  before/after memory-history snapshots)

Run on a GPU env. Example::

    python benchmarks/benchmark_turbo_cache.py \
        --model-path Qwen/Qwen2.5-0.5B-Instruct --ctx 4096 --steps 32 \
        --residual-lengths 32,64,128,256 --rotations dense,rht,none --out-dir outputs
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbo_kv import metrics, reporting  # noqa: E402
from turbo_kv.cache import TurboKVCache  # noqa: E402

_FWD_KW: dict = {}

_PASSAGE = (
    "The transformer architecture processes sequences by attending over keys "
    "and values cached from previous tokens. As context grows the key-value "
    "cache dominates memory, motivating low-bit quantization. Rotating the "
    "cached tensors with an orthogonal transform spreads energy across "
    "coordinates so that a simple per-token scalar quantizer becomes near "
    "optimal, preserving inner products and attention distributions. "
)


def load_model(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()
    if "num_logits_to_keep" in inspect.signature(model.forward).parameters:
        _FWD_KW["num_logits_to_keep"] = 1
    return model, tok


def build_inputs(tok, ctx):
    import torch

    text = _PASSAGE
    ids = tok(text, return_tensors="pt").input_ids
    while ids.shape[1] < ctx:
        text = text + _PASSAGE
        ids = tok(text, return_tensors="pt").input_ids
    return ids[:, :ctx]


def reference_decode(model, input_ids, steps):
    """Greedy BF16 reference: returns (ref_tokens [1,steps], ref_logits [steps,1,V])."""
    import torch
    from transformers import DynamicCache

    cache = DynamicCache()
    toks, logits = [], []
    with torch.no_grad():
        pos = torch.arange(input_ids.shape[1], device=model.device)
        out = model(input_ids=input_ids.to(model.device), past_key_values=cache,
                    use_cache=True, cache_position=pos, **_FWD_KW)
        lg = out.logits[:, -1, :].float()
        nxt = lg.argmax(-1, keepdim=True)
        logits.append(lg.cpu()); toks.append(nxt.cpu())
        for i in range(steps - 1):
            cp = torch.tensor([input_ids.shape[1] + i], device=model.device)
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True,
                        cache_position=cp, **_FWD_KW)
            lg = out.logits[:, -1, :].float()
            nxt = lg.argmax(-1, keepdim=True)
            logits.append(lg.cpu()); toks.append(nxt.cpu())
    return torch.cat(toks, dim=1), torch.stack(logits)


def teacher_force_turbo(model, cache, input_ids, ref_tokens):
    """Feed the BF16 reference token stream through ``cache``; return turbo logits."""
    import torch

    steps = ref_tokens.shape[1]
    logits = []
    with torch.no_grad():
        pos = torch.arange(input_ids.shape[1], device=model.device)
        out = model(input_ids=input_ids.to(model.device), past_key_values=cache,
                    use_cache=True, cache_position=pos, **_FWD_KW)
        logits.append(out.logits[:, -1, :].float().cpu())
        for i in range(steps - 1):
            tok = ref_tokens[:, i:i + 1].to(model.device)
            cp = torch.tensor([input_ids.shape[1] + i], device=model.device)
            out = model(input_ids=tok, past_key_values=cache, use_cache=True,
                        cache_position=cp, **_FWD_KW)
            logits.append(out.logits[:, -1, :].float().cpu())
    return torch.stack(logits)


def greedy_turbo(model, cache, input_ids, steps):
    """Independent greedy decode through ``cache``; returns generated token ids."""
    import torch

    toks = []
    with torch.no_grad():
        pos = torch.arange(input_ids.shape[1], device=model.device)
        out = model(input_ids=input_ids.to(model.device), past_key_values=cache,
                    use_cache=True, cache_position=pos, **_FWD_KW)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        toks.append(nxt.cpu())
        for i in range(steps - 1):
            cp = torch.tensor([input_ids.shape[1] + i], device=model.device)
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True,
                        cache_position=cp, **_FWD_KW)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            toks.append(nxt.cpu())
    return torch.cat(toks, dim=1)


def distinct_ratio(ids) -> float:
    flat = ids.flatten().tolist()
    return len(set(flat)) / max(1, len(flat))


def reconstruction_diagnostic(model, input_ids, rotations, residual_length):
    """Isolate the rotation roundtrip from attention.

    Runs one BF16 forward to capture the *real* per-layer K/V, then for each
    rotation pushes the evicted (compressed) tokens through the exact cache
    primitives (rotate -> per-token int4 -> bf16 scale/zero -> unpack ->
    dequant -> inverse-rotate) and measures the reconstruction error directly.
    This tells us whether a rotation hurts because the *reconstruction* is worse
    (a real quantization phenomenon) rather than any attention-path artifact.

    Also reports the mean per-token quant *step* (max-min over head_dim) in the
    rotated space: a good rotation flattens each token so the int4 step shrinks.
    """
    import torch
    from transformers import DynamicCache

    from turbo_kv import packing as Pmod
    from turbo_kv import rotations as Rmod

    cache = DynamicCache()
    with torch.no_grad():
        pos = torch.arange(input_ids.shape[1], device=model.device)
        model(input_ids=input_ids.to(model.device), past_key_values=cache,
              use_cache=True, cache_position=pos, **_FWD_KW)

    D = cache.key_cache[0].shape[-1]
    rows = []
    for rot in rotations:
        rot_obj = Rmod.make_rotation(rot, D, seed=0, device=model.device, dtype=torch.float32)

        def _roundtrip(x):
            xr = rot_obj.rotate(x.to(torch.float32))
            codes, scale, lo = Pmod.quantize_int4_per_token(xr)
            # mimic the cache storing scale/zero in bf16
            scale = scale.to(torch.bfloat16).to(torch.float32)
            lo = lo.to(torch.bfloat16).to(torch.float32)
            packed = Pmod.pack_int4(codes)
            codes2 = Pmod.unpack_int4(packed, x.shape[-1])
            deq = Pmod.dequantize_int4_per_token(codes2.to(torch.float32), scale, lo)
            return rot_obj.inverse(deq).to(torch.float32), xr

        for name, store in (("K", cache.key_cache), ("V", cache.value_cache)):
            k_rel, k_step, k_relp99 = [], [], []
            for layer in range(len(store)):
                x = store[layer][:, :, : input_ids.shape[1] - residual_length, :]  # evicted slice
                if x.shape[2] == 0:
                    continue
                xf = x.to(torch.float32)
                rec, xr = _roundtrip(xf)
                err = (xf - rec).norm(dim=-1)            # [B,H,T] per-token L2 error
                ref = xf.norm(dim=-1).clamp_min(1e-9)
                rel = (err / ref)                        # per-token relative error
                step = (xr.amax(-1) - xr.amin(-1))       # int4 step proxy in rotated space
                k_rel.append(rel.mean().item())
                k_relp99.append(rel.flatten().quantile(0.99).item())
                k_step.append(step.mean().item())
            row = dict(
                rotation=rot, tensor=name,
                recon_relerr_mean=round(float(sum(k_rel) / max(1, len(k_rel))), 6),
                recon_relerr_p99=round(float(sum(k_relp99) / max(1, len(k_relp99))), 6),
                rot_step_mean=round(float(sum(k_step) / max(1, len(k_step))), 6),
            )
            rows.append(row)
            print(f"[recon] {rot:5s} {name}: relerr_mean={row['recon_relerr_mean']:.4f} "
                  f"relerr_p99={row['recon_relerr_p99']:.4f} step={row['rot_step_mean']:.4f}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--residual-lengths", default="32,64,128,256")
    ap.add_argument("--rotations", default="dense,rht,none")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    import torch

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "traces"), exist_ok=True)
    out_csv = os.path.join(args.out_dir, "turbo_mse.csv")
    mem_csv = os.path.join(args.out_dir, "turbo_memory_kv.csv")
    recon_csv = os.path.join(args.out_dir, "turbo_recon.csv")

    model, tok = load_model(args.model_path)
    cfg = model.config
    nL, nKV, D = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    input_ids = build_inputs(tok, args.ctx)
    print(f"model: layers={nL} kv_heads={nKV} head_dim={D} ctx={args.ctx} steps={args.steps}")

    # BF16 reference (quality anchor + memory baseline).
    ref_tokens, ref_logits = reference_decode(model, input_ids, args.steps)
    bf16_bytes = metrics.kv_cache_bytes(nL, 1, args.ctx, nKV, D, bytes_per_val=2.0)
    print(f"BF16 closed-form KV bytes @ctx={args.ctx}: {bf16_bytes / 1024**2:.1f} MB")

    residuals = [int(x) for x in args.residual_lengths.split(",")]
    rotations = [r.strip() for r in args.rotations.split(",")]

    # Direct reconstruction diagnostic (isolates rotation roundtrip from attention).
    recon_rows = reconstruction_diagnostic(model, input_ids, rotations, residual_length=128)

    rows, mem_rows = [], []
    snap_done = False
    for rot in rotations:
        for rl in residuals:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # Optional memory-history snapshot for one representative config.
            do_snap = (not snap_done and rot == "dense" and rl == 128)
            if do_snap:
                torch.cuda.memory._record_memory_history(max_entries=100000)

            cache = TurboKVCache(residual_length=rl, rotation=rot, head_dim=D, bits=4)
            turbo_logits = teacher_force_turbo(model, cache, input_ids, ref_tokens)
            peak_mb = metrics.peak_allocated_mb()
            store = cache.memory_bytes()
            reduction = bf16_bytes / store["total_bytes"]

            # Quality vs BF16 reference (teacher-forced).
            kl = metrics.attention_kl(ref_logits.squeeze(1), turbo_logits.squeeze(1), dim=-1)
            argmax_match = (ref_logits.argmax(-1) == turbo_logits.argmax(-1)).float().mean().item()

            # Coherence: independent greedy continuation.
            gen = greedy_turbo(model, TurboKVCache(residual_length=rl, rotation=rot, head_dim=D, bits=4),
                               input_ids, args.steps)
            distinct = distinct_ratio(gen)

            if do_snap:
                path = os.path.join(args.out_dir, "traces", "turbo_mem_snapshot_dense_rl128.pickle")
                try:
                    torch.cuda.memory._dump_snapshot(path)
                except Exception as e:  # noqa: BLE001
                    print(f"snapshot dump failed: {e}")
                torch.cuda.memory._record_memory_history(enabled=None)
                snap_done = True

            row = dict(
                rotation=rot, residual_length=rl, bits=4,
                stored_total_mb=round(store["total_bytes"] / 1024**2, 4),
                packed_mb=round(store["packed_bytes"] / 1024**2, 4),
                scale_zero_mb=round(store["scale_zero_bytes"] / 1024**2, 4),
                window_mb=round(store["window_bytes"] / 1024**2, 4),
                bf16_mb=round(bf16_bytes / 1024**2, 4),
                reduction_x=round(reduction, 4),
                peak_alloc_mb=round(peak_mb, 2),
                tf_argmax_match=round(argmax_match, 4),
                tf_kl=round(kl, 6),
                gen_distinct_ratio=round(distinct, 4),
            )
            rows.append(row)
            print(f"[m3]  {rot:5s} rl={rl:4d}: stored={row['stored_total_mb']:.1f}MB "
                  f"reduction={reduction:.2f}x tf_kl={kl:.3e} match={argmax_match:.3f} "
                  f"distinct={distinct:.3f}")

    # Memory: closed-form vs measured for a fixed config (dense, rl=128) across bit-widths.
    for bits, bpv in [(16, 2.0), (4, 0.5), (3, 0.375), (2.5, 0.3125)]:
        theo = metrics.kv_cache_bytes(nL, 1, args.ctx, nKV, D, bytes_per_val=bpv)
        mem_rows.append(dict(bits=bits, bytes_per_val=bpv,
                             theoretical_mb=round(theo / 1024**2, 4),
                             reduction_vs_bf16=round(bf16_bytes / theo, 4)))

    for r in rows:
        reporting.append_row(out_csv, r)
    for r in mem_rows:
        reporting.append_row(mem_csv, r)
    for r in recon_rows:
        reporting.append_row(recon_csv, r)
    print(f"\nwrote {out_csv}\nwrote {mem_csv}\nwrote {recon_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
