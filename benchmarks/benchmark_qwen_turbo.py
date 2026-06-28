"""M4 — Patched Qwen2 attention end-to-end (rotate-query int4 KV) vs BF16.

Validates the M4 gates on a real Qwen2.5-0.5B context at ctx ∈ {8k, 16k}:

* **Quality** — teacher-forced next-token KL + argmax-match vs BF16, and the
  perplexity ratio (turbo / BF16) on the continuation. (Teacher-forced because
  greedy text cascades chaotically in BF16 too — see M1.)
* **Cost** — decode ms/token and peak allocator memory, BF16 vs TurboKV.
* **Per-layer divergence** — relative L2 error of each layer's attention-block
  output (proxy for the attn-KL heatmap), turbo vs BF16.

Profiler: `torch.profiler` op breakdown + `profile_memory` on one decode step,
with NVTX ranges around prefill/decode so an outer `nsys` capture (see the AML
yml) shows one inverse-rotation per head, not per token.

Outputs (CSV, under ``--out-dir``):
    turbo_e2e.csv        ctx × rotation × quality/latency/memory
    turbo_layer_kl.csv   ctx × rotation × layer × attn-output relerr
    turbo_ops.csv        top CUDA ops on a decode step (torch.profiler)

Run on a GPU env. Example::

    python benchmarks/benchmark_qwen_turbo.py \
        --model-path Qwen/Qwen2.5-0.5B-Instruct --contexts 8192,16384 \
        --steps 32 --rotations rht,none,dense --residual-length 128 --out-dir outputs
"""
from __future__ import annotations

import argparse
import inspect
import itertools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbo_kv import metrics, reporting  # noqa: E402
from turbo_kv.cache import TurboKVCache  # noqa: E402
from turbo_kv.qwen_patch import patch_qwen2_attention, unpatch_qwen2_attention  # noqa: E402

_FWD_KW: dict = {}

# A *diverse* multi-topic corpus. A single repeated paragraph makes the BF16
# reference near-deterministic (ppl ~ 1.0), which turns teacher-forced KL into a
# brittle near-one-hot metric that explodes on tie-flips (see the M4 attempt-1
# finding). Varied prose keeps the reference distribution non-degenerate so KL
# and perplexity are stable, meaningful quality signals.
_PARAGRAPHS = [
    "The transformer architecture processes sequences by attending over keys and "
    "values cached from previous tokens; as context grows the key-value cache "
    "dominates memory, motivating low-bit quantization of the stored tensors.",
    "In the high mountain valleys the shepherds moved their flocks at dawn, "
    "following narrow trails worn into the limestone over centuries, while the "
    "river below carried meltwater the color of pale jade toward the distant sea.",
    "Quarterly revenue rose nine percent on stronger demand for cloud services, "
    "though management warned that currency headwinds and a slower enterprise "
    "upgrade cycle could temper growth in the second half of the fiscal year.",
    "To reproduce the experiment, preheat the oven to two hundred degrees, fold "
    "the chilled butter into the flour until the mixture resembles coarse sand, "
    "then rest the dough for an hour before rolling it into thin rectangles.",
    "The committee debated whether the proposed amendment would withstand judicial "
    "review, citing precedent from earlier rulings that balanced individual "
    "liberty against the state's compelling interest in public safety.",
    "Photosynthesis converts carbon dioxide and water into glucose using energy "
    "absorbed by chlorophyll, releasing oxygen as a byproduct and forming the "
    "base of nearly every food web on the planet's surface and shallow oceans.",
    "She tightened the last bolt, wiped the grease from her hands, and listened as "
    "the rebuilt engine caught on the first turn, its idle settling into a low "
    "steady rhythm that told her the long winter of repairs was finally over.",
    "Modern compilers apply dozens of optimization passes, from inlining and loop "
    "unrolling to register allocation and vectorization, each transforming the "
    "intermediate representation while preserving the program's observable behavior.",
]
_PASSAGE = " ".join(_PARAGRAPHS) + " "

# Held-out novel text for the eval window — distinct topic, NOT in _PARAGRAPHS —
# so the BF16 reference distribution is non-degenerate (ppl >> 1) and the
# teacher-forced KL / perplexity actually measure int4-cache quality.
_EVAL_TEXT = (
    "Marine biologists tracking the migration discovered that the tagged turtles "
    "navigated by sensing minute variations in the planet's magnetic field, "
    "returning each season to the very beach where they had hatched decades "
    "earlier. The researchers noted that warming currents had shifted the timing "
    "of the journey by nearly three weeks, raising concerns about the synchrony "
    "between the animals' arrival and the availability of their preferred prey. "
    "Funding for the long-term study remained uncertain, yet the team pressed on, "
    "convinced that the data would prove indispensable for future conservation "
    "policy and for understanding how resilient the species might be to a rapidly "
    "changing ocean. By moonlight they catalogued each nest, measured the sand "
    "temperature, and released the hatchlings toward the glittering surf. "
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


def load_corpus_ids(tok, corpus):
    """Tokenize a real natural-text corpus once → [1, N] ids, or None for synthetic.

    ``wikitext`` loads WikiText-2 (raw, test split) — the standard LM-perplexity
    corpus — so the context is genuine non-repeating prose. This disentangles
    'int4 is lossy' from 'the tiled filler was pathological'.
    """
    if corpus == "synthetic":
        return None
    import torch

    if corpus == "wikitext":
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(t for t in ds["text"] if t and t.strip())
        ids = tok(text, return_tensors="pt").input_ids
        return ids
    raise ValueError(f"unknown corpus: {corpus}")


def build_inputs(tok, ctx, eval_tokens, corpus_ids=None):
    """Build a [1, ctx] sequence for the teacher-forced window eval.

    With ``corpus_ids`` (real prose) we take a natural, non-repeating chunk whose
    last ``eval_tokens`` are the genuine continuation — the standard LM eval.
    Otherwise we synthesize a tiled context + a *held-out* novel eval paragraph
    (the eval window must be novel so the BF16 reference is non-degenerate; a
    self-repeating tail makes KL brittle — see M4 attempts 1–2).
    """
    import torch

    if corpus_ids is not None:
        offset = 256  # skip leading headers/short lines
        if corpus_ids.shape[1] < offset + ctx:
            offset = max(0, corpus_ids.shape[1] - ctx)
        return corpus_ids[:, offset:offset + ctx]

    eval_ids = tok(_EVAL_TEXT, return_tensors="pt").input_ids
    while eval_ids.shape[1] < eval_tokens:
        eval_ids = tok(_EVAL_TEXT + " " + _EVAL_TEXT, return_tensors="pt").input_ids
    eval_ids = eval_ids[:, :eval_tokens]

    fill = ctx - eval_tokens
    text = _PASSAGE
    ids = tok(text, return_tensors="pt").input_ids
    while ids.shape[1] < fill:
        text = text + _PASSAGE
        ids = tok(text, return_tensors="pt").input_ids
    ctx_ids = ids[:, :fill]
    return torch.cat([ctx_ids, eval_ids], dim=1)


def teacher_forced_window(model, make_cache, input_ids, eval_tokens):
    """Teacher-force the last ``eval_tokens`` *real* tokens through ``make_cache()``.

    Returns ``(win_logits [W, V], targets [W])`` where ``win_logits[i]`` is the
    next-token distribution that predicts ``targets[i]``. Real tokens (not greedy)
    keep the reference distribution non-degenerate, and averaging over a wide
    window (W≈256) makes KL / perplexity stable instead of tie-flip brittle.
    """
    import torch

    T = input_ids.shape[1]
    ctx_len = T - eval_tokens
    cache = make_cache()
    logits = []
    with torch.no_grad():
        pos = torch.arange(ctx_len, device=model.device)
        out = model(input_ids=input_ids[:, :ctx_len].to(model.device),
                    past_key_values=cache, use_cache=True, cache_position=pos, **_FWD_KW)
        logits.append(out.logits[:, -1, :].float().cpu())     # predicts token ctx_len
        for i in range(eval_tokens - 1):
            tok = input_ids[:, ctx_len + i:ctx_len + i + 1].to(model.device)
            cp = torch.tensor([ctx_len + i], device=model.device)
            out = model(input_ids=tok, past_key_values=cache, use_cache=True,
                        cache_position=cp, **_FWD_KW)
            logits.append(out.logits[:, -1, :].float().cpu())
    win_logits = torch.cat(logits, dim=0)                     # [W, V]
    targets = input_ids[0, ctx_len:T].cpu()                   # [W]
    return win_logits, targets


def kl_stats(ref_logits, turbo_logits):
    """Per-position KL(ref ‖ turbo) over a [W, V] window: mean / median / p95."""
    import torch
    import torch.nn.functional as F

    log_p = F.log_softmax(ref_logits, dim=-1)
    log_q = F.log_softmax(turbo_logits, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)              # [W]
    kl, _ = kl.sort()
    n = kl.shape[0]
    return {
        "tf_kl": float(kl.mean()),
        "tf_kl_median": float(kl[n // 2]),
        "tf_kl_p95": float(kl[min(n - 1, int(0.95 * n))]),
    }


def window_perplexity(logits, targets):
    """exp(mean NLL) of ``targets`` [W] under ``logits`` [W, V]."""
    import torch
    import torch.nn.functional as F

    lp = F.log_softmax(logits, dim=-1)                        # [W, V]
    nll = -lp[torch.arange(targets.shape[0]), targets]
    return float(torch.exp(nll.mean()).item())


def decode_latency(model, input_ids, steps, make_cache):
    """Greedy decode ``steps`` tokens; return (ms_per_token p50, peak_alloc_mb)."""
    import torch

    samples = []
    metrics.reset_peak_memory()
    cache = make_cache()
    with torch.no_grad():
        pos = torch.arange(input_ids.shape[1], device=model.device)
        out = model(input_ids=input_ids.to(model.device), past_key_values=cache,
                    use_cache=True, cache_position=pos, **_FWD_KW)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        for i in range(steps):
            cp = torch.tensor([input_ids.shape[1] + i], device=model.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True,
                        cache_position=cp, **_FWD_KW)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
    summ = metrics.summarize_ms(samples)
    return summ["ms_p50"], metrics.peak_allocated_mb()


def per_layer_attn_relerr(model, input_ids, capture):
    """Run one forward capturing each layer's attention-block output via hooks.

    ``capture`` is a list filled with per-layer output tensors (last position).
    """
    import torch
    from transformers import DynamicCache

    handles = []
    capture.clear()
    slots = {}

    def mk_hook(idx):
        def hook(_mod, _inp, out):
            slots[idx] = out[0][:, -1, :].detach().float().cpu()
        return hook

    for idx, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(mk_hook(idx)))
    try:
        with torch.no_grad():
            pos = torch.arange(input_ids.shape[1], device=model.device)
            model(input_ids=input_ids.to(model.device), past_key_values=DynamicCache(),
                  use_cache=True, cache_position=pos, **_FWD_KW)
    finally:
        for h in handles:
            h.remove()
    for idx in sorted(slots):
        capture.append(slots[idx])


def profile_decode_ops(model, input_ids, make_cache, out_csv):
    """One decode step under torch.profiler; write top CUDA ops by self time."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    cache = make_cache()
    with torch.no_grad():
        pos = torch.arange(input_ids.shape[1], device=model.device)
        out = model(input_ids=input_ids.to(model.device), past_key_values=cache,
                    use_cache=True, cache_position=pos, **_FWD_KW)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        cp = torch.tensor([input_ids.shape[1]], device=model.device)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     profile_memory=True, record_shapes=False) as prof:
            with torch.cuda.nvtx.range("turbo_decode_step"):
                model(input_ids=nxt, past_key_values=cache, use_cache=True,
                      cache_position=cp, **_FWD_KW)
            torch.cuda.synchronize()
    rows = []
    for evt in prof.key_averages():
        cuda_us = getattr(evt, "self_device_time_total", 0) or getattr(evt, "self_cuda_time_total", 0)
        if cuda_us and cuda_us > 0:
            rows.append((evt.key, float(cuda_us)))
    rows.sort(key=lambda r: r[1], reverse=True)
    total = sum(r[1] for r in rows) or 1.0
    for name, us in rows[:20]:
        reporting.append_row(out_csv, dict(op=name[:48], cuda_us=round(us, 2),
                                           pct=round(100.0 * us / total, 2)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--contexts", default="8192,16384")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--eval-tokens", type=int, default=256)
    ap.add_argument("--rotations", default="rht,none,dense")
    ap.add_argument("--residual-length", type=int, default=128)
    ap.add_argument("--sink-lengths", default="0",
                    help="comma list of BF16 attention-sink token counts to sweep")
    ap.add_argument("--key-quants", default="per_token",
                    help="comma list of key quant axes: per_token and/or per_channel (KIVI)")
    ap.add_argument("--rope-modes", default="post",
                    help="comma list of RoPE modes: 'post' (RoPE before quant, current) "
                         "and/or 'pre' (KVQuant pre-RoPE key quant)")
    ap.add_argument("--bf16-layers", default="0",
                    help="comma list of INITIAL layer counts kept entirely in BF16 "
                         "(QJL per-layer bit allocation; 0 = all-int4, no change)")
    ap.add_argument("--key-outliers", default="0",
                    help="comma list of per-token KEY outlier counts kept in fp16 "
                         "(KVQuant/QJL dense-and-sparse; 0 = all-dense, no change)")
    ap.add_argument("--value-group-sizes", default="0",
                    help="comma list of VALUE group sizes (KIVI/AWQ group-wise int4 "
                         "values; 0 = whole-head per-token, no change; >0 must divide "
                         "head_dim)")
    ap.add_argument("--corpus", default="synthetic",
                    help="'synthetic' (tiled + held-out eval) or 'wikitext' (real prose)")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    import torch
    from transformers import DynamicCache

    os.makedirs(args.out_dir, exist_ok=True)
    e2e_csv = os.path.join(args.out_dir, "turbo_e2e.csv")
    layer_csv = os.path.join(args.out_dir, "turbo_layer_kl.csv")
    ops_csv = os.path.join(args.out_dir, "turbo_ops.csv")

    model, tok = load_model(args.model_path)
    cfg = model.config
    nL = cfg.num_hidden_layers
    nKV = cfg.num_key_value_heads
    D = cfg.hidden_size // cfg.num_attention_heads
    rl = args.residual_length
    contexts = [int(x) for x in args.contexts.split(",")]
    rotations = [r.strip() for r in args.rotations.split(",")]
    sink_lengths = [int(s) for s in args.sink_lengths.split(",")]
    key_quants = [k.strip() for k in args.key_quants.split(",")]
    rope_modes = [m.strip() for m in args.rope_modes.split(",")]
    bf16_layer_counts = [int(x) for x in args.bf16_layers.split(",")]
    key_outlier_counts = [int(x) for x in args.key_outliers.split(",")]
    value_group_sizes = [int(x) for x in args.value_group_sizes.split(",")]
    print(f"model: layers={nL} kv_heads={nKV} head_dim={D} residual_length={rl} "
          f"sinks={sink_lengths} key_quants={key_quants} rope_modes={rope_modes} "
          f"bf16_layers={bf16_layer_counts} key_outliers={key_outlier_counts} "
          f"value_group_sizes={value_group_sizes}")
    corpus_ids = load_corpus_ids(tok, args.corpus)
    if corpus_ids is not None:
        print(f"corpus={args.corpus}: {corpus_ids.shape[1]} tokens")

    ops_done = False
    for ctx in contexts:
        W = min(args.eval_tokens, ctx // 2)
        input_ids = build_inputs(tok, ctx, W, corpus_ids)

        # --- BF16 reference (unpatched): stable real-token window eval ---
        unpatch_qwen2_attention(model) if getattr(model, "_turbo_patched", False) else None
        ref_logits, targets = teacher_forced_window(model, lambda: DynamicCache(), input_ids, W)
        ppl_bf16 = window_perplexity(ref_logits, targets)
        bf16_ms, bf16_peak = decode_latency(model, input_ids, args.steps, lambda: DynamicCache())
        bf16_layers: list = []
        per_layer_attn_relerr(model, input_ids, bf16_layers)
        print(f"[ctx={ctx}] BF16 ppl={ppl_bf16:.3f} (W={W}) "
              f"decode={bf16_ms:.2f}ms/tok peak={bf16_peak:.0f}MB")

        for rot in rotations:
            patch_qwen2_attention(model, rotation=rot, seed=0)

            # per-layer attention-output relerr vs BF16 (int4-independent: uses a
            # non-quantized cache, so capture it once per rotation, not per sink).
            # Measured under the post-RoPE patch (rope-mode-independent).
            turbo_layers: list = []
            per_layer_attn_relerr(model, input_ids, turbo_layers)
            for li in range(len(turbo_layers)):
                ref = bf16_layers[li]
                rel = float((turbo_layers[li] - ref).norm() / ref.norm().clamp_min(1e-9))
                reporting.append_row(layer_csv, dict(ctx=ctx, rotation=rot, layer=li,
                                                     attn_out_relerr=round(rel, 6)))

            for sink, kq, mode, nbf, ko, vgs in itertools.product(
                    sink_lengths, key_quants, rope_modes, bf16_layer_counts,
                    key_outlier_counts, value_group_sizes):
                pre = (mode == "pre")
                # re-patch so the attention forward defers RoPE on keys iff pre.
                patch_qwen2_attention(model, rotation=rot, seed=0, pre_rope=pre)

                def make_turbo_cache():
                    return TurboKVCache(residual_length=rl, rotation="none",
                                        head_dim=D, bits=4, sink_length=sink,
                                        key_quant=kq, pre_rope=pre, bf16_layers=nbf,
                                        key_outliers=ko, value_group_size=vgs)

                turbo_logits, _ = teacher_forced_window(model, make_turbo_cache, input_ids, W)
                n_nonfinite = int((~torch.isfinite(turbo_logits)).sum())
                if n_nonfinite:
                    print(f"[m4][WARN] ctx={ctx} {rot} sink={sink}: {n_nonfinite} non-finite logits")
                kls = kl_stats(ref_logits, turbo_logits)
                argmax_match = (ref_logits.argmax(-1) == turbo_logits.argmax(-1)).float().mean().item()
                ppl_turbo = window_perplexity(turbo_logits, targets)
                turbo_ms, turbo_peak = decode_latency(model, input_ids, args.steps, make_turbo_cache)

                if (not ops_done and rot == "rht" and sink == sink_lengths[0]
                        and kq == key_quants[0] and mode == rope_modes[0]
                        and nbf == bf16_layer_counts[0] and ko == key_outlier_counts[0]
                        and vgs == value_group_sizes[0]):
                    profile_decode_ops(model, input_ids, make_turbo_cache, ops_csv)
                    ops_done = True

                reporting.append_row(e2e_csv, dict(
                    ctx=ctx, rotation=rot, residual_length=rl, sink_length=sink,
                    key_quant=kq, rope_mode=mode, bf16_layers=nbf, key_outliers=ko,
                    value_group_size=vgs,
                    corpus=args.corpus,
                    eval_tokens=W,
                    tf_kl=round(kls["tf_kl"], 6), tf_kl_median=round(kls["tf_kl_median"], 6),
                    tf_kl_p95=round(kls["tf_kl_p95"], 6), tf_argmax_match=round(argmax_match, 4),
                    n_nonfinite=n_nonfinite,
                    ppl_bf16=round(ppl_bf16, 4), ppl_turbo=round(ppl_turbo, 4),
                    ppl_ratio=round(ppl_turbo / ppl_bf16, 4),
                    decode_ms_bf16=round(bf16_ms, 3), decode_ms_turbo=round(turbo_ms, 3),
                    peak_mb_bf16=round(bf16_peak, 1), peak_mb_turbo=round(turbo_peak, 1),
                ))
                print(f"[m13] ctx={ctx} {rot:5s} sink={sink} key={kq} rope={mode} "
                      f"bf16L={nbf} ko={ko} vgs={vgs}: tf_kl={kls['tf_kl']:.3e} "
                      f"(med {kls['tf_kl_median']:.3e}) match={argmax_match:.3f} "
                      f"ppl_ratio={ppl_turbo / ppl_bf16:.3f} "
                      f"decode {turbo_ms:.2f}ms (bf16 {bf16_ms:.2f})")

    unpatch_qwen2_attention(model)
    print(f"\nwrote {e2e_csv}\nwrote {layer_csv}\nwrote {ops_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
