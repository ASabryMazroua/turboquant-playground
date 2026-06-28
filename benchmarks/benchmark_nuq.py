"""M12 — non-uniform quantization (NUQ) error sweep on real Qwen keys.

Pure numerics (no cache integration; PLAN §M12). KVQuant's premise: LLM
activations are heavy-tailed, so *non-uniform* reconstruction levels (fit by
empirical quantiles or 1-D k-means / Lloyd–Max) beat uniform affine levels at
the same bit budget — most at low bits and in the per-token regime where a few
large coordinates otherwise stretch a uniform grid.

Mirrors the M2 capture (hook ``k_proj``/``q_proj`` on a few layers of
Qwen2.5-0.5B, optionally rotate via :mod:`turbo_kv.rotations`) and, for each
``(layer, rotation, axis, bits, method)`` measures:

* key reconstruction MSE,
* inner-product RMSE over the full ``[nq, nk]`` query×key logit matrix,
* attention KL of ``softmax(qᵀk/√d)`` vs exact,
* realized ``effective_bits``,
* an honest ``codebook_bits_per_value`` overhead (``L*16 / group_size``) — the
  per-group fp16 codebook NUQ must store on top of the int indices.

Output (CSV, under ``--out-dir``):
    turbo_nuq.csv   rows: layer × rotation × axis × bits × method × metrics

Run on a GPU env. Example::

    python benchmarks/benchmark_nuq.py \
        --model-path Qwen/Qwen2.5-0.5B-Instruct --layers 0,11,23 \
        --ctx 2048 --query-rows 256 --key-cap 2048 --out-dir outputs
"""
from __future__ import annotations

import argparse
import math
import os
import sys

# Repo root on path (so turbo_kv imports when launched as a file).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbo_kv import metrics, quantizers as Q, rotations as R  # noqa: E402
from turbo_kv import reporting  # noqa: E402

BITS = [2.5, 3.0, 3.5, 4.0]
ROTATIONS = ["none", "rht"]          # NUQ is rotation-orthogonal; rht is the M2 winner
QUANT_AXES = ["token", "channel"]
# method ∈ uniform (affine baseline) vs NUQ (quantile / k-means codebooks).
METHODS = ["uniform", "nuq-quantile", "nuq-kmeans"]
KMEANS_ITERS = 10

_PASSAGE = (
    "The transformer architecture processes sequences by attending over keys "
    "and values cached from previous tokens. As context grows the key-value "
    "cache dominates memory, motivating low-bit quantization. Activation "
    "distributions in attention are heavy-tailed, so non-uniform reconstruction "
    "levels fit to the data density preserve inner products and attention "
    "distributions far better than a uniform grid at the same bit budget. "
)


def _build_inputs(tok, ctx: int):
    text = _PASSAGE
    ids = tok(text, return_tensors="pt").input_ids
    while ids.shape[1] < ctx:
        text = text + _PASSAGE
        ids = tok(text, return_tensors="pt").input_ids
    return ids[:, :ctx]


def _capture_projections(model, input_ids, layers):
    """Forward once, capturing k_proj/q_proj outputs of the requested layers."""
    import torch

    caps: dict = {}
    handles = []

    def mk_hook(tag):
        def hook(_module, _inp, out):
            caps[tag] = out.detach().float().cpu()

        return hook

    decoder_layers = model.model.layers
    for li in layers:
        attn = decoder_layers[li].self_attn
        handles.append(attn.k_proj.register_forward_hook(mk_hook(("k", li))))
        handles.append(attn.q_proj.register_forward_hook(mk_hook(("q", li))))

    with torch.no_grad():
        model(input_ids=input_ids.to(model.device), use_cache=False)

    for h in handles:
        h.remove()
    return caps


def _quantize(x, bits, *, axis, method):
    """Dispatch a method name to the uniform or NUQ quantizer for ``x``."""
    if method == "uniform":
        return Q.fake_quantize(x, bits, axis=axis, symmetric=False)
    if method == "nuq-quantile":
        return Q.fake_quantize_nuq(x, bits, axis=axis, method="quantile")
    if method == "nuq-kmeans":
        return Q.fake_quantize_nuq(x, bits, axis=axis, method="kmeans", iters=KMEANS_ITERS)
    raise ValueError(f"unknown method: {method!r}")


def _codebook_bits_per_value(method, bits, *, axis, head_dim, num_rows):
    """Honest per-group fp16 codebook overhead (0 for uniform).

    NUQ stores ``L`` fp16 levels per group. ``group_size`` is the number of
    values that share one codebook: ``head_dim`` for per-token codebooks, the
    token count for per-channel codebooks.
    """
    if method == "uniform":
        return 0.0
    L = Q.num_levels(bits)
    group_size = head_dim if axis == "token" else max(1, num_rows)
    return (L * 16.0) / group_size


def run_error_sweep(caps, layers, num_kv_heads, num_heads, head_dim,
                    query_rows, key_cap, out_csv):
    import torch

    rows = []
    for li in layers:
        k = caps[("k", li)].reshape(-1, num_kv_heads, head_dim)  # [T, Hkv, d]
        q = caps[("q", li)].reshape(-1, num_heads, head_dim)     # [T, Hq, d]
        T = k.shape[0]
        K = k.permute(1, 0, 2).reshape(-1, head_dim)             # [Hkv*T, d]
        if K.shape[0] > key_cap:
            idx = torch.linspace(0, K.shape[0] - 1, key_cap).long()
            K = K[idx]
        num_rows = K.shape[0]

        # Attention probe: query head 0 → kv head 0 over full [nq, nk] logits.
        kv0 = k[:, 0, :]
        q0 = q[:, 0, :]
        nq = min(query_rows, T)
        nk = min(key_cap, T)
        q_rows = q0[:nq]
        k_rows = kv0[:nk]
        inv_sqrt_d = 1.0 / math.sqrt(head_dim)

        for kind in ROTATIONS:
            rot = R.make_rotation(kind, head_dim, seed=0, dtype=torch.float32)
            RK = rot.rotate(K)
            Rq_rows = rot.rotate(q_rows)
            Rk_rows = rot.rotate(k_rows)
            exact_ip = Rq_rows @ Rk_rows.t()                     # [nq, nk]
            for axis in QUANT_AXES:
                for bits in BITS:
                    for method in METHODS:
                        RK_hat = _quantize(RK, bits, axis=axis, method=method)
                        key_mse = metrics.mse(RK, RK_hat)
                        Rk_rows_hat = _quantize(Rk_rows, bits, axis=axis, method=method)
                        approx_ip = Rq_rows @ Rk_rows_hat.t()
                        err = approx_ip - exact_ip
                        ip_rmse = torch.sqrt(torch.mean(err**2)).item()
                        akl = metrics.attention_kl(
                            exact_ip * inv_sqrt_d, approx_ip * inv_sqrt_d, dim=-1)
                        cb = _codebook_bits_per_value(
                            method, bits, axis=axis, head_dim=head_dim, num_rows=num_rows)
                        rows.append(dict(
                            layer=li, rotation=kind, axis=axis, bits=bits,
                            method=method,
                            effective_bits=round(Q.effective_bits(bits), 4),
                            key_mse=round(key_mse, 8),
                            ip_rmse=round(ip_rmse, 6),
                            attn_kl=round(akl, 8),
                            codebook_bits_per_value=round(cb, 6),
                        ))
                        print(f"[nuq]  L{li} {kind:4s} {axis:7s} bits={bits} "
                              f"{method:12s}: key_mse={key_mse:.4e} "
                              f"ip_rmse={ip_rmse:.4e} attn_kl={akl:.3e}")
    for r in rows:
        reporting.append_row(out_csv, r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layers", default="0,11,23")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--query-rows", type=int, default=256)
    ap.add_argument("--key-cap", type=int, default=2048)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]

    cfg = AutoConfig.from_pretrained(args.model_path)
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // num_heads
    print(f"model: heads={num_heads} kv_heads={num_kv_heads} head_dim={head_dim} "
          f"layers={cfg.num_hidden_layers}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32
    ).to(dev).eval()

    input_ids = _build_inputs(tok, args.ctx)
    print(f"captured context: {input_ids.shape[1]} tokens from layers {layers}")
    caps = _capture_projections(model, input_ids, layers)

    out_csv = os.path.join(args.out_dir, "turbo_nuq.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)

    run_error_sweep(caps, layers, num_kv_heads, num_heads, head_dim,
                    args.query_rows, args.key_cap, out_csv)

    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
