"""M2 — TurboQuant-MSE error sweep (rotation × bits) on real Qwen KV vectors.

Pure numerics (no custom attention yet, PLAN §M2). Collects real key/query
projections from Qwen2.5-0.5B, then for each rotation ∈ {none, dense, rht} and
bit-width ∈ {2.5, 3, 3.5, 4} measures how per-coordinate scalar quantization of
the *rotated* keys distorts:

* key reconstruction (MSE, cosine),
* inner products ``qᵀk`` (rmse / bias / max-abs),
* the attention distribution (KL of softmax(qᵀk/√d)).

Also emits (a) per-coordinate magnitude statistics before/after rotation (the
"Beta concentration" histogram source) and (b) dense-vs-RHT rotation latency vs
``head_dim`` via CUDA events.

Outputs (CSV, under ``--out-dir``):
    rotation_quant_error.csv   rows: layer × rotation × bits × metrics
    coord_magnitude.csv        rotation × coordinate variance + |value| histogram
    rotation_latency.csv       head_dim × rotation × ms quantiles

Run on a GPU env. Example::

    python benchmarks/benchmark_rotation_quant.py \
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
ROTATIONS = ["none", "dense", "rht"]
QUANT_AXES = ["token", "channel"]  # per-token is the regime where rotation helps

# A neutral English passage, tiled to reach the target context length so the
# captured projections reflect real (not random-token) statistics.
_PASSAGE = (
    "The transformer architecture processes sequences by attending over keys "
    "and values cached from previous tokens. As context grows the key-value "
    "cache dominates memory, motivating low-bit quantization. Rotating the "
    "cached tensors with an orthogonal transform spreads energy across "
    "coordinates so that a simple per-coordinate scalar quantizer becomes "
    "near optimal, preserving inner products and attention distributions. "
)


def _build_inputs(tok, ctx: int):
    import torch

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


def run_error_sweep(caps, layers, num_kv_heads, num_heads, head_dim, query_rows, key_cap, out_csv):
    import torch

    rows = []
    grp = num_heads // num_kv_heads  # query heads per kv head (GQA)
    for li in layers:
        k = caps[("k", li)].reshape(-1, num_kv_heads, head_dim)  # [T, Hkv, d]
        q = caps[("q", li)].reshape(-1, num_heads, head_dim)     # [T, Hq, d]
        T = k.shape[0]
        # Flatten keys over (token, kv-head) → coordinate matrix for quant scales.
        K = k.permute(1, 0, 2).reshape(-1, head_dim)             # [Hkv*T, d]
        if K.shape[0] > key_cap:
            idx = torch.linspace(0, K.shape[0] - 1, key_cap).long()
            K = K[idx]

        # Attention probe: query head 0 → kv head 0. IP-error and attn-KL are
        # measured over the FULL [nq, nk] query×key logit matrix (every q·k pair).
        kv0 = k[:, 0, :]                                          # [T, d]
        q0 = q[:, 0, :]                                           # [T, d]
        nq = min(query_rows, T)
        nk = min(key_cap, T)
        q_rows = q0[:nq]
        k_rows = kv0[:nk]
        inv_sqrt_d = 1.0 / math.sqrt(head_dim)

        for kind in ROTATIONS:
            rot = R.make_rotation(kind, head_dim, seed=0, dtype=torch.float32)
            RK = rot.rotate(K)                                    # rotated keys (quantize these)
            Rq_rows = rot.rotate(q_rows)
            Rk_rows = rot.rotate(k_rows)
            exact_ip = Rq_rows @ Rk_rows.t()                     # [nq, nk] exact q·k
            for axis in QUANT_AXES:
                for bits in BITS:
                    RK_hat = Q.fake_quantize(RK, bits, axis=axis, symmetric=False)
                    key_mse = metrics.mse(RK, RK_hat)
                    key_cos = metrics.cosine_similarity(RK, RK_hat)
                    # Quantize the probe keys with their own per-token/per-channel scales.
                    Rk_rows_hat = Q.fake_quantize(Rk_rows, bits, axis=axis, symmetric=False)
                    approx_ip = Rq_rows @ Rk_rows_hat.t()        # [nq, nk] q̂·k̂
                    err = approx_ip - exact_ip
                    ip_rmse = torch.sqrt(torch.mean(err**2)).item()
                    ip_bias = err.mean().item()
                    ip_max_abs = err.abs().max().item()
                    akl = metrics.attention_kl(exact_ip * inv_sqrt_d, approx_ip * inv_sqrt_d, dim=-1)
                    rows.append(dict(
                        layer=li, rotation=kind, quant_axis=axis, bits=bits,
                        effective_bits=round(Q.effective_bits(bits), 4),
                        key_mse=round(key_mse, 8),
                        key_cosine=round(key_cos, 6),
                        ip_rmse=round(ip_rmse, 6),
                        ip_bias=round(ip_bias, 6),
                        ip_max_abs=round(ip_max_abs, 6),
                        attn_kl=round(akl, 8),
                    ))
                    print(f"[err]  L{li} {kind:5s} {axis:7s} bits={bits}: key_mse={key_mse:.4e} "
                          f"ip_rmse={ip_rmse:.4e} attn_kl={akl:.3e}")
    for r in rows:
        reporting.append_row(out_csv, r)
    return rows


def run_coord_magnitude(caps, layers, num_kv_heads, head_dim, out_csv, nbins=40):
    """Per-coordinate variance + |value| histogram, before/after each rotation."""
    import torch

    # Pool keys across the requested layers for a stable distribution.
    Ks = [caps[("k", li)].reshape(-1, num_kv_heads, head_dim).permute(1, 0, 2).reshape(-1, head_dim)
          for li in layers]
    K = torch.cat(Ks, dim=0)                                      # [N, d]
    rows = []
    for kind in ROTATIONS:
        rot = R.make_rotation(kind, head_dim, seed=0, dtype=torch.float32)
        RK = rot.rotate(K)
        coord_var = RK.var(dim=0)                                 # [d]
        # Normalized energy concentration: max coord var / mean coord var.
        concentration = (coord_var.max() / coord_var.mean()).item()
        absvals = RK.abs().flatten()
        hi = torch.quantile(absvals, 0.999).item()
        edges = torch.linspace(0, max(hi, 1e-6), nbins + 1)
        counts = torch.histc(absvals, bins=nbins, min=0.0, max=max(hi, 1e-6))
        counts = (counts / counts.sum()).tolist()
        for b in range(nbins):
            rows.append(dict(
                rotation=kind, bin=b,
                edge_lo=round(edges[b].item(), 6),
                edge_hi=round(edges[b + 1].item(), 6),
                density=round(counts[b], 8),
                coord_var_concentration=round(concentration, 4),
            ))
        print(f"[hist] {kind:5s}: coord-var concentration (max/mean) = {concentration:.3f}")
    for r in rows:
        reporting.append_row(out_csv, r)
    return rows


def run_latency(out_csv, dims=(64, 128, 256, 512, 1024, 2048, 4096, 8192), batch=2048):
    """Dense O(d²) vs RHT O(d log d) rotation latency vs head_dim (CUDA events).

    Swept up to 8192 so the dense quadratic curve crosses above the RHT
    log-linear curve even though, at the model's actual head_dim=64, the single
    cuBLAS matmul of the dense rotation is the faster option.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for d in dims:
        x = torch.randn(batch, d, device=device, dtype=torch.float32)
        for kind in ("dense", "rht"):
            rot = R.make_rotation(kind, d, seed=0, device=device, dtype=torch.float32)
            if device == "cuda":
                t = metrics.time_cuda_ms(lambda: rot.rotate(x), warmup=10, reps=50)
                ms = t["ms_p50"]
                p10, p90 = t["ms_p10"], t["ms_p90"]
            else:  # CPU fallback timing
                import time
                for _ in range(5):
                    rot.rotate(x)
                ts = []
                for _ in range(20):
                    s = time.perf_counter()
                    rot.rotate(x)
                    ts.append((time.perf_counter() - s) * 1e3)
                ts.sort()
                ms, p10, p90 = ts[len(ts) // 2], ts[2], ts[-3]
            rows.append(dict(head_dim=d, rotation=kind,
                             ms_p50=round(ms, 6), ms_p10=round(p10, 6), ms_p90=round(p90, 6)))
            print(f"[lat]  d={d:5d} {kind:5s}: {ms:.4f} ms")
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
    print(f"model: heads={num_heads} kv_heads={num_kv_heads} head_dim={head_dim} layers={cfg.num_hidden_layers}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32
    ).to(dev).eval()

    input_ids = _build_inputs(tok, args.ctx)
    print(f"captured context: {input_ids.shape[1]} tokens from layers {layers}")
    caps = _capture_projections(model, input_ids, layers)

    err_csv = os.path.join(args.out_dir, "rotation_quant_error.csv")
    hist_csv = os.path.join(args.out_dir, "coord_magnitude.csv")
    lat_csv = os.path.join(args.out_dir, "rotation_latency.csv")
    for p in (err_csv, hist_csv, lat_csv):
        if os.path.exists(p):
            os.remove(p)

    run_error_sweep(caps, layers, num_kv_heads, num_heads, head_dim,
                    args.query_rows, args.key_cap, err_csv)
    run_coord_magnitude(caps, layers, num_kv_heads, head_dim, hist_csv)
    run_latency(lat_csv)

    print(f"\nwrote {err_csv}\nwrote {hist_csv}\nwrote {lat_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
