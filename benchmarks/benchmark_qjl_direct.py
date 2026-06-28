"""M11 — "QJL done right": direct large-m key sketch (+fp16 outliers) vs M6.

M6 sketched the *residual* after an MSE recon with a single ``m=d`` 1-bit sketch
and found it variance-limited (only neutral near ≈3.5-4 bits). The practical QJL
/ TurboQuant-prod (Zandieh et al.) never builds an MSE base: it sketches the
**rotated key itself** with a LARGE Gaussian sign sketch (``m`` up to 512) and
keeps only ``sign(S·Rk)`` plus ``‖Rk‖`` — variance ~ ``1/m`` — optionally storing
a few exact fp16 outlier coordinates. This study reuses the M6 capture harness
(real Qwen2 K/Q, RHT / none rotation) and, for each ``m ∈ {64,128,256,512}`` and
``n_outliers ∈ {0,8}``, compares the DIRECT estimate ``logits_direct`` vs the
TRUE ``Rq@Rk.T``, alongside the M6 ``prod-1b`` and ``mse-only`` baselines for the
Pareto picture.

Metrics over the [nq, nk] query×key logit matrix: signed IP **bias**, IP-RMSE,
attention-KL (softmax over keys), and the realized bits/value. Writes:

    turbo_qjl_direct.csv       layer × rotation × method × m × n_outliers → bits, ip_bias, ip_rmse, attn_kl
    turbo_qjl_direct_hist.csv  sampled signed IP errors (one config) for the histogram

Run on a GPU env::

    python benchmarks/benchmark_qjl_direct.py --model-path Qwen/Qwen2.5-0.5B-Instruct \
        --ctx 2048 --layers 0,6,12,18,23 --sketch-m 64,128,256,512 \
        --outliers 0,8 --base-bits 2,3,4 --out-dir outputs
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbo_kv import metrics, quantizers, reporting  # noqa: E402
from turbo_kv import rotations as R  # noqa: E402
from turbo_kv import qjl  # noqa: E402

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
    return model, tok


def build_inputs(tok, ctx):
    text = _PASSAGE
    ids = tok(text, return_tensors="pt").input_ids
    while ids.shape[1] < ctx:
        text = text + _PASSAGE
        ids = tok(text, return_tensors="pt").input_ids
    return ids[:, :ctx]


def capture_projections(model, input_ids, layers):
    import torch

    caps: dict = {}
    handles = []

    def mk_hook(tag):
        def hook(_m, _i, out):
            caps[tag] = out.detach().float().cpu()
        return hook

    for li in layers:
        attn = model.model.layers[li].self_attn
        handles.append(attn.k_proj.register_forward_hook(mk_hook(("k", li))))
        handles.append(attn.q_proj.register_forward_hook(mk_hook(("q", li))))
    with torch.no_grad():
        model(input_ids=input_ids.to(model.device), use_cache=False)
    for h in handles:
        h.remove()
    return caps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--layers", default="0,6,12,18,23")
    ap.add_argument("--sketch-m", default="64,128,256,512",
                    help="direct sign-sketch sizes m (sign bits) to sweep")
    ap.add_argument("--outliers", default="0,8",
                    help="number of exact fp16 outlier coords kept per key")
    ap.add_argument("--base-bits", default="2,3,4",
                    help="bit-widths for the M6 prod-1b / mse-only baselines")
    ap.add_argument("--early-layers", default="0,6",
                    help="layers that use a wider m (early-layer m doubling)")
    ap.add_argument("--query-rows", type=int, default=256)
    ap.add_argument("--key-cap", type=int, default=2048)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    import torch

    os.makedirs(args.out_dir, exist_ok=True)
    summary_csv = os.path.join(args.out_dir, "turbo_qjl_direct.csv")
    hist_csv = os.path.join(args.out_dir, "turbo_qjl_direct_hist.csv")

    model, tok = load_model(args.model_path)
    cfg = model.config
    nKV = cfg.num_key_value_heads
    nH = cfg.num_attention_heads
    D = cfg.hidden_size // nH
    layers = [int(x) for x in args.layers.split(",")]
    early = {int(x) for x in args.early_layers.split(",") if x.strip() != ""}
    m_list = [int(x) for x in args.sketch_m.split(",")]
    outlier_list = [int(x) for x in args.outliers.split(",")]
    base_bits = [float(b) for b in args.base_bits.split(",")]
    rotations = ["rht", "none"]
    inv_sqrt_d = 1.0 / math.sqrt(D)
    print(f"model: kv_heads={nKV} heads={nH} head_dim={D} layers={layers} "
          f"m={m_list} outliers={outlier_list} base_bits={base_bits}")

    input_ids = build_inputs(tok, args.ctx)
    caps = capture_projections(model, input_ids, layers)

    hist_done = False
    for li in layers:
        k = caps[("k", li)].reshape(-1, nKV, D)
        q = caps[("q", li)].reshape(-1, nH, D)
        T = k.shape[0]
        nq = min(args.query_rows, T)
        nk = min(args.key_cap, T)
        q0 = q[:nq, 0, :]                        # [nq, D] queries (head 0)
        k0 = k[:nk, 0, :]                        # [nk, D] keys    (kv head 0)

        for kind in rotations:
            rot = R.make_rotation(kind, D, seed=0, dtype=torch.float32)
            Rq = rot.rotate(q0)
            Rk = rot.rotate(k0)
            exact = Rq @ Rk.t()                  # [nq, nk] exact inner products

            def record(method, approx, realized, m, n_out):
                err = (approx - exact).flatten()
                ip_bias = float(err.mean())
                ip_rmse = float(torch.sqrt((err ** 2).mean()))
                attn_kl = metrics.attention_kl(exact * inv_sqrt_d, approx * inv_sqrt_d)
                reporting.append_row(summary_csv, dict(
                    layer=li, rotation=kind, method=method, m=m, n_outliers=n_out,
                    realized_bits=round(realized, 4),
                    ip_bias=round(ip_bias, 6), ip_abs_bias=round(abs(ip_bias), 6),
                    ip_rmse=round(ip_rmse, 6), attn_kl=round(attn_kl, 6),
                ))
                return err

            # Direct large-m sketch (the real QJL operating point).
            for m in m_list:
                m_eff = 2 * m if li in early else m  # early-layer m doubling
                sk = qjl.QJLSketch(D, m=m_eff, seed=0)
                for n_out in outlier_list:
                    signs, norm, oi, ov = qjl.encode_key_direct(Rk, sk, n_outliers=n_out)
                    approx = qjl.logits_direct(Rq, signs, norm, sk, out_idx=oi, out_val=ov)
                    realized = qjl.direct_bits_per_value(sk, head_dim=D, n_outliers=n_out)
                    method = "direct" if n_out == 0 else "direct+out"
                    err = record(method, approx, realized, m_eff, n_out)
                    # representative histogram: layer0, m=256, outliers on.
                    if (not hist_done and li == layers[0] and m == 256 and n_out == max(outlier_list)
                            and max(outlier_list) > 0):
                        idx = torch.randint(0, err.numel(), (4000,))
                        for e in err[idx].tolist():
                            reporting.append_row(hist_csv, dict(
                                rotation=kind, method=f"direct-m{m_eff}-o{n_out}",
                                ip_err=round(e, 6)))
                        if kind == rotations[-1]:
                            hist_done = True

            # M6 baselines at matched-ish bits for the Pareto comparison.
            for bits in base_bits:
                record("mse-only", qjl.logits_mse(Rq, Rk, bits, axis="token"),
                       quantizers.effective_bits(bits), 0, 0)
                sk1 = qjl.QJLSketch(D, m=D, seed=0)
                approx = qjl.logits_prod(Rq, Rk, bits, sk1, axis="token")
                realized = qjl.prod_bits_per_value(bits, sk1, head_dim=D)
                record("prod-1b", approx, realized, D, 0)

            print(f"[m11] L{li} {kind}: done ({len(m_list)} m × {len(outlier_list)} outliers "
                  f"+ {len(base_bits)} baseline bits)")

    print(f"\nwrote {summary_csv}\nwrote {hist_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
