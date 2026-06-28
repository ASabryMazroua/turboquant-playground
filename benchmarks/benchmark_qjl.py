"""M6 — TurboQuant-prod (QJL residual) numeric study vs MSE-only.

Reuses the M2 harness: capture real Qwen2 K/Q projections, rotate with RHT, then
for each bit-width compare two encodings of the keys at *matched* memory:

* **mse-only** — ``b``-bit per-token MSE reconstruction (biased inner products).
* **prod** — ``(b-1)``-bit recon + 1-bit QJL sign-sketch of the residual
  (unbiased inner products), the paper's TurboQuant-prod.

Metrics over the full [nq, nk] query×key logit matrix: signed IP **bias**
(the central claim — prod should be ~0), |bias|, IP-RMSE, and attention-KL
(softmax over keys). Writes:

    turbo_prod.csv         layer × bits × method → realized_bits, ip_bias, ip_rmse, attn_kl
    turbo_prod_iperr.csv   sampled signed IP errors (one config) for the histogram

Run on a GPU env::

    python benchmarks/benchmark_qjl.py --model-path Qwen/Qwen2.5-0.5B-Instruct \
        --ctx 2048 --layers 0,6,12,18,23 --bits 2,2.5,3,3.5,4 --out-dir outputs
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
    import torch

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
    ap.add_argument("--bits", default="2,2.5,3,3.5,4")
    ap.add_argument("--rotations", default="rht,none")
    ap.add_argument("--sketch-mults", default="1,2,4",
                    help="QJL sketch size as multiples of head_dim (1 = 1-bit/value)")
    ap.add_argument("--query-rows", type=int, default=256)
    ap.add_argument("--key-cap", type=int, default=2048)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    import torch

    os.makedirs(args.out_dir, exist_ok=True)
    summary_csv = os.path.join(args.out_dir, "turbo_prod.csv")
    iperr_csv = os.path.join(args.out_dir, "turbo_prod_iperr.csv")

    model, tok = load_model(args.model_path)
    cfg = model.config
    nKV = cfg.num_key_value_heads
    nH = cfg.num_attention_heads
    D = cfg.hidden_size // nH
    layers = [int(x) for x in args.layers.split(",")]
    bits_list = [float(b) for b in args.bits.split(",")]
    rotations = [r.strip() for r in args.rotations.split(",")]
    sketch_mults = [int(x) for x in args.sketch_mults.split(",")]
    inv_sqrt_d = 1.0 / math.sqrt(D)
    print(f"model: kv_heads={nKV} heads={nH} head_dim={D} layers={layers} "
          f"bits={bits_list} rotations={rotations} sketch_mults={sketch_mults}")

    input_ids = build_inputs(tok, args.ctx)
    caps = capture_projections(model, input_ids, layers)

    iperr_done = False
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

            def record(method, approx, realized, bits, sketch_m):
                err = (approx - exact).flatten()
                ip_bias = float(err.mean())
                ip_rmse = float(torch.sqrt((err ** 2).mean()))
                attn_kl = metrics.attention_kl(exact * inv_sqrt_d, approx * inv_sqrt_d)
                reporting.append_row(summary_csv, dict(
                    layer=li, rotation=kind, bits=bits, method=method, sketch_m=sketch_m,
                    realized_bits=round(realized, 4),
                    ip_bias=round(ip_bias, 6), ip_abs_bias=round(abs(ip_bias), 6),
                    ip_rmse=round(ip_rmse, 6), attn_kl=round(attn_kl, 6),
                ))
                return err

            for bits in bits_list:
                # MSE-only baseline (no sketch).
                record("mse-only", qjl.logits_mse(Rq, Rk, bits, axis="token"),
                       quantizers.effective_bits(bits), bits, 0)
                # prod with a QJL sketch of m = mult * head_dim projections.
                for mult in sketch_mults:
                    sk = qjl.QJLSketch(D, m=mult * D, seed=0)
                    approx = qjl.logits_prod(Rq, Rk, bits, sk, axis="token")
                    realized = qjl.prod_bits_per_value(bits, sk, head_dim=D)
                    err = record(f"prod-{mult}b", approx, realized, bits, mult * D)
                    # representative histogram config: layer0, 3-bit, 1-bit sketch
                    if (not iperr_done and li == layers[0] and abs(bits - 3.0) < 1e-6
                            and mult == 1):
                        mse_err = (qjl.logits_mse(Rq, Rk, bits, axis="token") - exact).flatten()
                        idx = torch.randint(0, err.numel(), (4000,))
                        for e in mse_err[idx].tolist():
                            reporting.append_row(iperr_csv, dict(rotation=kind, method="mse-only",
                                                                 ip_err=round(e, 6)))
                        for e in err[idx].tolist():
                            reporting.append_row(iperr_csv, dict(rotation=kind, method="prod-1b",
                                                                 ip_err=round(e, 6)))
                        if kind == rotations[-1]:
                            iperr_done = True
                print(f"[m6] L{li} {kind} b{bits}: done ({len(sketch_mults)} sketch sizes)")

    print(f"\nwrote {summary_csv}\nwrote {iperr_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
