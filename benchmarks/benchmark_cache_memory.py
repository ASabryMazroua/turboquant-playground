"""M1 — KV-cache memory benchmark: closed-form vs **measured** bytes.

Validates PLAN Appendix B's KV-byte formula against the PyTorch allocator by
actually materializing per-layer key/value tensors at each (context, batch) and
reading the allocator delta. The BF16 measured-vs-closed-form gap must be within
~5% (an M1 validation gate). Also tabulates the theoretical int4 / 3-bit
reductions.

Runs on an A100 GPU node; writes ``outputs/kv_bytes.csv``. Falls back to a
config-only (no-CUDA) closed-form table if CUDA is unavailable.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turbo_kv.metrics import kv_cache_bytes  # noqa: E402
from turbo_kv.reporting import append_row  # noqa: E402

# Known Qwen2.5-0.5B-Instruct architecture (used if AutoConfig is unavailable).
_FALLBACK = dict(num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2, hidden_size=896)


def _model_dims(model_path: str) -> dict[str, int]:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path)
        num_attn = cfg.num_attention_heads
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // num_attn)
        return dict(
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            source="AutoConfig",
        )
    except Exception as exc:  # offline / transformers missing → fallback constants
        print(f"[warn] AutoConfig unavailable ({exc}); using fallback Qwen2.5-0.5B dims")
        f = _FALLBACK
        return dict(
            num_layers=f["num_hidden_layers"],
            num_kv_heads=f["num_key_value_heads"],
            head_dim=f["hidden_size"] // f["num_attention_heads"],
            source="fallback",
        )


def _measure_bf16_bytes(num_layers, batch, seq_len, num_kv_heads, head_dim) -> float | None:
    """Allocate the BF16 KV tensors and return the measured allocator delta (bytes)."""
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    tensors = []
    try:
        for _ in range(num_layers):
            for _kv in range(2):  # K and V
                tensors.append(
                    torch.empty(
                        batch, num_kv_heads, seq_len, head_dim,
                        dtype=torch.bfloat16, device="cuda",
                    )
                )
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
    finally:
        del tensors
        torch.cuda.empty_cache()
    return float(after - before)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--contexts", default="2048,8192,16384,32768")
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--out", default="outputs/kv_bytes.csv")
    args = ap.parse_args()

    contexts = [int(x) for x in args.contexts.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    dims = _model_dims(args.model_path)
    print(f"model dims ({dims['source']}): {dims}")

    out_path = pathlib.Path(args.out)
    if out_path.exists():
        out_path.unlink()

    worst_pct = 0.0
    for batch in batches:
        for ctx in contexts:
            cf_bf16 = kv_cache_bytes(dims["num_layers"], batch, ctx, dims["num_kv_heads"], dims["head_dim"], 2.0)
            cf_int4 = kv_cache_bytes(dims["num_layers"], batch, ctx, dims["num_kv_heads"], dims["head_dim"], 0.5)
            cf_3bit = kv_cache_bytes(dims["num_layers"], batch, ctx, dims["num_kv_heads"], dims["head_dim"], 0.375)
            measured = _measure_bf16_bytes(
                dims["num_layers"], batch, ctx, dims["num_kv_heads"], dims["head_dim"]
            )
            pct_err = (
                abs(measured - cf_bf16) / cf_bf16 * 100.0 if measured is not None else float("nan")
            )
            if measured is not None:
                worst_pct = max(worst_pct, pct_err)
            row = dict(
                context=ctx,
                batch=batch,
                num_layers=dims["num_layers"],
                num_kv_heads=dims["num_kv_heads"],
                head_dim=dims["head_dim"],
                closed_form_bf16_mb=round(cf_bf16 / 1024**2, 3),
                measured_bf16_mb=round(measured / 1024**2, 3) if measured is not None else None,
                abs_pct_err=round(pct_err, 3) if measured is not None else None,
                closed_form_int4_mb=round(cf_int4 / 1024**2, 3),
                closed_form_3bit_mb=round(cf_3bit / 1024**2, 3),
                int4_reduction_x=round(cf_bf16 / cf_int4, 3),
            )
            append_row(out_path, row)
            print(
                f"ctx={ctx:6d} b={batch}: closed={row['closed_form_bf16_mb']:.1f}MB "
                f"measured={row['measured_bf16_mb']}MB err={row['abs_pct_err']}%"
            )

    gate = "n/a (no CUDA)" if worst_pct == 0.0 else f"{'PASS' if worst_pct <= 5.0 else 'FAIL'} (worst {worst_pct:.2f}%)"
    print(f"\nKV-bytes within-5% gate: {gate}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
