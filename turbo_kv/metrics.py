"""Metric library for turbo-kv-lab (PLAN Appendix B).

Reusable across milestones:

* **Memory** — closed-form KV-cache bytes + allocator-based peak helpers.
* **Speed** — CUDA-event timing with median / p10 / p90 quantiles.
* **Quality** — exact next-token match, attention-distribution KL, MSE, cosine,
  inner-product error ``|q·k - q̂·k̂|``, and end-to-end perplexity.

``torch`` is imported lazily inside the functions that need it so the
closed-form / pure-python helpers can be imported by non-GPU tooling.
"""
from __future__ import annotations

from typing import Callable, Sequence

# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #


def kv_cache_bytes(
    num_layers: int,
    batch: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_val: float = 2.0,
) -> float:
    """Closed-form KV-cache size in bytes (factor 2 = keys **and** values).

    ``bytes_per_val``: BF16 ⇒ 2.0, int8 ⇒ 1.0, int4 packed ⇒ 0.5, 3-bit ⇒ 0.375.
    """
    return 2.0 * num_layers * batch * seq_len * num_kv_heads * head_dim * bytes_per_val


def bytes_per_val_for_bits(bits: float) -> float:
    """Bytes per stored value for a given bit-width (e.g. 4 → 0.5)."""
    return bits / 8.0


def reset_peak_memory(device=None) -> None:
    import torch

    torch.cuda.reset_peak_memory_stats(device)


def peak_allocated_mb(device=None) -> float:
    import torch

    return torch.cuda.max_memory_allocated(device) / 1024**2


def peak_reserved_mb(device=None) -> float:
    import torch

    return torch.cuda.max_memory_reserved(device) / 1024**2


def current_allocated_mb(device=None) -> float:
    import torch

    return torch.cuda.memory_allocated(device) / 1024**2


# --------------------------------------------------------------------------- #
# Speed — CUDA events
# --------------------------------------------------------------------------- #


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of an already-sorted sequence."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def time_cuda_ms(
    fn: Callable[[], object],
    *,
    warmup: int = 10,
    reps: int = 50,
    quantiles: Sequence[float] = (0.5, 0.1, 0.9),
    reset_between: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Time ``fn`` with CUDA events; return ms quantiles.

    ``reset_between`` (optional) runs *before each timed rep* and is **not**
    timed — used to rebuild a fresh stateful cache between prefill reps.
    Keys returned: ``ms_p50``, ``ms_p10``, ``ms_p90``, ``ms_min``, ``ms_max``.
    """
    import torch

    for _ in range(warmup):
        if reset_between is not None:
            reset_between()
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(reps):
        if reset_between is not None:
            reset_between()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    samples.sort()
    out = {f"ms_p{int(q * 100):02d}": _quantile(samples, q) for q in quantiles}
    out["ms_min"] = samples[0]
    out["ms_max"] = samples[-1]
    return out


def summarize_ms(samples: Sequence[float]) -> dict[str, float]:
    """Quantile summary of a list of millisecond samples (already collected)."""
    s = sorted(samples)
    return {
        "ms_p50": _quantile(s, 0.5),
        "ms_p10": _quantile(s, 0.1),
        "ms_p90": _quantile(s, 0.9),
        "ms_min": s[0] if s else float("nan"),
        "ms_max": s[-1] if s else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #


def exact_match(a, b) -> float:
    """Fraction of equal elements between two integer tensors / sequences."""
    import torch

    a_t = torch.as_tensor(a)
    b_t = torch.as_tensor(b)
    if a_t.shape != b_t.shape:
        raise ValueError(f"exact_match shape mismatch: {a_t.shape} vs {b_t.shape}")
    if a_t.numel() == 0:
        return float("nan")
    return (a_t == b_t).float().mean().item()


def attention_kl(p_logits, q_logits, dim: int = -1) -> float:
    """Mean KL(softmax(p) ‖ softmax(q)) over all rows — for next-token or
    attention-weight distributions. ``p`` is the reference (e.g. BF16)."""
    import torch
    import torch.nn.functional as F

    p_logits = torch.as_tensor(p_logits, dtype=torch.float32)
    q_logits = torch.as_tensor(q_logits, dtype=torch.float32)
    log_p = F.log_softmax(p_logits, dim=dim)
    log_q = F.log_softmax(q_logits, dim=dim)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=dim)
    return kl.mean().item()


def mse(a, b) -> float:
    import torch

    a_t = torch.as_tensor(a, dtype=torch.float32)
    b_t = torch.as_tensor(b, dtype=torch.float32)
    return torch.mean((a_t - b_t) ** 2).item()


def cosine_similarity(a, b) -> float:
    import torch
    import torch.nn.functional as F

    a_t = torch.as_tensor(a, dtype=torch.float32).flatten()
    b_t = torch.as_tensor(b, dtype=torch.float32).flatten()
    return F.cosine_similarity(a_t, b_t, dim=0).item()


def inner_product_error(q, k, q_hat, k_hat) -> dict[str, float]:
    """Distortion of inner products ``q·k`` under quantization.

    ``q, k`` exact; ``q_hat, k_hat`` reconstructed. Returns mean **signed**
    error (bias), RMSE, and max abs error over the row-wise dot products.
    """
    import torch

    q_t = torch.as_tensor(q, dtype=torch.float32)
    k_t = torch.as_tensor(k, dtype=torch.float32)
    qh = torch.as_tensor(q_hat, dtype=torch.float32)
    kh = torch.as_tensor(k_hat, dtype=torch.float32)
    exact = (q_t * k_t).sum(dim=-1)
    approx = (qh * kh).sum(dim=-1)
    err = approx - exact
    return {
        "ip_bias": err.mean().item(),
        "ip_rmse": torch.sqrt(torch.mean(err**2)).item(),
        "ip_max_abs": err.abs().max().item(),
    }


def perplexity(model, input_ids, *, stride: int | None = None) -> float:
    """Standard causal-LM perplexity over ``input_ids`` ([1, T] or [B, T])."""
    import torch

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
        loss = out.loss
    if model_was_training:
        model.train()
    return torch.exp(loss).item()
