"""CUDA memory-history snapshot helper (profiler tool for M1/M3).

Wrap a workload to capture the PyTorch allocator timeline, then view the dumped
pickle at https://pytorch.org/memory_viz to show the KV cache dominating memory
and shrinking under int4 quantization.

As a context manager:
    from benchmarks.profiling.mem_snapshot import record_memory
    with record_memory("results/traces/decode_mem.pickle"):
        model.generate(...)
"""
from __future__ import annotations

import contextlib
from pathlib import Path


@contextlib.contextmanager
def record_memory(out_path: str | Path, max_entries: int = 100_000):
    """Record CUDA allocator history for the duration of the block and dump it."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("record_memory requires CUDA")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    try:
        yield
    finally:
        torch.cuda.memory._dump_snapshot(str(out_path))
        # Disable history recording again to avoid overhead afterwards.
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"memory snapshot -> {out_path}  (view at pytorch.org/memory_viz)")


def _demo() -> None:
    import torch

    with record_memory("results/traces/demo_mem.pickle"):
        xs = [torch.randn(1024, 1024, device="cuda") for _ in range(8)]
        _ = sum(x.sum() for x in xs)


if __name__ == "__main__":
    _demo()
