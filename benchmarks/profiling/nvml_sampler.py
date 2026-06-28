"""NVML utilization sampler (profiler tool for M0/M1/M4).

Samples GPU SM%, memory%, memory used, power, SM clock and temperature at a fixed
rate and writes a CSV usable by ``turbo_kv.reporting`` for the "GPU under-utilized
during decode" plot. Pure NVML sampler — no kernel-launch overhead, so it can run
alongside the timed workload without perturbing the headline numbers.

Usage:
    python benchmarks/profiling/nvml_sampler.py --hz 10 --duration 60 \
        --out results/traces/nvml.csv
Stop early with Ctrl+C; the CSV is flushed continuously.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

try:
    import pynvml  # provided by the `nvidia-ml-py` package
except ImportError:  # pragma: no cover - environment dependent
    pynvml = None

FIELDS = [
    "t_s",
    "gpu_index",
    "util_gpu_pct",
    "util_mem_pct",
    "mem_used_mb",
    "mem_total_mb",
    "power_w",
    "sm_clock_mhz",
    "temp_c",
]


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample GPU stats via NVML into a CSV.")
    ap.add_argument("--hz", type=float, default=10.0, help="samples per second")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds (0 = until Ctrl+C)")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index to sample")
    ap.add_argument("--out", type=Path, default=Path("results/traces/nvml.csv"))
    args = ap.parse_args()

    if pynvml is None:
        raise SystemExit("pynvml not installed; `pip install nvidia-ml-py`")

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    period = 1.0 / args.hz if args.hz > 0 else 0.1
    start = time.perf_counter()
    n = 0
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        try:
            while True:
                now = time.perf_counter()
                util = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))
                mem = _safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
                writer.writerow(
                    {
                        "t_s": round(now - start, 4),
                        "gpu_index": args.gpu,
                        "util_gpu_pct": getattr(util, "gpu", None),
                        "util_mem_pct": getattr(util, "memory", None),
                        "mem_used_mb": round(mem.used / 1024**2, 1) if mem else None,
                        "mem_total_mb": round(mem.total / 1024**2, 1) if mem else None,
                        "power_w": _safe(
                            lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        ),
                        "sm_clock_mhz": _safe(
                            lambda: pynvml.nvmlDeviceGetClockInfo(
                                handle, pynvml.NVML_CLOCK_SM
                            )
                        ),
                        "temp_c": _safe(
                            lambda: pynvml.nvmlDeviceGetTemperature(
                                handle, pynvml.NVML_TEMPERATURE_GPU
                            )
                        ),
                    }
                )
                fh.flush()
                n += 1
                if args.duration and (now - start) >= args.duration:
                    break
                sleep = period - (time.perf_counter() - now)
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            pass
    pynvml.nvmlShutdown()
    print(f"wrote {n} samples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
