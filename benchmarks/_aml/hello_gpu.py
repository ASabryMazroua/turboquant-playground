"""M0 smoke test: confirm the job is on a single A100 80GB with a working CUDA + Triton stack.

Prints a JSON report and exits non-zero if the HARD gates fail (CUDA available and
an A100 80GB visible) so a misplaced job is marked Failed in AML. Triton and the
Nsight tools are reported as warnings (needed later in M4/M5) but do not by
themselves fail the smoke test — we record their status either way.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (out.stdout or "") + (out.stderr or "")
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"<error: {exc!r}>"


def collect() -> dict:
    info: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            info["total_mem_gb"] = round(props.total_memory / 1024**3, 1)
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            info["cuda_runtime"] = torch.version.cuda
    except Exception as exc:
        info["torch_error"] = repr(exc)

    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as exc:
        info["triton_error"] = repr(exc)

    info["nsys"] = shutil.which("nsys")
    info["ncu"] = shutil.which("ncu")
    info["nvidia_smi"] = _run(["nvidia-smi"])
    return info


def main() -> int:
    info = collect()

    # Optional micro-sanity: a tiny bf16 matmul timed with CUDA events.
    try:
        import torch

        if torch.cuda.is_available():
            a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            for _ in range(3):  # warmup
                _ = a @ b
            torch.cuda.synchronize()
            start.record()
            _ = a @ b
            end.record()
            torch.cuda.synchronize()
            info["bf16_matmul_ms"] = round(start.elapsed_time(end), 4)
    except Exception as exc:
        info["matmul_error"] = repr(exc)

    print(json.dumps({k: v for k, v in info.items() if k != "nvidia_smi"}, indent=2))
    print("\n----- nvidia-smi -----")
    print(info.get("nvidia_smi"))

    name = str(info.get("device_name", ""))
    is_a100 = "A100" in name
    is_80gb = (info.get("total_mem_gb") or 0) >= 79.0
    hard_ok = bool(info.get("cuda_available")) and is_a100 and is_80gb

    print("\n----- M0 gates -----")
    print(f"  CUDA available : {info.get('cuda_available')}")
    print(f"  A100 detected  : {is_a100}  ({name})")
    print(f"  80GB memory    : {is_80gb}  ({info.get('total_mem_gb')} GB)")
    print(f"  Triton import  : {'triton' in info}  ({info.get('triton', info.get('triton_error'))})")
    print(f"  nsys present   : {bool(info.get('nsys'))}")
    print(f"  ncu present    : {bool(info.get('ncu'))}")
    print(f"\nRESULT: {'PASS' if hard_ok else 'FAIL'} (hard gates)")
    return 0 if hard_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
