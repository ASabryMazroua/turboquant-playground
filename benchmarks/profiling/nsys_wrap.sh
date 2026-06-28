#!/usr/bin/env bash
# Nsight Systems capture wrapper (profiler tool for M4/M5).
# Produces a system timeline (kernels, CUDA API, memcpy, NVTX ranges) under
# results/traces/. Use to confirm fused int4 kernels replaced the dequant->matmul
# sequence and that there is one inverse-rotation per head, not per token.
#
# Usage: benchmarks/profiling/nsys_wrap.sh <out_name> <python ...args>
set -euo pipefail
OUT="${1:?usage: nsys_wrap.sh <out_name> <cmd...>}"; shift
mkdir -p results/traces
nsys profile -t cuda,nvtx,osrt \
  -o "results/traces/${OUT}" --force-overwrite true \
  "$@"
echo "nsys report -> results/traces/${OUT}.nsys-rep"
