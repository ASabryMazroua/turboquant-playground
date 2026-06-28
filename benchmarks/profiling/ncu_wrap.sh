#!/usr/bin/env bash
# Nsight Compute per-kernel roofline wrapper (profiler tool for M5).
# Captures achieved DRAM GB/s, occupancy, warp-stall reasons and a roofline for
# the fused int4 kernels — the evidence that the kernels are memory-bound and how
# close to A100 peak HBM bandwidth they get.
#
# Usage: benchmarks/profiling/ncu_wrap.sh <out_name> <kernel_regex> <python ...args>
# Example kernel_regex: int4_(logits|values)
set -euo pipefail
OUT="${1:?usage: ncu_wrap.sh <out_name> <kernel_regex> <cmd...>}"; shift
KRE="${1:?missing <kernel_regex>}"; shift
mkdir -p results/traces
ncu --set roofline -k "regex:${KRE}" \
  -o "results/traces/${OUT}" --force-overwrite \
  "$@"
echo "ncu report -> results/traces/${OUT}.ncu-rep"
