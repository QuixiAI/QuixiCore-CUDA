#!/usr/bin/env bash
#
# QuixiCore CUDA bench entrypoint. Thin wrapper around the shared core
# (run_bench_core.sh, synced from the umbrella); this file is hand-written
# and backend-owned.
#
#   perf/harness/run_bench.sh --kernel layernorm --label ln-ab
#   perf/harness/run_bench.sh --dry-run
#
# Wraps the registry-driven perf/bench_kernels.py run phase. Know its real
# coverage before trusting it (see perf/perf.md, Measurement Harness): most
# kernel families are benchmarked by their standalone .out harnesses instead.
# SCAFFOLDING NOTE: this entrypoint has not yet been executed on a CUDA host;
# verify it there before treating its output as evidence.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QC_BACKEND="cuda"
QC_PYTHON="${QC_PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$QC_PYTHON" ] || QC_PYTHON="python3"

qc_out_dir() {
    # bench_kernels.py owns the run directory: perf/results/<date>/<run-id>
    echo "$REPO_ROOT/perf/results/$2/$1"
}

qc_bench_cmd() {
    qc_exec "$QC_PYTHON" "$REPO_ROOT/perf/bench_kernels.py" \
        --phase run --kernel "${QC_KERNELS:-all}" \
        --run-id "$(basename "$OUT_DIR")" \
        ${QC_PASSTHROUGH[@]+"${QC_PASSTHROUGH[@]}"}
}

qc_device_info() {
    command -v nvidia-smi >/dev/null 2>&1 && \
        echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
    command -v nvcc >/dev/null 2>&1 && echo "nvcc=$(nvcc --version 2>/dev/null | grep release)"
    echo "uname=$(uname -srm)"
}

source "$(dirname "${BASH_SOURCE[0]}")/run_bench_core.sh"
