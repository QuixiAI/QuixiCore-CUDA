#!/usr/bin/env bash
#
# Shared core of every backend's perf/harness/run_bench.sh.
#
# SYNCED COPY — the canonical file is tools/templates/run_bench_core.sh in the
# QuixiCore umbrella repo, distributed by tools/sync_perf_tooling.py. Edit it
# there, never here; tools/fleet_check.sh flags drift.
#
# The wrapper (run_bench.sh) sets QC_BACKEND and REPO_ROOT, defines
# qc_bench_cmd (builds the backend's runner argv and calls qc_exec with it)
# and optionally qc_device_info (prints `key=value` provenance lines) and
# qc_out_dir (overrides the run directory for run-id-style runners), then
# sources this file.
#
# Behavior, generalized from QuixiCore-ROCm's run_kernel_bench.sh:
#   * one run directory per invocation: perf/results/<date>/<label>/
#   * everything the harness prints is tee'd to bench.log
#   * provenance.json records git/host/container/command mechanically
#   * verdict guard: perf_diff.py validate must accept the run, otherwise
#     "do NOT record this as a measured win" (QC_VERDICT_MODE=stdout falls
#     back to grepping bench.log for an ALL PASS line)
#   * noise guard: validate warns on cv > QC_CV_LIMIT or spread >
#     QC_SPREAD_LIMIT (contention has already inverted one result fleet-wide)
#   * a pre-filled optimization_status.md entry is printed for review
#
# Flags: --label L  --preset P  --kernel K  --dry-run  --no-guards  -- <rest>

set -euo pipefail

QC_LABEL=""
QC_PRESET="${QC_PRESET:-}"
QC_KERNELS="${QC_KERNELS:-}"
QC_DRY_RUN=0
QC_NO_GUARDS=0
QC_CV_LIMIT="${QC_CV_LIMIT:-0.50}"
QC_SPREAD_LIMIT="${QC_SPREAD_LIMIT:-1.20}"
QC_PASSTHROUGH=()

qc_usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)   qc_usage 0 ;;
        --label)     QC_LABEL="${2:?--label needs a value}"; shift 2 ;;
        --preset)    QC_PRESET="${2:?--preset needs a value}"; shift 2 ;;
        --kernel)    QC_KERNELS="${2:?--kernel needs a value}"; shift 2 ;;
        --dry-run)   QC_DRY_RUN=1; shift ;;
        --no-guards) QC_NO_GUARDS=1; shift ;;
        --)          shift; QC_PASSTHROUGH=("$@"); break ;;
        *)           echo "unknown option: $1" >&2; qc_usage 2 ;;
    esac
done

[ -n "${REPO_ROOT:-}" ] || { echo "wrapper must set REPO_ROOT" >&2; exit 2; }
[ -n "${QC_BACKEND:-}" ] || { echo "wrapper must set QC_BACKEND" >&2; exit 2; }
HARNESS_DIR="$REPO_ROOT/perf/harness"
QC_DATE="$(date +%F)"
[ -n "$QC_LABEL" ] || QC_LABEL="${QC_BACKEND}-bench"

if declare -F qc_out_dir >/dev/null; then
    OUT_DIR="$(qc_out_dir "$QC_LABEL" "$QC_DATE")"
else
    OUT_DIR="$REPO_ROOT/perf/results/$QC_DATE/$QC_LABEL"
    n=2
    while [ -e "$OUT_DIR" ]; do
        OUT_DIR="$REPO_ROOT/perf/results/$QC_DATE/$QC_LABEL-$n"
        n=$((n + 1))
    done
fi

# qc_exec: the wrapper's qc_bench_cmd calls this with the full runner argv.
qc_exec() {
    if [ "$QC_DRY_RUN" -eq 1 ]; then
        echo "dry-run: would execute: $*"
        echo "dry-run: run directory:  $OUT_DIR"
        return 0
    fi
    mkdir -p "$OUT_DIR"
    QC_CMDLINE="$*"
    qc_write_provenance
    set +e
    "$@" 2>&1 | tee "$OUT_DIR/bench.log"
    QC_STATUS="${PIPESTATUS[0]}"
    set -e
}

qc_write_provenance() {
    local git_commit dirty devinfo
    git_commit="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    dirty=""
    # --porcelain also reports untracked files; a new-but-uncommitted kernel
    # is exactly the case where provenance matters most.
    if [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]; then
        dirty="-dirty"
    fi
    devinfo=""
    if declare -F qc_device_info >/dev/null; then
        devinfo="$(qc_device_info 2>/dev/null || true)"
    fi
    QC_GIT="$git_commit$dirty" QC_HOST="$(hostname)" \
    QC_STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    QC_CONTAINER="${QC_CONTAINER:-bare metal (no container)}" \
    QC_CMDLINE="$QC_CMDLINE" QC_DEVINFO="$devinfo" QC_BACKEND="$QC_BACKEND" \
    python3 - "$OUT_DIR/provenance.json" <<'PY'
import json, os, sys
info = {
    "backend": os.environ["QC_BACKEND"],
    "git": os.environ["QC_GIT"],
    "hostname": os.environ["QC_HOST"],
    "timestamp": os.environ["QC_STAMP"],
    "container": os.environ["QC_CONTAINER"],
    "command": os.environ["QC_CMDLINE"],
}
for line in os.environ.get("QC_DEVINFO", "").splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        info[k.strip()] = v.strip()
with open(sys.argv[1], "w") as f:
    json.dump(info, f, indent=2)
    f.write("\n")
PY
}

QC_STATUS=0
qc_bench_cmd

if [ "$QC_DRY_RUN" -eq 1 ]; then
    exit 0
fi

GUARD_FAILED=0
if [ "$QC_NO_GUARDS" -eq 0 ]; then
    if [ "${QC_VERDICT_MODE:-jsonl}" = "stdout" ]; then
        # Harness prints its own verdict; a missing verdict is a failure even
        # when the runner exits 0, so a harness that silently skips its checks
        # cannot masquerade as a passing run.
        if ! grep -q '^ALL PASS' "$OUT_DIR/bench.log" 2>/dev/null; then
            echo "!! no ALL PASS verdict in bench.log — do NOT record this as a measured win." >&2
            GUARD_FAILED=1
        fi
    else
        if ! python3 "$HARNESS_DIR/perf_diff.py" validate "$OUT_DIR" \
            --cv-limit "$QC_CV_LIMIT" --spread-limit "$QC_SPREAD_LIMIT"; then
            echo "!! validation failed — do NOT record this as a measured win." >&2
            GUARD_FAILED=1
        fi
    fi
fi

if [ "$QC_STATUS" -ne 0 ]; then
    echo "!! harness exited $QC_STATUS — do NOT record this as a measured win." >&2
fi

echo >&2
echo "Raw output: ${OUT_DIR#"$REPO_ROOT"/}" >&2
if [ -f "$OUT_DIR/results.jsonl" ]; then
    echo "--- paste into perf/optimization_status.md, then fill the <> fields ---"
    python3 "$HARNESS_DIR/perf_diff.py" entry "$OUT_DIR" || true
fi

if [ "$QC_STATUS" -ne 0 ]; then exit "$QC_STATUS"; fi
exit "$GUARD_FAILED"
