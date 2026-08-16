#!/usr/bin/env python3
"""Validate, compare, and gate QuixiCore benchmark results (schema 1).

The reporting format is defined in docs/benchmarking.md: a run directory
`perf/results/YYYY-MM-DD/<run-id>/` containing `run.json` (environment) and
`results.jsonl` (one JSON row per case). This tool is the mechanical half of
the decision rules the backend handbooks state as prose: the >=3% / >=8%
keep bars, the noise guard, and the rule that results from different hosts
are never comparable.

Subcommands:
  validate RUN            schema check + verdict/noise flags for one run
  diff BASELINE CANDIDATE compare two runs row-by-row
  gate RUN                diff RUN against the repo's committed baseline for
                          this host fingerprint (perf/baselines/<fp>/)
  promote RUN             curate RUN into perf/baselines/<fp>/ (only
                          subcommand that writes)
  entry RUN               print a pre-filled optimization_status.md skeleton
  fingerprint RUN         print the host fingerprint derived from run.json

RUN is a run directory or a bare results.jsonl path. Exit codes: 0 clean,
1 regression or validation failure, 2 usage/schema error, 3 no baseline for
this fingerprint, 4 host-fingerprint mismatch.

Copies of this tool are synced into each backend's perf/harness/ by the
umbrella's tools/sync_perf_tooling.py so standalone clones work; edit the
umbrella copy, never a synced one.
"""

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = 1
REQUIRED_ROW = ["kernel", "variant", "shape", "dtype", "status"]
STATUSES = {"ok", "skip", "fail"}
DEFAULT_CV_LIMIT = 0.50
DEFAULT_SPREAD_LIMIT = 1.20

# Mirror of registry/tolerances.yaml (rtol). The registry is the source of
# truth; this copy exists so synced backend copies work standalone.
TOLERANCES = {"fp32": 1e-5, "fp16": 1e-3, "bf16": 2e-3, "fp8": 2e-2, "quantized": 3e-2}


def fail(msg, code=2):
    print(f"perf_diff: {msg}", file=sys.stderr)
    sys.exit(code)


def slug(text):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]", "-", str(text).lower())).strip("-")


def load_run(arg):
    """Return (rows, run_meta, run_dir_or_None, label)."""
    p = Path(arg)
    if p.is_dir():
        jl = p / "results.jsonl"
        rj = p / "run.json"
        if not jl.exists():
            fail(f"{p}: no results.jsonl")
        meta = json.loads(rj.read_text()) if rj.exists() else {}
        run_dir = p
    elif p.is_file():
        jl = p
        rj = p.parent / "run.json"
        meta = json.loads(rj.read_text()) if rj.exists() else {}
        run_dir = p.parent
    else:
        fail(f"{arg}: not found")
    rows = []
    for n, line in enumerate(jl.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            fail(f"{jl}:{n}: bad JSON ({e})")
    return rows, meta, run_dir, str(arg)


def fingerprint(meta):
    backend = meta.get("backend") or "unknown"
    device = meta.get("device") or meta.get("cpu_model")
    arch = meta.get("arch")
    if not arch:
        platform = meta.get("platform", "")
        for cand in ("arm64", "x86_64", "aarch64"):
            if cand in platform:
                arch = cand
                break
    return f"{slug(backend)}-{slug(device) if device else 'unknown-host'}-{slug(arch) if arch else 'unknown'}"


def row_key(row):
    shape = json.dumps(row.get("shape", {}), sort_keys=True)
    return (row.get("kernel"), row.get("variant"), shape, row.get("dtype"), row.get("format") or "")


def variance(row):
    """Return (cv|None, spread|None)."""
    cv = row.get("target_cv")
    spread = row.get("target_spread")
    if spread is None:
        lo, hi = row.get("target_min_ms"), row.get("target_max_ms")
        if lo and hi and lo > 0:
            spread = hi / lo
    return cv, spread


def noisy(row, cv_limit, spread_limit):
    cv, spread = variance(row)
    if cv is not None and cv > cv_limit:
        return f"cv {cv:.2f} > {cv_limit}"
    if spread is not None and spread > spread_limit:
        return f"spread {spread:.2f}x > {spread_limit}x"
    return None


def tolerance_for(row):
    if row.get("format"):
        return "quantized", TOLERANCES["quantized"]
    dtype = str(row.get("dtype", "")).lower()
    if dtype in ("f32", "fp32", "float32"):
        return "fp32", TOLERANCES["fp32"]
    if dtype in ("f16", "fp16", "half", "float16"):
        return "fp16", TOLERANCES["fp16"]
    if dtype == "bf16":
        return "bf16", TOLERANCES["bf16"]
    if dtype.startswith("fp8") or dtype.startswith("e4m3") or dtype.startswith("e5m2"):
        return "fp8", TOLERANCES["fp8"]
    return None, None


def validate_rows(rows, meta, label, strict, cv_limit, spread_limit):
    """Returns (errors, warnings)."""
    errors, warnings = [], []
    if not rows:
        errors.append("results.jsonl is empty — no evidence, nothing to record")
    for i, row in enumerate(rows, 1):
        schema = row.get("schema", row.get("schema_version"))
        if schema != SCHEMA:
            errors.append(f"row {i}: schema {schema!r} != {SCHEMA}")
            continue
        for field in REQUIRED_ROW:
            if field not in row:
                errors.append(f"row {i}: missing required field {field!r}")
        status = row.get("status")
        if status not in STATUSES:
            errors.append(f"row {i}: status {status!r} not in {sorted(STATUSES)}")
        if status == "fail":
            errors.append(f"row {i} ({row.get('variant')}): status=fail")
        if row.get("check_passed") is False:
            errors.append(f"row {i} ({row.get('variant')}): check_passed=false")
        if status == "ok":
            if row.get("target_ms") is None:
                throughput = any(
                    row.get(k) for k in
                    ("gflops", "tflops", "tops", "gbps", "weight_gbps", "measurements")
                )
                correctness_only = row.get("check_passed") is True
                if throughput or correctness_only:
                    kind = "throughput-only" if throughput else "correctness-only"
                    warnings.append(
                        f"row {i} ({row.get('variant')}): ok row lacks target_ms"
                        f" ({kind} row; cannot be gated)"
                    )
                else:
                    errors.append(f"row {i} ({row.get('variant')}): ok row lacks target_ms")
            cv, spread = variance(row)
            if cv is None and spread is None:
                (errors if strict else warnings).append(
                    f"row {i} ({row.get('variant')}): no variance group "
                    "(target_p20/p80/cv or target_min/max/spread)"
                )
            reason = noisy(row, cv_limit, spread_limit)
            if reason:
                warnings.append(
                    f"row {i} ({row.get('variant')}): NOISY ({reason}) — not usable "
                    "for a keep/reject decision; re-run on an idle host"
                )
            name, rtol = tolerance_for(row)
            err = row.get("max_rel_err")
            if name and err is not None and err > rtol:
                (errors if strict else warnings).append(
                    f"row {i} ({row.get('variant')}): max_rel_err {err:.3g} exceeds "
                    f"{name} rtol {rtol:g} (registry/tolerances.yaml)"
                )
    # run.json checks: errors only under --strict so pre-schema runs validate
    for field in ("schema", "backend", "repo", "contract", "git", "timestamp", "warmup", "iters"):
        if field not in meta:
            (errors if strict else warnings).append(f"run.json: missing {field!r}")
    if fingerprint(meta).endswith("unknown-host-unknown"):
        (errors if strict else warnings).append("run.json: no host identity (device/cpu_model)")
    return errors, warnings


def cmd_validate(args):
    rows, meta, _, label = load_run(args.run)
    errors, warnings = validate_rows(
        rows, meta, label, args.strict, args.cv_limit, args.spread_limit
    )
    for w in warnings:
        print(f"WARN {w}")
    for e in errors:
        print(f"FAIL {e}")
    if errors:
        print(
            f"perf_diff validate: {label}: {len(errors)} failure(s) — this run is "
            "not evidence; do NOT record it as a measured win"
        )
        return 1
    ok = sum(1 for r in rows if r.get("status") == "ok")
    skip = sum(1 for r in rows if r.get("status") == "skip")
    print(f"perf_diff validate: {label}: {ok} ok, {skip} skip, {len(warnings)} warning(s)")
    return 0


def check_hosts(meta_a, meta_b, allow):
    fa, fb = fingerprint(meta_a), fingerprint(meta_b)
    if fa != fb or "unknown-host" in fa:
        msg = f"NOT COMPARABLE ACROSS HOSTS: baseline={fa} candidate={fb}"
        if not allow:
            print(msg)
            sys.exit(4)
        print(f"WARN {msg} (--allow-cross-host: reporting only, not gating)")
        return False
    return True


def diff_runs(base_rows, cand_rows, bar_low, bar_high, cv_limit, spread_limit):
    base = {row_key(r): r for r in base_rows if r.get("status") == "ok"}
    cand = {row_key(r): r for r in cand_rows if r.get("status") == "ok"}
    results = []
    for key in base:
        if key in cand:
            b, c = base[key], cand[key]
            if b.get("target_ms") is None or c.get("target_ms") is None:
                results.append((key, b, c, None, "untimed"))
                continue
            delta = (c["target_ms"] - b["target_ms"]) / b["target_ms"]
            reason = noisy(b, cv_limit, spread_limit) or noisy(c, cv_limit, spread_limit)
            if reason:
                verdict = "noisy"
            elif delta <= -bar_high:
                verdict = "improved**"
            elif delta <= -bar_low:
                verdict = "improved"
            elif delta >= bar_high:
                verdict = "regressed**"
            elif delta >= bar_low:
                verdict = "regressed"
            else:
                verdict = "neutral"
            results.append((key, b, c, delta, verdict))
        else:
            results.append((key, base[key], None, None, "only-in-baseline"))
    for key in cand:
        if key not in base:
            results.append((key, None, cand[key], None, "only-in-candidate"))
    return results


def print_diff(results, as_json):
    if as_json:
        out = []
        for key, b, c, delta, verdict in results:
            out.append({
                "kernel": key[0], "variant": key[1], "shape": json.loads(key[2]),
                "dtype": key[3], "format": key[4],
                "baseline_ms": b["target_ms"] if b else None,
                "candidate_ms": c["target_ms"] if c else None,
                "delta_pct": round(delta * 100, 2) if delta is not None else None,
                "verdict": verdict,
            })
        counts = summarize(results)
        print(json.dumps({"rows": out, "summary": counts}, indent=2))
        return
    width = max((len(f"{k[0]} · {k[1]}") for k, *_ in results), default=20)
    print(f"{'case':<{width}}  {'base ms':>10}  {'cand ms':>10}  {'Δ%':>7}  verdict")
    for key, b, c, delta, verdict in sorted(results, key=lambda r: (r[0][0], r[0][1])):
        case = f"{key[0]} · {key[1]}"
        bms = f"{b['target_ms']:.4f}" if b else "-"
        cms = f"{c['target_ms']:.4f}" if c else "-"
        d = f"{delta * 100:+.1f}" if delta is not None else "-"
        print(f"{case:<{width}}  {bms:>10}  {cms:>10}  {d:>7}  {verdict}")


def summarize(results):
    counts = {"compared": 0, "improved": 0, "regressed": 0, "neutral": 0,
              "noisy": 0, "unmatched": 0, "untimed": 0}
    for _, b, c, delta, verdict in results:
        if verdict.startswith("only-in"):
            counts["unmatched"] += 1
            continue
        if verdict == "untimed":
            counts["untimed"] += 1
            continue
        counts["compared"] += 1
        if verdict.startswith("improved"):
            counts["improved"] += 1
        elif verdict.startswith("regressed"):
            counts["regressed"] += 1
        elif verdict == "noisy":
            counts["noisy"] += 1
        else:
            counts["neutral"] += 1
    return counts


def run_diff(base_arg, cand_arg, args, gating):
    base_rows, base_meta, _, _ = load_run(base_arg)
    cand_rows, cand_meta, _, _ = load_run(cand_arg)
    comparable = check_hosts(base_meta, cand_meta, args.allow_cross_host)
    results = diff_runs(
        base_rows, cand_rows, args.bar_low, args.bar_high, args.cv_limit, args.spread_limit
    )
    print_diff(results, args.json)
    counts = summarize(results)
    if not args.json:
        line = (
            f"perf_diff: {counts['compared']} compared, {counts['improved']} improved"
            f"(>={args.bar_low:.0%}), {counts['regressed']} regressed(>={args.bar_low:.0%}), "
            f"{counts['noisy']} noisy, {counts['unmatched']} unmatched"
        )
        if counts["untimed"]:
            line += f", {counts['untimed']} untimed"
        print(line)
    if gating and comparable and counts["regressed"]:
        return 1
    return 0


def cmd_diff(args):
    return run_diff(args.baseline, args.candidate, args, gating=True)


def baselines_dir(run_dir, override):
    if override:
        return Path(override)
    # walk up from the run dir to the repo's perf/ directory
    p = run_dir.resolve()
    for parent in [p] + list(p.parents):
        if parent.name == "perf":
            return parent / "baselines"
    fail(f"cannot locate perf/ above {run_dir}; pass --baselines")


def cmd_gate(args):
    rows, meta, run_dir, _ = load_run(args.run)
    fp = fingerprint(meta)
    bdir = baselines_dir(run_dir, args.baselines) / fp
    if not (bdir / "results.jsonl").exists():
        print(f"perf_diff gate: no committed baseline for host {fp} under {bdir.parent}")
        return 3
    return run_diff(str(bdir), args.run, args, gating=True)


def cmd_promote(args):
    rows, meta, run_dir, label = load_run(args.run)
    errors, _ = validate_rows(rows, meta, label, False, args.cv_limit, args.spread_limit)
    if errors:
        for e in errors:
            print(f"FAIL {e}")
        print("perf_diff promote: refusing to promote a run that fails validation")
        return 1
    curated, dropped = [], 0
    seen = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("check_passed") is False:
            dropped += 1
            continue
        if noisy(row, args.cv_limit, args.spread_limit):
            dropped += 1
            continue
        seen[row_key(row)] = row  # latest wins
    curated = list(seen.values())
    if not curated:
        print("perf_diff promote: nothing survives curation; not writing")
        return 1
    fp = fingerprint(meta)
    if "unknown-host" in fp:
        fail("run.json has no host identity; cannot name a baseline directory", 1)
    bdir = baselines_dir(run_dir, args.baselines) / fp
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in curated))
    (bdir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"perf_diff promote: wrote {len(curated)} row(s) ({dropped} dropped) to {bdir}")
    print("commit this directory with a message stating the host and command line")
    return 0


def cmd_entry(args):
    rows, meta, run_dir, _ = load_run(args.run)
    prov = {}
    if run_dir and (run_dir / "provenance.json").exists():
        prov = json.loads((run_dir / "provenance.json").read_text())
    date = str(meta.get("timestamp", ""))[:10] or "YYYY-MM-DD"
    kernels = sorted({r.get("kernel", "?") for r in rows})
    env_bits = []
    for k in ("backend", "device", "cpu_model", "git", "warmup", "iters", "preset"):
        if meta.get(k) is not None:
            env_bits.append(f"{k}={meta[k]}")
    for k in ("gpu", "container", "hostname"):
        if prov.get(k):
            env_bits.append(f"{k}={prov[k]}")
    print(f"## {date}: {', '.join(kernels)} — <VERDICT: KEPT|REJECTED|INCONCLUSIVE|...>")
    print()
    print("Status: experimenting")
    print("Current implementation: <>")
    print("Hypothesis: <>")
    print(f"Environment: {'; '.join(env_bits) if env_bits else '<>'}")
    print("Correctness: "
          + (f"max_rel_err {max((r.get('max_rel_err') or 0) for r in rows):.3g} across "
             f"{sum(1 for r in rows if r.get('status') == 'ok')} ok rows"
             if rows else "<>"))
    print("Baseline:")
    print()
    print("| case | median ms | baseline ms | speedup |")
    print("|---|---:|---:|---:|")
    for r in rows:
        if r.get("status") != "ok":
            continue
        base = next(iter((r.get("baselines") or {}).values()), {})
        print(
            f"| {r.get('kernel')} · {r.get('variant')} | {r.get('target_ms'):.4f} | "
            f"{base.get('ms', float('nan')):.4f} | {base.get('speedup', float('nan')):.2f}x |"
        )
    print()
    print("Experiments: <one factor per row: change, before/after, keep/reject + why>")
    print("Decision: <>")
    print("Open questions: <>")
    print(f"Raw results: {run_dir}")
    return 0


def cmd_fingerprint(args):
    _, meta, _, _ = load_run(args.run)
    print(fingerprint(meta))
    return 0


def add_common(p):
    p.add_argument("--cv-limit", type=float, default=DEFAULT_CV_LIMIT)
    p.add_argument("--spread-limit", type=float, default=DEFAULT_SPREAD_LIMIT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate")
    v.add_argument("run")
    v.add_argument("--strict", action="store_true")
    add_common(v)
    v.set_defaults(func=cmd_validate)

    d = sub.add_parser("diff")
    d.add_argument("baseline")
    d.add_argument("candidate")
    d.add_argument("--bar-low", type=float, default=0.03)
    d.add_argument("--bar-high", type=float, default=0.08)
    d.add_argument("--allow-cross-host", action="store_true")
    d.add_argument("--json", action="store_true")
    add_common(d)
    d.set_defaults(func=cmd_diff)

    g = sub.add_parser("gate")
    g.add_argument("run")
    g.add_argument("--baselines")
    g.add_argument("--bar-low", type=float, default=0.03)
    g.add_argument("--bar-high", type=float, default=0.08)
    g.add_argument("--allow-cross-host", action="store_true")
    g.add_argument("--json", action="store_true")
    add_common(g)
    g.set_defaults(func=cmd_gate)

    pr = sub.add_parser("promote")
    pr.add_argument("run")
    pr.add_argument("--baselines")
    add_common(pr)
    pr.set_defaults(func=cmd_promote)

    e = sub.add_parser("entry")
    e.add_argument("run")
    e.set_defaults(func=cmd_entry)

    f = sub.add_parser("fingerprint")
    f.add_argument("run")
    f.set_defaults(func=cmd_fingerprint)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
