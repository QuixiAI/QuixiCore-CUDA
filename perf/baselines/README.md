# Committed Baselines

Curated benchmark baselines, one directory per host fingerprint:

```
perf/baselines/<host-fingerprint>/
    results.jsonl    # curated schema-1 rows (docs/benchmarking.md, umbrella)
    run.json         # provenance of the run they came from
```

The fingerprint is `slug(backend)-slug(device|cpu_model)-slug(arch)`, printed
by `python3 perf/harness/perf_diff.py fingerprint <run-dir>`. Baselines from
different fingerprints are never comparable; `perf_diff.py gate` refuses to
cross them.

## Refresh procedure

1. On an idle host, run the entrypoint at the backend's standard preset:
   `perf/harness/run_bench.sh --label baseline-refresh`
2. Inspect the run (`perf_diff.py validate`, eyeball the summary).
3. `python3 perf/harness/perf_diff.py promote perf/results/<date>/baseline-refresh`
   — promote filters to completed, correctness-checked, low-noise rows and
   refuses runs with failures.
4. Commit `perf/baselines/<fingerprint>/` with a message stating the host and
   the exact command line.

A refresh replaces the whole fingerprint directory; partial merges are not
supported. Baseline captures are measurements — they follow the performance
gate in `AGENTS.md` (idle host, completed run, recorded in the notebook).

No baseline has been captured yet for this backend.
