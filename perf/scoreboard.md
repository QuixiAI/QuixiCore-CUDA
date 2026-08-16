# CUDA Scoreboard

The backend's aggregate score lineage: the geometric mean of median latencies
(ms) across a run's timed rows, recorded at each promoted best. Lower is
better; scores are comparable only within one host fingerprint and row set,
so keep the row set stable (the standard preset) and note when it changes.
This is the loop's one hill-climbable number. Record with:

```bash
python3 perf/harness/perf_diff.py score <run-dir> \
    --record perf/scoreboard.md --note "<what changed>"
```

| date | score (geomean ms) | rows | host | git | what changed |
|---|---|---|---|---|---|
No scores recorded yet.
