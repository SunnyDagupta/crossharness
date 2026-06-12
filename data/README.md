# Phase 2a run data

| Field | Value |
|-------|-------|
| Date | 2026-06-11 |
| Model | hermes3:8b (Q4_0, 8.0B params, 131072-ctx) |
| Serving | Ollama 0.30.6, CPU, localhost:11434 |
| Harness | crossharness @ 8dc769d + stream:false patch (see eval_live.py) |
| Tasks | 24 (phase2a/tasks.py) |
| Conditions | none / syntactic / full |
| Samples | 5 per (task, condition) |
| Total rows | 360 |

## Files

- `hermes3-8b-q4-ollama-20260611.jsonl` — 360 rows (355 from main run +
  5 gap-fill for m5_assemble/full; gap-fill required because Ollama evicted
  the model on cell 356 after idle timeout; gap cells re-run blind, not
  re-rolled until success)

## Reproducing

```
OLLAMA_KEEP_ALIVE=24h ollama serve          # prevent idle eviction
python3 eval_live.py --samples 5 \
    --base-url http://localhost:11434/v1 \
    --model hermes3:8b
python3 analyze.py data/hermes3-8b-q4-ollama-20260611.jsonl
```

## Schema

Each line is a JSON object:

| Field | Type | Description |
|-------|------|-------------|
| task | string | task name (e.g. `w1_status`) |
| family | string | W / E / L / M |
| condition | string | none / syntactic / full |
| passed | bool | whether `task.check()` returned True |
| turns | int | conversation turns used |
| calls | list[str] | classified call types per turn |
| malformed_turns | int | turns where parse failed |
| model | string | model identifier |
