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

- `hermes3-8b-q4-ollama-20260611.jsonl` — 360 rows, advertisement=full
  (355 from main run + 5 gap-fill for m5_assemble/full; gap-fill required
  because Ollama evicted the model on cell 356 after idle timeout; gap
  cells re-run blind, not re-rolled until success)

### Advertisement-axis cells (70b-full / 70b-none / 70b-large)

The advertisement axis was re-baselined on **hermes-3-llama-3.1-70b via
OpenRouter** for the following reasons (recorded before any run):

- The measurement machine is an Intel MacBook Pro 15,4; Ollama on Intel
  does not use Metal/GPU and runs CPU-only at ~90 min/row (observed),
  making a 360-row run infeasible.
- `nousresearch/hermes-3-llama-3.1-8b` is not available on OpenRouter
  (confirmed 2026-06-12); only 70B and 405B Hermes-3 are listed.

**Design consequence:** Three cells are run on the same model and stack
(70B, OpenRouter). Within-stack comparisons (70b-full vs 70b-none vs
70b-large) are causal for advertisement effects. The 8B Ollama cell
(hermes3-8b-q4-ollama-20260611.jsonl) and the 70B OpenRouter cells are
NOT directly comparable; any 8B-vs-70B observation is advisory scale
data only. The pre-registered interpretation bands from
docs/advertisement-axis-protocol.md apply within the 70B cells.

Hosted quantization is provider-managed and unknown; it is noted in per-file
metadata as "openrouter/unknown".

- `hermes3-70b-openrouter-full-<date>.jsonl` — cell 70b-full (replication of cell 1 design on 70B)
- `hermes3-70b-openrouter-none-<date>.jsonl` — cell 70b-none (advertisement=none)
- `hermes3-70b-openrouter-large-<date>.jsonl` — cell 70b-large (advertisement=large)

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
| calls | list[dict] | `{class, turn}` per call; old cell-1 format is list[str] (backward-compat handled by analyze.py) |
| advertisement | string | full / none / large |
| malformed_turns | int | turns where parse failed |
| model | string | model identifier |
