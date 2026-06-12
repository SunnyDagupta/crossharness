# Advertisement-axis protocol addendum

Pre-registered before any cell A or cell B measurement data exists.
Constants held from cell 1 (HANDOFF.md): same model, same 24 tasks, same three
conditions (none / syntactic / full), temperature 0.6, max-turns 6, samples 5.

## Why this axis

Cell 1 (hermes3:8b, tools advertised in-prompt in the model's trained format) found
deviation 0/134 (95% upper ≈2.2%). The obvious critique: "you advertised the tools in
the trained format; remove that and measure again." This axis answers it with data
under two advertisement variants.

## Measurement cells

| Cell | Flag | System prompt | Advertised tools |
|------|------|--------------|-----------------|
| 1 (reference) | `--advertisement full` | Full Hermes-format catalog (Read/Write/LS) | 3 (executor set) |
| A | `--advertisement none` | Wire-format instruction only; no tool names or schemas | 0 |
| B | `--advertisement large` | Identical preamble; 15-tool catalog | 15 (real at positions 4/9/14) |

## System prompt rationale

**SYSTEM_PROMPT_NONE** retains the wire-format instruction (`<tool_call>` XML) but
removes all tool names and schemas. This isolates surface knowledge (whether the model
invokes executor-compatible names) from format discovery (whether the model emits
parseable `<tool_call>` blocks at all). Total silence about tool calling would
conflate two variables: format discovery and invocation willingness.

**SYSTEM_PROMPT_LARGE** uses an identical preamble to the cell 1 system prompt but
lists 15 tools. Real executor tools (Read, Write, LS) are at positions 4, 9, 14
(frozen; position effects noted as a limitation). Distractors are advertised but not
implemented; the executor remains strict and unchanged.

## Frozen distractor list (cell B)

The following 12 tool names are advertised in SYSTEM_PROMPT_LARGE but are not
implemented in the executor. They are classified as `advertised_distractor` —
semantic confusion, out of shim scope — and are never counted toward the deviation
metric.

```
web_search, exec_command, http_request,
message_send, calendar_create, image_generate, db_query,
sessions_list, cron_schedule, email_send, screenshot_capture,
browser_open
```

Alias table is FROZEN for the entire axis. No surfaces.py changes. Any
reasonable-but-unmapped names in cell A are reported as unmapped deviation; alias
expansion happens only after both cells, clearly labeled, with re-runs marked as
post-expansion.

## Primary metrics (per cell)

- **Invocation rate:** share of syntactic+full rows containing at least one parsed call.
  If <20% under cell A, the headline is "blind deployment fails at invocation,
  upstream of naming"; naming results are still reported but flagged low-n.
- **First-call deviation rate:** classification of each row's first parsed call —
  the pure prior measure, uncontaminated by error-feedback self-correction.
- **Post-first-call deviation rate:** calls after the first per row — the
  self-correction signal.
- **Task success per condition:** same three-condition decomposition as cell 1.
- **Surface recovery:** success(full) − success(syntactic), with pooled two-proportion SE.
- **advertised_distractor rate (cell B only):** frequency of calls that are in the
  advertised set but not in the executor. Reported separately; never counted as
  deviation.

## Pre-registered interpretation rules

### Cell A (no advertisement)

- **Invocation collapse:** if <20% of syntactic+full rows contain any parsed call,
  report "blind deployment fails at invocation, upstream of naming"; deviation results
  are reported but flagged low-n.
- **High first-call deviation AND surface recovery ≥15pp:** the surface layer has a
  measured domain (blind deployment); README scopes the shim's value claim to that
  domain.
- **High first-call deviation but surface recovery <5pp:** report "error-feedback
  substitutes for surface mapping" (verify via median turns-to-first-successful-call,
  syntactic vs full); the alias layer remains unjustified; benchmark story stands.
- **5–15pp surface recovery:** suggestive; no claims without a second model.

### Cell B (large catalog)

- **Deviation ≤5% with 15 advertised tools:** catalog noise does not induce notational
  drift for this model class.
- `advertised_distractor` calls are semantic confusion, out of shim scope; report the
  rate, never count it as deviation.

### Global

- Wire-format recovery = success(syntactic) − success(none).
- Surface recovery = success(full) − success(syntactic), pooled two-proportion SE.
- Rule of three for zero-event deviation upper CI.
- `none` condition is the definitional floor in all cells.

## Data file naming

```
data/hermes3-8b-q4-ollama-advnone-<YYYYMMDD>.jsonl    # cell A
data/hermes3-8b-q4-ollama-advlarge-<YYYYMMDD>.jsonl   # cell B
```

Each JSONL row includes an `"advertisement"` field set to `"none"` or `"large"`.

## Ops notes

- keep_alive=-1 sent automatically for local Ollama endpoints (prevents idle eviction).
  Override with OLLAMA_KEEP_ALIVE env var or --no-keep-alive flag.
- stream:false always sent (Ollama 0.30.6 /v1/chat/completions hangs without it).
- Transport retries: up to 3 attempts, 10/20 s backoff for TimeoutError/OSError/URLError.
