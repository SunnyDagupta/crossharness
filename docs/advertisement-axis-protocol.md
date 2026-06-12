# Advertisement-axis protocol addendum

Pre-registered before any cell A or cell B measurement data exists.
Original constants held from cell 1 (HANDOFF.md): same model, same 24 tasks, same
three conditions (none / syntactic / full), temperature 0.6, max-turns 6, samples 5.

## Re-baseline amendment (recorded before any 70B run; 2026-06-12)

The advertisement axis was re-baselined on **hermes-3-llama-3.1-70b via OpenRouter**
after two blocking constraints were discovered:

1. The measurement machine is an Intel MacBook Pro 15,4. Ollama on Intel Mac does not
   use Metal/GPU and runs CPU-only. Observed throughput: ~90 min/row for the none
   advertisement cell (verbose model responses without tool schemas). A 360-row run
   would take weeks; infeasible.
2. `nousresearch/hermes-3-llama-3.1-8b` is not listed on OpenRouter (confirmed
   2026-06-12). Only 70B and 405B Hermes-3 variants are available.

**Design consequence:** Three cells (70b-full / 70b-none / 70b-large) are run on the
same model and serving stack. Within-stack advertisement comparisons are causal.
The cell-1 8B/Ollama result and the 70B/OpenRouter results are NOT directly
comparable; any 8B-vs-70B observation is advisory scale data only and is explicitly
labeled as such in all reporting. The pre-registered interpretation bands below apply
within the 70B cells only.

## Why this axis

Cell 1 (hermes3:8b, tools advertised in-prompt in the model's trained format) found
deviation 0/134 (95% upper ≈2.2%). The obvious critique: "you advertised the tools in
the trained format; remove that and measure again." This axis answers it with data
under two advertisement variants. The re-baseline means the answer is now specifically
about hermes-3-llama-3.1-70b; generalization to 8B is not warranted without re-run.

## Measurement cells

| Cell | Flag | Model | System prompt | Advertised tools |
|------|------|-------|--------------|-----------------|
| cell-1 (8B reference) | `--advertisement full` | hermes3:8b / Ollama / CPU | Full Hermes-format catalog (Read/Write/LS) | 3 |
| 70b-full (replication) | `--advertisement full` | hermes-3-70b / OpenRouter | Full Hermes-format catalog (Read/Write/LS) | 3 |
| 70b-none | `--advertisement none` | hermes-3-70b / OpenRouter | Wire-format instruction only; no tool names or schemas | 0 |
| 70b-large | `--advertisement large` | hermes-3-70b / OpenRouter | Identical preamble; 15-tool catalog | 15 (real at 4/9/14) |

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

## Frozen distractor list (cell 70b-large)

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
reasonable-but-unmapped names in 70b-none are reported as unmapped deviation; alias
expansion happens only after all cells complete, clearly labeled, with re-runs marked
as post-expansion.

## Primary metrics (per cell)

- **Invocation rate:** share of syntactic+full rows containing at least one parsed
  call. If <20% under 70b-none, the headline is "blind deployment fails at invocation,
  upstream of naming"; naming results are still reported but flagged low-n.
- **First-call deviation rate:** classification of each row's first parsed call —
  the pure prior measure, uncontaminated by error-feedback self-correction.
- **Post-first-call deviation rate:** calls after the first per row — the
  self-correction signal.
- **Task success per condition:** same three-condition decomposition as cell 1.
- **Surface recovery:** success(full) − success(syntactic), with pooled two-proportion SE.
- **advertised_distractor rate (70b-large only):** frequency of calls that are in the
  advertised set but not in the executor. Reported separately; never counted as deviation.

## Pre-registered interpretation rules

### Cell 70b-none (no advertisement)

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

### Cell 70b-large (large catalog)

- **Deviation ≤5% with 15 advertised tools:** catalog noise does not induce notational
  drift for this model class.
- `advertised_distractor` calls are semantic confusion, out of shim scope; report the
  rate, never count it as deviation.

### Global

- Wire-format recovery = success(syntactic) − success(none).
- Surface recovery = success(full) − success(syntactic), pooled two-proportion SE.
- Rule of three for zero-event deviation upper CI.
- `none` condition is the definitional floor in all cells.
- Within-stack comparisons (70b-full vs 70b-none vs 70b-large) are causal.
  Cross-stack comparisons (cell-1 8B vs any 70B) are advisory scale observations only.

## Data file naming

```
data/hermes3-70b-openrouter-full-<YYYYMMDD>.jsonl   # cell 70b-full
data/hermes3-70b-openrouter-none-<YYYYMMDD>.jsonl   # cell 70b-none
data/hermes3-70b-openrouter-large-<YYYYMMDD>.jsonl  # cell 70b-large
```

Each JSONL row includes `"advertisement"` (full/none/large) and `"model"`.
Hosted quantization is provider-managed and unknown; metadata records
`"serving": "openrouter/unknown"`.

## Ops notes

- Secrets read from environment only (`OPENROUTER_API_KEY`); never inlined in commands.
- stream:false always sent.
- Transport retries: up to 3 attempts, 10/20 s backoff for TimeoutError/OSError/URLError.
- Cost guard: stop and report to owner if projected spend for three cells exceeds $25.
