# Phase 2a measurement protocol

Phase 1 proved the mechanism on constructed fixtures. Phase 2a replaces the
constructed inputs with live model outputs and turns the demo table into a
measured claim: how often do the failure modes occur, and how much does
each translation layer recover?

## Setup

- Model under test: any Hermes-format open model served through an
  OpenAI-compatible endpoint (reference target: a Hermes 3 class model;
  vLLM locally or any hosted provider).
- Harness: the strict Claude-Code-dialect executor from Phase 1
  (structured `tool_use` blocks; `Read` / `Write` / `LS` surface).
- Tools are advertised to the model Hermes-style in the system prompt,
  using the HARNESS surface (Claude Code names). Whether the model adheres
  to the advertised surface or falls back to trained generic conventions
  is the behavior being measured -- nothing in the prompt hints at the
  mismatch or at any translator.

## Task suite

24 tasks in 4 parameterized families (`phase2a/tasks.py`):

- W: direct file writes (6) -- can succeed in one call
- E: read a seeded structured file, extract a value, write it (8) --
  two dependent calls; result content must survive back-translation
- L: navigate a seeded directory tree, find a file by criterion, read
  it (5) -- success requires the model to actually receive file contents
- M: multi-step combinations of reads and writes (5)

Every task has a deterministic seed and a programmatic checker. No task
mentions tool names in its prompt.

## Conditions

Each (task, sample) runs independently under three conditions:

- `none`: the model's raw text goes to the harness untranslated. The
  structured API recognizes no tool call; the sample ends after the first
  turn. This is the floor: what mixing model and harness raw actually does.
- `syntactic`: wire-format translation only (LiteLLM-class capability).
  Tool names and argument keys pass through verbatim. The harness's
  error messages DO flow back to the model, so the model may self-correct
  -- this is deliberate. It makes the commodity baseline as strong as it
  honestly is, not a strawman.
- `full`: syntactic translation plus harness surface mapping (including
  the alias table in `crossharness/surfaces.py`).

## Sampling

- N samples per (task, condition); default 3, report with N >= 5.
- Temperature 0.6 (diversity matters: surface fallback is stochastic).
- Max 6 turns per sample; a turn with no parseable tool call ends the
  sample (the model believes it is done); the checker then decides.

## Metrics

Per parsed call, classify the surface the model emitted:

- `harness_native`: name matches the advertised Claude Code surface
- `mappable_generic`: name resolves via the generic profile (incl. aliases)
- `unmapped`: parsed, but no mapping exists
- `malformed`: not parseable from the wire format

Reported aggregates:

1. Deviation rate: share of calls that are not `harness_native`,
   measured on `full`-condition runs (those progress furthest, so they
   observe the most calls). This is the "does the failure mode occur"
   number.
2. Task success rate per condition, with bootstrap confidence intervals
   once N is large enough to bother.
3. Recovery decomposition:
   - wire-format recovery = success(syntactic) - success(none)
   - surface recovery = success(full) - success(syntactic)
   The second number is the project's reason to exist. If it is ~0 on
   live models, that is the published finding: inference-time coupling is
   notational only and training-time approaches (OpenEnv) are the fix.

## Threats to validity, stated up front

- The alias table can bias either direction: too thin and `full`
  under-recovers (unfair to the shim); tuned on the same captures it is
  evaluated on and it overfits (unfair to the baseline). Freeze the alias
  table before a measurement run; expand it only between runs and report
  the table version with results.
- Self-correction under error feedback strengthens `syntactic` -- by
  design, but it also means turn budget affects the gap. Report max-turns
  alongside results.
- One model, one task domain (file ops) is not a benchmark; it is one
  cell of the eventual model x harness matrix (Phase 2c). Do not
  generalize a single cell.
- Checker leniency: checkers accept any correct outcome, not specific
  call sequences, so a model that finds an unanticipated route to the
  goal still passes.

## Running

```
export OPENROUTER_API_KEY=...   # or any OpenAI-compatible key
python3 eval_live.py --model <model-slug> --samples 3
python3 eval_live.py --smoke    # no network: validates the pipeline
```

Results stream to `results/live_<timestamp>.jsonl`; the aggregate table
prints at the end. Cost at defaults is a few hundred small-model
completions per full run.
