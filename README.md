# crossharness

A cross-harness portability shim for tool-calling models: translate the
tool-calling dialect a model was trained on into the dialect a harness
expects, at inference time, and measure how much competence that recovers.

```
git clone <this repo> && cd crossharness && python3 demo.py
```

No dependencies, no install step, no network, no GPU. Python 3.9+.

## The problem

Frontier labs train models and harnesses hand in glove: the model learns
its harness's exact tool-calling conventions during post-training, so it is
fluent in that harness specifically. Open source breaks the coupling --
people pair any model with any harness, and a model trained on one
convention underperforms when driving another.

The mismatch has two distinct layers, and conflating them hides where the
unsolved problem lives:

| layer | example mismatch | status |
|---|---|---|
| wire encoding | in-band `<tool_call>` text vs structured `tool_use` blocks; arguments as JSON object vs JSON-encoded string; missing call ids | solved (commodity): LiteLLM translates provider formats, vLLM's `--tool-call-parser` parses in-band text |
| harness surface | model calls `read_file({"path": ...})`, harness exposes `Read({"file_path": ...})` | unsolved: no existing adapter maps tool names, schemas, and conventions between harnesses |

crossharness ships both layers behind one intermediate representation, and
-- the part that matters -- an eval that attributes recovery to each layer
separately.

## Where this sits

Every adjacent layer of the agent stack has a standard. This one does not:

| layer | standard | what it fixes |
|---|---|---|
| agent to tools | MCP | tool discovery and execution |
| editor/client to agent | ACP | sessions, file ops, permissions |
| model to harness, at training time | OpenEnv (Meta-PyTorch + Hugging Face) | train open models against standardized environments |
| model to harness, at inference time | nothing | a model fluent in dialect A driving a harness that speaks dialect B |

OpenEnv frames the coupling problem exactly right and attacks it by
training. crossharness is the complementary attack: no retraining, translate
at the boundary, and quantify what translation alone recovers. If the
recovered share turns out to be small, that is a finding too -- it means
the residual coupling is behavioral, not notational, and training-time
approaches are the only fix. The eval is built to be informative in either
direction.

## Quickstart

The 60-second demo -- one tool call crosses dialects and executes for real:

```
python3 demo.py
```

A recorded Hermes-format model turn (in-band text, no call ids, generic
snake_case surface) is parsed into the IR, surface-mapped, delivered to a
strict Claude-Code-dialect harness as a structured `tool_use` block,
executed against a sandboxed directory, and the `tool_result` is translated
back into the `<tool_response>` text the model knows how to read.

Then the number:

```
python3 eval.py
```

Three recorded sessions replay under three conditions:

```
condition        T1 write_haiku   T2 config_extract   T3 find_and_read   total
------------------------------------------------------------------------------
no translation   FAIL             FAIL                FAIL               0/3
syntactic only   PASS             FAIL                FAIL               1/3
full shim        PASS             PASS                PASS               3/3
```

Reading the table: wire-format translation alone (the LiteLLM-class
capability) recovers T1, where the model used the right tool names in the
wrong wire format. It cannot recover T2 or T3, where the model fell back to
its trained generic surface (`read_file`, `path`) against a harness that
exposes `Read` / `file_path` -- syntax arrives intact and the harness
correctly rejects it. The harness surface layer recovers those. The delta
between row two and row three is the contribution no existing adapter
ships.

## What this proves, and what it does not

Phase 1 proves the mechanism: dialect translation through a common IR
unblocks execution, and surface mapping recovers failures that wire-format
translation alone cannot, on fixtures representing documented failure
modes (models emitting trained conventions instead of advertised ones --
the same failure class that motivated vLLM's tool-call parsers and
OpenClaw's internal tool-call-repair package).

Phase 1 does not prove that live models hit these failure modes at any
particular rate, nor that translation recovers end-task performance on
real workloads. That is Phase 2: capture live outputs from open models
(`fixtures/capture.py` is the starting point), replay the same three-way
eval, and report "model X loses N points driving harness Y raw, recovers M
through the shim." The fixtures shipped here are constructed and labeled
as such -- determinism keeps the demo honest and reproducible, and the
capture path keeps it falsifiable.

## Architecture

```
model text (Hermes dialect)                 harness API (Claude Code dialect)
  prose + <tool_call>{...}</tool_call>        {"type":"tool_use","id":...,
        |                                          "name":"Read","input":{...}}
        v                                                  ^
  [encoding: parse in-band text,                [encoding: emit structured
   synthesize ids, normalize args]                blocks, correlate by id]
        \                                                 /
         ToolCall IR {id, name, args, origin, raw, canonical}
                            |
              [surface map via harness profiles]
              read_file -> fs.read -> Read
              path -> file_path
```

- `crossharness/ir.py` -- the IR: `ToolCall`, `ToolResult`, deterministic id synthesis
- `crossharness/encodings/` -- wire formats (`hermes` in-band text, `anthropic` structured blocks)
- `crossharness/surfaces.py` -- harness profiles: declarative tool-surface tables with alias support (models fall back to `read`, `cat`, `open_file` non-uniformly), the novel layer
- `crossharness/shim.py` -- composition + per-conversation ledger so results return in the model's own dialect
- `crossharness/executor.py` -- strict Claude-Code-style harness with real, sandboxed execution
- `eval.py` -- the three-way recovery eval (`--assert` for CI)
- `eval_live.py` -- the Phase 2a live runner (see below)
- `fixtures/` -- recorded sessions + `capture.py` to regenerate from a live endpoint

Adding a harness is writing one profile table (and one encoding module if
the wire format is new). Profiles are data, not code.

## Tests

```
python3 -m unittest discover -s tests -v
```

Covers round-trip stability (Hermes -> IR -> Anthropic -> IR -> Hermes),
id synthesis and correlation, double-encoded argument tolerance, surface
mapping in both directions, pass-through behavior for unmappable tools,
and executor strictness (unknown tools, schema violations, path escapes).

## Phase 2a: measuring on live models

The machinery to turn the demo table into a measured claim ships in this
repo. `phase2a/tasks.py` defines 24 seeded, programmatically-checked
file-ops tasks in 4 families; `eval_live.py` drives a live Hermes-format
model through all of them under the same three conditions and reports the
recovery decomposition (wire-format recovery vs surface recovery), per-call
surface classification, and the headline deviation rate. The full protocol,
including threats to validity, is in `docs/phase2a-protocol.md`.

```
export OPENROUTER_API_KEY=...    # or any OpenAI-compatible endpoint
python3 eval_live.py --model nousresearch/hermes-3-llama-3.1-8b
python3 eval_live.py --smoke     # no network: validates the pipeline
```

If surface recovery on live models turns out to be near zero, that is the
published finding, not a failure of the project: it would mean residual
model-harness coupling is behavioral rather than notational, and
training-time approaches are the only fix. The eval is informative in
either direction.

## Measured results (Phase 2a)

**Run metadata**

| Field | Value |
|-------|-------|
| Date | 2026-06-11 |
| Model | hermes3:8b (Q4_0, 8.0B params, 131072-ctx) |
| Serving | Ollama 0.30.6, localhost:11434 |
| Harness | crossharness @ 8dc769d + stream:false patch (see eval_live.py) |
| JSONL | data/hermes3-8b-q4-ollama-20260611.jsonl |
| Rows | 360 (24 tasks × 3 conditions × 5 samples) |

**Note:** The `none` condition is the definitional floor — without any
translation the harness never receives a well-formed tool call, so 0% is
by construction, not a model property.

**Success rate per condition**

| Condition | Success | Rate |
|-----------|---------|------|
| none | 0/120 | 0.0% |
| syntactic | 32/120 | 26.7% |
| full | 28/120 | 23.3% |

Wire-format recovery (syntactic − none): **+26.7%** (SE 4.4%)
Surface recovery (full − syntactic): **−3.3%** (SE 5.6%) — no detectable difference

**Call classification**

| Condition | Calls | harness_native | deviation | upper 95% CI |
|-----------|-------|----------------|-----------|--------------|
| syntactic | 77 | 77 | 0 | ≤3.9% |
| full | 57 | 57 | 0 | ≤5.3% |

Average calls/row under full: 0.47

**Success by family (full condition)**

| Family | none | syntactic | full |
|--------|------|-----------|------|
| E | 0/40 | 8/40 | 7/40 |
| L | 0/25 | 4/25 | 1/25 |
| M | 0/25 | 1/25 | 2/25 |
| W | 0/30 | 19/30 | 18/30 |

**Reading**

Wire-format recovery is +26.7%: without the syntactic shim, hermes3:8b
emits zero harness-compatible calls on every task. The shim's format
translation alone recovers 26.7% of tasks. Surface deviation did not
occur in this cell (0/57 calls used a generic surface, ≤5.3% upper 95%
CI), so the full alias layer made no detectable difference (−3.3%, 0.6
SE). The dominant residual failure is non-invocation (0.47 calls/row; W
family ~60% vs M family ~8%): the model often produces no tool call at
all, which is behavioral, not notational.

**Scope limits:** one 8B model, Q4_0 quant, 3-tool clean catalog
advertised in-prompt, Hermes-style prompt the model was trained on. The
Phase 1 constructed fixtures exhibit a surface-deviation failure mode
this live cell did not reproduce — whether that reflects the model
class, the catalog size, or the advertisement condition is the next
measurement axis.

## Roadmap

- Phase 2a (complete): hermes3:8b/Q4_0, wire-format recovery +26.7pp,
  surface recovery not detectable (−3.3pp, 0.6 SE); data in data/
- Phase 2a next axis: advertisement — larger and messier tool catalogs,
  no in-prompt advertisement, non-Hermes-trained models; pending owner
  decision (Phase 2b gated: <5pp surface recovery branch applies)
- Phase 2b (gated, not started): OpenAI/Codex dialect encoding
  (arguments as JSON-encoded strings, `tool_call_id` correlation);
  OpenClaw as a downstream target via its ACP surface
- Phase 2c: the recovery benchmark proper — model × harness matrix,
  task suite large enough to report points lost and recovered with
  confidence intervals

## License

MIT
