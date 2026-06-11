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

## Roadmap

- Phase 2a (machinery shipped, measurement pending): run the live suite
  against open Hermes-format models; publish measured failure-mode
  frequencies and recovery numbers
- Phase 2b: OpenAI/Codex dialect encoding (arguments as JSON-encoded
  strings, `tool_call_id` correlation); OpenClaw as a downstream target via
  its ACP surface
- Phase 2c: the recovery benchmark proper -- model x harness matrix, task
  suite large enough to report points lost and points recovered with
  confidence intervals

## License

MIT
