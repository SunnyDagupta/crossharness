# CLAUDE.md — crossharness

Standing context for working in this repository. Read fully before any change.

## What this project is

crossharness is a cross-harness portability shim for tool-calling models,
plus the eval that justifies it. Frontier labs train model and harness
hand in glove; open source mixes any model with any harness and loses that
coupling. This repo translates tool-calling dialects at inference time --
both the wire encoding (in-band `<tool_call>` text vs structured
`tool_use` blocks, id synthesis, argument encoding) and the harness
surface (tool names and schemas: `read_file({"path"})` vs
`Read({"file_path"})`) -- and measures how much competence each layer
recovers.

Positioning, one line per neighbor: MCP standardizes agent-to-tools. ACP
standardizes editor-to-agent. OpenEnv (Meta-PyTorch + Hugging Face) fixes
model-to-harness coupling at training time. crossharness is the
inference-time counterpart, and the measured recovery eval is the
project's center of gravity -- the shim is the intervention the eval
measures, not the other way round.

Strategic frame: wire-format translation is commodity (LiteLLM, vLLM
`--tool-call-parser` already do it). The unclaimed layer is harness
surface mapping. The eval exists to attribute recovery to each layer
separately. If live measurement shows surface recovery near zero, that is
a publishable finding (inference-time coupling is notational only), not a
failure; never spin it.

## Phase status

- Phase 0 (validation research): done.
- Phase 1 (mechanism): done, v0.2.0. Demo, three-way replay eval, 22 unit
  tests, CI workflow.
- Phase 2a (live measurement): machinery shipped (24-task suite + live
  runner, smoke-tested). The measurement itself has NOT run yet.
- Phase 2b/2c (Codex dialect, OpenClaw via ACP, model x harness matrix):
  not started; gated on 2a results and an explicit owner decision.

## Architecture map

- `crossharness/ir.py` -- ToolCall / ToolResult IR; deterministic
  sequential id synthesis; ParseError
- `crossharness/encodings/hermes.py` -- in-band `<tool_call>` parse/emit;
  tolerates double-encoded arguments; ids synthesized
- `crossharness/encodings/anthropic.py` -- structured tool_use /
  tool_result messages; id correlation
- `crossharness/surfaces.py` -- HarnessProfile / ToolSpec declarative
  surface tables with aliases; HERMES_GENERIC and CLAUDE_CODE profiles;
  the novel layer
- `crossharness/shim.py` -- composition; per-conversation ledger restores
  the model's own surface names on result back-translation
- `crossharness/executor.py` -- strict Claude-Code-style harness (Read /
  Write / LS), real sandboxed execution, path jail
- `demo.py` -- 60-second hero demo (deterministic, offline)
- `eval.py` -- three-way replay eval; `--assert` enforces the 0/3, 1/3,
  3/3 layer-attribution table for CI
- `eval_live.py` -- Phase 2a live runner: three conditions, per-call
  classification, JSONL output, recovery decomposition; `--smoke` is the
  offline pipeline check
- `phase2a/tasks.py` -- 24 seeded, programmatically checked tasks
  (families W6 / E8 / L5 / M5) + the Hermes-style system prompt
- `docs/phase2a-protocol.md` -- measurement protocol incl. threats to
  validity; read before touching anything measurement-adjacent
- `fixtures/` -- constructed Phase 1 fixtures (labeled as constructed) +
  `capture.py` for live regeneration
- `.github/workflows/tests.yml` -- unit tests + demo + `eval.py --assert`
  on Python 3.9 and 3.12

## Hard constraints (violating any of these is a regression)

1. Stdlib only. No pip dependencies, no requirements.txt, no third-party
   imports anywhere in the repo. The clone-and-run quickstart is a core
   promise.
2. Python 3.9+ compatibility.
3. No emojis anywhere: code, docs, output, commit messages.
4. Measurement honesty is the brand. Fixtures stay labeled as
   constructed. Negative or null results get reported plainly. Never
   tune the alias table on the same run it is evaluated against (freeze
   rule in docs/phase2a-protocol.md); report the alias-table version with
   any results.
5. The executor stays strict. No fuzzy tool-name matching, no
   auto-correction in the harness. Leniency belongs in surface profiles
   (data), never in the executor.
6. `eval.py --assert` (0/3, 1/3, 3/3) is a CI contract. Any change that
   alters the table requires deliberate intent and a README update, not a
   quiet fixture tweak.
7. Determinism in the offline paths: id synthesis stays sequential;
   demo and eval must produce identical output across runs.
8. Tone in all prose: plain, precise, no hype, caveats stated up front.

## Commands

```
python3 -m unittest discover -s tests      # 22 tests, all green
python3 demo.py                            # offline hero demo
python3 eval.py                            # replay eval table
python3 eval.py --assert                   # CI contract check (exit 0)
python3 eval_live.py --smoke               # offline live-pipeline check
python3 eval_live.py --model <slug> ...    # Phase 2a measurement (network)
```

## Known sharp edges

- vLLM serving for Phase 2a must NOT use `--tool-call-parser`. The
  runner parses raw in-band text itself; vLLM's parser would pre-translate
  and invalidate the measurement.
- `eval_live.py --tool-role` defaults to `user` for provider
  compatibility; `tool` is available where supported. Report which was
  used alongside results.
- The Hermes regex terminates at `}</tool_call>`; JSON string values
  containing that literal sequence will misparse. Known, acceptable,
  same failure class as production parsers; such calls score as
  malformed.
- `sandbox/`, `sandbox_live/`, `results/`, `__pycache__/` are runtime
  artifacts; never commit them.

## Owner's working style

Phase-by-phase with explicit check-ins at gates. Skeptical by default:
prefers a blunt feature-vs-product verdict over cheerleading. Wants the
smallest convincing demo, not the largest impressive one. Desktop-first
and design-tokens rules apply to web UI work only (none in this repo).
