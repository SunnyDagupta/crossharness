#!/usr/bin/env python3
"""The 60-second demo: one tool call crosses harness dialects and executes.

    python3 demo.py

A Hermes-format model output (in-band <tool_call> text, no ids, generic
snake_case surface) is translated through the IR into a Claude-Code-dialect
call (structured tool_use block, synthesized id, Read/Write surface),
executed for real against a sandboxed directory, and the result is
translated back into the <tool_response> text the model knows how to read.

No dependencies, no network, no model download. The model output is a
recorded fixture so the demo is deterministic; swap in a live model with
fixtures/capture.py.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crossharness import CLAUDE_CODE, HERMES_GENERIC, SandboxExecutor, Shim

SANDBOX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox")

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def style(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def header(n: int, title: str) -> None:
    print()
    print(style(f"[{n}] {title}", "1;36"))
    print(style("-" * 64, "2"))


def show_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


# The "model": a recorded Hermes-format turn. Generic surface
# (write_file/path), trained wire format (in-band tags), no call id.
MODEL_TURN_1 = """\
I'll save the note to disk now.
<tool_call>
{"name": "write_file", "arguments": {"path": "note.txt", "content": "tool calls are a dialect, not a language"}}
</tool_call>"""

MODEL_TURN_2 = """\
Reading it back to verify.
<tool_call>
{"name": "read_file", "arguments": {"path": "note.txt"}}
</tool_call>"""


def run_turn(shim: Shim, executor: SandboxExecutor, model_text: str, turn: int) -> None:
    base = (turn - 1) * 4

    header(base + 1, f"Model emits (Hermes dialect, turn {turn})")
    print(model_text)

    message, records = shim.model_to_harness(model_text)

    header(base + 2, "Shim translates: in-band text -> IR -> structured call")
    for record in records:
        print(
            f"IR: id={record.model_call.id} (synthesized)  "
            f"surface: {record.model_call.name} -> {record.harness_call.name}  "
            f"canonical: {record.harness_call.canonical}"
        )
        arg_moves = [
            f"{mk} -> {hk}"
            for mk, hk in zip(record.model_call.args, record.harness_call.args)
            if mk != hk
        ]
        if arg_moves:
            print(f"    args renamed: {', '.join(arg_moves)}")
    print()
    print("Anthropic-dialect message delivered to the harness:")
    show_json(message)

    results_message = executor.execute_message(message)

    header(base + 3, "Harness executes for real (sandboxed Claude-Code-style tools)")
    show_json(results_message)

    response_text = shim.harness_to_model(results_message)

    header(base + 4, "Shim translates back: the model sees its own dialect")
    print(response_text)


def main() -> None:
    if os.path.isdir(SANDBOX):
        shutil.rmtree(SANDBOX)

    print(style("crossharness demo", "1"))
    print(
        "A Hermes-fluent model drives a Claude-Code-dialect harness.\n"
        "Watch one tool call cross the border in both directions."
    )

    shim = Shim(model_profile=HERMES_GENERIC, harness_profile=CLAUDE_CODE)
    executor = SandboxExecutor(SANDBOX)

    run_turn(shim, executor, MODEL_TURN_1, turn=1)
    run_turn(shim, executor, MODEL_TURN_2, turn=2)

    note_path = os.path.join(SANDBOX, "note.txt")
    print()
    print(style("Proof on disk:", "1"))
    with open(note_path, "r", encoding="utf-8") as fh:
        print(f"  {note_path}: {fh.read()!r}")
    print()
    print("Next: python3 eval.py  (quantifies what each translation layer recovers)")


if __name__ == "__main__":
    main()
