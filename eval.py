#!/usr/bin/env python3
"""Three-way recovery eval: which translation layer earns the points?

    python3 eval.py

Three recorded Hermes-format sessions are replayed against the strict
Claude-Code-dialect harness under three conditions:

  none        raw model text is handed to the harness untranslated --
              the structured API sees prose, recognizes nothing
  syntactic   wire-format translation only (what LiteLLM-class adapters
              do): in-band text becomes structured calls with synthesized
              ids, but tool names/schemas pass through verbatim
  full        syntactic translation plus harness surface mapping
              (read_file -> Read, path -> file_path)

The number that matters is the delta between syntactic and full: that is
the contribution of the layer no existing adapter ships.

Fixtures are constructed, not sampled from a live model -- they represent
documented failure modes (see README "What this proves"). Regenerate
against a live endpoint with fixtures/capture.py.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crossharness import CLAUDE_CODE, HERMES_GENERIC, ParseError, SandboxExecutor, Shim
from crossharness.encodings import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
SANDBOX = os.path.join(HERE, "sandbox")

SENTINEL = "MEETING NOTES: ship the dialect layer."


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

def seed_task2(root: str) -> None:
    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"service": "gateway", "port": 8080, "tls": False}, fh, indent=2)


def seed_task3(root: str) -> None:
    docs = os.path.join(root, "docs")
    os.makedirs(docs, exist_ok=True)
    with open(os.path.join(docs, "notes.md"), "w", encoding="utf-8") as fh:
        fh.write(SENTINEL)


def check_task1(root: str, transcript: "List[dict]") -> bool:
    path = os.path.join(root, "haiku.txt")
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8") as fh:
        return "different glove" in fh.read()


def check_task2(root: str, transcript: "List[dict]") -> bool:
    path = os.path.join(root, "out.txt")
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip() == "8080"


def check_task3(root: str, transcript: "List[dict]") -> bool:
    # Success = the model actually received the file contents back:
    # the final tool result carries the sentinel and is not an error.
    for results_message in reversed(transcript):
        results = anthropic.parse_results(results_message)
        if not results:
            continue
        last = results[-1]
        return (not last.is_error) and SENTINEL in str(last.content)
    return False


@dataclass
class Task:
    name: str
    fixture: str
    seed: Optional[Callable[[str], None]]
    check: Callable[[str, "List[dict]"], bool]


TASKS = [
    Task("T1 write_haiku", "task1_write_haiku.txt", None, check_task1),
    Task("T2 config_extract", "task2_config_extract.txt", seed_task2, check_task2),
    Task("T3 find_and_read", "task3_find_and_read.txt", seed_task3, check_task3),
]

CONDITIONS = ("none", "syntactic", "full")


# ---------------------------------------------------------------------------
# Replay machinery
# ---------------------------------------------------------------------------

def load_turns(fixture_name: str) -> "List[str]":
    path = os.path.join(FIXTURES, fixture_name)
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    body = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    )
    return [turn.strip() for turn in body.split("---TURN---") if turn.strip()]


def run_task(task: Task, condition: str) -> "Tuple[bool, str]":
    """Replay one task under one condition. Returns (passed, failure_note)."""
    if os.path.isdir(SANDBOX):
        shutil.rmtree(SANDBOX)
    executor = SandboxExecutor(SANDBOX)
    if task.seed:
        task.seed(SANDBOX)

    shim = Shim(
        model_profile=HERMES_GENERIC,
        harness_profile=CLAUDE_CODE,
        syntactic_only=(condition == "syntactic"),
    )

    transcript: "List[dict]" = []
    note = ""
    for turn_text in load_turns(task.fixture):
        if condition == "none":
            # No translation: the structured harness receives the model's
            # raw text as a plain text block. No tool_use blocks exist, so
            # nothing is recognized or executed.
            message = {
                "role": "assistant",
                "content": [{"type": "text", "text": turn_text}],
            }
            results_message = executor.execute_message(message)
            transcript.append(results_message)
            if not anthropic.parse_results(results_message):
                note = "harness saw plain text; no tool call recognized"
            continue

        try:
            message, records = shim.model_to_harness(turn_text)
        except ParseError as exc:
            # Malformed model output is a data point, not a crash. Live
            # captures (Phase 2a) will contain these; they score as
            # failures with the parse error as the failure stage.
            note = f"malformed tool call: {exc}"
            break
        results_message = executor.execute_message(message)
        transcript.append(results_message)

        for result in anthropic.parse_results(results_message):
            if result.is_error and not note:
                note = str(result.content).splitlines()[0]
        # Translate results back to the model's dialect, as the live loop
        # would. With recorded fixtures the next turn is fixed, but the
        # back-translation path must still run (and is checked in tests).
        shim.harness_to_model(results_message)

    passed = task.check(SANDBOX, transcript)
    return passed, ("" if passed else note or "task checker failed")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def main() -> None:
    results: "dict[str, dict[str, Tuple[bool, str]]]" = {}
    for condition in CONDITIONS:
        results[condition] = {}
        for task in TASKS:
            results[condition][task.name] = run_task(task, condition)

    labels = {
        "none": "no translation",
        "syntactic": "syntactic only",
        "full": "full shim",
    }

    name_w = max(len(t.name) for t in TASKS) + 2
    cond_w = max(len(v) for v in labels.values()) + 2

    print()
    print("crossharness recovery eval")
    print(f"model dialect: Hermes in-band text, generic surface ({HERMES_GENERIC.name})")
    print(f"harness dialect: Anthropic structured blocks, Claude Code surface ({CLAUDE_CODE.name})")
    print()

    header = "condition".ljust(cond_w) + "".join(t.name.ljust(name_w) for t in TASKS) + "total"
    print(header)
    print("-" * len(header))
    for condition in CONDITIONS:
        row = labels[condition].ljust(cond_w)
        passed_count = 0
        for task in TASKS:
            passed, _ = results[condition][task.name]
            passed_count += passed
            row += ("PASS" if passed else "FAIL").ljust(name_w)
        row += f"{passed_count}/{len(TASKS)}"
        print(row)

    print()
    print("failure detail:")
    for condition in CONDITIONS:
        for task in TASKS:
            passed, note = results[condition][task.name]
            if not passed:
                print(f"  [{labels[condition]}] {task.name}: {note}")

    none_total = sum(results["none"][t.name][0] for t in TASKS)
    syn_total = sum(results["syntactic"][t.name][0] for t in TASKS)
    full_total = sum(results["full"][t.name][0] for t in TASKS)

    print()
    print("reading the table:")
    print(
        f"  wire-format translation recovers {syn_total - none_total} of "
        f"{len(TASKS)} tasks -- that part is commodity (LiteLLM-class)."
    )
    print(
        f"  harness surface mapping recovers {full_total - syn_total} more -- "
        "that is the layer no existing adapter ships."
    )
    print()
    print(
        "Fixtures are constructed to represent documented failure modes;\n"
        "Phase 2 measures their frequency on live models. See README."
    )

    if "--assert" in sys.argv:
        # CI mode: the layer-attribution pattern is the contract.
        expected = (none_total, syn_total, full_total) == (0, 1, len(TASKS))
        sys.exit(0 if expected else 1)


if __name__ == "__main__":
    main()
